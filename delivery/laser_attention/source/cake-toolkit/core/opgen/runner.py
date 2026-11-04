"""P5 identity-bound subprocess execution, with a read-only CPU sandbox.

The protected control directory owns receipts, logs and immutable snapshots.
Only a stage payload is writable. NPU launch remains unavailable until its
fresh device/ABI adapter is validated; CPU fixtures never count as NPU wins.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Literal
import fcntl,hashlib,os,re,secrets,shutil,signal,subprocess,time
from .._refs import Ref
from .architecture import Attribute,Record,VersionedRecord,_dump,parse_json


@dataclass(frozen=True)
class FileIdentity(Record):
    path: str
    sha256: str
    size_bytes: int
    role: Literal['source','harness','input','runtime','tool','output','log']

    def verify(self):
        actual=identify(self.path,self.role)
        if actual!=self:raise ValueError('File identity changed: '+self.path)
        return True


def identify(path,role):
    p=Path(path).resolve(strict=True)
    if not p.is_file():raise ValueError('Expected a regular file: '+str(p))
    data=p.read_bytes()
    return FileIdentity(str(p),hashlib.sha256(data).hexdigest(),len(data),role)


def tool_fingerprint():
    # 本仓自定: bind the complete opgen Python implementation, not just a version label.
    entries=[(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(Path(__file__).parent.glob('*.py'))]
    return hashlib.sha256(_dump(entries).encode()).hexdigest()


@dataclass(frozen=True)
class DeviceLease(Record):
    kind: Literal['cpu','npu']
    physical_card: int | None = None


@dataclass(frozen=True)
class AscendLease(Record):
    profile: FileIdentity
    kind: Literal['ascend'] = 'ascend'


@dataclass(frozen=True)
class Stage(Record):
    id: str
    kind: Literal['build','guard','check','measure','verify']
    argv: tuple[str,...]
    report: str
    outputs: tuple[str,...] = ()
    # 本仓自定: per-stage execution budget, not expected kernel duration.
    timeout_s: int | float = 60


@dataclass(frozen=True)
class MetricBinding(Record):
    metric: str
    model_fingerprint: str
    scope: str


@dataclass(frozen=True)
class RunPlan(VersionedRecord):
    id: str
    output_dir: str
    model_fingerprint: str
    contract: str
    workload: str
    software: tuple[Attribute,...]
    inputs: tuple[FileIdentity,...]
    sandbox: FileIdentity
    tool_fingerprint: str
    device: DeviceLease | AscendLease
    stages: tuple[Stage,...]
    env: tuple[Attribute,...] = ()
    # Explicit caller contract: these stages are deterministic and relocatable.
    cache_cpu: bool = False
    implementation: FileIdentity | None = None
    metric_bindings: tuple[MetricBinding,...] = ()
    schema_version: int = 1


@dataclass(frozen=True)
class Metric(Record):
    name: str
    unit: str
    value: int | float


@dataclass(frozen=True)
class StageReport(VersionedRecord):
    plan_fingerprint: str
    build_fingerprint: str
    kind: Literal['build','guard','check','measure','verify']
    passed: bool
    nonce: str
    metrics: tuple[Metric,...] = ()
    schema_version: int = 1


@dataclass(frozen=True)
class Product(Record):
    relative_path: str
    snapshot: FileIdentity


@dataclass(frozen=True)
class StageReceipt(Record):
    stage: str
    kind: Literal['build','guard','check','measure','verify']
    status: Literal['passed','reused','failed','timeout','canceled']
    returncode: int | None
    started_ns: int
    finished_ns: int
    stdout: FileIdentity
    stderr: FileIdentity
    products: tuple[Product,...]
    nonce: str
    error: str = ''
    preconditions: tuple[FileIdentity,...] = ()


@dataclass(frozen=True)
class RunReceipt(VersionedRecord):
    plan: RunPlan
    attempt: int
    work_dir: str
    state: Literal['incomplete','failed','verified']
    transitions: tuple[str,...]
    stages: tuple[StageReceipt,...]
    build_fingerprint: str
    prior: FileIdentity | None
    error: str
    schema_version: int = 1

    def verify(self,*,allow_incomplete=False):
        validate_plan(self.plan)
        expected=['planned'];build='';seen=[];nonces=set()
        if self.prior is not None:
            self.prior.verify()
            prior=RunReceipt.from_json(Path(self.prior.path).read_text())
            if prior.plan!=self.plan or prior.attempt>=self.attempt:raise ValueError('Resume provenance mismatch')
            prior.verify(allow_incomplete=True)
        if len(self.stages)>len(self.plan.stages):raise ValueError('Extra execution stages')
        for stage,row in zip(self.plan.stages,self.stages):
            if (row.stage,row.kind)!=(stage.id,stage.kind) or row.finished_ns<row.started_ns:
                raise ValueError('Execution stage ordering/time mismatch')
            row.stdout.verify();row.stderr.verify()
            for condition in row.preconditions:condition.verify()
            for product in row.products:product.snapshot.verify()
            if row.status in ('passed','reused'):
                if row.returncode!=0 or row.error:raise ValueError('Successful stage has an error')
                if row.status=='reused':
                    if not self.plan.cache_cpu or stage.kind not in ('build','check') or self.prior is None:
                        raise ValueError('Invalid reusable stage')
                    previous=[r for r in prior.stages if r.stage==stage.id and r.status in ('passed','reused')]
                    if len(previous)!=1 or _product_keys(previous[0])!=_product_keys(row):raise ValueError('Reused products differ from source proof')
                if not row.nonce or row.nonce in nonces:raise ValueError('Missing/duplicate stage challenge')
                nonces.add(row.nonce)
                if self.prior is not None and row.status!='reused' and row.nonce in {r.nonce for r in prior.stages}:raise ValueError('Fresh stage reused an old challenge')
                _report(self.plan,stage,row.products,build,row.nonce)
                if isinstance(self.plan.device,AscendLease) and stage.kind!='build':
                    from .ascend import load_profile,Occupancy,verify_occupancy
                    if len(row.preconditions)!=1:raise ValueError('Missing parent occupancy proof')
                    occupancy=Occupancy.from_json(Path(row.preconditions[0].path).read_text())
                    verify_occupancy(load_profile(self.plan.device),occupancy)
                    if not row.started_ns<=occupancy.captured_ns<=row.finished_ns:raise ValueError('Occupancy record is outside this stage')
                if stage.kind=='build':build=_build_key(row.products,stage,self.plan)
                expected.append(STATES[stage.kind]);seen.append(stage.kind)
            elif row!=self.stages[-1]:raise ValueError('Failed stage did not terminate execution')
        if tuple(expected)!=self.transitions or build!=self.build_fingerprint:raise ValueError('Execution state/build differs')
        complete=seen==list(KINDS)
        if self.state=='verified':
            if not complete or self.error:raise ValueError('Verified state without a complete successful chain')
        elif self.state=='failed':
            if not self.error or not self.stages or self.stages[-1].status in ('passed','reused'):raise ValueError('Failed run has no terminal failure evidence')
        elif complete or self.error or any(s.status not in ('passed','reused') for s in self.stages):raise ValueError('Invalid incomplete checkpoint')
        elif not allow_incomplete:raise ValueError('Incomplete checkpoint is not a terminal result')
        return True


KINDS=('build','guard','check','measure','verify')
STATES=dict(build='built',guard='guards_passed',check='checked',measure='measured',verify='verified')


def _relative(value):
    p=Path(value)
    if not value or value=='.' or p.is_absolute() or any(x in ('.','..') for x in p.parts) or str(p)!=value:
        raise ValueError('Expected a normalized relative output path')
    return p


def validate_plan(p):
    if p.implementation is not None:
        from .lowering import validate_run_binding
        validate_run_binding(p)
    if p.schema_version!=1 or p.tool_fingerprint!=tool_fingerprint():raise ValueError('Runner/tool identity changed')
    if isinstance(p.device,AscendLease):
        from .ascend import validate_binding
        validate_binding(p)
    elif p.device.kind=='npu':
        if p.device.physical_card==0:raise ValueError('Physical NPU0 is forbidden')
        raise ValueError('NPU adapter requires validated fresh occupancy/clean/stable/determinism and ABI binding; not enabled in this CPU first step')
    elif p.device.physical_card is not None:raise ValueError('CPU plan cannot identify an NPU')
    if len({b.metric for b in p.metric_bindings})!=len(p.metric_bindings) or any(not b.metric or not b.scope or not re.fullmatch('[0-9a-f]{64}',b.model_fingerprint) for b in p.metric_bindings):raise ValueError('Invalid/duplicate metric model binding')
    if not p.id or not re.fullmatch('[0-9a-f]{64}',p.model_fingerprint) or not p.contract or not p.workload or not p.software:
        raise ValueError('Run model, numerical contract, workload and software identity required')
    root=Path(p.output_dir)
    if not root.is_absolute() or root.resolve()!=root or len(root.parts)<3 or '.git' in root.parts:
        raise ValueError('Invalid run directory or symlink alias')
    if root.parts[1] in ('usr','etc','lib','lib64','bin','sbin','dev','proc','sys'):
        raise ValueError('Run directory cannot modify a shared system prefix')
    roles={x.role for x in p.inputs}
    if not {'source','harness','input','runtime','tool'}.issubset(roles):raise ValueError('Explicit source/harness/input/runtime/tool closure required')
    if len({x.path for x in p.inputs})!=len(p.inputs):raise ValueError('Duplicate input identities')
    for item in p.inputs:
        if Path(item.path).is_relative_to(root):raise ValueError('Frozen inputs cannot live in the writable run directory')
        item.verify()
    p.sandbox.verify()
    if p.sandbox.role!='tool':raise ValueError('Sandbox must be a frozen executable tool')
    if tuple(s.kind for s in p.stages)!=KINDS or len({s.id for s in p.stages})!=len(p.stages):
        raise ValueError('Exactly ordered build/guard/check/measure/verify stages required')
    declared={x.path for x in p.inputs if x.role in ('tool','runtime')};outputs=set()
    for stage in p.stages:
        if not re.fullmatch('[a-zA-Z0-9_-]+',stage.id) or not stage.argv or stage.timeout_s<=0 or not Path(stage.argv[0]).is_absolute():raise ValueError('Invalid stage id/argv/deadline')
        if str(Path(stage.argv[0]).resolve()) not in declared:raise ValueError('Stage executable is not identity-bound')
        if Path(stage.argv[0]).name in ('sh','bash','zsh','dash','ksh'):
            harnesses={item.path for item in p.inputs if item.role=='harness'}
            if len(stage.argv)<2 or stage.argv[1] not in harnesses:raise ValueError('Shell requires a frozen absolute harness script, not command templates/options')
        for path in (stage.report,)+stage.outputs:
            _relative(path)
            if path in outputs:raise ValueError('Stages must own distinct output paths')
            outputs.add(path)
    if not p.stages[0].outputs:raise ValueError('Build must produce explicit executable/artifact outputs')
    if len({x.name for x in p.env})!=len(p.env) or any(not isinstance(x.value,str) or x.name.startswith('CAKE_') for x in p.env):
        raise ValueError('Explicit unique string environment; CAKE_ is reserved')
    return True


def _safe_file(work,relative):
    p=work/_relative(relative)
    if p.resolve()!=p or not p.is_file() or p.stat().st_nlink!=1:raise ValueError('Missing/symlink/hardlink output: '+str(p))
    return p


def _product_keys(row):
    return tuple((x.relative_path,x.snapshot.sha256,x.snapshot.size_bytes) for x in row.products)


def _build_key(products,stage,plan):
    entries=[(p.relative_path,p.snapshot.sha256,p.snapshot.size_bytes) for p in products if p.relative_path in stage.outputs]
    return hashlib.sha256(_dump((plan.fingerprint(),entries)).encode()).hexdigest()


def _report(plan,stage,products,build,nonce):
    table={p.relative_path:p for p in products}
    if len(table)!=len(products) or set(table)!={stage.report,*stage.outputs}:raise ValueError('Stage output coverage differs')
    r=StageReport.from_json(Path(table[stage.report].snapshot.path).read_text())
    if (r.plan_fingerprint,r.build_fingerprint,r.kind,r.nonce)!=(plan.fingerprint(),build,stage.kind,nonce) or not r.passed:
        raise ValueError('Stage report identity/contract failed')
    if len({m.name for m in r.metrics})!=len(r.metrics) or any(not m.name or not m.unit or m.value<0 for m in r.metrics):raise ValueError('Invalid metric identities/values')
    if stage.kind=='measure' and not r.metrics:raise ValueError('Measurement has no metrics')
    if stage.kind=='measure' and not {b.metric for b in plan.metric_bindings}.issubset({m.name for m in r.metrics}):raise ValueError('Bound measurement metrics are missing')
    if isinstance(plan.device,AscendLease) and stage.kind=='guard':
        from .ascend import load_profile,verify_guard_audit
        verify_guard_audit(load_profile(plan.device),parse_json(Path(table['guard.audit.json'].snapshot.path).read_text()),plan.device.profile.path)
    return r


def _atomic(path,text):
    tmp=path.with_name(path.name+'.tmp-'+secrets.token_hex(8))
    with tmp.open('x') as stream:stream.write(text+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def _sandbox_argv(p,work,stage,build,attempt,nonce,filter_fd=None):
    hardware=isinstance(p.device,AscendLease) and stage.kind!='build'
    if hardware and filter_fd is None:raise ValueError('Ascend sandbox requires the enforced seccomp fd')
    command=[p.sandbox.path,'--unshare-all','--unshare-user','--die-with-parent','--new-session','--disable-userns','--cap-add' if hardware else '--cap-drop','ALL',
        '--ro-bind','/','/','--dev','/dev','--proc','/proc','--tmpfs','/tmp','--tmpfs','/run','--tmpfs','/root','--tmpfs','/home']
    for item in p.inputs:command+=['--ro-bind',item.path,item.path]
    if hardware:
        from .ascend import load_profile
        profile=load_profile(p.device)
        command+=['--tmpfs','/sys/fs/cgroup','--ro-bind','/proc/sys','/proc/sys',
                  '--ro-bind','/dev/null','/proc/kcore','--ro-bind','/dev/null','/proc/sysrq-trigger',
                  '--seccomp',str(filter_fd)]
        for node in profile.nodes:command+=['--dev-bind',node.path,node.path]
    command+=['--bind',str(work),str(work),'--chdir',str(work),'--clearenv']
    env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'};env.update({x.name:x.value for x in p.env})
    env.update(CAKE_PLAN_FP=p.fingerprint(),CAKE_BUILD_FP=build,CAKE_WORK=str(work),CAKE_NONCE=nonce,CAKE_MODEL_FP=p.model_fingerprint,CAKE_CONTRACT=p.contract)
    if stage.kind not in ('build','check'):env['CAKE_ATTEMPT']=str(attempt)
    for key,value in sorted(env.items()):command+=['--setenv',key,value]
    command+=['--']+[a.replace('{work}',str(work)) for a in stage.argv]
    return command


def _kill(process):
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    process.wait()


def _capture(work,directory,stage):
    products=[]
    for name in (stage.report,)+stage.outputs:
        source=_safe_file(work,name);target=directory/'snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        products.append(Product(name,identify(target,'output')))
    return tuple(products)


def _intact(work,rows):
    for row in rows:
        if row.status not in ('passed','reused'):continue
        for product in row.products:
            current=identify(_safe_file(work,product.relative_path),'output')
            if (current.sha256,current.size_bytes)!=(product.snapshot.sha256,product.snapshot.size_bytes):
                raise ValueError('Previously verified output changed: '+product.relative_path)


def _stage(p,work,directory,stage,build,attempt,previous,cancel):
    directory.mkdir();out=directory/'stdout.log';err=directory/'stderr.log';started=time.time_ns();status='failed';rc=None;error='';products=()
    process=None;nonce=secrets.token_hex(16);conditions=()
    try:
        validate_plan(p);_intact(work,previous)
        if isinstance(p.device,AscendLease) and stage.kind!='build':
            from .ascend import load_profile,capture_occupancy,verify_occupancy
            profile=load_profile(p.device);occupancy=capture_occupancy(profile)
            record=directory/'occupancy.json';record.write_text(occupancy.to_json(pretty=True)+'\n');conditions=(identify(record,'output'),)
            verify_occupancy(profile,occupancy)
        if any((work/x).exists() or (work/x).is_symlink() for x in (stage.report,)+stage.outputs):raise ValueError('Stage output was prewritten')
        with out.open('xb') as stdout,err.open('xb') as stderr:
            filter_fd=None
            try:
                if isinstance(p.device,AscendLease) and stage.kind!='build':
                    from .ascend import load_profile
                    filter_fd=os.open(load_profile(p.device).seccomp.path,os.O_RDONLY|os.O_CLOEXEC)
                process=subprocess.Popen(_sandbox_argv(p,work,stage,build,attempt,nonce,filter_fd),stdout=stdout,stderr=stderr,
                    env={'PATH':'/usr/bin:/bin'},start_new_session=True,close_fds=True,pass_fds=() if filter_fd is None else (filter_fd,))
            finally:
                if filter_fd is not None:os.close(filter_fd)
            deadline=time.monotonic()+stage.timeout_s
            while process.poll() is None:
                if cancel.exists():status='canceled';error='Explicit cancellation';_kill(process);break
                if time.monotonic()>=deadline:status='timeout';error='Stage deadline exceeded';_kill(process);break
                time.sleep(0.02)
            rc=process.wait()
        validate_plan(p);_intact(work,previous)
        if error:raise ValueError(error)
        if rc!=0:raise ValueError('Stage returned '+str(rc))
        products=_capture(work,directory,stage);_report(p,stage,products,build,nonce);status='passed'
    except BaseException as exc:
        if process is not None:_kill(process)
        if isinstance(exc,KeyboardInterrupt):status='canceled'
        error=str(exc) if str(exc) else type(exc).__name__
    for log in (out,err):
        if not log.exists():log.touch(exist_ok=False)
    return StageReceipt(stage.id,stage.kind,status,rc,started,time.time_ns(),identify(out,'log'),identify(err,'log'),products,nonce,error,conditions)


def execute(p:RunPlan,*,resume=False):
    validate_plan(p);root=Path(p.output_dir)
    if resume:
        if RunPlan.from_json((root/'plan.json').read_text())!=p:raise ValueError('Resume plan identity differs')
    else:
        root.mkdir();(root/'plan.json').write_text(p.to_json(pretty=True)+'\n')
    lease=os.open(root/'LOCK',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Run lease is active; do not restart its process')
        prior=None;prior_file=None
        if resume:
            prior=RunReceipt.from_json((root/'latest.json').read_text());prior.verify(allow_incomplete=True);_intact(Path(prior.work_dir),prior.stages)
            name='checkpoint-'+str(len(prior.stages))+'.json' if prior.state=='incomplete' else 'receipt.json'
            prior_file=identify(root/('attempt-'+str(prior.attempt))/name,'output')
            if RunReceipt.from_json(Path(prior_file.path).read_text())!=prior:raise ValueError('Latest pointer differs from immutable checkpoint')
        device_fd=None
        if isinstance(p.device,AscendLease):
            from .ascend import device_lease,load_profile
            device_fd=device_lease(load_profile(p.device))
        attempt=0 if prior is None else prior.attempt+1
        directory=root/('attempt-'+str(attempt));directory.mkdir();work=directory/'work';work.mkdir()
        rows=[];transitions=['planned'];build='';error=''
        def checkpoint():
            receipt=RunReceipt(p,attempt,str(work),'incomplete',tuple(transitions),tuple(rows),build,prior_file,'')
            path=directory/('checkpoint-'+str(len(rows))+'.json');path.write_text(receipt.to_json(pretty=True)+'\n')
            _atomic(root/'latest.json',receipt.to_json(pretty=True))
        checkpoint()
        for stage in p.stages:
            cached=None
            if prior is not None and p.cache_cpu and stage.kind in ('build','check'):
                cached=next((r for r in prior.stages if r.stage==stage.id and r.status in ('passed','reused')),None)
            if cached is not None:
                _intact(work,rows)
                for product in cached.products:
                    product.snapshot.verify();target=work/product.relative_path;target.parent.mkdir(parents=True,exist_ok=True)
                    with Path(product.snapshot.path).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
                stepdir=directory/stage.id;stepdir.mkdir();products=_capture(work,stepdir,stage)
                out=stepdir/'stdout.log';err=stepdir/'stderr.log';shutil.copyfile(cached.stdout.path,out);shutil.copyfile(cached.stderr.path,err)
                row=StageReceipt(stage.id,stage.kind,'reused',0,time.time_ns(),time.time_ns(),identify(out,'log'),identify(err,'log'),products,cached.nonce)
                _report(p,stage,products,build,cached.nonce)
            else:row=_stage(p,work,directory/stage.id,stage,build,attempt,rows,root/'CANCEL')
            rows.append(row)
            if row.status not in ('passed','reused'):error=row.error;break
            if stage.kind=='build':build=_build_key(row.products,stage,p)
            transitions.append(STATES[stage.kind])
            if len(rows)<len(p.stages):checkpoint()
        receipt=RunReceipt(p,attempt,str(work),'failed' if error else 'verified',tuple(transitions),tuple(rows),build,prior_file,error)
        if receipt.state=='verified':
            try:receipt.verify()
            except (OSError,ValueError) as exc:
                error='Final identity verification failed: '+str(exc)
                rows[-1]=replace(rows[-1],status='failed',error=error)
                receipt=RunReceipt(p,attempt,str(work),'failed',tuple(transitions[:-1]),tuple(rows),build,prior_file,error)
        path=directory/'receipt.json';path.write_text(receipt.to_json(pretty=True)+'\n');_atomic(root/'latest.json',receipt.to_json(pretty=True))
        return receipt
    finally:
        if 'device_fd' in locals() and device_fd is not None:os.close(device_fd)
        os.close(lease)


def cancel_run(p):
    validate_plan(p);root=Path(p.output_dir)
    if RunPlan.from_json((root/'plan.json').read_text())!=p:raise ValueError('Cancel target plan differs')
    with (root/'CANCEL').open('x') as stream:stream.write(p.fingerprint()+'\n')


def cli(args):
    import sys
    try:
        if args.output and (Path(args.output).exists() or Path(args.output).is_symlink()):raise ValueError('Output already exists; execution was not started')
        if args.selftest:
            if args.resume or args.output or args.cancel:raise ValueError('Selftest does not resume, cancel or write a run')
            from .runner_test import selftest
            return selftest()
        if args.verify:
            if args.resume or args.output or args.cancel:raise ValueError('Verification cannot override execution')
            receipt=RunReceipt.from_json(Path(args.verify).read_text());receipt.verify();text=_dump(dict(status='identity_verified',run_state=receipt.state),pretty=True);rc=0
        else:
            p=RunPlan.from_json(Path(args.plan).read_text())
            if args.cancel:
                if args.resume or args.output:raise ValueError('Cancellation is a separate action')
                cancel_run(p);text=_dump(dict(status='cancel_requested'));rc=0
            else:
                receipt=execute(p,resume=args.resume);text=receipt.to_json(pretty=True);rc=0 if receipt.state=='verified' else 1
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return rc
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr);return 2


def _nc_no_early_measurement():
    import sys
    base=Path(__file__).parent
    files=tuple(identify(base/name,role) for name,role in (('runner.py','source'),('architecture.py','harness'),('__init__.py','input'),('campaign.py','runtime')))+(identify(sys.executable,'tool'),)
    plan=RunPlan('nc','/tmp/nc-runner','0'*64,'contract','fixture',(Attribute('cpu','fixture'),),files,identify(sys.executable,'tool'),tool_fingerprint(),DeviceLease('cpu'),(Stage('early','measure',(sys.executable,),'report.json'),))
    try:validate_plan(plan)
    except ValueError as exc:return 'ordered' in str(exc)
    return False


REFS={'RUN1_identity_and_stage_order':Ref(
    what=['阶段顺序/内容身份/只读保护/最终rc与报告共同约束执行；恢复刷新guard/measure/verify，CPU缓存需显式确定性/可重定位契约。'],
    status='derived',evidence=['E107 runner_test namespace/fault/resume/timeout/identity controls; NPU adapter and delivery promotion remain pending.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'RUN1_identity_and_stage_order':_nc_no_early_measurement}
