"""P5 explicit Ascend binding, parent-visible occupancy and unchanged probes."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse,fcntl,json,os,re,runpy,shlex,stat,subprocess,sys,time
from .._refs import Ref
from .architecture import Record,VersionedRecord,_dump,parse_json
from .runner import FileIdentity,StageReport,identify
from .ascend_sandbox import POLICY,validate_filter


PROBES=('vc910_card_clean','vc910_card_stable','vc910_card_determinism')
SCRIPTS=('smoke_one.py','card_stability_probe.py','determinism_sweep.py')
REGEXES=('RMSE 0\\.000000e\\+00','不稳定 0 次','1 个形状里 0 个不确定')


@dataclass(frozen=True)
class DeviceNode(Record):
    path: str
    major: int
    minor: int


@dataclass(frozen=True)
class AscendProfile(VersionedRecord):
    physical_card: int
    logical_card: int
    python: str
    project_root: str
    toolkit_root: str
    cann_root: str
    cann_version: str
    soc: str
    registry: FileIdentity
    smi: FileIdentity
    nodes: tuple[DeviceNode,...]
    package: FileIdentity
    native: FileIdentity
    seccomp: FileIdentity | None = None
    sandbox_policy: str = POLICY
    schema_version: int = 1


@dataclass(frozen=True)
class Occupancy(VersionedRecord):
    physical_card: int
    captured_ns: int
    info_returncode: int
    info: str
    info_stderr: str
    mapping_returncode: int
    mapping: str
    mapping_stderr: str
    schema_version: int = 1


def freeze_nodes(card):
    paths=('/dev/davinci'+str(card),'/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc')
    result=[]
    for path in paths:
        st=os.lstat(path)
        if not stat.S_ISCHR(st.st_mode):raise ValueError('Expected literal device node: '+path)
        result.append(DeviceNode(path,os.major(st.st_rdev),os.minor(st.st_rdev)))
    return tuple(result)


def load_profile(binding):
    binding.profile.verify()
    p=AscendProfile.from_json(Path(binding.profile.path).read_text())
    if p.schema_version!=1 or p.physical_card<=0:raise ValueError('Physical NPU0 or negative card is forbidden')
    if p.logical_card!=p.physical_card:raise ValueError('This adapter preserves physical ordinals; remapping is unsupported')
    if p.soc!='Ascend910B2' or p.cann_version!='9.2.0':raise ValueError('This validated profile requires Ascend910B2/CANN9.2.0')
    if not Path(p.cann_root).is_dir():raise ValueError('CANN root missing')
    versions=[line.removeprefix('Version=') for line in (Path(p.cann_root)/'opp/version.info').read_text().splitlines() if line.startswith('Version=')]
    if versions!=[p.cann_version]:raise ValueError('Actual CANN version differs from profile')
    p.registry.verify();p.smi.verify();p.package.verify();p.native.verify()
    if Path(p.package.path).name!="__init__.py" or Path(p.native.path).name!="libvc_ascend_ops.so" or Path(p.package.path).parent!=Path(p.native.path).parent:
        raise ValueError("Guard package/native locations differ")
    if p.nodes!=freeze_nodes(p.physical_card):raise ValueError('Device node identity/set changed')
    if p.sandbox_policy!=POLICY:raise ValueError('Unsupported Ascend sandbox policy')
    validate_filter(p.seccomp)
    return p


def probe_argv(p,label):
    raw=parse_json(Path(p.registry.path).read_text());index=PROBES.index(label);spec=raw[label]
    tokens=shlex.split(spec['cmd'])
    if tokens[:4]!=['cd',p.project_root,'&&','python3'] or tokens[4]!='scripts/'+SCRIPTS[index]:
        raise ValueError('Registered probe launcher changed; audit before use')
    if spec['expect_rc']!=0 or spec['expect_regex']!=REGEXES[index]:raise ValueError('Registered probe criterion changed')
    tail=[x.replace('{card}',str(p.physical_card)) for x in tokens[5:]]
    if '--device' not in tail or tail[tail.index('--device')+1]!=str(p.physical_card):raise ValueError('Probe device is not bound')
    return (p.python,str(Path(p.project_root)/tokens[4]),*tail)


def guard_argv(p,profile_path,report):
    return (p.python,'-m','core.opgen.ascend_cli','--guard-profile',profile_path,'--report',report)


def probe_launcher_argv(p,profile_path,label):
    return (p.python,'-m','core.opgen.ascend_cli','--probe-profile',profile_path,'--probe-label',label)


def native_binding(p):
    return dict(profile_fingerprint=p.fingerprint(),physical_card=p.physical_card,package=p.package.path,native=p.native.path,native_sha256=p.native.sha256)


def bootstrap_probe(p):
    # Native static registration queries aclrtGetSocName. It must not depend on
    # an implicit default device before the requested physical card is bound.
    if p.physical_card<=0 or p.logical_card!=p.physical_card:raise ValueError('Physical NPU0/remapping forbidden in bootstrap')
    import torch,torch_npu,vc_attention_ascend as vc
    p.package.verify();p.native.verify()
    if str(Path(vc.__file__).resolve())!=p.package.path:raise ValueError('Guard imported a different installed package')
    torch.npu.set_device(p.physical_card)
    vc.load_native()
    mapped={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if len(line.split())>=6}
    if p.native.path not in mapped:raise ValueError('Guard native provider was not mapped')
    return native_binding(p)


def run_probe(p,label):
    binding=bootstrap_probe(p)
    print('CAKE_GUARD_NATIVE='+json.dumps(binding,sort_keys=True),flush=True)
    argv=probe_argv(p,label);sys.argv=list(argv[1:])
    runpy.run_path(argv[1],run_name='__main__')
    return 0


def validate_binding(plan):
    p=load_profile(plan.device);env={a.name:a.value for a in plan.env}
    if any(env.get(key) for key in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES')):
        raise ValueError('Physical-preserving adapter rejects visibility remapping')
    if Path(str(env.get('ASCEND_HOME_PATH',''))).resolve()!=Path(p.cann_root).resolve():raise ValueError('Plan CANN environment differs')
    if str(env.get('PYTHONPATH','')).split(':')[0]!=p.toolkit_root:raise ValueError('Trusted guard module must resolve from pinned toolkit first')
    if len(plan.stages)!=5:raise ValueError('Ascend execution requires all five stages')
    guard=plan.stages[1]
    if guard.argv!=guard_argv(p,plan.device.profile.path,guard.report) or guard.outputs!=('guard.audit.json',):
        raise ValueError('Ascend guard cannot be replaced by an arbitrary command/report')
    required={plan.device.profile.path,p.registry.path,p.smi.path,p.package.path,p.native.path,p.seccomp.path,str(Path(p.cann_root)/'opp/version.info'),*[str(Path(p.project_root)/'scripts'/s) for s in SCRIPTS]}
    if not required.issubset({i.path for i in plan.inputs}):raise ValueError('Profile/probe inputs are not frozen in run identity')
    for label in PROBES:probe_argv(p,label)
    if plan.cache_cpu:raise ValueError('NPU checks/guards are always fresh in this adapter')
    return p


def parse_occupancy(info,mapping,card):
    """Require explicit health and complete per-card process accounting."""
    parts=info.split('Process id')
    if len(parts)!=2:raise ValueError('NPU process table missing or ambiguous')
    health=re.findall(r'\|\s*'+str(card)+r'\s+910B2\s*\|\s*([^|]+)\|',parts[0])
    if len(health)!=1 or health[0].strip()!='OK':raise ValueError('Card missing or not healthy')
    empty=re.findall(r'No running processes found in NPU\s+(\d+)',parts[1])
    processes=re.findall(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|',parts[1])
    for line in parts[1].splitlines():
        prefix=re.match(r'\|\s*(\d+)\s+(\d+)\s*\|',line)
        if prefix and int(prefix[1])==card and not re.match(r'\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|',line):raise ValueError('Selected-card process row is malformed')
    rows=[int(pid) for npu,chip,pid in processes if int(npu)==card]
    if rows:raise ValueError('Physical card is occupied: '+str(rows))
    if empty.count(str(card))!=1:raise ValueError('No explicit unique empty-card record')
    mappings=re.findall(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+Ascend 910B2\s*$',mapping,re.M)
    found=[int(logic) for npu,chip,logic in mappings if int(npu)==card and int(chip)==0]
    if found!=[card]:raise ValueError('Physical/logical mapping missing or changed')
    return True


def capture_occupancy(p):
    first=subprocess.run([p.smi.path,'info'],capture_output=True,text=True,timeout=30)
    second=subprocess.run([p.smi.path,'info','-m'],capture_output=True,text=True,timeout=30)
    return Occupancy(p.physical_card,time.time_ns(),first.returncode,first.stdout,first.stderr,second.returncode,second.stdout,second.stderr)


def verify_occupancy(p,o):
    if o.physical_card!=p.physical_card or o.info_returncode!=0 or o.mapping_returncode!=0:raise ValueError('Occupancy query identity/exit status failed')
    return parse_occupancy(o.info,o.mapping,p.physical_card)


def device_lease(p):
    # 本仓自定: cooperative same-container lock; occupancy remains independent.
    root=Path('/tmp/cake-opgen-npu-leases');root.mkdir(mode=0o700,exist_ok=True)
    if root.is_symlink() or root.stat().st_uid!=os.getuid():raise ValueError('Untrusted device lease directory')
    fd=os.open(root/('physical-'+str(p.physical_card)+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:os.close(fd);raise
    return fd


def verify_guard_audit(p,audit,profile_path):
    if audit['physical_card']!=p.physical_card or audit['profile_fingerprint']!=p.fingerprint():raise ValueError('Guard audit profile mismatch')
    rows=audit['probes']
    if [r['label'] for r in rows]!=list(PROBES):raise ValueError('Guard audit is incomplete or reordered')
    for row in rows:
        index=PROBES.index(row['label'])
        if row['launch_argv']!=list(probe_launcher_argv(p,profile_path,row['label'])) or row['native']!=native_binding(p):
            raise ValueError('Guard bootstrap/native identity differs')
        if row['argv']!=list(probe_argv(p,row['label'])) or row['returncode']!=0 or not re.search(REGEXES[index],row['stdout']+row['stderr']):
            raise ValueError('Known artifact guard did not reproduce its criterion')
    return True


def capture_probe(p,label,profile_path):
    argv=probe_argv(p,label);launcher=probe_launcher_argv(p,profile_path,label)
    try:
        result=subprocess.run(launcher,cwd=p.project_root,capture_output=True,text=True,timeout=180)
        rc,stdout,stderr=result.returncode,result.stdout,result.stderr;timed_out=False
    except subprocess.TimeoutExpired as error:
        rc,stdout,stderr=None,error.stdout or '',error.stderr or '';timed_out=True
        if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
        if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
    markers=[line.removeprefix('CAKE_GUARD_NATIVE=') for line in stdout.splitlines() if line.startswith('CAKE_GUARD_NATIVE=')]
    binding=parse_json(markers[0]) if len(markers)==1 else None
    return dict(label=label,argv=list(argv),launch_argv=list(launcher),native=binding,returncode=rc,timed_out=timed_out,stdout=stdout,stderr=stderr)


def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--guard-profile');group.add_argument('--probe-profile');ap.add_argument('--probe-label',choices=PROBES);ap.add_argument('--report');args=ap.parse_args()
    if args.probe_profile:
        if not args.probe_label:ap.error('--probe-profile needs --probe-label')
        return run_probe(AscendProfile.from_json(Path(args.probe_profile).read_text()),args.probe_label)
    if not args.report:ap.error('--guard-profile needs --report')
    p=AscendProfile.from_json(Path(args.guard_profile).read_text());p.registry.verify()
    audit=dict(physical_card=p.physical_card,profile_fingerprint=p.fingerprint(),probes=[])
    work=Path(os.environ['CAKE_WORK'])
    for key in ('ASCEND_CACHE_PATH','ACL_OP_COMPILER_CACHE_DIR','ASCEND_PROCESS_LOG_PATH'):
        value=work/key.lower();value.mkdir(exist_ok=True);os.environ[key]=str(value)
    ok=True
    for label in PROBES:
        argv=probe_argv(p,label);print('guard '+label+' physical='+str(p.physical_card),flush=True)
        row=capture_probe(p,label,args.guard_profile)
        audit['probes'].append(row);print(row['stdout'],flush=True);print(row['stderr'],flush=True)
        (work/'guard.audit.json').write_text(_dump(audit,pretty=True)+'\n')
        index=PROBES.index(label)
        if row['returncode']!=0 or not re.search(REGEXES[index],row['stdout']+row['stderr']):ok=False;break
    if ok:verify_guard_audit(p,audit,args.guard_profile)
    report=StageReport(os.environ['CAKE_PLAN_FP'],os.environ['CAKE_BUILD_FP'],'guard',ok,os.environ['CAKE_NONCE'])
    Path(args.report).write_text(report.to_json(pretty=True)+'\n')
    return 0 if ok else 1


def _nc_unknown_is_not_idle():
    try:parse_occupancy('unavailable','unavailable',5)
    except ValueError:return True
    return False


REFS={'ASCEND1_parent_occupancy_and_artifacts':Ref(
    what=['物理映射/节点/父域完整占用表及原三守卫绑定；未知不当空闲，NPU0和任意guard旁路拒绝。'],
    status='derived',evidence=['E108 ascend_test parser/binding negative controls and real guarded attention driver validation; hardware availability remains fresh.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'ASCEND1_parent_occupancy_and_artifacts':_nc_unknown_is_not_idle}

if __name__=='__main__':raise SystemExit(main())
