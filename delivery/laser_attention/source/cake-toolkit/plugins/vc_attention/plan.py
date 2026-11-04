"""Freeze the existing attention workload into an identity-bound Ascend plan.

This first adapter has one explicit E97 target envelope, not arbitrary models.
No NPU workload is launched while inspecting the stack or constructing a plan.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,os,sys,tarfile
from core.opgen.architecture import (Architecture,Attribute,Value,SemanticProgram,Contract,Storage,Resource,Allocation,Instance,Layout,Event,Region,Demand,Binding,Evidence,Platform)
from core.opgen.runner import FileIdentity,AscendLease,RunPlan,Stage,identify,tool_fingerprint,validate_plan,MetricBinding
from core.opgen.ascend import AscendProfile,freeze_nodes,guard_argv
from core.opgen.ascend_sandbox import filter_bytes


def inspect_stack():
    import torch,importlib.metadata,importlib.util
    keys=('HOME','PATH','LD_LIBRARY_PATH','PYTHONPATH','ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_AICPU_PATH','ASCEND_TOOLKIT_HOME','CMAKE_PREFIX_PATH','SOC_VERSION','OMP_NUM_THREADS','MAX_JOBS','VIRTUAL_ENV')
    return dict(python=sys.executable,runtime=dict(torch=torch.__version__,torch_npu=importlib.metadata.version('torch_npu'),abi=int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        env={k:os.environ[k] for k in keys if k in os.environ},torch_root=os.path.dirname(torch.__file__),
        torch_npu_root=os.path.dirname(importlib.util.find_spec('torch_npu').origin),guard_package=str(Path(importlib.util.find_spec('vc_attention_ascend').origin).resolve()))


def build_plan(project,output,card,stack,sandbox,lowering=None,*,candidate_kind='existing_E97'):
    project=Path(project).resolve();out=Path(output).resolve();cake=Path(__file__).resolve().parents[2]
    if card<=0:raise ValueError('Physical NPU0 or negative card is forbidden')
    if candidate_kind not in ('existing_E97','online_top8','window_top8','inline_pv'):raise ValueError('Unknown attention candidate kind')
    if candidate_kind!='existing_E97' and lowering is not None:raise ValueError('Online producer requires its own lowering, not the E97 certificate')
    online=candidate_kind!='existing_E97'
    online_manifest='sparse_inline_pv_cann920_build.json' if candidate_kind=='inline_pv' else 'sparse_window_top8_cann920_build.json' if candidate_kind=='window_top8' else 'sparse_online_top8_cann920_build.json'
    if not project.is_dir():raise ValueError('Operator project missing')
    out.mkdir()
    python=stack['python'];cann=Path(stack['env']['ASCEND_HOME_PATH']).resolve()
    tag=out.name
    filterpath=out/'sandbox.bpf';filterpath.write_bytes(filter_bytes())
    profile=AscendProfile(card,card,python,str(project),str(cake),str(cann),'9.2.0','Ascend910B2',identify(cake/'core/_devprobe.json','harness'),identify('/usr/local/sbin/npu-smi','tool'),freeze_nodes(card),identify(stack['guard_package'],'harness'),identify(Path(stack['guard_package']).with_name('libvc_ascend_ops.so'),'runtime'),identify(filterpath,'tool'))
    profilepath=out/'profile.json';profilepath.write_text(profile.to_json(pretty=True))
    shape=(1,1,38139,128);size=2*38139*128;reference=project/'scripts/mindie_laser_sparse_finalize.py';refsha=hashlib.sha256(reference.read_bytes()).hexdigest()
    contract=Contract('block1-top8-online' if online else 'block1-top8-existing-E97','approximate',str(reference)+'#finalize_reference@sha256:'+refsha,
        'BF16 [1,1,38139,128] std5 target; block1/global-top8 and FP32 stable softmax/half weights/native FP32 PV/public cast'+('; online tie relation: descending scores exact, IDs belong to original per-block maxima, torch.TopK ties may differ' if online else ''),
        'Own-weight FP64 PV rel<=1e-5 abs<=1e-4; producer/TopK/packet/counters and public conversion checked; dense sampled diagnostic and video unverified')
    values=tuple(Value(v,'input',shape,'bf16') for v in ('q','k','v'))+(Value('out','block1_top8_attention',shape,'bf16',('q','k','v'),(Attribute('block',128),Attribute('candidates_per_block',1),Attribute('rank',8)),str(reference)+'@'+refsha),)
    layout=Layout(shape,(size,size,256,2))
    model=Architecture('online-top8-complete-api-boundary' if online else 'E97-complete-api-boundary',SemanticProgram(values,('q','k','v'),('out',),contract),(Storage('public','GM','global',None),),(Resource('callable','host',1,Evidence('assumed','single-call API boundary; not hardware pipe capacity')),),
        tuple(Allocation(v,'public',size) for v in ('q','k','v','out')),tuple(Instance(v,v,v,layout) for v in ('q','k','v','out')),
        (Event('complete_api','host','compute',tuple(Region(v,0,size) for v in ('q','k','v')),(Region('out',0,size),),(Demand('callable'),),('out',)),),(),
        tuple(Binding(v,() if v!='out' else ('complete_api',),(v,)) for v in ('q','k','v','out')),platform=Platform('Ascend910B2'),label='Public API boundary only. Internal layout/workspace/scheduling remains explicitly unknown; no ideal-pipeline certificate')
    implementation=None
    if lowering is not None:
        from core.opgen.lowering import LoweringPlan
        implementation=identify(lowering,'input');bound=LoweringPlan.from_json(Path(implementation.path).read_text());bound.verify()
        if bound.device!=card:raise ValueError('Lowering physical device differs')
        from .architecture import verify_native_declaration
        verify_native_declaration(bound,json.loads((project/'results/mindie_laser_sparse_finalize_cann920_build.json').read_text()))
        model=bound.architecture;contract=model.semantic.contract
    modelpath=out/'model.json';modelpath.write_text(model.to_json(pretty=True))
    manifestnames=['mindie_laser_cann920_build.json','mindie_laser_cann920_bridge.json','mindie_laser_sparse_finalize_cann920_build.json','mindie_laser_sparse_candidates_score_chunk_cann920_build.json','mindie_laser_sparse_pv_cann920_build.json']
    if online:manifestnames.append(online_manifest)
    files={}
    def add(path,role):
        path=Path(path)
        if path.is_file():
            item=identify(path,role);files[item.path]=item
    # Source and evaluation harness are read-only and version-bound.
    for path in (project/'scripts').glob('*.py'):add(path,'harness')
    for path in (project/'ascend/vc_attention_ascend').glob('*.py'):add(path,'harness')
    for path in Path(profile.package.path).parent.glob('*.py'):add(path,'harness')
    add(profile.native.path,'runtime')
    for path in (cake/'plugins/vc_attention').glob('*.py'):add(path,'harness')
    for path in (cake/'core/devguard.py',cake/'core/_refs.py'):add(path,'harness')
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('path','library_path','bridge_path') and isinstance(item,str) and Path(item).is_file():
                    suffix=Path(item).suffix;add(item,'source' if suffix in ('.h','.cpp','.txt','.cmake') else 'harness' if suffix=='.py' else 'runtime' if '.so' in Path(item).name else 'input')
                visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for name in manifestnames:
        file=project/'results'/name;add(file,'input');visit(json.loads(file.read_text()))
    old=json.loads((project/'results/laser_cann920_e97_finalize_long.json').read_text())
    libraries=[]
    for mapping in (old['loaded_library_sha256'],old['candidate_mapped_library_sha256']):
        for path,sha in mapping.items():
            if identify(path,'runtime').sha256!=sha:raise ValueError('Existing provider differs from frozen E97: '+path)
            libraries.append(dict(path=path,sha256=sha));add(path,'runtime')
    if online:
        native=json.loads((project/'results'/online_manifest).read_text())
        item=identify(native['library_path'],'runtime')
        if item.sha256!=native['library_sha256']:raise ValueError('Online library differs from manifest')
        libraries.append(dict(path=item.path,sha256=item.sha256))
    for path in (cann/'opp/version.info',cann/'set_env.sh',Path(stack['torch_root'])/'__init__.py',Path(stack['torch_npu_root'])/'__init__.py',Path(python).parent.parent/'pyvenv.cfg'):
        add(path,'runtime')
    # Bind critical actual runtime binaries, not just version strings.
    for path in (Path(stack['torch_root'])/'lib/libtorch_cpu.so',Path(stack['torch_root'])/'lib/libc10.so',Path(stack['torch_npu_root'])/'lib/libtorch_npu.so'):
        add(path,'runtime')
    add(python,'tool');add(profilepath,'input');add(profile.registry.path,'harness');add(profile.smi.path,'tool');add(profile.seccomp.path,'tool');add(modelpath,'input')
    # Actual compiled source files provide the source role (no code is rebuilt).
    if not any(i.role=='source' for i in files.values()):raise ValueError('No compiled source identities found')
    if implementation is not None:
        add(implementation.path,'input')
        for item in bound.sources:files[item.path]=item
        for entry in bound.entries:files[entry.library.path]=entry.library
    config=dict(project_root=str(project),cann_root=str(cann),model=str(modelpath),device=card,shape=list(shape),std=5,seed=0,warmup=3,repeats=10,trials=9,
        candidate_manifest=str(project/'results'/(online_manifest if online else 'mindie_laser_sparse_finalize_cann920_build.json')),reference_manifest=str(project/'results'/('mindie_laser_sparse_finalize_cann920_build.json' if online else 'mindie_laser_sparse_candidates_score_chunk_cann920_build.json')),pv_manifest=str(project/'results/mindie_laser_sparse_pv_cann920_build.json'),runtime=stack['runtime'],native_libraries=libraries,
        identities=[dict(path=i.path,sha256=i.sha256) for i in files.values() if i.role in ('source','harness','runtime')])
    if online:config['host_only_trace_variants']=['candidate_selection']
    if implementation is not None:config['lowering']=implementation.path
    configpath=out/'attention.json';configpath.write_text(json.dumps(config,indent=2));add(configpath,'input')
    env=dict(stack['env']);env['PYTHONPATH']=str(cake)+':'+env.get('PYTHONPATH','');env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONFAULTHANDLER']='1'
    stages=[]
    for name,outputs in [('build',('native.manifest.json',)),('guard',('guard.audit.json',)),('check',('check.raw.tar','check.summary.json')),('measure',('measure.raw.tar','measure.summary.json')),('verify',())]:
        argv=guard_argv(profile,str(profilepath),'guard.json') if name=='guard' else (python,'-m','plugins.vc_attention.execution','--config',str(configpath),'--stage',name)
        stages.append(Stage(name,name,argv,name+'.json',outputs,600 if name=='guard' else 720))
    plan=RunPlan(tag,str(out/'run'),model.fingerprint(),contract.id,json.dumps(dict(shape=list(shape),dtype='bfloat16',std=5,seed=0)),
        (Attribute('soc','Ascend910B2'),Attribute('cann_version','9.2.0'),Attribute('torch',stack['runtime']['torch']),Attribute('torch_npu',stack['runtime']['torch_npu']),Attribute('abi',stack['runtime']['abi'])),tuple(files.values()),identify(sandbox,'tool'),tool_fingerprint(),AscendLease(identify(profilepath,'input')),tuple(stages),tuple(Attribute(k,v) for k,v in sorted(env.items())),False,implementation,(MetricBinding('candidate_device_task_sum',model.fingerprint(),'complete prepared attention, device task sum'),)+( (MetricBinding('previous_device_task_sum',bound.transform.versions[0].fingerprint(),'complete prepared attention, device task sum'),) if implementation else ()))
    validate_plan(plan);(out/'plan.json').write_text(plan.to_json(pretty=True))
    # Save exact mutable source/tool bytes before running. Runtime binaries retain
    # explicit identities and require the matching environment for later replay.
    snapshot={i.path:i for i in plan.inputs if i.role in ('source','harness','tool')}
    for path in (cake/'core/opgen').glob('*.py'):
        item=identify(path,'tool');snapshot[item.path]=item
    with tarfile.open(out/'sources.tar','w') as stream:
        for path in sorted(snapshot):stream.add(path,arcname=path.lstrip('/'),recursive=False)
    (out/'sources.json').write_text(json.dumps([asdict(i) for i in snapshot.values()],indent=2))
    return plan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inspect-stack',action='store_true');ap.add_argument('--project');ap.add_argument('--stack');ap.add_argument('--card',type=int);ap.add_argument('--output');ap.add_argument('--sandbox');ap.add_argument('--lowering');ap.add_argument('--candidate-kind',choices=('existing_E97','online_top8','window_top8','inline_pv'),default='existing_E97');args=ap.parse_args()
    if args.inspect_stack:
        print(json.dumps(inspect_stack(),indent=2));return 0
    if any(v is None for v in (args.project,args.stack,args.card,args.output,args.sandbox)):ap.error('plan requires --project --stack --card --output --sandbox')
    plan=build_plan(args.project,args.output,args.card,json.loads(Path(args.stack).read_text()),args.sandbox,args.lowering,candidate_kind=args.candidate_kind)
    print(json.dumps(dict(plan=str(Path(plan.output_dir).parent/'plan.json'),inputs=len(plan.inputs),input_bytes=sum(i.size_bytes for i in plan.inputs),model=plan.model_fingerprint,device=args.card),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
