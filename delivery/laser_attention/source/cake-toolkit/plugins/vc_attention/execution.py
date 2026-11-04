"""Protected adapter around the existing E97 attention worker, not new kernels."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,subprocess,sys,tarfile
from core.opgen.architecture import Architecture,parse_json,_dump
from core.opgen.runner import Metric,StageReport


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_config(c):
    model=Architecture.from_json(Path(c['model']).read_text())
    if model.fingerprint()!=os.environ['CAKE_MODEL_FP'] or model.semantic.contract.id!=os.environ['CAKE_CONTRACT']:
        raise ValueError('Actual model/contract file is not bound to the run')
    if c['shape']!=[1,1,38139,128] or c['std']!=5:raise ValueError('This driver validates the requested target envelope')
    for item in c['identities']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Attention dependency changed: '+item['path'])
    if Path(os.environ.get('ASCEND_HOME_PATH','')).resolve()!=Path(c['cann_root']).resolve():raise ValueError('Attention CANN root differs')
    if c.get('lowering'):
        from core.opgen.lowering import LoweringPlan
        plan=LoweringPlan.from_json(Path(c['lowering']).read_text());plan.verify()
        if plan.architecture!=model or plan.device!=c['device']:raise ValueError('Actual lowering/model/device differs')
        from .architecture import verify_native_declaration
        verify_native_declaration(plan,parse_json(Path(c['candidate_manifest']).read_text()))
    return model


def validate_report(c,r,*,timed):
    if r.get('status')!='passed' or r.get('returncode')!=0:raise ValueError('Original attention worker failed: '+str(r.get('error')))
    if not r.get('finalize_checks'):raise ValueError('Native finalize checks are missing')
    if not timed:return {}
    if (r['shape'],r['std'],r['seed'])!=(c['shape'],c['std'],c['seed']):raise ValueError('Actual workload differs')
    groups=r['paired_groups']['attention']
    if groups['completed_trials']!=c['trials']:raise ValueError('Incomplete paired timing')
    mapped=r['loaded_library_sha256'];candidate_mapped=r['candidate_mapped_library_sha256']
    for item in c['native_libraries']:
        actual=mapped.get(item['path'],candidate_mapped.get(item['path']))
        if actual!=item['sha256']:raise ValueError('Actual native provider differs: '+item['path'])
    summary=dict(scope='Declared candidate/previous/Laser; profiler device task sum/union, paired synchronized wall API and full/preparation kept distinct',
        shape=r['shape'],std=r['std'],seed=r['seed'],video='unverified',device={},paired_api={},stages={},source_report_sha256=None,observations={})
    variants=r['prepared_trace']['variants']
    for name,v in variants.items():
        if not v['capture_finalized'] or v['submitted_calls']<=0:raise ValueError('Incomplete device capture')
        if 'device_summary' not in v:
            # An eliminated stage may legitimately have only CPU trace events.
            # Absence of a device table is NOT a measured zero device duration.
            allowed=c.get('host_only_trace_variants',[])
            if (allowed!=['candidate_selection'] or name!='candidate_selection' or v.get('status')!='raw_only'
                    or not v.get('trace_files') or v.get('kernel_table_files')!=[] or v.get('error')
                    or v.get('reason')!='No kernel_details.csv; no device timings inferred'):
                raise ValueError('Missing device summary for undeclared or incomplete capture: '+name)
            summary['observations'][name]=dict(scope='Declared host-only selection; finalized CPU trace only, no device timing inferred',
                device_timing='unavailable',submitted_calls=v['submitted_calls'],trace_files=v['trace_files'])
            continue
        d=v['device_summary'];timeline=d['device_timeline']
        if timeline['status']!='verified' or set(timeline['devices'])!={str(c['device'])}:raise ValueError('Profiler device identity differs')
        summary['device'][name]=dict(task_sum_ms=d['all_device_tasks_sum_ms_per_call'],
            union_ms=timeline['devices'][str(c['device'])]['union_ms_per_call'],
            tasks_per_call=d['device_tasks']/v['submitted_calls'],capture_span_ms=timeline['devices'][str(c['device'])]['capture_span_ms'])
        if name in groups['rows']:summary['paired_api'][name]=groups['rows'][name]
    summary['stages']=r['timing_ms'];summary['target_check']=r['target_check']
    summary['candidate_over_laser_device_sum']=summary['device']['candidate']['task_sum_ms']/summary['device']['laser']['task_sum_ms']
    return summary


def audit_architecture(c,r,data):
    from core.opgen.lowering import LoweringPlan,LoweringReceipt
    from .architecture import attest_boundary,audit_capture
    plan=LoweringPlan.from_json(Path(c['lowering']).read_text());plan.verify()
    receipt=attest_boundary(plan,r.get('architecture_boundary'),r['candidate_mapped_library_sha256'])
    info=r.get('architecture_binding',{})
    if info.get('model')!=plan.architecture.fingerprint() or info.get('status')!=receipt.status:raise ValueError('Original worker did not bind the model before timing')
    path=Path(info['path'])
    if not path.resolve().is_relative_to(data.resolve()) or digest(path)!=info['sha256']:raise ValueError('Native ABI receipt path/identity differs')
    actual=LoweringReceipt.from_json(path.read_text());actual.verify()
    if actual!=receipt:raise ValueError('Native ABI replay differs')
    mappings={}
    for name in ('candidate','previous'):
        audit=audit_capture(r['prepared_trace']['variants'][name],name,c['device'],data/('architecture-'+name))
        mappings[name]=dict(ai_tasks=len(audit['ai_tasks']),control_tasks=len(audit['hardware_controls']),flow_events=len(audit['hardware_flows']),
            raw_records=audit['raw_records'],unsupported=audit['unsupported'],groups=audit['groups'],mapping_sha256=digest(data/('architecture-'+name)/'mapping.json'))
    return dict(model=plan.architecture.fingerprint(),scope=plan.scope,boundary_receipt=info['sha256'],mappings=mappings,
        warning='Sample addresses belong to target_validation only; pattern-derived invocation groups do not prove per-pipe ideal scheduling')


def verify_archived_architecture(c,r,archivepath):
    # Replay receipt from archived bytes, not a mutable report fingerprint alone.
    from core.opgen.lowering import LoweringPlan,LoweringReceipt
    from .architecture import attest_boundary
    plan=LoweringPlan.from_json(Path(c['lowering']).read_text());plan.verify();expected=attest_boundary(plan,r.get('architecture_boundary'),r['candidate_mapped_library_sha256'])
    info=r['architecture_binding'];result={}
    with tarfile.open(archivepath,'r') as stream:
        def read(name):
            member=stream.getmember(name)
            if not member.isfile():raise ValueError('Architecture artifact is not a regular file')
            return stream.extractfile(member).read()
        # The worker's relative path is fixed by its original output convention.
        matches=[m.name for m in stream.getmembers() if m.name.endswith('/lowering-receipt.json')]
        if len(matches)!=1:raise ValueError('Missing/ambiguous archived lowering receipt')
        raw=read(matches[0]);receipt=LoweringReceipt.from_json(raw.decode());receipt.verify()
        if receipt!=expected or hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError('Archived ABI receipt differs')
        for name in ('candidate','previous'):
            mapping=read('architecture-'+name+'/mapping.json');audit=parse_json(mapping.decode())
            # Replay immutable raw files and classifications, not stored counts alone.
            from core.opgen.trace import TraceReceipt
            from .architecture import audit_capture
            trace=TraceReceipt.from_json(read('architecture-'+name+'/receipt.json').decode());trace.verify()
            spec=r['prepared_trace']['variants'][name];data=archivepath.parent/'measure-data'
            for file in spec['trace_files']+spec['kernel_table_files']:
                path=Path(file)
                relative=path.resolve().relative_to(data.resolve())
                if read(str(relative))!=path.read_bytes():raise ValueError('Archived raw trace/table differs')
            import tempfile
            with tempfile.TemporaryDirectory(prefix='architecture-replay-',dir=archivepath.parent) as directory:
                replay=audit_capture(spec,name,c['device'],Path(directory)/name)
            if replay!=audit:raise ValueError('Archived architecture mapping differs from raw replay')
            result[name]=dict(ai_tasks=len(audit['ai_tasks']),control_tasks=len(audit['hardware_controls']),flow_events=len(audit['hardware_flows']),raw_records=audit['raw_records'],unsupported=audit['unsupported'],groups=audit['groups'],mapping_sha256=hashlib.sha256(mapping).hexdigest())
    return dict(model=plan.architecture.fingerprint(),scope=plan.scope,boundary_receipt=info['sha256'],mappings=result,
        warning='Sample addresses belong to target_validation only; pattern-derived invocation groups do not prove per-pipe ideal scheduling')


def archive(directory,target):
    # Never follow output symlinks. Preserve their spelling as audit metadata.
    skipped=[]
    with tarfile.open(target,'w') as stream:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():skipped.append(dict(path=str(path.relative_to(directory)),target=os.readlink(path)));continue
            if path.is_file():stream.add(path,arcname=str(path.relative_to(directory)),recursive=False)
    return skipped


def benchmark(c,work,kind):
    project=Path(c['project_root']);data=work/(kind+'-data');data.mkdir()
    report=data/'report.json'
    argv=[sys.executable,str(project/'scripts/probe_attention_sparse_components.py'),'--device',str(c['device']),
        '--candidate-manifest',c['candidate_manifest'],'--reference-manifest',c['reference_manifest'],'--fused-pv-build',c['pv_manifest'],
        '--rows',str(c['shape'][2]),'--keys',str(c['shape'][2]),'--dim','128','--seed',str(c['seed']),
        '--warmup',str(c['warmup']),'--repeats',str(c['repeats']),'--trials',str(c['trials']),
        '--timeout','600','--output',str(report)]
    if kind=='check':argv+=['--smoke-only']
    else:argv+=['--trace']
    # Input source/manifest paths are absolute in the original worker. Its cwd
    # can therefore be writable without touching the operator checkout.
    with (data/'launcher.stdout').open('w') as so,(data/'launcher.stderr').open('w') as se:
        child_env=dict(os.environ)
        if c.get('lowering'):child_env['CAKE_ATTENTION_LOWERING']=c['lowering']
        result=subprocess.run(argv,cwd=work,stdout=so,stderr=se,timeout=660,env=child_env)
    skipped=archive(data,work/(kind+'.raw.tar'))
    if result.returncode:raise ValueError('Attention driver returned '+str(result.returncode)+'; see '+str(data))
    r=parse_json(report.read_text());summary=validate_report(c,r,timed=kind=='measure')
    if kind=='measure':
        summary['source_report_sha256']=digest(report)
        if c.get('lowering'):
            summary['architecture']=audit_architecture(c,r,data)
    summary['archive_symlinks']=skipped
    (work/(kind+'.summary.json')).write_text(_dump(summary,pretty=True)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',choices=('build','check','measure','verify'),required=True);args=ap.parse_args()
    c=parse_json(Path(args.config).read_text());model=audit_config(c);work=Path(os.environ['CAKE_WORK']);metrics=[]
    for key in ('ASCEND_CACHE_PATH','ACL_OP_COMPILER_CACHE_DIR','ASCEND_PROCESS_LOG_PATH'):
        path=work/key.lower();path.mkdir(exist_ok=True);os.environ[key]=str(path)
    if args.stage=='build':
        # Reuse is explicit and every source/library hash is checked above.
        # This is not represented as a fresh compiler invocation.
        import torch
        actual=dict(torch=torch.__version__,torch_npu=importlib.metadata.version('torch_npu'),abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))
        if actual!=c['runtime']:raise ValueError('Actual torch/NPU ABI differs')
        (work/'native.manifest.json').write_text(_dump(dict(mode='identity-checked existing binaries',model=model.fingerprint(),runtime=actual,libraries=c['native_libraries']),pretty=True)+'\n')
    elif args.stage in ('check','measure'):
        summary=benchmark(c,work,args.stage)
        if args.stage=='measure':
            for name,row in summary['device'].items():
                metrics.extend((Metric(name+'_device_task_sum','ms',row['task_sum_ms']),Metric(name+'_device_union','ms',row['union_ms'])))
            for name,row in summary['paired_api'].items():metrics.append(Metric(name+'_prepared_api_median','ms',row['median_ms']))
            for name,row in summary['stages'].items():metrics.append(Metric(name+'_api_median','ms',row['median']))
    else:
        with tarfile.open(work/'measure.raw.tar','r') as stream:
            member=stream.getmember('report.json')
            if not member.isfile():raise ValueError('Archived report is not a regular file')
            data=stream.extractfile(member).read();r=parse_json(data.decode())
        summary=validate_report(c,r,timed=True);current=parse_json((work/'measure.summary.json').read_text())
        if c.get('lowering'):
            summary['architecture']=verify_archived_architecture(c,r,work/'measure.raw.tar')
            if summary['architecture']!=current.get('architecture'):raise ValueError('Architecture raw/summary differs')
        if hashlib.sha256(data).hexdigest()!=current['source_report_sha256'] or any(summary[k]!=current[k] for k in ('device','paired_api','stages','target_check','candidate_over_laser_device_sum','observations')):
            raise ValueError('Raw capture and protected summary differ')
    report=StageReport(os.environ['CAKE_PLAN_FP'],os.environ['CAKE_BUILD_FP'],args.stage,True,os.environ['CAKE_NONCE'],tuple(metrics))
    (work/(args.stage+'.json')).write_text(report.to_json(pretty=True)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
