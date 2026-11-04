"""Replay named-core timings from archived CSV using the frozen existing parser.

This derives an analysis artifact. It is not a new runner, delivery metric,
confidence interval, phase clock calibration or critical-path certificate.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,tarfile,tempfile
from core.opgen.runner import RunReceipt,identify

def named_group(groups,calls,name):
    if type(calls) is not int or calls<=0:raise ValueError('Positive actual submitted call count required')
    found=[x for x in groups if x.get('name')==name]
    if len(found)!=1 or found[0].get('tasks')!=calls:raise ValueError('Missing/ambiguous core or incomplete one-core-per-call coverage: '+name)
    x=found[0];value=x.get('sum_duration_ms_per_call')
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Invalid core duration')
    return dict(kernel=name,calls=calls,mean_device_task_ms=value,counters_as_exported=x.get('counters_as_exported',{}))

def analyze(receipt_path,*,candidate_kernel,previous_kernel,laser_kernel):
    receipt=RunReceipt.from_json(Path(receipt_path).read_text());receipt.verify()
    if receipt.state!='verified':raise ValueError('Require an actually verified complete run')
    stage=next(x for x in receipt.plan.stages if x.kind=='measure');cfg=Path(stage.argv[stage.argv.index('--config')+1]);config=json.loads(cfg.read_text());work=Path(receipt.work_dir);data=work/'measure-data';archive=work/'measure.raw.tar'
    parser=Path(config['project_root'])/'scripts/mindie_laser_quant_trace.py';pin=next(x for x in receipt.plan.inputs if x.path==str(parser.resolve()));pin.verify()
    spec=importlib.util.spec_from_file_location('_frozen_named_core_csv_parser',parser);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows={}
    with tarfile.open(archive,'r') as stream,tempfile.TemporaryDirectory(prefix='attention-core-replay-') as folder:
        def read(member):
            item=stream.getmember(member)
            if not item.isfile():raise ValueError('Archived evidence is not a regular file')
            return stream.extractfile(item).read()
        raw=read('report.json');report=json.loads(raw)
        for role,variant,name in [('candidate','candidate_producer',candidate_kernel),('previous','previous',previous_kernel),('laser','laser',laser_kernel)]:
            capture=report['prepared_trace']['variants'][variant]
            if capture.get('capture_finalized') is not True or len(capture.get('kernel_table_files',[]))!=1:raise ValueError('Incomplete/ambiguous device capture')
            original=Path(capture['kernel_table_files'][0]);relative=original.resolve().relative_to(data.resolve());table=read(str(relative));path=Path(folder)/(role+'.csv');path.write_bytes(table)
            parsed=module.summarize_kernel_csv(path,capture['submitted_calls'])
            if parsed['kernel_groups']!=capture['device_summary']['kernel_groups']:raise ValueError('Archived raw kernel table disagrees with original reported groups')
            rows[role]=dict(named_group(parsed['kernel_groups'],capture['submitted_calls'],name),variant=variant,archived_member=str(relative),sha256=hashlib.sha256(table).hexdigest())
    a,b,l=(rows[x]['mean_device_task_ms'] for x in ('candidate','previous','laser'))
    dense=report['target_check']['dense_error']['relative_rmse'];ldense=report['laser_dense_error']['relative_rmse']
    return dict(status='archived_raw_replayed',receipt=identify(receipt_path,'output').__dict__,source=identify(__file__,'harness').__dict__,parser=pin.__dict__,report_sha256=hashlib.sha256(raw).hexdigest(),core=rows,gain_vs_previous_fraction=1-a/b,gain_vs_laser_fraction=1-a/l,beats_both_core=a<b and a<l,quality=dict(candidate_RMSE_ratio=dense,candidate_RMSE_percent=dense*100,laser_RMSE_ratio=ldense,laser_RMSE_percent=ldense*100,scope='Actual original32query sample; not video or all rows quality'),device_scope='Named core mean device task durations across3 submitted profiler calls; not API wall or full prepared sums',busy_scope='Exported counters overlap; not additive phase cost or critical path',shape=report['shape'],std=report['std'],seed=report['seed'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--candidate-kernel',required=True);p.add_argument('--previous-kernel',default='vc_sparse_pair_fp16');p.add_argument('--laser-kernel',default='VcLaserAttention');p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Preserve existing analysis')
    r=analyze(a.receipt,candidate_kernel=a.candidate_kernel,previous_kernel=a.previous_kernel,laser_kernel=a.laser_kernel)
    with Path(a.output).open('x') as out:json.dump(r,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(dict(core_ms={k:v['mean_device_task_ms'] for k,v in r['core'].items()},beats_both_core=r['beats_both_core'],quality=r['quality'])));return 0
if __name__=='__main__':raise SystemExit(main())
