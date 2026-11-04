#!/usr/bin/env python3
"""Verify publication bytes and replay archived kernel CSVs; never use a device."""
from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'FILE_MANIFEST.json').read_text())
    for row in manifest['files']:
        path=ROOT/row['path']
        if path.is_symlink() or not path.is_file():raise ValueError('Missing/symlink: '+row['path'])
        data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Snapshot identity changed: '+row['path'])
    origins=json.loads((ROOT/'evidence/source_origins.json').read_text())
    best=[x for x in origins['native_sources'] if x['operation']=='sparse_delay']
    if len(best)!=14:raise ValueError('Expected14frozen best native sources')
    primary=json.loads((ROOT/'evidence/runs/primary_d/run_receipt.json').read_text())
    pinned={x['path']:x for x in primary['plan']['inputs']}
    for row in best:
        origin=origins['files'][row['source']]['original_path']
        if pinned[origin]['sha256']!=row['sha256']:raise ValueError('Best source differs from measured run input')
    for row in origins['native_sources']:
        if hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Original native source mismatch')
    path=ROOT/'source/vc-attention/scripts/mindie_laser_quant_trace.py';spec=importlib.util.spec_from_file_location('_published_frozen_csv_parser',path);module=importlib.util.module_from_spec(spec);exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    index=json.loads((ROOT/'evidence/index.json').read_text());results=[]
    for run in index['runs']:
        report=json.loads((ROOT/run['worker_report']).read_text());core=json.loads((ROOT/run['core_analysis']).read_text());groups={}
        for table in run['tables']:
            parsed=module.summarize_kernel_csv(ROOT/table['path'],table['submitted_calls'])
            expected=report['prepared_trace']['variants'][table['variant']]['device_summary']['kernel_groups']
            if parsed['kernel_groups']!=expected:raise ValueError('Original report/CSV mismatch: '+table['path'])
            groups[table['variant']]=parsed['kernel_groups']
        for role in ('candidate','previous','laser'):
            entry=core['core'][role];rows=[r for r in groups[entry['variant']] if r['name']==entry['kernel']]
            if len(rows)!=1 or rows[0]['tasks']!=entry['calls'] or abs(rows[0]['sum_duration_ms_per_call']-entry['mean_device_task_ms'])>1e-12:raise ValueError('Named core evidence mismatch')
        results.append(dict(run=run['id'],core_ms=run['recorded_core_ms']))
    print(json.dumps(dict(status='snapshot_and_archived_CSV_verified',files=len(manifest['files']),best_native_sources=14,native_components=len(origins['native_sources']),runs=results,NPU_executions=0,fullquery_tensor_replay=False),ensure_ascii=False))
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError) as error:print(str(error),file=sys.stderr);raise SystemExit(1)
