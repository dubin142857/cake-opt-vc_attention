"""Coverage and support-loss diagnostics; no new runner or attention oracle.

Actual archived outputs remain the candidates. Counterfactual weights/support
are explicitly CPU diagnostics and can never replace them in quality metrics.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import resource
import sys
import tarfile
import time
import torch
from core.opgen.runner import FileIdentity, RunReceipt, identify


def write_json(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def verify_identity(value):
    identity=FileIdentity(**value);identity.verify();return Path(identity.path)


def load_cpu(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def load_target(archive):
    """Read a unique regular target directly from the frozen archive."""
    with tarfile.open(archive) as stream:
        members=[x for x in stream.getmembers() if x.name.endswith('/baseline/local/target.pt')]
        if len(members)!=1 or not members[0].isfile():raise ValueError('Unique regular archived target required')
        # No extraction to a mutable folder, no filesystem traversal.
        target=load_cpu(io.BytesIO(stream.extractfile(members[0]).read()))
    return target,members[0].name


def checked_target(target, *, expected_shape=None):
    inputs=target.get('inputs')
    if not isinstance(inputs,(list,tuple)) or len(inputs)!=3:raise ValueError('Three original CPU inputs required')
    output=target.get('output')
    for tensor in (*inputs,output):
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or tensor.ndim!=4 or not tensor.is_floating_point() or not torch.isfinite(tensor).all():raise ValueError('Finite floating CPU BHSD tensors required')
    shape=tuple(inputs[0].shape)
    if any(tuple(x.shape)!=shape for x in (*inputs,output)) or min(shape)<1:raise ValueError('Matching MHA shape required')
    if expected_shape is not None and shape!=tuple(expected_shape):raise ValueError('Frozen full-query shape differs')
    if any(x.dtype!=torch.bfloat16 for x in (*inputs,output)):raise ValueError('Original public BF16 tensors required')
    return tuple(inputs),output


def same_inputs(actual, expected):
    if len(actual)!=len(expected) or any(a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b) for a,b in zip(actual,expected)):
        raise ValueError('Archived branches do not have identical original inputs')


def checked_selected(target, queries, keys):
    ids=target.get('selected_indices')
    if not isinstance(ids,torch.Tensor) or ids.device.type!='cpu' or ids.ndim!=2 or ids.shape[0]<queries or ids.shape[1]!=8 or ids.dtype not in (torch.int32,torch.int64):raise ValueError('Archived [Qwork,8] support required')
    ids=ids[:queries].long()
    if (ids<0).any() or (ids>=keys).any() or (ids.sort(-1).values.diff(dim=-1)==0).any():raise ValueError('Invalid or duplicate selected key IDs')
    return ids


def error_summary(actual, reference, error, row_error):
    if actual.shape!=reference.shape or actual.ndim!=4 or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():raise ValueError('Matching finite BHQD outputs required')
    queries=reference.shape[2]
    global_error=error(actual,reference)
    rows,relative=row_error(actual.reshape(-1,actual.shape[-1]),reference.reshape(-1,reference.shape[-1]),queries,worst=64)
    thresholds={str(threshold):dict(count=int((relative>threshold).sum()),fraction=float((relative>threshold).double().mean())) for threshold in (.01,.05,.1)}
    ratio=global_error['relative_rmse']
    return dict(global_error=global_error,relative_RMSE_ratio=ratio,relative_RMSE_percent=None if ratio is None else ratio*100,rows=rows,thresholds=thresholds,logical_rows=relative.numel()),relative


def import_reference(identity, module_name):
    path=verify_identity(identity)
    # The benchmark imports its pinned sibling source lock at module import.
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def peak_mib():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024


def check_budget(start, *, deadline_s=360, memory_mib=4096):
    if time.monotonic()-start>deadline_s:raise TimeoutError('Frozen CPU wall-time budget reached')
    if peak_mib()>memory_mib:raise MemoryError('Observed CPU RSS budget reached')


def diagnose(q,k,v,ids,selected,reference,probe,error):
    report,raw=probe(q,k,v,ids,logit_mode='fp32',topks=(1,2,4,8,16),masses=(.999,.9999))
    oracle_error=error(raw['dense_output'],reference.index_select(2,ids))
    if oracle_error['relative_rmse'] is None or oracle_error['relative_rmse']>1e-4:raise ValueError('Independent support probe disagrees with full reference')
    logits=raw['logits'];chosen=selected[ids]
    actual_scores=logits[0,0].gather(-1,chosen)
    weights=actual_scores.double().softmax(-1)
    ideal_actual=(v[0,0,chosen].double()*weights[:,:,None]).sum(1)[None,None]
    global_probability=logits.double().softmax(-1)
    mass=global_probability[0,0].gather(-1,chosen).sum(-1)
    true=raw['top_indices'][0,0].long();present=(true[:,:2,None]==chosen[:,None,:]).any(-1)
    same_block=(true[:,0]//128)==(true[:,1]//128)
    same_block_missing=(same_block & ~present.all(-1))
    detail=dict(query_ids=ids.tolist(),rows=len(ids),probe_vs_full_reference=oracle_error,
        ideal_global_Top8=report['topk']['8'],ideal_actual_saved_support=error(ideal_actual,raw['dense_output']),
        missing_true_top1_rows=int((~present[:,0]).sum()),missing_any_true_top2_rows=int((~present.all(-1)).sum()),
        missing_any_true_top2_fraction=float((~present.all(-1)).double().mean()),
        true_top2_same_K128_block_rows=int(same_block.sum()),same_block_and_missing_rows=int(same_block_missing.sum()),
        actual_support_retained_mass_min=float(mass.min()),actual_support_retained_mass_mean=float(mass.mean()),
        probe=report,scope='Worst actual E122 rows; all ideal outputs are CPU counterfactuals, not a candidate')
    saved=dict(query_ids=ids,selected_indices=chosen,true_indices=true,logits=logits,dense_output=raw['dense_output'],ideal_actual_support=ideal_actual,ideal_global_top8=raw['topk_outputs']['8'],retained_mass=mass,top2_present=present,same_block=same_block)
    return detail,saved


def run(frozen, output):
    freeze_path=Path(frozen).resolve();freeze=json.loads(freeze_path.read_text());out=Path(output).resolve()
    if out.exists():raise FileExistsError('Unique audit output directory required')
    out.mkdir(parents=True);torch.set_num_threads(4)
    report=dict(status='preflight',freeze=asdict(identify(freeze_path,'input')),cases=[],native_execution=False,
        scope='All logical queries, two already-exposed std5 seeds; no general-input or video quality claim')
    write_json(out/'report.json',report)
    try:
        # Enforce identities before importing helpers or deserializing tensors.
        for case in freeze['cases']:
            verify_identity(case['cached_FP64'])
            for branch in case['branches'].values():
                path=verify_identity(branch['receipt']);verify_identity(branch['archive'])
                receipt=RunReceipt.from_json(path.read_text());receipt.verify()
                if receipt.state!='verified' or receipt.plan.model_fingerprint!=branch['model'] or Path(receipt.work_dir)/'measure.raw.tar'!=Path(branch['archive']['path']):raise ValueError('Frozen verified receipt/archive/model linkage differs')
        for source in freeze['sources']:verify_identity(source)
        dense_module=import_reference(freeze['sources'][0],'e123_pinned_dense')
        sparse_module=import_reference(freeze['sources'][1],'e123_pinned_sparse')
        sources=[Path(__file__),Path(__file__).with_name('__init__.py'),freeze_path]+[Path(x['path']) for x in freeze['sources']]
        report['sources']=[asdict(identify(x,'source')) for x in sources]
        with tarfile.open(out/'named-measured-sources.tar','x') as archive:
            for path in sources:archive.add(path,arcname=str(path.resolve()).lstrip('/'),recursive=False)
        start=time.monotonic();report.update(status='running',threads=torch.get_num_threads(),identity_preflight='passed',inner_chunk=64,outer_chunk=1024)
        write_json(out/'report.json',report)
        raw_bytes=0
        for case in freeze['cases']:
            check_budget(start);outputs={};targets={};original=None;members={}
            for name,branch in case['branches'].items():
                target,member=load_target(branch['archive']['path']);inputs,actual=checked_target(target,expected_shape=(1,1,38139,128))
                if original is None:original=inputs
                else:same_inputs(inputs,original)
                outputs[name]=actual;members[name]=member
                if name=='E122_delay':targets[name]=target
            q,k,v=original;queries=q.shape[2];all_ids=torch.arange(queries,dtype=torch.int64)
            selected=checked_selected(targets['E122_delay'],queries,k.shape[2]);targets.clear()
            parts=[]
            for ids in all_ids.split(1024):
                parts.append(dense_module.dense_reference(q,k,v,ids,chunk=64));check_budget(start)
                if int(ids[-1])%4096>=3072 or int(ids[-1])==queries-1:
                    print(json.dumps(dict(seed=case['seed'],reference_queries=int(ids[-1])+1,seconds=time.monotonic()-start,peak_RSS_MiB=peak_mib())),flush=True)
            reference=torch.cat(parts,dim=2);del parts
            cached=load_cpu(case['cached_FP64']['path']);cached_ids=cached['query_ids']
            if cached_ids.dtype!=torch.int64 or cached_ids.ndim!=1 or len(cached_ids)!=349 or (cached_ids<0).any() or (cached_ids>=queries).any() or len(cached_ids.unique())!=349:raise ValueError('Invalid independent cached query scope')
            cached_dense=cached['dense_output'];actual_subset=reference[0,0,cached_ids]
            if cached_dense.shape!=actual_subset.shape or not torch.isfinite(cached_dense).all():raise ValueError('Independent oracle tensor differs')
            oracle_error=sparse_module._error(actual_subset,cached_dense)
            if oracle_error['relative_rmse'] is None or oracle_error['relative_rmse']>1e-4:raise ValueError('Full FP32 oracle differs from independent cached FP64')
            entry=dict(seed=case['seed'],std=5,shape=list(q.shape),archived_members=members,oracle349=oracle_error,branches={})
            row_relative=None
            for name,actual in outputs.items():
                summary,relative=error_summary(actual,reference,sparse_module._error,sparse_module._row_error)
                entry['branches'][name]=summary
                if name=='E122_delay':row_relative=relative
            worst=torch.argsort(torch.where(torch.isfinite(row_relative),row_relative,float('inf')),descending=True,stable=True)[:64]
            diagnostic,saved=diagnose(q,k,v,worst,selected,reference,sparse_module.support_probe,sparse_module._error)
            entry['diagnosis']=diagnostic
            raw_path=out/f"seed{case['seed']}-reference-and-worst.pt"
            with raw_path.open('xb') as stream:torch.save(dict(full_reference=reference,worst=saved),stream)
            entry['artifact']=asdict(identify(raw_path,'output'));raw_bytes+=entry['artifact']['size_bytes']
            if raw_bytes>128*1024**2:raise ValueError('Frozen new raw-tensor byte budget exceeded')
            report['cases'].append(entry);report.update(cpu_wall_seconds=time.monotonic()-start,raw_bytes=raw_bytes,peak_RSS_MiB=peak_mib());write_json(out/'report.json',report)
            print(json.dumps(dict(seed=case['seed'],percent={n:r['relative_RMSE_percent'] for n,r in entry['branches'].items()},diagnosis={k:diagnostic[k] for k in ('missing_any_true_top2_rows','true_top2_same_K128_block_rows')})),flush=True)
            del original,inputs,target,q,k,v,reference,outputs,cached,saved,selected,row_relative
            check_budget(start)
        for source in report['sources']:FileIdentity(**source).verify()
        report.update(status='completed',cpu_wall_seconds=time.monotonic()-start,peak_RSS_MiB=peak_mib());write_json(out/'report.json',report)
        return report
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',peak_RSS_MiB=peak_mib());write_json(out/'report.json',report);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--frozen',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    run(args.frozen,args.output);return 0
if __name__=='__main__':raise SystemExit(main())
