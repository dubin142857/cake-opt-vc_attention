#!/usr/bin/env python3
"""Native builtin component instrument; precomputed scores/indices, not attention.

Uses the existing isolated Laser provider, paired timer and trace machinery.
No component duration is added to another or presented as an attention floor.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import benchmark_mindie_laser_quant as quant

ROOT=Path(__file__).resolve().parents[1]
base=quant.baseline
common=base.common


def make_indices(torch,rows,keys,rank,seed,pattern):
    if min(rows,keys,rank)<1 or rank>keys:raise ValueError('Invalid sparse dimensions')
    generator=torch.Generator().manual_seed(seed)
    if pattern=='contiguous':
        starts=torch.randint(keys-rank+1,(rows,1),generator=generator)
        return starts+torch.arange(rank)
    if pattern!='random':raise ValueError('Unknown index pattern')
    result=torch.randint(keys,(rows,rank),generator=generator)
    # Rejection gives independent uniform samples without replacement per row.
    for col in range(1,rank):
        duplicate=(result[:,:col]==result[:,col:col+1]).any(-1)
        while duplicate.any():
            result[duplicate,col]=torch.randint(keys,(int(duplicate.sum()),),generator=generator)
            duplicate=(result[:,:col]==result[:,col:col+1]).any(-1)
    return result


def check_selection(cpu,values,indices,torch):
    values,indices=values.cpu(),indices.cpu()
    if tuple(indices.shape)!=tuple(values.shape) or indices.shape[0]!=cpu.shape[0]:
        raise AssertionError('TopK output layout')
    if (indices<0).any() or (indices>=cpu.shape[1]).any():raise AssertionError('TopK invalid indices')
    ordered=indices.sort(-1).values
    if (ordered[:,1:]==ordered[:,:-1]).any():raise AssertionError('TopK repeated index')
    wrong=int((values.contiguous().view(torch.int16)!=cpu.gather(1,indices).contiguous().view(torch.int16)).sum())
    rows=torch.linspace(0,cpu.shape[0]-1,min(32,cpu.shape[0])).long().unique()
    expected=cpu.index_select(0,rows).topk(values.shape[1],dim=-1,sorted=True).values
    sample_wrong=int((values.index_select(0,rows).contiguous().view(torch.int16)!=expected.contiguous().view(torch.int16)).sum())
    if wrong or sample_wrong:raise AssertionError('TopK selected values disagree')
    return dict(all_rows=cpu.shape[0],sampled_rows=rows.tolist(),gather_value_bit_differences=wrong,
                sampled_topk_value_bit_differences=sample_wrong)


def sparse_calls(value,indices,weights,torch):
    rows,rank=indices.shape;dim=value.shape[-1]
    gather=lambda:value.index_select(0,indices.reshape(-1)).reshape(rows,rank,dim)
    packed=gather()
    weighted=lambda:(gather().float()*weights.float().unsqueeze(-1)).sum(1)
    bmm=lambda:torch.bmm(weights.unsqueeze(1),gather()).squeeze(1)
    pv=lambda:(packed.float()*weights.float().unsqueeze(-1)).sum(1)
    return dict(gather=gather,weighted_sum=weighted,weighted_bmm=bmm,pv_only_sum=pv)


def check_sparse(cpu_value,cpu_indices,cpu_weights,calls,torch):
    import mindie_laser_sparse_reference as reference
    packed=cpu_value.index_select(0,cpu_indices.reshape(-1)).reshape(*cpu_indices.shape,cpu_value.shape[-1])
    expected=(packed.double()*cpu_weights.double().unsqueeze(-1)).sum(1)
    gather=calls['gather']().cpu()
    wrong=int((gather.view(torch.int16)!=packed.view(torch.int16)).sum())
    if wrong:raise AssertionError('Gather changed storage bits')
    row={'gather_bit_differences':wrong,'weighted':{}}
    for name in ('weighted_sum','weighted_bmm','pv_only_sum')+ (('native_fused',) if 'native_fused' in calls else ()):
        got=calls[name]().cpu()
        if not torch.isfinite(got).all():raise AssertionError('Weighted PV nonfinite')
        error=reference._error(got,expected)
        limit=.001 if name=='weighted_bmm' else 1e-5
        absolute=.02 if name=='weighted_bmm' else 1e-4
        if error['relative_rmse'] is None or error['relative_rmse']>limit or error['max_abs']>absolute:
            raise AssertionError(name+' differs from FP64 reference: '+str(error))
        row['weighted'][name]=error
    return row



def native_result_check(adapter,value,indices,weights,torch,*,invalid_rows=0):
    from mindie_laser_sparse_pv import validate_counters
    import mindie_laser_sparse_reference as reference
    payload=adapter.prepare(value,indices,weights);result=adapter.run_native(payload)
    counts=validate_counters(result,payload.rows,payload.rank,invalid_rows=invalid_rows)
    cpu_value=value.cpu();cpu_indices=indices.cpu();cpu_weights=weights.cpu()
    bad=((cpu_indices<0)|(cpu_indices>=cpu_value.shape[0])).any(-1)
    safe=torch.where((cpu_indices>=0)&(cpu_indices<cpu_value.shape[0]),cpu_indices,0).long()
    expected=(cpu_value.index_select(0,safe.reshape(-1)).reshape(*safe.shape,cpu_value.shape[-1]).double()*cpu_weights.double().unsqueeze(-1)).sum(1)
    expected[bad]=0
    actual=result[0].cpu();error=reference._error(actual,expected)
    if not torch.isfinite(actual).all() or error['relative_rmse'] is None or error['relative_rmse']>1e-5 or error['max_abs']>1e-4:
        raise AssertionError('Native sparse PV numerical disagreement: '+str(error))
    unpacked=payload.indices.cpu().transpose(1,2).reshape(-1,16)[:payload.rows,:payload.rank].long()
    want=torch.where((cpu_indices>=0)&(cpu_indices<cpu_value.shape[0]),cpu_indices,-1).long()
    unpacked_weights=payload.weights.cpu().transpose(1,2).reshape(-1,16)[:payload.rows,:payload.rank]
    if not torch.equal(unpacked,want) or not torch.equal(unpacked_weights.view(torch.int16),cpu_weights.view(torch.int16)):
        raise AssertionError('Native packet roundtrip mismatch')
    return payload,dict(counters=counts,error=error,packet_bit_differences=0)


def worker(args):
    if args.candidate_build or args.candidate_manifest:
        from probe_mindie_laser_sparse_attention import worker as candidate_worker
        return candidate_worker(args)
    directory=Path(args._run_dir)/'baseline/local';directory.mkdir(parents=True,exist_ok=True)
    report={'status':'running','mode':'sparse_components','stage':'provenance','scope':'native components, precomputed scores/indices; not an attention candidate',
            'rows':args.rows,'keys':args.keys,'dim':args.dim,'rank':args.rank,'seed':args.seed,
            'source_sha256':common.sha256_file(__file__),'paired_groups':{}}
    save=lambda:common.write_json(directory/'report.json',report)
    save()
    try:
        resolved=quant.build_arguments(args,'baseline');report['provenance']=base.provenance(resolved)
        import torch
        import torch_npu
        from mindie_laser_quant_pairing import run_paired
        from mindie_laser_quant_trace import capture_native_traces
        torch.set_num_threads(args.cpu_threads);torch.npu.set_device(args.device)
        torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE':'disable','ACL_OP_COMPILER_CACHE_DIR':os.environ['ACL_OP_COMPILER_CACHE_DIR']})
        if hasattr(torch.npu,'matmul'):torch.npu.matmul.allow_hf32=False
        source=json.loads(Path(resolved.source_manifest).read_text())
        laser,transforms=base.load_adapter(resolved,source)
        report['transforms']=transforms;device=f'npu:{args.device}'
        generator=torch.Generator().manual_seed(args.seed)
        fused=None
        if args.fused_pv_build:
            from mindie_laser_sparse_pv import load_sparse_pv
            fused=load_sparse_pv(args.fused_pv_build,args.device)
            report['fused_pv_provider']=dict(metadata=fused.metadata,build=fused.evidence)
        with torch.inference_mode():
            report.update(stage='small_checks',small_selection=[],small_sparse=[]);save()
            for rows,cols in ((1,1),(17,7),(129,17),(17,298),(17,2048),(17,38139)):
                for pattern in ('random','ties','peaks'):
                    cpu=(torch.randn(rows,cols,generator=generator)*25).half() if pattern=='random' else torch.zeros(rows,cols,dtype=torch.float16)
                    if pattern=='peaks':
                        positions=torch.linspace(0,cols-1,min(8,cols)).long().unique()
                        cpu[:,positions]=torch.arange(1,len(positions)+1).half()*2
                    npu=cpu.to(device);values,indices=npu.topk(min(8,cols),dim=-1,sorted=True)
                    row=check_selection(cpu,values,indices,torch)
                    row.update(shape=[rows,cols],pattern=pattern);report['small_selection'].append(row);save()
            for rows,keys,dim,rank in ((1,17,64,1),(17,385,128,8),(129,17,128,8),(17,385,64,8)):
                for pattern in ('random','contiguous'):
                    value=(torch.randn(keys,dim,generator=generator)*5).half();value[:,0]=2
                    indices=make_indices(torch,rows,keys,rank,args.seed,pattern)
                    weight=torch.rand(rows,rank,generator=generator);weight=(weight/weight.sum(-1,keepdim=True)).half()
                    calls=sparse_calls(value.to(device),indices.to(device),weight.to(device),torch)
                    native_check=None
                    if fused is not None:
                        packet,native_check=native_result_check(fused,value.to(device),indices.to(device),weight.to(device),torch)
                        calls['native_fused']=lambda packet=packet:fused.run_prepared(packet)
                    row=check_sparse(value,indices,weight,calls,torch);row.update(rows=rows,keys=keys,dim=dim,rank=rank,pattern=pattern)
                    if native_check is not None:row['native']=native_check
                    report['small_sparse'].append(row);save()
            if fused is not None:
                report['native_extra_checks']=[]
                for rank in (1,8,16):
                    for std in (0,1,5):
                        dim=64 if rank==16 else 128
                        value=(torch.randn(257,dim,generator=generator)*std).half()
                        indices=torch.randint(257,(33,rank),generator=generator);indices[0]=7
                        weights=torch.randn(33,rank,generator=generator).half()
                        _,check=native_result_check(fused,value.to(device),indices.to(device),weights.to(device),torch)
                        check.update(rank=rank,std=std,dim=dim,case='duplicate/negative weights')
                        report['native_extra_checks'].append(check);save()
                value=torch.randn(6*385,64,generator=generator).half()
                indices=torch.randint(385,(6*17,8),generator=generator)+torch.arange(6).repeat_interleave(17)[:,None]*385
                weights=torch.rand(6*17,8,generator=generator).half()
                _,check=native_result_check(fused,value.to(device),indices.to(device),weights.to(device),torch)
                check['case']='B2H3 flattened independent heads';report['native_extra_checks'].append(check)
                indices[0,0]=-1;indices[-1,-1]=2**32+5
                _,check=native_result_check(fused,value.to(device),indices.to(device),weights.to(device),torch,invalid_rows=2)
                check['case']='two invalid rows safely zeroed';report['native_extra_checks'].append(check);save()
            if args.smoke_only:
                report.update(status='passed',stage='smoke_completed');save();return 0
            report['stage']='laser_reference';save()
            original=quant.synthetic_inputs(torch,[1,1,args.keys,args.dim],5,args.seed)
            report['laser_shape']=list(original[0].shape)
            inputs=[x.to(device) for x in original];payload=laser.prepare(*inputs)
            marker=Path(os.environ[base.MARKER_ENV]);offset=marker.stat().st_size if marker.exists() else 0
            output=laser.run_prepared(payload).cpu();torch.npu.synchronize()
            with marker.open('rb') as stream:stream.seek(offset);fresh=stream.read().decode()
            report['host_markers']=base.parse_markers(fresh,os.getpid())
            build=report['provenance']['build_manifest'];cann=build.get('cann_root') or build['definitions']['CUSTOM_ASCEND_CANN_PACKAGE_PATH']
            base.inspect_provider('local',resolved.vendor_root,resolved.bridge_path,report,directory,cann_root=cann,soc=build['soc'])
            if fused is not None:
                library=fused.evidence['library_path']
                if library not in report['mapped_library_paths']:raise AssertionError('Sparse PV library not mapped')
                report['loaded_library_sha256'][library]=common.sha256_file(library)
                if report['loaded_library_sha256'][library]!=fused.evidence['library_sha256']:raise AssertionError('Sparse PV loaded bytes changed')
            os.environ.pop(base.MARKER_ENV,None);report['marker_disabled_before_timing']=True
            sample_ids=torch.linspace(0,args.keys-1,min(32,args.keys)).long().unique()
            expected=common.dense_reference(*original,sample_ids)
            base.record_gate(report,'laser_dense_fp32',output.index_select(2,sample_ids),expected,.01,save)
            report.update(stage='large_selection_checks',selection={},input_preparation={});save()
            selection_calls={};scores_cpu={};scores_npu={}
            for cols in ([] if fused is not None else args.score_cols):
                started=time.perf_counter();cpu=(torch.randn(args.rows,cols,generator=generator)*25).half()
                fingerprint=common.tensor_sha256(cpu);npu=cpu.to(device);torch.npu.synchronize()
                name='select_'+str(cols);scores_cpu[name]=cpu;scores_npu[name]=npu
                report['input_preparation'][name]={'shape':list(cpu.shape),'dtype':'float16','sha256':fingerprint,
                    'cpu_generate_hash_h2d_wall_s':time.perf_counter()-started,'bytes':cpu.numel()*cpu.element_size()}
                selection_calls[name]=lambda x=npu:x.topk(args.rank,dim=-1,sorted=True)
                values,indices=selection_calls[name]();report['selection'][name]=check_selection(cpu,values,indices,torch)
                score_ids=torch.linspace(0,args.rows-1,min(32,args.rows)).long().unique()
                torch.save({'query_ids':score_ids,'score_rows':cpu.index_select(0,score_ids),
                            'selected_values':values.cpu(),'selected_indices':indices.cpu()},directory/(name+'_samples.pt'))
                save()
            cpu_value=original[2][0,0].half();value=cpu_value.to(device)
            report.update(stage='large_sparse_checks',sparse={});save()
            families={};cpu_states={};fused_payloads={};device_states={}
            for pattern in ('random','contiguous'):
                indices=make_indices(torch,args.rows,args.keys,args.rank,args.seed,pattern)
                weights=torch.rand(args.rows,args.rank,generator=generator);weights=(weights/weights.sum(-1,keepdim=True)).half()
                if args.sparse_reference:
                    reference_path=Path(args.sparse_reference).resolve()
                    reference_report=json.loads(reference_path.read_text())
                    if reference_report.get('status')!='passed':raise ValueError('Passed sparse input reference required')
                    artifact=Path(reference_report['artifact_dir'])/'baseline/local'/('sparse_'+pattern+'_inputs.pt')
                    recorded=torch.load(artifact,map_location='cpu',weights_only=True)
                    if not torch.equal(recorded['value'],cpu_value) or recorded['indices'].shape!=(args.rows,args.rank) or recorded['weights'].shape!=(args.rows,args.rank) or recorded['weights'].dtype!=torch.float16:
                        raise AssertionError('Sparse input reference shape/value mismatch')
                    indices,weights=recorded['indices'],recorded['weights']
                    report.setdefault('sparse_input_reference',{})[pattern]={
                        'report':str(reference_path),'report_sha256':common.sha256_file(reference_path),
                        'artifact':str(artifact),'artifact_sha256':common.sha256_file(artifact),
                        'tensor_sha256':{name:common.tensor_sha256(recorded[name]) for name in ('value','indices','weights')}}
                device_states[pattern]=(value,indices.to(device),weights.to(device))
                families[pattern]=sparse_calls(*device_states[pattern],torch)
                if fused is not None:
                    packet,native_check=native_result_check(fused,*device_states[pattern],torch)
                    fused_payloads[pattern]=packet
                    families[pattern]['native_fused']=lambda packet=packet:fused.run_prepared(packet)
                report['sparse'][pattern]=check_sparse(cpu_value,indices,weights,families[pattern],torch)
                if fused is not None:report['sparse'][pattern]['native']=native_check
                cpu_states[pattern]={'value':cpu_value,'indices':indices,'weights':weights}
                torch.save(cpu_states[pattern],directory/('sparse_'+pattern+'_inputs.pt'));save()
            report['stage']='timing';save()
            def paired(label,calls,trials):
                def update(value):report['paired_groups'][label]=value;save()
                report['paired_groups'][label]=run_paired(calls,synchronize=torch.npu.synchronize,warmup=args.warmup,
                    repeats=args.repeats,trials=trials,seed=args.seed,on_trial=update);save()
            if fused is None:
                paired('selection',selection_calls,9)
                grouped={kind+'_'+pattern:families[pattern][kind] for pattern in families for kind in ('gather','weighted_sum')}
                paired('gather',grouped,12)
                paired('pv',{name:families['random'][name] for name in ('weighted_sum','weighted_bmm','pv_only_sum')},9)
            else:
                for pattern in families:
                    paired('fused_pv_'+pattern,{name:families[pattern][name] for name in ('weighted_sum','weighted_bmm','native_fused')},9)
                quant.benchmark_calls({'sparse_prepare':lambda:fused.prepare(*device_states['random']),
                    'sparse_full':lambda:fused(*device_states['random'])},args,torch,report,save)
            quant.benchmark_calls({'laser':lambda:laser.run_prepared(payload),'laser_full':lambda:laser(*inputs),
                                  'laser_preprocessing':lambda:laser.prepare(*inputs)},args,torch,report,save)
            if args.trace:
                calls={'laser':lambda:laser.run_prepared(payload),**selection_calls,
                       **{kind+'_'+pattern:families[pattern][kind] for pattern in families for kind in ('gather','weighted_sum')},
                       'weighted_bmm_random':families['random']['weighted_bmm'],'pv_only_sum_random':families['random']['pv_only_sum']}
                if fused is not None:
                    calls={'laser':lambda:laser.run_prepared(payload),
                           **{name+'_'+pattern:families[pattern][name] for pattern in families for name in ('weighted_sum','weighted_bmm','native_fused')},
                           'sparse_full':lambda:fused(*device_states['random'])}
                report['stage']='trace';save()
                report['prepared_trace']=capture_native_traces(calls,torch=torch,torch_npu=torch_npu,directory=directory/'trace',repeats=3)
            report.update(status='passed',stage='completed');save();return 0
    except Exception as e:
        report.update(status='failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());save();traceback.print_exc();return 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-build',default=str(ROOT/'results/mindie_laser_cann920_build.json'))
    parser.add_argument('--bridge-manifest',default=str(ROOT/'results/mindie_laser_cann920_bridge.json'))
    parser.add_argument('--device',type=int,default=5)
    parser.add_argument('--rows',type=int,default=38139)
    parser.add_argument('--keys',type=int,default=38139)
    parser.add_argument('--dim',type=int,default=128)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--score-cols',type=int,nargs='+',default=[298,2048,38139])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--trials',type=int,default=9,help='Independent Laser/prep/full trials; stage groups use balanced 9/12/9')
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--candidate-cold-only',action='store_true',help='Supplement shape-cache cold/hot costs without repeating the primary comparison')
    parser.add_argument('--candidate-metrics',nargs='+',choices=('PipeUtilization','Memory','MemoryAccess','L2Cache','ResourceConflictRatio'),help='Independent hardware metric captures; explicit candidate/reference manifests and --trace required')
    parser.add_argument('--candidate-manifest',help='Explicit candidate manifest; mutually exclusive with legacy candidate chain')
    parser.add_argument('--reference-manifest',help='Explicit previous candidate manifest for paired comparison')
    parser.add_argument('--candidate-dma-build',help='Output DMA overlap variant; --candidate-lookup-build supplies the E88 reference')
    parser.add_argument('--candidate-lookup-build',help='Compare exact-mask lookup candidate to --candidate-build and Laser')
    parser.add_argument('--candidate-build',help='Full block2/top8 candidate manifest; reuses this isolated worker')
    parser.add_argument('--fused-pv-build',help='Independent fused PV manifest; selects PV comparison instead of full score selection')
    parser.add_argument('--sparse-reference',help='Passed component report whose exact saved sparse inputs will be reused')
    parser.add_argument('--output',type=Path,default=ROOT/'results/laser_cann920_e85_sparse_components.json')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--_run-dir',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.device<0 or quant.physical_device(args.device)==0:parser.error('Physical NPU0 is reserved')
    if min(args.rows,args.keys,args.rank,args.cpu_threads,args.repeats,args.trials,args.timeout)<1 or args.rank>args.keys or args.dim not in (64,128) or args.warmup<0 or not 0<=args.seed<2**63:
        parser.error('Invalid component dimensions or timing settings')
    if not args.score_cols or min(args.score_cols)<args.rank or len(args.score_cols)!=len(set(args.score_cols)):
        parser.error('Distinct score extents at least rank required')
    if args.fused_pv_build and args.rank>16:parser.error('Fused sparse PV supports rank1..16')
    if args.candidate_manifest and any((args.candidate_build,args.candidate_lookup_build,args.candidate_dma_build)):parser.error('Explicit candidate manifest cannot mix with legacy chain')
    if args.reference_manifest and not args.candidate_manifest:parser.error('--reference-manifest requires --candidate-manifest')
    if args.candidate_manifest and not args.fused_pv_build:parser.error('--candidate-manifest requires --fused-pv-build')
    if args.candidate_dma_build and not args.candidate_lookup_build:parser.error('--candidate-dma-build requires --candidate-lookup-build')
    if args.candidate_lookup_build and not args.candidate_build:parser.error('--candidate-lookup-build requires --candidate-build')
    if args.candidate_cold_only and not (args.candidate_build or args.candidate_manifest):parser.error('--candidate-cold-only requires --candidate-build')
    if args.candidate_build and not args.fused_pv_build:parser.error('--candidate-build requires --fused-pv-build')
    if args.candidate_metrics and (not args.candidate_manifest or not args.reference_manifest or not args.trace or args.candidate_cold_only):parser.error('--candidate-metrics needs explicit candidate/reference manifests and --trace, without --candidate-cold-only')
    if args.candidate_metrics and len(args.candidate_metrics)!=len(set(args.candidate_metrics)):parser.error('Distinct candidate metrics required')
    if args._worker:return worker(args)
    args.output=args.output.resolve();args.output.parent.mkdir(parents=True,exist_ok=True)
    args._run_dir=tempfile.mkdtemp(prefix=args.output.stem+'_',dir=args.output.parent)
    resolved=quant.build_arguments(args,'baseline');base.provenance(resolved)
    if args.fused_pv_build:
        from mindie_laser_quant_fused import validate_preprocess_manifest
        validate_preprocess_manifest(args.fused_pv_build,operation='sparse_pv')
    if args.candidate_manifest:
        from mindie_laser_sparse_candidates import candidate_manifest_operation
        candidate_manifest_operation(args.candidate_manifest)
        if args.reference_manifest:candidate_manifest_operation(args.reference_manifest)
    if args.candidate_build:
        from mindie_laser_quant_fused import validate_preprocess_manifest
        validate_preprocess_manifest(args.candidate_build,operation='sparse_candidates')
    if args.candidate_lookup_build:
        from mindie_laser_quant_fused import validate_preprocess_manifest
        validate_preprocess_manifest(args.candidate_lookup_build,operation='sparse_candidates_lookup')
    if args.candidate_dma_build:
        from mindie_laser_quant_fused import validate_preprocess_manifest
        validate_preprocess_manifest(args.candidate_dma_build,operation='sparse_candidates_dma_overlap')
    env=base.worker_environment(os.environ,'local',resolved.vendor_root,Path(resolved._run_dir))
    for key in ('ACL_OP_COMPILER_CACHE_DIR','ASCEND_CACHE_PATH','XDG_CACHE_HOME'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    directory=Path(resolved._run_dir)/'local';directory.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv),'--_worker','--_run-dir',args._run_dir]
    try:
        with (directory/'worker.log').open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        report=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {'status':'failed','error':'Worker exited without report'}
        if result.returncode and report.get('status')=='passed':report.update(status='failed',error='Worker exited after reporting pass')
        report.update(returncode=result.returncode,artifact_dir=args._run_dir,command=command)
    except subprocess.TimeoutExpired:
        report={'status':'failed','error':'Worker timeout; revalidate card before further NPU judgments','artifact_dir':args._run_dir}
    common.write_json(args.output,report)
    print(json.dumps({'status':report['status'],'error':report.get('error'),'report':str(args.output),'artifact_dir':args._run_dir}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
