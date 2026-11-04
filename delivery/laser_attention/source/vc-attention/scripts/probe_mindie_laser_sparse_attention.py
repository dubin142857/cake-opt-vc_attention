"""Full block-two/top-eight attention experiment within the existing worker."""
from pathlib import Path
import json,os,traceback
from probe_attention_sparse_components import quant,base,common
from mindie_laser_sparse_candidates import load_candidate_attention,validate_producer_counters,reference_candidates
from mindie_laser_sparse_reference import _error

def check_selector_indices(got,ids,masked,valid):
    import torch
    if (ids<0).any() or (ids>=128).any():raise AssertionError('Selector out-of-bounds index')
    finite=torch.isfinite(got)
    if not torch.equal(got[finite],masked.gather(-1,ids)[finite]):raise AssertionError('Selector finite value/index mismatch')
    if valid>1 and ((ids[:,0]==ids[:,1]).any() or (ids>=valid).any()):raise AssertionError('Selector did not exclude one exact position')

def selector_checks(adapter,device):
    import torch
    generator=torch.Generator().manual_seed(8701);records=[]
    cpu_geometry=adapter.make_positions('cpu');position=cpu_geometry.to(device)
    if not torch.equal(position.cpu().view(torch.int16),cpu_geometry.view(torch.int16)):raise AssertionError('Geometry H2D changed storage bits')
    for rows in (1,17,64,129):
        padded=(rows+63)//64*64
        for valid in (0,1,2,3,77,128):
            for pattern in ('integer','ties','peaks'):
                cpu=torch.randint(-31,32,(padded,128),generator=generator).half() if pattern=='integer' else torch.zeros(padded,128,dtype=torch.float16)
                if pattern=='peaks':
                    cpu[:,:]=-7;cpu[:,0]=3;cpu[:,valid-1]=7
                    if valid>2:cpu[:,valid//2]=5
                got,ids=adapter.probe(cpu.to(device),position,rows,valid)
                got,ids=got.cpu().t()[:rows],ids.cpu().t()[:rows].long()
                masked=cpu[:rows].clone();masked[:,valid:]=-float('inf')
                want=masked.topk(2,-1).values
                if not torch.equal(got.contiguous().view(torch.int16),want.contiguous().view(torch.int16)):
                    raise AssertionError('Selector values disagree '+str((rows,valid,pattern,got[:2].tolist(),want[:2].tolist(),ids[:2].tolist())))
                check_selector_indices(got,ids,masked,valid)
                records.append(dict(rows=rows,valid=valid,pattern=pattern,value_bit_differences=0,indices_legal=True))
    return records


def check_pipeline(adapter,payload,state,torch):
    if payload.layout.get('coarse_refine'):
        from plugins.vc_attention_e121.checks import pipeline
        return pipeline(adapter,payload,state,'npu:'+str(adapter.device))
    from mindie_laser_sparse_pv import validate_counters
    g=payload.layout;p=payload.base;out,producer,selected,pv,packet=state
    counts=validate_producer_counters(producer,g)
    if g.get('inline_pv'):
        from mindie_laser_candidate_inline_pv import validate_inline_pv_counters
        pv_counts=validate_inline_pv_counters(pv,g)
    else:pv_counts=validate_counters(pv,g['heads']*g['qwork'],g['rank'])
    ids,weights,_=selected;ids=ids.cpu().long();weights=weights.cpu()
    head=torch.arange(g['heads']).repeat_interleave(g['qwork'])[:,None]
    if (ids<0).any() or (ids>=g['heads']*g['kpad']).any() or not (ids//g['kpad']==head).all() or (ids%g['kpad']>=g['keys']).any():raise AssertionError('Cross-head or padded selected V index')
    if ids.shape[-1]>1 and (ids.sort(-1).values[:,1:]==ids.sort(-1).values[:,:-1]).any():raise AssertionError('Repeated selected key')
    cpu_value=p.tensors[2].cpu().view(-1,128)
    reference=(cpu_value[ids].double()*weights.double()[...,None]).sum(1)
    core_error=_error(pv[0].cpu(),reference)
    if core_error['relative_rmse'] is None or core_error['relative_rmse']>1e-5 or core_error['max_abs']>1e-4:raise AssertionError('PV contract discrepancy: '+str(core_error))
    unpack=packet.indices.cpu().transpose(1,2).reshape(-1,16)[:len(ids),:g['rank']].long()
    unpackw=packet.weights.cpu().transpose(1,2).reshape(-1,16)[:len(ids),:g['rank']]
    if not torch.equal(unpack,ids) or not torch.equal(unpackw.contiguous().view(torch.int16),weights.contiguous().view(torch.int16)):raise AssertionError('Packet roundtrip changed bits')
    expected=reference.view(p.shape[0],p.shape[1],g['qwork'],128)[:,:,:g['queries'],:p.shape[3]]
    error=_error(out.cpu(),expected)
    if not torch.isfinite(out).all():raise AssertionError('Nonfinite complete candidate output')
    # Output conversion is deliberately separate from the FP32 PV check.
    result=dict(producer_counters=counts,pv_counters=pv_counts,fp32_pv_error=core_error,public_output_error=error,packet_bit_differences=0)
    if g.get('delayed_temperature'):
        from plugins.vc_attention_e122.checks import supplement
        result.update(supplement(adapter,payload,state))
    return result

def basis_producer_checks(adapter,torch,device,reference=None):
    if adapter.metadata.get('delayed_temperature'):
        from plugins.vc_attention_e122.checks import basis
        return basis(adapter,torch,device)
    if adapter.metadata.get('coarse_refine'):
        from plugins.vc_attention_e121.checks import basis
        return basis(adapter,torch,device)
    records=[]
    for b,h,rows,keys,dim in ((1,1,1,1,64),(1,1,129,385,64),(2,3,17,257,128),(1,3,385,77,128)):
        q=torch.zeros(b,h,rows,dim,dtype=torch.bfloat16);q[...,0]=8
        k=((torch.arange(b*h*keys*dim).reshape(b,h,keys,dim)%127)-63).to(torch.bfloat16)
        v=(k.float()/8).to(torch.bfloat16)
        payload=adapter.prepare(q.to(device),k.to(device),v.to(device));state=adapter.run_native(payload)
        entry=check_pipeline(adapter,payload,state,torch);g=payload.layout
        expected=(q.float()@k.float().transpose(-1,-2)).half()
        expected=(expected.float()*torch.tensor(dim**-.5,dtype=torch.float16).float()).half()
        cpb=g.get('candidates_per_block',2)
        expected=torch.nn.functional.pad(expected,(0,g['kpad']-keys),value=-float('inf')).reshape(b*h,rows,g['kpad']//128,128).topk(cpb,-1).values.flatten(-2)
        online=g.get('producer_counter_version') in (5,6,7,8)
        if online:
            expected=expected.topk(g['rank'],-1).values
            actual=state[1][0].cpu().view(b*h,g['qwork'],8)[:,:rows,:g['rank']]
        else:
            actual=state[1][0].cpu()[:,:cpb*(g['kpad']//128),:rows].transpose(1,2)
        if not torch.equal(actual.contiguous().view(torch.int16),expected.contiguous().view(torch.int16)):raise AssertionError('Basis QK producer values not exact')
        indices=state[1][1].cpu()[:,:g['rank'] if online else cpb*(g['kpad']//128),:rows].transpose(1,2).long()
        # Validate every finite candidate against its original K position.
        for head in range(b*h):
            ids=indices[head]-head*g['kpad'];finite=torch.isfinite(actual[head]);safe=ids.clamp(0,keys-1)
            logits=(q.float()@k.float().transpose(-1,-2)).flatten(0,1)[head].half()
            logits=(logits.float()*torch.tensor(dim**-.5,dtype=torch.float16).float()).half()
            if ((ids<0)|(ids>=keys))[finite].any() or not torch.equal(actual[head][finite],logits.gather(-1,safe)[finite]):raise AssertionError('Basis QK candidate key mapping')
        if reference is not None:entry['reference_bits']=compare_candidate_states(state,reference.run_native(reference.prepare(q.to(device),k.to(device),v.to(device))),g,torch,reference_candidates_per_block=reference.metadata['candidates_per_block'],reference_bulk_columns=reference.metadata.get('output_bulk_columns',1),reference_score_chunk_blocks=reference.metadata.get('score_chunk_blocks',1))
        entry.update(shape=[b,h,rows,keys,dim],candidate_value_bit_differences=0);records.append(entry)
    return records


def compare_finalized_states(actual,reference,layout,torch):
    if layout.get('candidates_per_block')!=1:raise ValueError('Finalize needs block1 comparison')
    validate_producer_counters(actual[1],layout);validate_producer_counters(reference[1],layout)
    columns=layout['kpad']//128;result={}
    pairs=[('producer_counter',actual[1][2],reference[1][2]),('pv_counter',actual[3][1],reference[3][1]),
           ('ids',actual[2][0],reference[2][0]),('scores',actual[2][2],reference[2][2]),('packet_ids',actual[4].indices,reference[4].indices)]
    for i in (0,1):pairs.append(('candidate_'+str(i),actual[1][i][:,:columns],reference[1][i][:,:columns]))
    for name,a,b in pairs:
        a,b=a.cpu().contiguous(),b.cpu().contiguous()
        if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Finalize reference shape/type: '+name)
        wrong=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum());result[name]=wrong
        if wrong:raise AssertionError('Finalize fixed-stage bit mismatch: '+name)
    weight=actual[2][1].cpu();oldweight=reference[2][1].cpu();delta=(weight.float()-oldweight.float()).abs()
    result['weights_max_abs']=float(delta.max());result['weights_sum_max_abs']=float((weight.float().sum(-1)-1).abs().max())
    if not torch.isfinite(weight).all() or (weight<0).any() or result['weights_max_abs']>1e-3 or result['weights_sum_max_abs']>1e-3:raise AssertionError('Finalize weight contract')
    from mindie_laser_sparse_pv import pack_rows
    pi,pw=pack_rows(actual[2][0].cpu(),weight,2**31-1)
    for name,a,b in [('packet_index_roundtrip',actual[4].indices.cpu(),pi),('packet_weight_roundtrip',actual[4].weights.cpu(),pw)]:
        if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):raise AssertionError('Finalize '+name)
    public=actual[0].cpu();g=layout
    expected=actual[3][0].cpu().view(public.shape[0],public.shape[1],g['qwork'],128)[:,:,:g['queries'],:public.shape[3]].to(public.dtype).contiguous()
    if not torch.equal(public.contiguous().view(torch.uint8),expected.view(torch.uint8)):raise AssertionError('Finalize public conversion')
    result.update(scope='Producer, builtin TopK scores/IDs, packet IDs and counters exact. FP32 softmax/half weights have a separate contract; PV checked against own weights.',public_vs_previous=_error(public,reference[0].cpu()),pv_vs_previous=_error(actual[3][0].cpu(),reference[3][0].cpu()))
    return result


def compare_candidate_states(actual,reference,layout,torch,*,reference_candidates_per_block=2,reference_bulk_columns=1,reference_score_chunk_blocks=1):
    if layout.get('coarse_refine') or layout.get('delayed_temperature'):
        return dict(scope='Distinctdeclaredcoarse/refineapproximation; ownsource/point/PVchecks govern',public_vs_previous=_error(actual[0].cpu(),reference[0].cpu()))
    if layout.get('postmax_scale'):
        from mindie_laser_postmax_checks import compare_postmax_states
        return compare_postmax_states(actual,reference,layout,torch)
    if layout.get('producer_counter_version') in (5,6,7):
        if (reference_candidates_per_block,reference_bulk_columns,reference_score_chunk_blocks)!=(1,8,8):raise ValueError('Online reference must be E94/E97 block1 score8')
        from mindie_laser_online_checks import compare_online_states
        return compare_online_states(actual,reference,layout,torch)
    if layout.get('native_finalize',False):
        if reference_candidates_per_block!=1 or reference_bulk_columns!=8 or reference_score_chunk_blocks!=8:raise ValueError('Finalize reference contract mismatch')
        return compare_finalized_states(actual,reference,layout,torch)
    cpb=layout.get('candidates_per_block',2)
    actual_count,reference_count=actual[1][2],reference[1][2]
    counter_scope=None
    if layout.get('output_bulk_columns',1)!=reference_bulk_columns or layout.get('score_chunk_blocks',1)!=reference_score_chunk_blocks:
        validate_producer_counters(actual[1],layout)
        rg=dict(layout,output_bulk_columns=reference_bulk_columns,producer_counter_version=4 if reference_score_chunk_blocks==32 else 3 if reference_score_chunk_blocks>1 else 1 if reference_bulk_columns==1 else 2,
                score_chunk_blocks=reference_score_chunk_blocks,
                candidates_per_block=reference_candidates_per_block)
        validate_producer_counters(reference[1],rg)
        cores=min(24,layout['heads']*(layout['qwork']//128))
        actual_count,reference_count=actual_count.cpu().clone(),reference_count.cpu().clone()
        actual_count[cores:,4:6]=0;reference_count[cores:,4:6]=0
        if layout.get('score_chunk_blocks',1)!=reference_score_chunk_blocks:
            # Protocol-dependent fields are fully validated above on each side.
            actual_count[:cores,2:5]=0;reference_count[:cores,2:5]=0
            actual_count[cores:,2]=0;reference_count[cores:,2]=0
            counter_scope='Logical counter bytes exact; score-group protocol and output DMA counters independently validated'
        else:counter_scope='Legacy counter bytes exact; version-specific output batches/DMA calls independently validated'

    if cpb==1 and reference_candidates_per_block==2:
        blocks=layout['kpad']//128;result={}
        for i in (0,1):
            a=actual[1][i][:,:blocks].cpu().contiguous()
            b=reference[1][i][:,:2*blocks:2].cpu().contiguous()
            if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Block1/rank0 candidate shape or dtype')
            wrong=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum());result['producer_rank0_'+str(i)]=wrong
            if wrong:raise AssertionError('Block1 differs from reference rank0: '+str((i,wrong)))
        a,b=actual_count.cpu(),reference_count.cpu()
        if not torch.equal(a,b):raise AssertionError('Block1 producer counter coverage differs')
        result.update(scope='Only same-contract producer rank0 is bitwise compared; complete algorithms differ',
                      complete_algorithm_error=_error(actual[0].cpu(),reference[0].cpu()))
        if counter_scope:result['counter_scope']=counter_scope
        return result

    if cpb!=reference_candidates_per_block:raise ValueError('Unsupported candidate contract comparison')
    pairs=[('public',actual[0],reference[0]),('pv',actual[3][0],reference[3][0]),
           ('producer_count',actual_count,reference_count),('pv_count',actual[3][1],reference[3][1]),
           ('packet_index',actual[4].indices,reference[4].indices),('packet_weight',actual[4].weights,reference[4].weights)]
    columns=layout['kpad']//128*cpb
    for i in (0,1):pairs.append(('real_candidate_'+str(i),actual[1][i][:,:columns],reference[1][i][:,:columns]))
    for i in (0,1,2):pairs.append(('selection_'+str(i),actual[2][i],reference[2][i]))
    result={}
    for name,a,b in pairs:
        a,b=a.cpu().contiguous(),b.cpu().contiguous()
        if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Candidate reference shape/type: '+name)
        wrong=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum());result[name]=wrong
        if wrong:raise AssertionError('Candidate reference bit mismatch: '+name+' '+str(wrong))
    if counter_scope:result['counter_scope']=counter_scope
    return result

def finalized_boundary_metadata(payload,state):
    """One already checked invocation, outside all timing/profiler loops."""
    selected=state[2]
    if not hasattr(selected,'topk'):return None
    tensors=dict(value=payload.base.tensors[2].view(-1,128),candidates=selected.topk.candidates,
        scores=selected.topk.scores,positions=selected.topk.positions,ids=selected[0],weights=selected[1],
        packet_ids=state[4].indices,packet_weights=state[4].weights,pv=state[3][0],counters=state[3][1],geometry=payload.finalize_geometry)
    rows={}
    for name,tensor in tensors.items():
        storage=tensor.untyped_storage();width=tensor.element_size()
        rows[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),strides_bytes=[n*width for n in tensor.stride()],
            data_address=tensor.data_ptr(),storage_address=storage.data_ptr(),storage_bytes=storage.nbytes(),
            storage_offset_bytes=tensor.storage_offset()*width,device=tensor.device.index)
    return dict(scope='target_validation invocation only; ABI/shared tensors, not private UB or profiler-call addresses',
        layout=payload.layout,tensors=rows,geometry_sha256=common.tensor_sha256(payload.finalize_geometry.cpu()))


def worker(args):
    directory=Path(args._run_dir)/'baseline/local';directory.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',mode='sparse_attention',stage='provenance',
        scope='complete full-QK block2/top8 approximate attention, including selection/layout/PV/cast; video unverified',
        sources={str(p):common.sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('mindie_laser_sparse_candidates.py'),Path(__file__).with_name('probe_attention_sparse_components.py')]},paired_groups={})
    save=lambda:common.write_json(directory/'report.json',report);save()
    try:
        resolved=quant.build_arguments(args,'baseline');report['provenance']=base.provenance(resolved)
        import torch,torch_npu
        from mindie_laser_quant_pairing import run_paired
        from mindie_laser_quant_trace import capture_native_traces
        from mindie_laser_global import load_global_attention
        torch.set_num_threads(args.cpu_threads);torch.npu.set_device(args.device)
        torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE':'disable','ACL_OP_COMPILER_CACHE_DIR':os.environ['ACL_OP_COMPILER_CACHE_DIR']})
        if hasattr(torch.npu,'matmul'):torch.npu.matmul.allow_hf32=False
        source=json.loads(Path(resolved.source_manifest).read_text());laser,transforms=base.load_adapter(resolved,source)
        device=f'npu:{args.device}'
        reference=None
        if args.candidate_manifest:
            from mindie_laser_sparse_candidates import load_candidate_variant,CANDIDATE_LOADERS,candidate_manifest_operation
            candidate=load_candidate_variant(laser,args.candidate_manifest,args.fused_pv_build,args.device)
            if args.reference_manifest:
                reference=load_candidate_variant(laser,args.reference_manifest,args.fused_pv_build,args.device)
                if candidate.metadata['candidates_per_block']==1 and reference.metadata['candidates_per_block'] not in (1,2):
                    raise ValueError('Block1 needs a compatible block1 or block2 reference')
                report['reference_provider']=dict(build=reference.evidence,metadata=reference.metadata)
            for manifest in filter(None,(args.candidate_manifest,args.reference_manifest)):
                module,_=CANDIDATE_LOADERS[candidate_manifest_operation(manifest)]
                path=Path(__file__).with_name(module+'.py');report['sources'][str(path)]=common.sha256_file(path)
        else:
            candidate=load_candidate_attention(laser,args.candidate_build,args.fused_pv_build,args.device)
        if args.candidate_lookup_build:
            from mindie_laser_candidate_lookup import load_candidate_lookup_attention
            reference=candidate;candidate=load_candidate_lookup_attention(laser,args.candidate_lookup_build,args.fused_pv_build,args.device)
            report['reference_provider']=dict(build=reference.evidence,metadata=reference.metadata)
            lookup_source=Path(__file__).with_name('mindie_laser_candidate_lookup.py');report['sources'][str(lookup_source)]=common.sha256_file(lookup_source)

        if args.candidate_dma_build:
            from mindie_laser_candidate_dma_overlap import load_candidate_dma_attention
            report['unmeasured_loaded_provider']=report['reference_provider']
            reference=candidate;candidate=load_candidate_dma_attention(laser,args.candidate_dma_build,args.fused_pv_build,args.device)
            report['reference_provider']=dict(build=reference.evidence,metadata=reference.metadata)
            dma_source=Path(__file__).with_name('mindie_laser_candidate_dma_overlap.py');report['sources'][str(dma_source)]=common.sha256_file(dma_source)
        incumbent=None
        if os.environ.get('VC_ATTENTION_INCUMBENT_MANIFEST'):
            from mindie_laser_sparse_candidates import load_candidate_variant
            incumbent=load_candidate_variant(laser,os.environ['VC_ATTENTION_INCUMBENT_MANIFEST'],args.fused_pv_build,args.device)
            report['measured_incumbent']=dict(build=incumbent.evidence,metadata=incumbent.metadata)
        report['scope']=f"complete full-QK block{candidate.metadata['candidates_per_block']}/top8 approximate attention, including selection/layout/PV/cast; refinement and video quality unverified"
        report.update(provider=dict(build=candidate.evidence,metadata=candidate.metadata,pv=candidate.pv.evidence),transforms=transforms,stage='selector_checks');save()
        with torch.inference_mode():
            if hasattr(candidate,'finalizer'):
                from probe_mindie_laser_sparse_finalize import component_checks,audit_loaded_providers
                audit_loaded_providers(report,candidate,reference,common)
                component_source=Path(__file__).with_name('probe_mindie_laser_sparse_finalize.py');report['sources'][str(component_source)]=common.sha256_file(component_source)
                report['finalize_checks']=component_checks(candidate,reference,torch,device,directory/'finalize_checks');save()
                report['stage_scopes']=candidate.metadata['stage_scopes']
                helper_source=Path(__file__).with_name('mindie_laser_sparse_finalize.py');report['sources'][str(helper_source)]=common.sha256_file(helper_source)
            elif candidate.metadata.get('inline_pv'):
                from probe_mindie_laser_sparse_finalize import audit_loaded_providers
                from probe_mindie_laser_inline_pv import component_checks_inline
                audit_loaded_providers(report,candidate,reference,common)
                if incumbent is not None:
                    evidence={};audit_loaded_providers(evidence,incumbent,None,common)
                    report['candidate_mapped_library_sha256'].update(evidence['candidate_mapped_library_sha256'])
                if candidate.metadata.get('pv_arithmetic'):
                    from plugins.vc_attention_e118.math_probe import primitive_checks
                    report['stage']='primitive_check';save()
                    report['primitive_checks']=primitive_checks(candidate,torch,device,directory/'fma-primitive');save()
                    if not report['primitive_checks']['selected_mode_passed']:raise AssertionError('Actual vmla primitive premise failed')
                if getattr(candidate,'pair_batch',False):
                    from plugins.vc_attention_e119.pair_probe import primitive_checks as pair_checks
                    report['pair_checks']=pair_checks(candidate,torch,device,directory/'pair-primitive');save()
                    if not report['pair_checks']['passed']:raise AssertionError('Paired reduction premise failed')
                if candidate.metadata.get('coarse_refine'):
                    from plugins.vc_attention_e121.checks import primitives,components
                    report['refine_primitive_checks']=primitives(candidate,device,directory/'refine-primitive');save()
                    if os.environ.get('VC_ATTENTION_REFINE_DIAGNOSTIC_ONLY')=='1':raise RuntimeError('Intentional check-only terminal after primitive archive; no combined target measurement')
                    report['finalize_checks']=components(candidate,torch,device,directory/'refine-components');save()
                else:
                    report['finalize_checks']=component_checks_inline(candidate,reference,torch,device,incumbent=None if candidate.metadata.get('delayed_temperature') else incumbent,directory=directory/'delay-components');save()
                report['stage_scopes']=candidate.metadata['stage_scopes']
                for source_name in ('probe_mindie_laser_inline_pv.py','mindie_laser_candidate_inline_pv.py','mindie_laser_online_checks.py'):
                    source_path=Path(__file__).with_name(source_name);report['sources'][str(source_path)]=common.sha256_file(source_path)
            if args.candidate_cold_only:
                cpu=quant.synthetic_inputs(torch,[1,1,args.rows,args.dim],5,args.seed);inputs=[x.to(device) for x in cpu]
                candidate.prepare(*inputs)
                def cold_prepare():
                    candidate.geometry_cache.clear()
                    return candidate.prepare(*inputs)
                def cold_full():
                    candidate.geometry_cache.clear()
                    return candidate(*inputs)
                calls=dict(candidate_prepare_hot=lambda:candidate.prepare(*inputs),candidate_prepare_cold=cold_prepare,candidate_full_cold=cold_full)
                report.update(stage='cold_geometry_cost',scope='independent shape-cache cold/hot supplement; no new primary attention comparison',shape=list(cpu[0].shape),std=5,seed=args.seed);save()
                component_calls={}
                if reference is not None:
                    current_payload=candidate.prepare(*inputs);old_payload=reference.prepare(*inputs)
                    current_state=candidate.run_native(current_payload);old_state=reference.run_native(old_payload)
                    report['reference_bits']=compare_candidate_states(current_state,old_state,current_payload.layout,torch,reference_candidates_per_block=reference.metadata['candidates_per_block'],reference_bulk_columns=reference.metadata.get('output_bulk_columns',1),reference_score_chunk_blocks=reference.metadata.get('score_chunk_blocks',1));save()
                    # Capture initialized real candidate columns outside timing, for full-row semantic studies.
                    physical_columns=current_payload.layout['kpad']//128*candidate.metadata['candidates_per_block']
                    semantic_path=directory/'candidate_semantic_state.pt'
                    torch.save(dict(layout=current_payload.layout,shape=current_payload.base.shape,dtype=str(current_payload.base.dtype),qkv_cpu=cpu,
                        value_half=current_payload.base.tensors[2].cpu().contiguous(),
                        candidate_values=current_state[1][0][:,:physical_columns].cpu().contiguous(),candidate_indices=current_state[1][1][:,:physical_columns].cpu().contiguous(),
                        selected_indices=current_state[2][0].cpu(),selected_weights=current_state[2][1].cpu(),selected_scores=current_state[2][2].cpu(),
                        pv_float=current_state[3][0].cpu(),public_output=current_state[0].cpu(),producer_counters=current_state[1][2].cpu(),pv_counters=current_state[3][1].cpu()),semantic_path)
                    report['semantic_artifact']=dict(path=str(semantic_path),sha256=common.sha256_file(semantic_path),bytes=semantic_path.stat().st_size,
                        scope='Initialized physical candidate columns and padded query rows; original QKV, actual padded half V, native selected top8 and outputs. Not full QK/dense attention.')
                    save()
                    component_calls={
                        'producer':dict(previous=lambda:reference.produce(old_payload),candidate=lambda:candidate.produce(current_payload)),
                        'selection':dict(previous=lambda:reference.select(old_payload,old_state[1]),candidate=lambda:candidate.select(current_payload,current_state[1])),
                        'consumer':dict(previous=lambda:reference.consume(old_payload,old_state[2]),candidate=lambda:candidate.consume(current_payload,current_state[2]))}
                    for label,components in component_calls.items():
                        def update(row,label=label):report['paired_groups'][label]=row;save()
                        report['paired_groups'][label]=run_paired(components,synchronize=torch.npu.synchronize,warmup=args.warmup,repeats=args.repeats,trials=args.trials+(args.trials%2),seed=args.seed,on_trial=update);save()
                quant.benchmark_calls(calls,args,torch,report,save)
                if args.trace:
                    traced=dict(calls,**{label+'_'+arm:fn for label,components in component_calls.items() for arm,fn in components.items()})
                    report['prepared_trace']=capture_native_traces(traced,torch=torch,torch_npu=torch_npu,directory=directory/'trace',repeats=3)
                report.update(status='passed',stage='completed');save();return 0
            report['selector_checks']=selector_checks(candidate,device);save()
            report['basis_checks']=basis_producer_checks(candidate,torch,device,reference);save()
            report['small_checks']=[]
            for b,h,length,dim,std in ((1,1,17,64,0),(1,1,129,128,1),(2,3,257,64,5),(1,1,385,128,5)):
                cpu=quant.synthetic_inputs(torch,[b,h,length,dim],std,args.seed)
                inputs=[x.to(device) for x in cpu];payload=candidate.prepare(*inputs);state=candidate.run_native(payload)
                entry=check_pipeline(candidate,payload,state,torch)
                if reference is not None:entry['reference_bits']=compare_candidate_states(state,reference.run_native(reference.prepare(*inputs)),payload.layout,torch,reference_candidates_per_block=reference.metadata['candidates_per_block'],reference_bulk_columns=reference.metadata.get('output_bulk_columns',1),reference_score_chunk_blocks=reference.metadata.get('score_chunk_blocks',1))
                ids=torch.arange(length);dense=common.dense_reference(*cpu,ids)
                entry.update(shape=[b,h,length,dim],std=std,dense_error=_error(state[0].cpu(),dense));report['small_checks'].append(entry);save()
            if args.smoke_only:report.update(status='passed',stage='smoke_completed');save();return 0
            if args.rows!=args.keys:raise ValueError('Current target benchmark requires equal Q/K lengths; operator supports separate extents')
            report['stage']='target_validation';save()
            cpu=quant.synthetic_inputs(torch,[1,1,args.rows,args.dim],5,args.seed);inputs=[x.to(device) for x in cpu]
            laser_payload=laser.prepare(*inputs);payload=candidate.prepare(*inputs)
            state=candidate.run_native(payload);report['target_check']=check_pipeline(candidate,payload,state,torch)
            report['architecture_boundary']=finalized_boundary_metadata(payload,state)
            if report['architecture_boundary'] is not None:
                if os.environ.get('CAKE_ATTENTION_LOWERING'):
                    from plugins.vc_attention.architecture import finalizer_binding
                    report['architecture_boundary']['native']=finalizer_binding(candidate.finalizer)
            if os.environ.get('CAKE_ATTENTION_LOWERING'):
                from core.opgen.lowering import LoweringPlan
                from plugins.vc_attention.architecture import attest_boundary
                lowering=LoweringPlan.from_json(Path(os.environ['CAKE_ATTENTION_LOWERING']).read_text())
                binding=attest_boundary(lowering,report['architecture_boundary'],report['candidate_mapped_library_sha256'])
                binding_path=directory/'lowering-receipt.json';binding_path.write_text(binding.to_json(pretty=True)+'\n')
                report['architecture_binding']=dict(path=str(binding_path),sha256=common.sha256_file(binding_path),status=binding.status,model=lowering.architecture.fingerprint(),scope=lowering.scope)
            if reference is not None:
                reference_payload=reference.prepare(*inputs);reference_state=reference.run_native(reference_payload)
                report['target_check']['reference_bits']=compare_candidate_states(state,reference_state,payload.layout,torch,reference_candidates_per_block=reference.metadata['candidates_per_block'],reference_bulk_columns=reference.metadata.get('output_bulk_columns',1),reference_score_chunk_blocks=reference.metadata.get('score_chunk_blocks',1))
            if incumbent is not None:
                incumbent_payload=incumbent.prepare(*inputs);incumbent_state=incumbent.run_native(incumbent_payload)
                report['target_check']['incumbent_pipeline']=check_pipeline(incumbent,incumbent_payload,incumbent_state,torch)
                report['target_check']['incumbent_reference']=compare_candidate_states(incumbent_state,reference_state,incumbent_payload.layout,torch,reference_candidates_per_block=1,reference_bulk_columns=8,reference_score_chunk_blocks=8)
                from mindie_laser_fma_checks import compare_incumbent
                if not candidate.metadata.get('coarse_refine') and not candidate.metadata.get('delayed_temperature'):
                    report['target_check']['incumbent_bits']=compare_incumbent(state,incumbent_state,torch)
                else:
                    report['target_check']['incumbent_algorithm_error']=_error(state[0].cpu(),incumbent_state[0].cpu())
            report.update(shape=list(cpu[0].shape),std=5,seed=args.seed)
            query_ids=torch.linspace(0,args.rows-1,min(32,args.rows)).long().unique();dense=common.dense_reference(*cpu,query_ids)
            got=state[0].cpu();report['target_check']['dense_error']=_error(got.index_select(2,query_ids),dense)
            # Independent CPU block2 reference: full QK for these sampled rows.
            from mindie_laser_sparse_reference import support_probe
            from mindie_laser_sparse_candidates import reference_output
            dots=cpu[0].index_select(2,query_ids).float()@cpu[1].float().transpose(-1,-2)
            logits=(dots.half().float()*torch.tensor(args.dim**-.5,dtype=torch.float16).float()).half().float()
            expected,_=reference_output(logits,cpu[2],candidates_per_block=candidate.metadata['candidates_per_block']);report['target_check']['cpu_block_reference_error']=_error(got.index_select(2,query_ids),expected)
            if candidate.metadata['candidates_per_block']==2:report['target_check']['cpu_block2_error']=report['target_check']['cpu_block_reference_error']
            extra={}
            if candidate.metadata.get('coarse_refine'):
                extra=dict(coarse_ids=state[1][9].cpu(),coarse_scores=state[1][10].cpu(),quantized=tuple(x.cpu() for x in payload.quantized),independent_point_scores=candidate.latest_point_scores)
            if candidate.metadata.get('delayed_temperature'):
                extra=dict(raw_reference_scores=candidate.latest_raw_reference[0],raw_reference_ids=candidate.latest_raw_reference[1],selected_scores=state[2][2].cpu())
            torch.save(dict(inputs=cpu,query_ids=query_ids,output=got,selected_indices=state[2][0].cpu(),selected_weights=state[2][1].cpu(),producer_counters=state[1][2].cpu(),pv_counters=state[3][1].cpu(),**extra),directory/'target.pt')
            # Fresh vendor call after all candidate correctness checks.
            marker=Path(os.environ[base.MARKER_ENV]);offset=marker.stat().st_size if marker.exists() else 0
            laser_output=laser.run_prepared(laser_payload).cpu();torch.npu.synchronize()
            with marker.open('rb') as stream:stream.seek(offset);fresh=stream.read().decode()
            report['host_markers']=base.parse_markers(fresh,os.getpid())
            build=report['provenance']['build_manifest'];cann=build.get('cann_root') or build['definitions']['CUSTOM_ASCEND_CANN_PACKAGE_PATH']
            base.inspect_provider('local',resolved.vendor_root,resolved.bridge_path,report,directory,cann_root=cann,soc=build['soc'])
            report['laser_dense_error']=_error(laser_output.index_select(2,query_ids),dense);save()
            if reference is not None:
                calls=dict(laser=lambda:laser.run_prepared(laser_payload),previous=lambda:(incumbent.run_prepared(incumbent_payload) if incumbent is not None else reference.run_prepared(reference_payload)),candidate=lambda:candidate.run_prepared(payload))
            else:
                overlap=load_global_attention(laser,Path(__file__).resolve().parents[1]/'results/mindie_laser_global_stream_overlap_cann920_build.json',args.device,operation='global_stream_overlap')
                calls=dict(laser=lambda:laser.run_prepared(laser_payload),overlap=lambda:overlap.run_prepared(laser_payload),candidate=lambda:candidate.run_prepared(payload))
            report['stage']='timing';save()
            def update(row):report['paired_groups']['attention']=row;save()
            report['paired_groups']['attention']=run_paired(calls,synchronize=torch.npu.synchronize,warmup=args.warmup,repeats=args.repeats,trials=args.trials,seed=args.seed,on_trial=update);save()
            selection=state[2];producer=state[1]
            stages=dict(candidate_producer=lambda:candidate.produce(payload),candidate_selection=lambda:candidate.select(payload,producer),
                candidate_consumer=lambda:candidate.consume(payload,selection),candidate_full=lambda:candidate(*inputs),candidate_prepare=lambda:candidate.prepare(*inputs),
                laser_full=lambda:laser(*inputs),laser_preprocessing=lambda:laser.prepare(*inputs))
            quant.benchmark_calls(stages,args,torch,report,save)
            if args.candidate_metrics:
                if reference is None:raise ValueError('Metrics diagnostic requires a paired previous provider')
                report.update(mode='sparse_metric_diagnostic',stage='paired_producer_metrics',
                    metric_scope='Independent metric captures; no phase percentages or cross-metric counter sum. API timing above excludes profiler.')
                trace_source=Path(__file__).with_name('mindie_laser_quant_trace.py');report['sources'][str(trace_source)]=common.sha256_file(trace_source);save()
                producer_calls=dict(previous=lambda:reference.produce(reference_payload),candidate=lambda:candidate.produce(payload))
                def producer_update(row):report['paired_groups']['producer']=row;save()
                report['paired_groups']['producer']=run_paired(producer_calls,synchronize=torch.npu.synchronize,warmup=args.warmup,
                    repeats=args.repeats,trials=args.trials+(args.trials%2),seed=args.seed,on_trial=producer_update);save()
                metric_calls=dict(laser=calls['laser'],producer_previous=producer_calls['previous'],producer_candidate=producer_calls['candidate'],prepared_candidate=calls['candidate'])
                report['metric_captures']={}
                for metric in args.candidate_metrics:
                    report.update(stage='metric_'+metric);save()
                    capture=capture_native_traces(metric_calls,torch=torch,torch_npu=torch_npu,directory=directory/('metric_'+metric),
                        repeats=3,metric=metric,all_numeric_fields=True)
                    report['metric_captures'][metric]=capture;save()
                    if capture['status']=='failed':raise RuntimeError('Hardware metric capture failed; inspect partial report: '+metric)
                report.update(status='passed',stage='metric_diagnostic_completed');save();return 0
            if args.trace:
                report['stage']='trace';save()
                report['prepared_trace']=capture_native_traces(dict(calls,**stages),torch=torch,torch_npu=torch_npu,directory=directory/'trace',repeats=3)
            report.update(status='passed',stage='completed');save();return 0
    except Exception as e:
        report.update(status='failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());save();traceback.print_exc();return 1
