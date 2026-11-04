"""Actual inline pipeline checks; no external TopK-position input exists here."""
def component_checks_inline(adapter,reference,torch,device,incumbent=None,directory=None):
    if adapter.metadata.get('delayed_temperature') and directory is None:raise ValueError('Explicit component archive directory required')
    from probe_mindie_laser_sparse_attention import check_pipeline,compare_candidate_states
    from mindie_laser_sparse_reference import _error
    records=[]
    cases=((1,1,1,1,64,0),(1,1,17,257,64,1),(1,1,128,385,128,5),
        (2,3,17,769,128,5),(1,3,129,1153,64,5),(1,1,129,65537,64,5),
        (1,1,17,1025,128,0),(1,1,257,129,64,5),(1,1,17,385,128,1),
        (1,1,1,257,128,5),(1,1,64,641,128,5),(1,1,65,897,64,5),(1,2,128,1153,128,5))
    for index,(b,h,q,k,d,std) in enumerate(cases):
        gen=torch.Generator().manual_seed(11400+index)
        inputs=[(torch.randn(b,h,n,d,generator=gen)*scale).to(torch.bfloat16).to(device)
            for n,scale in ((q,std),(k,std),(k,5))]
        payload=adapter.prepare(*inputs);state=adapter.run_native(payload)
        try:entry=check_pipeline(adapter,payload,state,torch)
        except Exception as exc:
            if adapter.metadata.get('delayed_temperature'):
                from pathlib import Path
                import json
                folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
                torch.save(dict(inputs=tuple(x.cpu() for x in inputs),geometry=payload.layout,base_tensors=tuple(x.cpu() for x in payload.base.tensors[:3]),producer=tuple(x.cpu() for x in state[1]),output=state[0].cpu(),error=str(exc)),folder/'failed.pt')
                (folder/'partial.json').write_text(json.dumps(dict(completed=records,failed_case=index,error=str(exc)),indent=2)+'\n')
            raise
        if adapter.metadata.get('delayed_temperature'):
            other=adapter.run_native(payload)
            if any(not torch.equal(a.cpu().contiguous().view(torch.uint8),b.cpu().contiguous().view(torch.uint8)) for a,b in zip(state[1],other[1])):raise AssertionError('Delay initialized state nondeterminism')
            entry['initialized_state_repeat_byte_errors']=0
        if reference is not None:
            entry['reference_bits']=compare_candidate_states(state,reference.run_native(reference.prepare(*inputs)),payload.layout,torch,
                reference_candidates_per_block=1,reference_bulk_columns=8,reference_score_chunk_blocks=8)
        if incumbent is not None:
            from mindie_laser_fma_checks import compare_incumbent
            entry['incumbent_bits']=compare_incumbent(state,incumbent.run_native(incumbent.prepare(*inputs)),torch)
        # The existing independent PV kernel consumes the actual new packet.
        replay=adapter.pv.run_native(state[4])
        entry['independent_pv_error']=_error(state[3][0].cpu(),replay[0].cpu())
        entry.update(shape=[b,h,q,k,d],std=std,scope='Actual new inline qkv pipeline; own FP64 PV + source relation + original PV replay; not unused old-finalizer component tests')
        records.append(entry)
        if adapter.metadata.get('delayed_temperature'):
            from pathlib import Path
            import json
            folder=Path(directory);folder.mkdir(parents=True,exist_ok=True);(folder/'partial.json').write_text(json.dumps(dict(completed=records),indent=2)+'\n')
    return records
