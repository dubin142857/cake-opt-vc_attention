"""Correctness-only finalize component checks, reused by the existing attention worker."""
from pathlib import Path
from mindie_laser_sparse_finalize import finalize_geometry,finalize_reference
from mindie_laser_sparse_pv import pack_rows,validate_counters,SparsePvPayload
from mindie_laser_sparse_reference import _error

def component_checks(adapter,reference,torch,device,directory=None):
    cases=[(1,128,32,1,64),(1,128,32,3,128),(3,128,32,8,64),(1,256,320,8,128),
           (3,128,320,3,128),(1,128,512,8,128),(1,256,512,1,64),(3,256,320,5,128)]
    records=[]
    for i,(h,q,c,rank,dim) in enumerate(cases+[(1,128,32,3,128)]*4):
        gen=torch.Generator().manual_seed(9700+i);rows=h*q;keys=h*257
        v=(torch.randn(keys,dim,generator=gen)*5).half()
        rr=torch.arange(q)[None,None,:];cc=torch.arange(c)[None,:,None];hh=torch.arange(h)[:,None,None]
        candidate=(hh*257+(cc*17+rr)%257).to(torch.int32).contiguous()
        matrix=(torch.randn(rows,c,generator=gen)*25).half();scores,pos=matrix.topk(rank,-1,sorted=True)
        mutation=None
        if i>=len(cases):
            mutation=['negative_position','wide_position','outside_position','outside_value'][i-len(cases)]
            if mutation=='negative_position':pos[0,0]=-1
            if mutation=='wide_position':pos[0,0]=2**32+1
            if mutation=='outside_position':pos[0,0]=c+1
            if mutation=='outside_value':candidate[0,int(pos[0,0]),0]=keys+1
        native=adapter.finalizer.run_native(v.to(device),candidate.to(device),scores.to(device),pos.to(device),finalize_geometry(rank).to(device))
        got,ids,w,pi,pw,count=[x.cpu() for x in native]
        def fail(message):
            suffix=''
            if directory is not None:
                directory.mkdir(parents=True,exist_ok=True);path=directory/('failed_case_'+str(i)+'.pt')
                if path.exists():raise RuntimeError('Refusing to overwrite finalize diagnostic: '+str(path))
                torch.save(dict(shape=[h,q,c,rank,dim],mutation=mutation,value=v,candidates=candidate,scores=scores,positions=pos,geometry=finalize_geometry(rank),native=[x.cpu() for x in native]),path)
                suffix='; diagnostic='+str(path)
            raise AssertionError(message+'; case='+str(i)+suffix)
        expected_ids,expected_w,_=finalize_reference(v,candidate,scores,pos)
        if not torch.equal(ids,expected_ids):fail('Finalize component mapped IDs')
        maxw=float((w.float()-expected_w.float()).abs().max());sumw=float((w.float().sum(-1)-1).abs().max())
        if not torch.isfinite(w).all() or (w<0).any() or maxw>1e-3 or sumw>1e-3:fail('Finalize component weights max='+str(maxw)+' sum='+str(sumw))
        expected=(v[ids.long().clamp(0,keys-1)].double()*w.double()[...,None]).sum(1)
        bad=(ids<0).any(1);expected[bad]=0
        error=_error(got,expected)
        if error['relative_rmse'] is None or error['relative_rmse']>1e-5 or error['max_abs']>1e-4:fail('Finalize component PV: '+str(error))
        epi,epw=pack_rows(ids,w,keys)
        if not torch.equal(pi,epi) or not torch.equal(pw.view(torch.int16),epw.view(torch.int16)):fail('Finalize component packet padding/order')
        counters=validate_counters((got,count),rows,rank,invalid_rows=int(bad.sum()))
        replay=adapter.pv.run_native(SparsePvPayload(v.to(device),native[3],native[4],rows,rank))
        bitdiff=int((got.contiguous().view(torch.uint8)!=replay[0].cpu().contiguous().view(torch.uint8)).sum())
        records.append(dict(shape=[h,q,c,rank,dim],mutation=mutation,weights_max_abs=maxw,weights_sum_max_abs=sumw,pv_error=error,counters=counters,old_pv_replay_byte_differences=bitdiff))
    if reference is not None:
        gen=torch.Generator().manual_seed(9799)
        cpu=[(torch.randn(1,1,n,64,generator=gen)*5).to(torch.bfloat16) for n in (17,65537,65537)]
        inputs=[x.to(device) for x in cpu];payload=adapter.prepare(*inputs)
        online=payload.layout.get('producer_counter_version') in (5,6)
        if payload.layout['native_finalize'] and not online:raise AssertionError('Large C must select fallback')
        actual=adapter.run_native(payload);old=reference.run_native(reference.prepare(*inputs))
        from probe_mindie_laser_sparse_attention import compare_candidate_states
        bits=compare_candidate_states(actual,old,payload.layout,torch,reference_candidates_per_block=1,reference_bulk_columns=8,reference_score_chunk_blocks=8)
        records.append(dict(scope='Online C32 for K65537' if online else 'C>512 fallback',shape=[1,1,17,65537,64],native_finalize=payload.layout['native_finalize'],reference_bits=bits))
    return records


def audit_loaded_providers(report,candidate,reference,common):
    paths=set(common.mapped_libraries(Path('/proc/self/maps').read_text()));expected={}
    def collect(x):
        if isinstance(x,dict):
            if 'library_path' in x and 'library_sha256' in x:
                path=str(Path(x['library_path']).resolve());digest=x['library_sha256']
                if path in expected and expected[path]!=digest:raise ValueError('Ambiguous provider library SHA')
                expected[path]=digest
            for v in x.values():collect(v)
        elif isinstance(x,(list,tuple)):
            for v in x:collect(v)
    collect(candidate.evidence);collect(candidate.pv.evidence)
    if reference is not None:collect(reference.evidence);collect(reference.pv.evidence)
    for path,digest in expected.items():
        if path not in paths or common.sha256_file(path)!=digest:raise RuntimeError('Candidate library absent/changed in actual process mappings: '+path)
    report['candidate_mapped_library_sha256']=expected
