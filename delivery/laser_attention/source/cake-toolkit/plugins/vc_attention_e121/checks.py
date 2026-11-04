"""Whole-pipeline source/primitive replay for the explicitly changed algorithm."""
import math,time
import torch
def _error(actual,reference):
    from mindie_laser_sparse_reference import _error as measure
    return measure(actual,reference)

def zero_errors(result):
    if any(v for v in result.values()):raise AssertionError('Nonzero semantic errors: '+str(result))

def dot_error(actual,q,k):
    if q.shape!=k.shape or q.shape[-1]!=128 or actual.shape!=q.shape[:-1] or not all(torch.isfinite(x).all() for x in (actual,q,k)):raise AssertionError('Nonfinite or malformed point source/output')
    expected=(q.double()*k.double()).sum(-1);mass=(q.double()*k.double()).abs().sum(-1);error=(actual.double()-expected).abs()
    # FP32 half-factor products are exact; an at-most128-add tree has this
    # absolute forward error, including ill-conditioned cancellation.
    u=2.**-24;gamma=128*u/(1-128*u);bound=gamma*mass+2.**-44
    if (error>bound).any():raise AssertionError('Original dot exceeds FP32 absolute forward-error bound')
    return dict(max_abs=float(error.max()),max_bound=float(bound.max()),bound_violations=0,frozen_relative_prediction_exceeded=int((error>2e-5+2e-6*expected.abs()).sum()))

def primitives(adapter,device,directory):
    from pathlib import Path
    import json
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    gen=torch.Generator().manual_seed(121);deq={x:0 for x in ('f32_saturating','f64_saturating')};qk_rows=[]
    for pattern in range(3):
        q=torch.randint(-127,128,(128,128),generator=gen,dtype=torch.int32).to(torch.int8);k=torch.randint(-127,128,(128,128),generator=gen,dtype=torch.int32).to(torch.int8)
        if pattern==1:q.zero_();q[:,0]=torch.arange(128).to(torch.int8);k.zero_();k[:,0]=torch.arange(128).flip(0).to(torch.int8)
        if pattern==2:q[:,1::2]=-q[:,::2];k[:,1::2]=k[:,::2]
        dots=q.float()@k.float().T
        for gain in (1.,2.**-12,.0234567):
            f=torch.tensor(gain,dtype=torch.float32);f=(f.view(torch.int32)&-8192).view(torch.float32)
            runs=[adapter.qk_tile(q.to(device),k.to(device),gain).cpu() for _ in range(2)]
            if not torch.equal(runs[0].view(torch.int16),runs[1].view(torch.int16)):raise AssertionError('QKtile nondeterminism')
            torch.save(dict(q=q,k=k,gain=gain,actual=runs,expected_f32=(dots*f).half(),expected_f64=(dots.double()*f.double()).half()),directory/f'qk-{pattern}-{gain}.pt')
            counts={}
            for mode in deq:
                expected=((dots*f) if mode=='f32_saturating' else (dots.double()*f.double())).clamp(-65504,65504).half()
                counts[mode]=int((runs[0]!=expected).sum());deq[mode]+=counts[mode]
            qk_rows.append(dict(pattern=pattern,gain=gain,errors=counts))
    possible=[x for x,n in deq.items() if n==0]
    adapter.deq_mode='f32_saturating' if 'f32_saturating' in possible else (possible[0] if possible else None)
    records=[]
    for pattern in range(10):
        q=torch.randn(64,128,generator=gen).half();k=torch.randn(64,128,generator=gen).half()
        if pattern==0:q.zero_();k.zero_()
        elif pattern==1:q.fill_(1);k.copy_(torch.arange(128).expand(64,128).half())
        elif pattern==2:q.mul_(5);k.mul_(5)
        elif pattern==3:q[:,1::2]=-q[:,::2];k[:,1::2]=k[:,::2]
        elif pattern==4:q.zero_();k.zero_();q[:,:3]=torch.tensor([4096.,1.,4096.]);k[:,:3]=torch.tensor([4096.,1.,-4096.])
        elif pattern==5:q.fill_(2.**-24);k.fill_(1)
        elif pattern==6:q.fill_(65504);k.fill_(.125)
        elif pattern==7:q.zero_();k.zero_();q[:,127]=torch.arange(64).half();k[:,127]=1
        elif pattern==8:q[:,64:]=0;k[:,:64]=0
        elif pattern==9:q[:,::2]=0;k[:,1::2]=0
        runs=[adapter.point_probe(q.to(device),k.to(device)).cpu() for _ in range(2)]
        if not torch.equal(runs[0].view(torch.int32),runs[1].view(torch.int32)):raise AssertionError('Pointprobe nondeterminism')
        torch.save(dict(q=q,k=k,actual=runs),directory/f'point-{pattern}.pt')
        records.append(dict(pattern=pattern,**dot_error(runs[0],q,k)))
    result=dict(qk_cases=qk_rows,qk_model_errors=deq,selected_deq_model=adapter.deq_mode,point_cases=records,repeats=2,passed=bool(possible))
    (directory/'detail.json').write_text(json.dumps(result,indent=2)+'\n')
    if not possible:raise AssertionError('No declared CPU DEQ model matchesactualtile; raw primitives saved')
    return result

def counters(producer,pv,g):
    c=producer[2].cpu();tasks=g['heads']*(g['qwork']//128);cores=min(24,tasks);blocks=g['kpad']//128;groups=(blocks+7)//8;windows=(blocks+31)//32
    if c.shape!=(3*cores,64) or c.dtype!=torch.int64 or (c<0).any():raise AssertionError('Counterlayout')
    for core in range(cores):
        n=len(range(core,tasks,cores));want=[n,n*blocks,n*max(groups-2,0),n*groups,n*min(2,groups)]
        if c[core,:5].tolist()!=want or c[core,5:].any():raise AssertionError('AICprotocol coverage')
        want=[n,n*blocks,n*groups,n*g['keys'],n*windows,6*n,512*n,1536*n,2*windows*n,n*blocks*64,n*1024,n*1024,n*64*(16-g['coarse_rank']),n*1028]
        for sub in range(2):
            actual=c[cores+2*core+sub]
            if actual[:14].tolist()!=want or actual[14:].any():raise AssertionError('AIVcoarse/refinecounter mismatch '+str((core,sub,actual[:14].tolist(),want)))
    from mindie_laser_candidate_inline_pv import validate_inline_pv_counters
    return dict(query_tiles=tasks,qk_blocks=tasks*blocks,coarse_points=tasks*2*1024,coarse_rank=g['coarse_rank'],point_invalid=int(c[cores:,12].sum()),pv=validate_inline_pv_counters(pv,g))

def deq_gain(qs,ks):return ((qs[:,None]*ks[None,:]).contiguous().view(torch.int32)&-8192).view(torch.float32)

def coarse_check(adapter,payload,state):
    g=payload.layout;h=g['heads'];qw=g['qwork'];kp=g['kpad'];cr=g['coarse_rank'];d=payload.base.shape[3];nkb=kp//128
    qi,ki,qs,ks=(x.cpu() for x in payload.quantized);qi=qi.reshape(h,-1,128);ki=ki.reshape(h,kp,128);qs=qs.reshape(h,-1);ks=ks.reshape(h,-1)
    ids=state[1][9].cpu().transpose(1,2).long();scores=state[1][10].cpu().transpose(1,2);total=0;then=time.monotonic()
    if ids.shape!=(h,qw,16) or scores.shape!=ids.shape or scores.dtype!=torch.float32:raise AssertionError('Coarsewitnesslayout')
    if not torch.isfinite(scores[:,:,:cr]).all() or not torch.isneginf(scores[:,:,cr:]).all():raise AssertionError('Coarse finite/padded contract')
    if adapter.deq_mode not in ('f32_saturating','f64_saturating'):raise AssertionError('DEQ model notprobed')
    for head in range(h):
        kval=ki[head].float()
        for start in range(0,qw,256):
            end=min(start+256,qw);dots=(qi[head,start:end].float()@kval.T).reshape(end-start,nkb,128)
            gain=deq_gain(qs[head,torch.arange(start,end)//128],ks[head])
            raw=((dots*gain[:,:,None]) if adapter.deq_mode=='f32_saturating' else (dots.double()*gain.double()[:,:,None])).clamp(-65504,65504).half()
            raw=raw.flatten(1);raw[:,g['keys']:]=-float('inf');raw=raw.reshape(end-start,nkb,128)
            maximum=raw.max(-1).values;values=(maximum.float()*float(torch.tensor(d**-.5,dtype=torch.float16))).half().float()
            wanted=values.sort(-1,descending=True).values[:,:cr];actual=scores[head,start:end,:cr]
            if not torch.equal(actual,wanted):raise AssertionError('Coarse globalTop16 value mismatch '+str((head,start,int((actual!=wanted).sum()))))
            local=ids[head,start:end,:cr]-head*kp
            if ((local<0)|(local>=g['keys'])).any():raise AssertionError('Coarse IDs crosshead/tail')
            block=local//128
            if cr>1 and (block.sort(-1).values.diff(dim=-1)==0).any():raise AssertionError('Duplicatecoarseblock')
            ownraw=raw.flatten(1).gather(1,local)
            if not torch.equal(ownraw,maximum.gather(1,block)) or not torch.equal(actual,values.gather(1,block)):raise AssertionError('Coarse value/ID not originalblockmax')
            total+=end-start
    return dict(rows_checked=total,keys_checked=g['keys'],rank=cr,global_value_errors=0,illegal_source_pairs=0,deq_model=adapter.deq_mode,cpu_wall_seconds=time.monotonic()-then)

def fine_check(adapter,payload,state,device):
    g=payload.layout;h=g['heads'];qw=g['qwork'];cr=g['coarse_rank'];rank=g['rank'];qp=payload.base.tensors[0].shape[2];kp=g['kpad'];dim=payload.base.shape[3]
    q=payload.base.tensors[0].cpu().reshape(h,qp,128)[:,:qw].reshape(-1,128);k=payload.base.tensors[1].cpu().reshape(-1,128)
    source_ids=state[1][9].cpu().transpose(1,2).reshape(-1,16)[:,:cr].long();fine_ids=state[2][0].cpu().long();fine_scores=state[2][2].cpu()
    if fine_ids.shape!=(h*qw,rank) or fine_scores.shape!=fine_ids.shape or fine_scores.dtype!=torch.float32:raise AssertionError('FineScore/IDshape')
    dotmax=0.;predmiss=0;points=0;replayed=[]
    for start in range(0,len(q),256):
        end=min(start+256,len(q));ci=source_ids[start:end];qr=q[start:end,None,:].expand(-1,cr,-1).reshape(-1,128).contiguous();kr=k[ci].reshape(-1,128).contiguous()
        raw=adapter.point_probe(qr.to(device),kr.to(device)).cpu();audit=dot_error(raw,qr,kr);dotmax=max(dotmax,audit['max_abs']);predmiss+=audit['frozen_relative_prediction_exceeded']
        expected=(raw.reshape(end-start,cr)*torch.tensor(dim**-.5,dtype=torch.float32));wanted=expected.sort(-1,descending=True).values[:,:rank]
        if not torch.equal(fine_scores[start:end],wanted):raise AssertionError('FineTop8 not independentpointprimitivevalues')
        matches=fine_ids[start:end,:,None]==ci[:,None,:]
        if not matches.any(-1).all():raise AssertionError('FineIDoutsidecoarsepool')
        selected=expected.gather(1,matches.to(torch.int64).argmax(-1))
        if not torch.equal(selected,fine_scores[start:end]):raise AssertionError('FineScore/ID mismatch independentpoint')
        points+=raw.numel();replayed.append(expected)
    adapter.latest_point_scores=torch.cat(replayed)
    weights=state[2][1].cpu();want=fine_scores.double().softmax(-1).half();lo=torch.nextafter(want,torch.full_like(want,-float('inf')));hi=torch.nextafter(want,torch.full_like(want,float('inf')))
    if not torch.isfinite(weights).all() or (weights<0).any() or ((weights<lo)|(weights>hi)).any() or (weights.float().sum(-1)-1).abs().max()>1e-3:raise AssertionError('FP32softmax/halfweightcontract')
    return dict(points_checked=points,score_value_errors=0,illegal_pool_IDs=0,dot_CPU_FP64_max_abs=dotmax,frozen_relative_prediction_exceeded=predmiss,halfweight_boundary_differences=int((weights!=want).sum()))

def quantization_check(original,quantized,value,scale):
    from mindie_laser_quant_backend import QuantLaserPayload
    from mindie_laser_quant_preprocessing import check_preprocessing
    qi,ki,qs,ks=quantized;b,h=original[0].shape[:2]
    actual=QuantLaserPayload(qi,ki,value,qs,ks,torch.ones(b,h,128),original[0].dtype,tuple(original[0].shape),original[1].shape[2],scale,False)
    return check_preprocessing(original,actual)

def pipeline(adapter,payload,state,device):
    g=payload.layout;out,producer,selected,pv,packet=state;ids,weights,scores=selected
    counts=counters(producer,pv,g);ids=ids.cpu().long();weights=weights.cpu();head=torch.arange(g['heads']).repeat_interleave(g['qwork'])[:,None]
    if (ids<0).any() or (ids>=g['heads']*g['kpad']).any() or not (ids//g['kpad']==head).all() or (ids%g['kpad']>=g['keys']).any():raise AssertionError('SelectedIDshead/tail')
    if g['rank']>1 and (ids.sort(-1).values.diff(dim=-1)==0).any():raise AssertionError('Repeatedselectedkey')
    values=payload.base.tensors[2].cpu().reshape(-1,128);reference=(values[ids].double()*weights.double()[:,:,None]).sum(1);err=_error(pv[0].cpu(),reference)
    if err['relative_rmse'] is None or err['relative_rmse']>1e-5 or err['max_abs']>1e-4:raise AssertionError('OriginalownPVgate '+str(err))
    unpack=packet.indices.cpu().transpose(1,2).reshape(-1,16)[:len(ids),:g['rank']].long();unpackw=packet.weights.cpu().transpose(1,2).reshape(-1,16)[:len(ids),:g['rank']]
    if not torch.equal(unpack,ids) or not torch.equal(unpackw.view(torch.int16),weights.view(torch.int16)):raise AssertionError('Packetroundtrip')
    want=pv[0].cpu().reshape(*payload.base.shape[:2],g['qwork'],128)[:,:,:g['queries'],:payload.base.shape[3]].to(out.dtype).contiguous()
    if not torch.isfinite(out).all() or not torch.equal(out.cpu().view(torch.uint8),want.view(torch.uint8)):raise AssertionError('Publiccastnonfinite/bytes')
    quantcheck=quantization_check(tuple(x.cpu() for x in payload.original),tuple(x.cpu() for x in payload.quantized),payload.base.tensors[2].cpu(),payload.base.scale)
    if not quantcheck['passed']:raise AssertionError('Actualquantpayload '+str(quantcheck['errors']))
    coarse=coarse_check(adapter,payload,state);fine=fine_check(adapter,payload,state,device)
    return dict(producer_counters=counts,pv_counters=counts['pv'],fp32_pv_error=err,packet_bit_differences=0,public_cast_byte_differences=0,quantization=quantcheck,coarse_source=coarse,fine_source=fine,implementation_contract='int8coarse16/originalFP32point/Top8/ownPV; no oldE119bitwiseequivalence claim')

def components(adapter,torch,device,directory):
    from pathlib import Path
    import json
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[];cases=((1,1,1,1,64,0),(1,1,17,257,64,1),(1,1,128,385,128,5),(2,3,17,769,128,5),(1,3,129,1153,64,5),(1,1,129,65537,64,5),(1,1,17,1025,128,0),(1,1,257,129,64,5),(1,1,17,385,128,1),(1,1,1,257,128,5),(1,1,64,641,128,5),(1,1,65,897,64,5),(1,2,128,1153,128,5))
    gen=torch.Generator().manual_seed(12100)
    for index,(b,h,q,k,d,std) in enumerate(cases):
        cpu=[(torch.randn(b,h,n,d,generator=gen)*scale).bfloat16() for n,scale in ((q,std),(k,std),(k,5))];payload=adapter.prepare(*(x.to(device) for x in cpu));state=adapter.run_native(payload)
        try:entry=pipeline(adapter,payload,state,device)
        except Exception as exc:
            torch.save(dict(case_index=index,inputs=cpu,geometry=payload.layout,base_tensors=tuple(x.cpu() for x in payload.base.tensors[:3]),quantized=tuple(x.cpu() for x in payload.quantized),positions=payload.positions.cpu(),output=state[0].cpu(),producer=tuple(x.cpu() for x in state[1]),selected=tuple(x.cpu() for x in state[2]),error=str(exc)),directory/'failed.pt')
            (directory/'partial.json').write_text(json.dumps(dict(completed=records,failed_case=index,error=str(exc)),indent=2)+'\n');raise
        other=adapter.run_native(payload)
        for a,buff in zip(state[1],other[1]):
            if not torch.equal(a.cpu().contiguous().view(torch.uint8),buff.cpu().contiguous().view(torch.uint8)):raise AssertionError('Nondeterministic initialized nativefields')
        entry.update(shape=[b,h,q,k,d],std=std,initialized_state_repeat_byte_errors=0);records.append(entry)
        (directory/'partial.json').write_text(json.dumps(dict(completed=records),indent=2)+'\n')
    return records

def basis(adapter,torch,device):
    records=[]
    for b,h,q,k,d in ((1,1,1,1,64),(1,1,129,385,64),(2,3,17,257,128),(1,3,385,77,128)):
        x=torch.zeros(b,h,q,d,dtype=torch.bfloat16);x[...,0]=8;key=((torch.arange(b*h*k*d).reshape(b,h,k,d)%127)-63).bfloat16();v=(key.float()/8).bfloat16();payload=adapter.prepare(x.to(device),key.to(device),v.to(device));state=adapter.run_native(payload);records.append(pipeline(adapter,payload,state,device))
    return records
