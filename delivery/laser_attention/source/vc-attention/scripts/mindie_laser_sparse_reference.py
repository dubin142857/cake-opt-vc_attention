"""CPU semantic probes for row-adaptive sparse PV after complete QK.

These are approximate algorithms, not a CANN arithmetic emulator. Selecting
keys needs the complete score row; its cost is not included in PV coverage.
"""
from __future__ import annotations
import math
import torch


def _error(actual,reference):
    delta=(actual-reference).double()
    rms=float(reference.double().square().mean().sqrt())
    error=float(delta.square().mean().sqrt())
    return dict(relative_rmse=error/rms if rms else (0. if error==0 else None),
                rmse=error,reference_rms=rms,max_abs=float(delta.abs().max()))


@torch.no_grad()
def support_probe(q,k,v,query_ids,*,logit_mode='fp32',topks=(1,2,4,8,16,32,64),masses=(.999,.9999)):
    """FP32/FP16 logits, FP64 semantic softmax/PV; not a device emulator."""
    if any(x.device.type!='cpu' or x.ndim!=4 for x in (q,k,v)):
        raise ValueError('CPU BHSD inputs required')
    if q.shape[:2]!=k.shape[:2] or k.shape!=v.shape or q.shape[-1]!=k.shape[-1] or not q.numel() or not k.numel():
        raise ValueError('Invalid MHA shapes')
    if not all(torch.isfinite(x).all() for x in (q,k,v)):
        raise ValueError('Nonfinite input')
    if query_ids.ndim!=1 or not query_ids.numel() or query_ids.dtype!=torch.int64 or (query_ids<0).any() or (query_ids>=q.shape[2]).any():
        raise ValueError('Invalid query sample')
    if any(type(x)!=int or x<1 for x in topks) or not topks:
        raise ValueError('Positive integer ranks required')
    if any(not math.isfinite(x) or not 0<x<1 for x in masses):
        raise ValueError('Mass targets must lie strictly between zero and one')
    dots=q.index_select(2,query_ids).float()@k.float().transpose(-1,-2)
    scale=q.shape[-1]**-.5
    if logit_mode=='fp32':logits=dots*scale
    elif logit_mode=='fp16_qk_scale':
        logits=(dots.half().float()*torch.tensor(scale,dtype=torch.float16).float()).half().float()
    else:raise ValueError('Unknown logit mode')
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite score cast')
    # FP64 normalization makes the CDF rank well-defined even in long equal rows.
    weights=(logits.double()-logits.double().amax(-1,keepdim=True)).exp()
    weights/=weights.sum(-1,keepdim=True)
    ranked,indices=torch.sort(logits,dim=-1,descending=True,stable=True)
    sorted_weights=weights.gather(-1,indices)
    cdf=sorted_weights.cumsum(-1)
    dense=weights@v.double()
    report={'logit_mode':logit_mode,'sampled_rows':logits.numel()//logits.shape[-1],
            'query_ids':query_ids.tolist(),'keys':k.shape[2],'topk':{},'adaptive':{}}
    raw={'logits':logits,'query_ids':query_ids,'dense_output':dense,'topk_outputs':{},'adaptive_ranks':{}}
    for rank in dict.fromkeys(min(x,k.shape[2]) for x in topks):
        ids=indices[...,:rank];w=sorted_weights[...,:rank]
        retained=w.sum(-1)
        selected_v=v.double().unsqueeze(2).expand(-1,-1,len(query_ids),-1,-1).gather(3,ids[...,None].expand(*ids.shape,v.shape[-1]))
        output=((w/retained[...,None])[...,None]*selected_v).sum(-2)
        report['topk'][str(rank)]={'retained_mass_mean':float(retained.mean()),
            'retained_mass_min':float(retained.min()),'key_fraction':rank/k.shape[2],**_error(output,dense)}
        raw['topk_outputs'][str(rank)]=output
    max_rank=min(max(topks),k.shape[2]);raw['top_indices']=indices[...,:max_rank].to(torch.int32)
    for mass in masses:
        required=(cdf<mass).sum(-1)+1;required.clamp_(max=k.shape[2])
        mask=torch.arange(k.shape[2])<required[...,None]
        chosen=sorted_weights*mask
        retained=chosen.sum(-1,keepdim=True)
        sparse=torch.zeros_like(weights).scatter_(-1,indices,chosen/retained)
        output=sparse@v.double()
        key=str(mass);flat=required.double().flatten()
        report['adaptive'][key]={'rank_min':int(flat.min()),'rank_max':int(flat.max()),
            'rank_mean':float(flat.mean()),'rank_p50':float(torch.quantile(flat,.5)),
            'rank_p99':float(torch.quantile(flat,.99)),
            'rank_fraction_mean':float(flat.mean())/k.shape[2],
            'retained_mass_min':float(retained.min()),**_error(output,dense)}
        raw['adaptive_ranks'][key]=required
    return report,raw


def refined_pool_selection(scores,indices,*,rank=8,refine=0):
    """Stable CPU top8 from block maxima, optionally adding runner-up candidates.

    scores/indices are row-major, adjacent columns are a block's first/second.
    Only retained candidate information is available; this is not a dense oracle.
    """
    if scores.device.type!='cpu' or indices.device.type!='cpu' or scores.ndim!=2 or scores.shape!=indices.shape or not scores.numel() or scores.shape[1]%2:
        raise ValueError('Matching nonempty CPU row/pair matrices required')
    if indices.dtype not in (torch.int32,torch.int64) or not scores.is_floating_point():raise ValueError('Floating scores and integer indices required')
    if not torch.isfinite(scores[:,0::2]).any(-1).all() or torch.isnan(scores).any() or torch.isposinf(scores).any() or (scores[:,0::2]<scores[:,1::2]).any():raise ValueError('Finite first candidates and ordered pairs required')
    if type(rank)!=int or not 1<=rank<=16 or type(refine)!=int or not 0<=refine<=rank:raise ValueError('Rank1..16 and refinement0..rank required')
    coarse=torch.argsort(scores[:,0::2],dim=-1,descending=True,stable=True)[:,:min(rank,scores.shape[1]//2)]
    columns=torch.cat((coarse*2,coarse[:,:min(refine,coarse.shape[1])]*2+1),dim=-1).sort(-1).values
    order=torch.argsort(scores.gather(-1,columns),dim=-1,descending=True,stable=True)[:,:min(rank,columns.shape[-1])]
    columns=columns.gather(-1,order)
    return scores.gather(-1,columns),indices.gather(-1,columns),columns,coarse

def _pool_pv(value,indices,weights,chunk=256):
    outputs=[]
    for start in range(0,len(indices),chunk):
        i=indices[start:start+chunk].long();w=weights[start:start+chunk].double()
        outputs.append((value[i].double()*w[...,None]).sum(1))
    return torch.cat(outputs)

def _row_error(actual,reference,queries,worst=16):
    mse=(actual-reference).double().square().mean(-1);den=reference.double().square().mean(-1)
    rel=torch.where(den>0,(mse/den).sqrt(),torch.where(mse==0,0.,float('nan')))
    valid=torch.isfinite(rel);finite=rel[valid]
    order=torch.argsort(torch.where(valid,rel,float('inf')),descending=True,stable=True)[:worst]
    row=dict(undefined_relative_rows=int((~valid).sum()),
        p50=float(torch.quantile(finite,.5)) if finite.numel() else None,
        p99=float(torch.quantile(finite,.99)) if finite.numel() else None,
        p999=float(torch.quantile(finite,.999)) if finite.numel() else None,
        max=float(finite.max()) if finite.numel() else None,
        worst=[dict(head=int(i)//queries,query=int(i)%queries,relative=float(rel[i]) if valid[i] else None,rmse=float(mse[i].sqrt())) for i in order])
    return row,rel

@torch.no_grad()
def candidate_pool_probe(candidate_values,candidate_indices,value,*,queries,keys,kpad,rank=8,refinements=(0,1,2,4,7,8),native=None):
    """All real rows of an initialized two-candidate-per-K128 pool, CPU only.

    The oracle softmax/PV is FP64. Native fidelity uses its stored half weights.
    No grouping cost or device latency is predicted from the logical counts.
    """
    if any(t.device.type!='cpu' for t in (candidate_values,candidate_indices,value)):raise ValueError('CPU tensors required')
    if candidate_values.ndim!=3 or candidate_values.shape!=candidate_indices.shape or value.ndim!=2 or not value.numel():raise ValueError('H,C,Qwork candidates and flattened K/V required')
    if not candidate_values.is_floating_point() or candidate_indices.dtype not in (torch.int32,torch.int64) or not value.is_floating_point():raise ValueError('Floating scores/V and integer candidate IDs required')
    heads,columns,qwork=candidate_values.shape
    if any(type(x)!=int or x<=0 for x in (queries,keys,kpad,rank)) or queries>qwork or keys>kpad or kpad%128 or columns!=2*(kpad//128) or value.shape[0]!=heads*kpad:raise ValueError('Candidate geometry mismatch')
    if rank>16 or not refinements or len(set(refinements))!=len(refinements) or any(type(t)!=int or not 0<=t<=rank for t in refinements):raise ValueError('Invalid ranks/refinements')
    if not torch.isfinite(value).all():raise ValueError('Nonfinite V')
    scores=candidate_values.transpose(1,2)[:,:queries].reshape(-1,columns)
    ids=candidate_indices.transpose(1,2)[:,:queries].reshape(-1,columns).long()
    rows=len(scores);blocks=columns//2
    finite=torch.isfinite(scores);expected=(torch.arange(columns)//2*128+torch.arange(columns)%2)<keys
    if torch.isnan(scores).any() or torch.isposinf(scores).any() or not (finite==expected).all() or (scores[:,0::2]<scores[:,1::2]).any():raise ValueError('Incomplete/nonfinite/unordered candidate pool')
    h=torch.arange(heads).repeat_interleave(queries)[:,None]
    local=ids-h*kpad;b=torch.arange(columns)//2
    if ((local<0)|(local>=keys)|(local//128!=b))[finite].any():raise ValueError('Candidate index crosses head/block or valid keys')
    both=finite[:,1::2]
    if (ids[:,0::2]==ids[:,1::2])[both].any():raise ValueError('Repeated finite block candidates')
    order=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:min(rank,columns)]
    ref_scores=scores.gather(-1,order);ref_ids=ids.gather(-1,order);ref_w=ref_scores.double().softmax(-1)
    safe_ref=torch.where(torch.isfinite(ref_scores),ref_ids,0)
    reference=_pool_pv(value,safe_ref,ref_w)
    report=dict(scope='CPU block2-pool stable top8 oracle, not full QK/dense or CANN timing',heads=heads,queries=queries,real_rows=rows,keys=keys,kpad=kpad,qwork=qwork,rank=rank,variants={})
    raw=dict(reference_output=reference,reference_columns=order,reference_indices=ref_ids,reference_scores=ref_scores,reference_weights=ref_w,variants={})
    if native is not None:
        ni,nw,no=native
        if ni.dtype not in (torch.int32,torch.int64) or nw.dtype!=torch.float16 or no.dtype!=torch.float32:raise ValueError('Native integer IDs/half weights/FP32 output required')
        if ni.shape!=nw.shape or ni.shape[0]!=heads*qwork or no.shape!=(heads*qwork,value.shape[-1]):raise ValueError('Native artifact shape mismatch')
        def real(x):return x.reshape(heads,qwork,*x.shape[1:])[:,:queries].reshape(rows,*x.shape[1:])
        ni,nw,no=real(ni).long(),real(nw),real(no)
        if ((ni<0)|(ni>=heads*kpad)|(ni//kpad!=h)|(ni%kpad>=keys)).any() or not torch.isfinite(nw).all():raise ValueError('Invalid stored native selection')
        rebuilt=_pool_pv(value,ni,nw);err=_error(rebuilt,no)
        if err['relative_rmse'] is None or err['relative_rmse']>1e-5 or err['max_abs']>1e-4:raise AssertionError('Stored native PV fidelity failed: '+str(err))
        report['native_fidelity']=err;report['stable_oracle_vs_native']=_error(reference,no)
    for refine in refinements:
        selected,key_ids,chosen,coarse=refined_pool_selection(scores,ids,rank=rank,refine=refine)
        weights=selected.double().softmax(-1);safe=torch.where(torch.isfinite(selected),key_ids,0)
        output=_pool_pv(value,safe,weights)
        chosen_finite=torch.isfinite(selected);ref_finite=torch.isfinite(ref_scores)
        present=((ref_ids[...,None]==key_ids[:,None,:])&chosen_finite[:,None,:]).any(-1)
        missing=ref_finite&~present
        padded=torch.nn.functional.pad(selected,(0,ref_scores.shape[-1]-selected.shape[-1]),value=-math.inf)
        wrong_bits=(padded.contiguous().view(torch.uint8)!=ref_scores.contiguous().view(torch.uint8)).any(-1)
        row_error,rel=_row_error(output,reference,queries)
        picked_blocks=coarse[:,:min(refine,coarse.shape[1])]
        valid_blocks=torch.isfinite(scores[:,0::2].gather(-1,picked_blocks))
        labels=(picked_blocks+h*blocks)[valid_blocks]
        hist=torch.bincount(labels,minlength=heads*blocks).reshape(heads,blocks)
        grouped={str(tile):dict(padded_query_assignments=int(((hist+tile-1)//tile*tile).sum()),cube_tasks=int(((hist+tile-1)//tile).sum())) for tile in (16,128)}
        cells=dict(assignments=int(hist.sum()),active_head_blocks=int((hist>0).sum()),max_queries_per_head_block=int(hist.max()),
            logical_qk_pairs=int(hist.sum())*128,padded_M=grouped,full_qk_pairs=rows*keys,additional_qk_ratio=int(hist.sum())*128/(rows*keys),
            note='Counts only; initial full QK remains. Label/prefix/reorder/refinement/merge costs are not measured.')
        entry=dict(refine_blocks=refine,relative_to_stable_block2=_error(output,reference),row_error=row_error,
            rows_missing_any_top8=int(missing.any(-1).sum()),rows_missing_top1=int(missing[:,:1].any(-1).sum()),rows_missing_top2=int(missing[:,:min(2,missing.shape[-1])].any(-1).sum()),
            retained_top8_mass_mean=float((ref_w*present).sum(-1).mean()),retained_top8_mass_min=float((ref_w*present).sum(-1).min()),
            rows_with_different_score_storage=int(wrong_bits.sum()),
            guaranteed_prefix=min(refine+1,rank),work=cells)
        # T>=rank-1 must recover stable top-rank columns, not just equal scores.
        if refine>=rank-1 and not torch.equal(chosen,order):raise AssertionError('Refinement prefix theorem violated')
        if native is not None:
            entry['relative_to_native']=_error(output,no)
            native_present=((ni[...,None]==key_ids[:,None,:])&chosen_finite[:,None,:]).any(-1)
            entry['rows_missing_native_top1']=int((~native_present[:,:1]).any(-1).sum())
            entry['rows_missing_native_top2']=int((~native_present[:,:min(2,ni.shape[-1])]).any(-1).sum())
        report['variants'][str(refine)]=entry
        raw['variants'][str(refine)]=dict(output=output,indices=key_ids,columns=chosen,scores=selected,weights=weights,relative_per_row=rel,bucket_counts=hist)
    return report,raw
