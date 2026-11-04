"""Raw Top8 source from the actual immutable E119 kernel, not a new QK oracle."""
import torch

def exact_source(actual,reference,scale):
    # Both public producer witnesses initialize all8 score columns/all32 ID columns.
    if len(actual)!=9 or len(reference)!=9:raise AssertionError('Producer state ABI')
    raw=reference[0].cpu();scores=actual[0].cpu();expected=raw.float()*torch.tensor(scale,dtype=torch.float32)
    if raw.ndim!=2 or raw.shape[1]!=8 or not torch.isfinite(raw[:,:reference[5].shape[-1]]).all():raise AssertionError('Nonfinite raw source')
    if scores.shape!=expected.shape or scores.dtype!=torch.float32 or raw.dtype!=torch.float16 or not torch.equal(scores.contiguous().view(torch.int32),expected.contiguous().view(torch.int32)):raise AssertionError('Scaled score not original raw Top8')
    for index in (1,5):
        if actual[index].shape!=reference[index].shape or not torch.equal(actual[index].cpu(),reference[index].cpu()):raise AssertionError('Raw source ID mismatch')
    return dict(rows_checked=len(raw),score_columns=8,padding_columns=32,score_bit_errors=0,ID_errors=0,scope='Original unchanged E119 binary withtemperature1; FP32scaled score andrawTop8ID exact; no dense/video equivalence')

def validate_halfweights(actual,raw,scale):
    if raw.ndim!=2 or not torch.isfinite(raw).all():raise AssertionError('Nonfinite centered source')
    raw=raw.float();delta=(raw-raw[:,:1])*torch.tensor(scale,dtype=torch.float32);want=delta.double().softmax(-1).half();a=actual.cpu()
    lo=torch.nextafter(want,torch.full_like(want,-float('inf')));hi=torch.nextafter(want,torch.full_like(want,float('inf')))
    if a.shape!=want.shape or a.dtype!=torch.float16 or not torch.isfinite(a).all() or (a<0).any() or ((a<lo)|(a>hi)).any() or (a.float().sum(-1)-1).abs().max()>1e-3:raise AssertionError('Centered FP32 softmax/halfweight mismatch')
    return dict(rows_checked=len(raw),boundary_differences=int((a!=want).sum()),max_half_neighbor_steps=1)

def supplement(adapter,payload,state):
    from mindie_laser_sparse_candidates import validate_producer_counters
    g=payload.layout;p=payload.base
    reference=adapter.raw_reference(*p.tensors[:3],payload.positions,g['queries'],g['keys'],p.shape[3],adapter.raw_library)
    rg=dict(g,producer_counter_version=8);rc=validate_producer_counters(reference,rg)
    raw=reference[0].cpu();proof=exact_source(state[1],reference,p.scale);weights=validate_halfweights(state[2][1],raw[:,:g['rank']],p.scale)
    actual=state[0].cpu();want=state[3][0].cpu().reshape(*p.shape[:2],g['qwork'],128)[:,:,:g['queries'],:p.shape[3]].to(actual.dtype).contiguous()
    if not torch.isfinite(actual).all() or not torch.equal(actual.contiguous().view(torch.uint8),want.contiguous().view(torch.uint8)):raise AssertionError('Public trim/cast bytes')
    adapter.latest_raw_reference=(raw,reference[5].cpu())
    return dict(raw_source=proof,raw_reference_counters=rc,centered_halfweights=weights,public_cast_byte_differences=0)

def analytic_source(q,k,g,producer):
    """Basis q has one8, K small integers: dot and half conversion are exact."""
    b,h,rows,d=q.shape;raw=(q.float()@k.float().transpose(-1,-2)).half().float();flat=raw.reshape(b*h,rows,g['keys']);pad=torch.nn.functional.pad(flat,(0,g['kpad']-g['keys']),value=-float('inf'));block=pad.reshape(b*h,rows,g['kpad']//128,128).max(-1).values
    scale=torch.tensor(1/(d**.5),dtype=torch.float32);wanted=block.sort(-1,descending=True).values[:,:,:g['rank']]*scale;actual=producer[0].cpu().reshape(b*h,g['qwork'],8)[:,:rows,:g['rank']]
    if actual.dtype!=torch.float32 or not torch.equal(actual,wanted):raise AssertionError('Analytic rawTop8 FP32scaled values')
    ids=producer[1].cpu()[:,:g['rank'],:rows].transpose(1,2).long()-torch.arange(b*h)[:,None,None]*g['kpad']
    if ((ids<0)|(ids>=g['keys'])).any() or not torch.equal(actual,(flat*scale).gather(-1,ids)):raise AssertionError('Analytic basis ID/score source')
    return dict(rows_checked=b*h*rows,raw_Top8_scaled_value_errors=0,value_ID_errors=0)

def basis(adapter,torch,device):
    from probe_mindie_laser_sparse_attention import check_pipeline
    records=[]
    for b,h,rows,keys,d in ((1,1,1,1,64),(1,1,129,385,64),(2,3,17,257,128),(1,3,385,77,128)):
        q=torch.zeros(b,h,rows,d,dtype=torch.bfloat16);q[...,0]=8;k=((torch.arange(b*h*keys*d).reshape(b,h,keys,d)%127)-63).bfloat16();v=(k.float()/8).bfloat16();payload=adapter.prepare(q.to(device),k.to(device),v.to(device));state=adapter.run_native(payload)
        entry=check_pipeline(adapter,payload,state,torch);entry.update(analytic_source(q,k,payload.layout,state[1]));entry['shape']=[b,h,rows,keys,d];records.append(entry)
    return records
