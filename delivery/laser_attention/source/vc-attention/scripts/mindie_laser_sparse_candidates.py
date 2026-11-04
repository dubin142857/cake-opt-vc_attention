"""Block-two candidate approximation; full QK is still required."""
from __future__ import annotations
import math

def geometry(heads,queries,keys,rank=8):
    if any(type(x)!=int or x<1 for x in (heads,queries,keys,rank)) or rank>16:
        raise ValueError('Positive extents and rank1..16 required')
    qpad=(queries+255)//256*256;kpad=(keys+255)//256*256
    qwork=(queries+127)//128*128;columns=((kpad//128*2+31)//32)*32
    if heads*kpad>2**31-1 or heads*columns*qwork>2**31-1:
        raise ValueError('Candidate int32 indexing extent exceeded')
    finite_candidates=2*(keys//128)+min(keys%128,2)
    return dict(heads=heads,queries=queries,keys=keys,qpad=qpad,kpad=kpad,
                qwork=qwork,columns=columns,rank=min(rank,finite_candidates))

def reference_candidates(logits,rank=8,*,candidates_per_block=2):
    import torch
    if logits.ndim<2 or not logits.shape[-1] or not torch.isfinite(logits).all():
        raise ValueError('Finite score rows required')
    if type(rank)!=int or not 1<=rank<=16:raise ValueError('Rank1..16 required')
    if type(candidates_per_block)!=int or candidates_per_block not in (1,2):raise ValueError("One or two candidates per block required")
    keys=logits.shape[-1];blocks=(keys+127)//128
    padded=torch.nn.functional.pad(logits,(0,blocks*128-keys),value=-math.inf)
    values,ids=padded.reshape(*logits.shape[:-1],blocks,128).topk(candidates_per_block,dim=-1)
    ids=ids+torch.arange(blocks,device=ids.device)[:,None]*128
    values=values.flatten(-2);ids=ids.flatten(-2)
    finite=candidates_per_block*(keys//128)+min(keys%128,candidates_per_block)
    chosen,pos=values.topk(min(rank,finite),dim=-1)
    return chosen,ids.gather(-1,pos)

def reference_output(logits,value,rank=8,*,candidates_per_block=2):
    import torch
    values,ids=reference_candidates(logits,rank,candidates_per_block=candidates_per_block)
    weights=values.double().softmax(-1)
    flat=ids.reshape(-1,ids.shape[-2],ids.shape[-1]);v=value.reshape(-1,value.shape[-2],value.shape[-1])
    selected=torch.stack([v[h][flat[h]] for h in range(len(v))]).reshape(*ids.shape,value.shape[-1])
    return (selected.double()*weights[...,None]).sum(-2),ids


class CandidatePayload:
    def __init__(self,base,layout,positions,row_offsets):
        self.base,self.layout,self.positions,self.row_offsets=base,layout,positions,row_offsets

class CandidateAttention:
    def __init__(self,base,native,probe,pv,evidence,metadata,device):
        self.base,self.native,self.probe,self.pv=base,native,probe,pv
        self.evidence,self.metadata,self.device=evidence,metadata,device
        self.geometry_cache={}
    def make_layout(self,heads,queries,keys):
        return geometry(heads,queries,keys)
    def make_positions(self,device):
        import torch
        return torch.arange(128,dtype=torch.float16).repeat(64,1).to(device)
    def prepare(self,q,k,v):
        import torch
        from mindie_laser_backend import validate_mha_inputs
        validate_mha_inputs(q,k,v)
        if any(x.device.type!='npu' or x.device.index!=self.device or x.requires_grad for x in (q,k,v)):
            raise ValueError('Configured inference NPU required')
        h=q.shape[0]*q.shape[1];g=self.make_layout(h,q.shape[2],k.shape[2]);key=(h,q.shape[2],k.shape[2],q.shape[3])
        if key not in self.geometry_cache:
            position=self.make_positions(q.device)
            row=torch.arange(h*g['qwork'],dtype=torch.int64)
            offsets=((row//g['qwork'])*g['columns']*g['qwork']+row%g['qwork']).to(torch.int32).unsqueeze(1).to(q.device)
            self.geometry_cache[key]=(position,offsets)
        position,offsets=self.geometry_cache[key]
        return CandidatePayload(self.base.prepare(q,k,v),g,position,offsets)
    def produce(self,payload):
        p,g=payload.base,payload.layout
        return self.native(p.tensors[0],p.tensors[1],payload.positions,g['queries'],g['keys'],p.shape[3])
    def select(self,payload,state):
        values,indices,_=state;g=payload.layout
        rows=values.transpose(1,2).contiguous().view(-1,g['columns'])
        selected,positions=rows.topk(g['rank'],dim=-1,sorted=True)
        offsets=positions.to(__import__('torch').int32)*g['qwork']+payload.row_offsets
        ids=indices.view(-1).index_select(0,offsets.reshape(-1)).view(-1,g['rank'])
        weights=selected.float().softmax(-1).half()
        return ids,weights,selected
    def consume(self,payload,selection):
        p,g=payload.base,payload.layout;ids,weights,_=selection
        packet=self.pv.prepare(p.tensors[2].view(-1,128),ids,weights)
        result=self.pv.run_native(packet)
        output=result[0].view(p.shape[0],p.shape[1],g['qwork'],128)[:,:,:g['queries'],:p.shape[3]].to(p.dtype).contiguous()
        return output,result,packet
    def run_native(self,payload):
        state=self.produce(payload);selected=self.select(payload,state);out,result,packet=self.consume(payload,selected)
        return out,state,selected,result,packet
    def run_prepared(self,payload):return self.run_native(payload)[0]
    def __call__(self,q,k,v):return self.run_prepared(self.prepare(q,k,v))

def validate_producer_counters(state,layout):
    import torch
    count=state[2].cpu();g=layout;tasks=g['heads']*(g['qwork']//128);cores=min(24,tasks);blocks=g['kpad']//128
    bulk=g.get('output_bulk_columns',1);version=g.get('producer_counter_version',1)
    chunk=g.get('score_chunk_blocks',1)
    if (bulk,version,chunk) not in ((1,1,1),(8,2,1),(8,3,8),(8,4,32),(8,5,8),(8,6,8),(8,7,8), (8,8,8), (8,10,8)):raise AssertionError('Producer output/counter contract mismatch')
    groups=(blocks+chunk-1)//chunk
    if count.shape!=(3*cores,64) or count.dtype!=torch.int64 or (count<0).any():raise AssertionError('Producer counter layout')
    for core in range(cores):
        n=len(range(core,tasks,cores));want=[n,n*blocks,n*max(groups-2,0),n*groups,n*min(2,groups)]
        if count[core,:5].tolist()!=want or count[core,5:].any():raise AssertionError('AIC coverage incomplete')
        for sub in range(2):
            v=count[cores+core*2+sub]
            expected=[n,n*blocks,n*groups,n*g['keys']]
            if version in (6,7,8,10):
                windows=(blocks+31)//32;expected += [n*windows,4*n,512*n,1536*n,n*(2*windows-1)]
            elif version==5:expected += [n*groups,4*n,512*n,1536*n]
            elif bulk>1:expected += [n*((blocks+bulk-1)//bulk),2*n*((blocks+bulk-1)//bulk)]
            if version in (8,10):expected += [n*blocks*64]
            if version==10:expected += [n*1536]
            if v[:len(expected)].tolist()!=expected or v[len(expected):].any():raise AssertionError('AIV coverage incomplete')
    result=dict(query_tiles=tasks,qk_blocks=tasks*blocks,candidates=tasks*blocks*g.get('candidates_per_block',2)*128,cores=cores)
    if version in (5,6,7,8,10):result.update(online_merges=int(count[cores:,4].sum()),output_dma_calls=int(count[cores:,5].sum()),score_entries=int(count[cores:,6].sum()),padding_id_entries=int(count[cores:,7].sum()))
    elif bulk>1:result.update(output_bulk_columns=bulk,output_batches=int(count[cores:,4].sum()),output_dma_calls=int(count[cores:,5].sum()))
    if version in (6,7,8,10):result.update(selection_window_blocks=32,sort32_calls=int(count[cores:,8].sum()))
    if chunk>1:result.update(score_chunk_blocks=chunk,score_groups=tasks*groups,score_reuse_waits=int(count[:cores,2].sum()),score_tail_waits=int(count[:cores,4].sum()),score_readfree=int(count[cores:,2].sum()))
    if version==8:result.update(scale_elements=int(count[cores:,9].sum()),scale_elements_per_block=64)
    if version==10:result.update(raw_reduction_elements=int(count[cores:,9].sum()),scale_elements=int(count[cores:,10].sum()),scale_elements_per_vector_tile=1536)
    return result

def load_candidate_attention(base,manifest,pv_manifest,device=5):
    import json,os
    from pathlib import Path
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    from mindie_laser_sparse_pv import load_sparse_pv
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_candidates')
    build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Candidate build/runtime CANN mismatch')
    import torch
    import torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_candidates
    expected=dict(soc=evidence['soc'],operation='block2_candidates',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=51968,score_slots=2,candidates_per_block=2)
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Candidate producer metadata mismatch')
    native=ns.produce.default;probe=ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*3:raise ValueError('Candidate schema mismatch')
    if [str(x.type) for x in probe._schema.arguments]!=['Tensor']*2+['int']*2 or [str(x.type) for x in probe._schema.returns]!=['Tensor']*2:raise ValueError('Probe schema mismatch')
    return CandidateAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)


CANDIDATE_LOADERS={
    'sparse_candidates':('mindie_laser_sparse_candidates','load_candidate_attention'),
    'sparse_candidates_lookup':('mindie_laser_candidate_lookup','load_candidate_lookup_attention'),
    'sparse_candidates_dma_overlap':('mindie_laser_candidate_dma_overlap','load_candidate_dma_attention'),
    'sparse_candidates_block1':('mindie_laser_candidate_block1','load_candidate_block1_attention'),
    'sparse_candidates_read_ack':('mindie_laser_candidate_read_ack','load_candidate_read_ack_attention'),
    'sparse_candidates_bulk_output':('mindie_laser_candidate_bulk_output','load_candidate_bulk_output_attention'),
    'sparse_candidates_score_chunk':('mindie_laser_candidate_score_chunk','load_candidate_score_chunk_attention'),
    'sparse_candidates_score_chunk32':('mindie_laser_candidate_score_chunk32','load_candidate_score_chunk32_attention'),
    'sparse_online_top8':('mindie_laser_candidate_online_top8','load_online_top8_attention'),
    'sparse_window_top8':('mindie_laser_candidate_online_top8','load_online_top8_attention'),
    'sparse_inline_pv':('mindie_laser_candidate_inline_pv','load_inline_pv_attention'),
    'sparse_postmax_scale':('mindie_laser_candidate_postmax_scale','load_postmax_attention'),
    'sparse_delay':('mindie_laser_candidate_delay','load_delay_attention'),
    'sparse_pair':('mindie_laser_candidate_pair','load_pair_attention'),
    'sparse_fma_float':('mindie_laser_candidate_fma','load_fma_attention'),
    'sparse_fma_half':('mindie_laser_candidate_fma','load_fma_attention'),
    'sparse_finalize':('mindie_laser_candidate_finalize','load_candidate_finalize_attention')}

def candidate_manifest_operation(manifest):
    import json
    from pathlib import Path
    from mindie_laser_quant_fused import validate_preprocess_manifest
    operation=json.loads(Path(manifest).read_text()).get('operation')
    if operation not in CANDIDATE_LOADERS:raise ValueError('Unsupported candidate manifest operation')
    validate_preprocess_manifest(manifest,operation=operation)
    return operation

def load_candidate_variant(base,manifest,pv_manifest,device):
    import importlib
    operation=candidate_manifest_operation(manifest)
    module,name=CANDIDATE_LOADERS[operation]
    return getattr(importlib.import_module(module),name)(base,manifest,pv_manifest,device)
