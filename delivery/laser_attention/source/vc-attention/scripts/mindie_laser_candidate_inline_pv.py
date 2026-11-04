"""Same-AIV window Top8, normalization and FP32 PV with explicit evidence outputs."""
from pathlib import Path
import json,os
from mindie_laser_candidate_score_chunk import CandidateScoreChunkAttention
from mindie_laser_candidate_online_top8 import online_geometry
from mindie_laser_sparse_pv import SparsePvPayload

def inline_geometry(heads,queries,keys):
    g=online_geometry(heads,queries,keys,32)
    g.update(producer_counter_version=7,pv_counter_version=2,inline_pv=True)
    return g

def validate_inline_pv_counters(state,g):
    import torch
    count=state[1].cpu();tasks=g['heads']*(g['qwork']//128);cores=min(24,tasks)
    if g.get('pv_counter_version')!=2 or count.shape!=(2*cores,64) or count.dtype!=torch.int64 or (count<0).any():raise AssertionError('Inline PV counter contract')
    for core in range(cores):
        n=len(range(core,tasks,cores));wanted=[64*n,64*n*g['rank'],0,4*n,20*n]
        for sub in range(2):
            v=count[2*core+sub]
            if v[:5].tolist()!=wanted or v[5:].any():raise AssertionError('Inline PV actual query assignment/counters')
    return dict(query_rows=64*tasks*2,value_rows_read=64*tasks*2*g['rank'],invalid_rows=0,groups=tasks*8,cores=2*cores,output_dma_calls=tasks*40,pv_counter_version=2)

class InlineSelection(tuple):
    def __new__(cls,state,rank):
        obj=super().__new__(cls,(state[5],state[6],state[0][:,:rank].contiguous()));obj.native=state;return obj

class CandidateInlineAttention(CandidateScoreChunkAttention):
    def __init__(self,base,native,probe,pv,evidence,metadata,device):
        super().__init__(base,native,probe,pv,evidence,metadata,device)
        self.metadata=dict(metadata,stage_scopes=dict(producer='Full QK/blockmax/window Top8 plus same-AIV FP32 softmax/FP32 PV and all evidence outputs',
            selection='Native result views only; no TopK',consumer='Public trim/cast only; PV was executed in producer',tie='Descending block-max values and original per-block ID/score relation; no Torch.TopK tie identity claim'))
    def make_layout(self,heads,queries,keys):return inline_geometry(heads,queries,keys)
    def produce(self,payload):
        p,g=payload.base,payload.layout
        return self.native(*p.tensors[:3],payload.positions,g['queries'],g['keys'],p.shape[3])
    def select(self,payload,state):return InlineSelection(state,payload.layout['rank'])
    def consume(self,payload,selection):
        p,g=payload.base,payload.layout;s=selection.native
        result=(s[3],s[4]);packet=SparsePvPayload(p.tensors[2].view(-1,128),s[7],s[8],g['heads']*g['qwork'],g['rank'])
        out=s[3].view(p.shape[0],p.shape[1],g['qwork'],128)[:,:,:g['queries'],:p.shape[3]].to(p.dtype).contiguous()
        return out,result,packet
    def run_native(self,payload):
        state=self.produce(payload);selected=self.select(payload,state);out,result,packet=self.consume(payload,selected)
        return out,state,selected,result,packet

def load_inline_pv_attention(base,manifest,pv_manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    from mindie_laser_sparse_pv import load_sparse_pv
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_inline_pv');build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Inline build/runtime CANN mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_inline_pv
    expected=dict(soc=evidence['soc'],operation='block1_window_top8_inline_pv',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=170624,
        score_slots=2,candidates_per_block=1,score_chunk_blocks=8,selection_window_blocks=32,output_rank=8,output_columns=32,
        producer_counter_version=7,pv_counter_version=2,inline_pv=True,selection='window32_top8_then_resident_top8',tie='Sort32 observed input-stable; validate value/ID relation')
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Inline native metadata mismatch')
    native,probe=ns.produce.default,ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*4+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*9:raise ValueError('Inline native schema mismatch')
    if [str(x.type) for x in probe._schema.arguments]!=['Tensor']*2+['int']*2 or [str(x.type) for x in probe._schema.returns]!=['Tensor']*2:raise ValueError('Inline probe schema mismatch')
    return CandidateInlineAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
