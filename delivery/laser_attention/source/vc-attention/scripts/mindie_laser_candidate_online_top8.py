"""Online Top8 over E97 block maxima; dense/video quality is not certified."""
from pathlib import Path
import json,os
from mindie_laser_candidate_score_chunk import CandidateScoreChunkAttention
from mindie_laser_candidate_block1 import block1_geometry
from mindie_laser_candidate_finalize import CandidateFinalizeAttention
from mindie_laser_sparse_finalize import TopkInputs,load_sparse_finalize


def online_geometry(heads,queries,keys,window_blocks=8):
    if window_blocks not in (8,32):raise ValueError("Supported online windows8/32")
    g=block1_geometry(heads,queries,keys)
    g.update(columns=32,output_bulk_columns=8,producer_counter_version=5,score_chunk_blocks=8,native_finalize=True)
    if window_blocks==32:g.update(producer_counter_version=6,selection_window_blocks=32)
    return g

class OnlineTop8Producer(CandidateScoreChunkAttention):
    def make_layout(self,heads,queries,keys):return online_geometry(heads,queries,keys,self.metadata.get("selection_window_blocks",8))

class OnlineTop8Attention(CandidateFinalizeAttention):
    def __init__(self,producer,finalizer):
        super().__init__(producer,finalizer)
        self.metadata.update(operation='block1_online_top8_finalize',stage_scopes=dict(
            producer='Full QK + block maximum + UB online Top8 + compact score/index output',
            selection='Direct score views and prepared constant rank positions; no TopK',
            consumer='Original map + FP32 softmax + packet + PV + trim/cast',
            tie='Descending block-max values exact; tied IDs may differ from torch.TopK; same source-pair obligation'))
    def make_layout(self,heads,queries,keys):return online_geometry(heads,queries,keys,self.metadata.get("selection_window_blocks",8))
    def prepare(self,q,k,v):
        import torch
        payload=super().prepare(q,k,v);g=payload.layout
        key=('online_positions',g['heads']*g['qwork'],g['rank'])
        if key not in self.geometry_cache:
            self.geometry_cache[key]=torch.arange(g['rank'],dtype=torch.int64).repeat(key[1],1).to(q.device)
        payload.top8_positions=self.geometry_cache[key]
        return payload
    def select(self,payload,state):
        return TopkInputs(state[0][:,:payload.layout['rank']].contiguous(),payload.top8_positions,state[1])

def load_online_top8_attention(base,manifest,pv_manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    from mindie_laser_sparse_pv import load_sparse_pv
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    operation=json.loads(Path(manifest).read_text()).get('operation')
    if operation not in ('sparse_online_top8','sparse_window_top8'):raise ValueError('Online/window producer operation required')
    window=operation=='sparse_window_top8'
    evidence=validate_preprocess_manifest(manifest,operation=operation)
    build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Online build/runtime CANN mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=getattr(torch.ops,'vc_laser_sparse_window_top8' if window else 'vc_laser_sparse_online_top8')
    expected=dict(soc=evidence['soc'],operation='block1_online_top8',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,
        ub_bytes=109184,score_slots=2,candidates_per_block=1,score_chunk_blocks=8,output_rank=8,output_columns=32,
        producer_counter_version=5,selection='UB_top8_then_direct_scores',tie='Sort32 observed input-stable; validate value/ID relation')
    if window:expected.update(operation='block1_window_top8',ub_bytes=133760,producer_counter_version=6,selection_window_blocks=32,selection='window32_top8_then_resident_top8')
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Online native metadata mismatch')
    native,probe=ns.produce.default,ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*3:raise ValueError('Online producer schema mismatch')
    if [str(x.type) for x in probe._schema.arguments]!=['Tensor']*2+['int']*2 or [str(x.type) for x in probe._schema.returns]!=['Tensor']*2:raise ValueError('Online probe schema mismatch')
    producer=OnlineTop8Producer(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
    final=Path(__file__).resolve().parents[1]/'results/mindie_laser_sparse_finalize_cann920_build.json'
    return OnlineTop8Attention(producer,load_sparse_finalize(final,device))
