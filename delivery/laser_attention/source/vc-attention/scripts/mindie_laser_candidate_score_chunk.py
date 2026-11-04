"""Block1 attention with eight-score ownership groups and bulk candidate output."""
from pathlib import Path
import os,json
from mindie_laser_candidate_block1 import CandidateBlock1Attention

def load_candidate_score_chunk_attention(base,manifest,pv_manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    from mindie_laser_sparse_pv import load_sparse_pv
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_candidates_score_chunk')
    build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Lookup CANN root mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_candidates_score_chunk
    expected=dict(soc=evidence['soc'],operation='block1_score_chunk',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=54656,score_slots=2,candidates_per_block=1,geometry='mask_words_in_half_storage',lookup_bytes=6176,output_slots=8,defer_first_ack=False,readiness_events=2,score_ack='after_mte2_read',score_ack_pipe='MTE2',output_bulk_columns=8,output_batch_buffers=1,producer_counter_version=3,score_chunk_blocks=8,bulk_dma_api='DataCopyPadExt')
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Lookup metadata mismatch')
    native=ns.produce.default;probe=ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*3:raise ValueError('Lookup producer schema mismatch')
    if [str(x.type) for x in probe._schema.arguments]!=['Tensor']*2+['int']*2 or [str(x.type) for x in probe._schema.returns]!=['Tensor']*2:raise ValueError('Lookup probe schema mismatch')
    return CandidateScoreChunkAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)


class CandidateScoreChunkAttention(CandidateBlock1Attention):
    def make_layout(self,heads,queries,keys):
        g=super().make_layout(heads,queries,keys)
        g.update(output_bulk_columns=8,producer_counter_version=3,score_chunk_blocks=8)
        return g
