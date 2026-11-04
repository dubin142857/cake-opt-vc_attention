"""Output DMA overlap with unchanged E88 bitmask and arithmetic."""
from pathlib import Path
import os,json
from mindie_laser_candidate_lookup import CandidateLookupAttention

def load_candidate_dma_attention(base,manifest,pv_manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    from mindie_laser_sparse_pv import load_sparse_pv
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_candidates_dma_overlap')
    build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Lookup CANN root mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_candidates_dma_overlap
    expected=dict(soc=evidence['soc'],operation='block2_lookup_dma_overlap',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=52352,score_slots=2,candidates_per_block=2,geometry='mask_words_in_half_storage',lookup_bytes=6176,output_slots=2,defer_first_ack=True,readiness_events=2)
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Lookup metadata mismatch')
    native=ns.produce.default;probe=ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*3:raise ValueError('Lookup producer schema mismatch')
    if [str(x.type) for x in probe._schema.arguments]!=['Tensor']*2+['int']*2 or [str(x.type) for x in probe._schema.returns]!=['Tensor']*2:raise ValueError('Lookup probe schema mismatch')
    return CandidateLookupAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
