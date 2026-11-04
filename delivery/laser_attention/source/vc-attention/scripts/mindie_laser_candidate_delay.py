"""Raw selection followed by centered FP32 temperature; explicit new rounding graph."""
from pathlib import Path
import json,os
from mindie_laser_candidate_inline_pv import CandidateInlineAttention,inline_geometry
from mindie_laser_sparse_pv import load_sparse_pv
class CandidateDelayAttention(CandidateInlineAttention):
    def make_layout(self,heads,queries,keys):
        g=inline_geometry(heads,queries,keys);g.update(delayed_temperature=True,producer_counter_version=10);return g

def load_delay_attention(base,manifest,pv_manifest,device=7):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    if physical_device(device)==0:raise ValueError('Physical0reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_delay');build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('CANNroot differs')
    parent=Path(os.environ['VC_ATTENTION_INCUMBENT_MANIFEST']);raw=validate_preprocess_manifest(parent,operation='sparse_pair')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(raw['library_path']);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_delay
    expected={'soc': 'Ascend910B2', 'operation': 'block1_raw_top8_center_scale_pv', 'contract_version': 1, 'q_rows': 128, 'k_columns': 128, 'vector_rows': 64, 'ub_bytes': 155520, 'score_slots': 2, 'candidates_per_block': 1, 'score_chunk_blocks': 8, 'selection_window_blocks': 32, 'output_rank': 8, 'output_columns': 32, 'producer_counter_version': 10, 'pv_counter_version': 2, 'inline_pv': True, 'pv_arithmetic': 'fp32_vmla_half', 'delayed_temperature': True, 'scale_elements_per_query_tile': 1536, 'scale_order': 'raw_Top8_then_FP32_center_scale', 'score_dtype': 'float32_scaled', 'raw_reference': 'original_E119_binary_temperature1', 'selection': 'raw_window32_top8_then_resident_top8', 'tie': 'Raw half order; scaled-half ties may select different V; raw reference exact source contract'}
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Delay native metadata mismatch')
    for fn,args,returns in ((ns.produce.default,['Tensor']*4+['int']*3,['Tensor']*9),(ns.raw_reference.default,['Tensor']*4+['int']*3+['str'],['Tensor']*9)):
        if [str(x.type) for x in fn._schema.arguments]!=args or [str(x.type) for x in fn._schema.returns]!=returns:raise ValueError('Delay/rawreference ABI')
    evidence=dict(evidence,raw_reference_provider=raw)
    obj=CandidateDelayAttention(base,ns.produce.default,ns.probe.default,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
    obj.raw_reference=ns.raw_reference.default;obj.raw_library=raw['library_path'];obj.fma_probe=ns.fma_probe.default;obj.pair_probe=ns.pair_probe.default;obj.pair_batch=True
    obj.metadata['stage_scopes']=dict(producer='Original half fullQK/rawTop8, centeredFP32temperature/softmax andsameFP32PV includingproof outputs',selection='Native resultviews only',consumer='Publictrim/cast',source_reference='Immutable E119binarytemperature1 diagnostic only; untimed; ordinary E119timed separately')
    return obj
