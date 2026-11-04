"""Exact-PV rewrite within the explicitly approximate postmax attention contract."""
from pathlib import Path
import json,os
from mindie_laser_candidate_postmax_scale import CandidatePostmaxAttention
from mindie_laser_sparse_pv import load_sparse_pv

def load_fma_attention(base,manifest,pv_manifest,device=7):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    build=json.loads(Path(manifest).read_text());operation=build['operation'];mode=operation.removeprefix('sparse_fma_')
    if mode not in ('float','half'):raise ValueError('FMA manifest operation')
    evidence=validate_preprocess_manifest(manifest,operation=operation)
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Postmax CANN root mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=getattr(torch.ops,'vc_laser_sparse_fma_'+mode)
    expected=dict(soc=evidence['soc'],operation='block1_window_top8_fma_'+mode+'_pv',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=170624,
        score_slots=2,candidates_per_block=1,score_chunk_blocks=8,selection_window_blocks=32,output_rank=8,output_columns=32,producer_counter_version=8,pv_counter_version=2,
        inline_pv=True,pv_arithmetic='fp32_vmla_'+mode,scale_elements_per_block=64,scale_order='raw_block_max_then_half_scale',selection='window32_top8_then_resident_top8',tie='Sort32 observed input-stable; validate value/ID relation')
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Postmax native metadata mismatch')
    native,probe=ns.produce.default,ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*4+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*9:raise ValueError('Postmax ABI mismatch')
    if [str(x.type) for x in ns.qk_tile.default._schema.arguments]!=['Tensor']*2:raise ValueError('Postmax QK audit ABI mismatch')
    mech=validate_preprocess_manifest(Path(manifest).with_name('max_scale_probe_cann920_build.json'),operation='max_scale_probe')
    torch.ops.load_library(mech['library_path']);evidence=dict(evidence,scale_mechanism=mech)
    result=CandidatePostmaxAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
    result.fma_probe=ns.fma_probe.default
    if [str(x.type) for x in result.fma_probe._schema.arguments]!=['Tensor']*3 or [str(x.type) for x in result.fma_probe._schema.returns]!=['Tensor']*3:raise ValueError('FMA probe ABI mismatch')
    result.qk_tile=ns.qk_tile.default;result.scale_probe=torch.ops.vc_laser_max_scale_probe.run.default;result.scale_tables={}
    result.metadata['stage_scopes']['tie']='Raw first block argmax; scaled rounded ties may choose another V. Every selected ID independently checked as legal scaled block maximum; video unverified.'
    return result
