"""Post-block-max scale, explicit collapsed-tie approximation, independent source audit."""
from pathlib import Path
import json,os
from mindie_laser_candidate_inline_pv import CandidateInlineAttention,inline_geometry
from mindie_laser_sparse_pv import load_sparse_pv

class AuditedState(tuple):
    def __new__(cls,state,adapter,payload):
        obj=super().__new__(cls,state);obj.source_context=(adapter,payload);return obj

class CandidatePostmaxAttention(CandidateInlineAttention):
    def make_layout(self,heads,queries,keys):
        g=inline_geometry(heads,queries,keys);g.update(producer_counter_version=8,postmax_scale=True);return g
    def run_native(self,payload):return AuditedState(super().run_native(payload),self,payload)
    def scale_table(self,dimension):
        if dimension in self.scale_tables:return self.scale_tables[dimension]
        import torch
        from plugins.vc_attention_e116.math_probe import audit_map
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16);finite=(bits.int()&0x7c00)!=0x7c00
        values=torch.cat((bits[finite],torch.tensor([0x7c00,-1024],dtype=torch.int16),torch.zeros(2046,dtype=torch.int16))).view(torch.float16).reshape(512,128)
        source=values.to(f'npu:{self.device}');scale=float(torch.tensor(dimension**-.5,dtype=torch.float16))
        runs=[tuple(t.cpu() for t in self.scale_probe(source,scale)) for _ in range(2)]
        if any(not torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)) for a,b in zip(*runs)):raise AssertionError('Postmax scale mechanism nondeterminism')
        report=audit_map(values,*runs[0],torch)
        if report['monotonicity_violations'] or report['composition_value_bit_mismatches']:raise AssertionError('Postmax scalar mechanism premise falsified:'+str(report))
        table=torch.zeros(65536,dtype=torch.int16);codes=values.flatten().view(torch.int16).int()&65535;table[codes.long()]=runs[0][0].flatten().view(torch.int16)
        self.scale_tables[dimension]=(table,report);return table,report

def load_postmax_attention(base,manifest,pv_manifest,device=7):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    build=json.loads(Path(manifest).read_text());evidence=validate_preprocess_manifest(manifest,operation='sparse_postmax_scale')
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Postmax CANN root mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_postmax_scale
    expected=dict(soc=evidence['soc'],operation='block1_window_top8_postmax_scale_pv',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=170624,
        score_slots=2,candidates_per_block=1,score_chunk_blocks=8,selection_window_blocks=32,output_rank=8,output_columns=32,producer_counter_version=8,pv_counter_version=2,
        inline_pv=True,scale_elements_per_block=64,scale_order='raw_block_max_then_half_scale',selection='window32_top8_then_resident_top8',tie='Sort32 observed input-stable; validate value/ID relation')
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Postmax native metadata mismatch')
    native,probe=ns.produce.default,ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*4+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*9:raise ValueError('Postmax ABI mismatch')
    if [str(x.type) for x in ns.qk_tile.default._schema.arguments]!=['Tensor']*2:raise ValueError('Postmax QK audit ABI mismatch')
    mech=validate_preprocess_manifest(Path(manifest).with_name('max_scale_probe_cann920_build.json'),operation='max_scale_probe')
    torch.ops.load_library(mech['library_path']);evidence=dict(evidence,scale_mechanism=mech)
    result=CandidatePostmaxAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
    result.qk_tile=ns.qk_tile.default;result.scale_probe=torch.ops.vc_laser_max_scale_probe.run.default;result.scale_tables={}
    result.metadata['stage_scopes']['tie']='Raw first block argmax; scaled rounded ties may choose another V. Every selected ID independently checked as legal scaled block maximum; video unverified.'
    return result
