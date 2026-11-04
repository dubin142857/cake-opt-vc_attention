"""Same E87 candidate arithmetic with exact-position bitmap lookup."""
from pathlib import Path
import os,json
from mindie_laser_sparse_candidates import CandidateAttention

def mask_geometry():
    import torch
    def signed(x):return x if x<2**31 else x-2**32
    words=[]
    for index in range(128):
        for word in range(4):
            x=0xffffffff^(1<<(index%32)) if word==index//32 else 0xffffffff
            words.append(signed(x))
    for valid in range(129):
        for word in range(4):
            n=max(0,min(32,valid-word*32));words.append(signed((1<<n)-1))
    words.extend([0]*4)
    assert len(words)==1032
    words.extend((i//4)*4 for i in range(256))
    words.extend((i%4)*4 for i in range(256))
    assert len(words)==1544
    carrier=torch.zeros(8192,dtype=torch.float16)
    carrier.view(torch.int32)[:1544]=torch.tensor(words,dtype=torch.int32)
    return carrier.view(64,128)

class CandidateLookupAttention(CandidateAttention):
    def make_positions(self,device):return mask_geometry().to(device)

def load_candidate_lookup_attention(base,manifest,pv_manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    from mindie_laser_sparse_pv import load_sparse_pv
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_candidates_lookup')
    build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Lookup CANN root mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_candidates_lookup
    expected=dict(soc=evidence['soc'],operation='block2_lookup',contract_version=1,q_rows=128,k_columns=128,vector_rows=64,ub_bytes=51968,score_slots=2,candidates_per_block=2,geometry='mask_words_in_half_storage',lookup_bytes=6176)
    metadata=json.loads(ns.metadata())
    if metadata!=expected:raise ValueError('Lookup metadata mismatch')
    native=ns.produce.default;probe=ns.probe.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*3 or [str(x.type) for x in native._schema.returns]!=['Tensor']*3:raise ValueError('Lookup producer schema mismatch')
    if [str(x.type) for x in probe._schema.arguments]!=['Tensor']*2+['int']*2 or [str(x.type) for x in probe._schema.returns]!=['Tensor']*2:raise ValueError('Lookup probe schema mismatch')
    return CandidateLookupAttention(base,native,probe,load_sparse_pv(pv_manifest,device),evidence,metadata,device)
