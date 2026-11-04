"""Map builtin TopK results, normalize, pack and accumulate sparse PV in one AIV kernel."""
from pathlib import Path
from dataclasses import dataclass
import json,os

def finalize_geometry(rank):
    import torch
    if type(rank)!=int or not 1<=rank<=8:raise ValueError('Finalize rank1..8 required')
    x=torch.zeros(1056,dtype=torch.int32)
    for key in range(16):
        for row in range(16):
            j=key*16+row
            x[j]=(row*8+key)*8 if key<rank else 1024
            x[256+j]=row*4
            x[512+row*16+key]=(key*16+row)*4
            x[768+j]=(row*16+key)*2
    for row in range(16):x[1024+row]=row*16*4
    return x


def finalize_reference(value,candidates,scores,positions):
    import torch
    h,c,q=candidates.shape;rows,rank=scores.shape
    if rows!=h*q or positions.shape!=scores.shape:raise ValueError('Finalize reference shape')
    rr=torch.arange(rows);raw=positions.long();safe=raw.clamp(0,c-1)
    ids=candidates[rr[:,None]//q,safe,rr[:,None]%q].long()
    invalid=(raw<0)|(raw>=c)|(ids<0)|(ids>=len(value))
    ids=torch.where(invalid,-1,ids).to(torch.int32)
    weights=scores.float().softmax(-1).half()
    pv=(value[ids.long().clamp(0,len(value)-1)].double()*weights.double()[...,None]).sum(1)
    pv[invalid.any(1)]=0
    return ids,weights,pv


@dataclass(frozen=True)
class TopkInputs:
    scores: object
    positions: object
    candidates: object

class FinalizedSelection(tuple):
    def __new__(cls,ids,weights,scores,topk):
        obj=super().__new__(cls,(ids,weights,scores));obj.topk=topk;return obj

class SparseFinalize:
    def __init__(self,native,evidence,metadata,device):self.native,self.evidence,self.metadata,self.device=native,evidence,metadata,device
    def run_native(self,value,candidates,scores,positions,geometry):return self.native(value,candidates,scores,positions,geometry)


def load_sparse_finalize(manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_finalize');build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Finalize CANN mismatch')
    import torch,torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path']);ns=torch.ops.vc_laser_sparse_finalize
    metadata=json.loads(ns.metadata())
    expected=dict(soc=evidence['soc'],operation='mapped_softmax_sparse_pv',contract_version=1,rows=16,rank_max=8,columns_max=512,geometry_words=1056,ub_bytes=94560,softmax='fp32_exp_sum_div_half',output='float32')
    if metadata!=expected:raise ValueError('Finalize metadata mismatch')
    native=ns.finalize.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*5 or [str(x.type) for x in native._schema.returns]!=['Tensor']*6:raise ValueError('Finalize schema mismatch')
    return SparseFinalize(native,evidence,metadata,device)
