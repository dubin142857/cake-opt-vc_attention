"""Independent fused sparse weighted-PV component; not full attention."""
from dataclasses import dataclass
from pathlib import Path
import json
import os


@dataclass
class SparsePvPayload:
    value: object
    indices: object
    weights: object
    rows: int
    rank: int


def pack_rows(indices,weights,keys):
    """Rank-major 16-row packets; invalid int64 IDs remain invalid after cast."""
    import torch
    import torch.nn.functional as functional
    if indices.ndim!=2 or indices.shape!=weights.shape or not indices.numel():raise ValueError('Matching nonempty row/rank tensors required')
    rows,rank=indices.shape
    if not 1<=rank<=16 or not 0<keys<=2**31-1:raise ValueError('Rank1..16 and int32 key extent required')
    if indices.dtype not in (torch.int32,torch.int64) or weights.dtype!=torch.float16 or indices.device!=weights.device:
        raise ValueError('Matching-device integer indices and half weights required')
    if indices.dtype==torch.int64:
        indices=torch.where((indices>=0)&(indices<keys),indices,-1).to(torch.int32)
    padded=(rows+15)//16*16
    def packet(x):return functional.pad(x,(0,16-rank,0,padded-rows)).reshape(-1,16,16).transpose(1,2).contiguous()
    return packet(indices),packet(weights)


def validate_counters(result,rows,rank,*,invalid_rows=0):
    import torch
    count=result[1].cpu()
    groups=(rows+15)//16;cores=min(groups,48)
    if count.shape!=(cores,64) or count.dtype!=torch.int64 or (count<0).any() or count[:,4:].any():
        raise AssertionError('Sparse PV invalid counter layout/content')
    for core in range(cores):
        assigned=list(range(core,groups,cores));queries=sum(min(16,rows-group*16) for group in assigned)
        if tuple(count[core,[0,1,3]].tolist())!=(queries,queries*rank,len(assigned)) or count[core,2]>queries:
            raise AssertionError('Sparse PV incomplete/incorrect coverage')
    totals=count.sum(0)
    if int(totals[2])!=invalid_rows:raise AssertionError('Sparse PV invalid index rows detected: '+str(int(totals[2])))
    return dict(query_rows=int(totals[0]),value_rows_read=int(totals[1]),invalid_rows=int(totals[2]),groups=int(totals[3]),cores=cores)


class SparsePv:
    def __init__(self,native,metadata,evidence,device):self.native,self.metadata,self.evidence,self.device=native,metadata,evidence,device
    def prepare(self,value,indices,weights):
        import torch
        import torch_npu
        if value.ndim!=2 or not value.numel() or value.shape[1] not in (64,128) or value.dtype!=torch.float16:
            raise ValueError('Half V[keys,D64/128] required')
        if any(x.device.type!='npu' or x.device.index!=self.device or x.requires_grad for x in (value,indices,weights)):
            raise ValueError('Configured inference NPU inputs required')
        packed_indices,packed_weights=pack_rows(indices,weights,value.shape[0])
        value=torch_npu.npu_format_cast(value.contiguous(),2)
        if value.data_ptr()%32:value=value.clone()
        return SparsePvPayload(value,packed_indices,packed_weights,indices.shape[0],indices.shape[1])
    def run_native(self,payload):
        if not isinstance(payload,SparsePvPayload) or payload.value.device.index!=self.device:raise ValueError('Prepared sparse payload required')
        return self.native(payload.value,payload.indices,payload.weights,payload.rows,payload.rank)
    def run_prepared(self,payload):return self.run_native(payload)[0]
    def __call__(self,value,indices,weights):return self.run_prepared(self.prepare(value,indices,weights))


def load_sparse_pv(manifest,device=5):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    if physical_device(device)==0:raise ValueError('Physical NPU0 reserved')
    evidence=validate_preprocess_manifest(manifest,operation='sparse_pv')
    build=json.loads(Path(manifest).read_text())
    if Path(build['cann_root']).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():raise ValueError('Sparse PV build/runtime CANN mismatch')
    import torch
    import torch_npu
    torch.npu.set_device(device);torch.ops.load_library(evidence['library_path'])
    namespace=torch.ops.vc_laser_sparse_pv;native=namespace.weighted.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*2 or [str(x.type) for x in native._schema.returns]!=['Tensor']*2:
        raise ValueError('Sparse PV native schema mismatch')
    metadata=json.loads(namespace.metadata())
    expected=dict(soc=evidence['soc'],operation='weighted_sparse_pv',contract_version=1,rows=16,rank_max=16,ub_bytes=84032,output='float32')
    if metadata!=expected:raise ValueError('Sparse PV native metadata mismatch')
    return SparsePv(native,metadata,evidence,device)
