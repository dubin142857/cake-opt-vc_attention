"""Independent full-K/global-softmax experiment; no implicit attention fallback."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path


class GlobalAttention:
    def __init__(self,base,native,evidence,device,skip_zero=True):
        self.base,self.native=base,native
        self.device,self.skip_zero=device,bool(skip_zero)
        self.metadata=dict(evidence,skip_zero=self.skip_zero,
            contract='FP16 QK result/scale/sub/exp, global maximum; FP32 rowsum/PV; FP16 intermediate output',
            scheduling='Q32 full-K score, global softmax, then PV; one ring slot per Cube',
            equivalence='Zero-probability PV blocks may be skipped; no threshold approximation')

    def prepare(self,q,k,v):
        import torch_npu
        from mindie_laser_backend import validate_mha_inputs
        validate_mha_inputs(q,k,v)
        if any(x.device.index!=self.device or x.requires_grad for x in (q,k,v)):
            raise ValueError('Inference inputs must use the configured NPU')
        prepared=[]
        for x in (q,k,v):
            x=x.contiguous()
            if torch_npu.get_npu_format(x) not in (0,2):x=torch_npu.npu_format_cast(x,2)
            if x.data_ptr()%32:x=x.clone()
            prepared.append(x)
        return self.base.prepare(*prepared)

    def run_native(self,payload):
        import torch
        from mindie_laser_backend import LaserPayload
        if not isinstance(payload,LaserPayload) or payload.tensors[0].device.index!=self.device:
            raise ValueError('Original Laser prepared payload on the configured device required')
        padding=0 if payload.pre_tokens==2147483647 else payload.pre_tokens
        keys=payload.tensors[1].shape[2]-padding
        return self.native(*payload.tensors,payload.shape[2],keys,payload.shape[3],
                           payload.dtype==torch.bfloat16,self.skip_zero)

    def run_prepared(self,payload):return self.run_native(payload)[0]
    def __call__(self,q,k,v):return self.run_prepared(self.prepare(q,k,v))
    def describe_payload(self,payload):
        return dict(self.base.describe_payload(payload),global_softmax=True,skip_zero=self.skip_zero,
            fixed_core_scope='Single MIX kernel + output/counter/ring/stat allocation and native dispatch; no preparation')


def load_global_attention(base,manifest,device=3,skip_zero=True,*,operation="global"):
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import validate_preprocess_manifest
    if physical_device(device)==0:raise ValueError('Physical NPU 0 is reserved')
    if operation not in ('global','global_half_sum','global_pipe','global_q64','global_q128','global_stream','global_stream_q64','global_stream_onepass','global_stream_overlap'):raise ValueError('Unsupported global operation')
    evidence=validate_preprocess_manifest(manifest,operation=operation)
    build=json.loads(Path(manifest).read_text())
    if Path(build.get('cann_root','')).resolve()!=Path(os.environ.get('ASCEND_HOME_PATH','')).resolve():
        raise ValueError('Global build/runtime CANN roots disagree')
    import torch
    import torch_npu
    torch.npu.set_device(device)
    torch.ops.load_library(evidence['library_path'])
    namespace=getattr(torch.ops,{'global':'vc_laser_global','global_half_sum':'vc_laser_global_half_sum','global_pipe':'vc_laser_global_pipe','global_q64':'vc_laser_global_q64','global_q128':'vc_laser_global_q128','global_stream':'vc_laser_global_stream','global_stream_q64':'vc_laser_global_stream_q64','global_stream_onepass':'vc_laser_global_stream_onepass','global_stream_overlap':'vc_laser_global_stream_overlap'}[operation])
    native=namespace.attention.default
    if [str(x.type) for x in native._schema.arguments]!=['Tensor']*3+['int']*3+['bool']*2 or [str(x.type) for x in native._schema.returns]!=['Tensor']*4:
        raise ValueError('Unrecognized global attention native schema')
    metadata=json.loads(namespace.metadata())
    if metadata.get('soc')!=evidence['soc'] or metadata.get('operation')!={'global':'global_attention','global_half_sum':'global_attention_half_sum','global_pipe':'global_attention_pipe','global_q64':'global_attention_q64','global_q128':'global_attention_q128','global_stream':'global_attention_stream','global_stream_q64':'global_attention_stream_q64','global_stream_onepass':'global_attention_stream_onepass','global_stream_overlap':'global_attention_stream_overlap'}[operation] or metadata.get('contract_version')!=evidence['contract_version']:
        raise ValueError('Global metadata/build mismatch')
    result=GlobalAttention(base,native,dict(build=evidence,native=metadata,schema=str(native._schema)),device,skip_zero)
    if operation in ('global_half_sum','global_pipe','global_q64','global_q128'):
        result.metadata['contract']='FP16 QK/P; FP16 four-K128 chunk fold and vcadd; FP32 cross-chunk rowsum/PV; FP16 intermediate output'
    if operation in ('global_pipe','global_q64','global_q128'):result.metadata['scheduling']='QK/PV phase-local Cube double buffering; serial global softmax between phases'
    if operation in ('global_q64','global_q128'):
        from prepare_mindie_laser_global_geometry_source import geometry
        g=geometry(int(operation.removeprefix('global_q')))
        if any(metadata.get(k)!=v for k,v in g.items()):raise ValueError('Native Q-geometry mismatch')
        result.metadata['scheduling']=f'Q{g["q_rows"]} full-K; QK/PV phase-local Cube double buffering'
    if operation.startswith('global_stream'):
        if metadata.get('stream_k_blocks')!=16 or metadata.get('online_softmax') is not True:
            raise ValueError('Native online-stream metadata mismatch')
        result.metadata.update(contract='FP16 QK/scale/P/alpha; FP16 four-K128 rowsum; FP32 online rowsum/output recurrence; final FP16 intermediate',
                               scheduling='Q128 K2048 bounded ring; QK -> vector online softmax -> PV -> vector FP32 recurrence')
    if operation in ('global_stream_q64','global_stream_onepass'):
        from prepare_mindie_laser_stream_onepass_source import geometry
        g=geometry(operation)
        if any(metadata.get(k)!=v for k,v in g.items()):raise ValueError('One-pass/control native metadata mismatch')
        result.metadata['scheduling']='Q64 K2048: '+('one UB score pass' if g['onepass'] else 'two score passes')
    elif operation=='global_stream' and any(metadata.get(k)!=v for k,v in dict(q_rows=128,vector_rows=64,ub_bytes=167936).items()):
        raise ValueError('Original stream native geometry mismatch')
    if operation=='global_stream_overlap':
        expected=dict(q_rows=128,vector_rows=64,ub_bytes=167936,score_slots_per_cube=2,lookahead_chunks=1)
        if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('Lookahead native layout mismatch')
        result.metadata['scheduling']='Q128 K2048: two GM score/P slots, one-chunk QK lookahead'
    return result


def last_stream_ring(state,payload):
    """Normalize the persisted last-stream P view; no unused slots are semantic.

    E81 uses [core,16,Q,128]; lookahead uses [core,2,16,128,128].
    Native buffers retain stale/undefined data outside the valid last prefix.
    """
    ring=state[2]
    if ring.ndim==4:
        if ring.shape[1]!=16 or ring.shape[2] not in (64,128) or ring.shape[3]!=128:
            raise ValueError('Invalid single stream-ring layout')
        return state
    if ring.ndim!=5 or tuple(ring.shape[1:])!=(2,16,128,128):
        raise ValueError('Invalid lookahead stream-ring layout')
    keys=payload.tensors[1].shape[2]
    if keys<=0 or keys%128:raise ValueError('Invalid padded K extent')
    slot=((keys//128-1)//16)&1
    return (state[0],state[1],ring[:,slot],state[3])


def last_task_coordinates(payload,cores,tile_rows=32):
    """Map persisted ring rows to true (flattened B/H, query); omit padding."""
    if tile_rows not in (32,64,128):raise ValueError('Unsupported tile rows')
    qtiles=(payload.shape[2]+tile_rows-1)//tile_rows
    tasks=payload.shape[0]*payload.shape[1]*qtiles
    if not 1<=cores<=tasks:raise ValueError('Invalid last-task core count')
    coordinates={}
    for core in range(cores):
        task=core+(tasks-1-core)//cores*cores
        for row in range(tile_rows):
            query=task%qtiles*tile_rows+row
            if query<payload.shape[2]:coordinates[(task//qtiles,query)]=(core,row)
    return coordinates


def last_task_query_mask(payload,cores,torch,tile_rows=32):
    mask=torch.zeros((cores,tile_rows),dtype=torch.bool)
    for core,row in last_task_coordinates(payload,cores,tile_rows).values():mask[core,row]=True
    return mask


def compare_last_task_state(actual,reference,payload,torch,valid_blocks=None):
    """Compare P/max/sum on matching real queries, never same slot addresses.

    Ring layout is [core,Kblock,Qrow,128], stats [core,max(Q),sum(Q)].
    Branch counters legitimately differ with Q coarsening and are excluded.
    """
    maps=[]
    for state in (actual,reference):
        ring,stats=state[2:4]
        if ring.ndim!=4 or ring.shape[3]!=128 or stats.shape!=(ring.shape[0],2*ring.shape[2]):
            raise ValueError('Invalid persisted global geometry')
        maps.append(last_task_coordinates(payload,ring.shape[0],ring.shape[2]))
    keys=sorted(maps[0].keys() & maps[1].keys())
    if not keys:raise AssertionError('No common real queries in persisted states')
    selected=[]
    for state,mapping in zip((actual,reference),maps):
        cores,rows=zip(*(mapping[k] for k in keys))
        cores=torch.tensor(cores,dtype=torch.int64);rows=torch.tensor(rows,dtype=torch.int64)
        ring,stats=state[2:4]
        if valid_blocks is not None and not 1<=valid_blocks<=ring.shape[1]:raise ValueError("Invalid valid ring extent")
        selected.append((ring[cores,:valid_blocks,rows,:].contiguous(),stats[cores,rows].contiguous(),
                         stats[cores,ring.shape[2]+rows].contiguous()))
    differences=[int((a.view(torch.int16 if i==0 else torch.int32)!=b.view(torch.int16 if i==0 else torch.int32)).sum())
                 for i,(a,b) in enumerate(zip(*selected))]
    return dict(common_real_queries=len(keys),probability_bit_differences=differences[0],
                maximum_bit_differences=differences[1],rowsum_bit_differences=differences[2])


def half_chunk_sums(probabilities,torch):
    """CPU model: half cross-block folding, then FP32 sum rounded to half.

    The final hardware vcadd tree can differ; this is not a bitwise oracle.
    Global accumulation stays FP32, including long equal-weight cases.
    """
    padded=torch.nn.functional.pad(probabilities,(0,(-probabilities.shape[-1])%512))
    chunks=padded.reshape(*probabilities.shape[:-1],-1,4,128).half()
    folded=(chunks[...,0,:]+chunks[...,1,:]).half()
    for part in (2,3):folded=(folded+chunks[...,part,:]).half()
    return folded.float().sum(-1).half().float().sum(-1)


def decoded_reference(payload,ids,torch,*,half_sum=False):
    """Independent CPU model of the new arithmetic, not old online rescaling."""
    padding=0 if payload.pre_tokens==2147483647 else payload.pre_tokens
    q,k,v=payload.tensors
    q=q.index_select(2,ids.to(q.device)).float().cpu();k=k.float().cpu();v=v.float().cpu()
    scores=((q@k.transpose(-1,-2)).half()*torch.tensor(payload.scale,dtype=torch.float16)).half()
    if padding:scores[...,-padding:]=-float('inf')
    maxima=scores.amax(-1)
    probabilities=(scores-maxima.unsqueeze(-1)).half().exp().float()
    if half_sum:
        sums=half_chunk_sums(probabilities,torch)
    else:sums=probabilities.sum(-1)
    output=(probabilities@v)/sums.unsqueeze(-1)
    output=output[...,:payload.shape[3]].half().to(payload.dtype)
    return dict(output=output,probabilities=probabilities,maxima=maxima.float(),sums=sums)


def validate_counters(result,payload,torch,tile_rows=32,stream_blocks=None):
    """Counters describe actual branches, including forced first-block init."""
    counts=result[1].cpu()
    if counts.dtype!=torch.int64 or counts.ndim!=2 or counts.shape[1]!=64 or bool((counts<0).any()):
        raise AssertionError('Invalid global counter storage')
    if tile_rows not in (32,64,128):raise ValueError("Unsupported counter geometry")
    tasks=payload.shape[0]*payload.shape[1]*((payload.shape[2]+tile_rows-1)//tile_rows)
    totals=counts[:,:8].sum(0).tolist()
    blocks=payload.tensors[1].shape[2]//128
    if totals[0]!=tasks or totals[1]!=tasks*blocks or totals[1]!=totals[2]+totals[3] or totals[1]!=totals[4]+totals[5]:
        raise AssertionError(f'Incomplete/stale branch counters: {totals}')
    if stream_blocks is not None:
        if stream_blocks!=16 or totals[7]!=tasks*((blocks+stream_blocks-1)//stream_blocks):
            raise AssertionError('Incomplete/stale stream counters')
    return dict(stream_chunks=totals[7],query_tiles=totals[0],qk_blocks=totals[1],pv_blocks=totals[2],skipped_blocks=totals[3],
                actual_zero_blocks=totals[4],actual_nonzero_blocks=totals[5],forced_first_zero=totals[6],
                skipped_fraction=totals[3]/totals[1])


def stream_decoded_reference(payload,ids,torch,*,stats_only=False):
    """CPU online contract, including half alpha but FP32 old output.

    This is a numerical oracle, not a vcadd/Exp bitwise emulator. A stream
    contains 16 K128 blocks; denominators add four-block half folds in order.
    """
    q,k,v=payload.tensors
    q=q.index_select(2,ids.to(q.device)).float().cpu();k=k.float().cpu()
    if not stats_only:v=v.float().cpu()
    scores=((q@k.transpose(-1,-2)).half()*torch.tensor(payload.scale,dtype=torch.float16)).half()
    padding=0 if payload.pre_tokens==2147483647 else payload.pre_tokens
    if padding:scores[...,-padding:]=-float('inf')
    maximum=torch.full(scores.shape[:-1],-float('inf'),dtype=torch.float16)
    sums=torch.zeros_like(maximum,dtype=torch.float32)
    output=None if stats_only else torch.zeros((*maximum.shape,128),dtype=torch.float32)
    for begin in range(0,scores.shape[-1],2048):
        chunk=scores[...,begin:begin+2048]
        new_max=torch.maximum(maximum,chunk.amax(-1))
        alpha=(maximum-new_max).half().exp().float()
        probability=(chunk-new_max.unsqueeze(-1)).half().exp().float()
        sums=sums*alpha
        for offset in range(0,probability.shape[-1],512):
            sums=sums+half_chunk_sums(probability[...,offset:offset+512],torch)
        if not stats_only:output=output*alpha.unsqueeze(-1)+probability@v[...,begin:begin+2048,:]
        maximum=new_max
    if not stats_only:output=(output/sums.unsqueeze(-1))[...,:payload.shape[3]].half().to(payload.dtype)
    return dict(output=output,probabilities=probability,maxima=maximum.float(),sums=sums)


def stream_stat_samples(state,payload,torch):
    """All persisted *valid* rows, paired by head/query with the CPU contract."""
    tile_rows=state[3].shape[1]//2
    mapping=last_task_coordinates(payload,state[3].shape[0],tile_rows)
    keys=sorted(mapping);query_ids=sorted({query for _,query in keys})
    model=stream_decoded_reference(payload,torch.tensor(query_ids),torch,stats_only=True)
    lookup={q:i for i,q in enumerate(query_ids)}
    heads=torch.tensor([head for head,_ in keys]);indices=torch.tensor([lookup[q] for _,q in keys])
    cores,rows=zip(*(mapping[key] for key in keys));cores=torch.tensor(cores);rows=torch.tensor(rows)
    actual=state[3]
    return dict(coordinates=torch.tensor(keys),actual_max=actual[cores,rows],actual_sum=actual[cores,tile_rows+rows],
                reference_max=model['maxima'].flatten(0,1)[heads,indices],reference_sum=model['sums'].flatten(0,1)[heads,indices])
