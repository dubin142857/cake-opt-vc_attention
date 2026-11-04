"""Finite post-TopK fusion case and source-free CANN capture accounting.

Retrospective E94 -> E97 reconstruction, not a new optimization or proof of
whole attention equivalence. Hardware pipe internals remain unobserved.
"""
from pathlib import Path
from decimal import Decimal,InvalidOperation
from dataclasses import asdict
import argparse,csv,json,math
from core.opgen.architecture import (Architecture,Value,Contract,SemanticProgram,Storage,Resource,Allocation,Instance,Layout,Event,Region,Demand,Binding,Evidence,Platform,Dependency,Duration,DTYPE_BYTES)
from core.opgen.transforms import TransformPlan,FuseEvents,apply_plan
from core.opgen.schedule import SchedulePlan,Cost,solve
from core.opgen.lowering import LoweringPlan,NativeEntry,TensorBoundary,TensorObservation,instantiate
from core.opgen.runner import identify
from core.opgen.trace import make_plan,ingest


# 来源: vc-attention/ascend/laser_sparse_finalize/sparse_finalize_torch.cpp Check/Finalize.
def build_case(project,device,*,rows=38144,keys=38144,columns=320,dim=128,rank=8):
    project=Path(project).resolve()
    if device<=0:raise ValueError('Physical NPU0 is forbidden')
    if rows<=0 or rows%128 or keys<=0 or columns<32 or columns>512 or columns%32 or dim not in (64,128) or not 1<=rank<=8:
        raise ValueError('Case geometry exceeds the native ABI')
    manifest=json.loads((project/'results/mindie_laser_sparse_finalize_cann920_build.json').read_text())
    if manifest['status']!='passed' or manifest['soc']!='Ascend910B2':raise ValueError('Unvalidated native provider')
    if manifest['torch_api']!='torch.ops.vc_laser_sparse_finalize.finalize':raise ValueError('Native entry declaration differs')
    sources=[]
    for spec in manifest['source_files']:
        item=identify(spec['path'],'source')
        if item.sha256!=spec['sha256']:raise ValueError('Compiled source changed')
        sources.append(item)
    library=identify(manifest['library_path'],'runtime')
    if library.sha256!=manifest['library_sha256']:raise ValueError('Compiled native changed')
    for name in ('mindie_laser_sparse_finalize.py','mindie_laser_candidate_finalize.py','mindie_laser_sparse_pv.py','probe_mindie_laser_sparse_attention.py'):
        sources.append(identify(project/'scripts'/name,'harness'))
    ref=';'.join(s.path+'@sha256:'+s.sha256 for s in sources)
    # 本仓自定: explicit approximate boundary shared by source/fused variants.
    contract=Contract('post-topk-map-softmax-packet-pv-v1','approximate',ref,
        'H=1; finite half scores, valid/invalid int64 positions, candidate global IDs; padded half V; native ABI geometry',
        'IDs/packet/counters follow existing checks; softmax half weight maxabs<=1e-3 and rowsum<=1e-3; own-weight FP64 PV rel<=1e-5 abs<=1e-4. Not dense/video quality or universal bitwise equality')
    specs=(('value',(keys,dim),'f16',()),('candidates',(1,columns,rows),'i32',()),('scores',(rows,rank),'f16',()),('positions',(rows,rank),'i64',()),
        ('ids',(rows,rank),'i32',('candidates','positions')),('weights',(rows,rank),'f16',('scores',)),
        ('packet_ids',(rows//16,16,16),'i32',('ids',)),('packet_weights',(rows//16,16,16),'f16',('weights',)),
        ('pv',(rows,dim),'f32',('value','packet_ids','packet_weights')),('counters',(min(48,rows//16),64),'i64',('packet_ids',)))
    values=tuple(Value(n,'input' if not parents else 'opaque_'+n,shape,dtype,parents,external_reference='' if not parents else ref) for n,shape,dtype,parents in specs)
    sizes={v.id:math.prod(v.shape)*DTYPE_BYTES[v.dtype] for v in values}
    def layout(v):
        stride=DTYPE_BYTES[v.dtype];result=[]
        for n in reversed(v.shape):result.append(stride);stride*=n
        return Layout(v.shape,tuple(reversed(result)))
    def regions(names):return tuple(Region(n,0,sizes[n]) for n in names)
    # 本仓自定: E109 frozen prior intervals, not observed hardware bounds.
    jobs=(('map',('candidates','positions'),('ids',),200000,350000),('normalize',('scores',),('weights',),30000,120000),
          ('packet',('ids','weights'),('packet_ids','packet_weights'),40000,180000),('pv',('value','packet_ids','packet_weights'),('pv','counters'),180000,350000))
    events=tuple(Event(n,'stream','compute',regions(read),regions(write),(Demand('stream'),),write,
        Duration(lo,hi,Evidence('assumed','ledger://E109/F467','post-TopK macros','not hardware bounds'))) for n,read,write,lo,hi in jobs)
    source=Architecture('E94-post-topk',SemanticProgram(values,tuple(v.id for v in values if not v.inputs),tuple(v.id for v in values if v.inputs),contract),
        tuple(Storage(v.id,'GM','global',sizes[v.id],Evidence('assumed','native ABI reservation','per-tensor arena','not total device memory')) for v in values),
        (Resource('stream','global',1,Evidence('assumed','sequential stream macros; not individual hardware pipes')),),
        tuple(Allocation(v.id,v.id,sizes[v.id]) for v in values),tuple(Instance(v.id,v.id,v.id,layout(v)) for v in values),events,
        (Dependency('map','normalize','program'),Dependency('map','packet','data'),Dependency('normalize','packet','data'),Dependency('packet','pv','data')),
        tuple(Binding(v.id,tuple(e.id for e in events if v.id in e.semantic_values),(v.id,)) for v in values),dispatch_priority=tuple(e.id for e in events),platform=Platform('Ascend910B2'),
        label='Retrospective source macro graph; all boundary tensors retained; private UB and geometry constant outside shared semantic boundary')
    transform=apply_plan(source,TransformPlan(source.fingerprint(),FuseEvents(tuple(e.id for e in events),'finalize')))
    target=transform.architecture
    schedule=solve(target,SchedulePlan(target.fingerprint(),costs=(Cost('finalize',Duration(350000,650000,Evidence('assumed','ledger://E109/F467','fused prior','may be slower; not summed source costs'))),)))
    result=LoweringPlan('E97-native-finalize',transform,schedule,tuple(TensorBoundary(v.id,v.id,32) for v in values),
        (NativeEntry('finalize',library,'vc_laser_sparse_finalize.finalize','vc_sparse_finalize_fp16'),),tuple(sources),
        'post-TopK shared ABI only; one target_validation invocation; geometry constant pinned separately; private UB/profiler-call addresses and whole attention quality unproved',device)
    result.verify();return result


def finalizer_binding(finalizer):
    # Read the called OpOverload schema, not a field absent from cropped evidence.
    name=finalizer.native._schema.name
    if name!='vc_laser_sparse_finalize::finalize':raise ValueError('Called native schema differs')
    return dict(library=finalizer.evidence['library_path'],sha256=finalizer.evidence['library_sha256'],
        symbol=name.replace('::','.'),kernel='vc_sparse_finalize_fp16')


def verify_native_declaration(plan,manifest):
    if len(plan.entries)!=1:raise ValueError('Native entry coverage differs')
    entry=plan.entries[0]
    expected=('finalize',manifest['torch_api'].removeprefix('torch.ops.'),'vc_sparse_finalize_fp16',str(Path(manifest['library_path']).resolve()),manifest['library_sha256'])
    if (entry.event,entry.symbol,entry.kernel,entry.library.path,entry.library.sha256)!=expected:raise ValueError('Native entry differs from the pinned finalizer declaration')
    return True


def attest_boundary(plan,metadata,mapped):
    if not metadata or set(metadata['tensors'])!={b.tensor for b in plan.boundaries}|{'geometry'}:raise ValueError('Incomplete native ABI sample')
    types={'torch.float16':'f16','torch.float32':'f32','torch.int32':'i32','torch.int64':'i64'};tensors=[]
    for b in plan.boundaries:
        x=metadata['tensors'][b.tensor]
        tensors.append(TensorObservation(b.tensor,tuple(x['shape']),types.get(x['dtype'],'unsupported'),tuple(x['strides_bytes']),
            x['data_address'],x['storage_address'],x['storage_bytes'],x['storage_offset_bytes'],x['device']))
    geometry=metadata['tensors']['geometry']
    if geometry['shape']!=[1056] or geometry['dtype']!='torch.int32' or geometry['strides_bytes']!=[4] or geometry['device']!=plan.device or not metadata['geometry_sha256']:
        raise ValueError('Native geometry ABI mismatch')
    native=metadata.get('native',{})
    if len(plan.entries)!=1:raise ValueError('Native entry coverage differs')
    libraries=[]
    for entry in plan.entries:
        if (entry.event,entry.symbol,entry.kernel,entry.library.path,entry.library.sha256)!=('finalize',native.get('symbol'),native.get('kernel'),native.get('library'),native.get('sha256')):
            raise ValueError('Native entry does not match the actual finalizer')
        if mapped.get(entry.library.path)!=entry.library.sha256:raise ValueError('Mapped native binary differs')
        if entry.library not in libraries:libraries.append(entry.library)
    return instantiate(plan,tuple(tensors),tuple(libraries),'target_validation')


# 来源: original E94/E97 complete captures, cross-checked with submitted harness.
# Order identifies logical call groups, not hardware-observed host launch IDs.
CAST='aclnnCast_CastAiCore_Cast';TRANSPOSE='aclnnInplaceCopy_TransposeAiCore_Transpose'
COMMON=('aclnnInplaceFillScalar_FillAiCore_Fill','vc_sparse_candidates_score_chunk_fp16',TRANSPOSE,'aclnnTopk_TopkV2AiCore_TopKV2','aclnnTopk_CastAiCore_Cast')
MACROS=(('map',(CAST,'aclnnMuls_MulAiCore_Mul','aclnnAdd_AddAiCore_Add','aclnnIndexSelect_GatherV3AiCore_GatherV3')),
    ('normalize',(CAST,'aclnnSoftmax_SoftmaxAiCore_SoftmaxV2',CAST)),
    ('packet',('aclnnConstantPadNd_PadV3AiCore_MemSet','aclnnConstantPadNd_PadV3AiCore_PadV3',TRANSPOSE,'aclnnConstantPadNd_PadV3AiCore_MemSet','aclnnConstantPadNd_PadV3AiCore_PadV3',TRANSPOSE)),('pv',('vc_sparse_pv_fp16',)))
AI_TYPES={'MIX_AIC','MIX_AIV','AI_VECTOR_CORE','AI_CORE'}
CONTROL_TYPES={'EVENT RECORD','PROFILING_ENABLE','PROFILING_DISABLE'}


def partition_kernels(kernels,variant,calls):
    if variant=='candidate':middle=(('finalize',('vc_sparse_finalize_fp16',)),)
    elif variant=='previous':middle=MACROS
    else:raise ValueError('Unknown architecture variant')
    pattern=COMMON+tuple(name for _,names in middle for name in names)+(CAST,)
    if type(calls) is not int or calls<=0 or tuple(k['name'] for k in kernels)!=pattern*calls:raise ValueError('Complete per-call kernel pattern differs; cannot infer invocation grouping')
    if len({k['stream'] for k in kernels})!=1:raise ValueError('Multi-stream grouping requires explicit launch correlation')
    groups=[]
    for call in range(calls):
        chunk=kernels[call*len(pattern):(call+1)*len(pattern)];offset=len(COMMON);events={}
        for event,names in middle:
            subset=chunk[offset:offset+len(names)];offset+=len(names)
            events[event]=dict(tasks=[x['event'] for x in subset],task_sum_ns=sum(x['duration_ns'] for x in subset),start_ns=subset[0]['start_ns'],end_ns=subset[-1]['start_ns']+subset[-1]['duration_ns'])
        groups.append(dict(logical_invocation=call,events=events,context_tasks=[x['event'] for x in chunk[:len(COMMON)]+chunk[-1:]],
            scope='complete ordered capture pattern and sequential harness; not directly observed launch IDs'))
    return groups


def audit_capture(spec,variant,device,output):
    output=Path(output);output.mkdir()
    if not spec['capture_finalized']:raise ValueError('Unfinalized capture')
    if len(spec['trace_files'])!=1 or len(spec['kernel_table_files'])!=1:raise ValueError('Ambiguous capture file selection')
    trace=Path(spec['trace_files'][0]);raw=json.loads(trace.read_text());rows=raw['traceEvents'] if isinstance(raw,dict) else raw
    pids={r['pid'] for r in rows if r.get('ph')=='M' and r.get('name')=='process_name' and r.get('args',{}).get('name')=='Ascend Hardware'}
    devices={r['pid']:str(r.get('args',{}).get('labels','')) for r in rows if r.get('ph')=='M' and r.get('name')=='process_labels'}
    if not pids or any(devices.get(pid)!='NPU '+str(device) for pid in pids):raise ValueError('Raw trace physical device metadata differs')
    stamps=[]
    for r in rows:
        if type(r.get('ts')) in (str,int,float):
            try:t=Decimal(str(r['ts']))*1000
            except InvalidOperation:continue
            if t.is_finite():stamps.append(t)
    origin=int(min(stamps).to_integral_value(rounding='ROUND_FLOOR'))
    plan=make_plan(trace,identity=variant+'-capture',clock='cann-export',time_encoding='decimal_ns',origin_ns=origin)
    receipt=ingest(plan);receipt.verify();(output/'plan.json').write_text(plan.to_json(pretty=True));(output/'receipt.json').write_text(receipt.to_json(pretty=True))
    kernels=[];controls=[];flows=[]
    for event in receipt.observation.events:
        attrs={x.name:x.value for x in event.attributes};begin=json.loads(attrs['trace.begin']);args=begin.get('args',{});task=args.get('Task Type')
        if begin.get('pid') not in pids:continue
        if attrs['trace.phase']!='X':
            if attrs['trace.phase']=='FLOW' and task is None:
                flows.append(dict(event=event.id,record=event.record,phase=attrs['trace.phase'],name=event.name));continue
            raise ValueError('Unsupported hardware event phase')
        row=dict(event=event.id,name=event.name,task=args.get('Task Id'),stream=str(args.get('Physic Stream Id')),start_ns=event.start_ns,duration_ns=event.end_ns-event.start_ns,record=event.record)
        if task in AI_TYPES:kernels.append(row)
        elif task in CONTROL_TYPES:controls.append(row)
        else:raise ValueError('Unknown hardware task requires classification: '+str(task))
    kernels.sort(key=lambda x:(x['start_ns'],x['event']))
    csvfile=identify(spec['kernel_table_files'][0],'input')
    with Path(csvfile.path).open() as stream:table=list(csv.DictReader(stream))
    matched=set()
    for row in table:
        matches=[k for k in kernels if str(k['task'])==row['Task ID'] and k['stream']==row['Stream ID'] and k['name']==row['Name']]
        if len(matches)!=1 or matches[0]['event'] in matched:raise ValueError('Ambiguous/duplicate CSV-to-trace kernel')
        k=matches[0];matched.add(k['event'])
        start=Decimal(row['Start Time(us)'].strip())*1000-origin;duration=Decimal(row['Duration(us)'].strip())*1000
        if row['Device_id']!=str(device) or abs(Decimal(k['start_ns'])-start)>1 or abs(Decimal(k['duration_ns'])-duration)>1:raise ValueError('CSV-to-trace device/timing mismatch')
    csvfile.verify()
    if len(matched)!=len(kernels):raise ValueError('Unaccounted AI task')
    groups=partition_kernels(kernels,variant,spec['submitted_calls'])
    # Every input row remains in TraceReceipt metadata/normalized/unsupported.
    a=receipt.accounting;indices=[x.index for x in a.metadata]+[i for x in a.normalized for i in x.records]+[x.index for x in a.unsupported]
    if sorted(indices)!=list(range(len(rows))):raise ValueError('Raw trace row accounting differs')
    result=dict(variant=variant,device=device,origin_ns=origin,trace_plan=plan.fingerprint(),csv=asdict(csvfile),raw_records=len(rows),
        metadata=len(a.metadata),normalized=len(a.normalized),unsupported=len(a.unsupported),ai_tasks=kernels,hardware_controls=controls,hardware_flows=flows,groups=groups,
        limitations='One capture; graph macros exclude six context kernels per call. Source-free export lacks per-pipe causal edges/private addresses; no ideal hardware schedule proof.')
    (output/'mapping.json').write_text(json.dumps(result,indent=2));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--card',type=int,required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    plan=build_case(args.project,args.card);path=Path(args.output)
    with path.open('x') as stream:stream.write(plan.to_json(pretty=True)+'\n')
    print(json.dumps(dict(lowering=str(path.resolve()),model=plan.architecture.fingerprint(),events=len(plan.architecture.events),scope=plan.scope),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
