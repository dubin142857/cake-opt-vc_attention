"""P2 source-free Chrome/CANN trace ingestion and independently scoped checks.

Only explicit, hash-bound trace/report inputs are opened. No source paths in
trace arguments are followed; no launch or data address is inferred from gaps,
program counters or arbitrary register values.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,replace
from decimal import Decimal
import hashlib
import math
from pathlib import Path
import re
from typing import Literal
from .._refs import Ref
from .architecture import (Attribute,Evidence,Record,VersionedRecord,_dump,_typed,parse_json)
from .observation import (Artifact,Provenance,ObservedAccess,ObservedEvent,ObservedRelation,
                          Measurement,Coverage,ObservedExecution,validate_observation)

# Format contract, not a hardware calibration. Chrome ts/dur are microseconds;
# displayTimeUnit controls display. Explicit ns input is an adapter override.
# https://perfetto.dev/docs/getting-started/other-formats#chrome-json-format
ADAPTER_VERSION='1'


@dataclass(frozen=True)
class TracePlan(VersionedRecord):
    id: str
    trace: Artifact
    timestamp_unit: Literal['us','ns']
    clock: str
    sim_report: Artifact | None = None
    single_launch: str | None = None
    cores_expected: tuple[str,...] | None = None
    relative_tolerance: int | float | None = None
    absolute_tolerance_ns: int | float | None = None
    comparison_layers: tuple[Literal['X','BE','FLOW'],...] = ('X',)
    time_encoding: Literal['number','decimal_ns'] = 'number'
    origin_ns: int = 0
    schema_version: int = 1


@dataclass(frozen=True)
class InputRead(Record):
    artifact: str
    location: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RawRecord(Record):
    index: int
    json: str


@dataclass(frozen=True)
class UnsupportedRecord(Record):
    index: int
    phase: str
    reason: str
    json: str


@dataclass(frozen=True)
class RecordGroup(Record):
    event: str
    records: tuple[int,...]


@dataclass(frozen=True)
class Accounting(Record):
    raw_records: int
    metadata: tuple[RawRecord,...]
    normalized: tuple[RecordGroup,...]
    unsupported: tuple[UnsupportedRecord,...]
    input_reads: tuple[InputRead,...]
    trace_header: str
    notes: tuple[str,...]


@dataclass(frozen=True)
class Comparison(Record):
    core: str
    launch: str | None
    clock: str
    observed_span_ns: int | float | None
    reported_span_ns: int | float | None
    difference_ns: int | float | None
    relative_error: int | float | None
    allowed_error_ns: int | float | None
    status: Literal['matched','mismatch','compared','unscoped','missing_trace','missing_report','clock_mismatch']


@dataclass(frozen=True)
class TraceReceipt(VersionedRecord):
    id: str
    plan: TracePlan
    observation: ObservedExecution
    accounting: Accounting
    comparisons: tuple[Comparison,...]
    status: Literal['parsed_conditionally'] = 'parsed_conditionally'
    schema_version: int = 1

    def verify(self):
        expected=ingest(self.plan)
        if expected!=self:
            raise ValueError('Receipt differs from deterministic replay of its bound inputs')
        return True


@dataclass(frozen=True)
class SyncAnnotation(Record):
    action: Literal['signal','wait','barrier','none']
    channel: str
    generation: str
    scope: str


def _number(value,label):
    if type(value) not in (int,float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(label+' requires a finite number')
    return value


def _time(value,unit,label,plan=None):
    if plan is not None and plan.time_encoding=='decimal_ns':
        if type(value) not in (int,float,str) or (type(value) is str and not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',value)):
            raise ValueError(label+' requires a finite decimal timestamp')
        number=Decimal(str(value))
        if not number.is_finite():raise ValueError(label+' requires a finite decimal timestamp')
        point=number*(1000 if unit=='us' else 1)
        if label!='dur':point-=plan.origin_ns
        # 本仓自定: this codec preserves integral signed-64 nanoseconds; it never
        # rounds sub-nanosecond values or silently converts a huge epoch to float.
        if point!=point.to_integral_value() or abs(point)>((1<<63)-1):
            raise ValueError(label+' is not an exact signed-64 nanosecond value after rebasing')
        return int(point)
    value=_number(value,label)
    result=float(Decimal(str(value))*(1000 if unit=='us' else 1))
    if not math.isfinite(result):
        raise ValueError(label+' overflows nanoseconds')
    return result


def _read_bound(artifact,role):
    if artifact.kind!=role or not Path(artifact.location).is_absolute() or not re.fullmatch('[0-9a-f]{64}',artifact.sha256) or artifact.size_bytes<0:
        raise ValueError('Input role/absolute path/hash/size is invalid: '+artifact.id)
    # At most declared bytes + one: a concurrently growing file cannot expand the
    # read beyond the pinned scope. Reading only once also binds parsed bytes.
    with Path(artifact.location).open('rb') as stream:
        raw=stream.read(artifact.size_bytes+1)
    digest=hashlib.sha256(raw).hexdigest()
    if len(raw)!=artifact.size_bytes or digest!=artifact.sha256:
        raise ValueError('Input identity changed: '+artifact.id)
    read=InputRead(artifact.id,str(Path(artifact.location).resolve()),digest,len(raw))
    return raw,read


def _pin(path,role,identity):
    path=Path(path).resolve();size=path.stat().st_size
    with path.open('rb') as stream:
        raw=stream.read(size+1)
    if len(raw)!=size:
        raise ValueError('Input changed while pinning: '+str(path))
    return Artifact(identity,role,str(path),hashlib.sha256(raw).hexdigest(),size)


def make_plan(trace,sim_report=None,*,identity='capture',timestamp_unit='us',clock='trace_time',
              single_launch=None,cores_expected=None,relative_tolerance=None,absolute_tolerance_ns=None,time_encoding='number',origin_ns=0):
    return TracePlan(identity,_pin(trace,'trace','trace'),timestamp_unit,clock,
                     _pin(sim_report,'sim_report','sim_report') if sim_report is not None else None,
                     single_launch,cores_expected,relative_tolerance,absolute_tolerance_ns,time_encoding=time_encoding,origin_ns=origin_ns)


def _scope(row,plan):
    args=row.get('args',{})
    if type(args) is not dict:
        raise ValueError('args must be an object')
    launch=args.get('launch_id')
    if launch is not None and (type(launch) is not str or not launch):
        raise ValueError('launch_id must be a nonempty string')
    if launch is not None and plan.single_launch is not None and launch!=plan.single_launch:
        raise ValueError('Per-event launch contradicts declared single_launch')
    return launch if launch is not None else plan.single_launch


def _event(begin,end,indices,plan):
    for key in ('pid','tid','name'):
        if type(begin.get(key)) not in (int,str) or not str(begin[key]):
            raise ValueError('Missing/invalid '+key)
    start=_time(begin.get('ts'),plan.timestamp_unit,'ts',plan)
    launch=_scope(begin,plan)
    flow=end is None and begin.get('ph') in ('s','t','f')
    if flow:
        duration=0;finish=start
    elif end is None:
        duration=_time(begin.get('dur'),plan.timestamp_unit,'dur',plan)
        if duration<0:
            raise ValueError('Negative duration')
        finish=(start+duration) if plan.time_encoding=='decimal_ns' else float(Decimal(str(begin['ts']))*(1000 if plan.timestamp_unit=='us' else 1)
                     +Decimal(str(begin['dur']))*(1000 if plan.timestamp_unit=='us' else 1))
    else:
        finish=_time(end.get('ts'),plan.timestamp_unit,'end ts',plan)
        duration=(finish-start) if plan.time_encoding=='decimal_ns' else Decimal(str(end['ts']))-Decimal(str(begin['ts']))
        if duration<0:
            raise ValueError('End precedes begin')
        end_launch=_scope(end,plan)
        if end_launch is not None and launch is not None and end_launch!=launch:
            raise ValueError('B/E launch scopes differ')
        launch=launch if launch is not None else end_launch
    if not math.isfinite(finish) or (duration>0 and finish<=start):
        raise ValueError('Timestamp precision cannot represent this positive interval')
    if plan.time_encoding=='decimal_ns' and abs(finish)>((1<<63)-1):raise ValueError('End timestamp exceeds signed-64 nanosecond codec')
    args=begin.get('args',{});attributes=[Attribute('trace.phase','FLOW' if flow else ('X' if end is None else 'BE')),
        Attribute('trace.begin',_dump(begin))]
    if plan.time_encoding!='number':attributes.extend((Attribute('trace.time_encoding',plan.time_encoding),Attribute('trace.origin_ns',plan.origin_ns)))
    if end is not None:
        attributes.append(Attribute('trace.end',_dump(end)))
    for key in ('detail','pc_addr','code'):
        if key in args:
            # Retained as metadata, never interpreted as another file to open.
            attributes.append(Attribute('native.'+key,_dump(args[key])))
    accesses=();memory='unknown'
    if 'opgen_accesses' in args:
        accesses=_typed(tuple[ObservedAccess,...],args['opgen_accesses'],'opgen_accesses',json_input=True)
        memory='access' if accesses else 'none'
    sync=None;sync_kind='unknown';name=str(begin['name'])
    if 'opgen_sync' in args:
        sync=_typed(SyncAnnotation,args['opgen_sync'],'opgen_sync',json_input=True)
        if sync.action in ('signal','wait') and not all((sync.channel,sync.generation,sync.scope)):
            raise ValueError('Explicit synchronization annotation needs channel/generation/scope')
        sync_kind=sync.action
    elif name in ('SET_FLAG','SET_FLAG_DEVI'):
        sync_kind='signal'
    elif name in ('WAIT_FLAG','WAIT_FLAG_DEVI'):
        sync_kind='wait'
    elif name=='BAR':
        sync_kind='barrier'
    identity=plan.trace.id+':'+':'.join(str(x) for x in indices)
    event=ObservedEvent(identity,str(begin['pid']),str(begin['tid']),plan.clock,start,finish,
                        name,plan.trace.id,'traceEvents['+','.join(str(x) for x in indices)+']',
                        launch,accesses=accesses,memory_kind=memory,sync_kind=sync_kind,
                        attributes=tuple(attributes))
    return event,sync


def _normalize(raw,plan):
    if type(raw) is not dict or type(raw.get('traceEvents')) is not list:
        raise ValueError('Expected a Chrome trace object with traceEvents; aggregate CSV is not an execution timeline')
    metadata=[];unsupported=[];groups=[];events=[];annotations={};identified=defaultdict(list);unidentified=defaultdict(list)
    notes=[]
    def reject(index,row,reason):
        phase=str(row.get('ph','unknown')) if type(row) is dict else 'unknown'
        unsupported.append(UnsupportedRecord(index,phase,reason,_dump(row)))
    def accept(begin,end,indices):
        try:
            event,sync=_event(begin,end,indices,plan)
        except (ValueError,TypeError) as exc:
            for index,row in zip(indices,(begin,) if end is None else (begin,end)):
                reject(index,row,str(exc))
            return
        events.append(event);groups.append(RecordGroup(event.id,indices))
        if sync is not None:
            annotations[event.id]=sync
    for index,row in enumerate(raw['traceEvents']):
        if type(row) is not dict:
            reject(index,row,'Record is not an object');continue
        phase=row.get('ph')
        if phase=='M':
            metadata.append(RawRecord(index,_dump(row)))
        elif phase=='X':
            accept(row,None,(index,))
        elif phase in ('s','t','f'):
            if type(row.get('id')) not in (int,str):
                reject(index,row,'Flow endpoint needs explicit correlation id')
            else:
                accept(row,None,(index,))
        elif phase in ('B','E'):
            try:
                _time(row.get('ts'),plan.timestamp_unit,'B/E timestamp',plan);launch=_scope(row,plan)
                if 'pid' not in row or 'tid' not in row:
                    raise ValueError('B/E record needs pid/tid')
                if type(row['pid']) not in (int,str) or type(row['tid']) not in (int,str):
                    raise ValueError('Invalid B/E pid/tid')
                if 'id' in row:
                    if type(row['id']) not in (int,str):
                        raise ValueError('Invalid B/E correlation id')
                    identified[(str(row['pid']),str(row['tid']),_dump(row['id']))].append((index,row))
                else:
                    unidentified[(str(row['pid']),str(row['tid']))].append((index,row))
            except (TypeError,ValueError) as exc:
                reject(index,row,str(exc))
        else:
            reject(index,row,'Unsupported execution phase')
    for rows in identified.values():
        begins=sorted(((i,r) for i,r in rows if r['ph']=='B'),key=lambda x:(_time(x[1]['ts'],plan.timestamp_unit,'ts',plan),x[0]))
        ends=sorted(((i,r) for i,r in rows if r['ph']=='E'),key=lambda x:(_time(x[1]['ts'],plan.timestamp_unit,'ts',plan),x[0]))
        pairs=list(zip(begins,ends))
        unambiguous=(len(begins)==len(ends) and all(_time(b['ts'],plan.timestamp_unit,'ts',plan)<=_time(e['ts'],plan.timestamp_unit,'ts',plan) and (not e.get('name') or e['name']==b.get('name')) for (_,b),(_,e) in pairs)
                     and all(_time(pairs[i][1][1]['ts'],plan.timestamp_unit,'ts',plan)<=_time(pairs[i+1][0][1]['ts'],plan.timestamp_unit,'ts',plan) for i in range(len(pairs)-1)))
        if unambiguous:
            for (bi,b),(ei,e) in pairs:
                accept(b,e,(bi,ei))
        else:
            for index,row in rows:
                reject(index,row,'Unpaired, reversed or ambiguous correlated B/E records')
    for rows in unidentified.values():
        stack=[]
        for index,row in sorted(rows,key=lambda x:(_time(x[1]['ts'],plan.timestamp_unit,'ts',plan),x[0])):
            if row['ph']=='B':
                stack.append((index,row))
            elif stack and (not row.get('name') or row['name']==stack[-1][1].get('name')):
                bi,b=stack.pop();accept(b,row,(bi,index))
            else:
                reject(index,row,'Unmatched or non-nested end record')
        for index,row in stack:
            reject(index,row,'Unmatched begin record')
    covered=[x.index for x in metadata]+[i for g in groups for i in g.records]+[x.index for x in unsupported]
    if sorted(covered)!=list(range(len(raw['traceEvents']))):
        raise ValueError('Internal accounting lost or duplicated source records')
    if not events:
        raise ValueError('No supported execution intervals; unsupported records='+str(len(unsupported)))
    events.sort(key=lambda x:(x.start_ns,x.end_ns,x.core,x.pipe,x.id));groups.sort(key=lambda x:x.event)
    unsupported.sort(key=lambda x:x.index)
    # Observed adjacency, split by scope and visual layer; never required dataflow.
    relations=[];pipes=defaultdict(list)
    for event in events:
        layer=next(x.value for x in event.attributes if x.name=='trace.phase')
        pipes[(event.clock,event.core,event.pipe,event.launch,layer)].append(event)
    for sequence in pipes.values():
        for left,right in zip(sequence,sequence[1:]):
            if left.end_ns<=right.start_ns:
                relations.append(ObservedRelation(left.id,right.id,'observed_order',
                    Evidence('derived',plan.trace.sha256,'observed intervals','Temporal adjacency only; not a required dependency')))
    # Native flow IDs identify drawing connections. Without an explicit global
    # namespace, match within one pid/core only; cross-core meaning stays unknown.
    flows=defaultdict(list)
    for event in events:
        raw_event=parse_json(next(x.value for x in event.attributes if x.name=='trace.begin'))
        if raw_event.get('ph') in ('s','t','f'):
            key=(event.core,event.clock,event.launch,_dump(raw_event.get('cat')),_dump(raw_event['id']))
            flows[key].append((event,raw_event['ph']))
    for sequence in flows.values():
        sequence.sort(key=lambda x:(x[0].start_ns,x[0].id))
        if sum(phase=='s' for _,phase in sequence)!=1 or sequence[0][1]!='s':
            notes.append('Unmatched/ambiguous flow namespace: '+sequence[0][0].id)
            continue
        previous=None;closed=False
        for current,phase in sequence:
            if closed:
                notes.append('Flow endpoint after finish: '+current.id);continue
            if previous is not None:
                relations.append(ObservedRelation(previous.id,current.id,'synchronization_candidate',
                    Evidence('derived',plan.trace.sha256,'explicit trace flow','Drawing connection only; not hardware dataflow or visibility proof')))
            previous=current
            if phase=='f':closed=True
    signals=defaultdict(list)
    by_id={e.id:e for e in events}
    for eid,annotation in annotations.items():
        if annotation.action=='signal':
            event=by_id[eid]
            signals[(annotation.channel,annotation.generation,annotation.scope,event.launch,event.clock)].append(event)
    for eid,annotation in annotations.items():
        if annotation.action!='wait':continue
        wait=by_id[eid];key=(annotation.channel,annotation.generation,annotation.scope,wait.launch,wait.clock)
        candidates=[e for e in signals[key] if e.end_ns<=wait.end_ns]
        if len(candidates)==1:
            relations.append(ObservedRelation(candidates[0].id,eid,'synchronization_candidate',
                Evidence('assumed',plan.trace.sha256,'explicit annotation','Channel/generation match requires hardware semantics and visibility verification')))
        else:
            notes.append('Unmatched/ambiguous synchronization annotation: '+eid)
    return tuple(events),tuple(relations),tuple(metadata),tuple(groups),tuple(unsupported),tuple(notes)


def _report(raw,plan):
    if type(raw) is not dict:
        raise ValueError('Simulator report must be an object')
    if raw.get('format')=='opgen.report.v1':
        rows=raw.get('rows')
        if type(rows) is not list:
            raise ValueError('Versioned report requires rows')
    else:
        rows=[]
        for core,values in raw.items():
            if type(values) is not dict:
                raise ValueError('Legacy report core row must be an object')
            rows.append(dict(values,core=core,launch=plan.single_launch,clock=plan.clock))
    measurements=[];seen=set()
    for index,row in enumerate(rows):
        if type(row) is not dict or type(row.get('core')) is not str or not row['core']:
            raise ValueError('Report row needs a core')
        launch=row.get('launch');clock=row.get('clock',plan.clock)
        if (launch is not None and (type(launch) is not str or not launch)) or type(clock) is not str or not clock:
            raise ValueError('Invalid report launch/clock')
        key=(row['core'],launch,clock)
        if key in seen:
            raise ValueError('Duplicate simulator report scope')
        seen.add(key)
        for field,metric in (('duration_us','span'),('running_us','running')):
            value=row.get(field)
            if value is not None:
                value=_time(value,'us',field)
                if value<0:raise ValueError('Negative reported duration')
            measurements.append(Measurement('report:'+str(index)+':'+metric,row['core'],metric,'ns',value,
                plan.sim_report.id,Evidence('measured',plan.sim_report.sha256,'simulator self-report','Input identity checked; reporting semantics still scoped by caller'),clock,launch))
    return tuple(measurements)


def reconcile(observation,plan):
    ranges={}
    for event in observation.events:
        layer=next(x.value for x in event.attributes if x.name=='trace.phase')
        if layer not in plan.comparison_layers:
            continue
        key=(event.core,event.launch,event.clock)
        if key in ranges:
            lo,hi=ranges[key];ranges[key]=(min(lo,event.start_ns),max(hi,event.end_ns))
        else:ranges[key]=(event.start_ns,event.end_ns)
    reports={(m.subject,m.launch,m.clock):m.value for m in observation.measurements if m.metric=='span'}
    comparisons=[]
    for key in sorted(set(ranges)|set(reports),key=lambda x:tuple('' if v is None else v for v in x)):
        core,launch,clock=key;span=None
        if key in ranges:
            lo,hi=ranges[key];span=hi-lo
        reference=reports.get(key);difference=None;relative=None;allowed=None
        if launch is None:
            status='unscoped'
        elif key not in ranges:
            status='clock_mismatch' if any(k[:2]==key[:2] for k in ranges) else 'missing_trace'
        elif reference is None:
            status='clock_mismatch' if any(k[:2]==key[:2] for k in reports) else 'missing_report'
        else:
            difference=abs(span-reference)
            relative=difference/reference if reference else (0.0 if difference==0 else None)
            if plan.relative_tolerance is None and plan.absolute_tolerance_ns is None:
                status='compared'
            else:
                allowed=(plan.absolute_tolerance_ns if plan.absolute_tolerance_ns is not None else 0)+(plan.relative_tolerance if plan.relative_tolerance is not None else 0)*reference
                status='matched' if difference<=allowed else 'mismatch'
        comparisons.append(Comparison(core,launch,clock,span,reference,difference,relative,allowed,status))
    return tuple(comparisons)


def ingest(plan: TracePlan):
    if plan.schema_version!=1 or not plan.id or not plan.clock or plan.single_launch=='':
        raise ValueError('Invalid plan identity/version/clock/launch')
    if plan.origin_ns and plan.time_encoding!='decimal_ns':raise ValueError('Explicit origin requires decimal_ns encoding')
    if not plan.comparison_layers or len(set(plan.comparison_layers))!=len(plan.comparison_layers):
        raise ValueError('Comparison layers must be explicit and unique')
    for value in (plan.relative_tolerance,plan.absolute_tolerance_ns):
        if value is not None and value<0:raise ValueError('Negative comparison tolerance')
    if plan.sim_report is not None and plan.sim_report.id==plan.trace.id:
        raise ValueError('Trace and report artifact identities must differ')
    trace,read=_read_bound(plan.trace,'trace');raw=parse_json(trace);array_source=type(raw) is list
    if array_source:raw={'traceEvents':raw}
    events,relations,metadata,groups,unsupported,notes=_normalize(raw,plan)
    if array_source:
        events=tuple(replace(e,record=e.record.removeprefix('traceEvents')) for e in events)
        notes=notes+('Source container was a top-level array; record indices refer directly to that array',)
    reads=[read];artifacts=[plan.trace];measurements=()
    if plan.sim_report is not None:
        report,read=_read_bound(plan.sim_report,'sim_report');reads.append(read);artifacts.append(plan.sim_report)
        measurements=_report(parse_json(report),plan)
    provenance_notes=('Only explicit trace/report inputs read; embedded code paths are metadata, not opened',
        'No launch inferred from timing gaps; single_launch, if supplied, is an explicit caller assumption',
        'X/BE/FLOW layers are separate; comparison_layers selects the report scope, never summed instruction work')
    if plan.time_encoding!='number':provenance_notes+=('Exact '+plan.time_encoding+' timestamps, origin_ns='+str(plan.origin_ns)+'; origin is declared, not a cross-clock calibration',)
    coverage=Coverage(len(events)+len(unsupported),len(events),len(unsupported),cores_expected=plan.cores_expected,
                      addresses='partial' if any(e.accesses for e in events) else 'unknown',synchronization='partial',
                      notes=('Counts use normalized intervals plus unsupported raw records; Accounting partitions all original rows',))
    observation=ObservedExecution(plan.id,events,relations,measurements,
        Provenance('opgen.trace','2' if array_source or plan.time_encoding!='number' else ADAPTER_VERSION,tuple(artifacts),notes=provenance_notes),coverage)
    errors=[x for x in validate_observation(observation).issues if x.severity=='error']
    if errors:raise ValueError('Invalid parsed observation: '+str(errors))
    accounting=Accounting(len(raw['traceEvents']),metadata,groups,unsupported,tuple(reads),
        _dump({'source_container':'array'} if array_source else {key:value for key,value in raw.items() if key!='traceEvents'}),notes)
    return TraceReceipt(plan.id,plan,observation,accounting,reconcile(observation,plan))


def cli(args):
    import sys
    try:
        if args.transform:
            raise ValueError('Trace actions cannot apply architecture transforms')
        overrides=any(x is not None for x in (args.sim_report,args.trace_unit,args.trace_clock,args.single_launch,args.relative_tolerance,args.absolute_tolerance_ns,args.trace_time_encoding,args.trace_origin_ns)) or bool(args.expected_core)
        if (args.verify_trace or args.trace_plan) and overrides:
            raise ValueError('Bound plans/receipts cannot be overridden; create a new explicit plan')
        if args.verify_trace:
            receipt=TraceReceipt.from_json(Path(args.verify_trace).read_text());receipt.verify()
            text=_dump(dict(status='replayed_inputs',observation_fingerprint=receipt.observation.fingerprint(),
                             scope='Bound input consistency, not algorithm or hardware certification'),pretty=True);code=0
        else:
            if args.trace_plan:
                plan=TracePlan.from_json(Path(args.trace_plan).read_text())
            else:
                plan=make_plan(args.trace,args.sim_report,
                    timestamp_unit='us' if args.trace_unit is None else args.trace_unit,
                    clock='trace_time' if args.trace_clock is None else args.trace_clock,
                    single_launch=args.single_launch,cores_expected=tuple(args.expected_core) if args.expected_core else None,
                    relative_tolerance=args.relative_tolerance,absolute_tolerance_ns=args.absolute_tolerance_ns,
                    time_encoding=args.trace_time_encoding or 'number',origin_ns=args.trace_origin_ns or 0)
            receipt=ingest(plan);text=receipt.to_json(pretty=True)
            code=1 if any(x.status=='mismatch' for x in receipt.comparisons) else 0
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return code
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr);return 2


def _nc_read_boundary():
    artifact=Artifact('src','kernel_source','/not/opened','0'*64,0)
    try:
        _read_bound(artifact,'trace')
        return False
    except ValueError:
        return True


REFS={'TRACE1_bound_inputs_and_scope':Ref(
    what=['仅读取显式hash绑定trace/report；X/B/E及缺口全量记账，单位/launch/clock有界对账，不读嵌入源码路径。'],
    status='derived',evidence=['E102 trace_test real-format and synthetic counterexamples; Perfetto Chrome format specification; archived simulator report comparisons.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'TRACE1_bound_inputs_and_scope':_nc_read_boundary}
