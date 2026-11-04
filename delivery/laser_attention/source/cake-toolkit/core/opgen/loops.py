"""P3 finite template expansion with explicit views, boundary tokens and IO.

This is a conditional mapping of a supplied global semantic program, not an
automatic proof of arbitrary loop recurrence or state-history mathematics.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from collections import defaultdict
import math
from typing import Literal
from .._refs import Ref
from .architecture import (Architecture,Binding,Dependency,Duration,Evidence,Event,
    Layout,Record,Region,VersionedRecord,_dag,layout_span_bytes,validate)
from .schedule import SchedulePlan,ScheduleReceipt,solve


@dataclass(frozen=True)
class EdgeKey(Record):
    source: str
    target: str
    kind: Literal['data','program','synchronization','reuse']
    distance: int

    @classmethod
    def of(cls,edge):
        return cls(edge.source,edge.target,edge.kind,edge.iteration_distance)


@dataclass(frozen=True)
class Domain(Record):
    event: str
    start: int = 0
    stop_offset: int = 0


@dataclass(frozen=True)
class EdgeDomain(Record):
    edge: EdgeKey
    start: int = 0
    stop_offset: int = 0


@dataclass(frozen=True)
class SlotMap(Record):
    allocation: str
    count: int


@dataclass(frozen=True)
class ViewMap(Record):
    instance: str
    origin: tuple[int,...]
    step: tuple[int,...]
    offset_step_bytes: int = 0
    clip_tail: bool = False


@dataclass(frozen=True)
class AccessMap(Record):
    event: str
    mode: Literal['reads','writes']
    index: int
    iteration_shift: int = 0
    span: Literal['template','instance'] = 'template'


@dataclass(frozen=True)
class InterfaceRegion(Record):
    instance: str
    offset_bytes: int
    span_bytes: int
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Seed(Record):
    edge: EdgeKey
    target_iteration: int
    ready: bool
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class LoopPlan(VersionedRecord):
    source_fingerprint: str
    iterations: int
    slots: tuple[SlotMap,...] = ()
    views: tuple[ViewMap,...] = ()
    domains: tuple[Domain,...] = ()
    edge_domains: tuple[EdgeDomain,...] = ()
    accesses: tuple[AccessMap,...] = ()
    inputs: tuple[InterfaceRegion,...] = ()
    outputs: tuple[InterfaceRegion,...] = ()
    seeds: tuple[Seed,...] = ()
    cost_policy: Literal['invalidate','assume_unchanged'] = 'invalidate'
    require_complete_outputs: bool = True
    # 本仓自定: bounded finite expansion, not a claim about hardware feasibility.
    max_events: int = 2000
    max_objects: int = 10000
    schema_version: int = 1


@dataclass(frozen=True)
class EventRef(Record):
    event: str
    template: str
    iteration: int


@dataclass(frozen=True)
class InstanceRef(Record):
    instance: str
    template: str
    iteration: int | None


@dataclass(frozen=True)
class Boundary(Record):
    edge: EdgeKey
    target_iteration: int
    source_iteration: int
    status: Literal['seeded','missing','data_history_required']
    seed_event: str = ''


@dataclass(frozen=True)
class OutputCoverage(Record):
    value: str
    covered_elements: int | None
    total_elements: int
    status: Literal['complete','partial','unknown']


@dataclass(frozen=True)
class LoopReceipt(VersionedRecord):
    source: Architecture
    plan: LoopPlan
    status: Literal['expanded_conditionally','unknown']
    architecture: Architecture | None
    events: tuple[EventRef,...]
    instances: tuple[InstanceRef,...]
    boundaries: tuple[Boundary,...]
    exits: tuple[str,...]
    prologue: tuple[str,...]
    steady: tuple[str,...]
    epilogue: tuple[str,...]
    drain: tuple[str,...]
    output_coverage: tuple[OutputCoverage,...]
    conditions: tuple[str,...]
    schema_version: int = 1

    def verify(self):
        if expand(self.source,self.plan)!=self:
            raise ValueError('Loop receipt differs from deterministic expansion')
        return True


@dataclass(frozen=True)
class LoopScheduleReceipt(VersionedRecord):
    loop: LoopReceipt
    schedule: ScheduleReceipt | None
    status: Literal['scheduled_conditionally','unknown']
    upper_ns: int | None
    schema_version: int = 1

    def verify(self):
        self.loop.verify()
        if self.loop.architecture is None:
            if self.schedule is not None or self.status!='unknown' or self.upper_ns is not None:
                raise ValueError('Unexpanded loop cannot carry a scheduling witness')
        else:
            if self.schedule is None or self.schedule.architecture!=self.loop.architecture:
                raise ValueError('Schedule is not bound to the verified expanded architecture')
            self.schedule.verify()
            if self.status!=self.schedule.status or self.upper_ns!=self.schedule.upper_ns:
                raise ValueError('Pipeline summary differs from schedule')
        return True


def schedule_loop(loop:LoopReceipt,plan:SchedulePlan | None=None):
    loop.verify()
    if loop.architecture is None:
        if plan is not None:raise ValueError('A schedule plan requires a concrete verified loop expansion')
        return LoopScheduleReceipt(loop,None,'unknown',None)
    plan=SchedulePlan(loop.architecture.fingerprint()) if plan is None else plan
    scheduled=solve(loop.architecture,plan)
    return LoopScheduleReceipt(loop,scheduled,scheduled.status,scheduled.upper_ns)


class LoopError(ValueError):
    pass


class _Count:
    def __init__(self,limit):self.limit=limit;self.used=0
    def take(self,amount):
        if amount<0 or self.used+amount>self.limit:
            raise LoopError('Finite expansion object budget exceeded')
        self.used+=amount


def _id(name,iteration):
    return 'loop:'+str(iteration)+'|'+name


def _covered(start,end,intervals):
    upto=start
    for lo,hi in sorted(intervals):
        if upto>=end:return True
        if hi<=upto:continue
        if lo>upto:return False
        upto=max(upto,hi)
    return upto>=end


def _origin(instance,values):
    return instance.logical_origin or (0,)*len(values[instance.value].shape)


def _compatible(view,interface,values):
    if view.value!=interface.value or view.allocation!=interface.allocation:return False
    vo,io=_origin(view,values),_origin(interface,values)
    if len(vo)!=len(io):return False
    if any(v<i or v+n>i+d for v,n,i,d in zip(vo,view.layout.extents,io,interface.layout.extents)):return False
    if any(n>1 and a!=b for n,a,b in zip(view.layout.extents,view.layout.strides_bytes,interface.layout.strides_bytes)):return False
    expected=interface.offset_bytes+sum((v-i)*stride for v,i,stride in zip(vo,io,interface.layout.strides_bytes))
    return view.offset_bytes==expected


def _initialization(a,inputs,outputs):
    """Read bytes need an interface view or a preceding local producer.

    A different logical view writing the same slot must be ordered outside the
    reaching definition, not merely somewhere before the read.
    """
    instances={x.id:x for x in a.instances};values={x.id:x for x in a.semantic.values};events={x.id:x for x in a.events}
    _,ancestors=_dag(events,[(d.source,d.target) for d in a.dependencies])
    writes=[];issues=[]
    for event in a.events:
        for region in event.writes:
            if region.span_bytes:
                instance=instances[region.instance];start=instance.offset_bytes+region.offset_bytes
                writes.append((event.id,instance.id,instance.allocation,start,start+region.span_bytes))
    for event in a.events:
        for region in event.reads:
            if not region.span_bytes:continue
            instance=instances[region.instance];start=instance.offset_bytes+region.offset_bytes;end=start+region.span_bytes
            own=[w for w in writes if w[1]==instance.id and w[0] in ancestors[event.id]]
            initial=[(instances[r.instance].offset_bytes+r.offset_bytes,instances[r.instance].offset_bytes+r.offset_bytes+r.span_bytes)
                     for r in inputs if _compatible(instance,instances[r.instance],values)]
            intervals=initial+[(w[3],w[4]) for w in own]
            if not _covered(start,end,intervals):
                issues.append('Uninitialized or unmapped read: '+event.id+'/'+instance.id)
            for foreign in writes:
                eid,iid,allocation,lo,hi=foreign
                if allocation!=instance.allocation or iid==instance.id or max(lo,start)>=min(hi,end):continue
                if event.id in ancestors[eid]:continue
                killed=[(w[3],w[4]) for w in own if eid in ancestors[w[0]]]
                if not _covered(max(lo,start),min(hi,end),killed):
                    issues.append('Stale or unordered slot overwrite: '+eid+' -> '+event.id)
        for region in event.writes:
            instance=instances[region.instance]
            if instance.value not in a.semantic.outputs or not region.span_bytes:continue
            start=instance.offset_bytes+region.offset_bytes;end=start+region.span_bytes
            witnesses=[(instances[r.instance].offset_bytes+r.offset_bytes,instances[r.instance].offset_bytes+r.offset_bytes+r.span_bytes)
                       for r in outputs if _compatible(instance,instances[r.instance],values)]
            if not _covered(start,end,witnesses):
                issues.append('Output physical/logical interface mismatch: '+event.id+'/'+instance.id)
    return tuple(sorted(set(issues)))


def _logical_interval(instance,value):
    origin=instance.logical_origin or (0,)*len(value.shape);extents=instance.layout.extents
    # Only flatten truly contiguous logical rectangles, never fill their holes.
    changed=False
    for o,n,d in zip(origin,extents,value.shape):
        if changed and (o!=0 or n!=d):return None
        if n>1:changed=True
    offset=0
    for o,d in zip(origin,value.shape):offset=offset*d+o
    return offset,offset+math.prod(extents)


def _coverage(a):
    values={v.id:v for v in a.semantic.values};instances={i.id:i for i in a.instances};regions=defaultdict(list);unknown=set()
    for event in a.events:
        for region in event.writes:
            instance=instances[region.instance]
            if instance.value not in a.semantic.outputs or not region.span_bytes:continue
            span=layout_span_bytes(instance.layout,values[instance.value].dtype)
            if region.offset_bytes or region.span_bytes!=span:
                unknown.add(instance.value);continue
            interval=_logical_interval(instance,values[instance.value])
            if interval is None:unknown.add(instance.value)
            else:regions[instance.value].append(interval)
    result=[]
    for name in a.semantic.outputs:
        total=math.prod(values[name].shape)
        if name in unknown:
            result.append(OutputCoverage(name,None,total,'unknown'));continue
        end=0;count=0
        for lo,hi in sorted(regions[name]):
            if hi>end:count+=hi-max(lo,end);end=hi
        result.append(OutputCoverage(name,count,total,'complete' if count==total else 'partial'))
    return tuple(result)


CONDITIONS=(
    'Global semantics and event-to-tile computations are supplied declarations, not automatically proven loop mathematics',
    'Control seeds are explicit initial protocol assumptions; they never synthesize numerical history values',
    'Interface regions assert initialized input and public output bindings; actual addresses and data still require execution checks',
    'Repeated/tail costs are invalidated unless explicitly reused as assumptions; no new hardware measurement is created',
    'Drain includes finite event sinks and all resource tails; last output readiness alone is not a kernel completion proof',
)


def expand(source:Architecture,plan:LoopPlan):
    if plan.schema_version!=1 or plan.source_fingerprint!=source.fingerprint():raise LoopError('Loop plan source/version mismatch')
    if plan.iterations<1 or plan.max_events<0 or plan.max_objects<0:raise LoopError('Positive finite iterations and nonnegative budgets required')
    errors=[x for x in validate(source).issues if x.severity=='error']
    if errors:raise LoopError('Invalid template: '+str(errors))
    counter=_Count(plan.max_objects)
    values={v.id:v for v in source.semantic.values};templates={e.id:e for e in source.events}
    original_instances={i.id:i for i in source.instances};allocations={a.id:a for a in source.allocations}
    domains={e:(0,plan.iterations) for e in templates};seen=set()
    for domain in plan.domains:
        if domain.event not in templates or domain.event in seen:raise LoopError('Invalid/duplicate event domain')
        seen.add(domain.event);stop=plan.iterations+domain.stop_offset
        if stop<domain.start:raise LoopError('Reversed event domain')
        domains[domain.event]=(domain.start,stop)
    event_count=sum(stop-start for start,stop in domains.values())
    if event_count>plan.max_events:raise LoopError('Finite event expansion budget exceeded')
    counter.take(event_count)
    edge_keys={EdgeKey.of(d) for d in source.dependencies};edge_domains={};seen=set()
    for domain in plan.edge_domains:
        if domain.edge not in edge_keys or domain.edge in seen:raise LoopError('Invalid/duplicate edge domain')
        if plan.iterations+domain.stop_offset<domain.start:raise LoopError('Reversed edge domain')
        seen.add(domain.edge);edge_domains[domain.edge]=(domain.start,plan.iterations+domain.stop_offset)
    interface_ids=set();interface_allocations=set()
    for kind,regions,semantic in (('input',plan.inputs,source.semantic.inputs),('output',plan.outputs,source.semantic.outputs)):
        for region in regions:
            instance=original_instances.get(region.instance)
            if instance is None or instance.value not in semantic:raise LoopError('Invalid '+kind+' interface instance')
            span=layout_span_bytes(instance.layout,values[instance.value].dtype)
            if min(region.offset_bytes,region.span_bytes)<0 or region.offset_bytes+region.span_bytes>span:
                raise LoopError('Interface region exceeds its view')
            interface_ids.add(instance.id);interface_allocations.add(instance.allocation)
    slots={};seen=set();expanded_allocations=[]
    for mapping in plan.slots:
        if mapping.allocation not in allocations or mapping.allocation in seen or mapping.count<1 or mapping.allocation in interface_allocations:
            raise LoopError('Invalid/duplicate slot map or attempted public IO rotation')
        seen.add(mapping.allocation);slots[mapping.allocation]=mapping.count
    counter.take(sum(slots.get(a.id,1) for a in source.allocations))
    slot_ids={}
    for allocation in source.allocations:
        count=slots.get(allocation.id,1);names=[]
        for index in range(count):
            name=allocation.id if count==1 else 'slot:'+str(index)+'|'+allocation.id
            names.append(name);expanded_allocations.append(replace(allocation,id=name))
        slot_ids[allocation.id]=tuple(names)
    if len({a.id for a in expanded_allocations})!=len(expanded_allocations):raise LoopError('Expanded allocation identity collision')
    views={}
    for view in plan.views:
        if view.instance not in original_instances or view.instance in views or view.instance in interface_ids:
            raise LoopError('Invalid/duplicate affine view or changing a public interface view')
        rank=len(values[original_instances[view.instance].value].shape)
        if len(view.origin)!=rank or len(view.step)!=rank:raise LoopError('Affine view rank mismatch')
        views[view.instance]=view
    access_maps={}
    for mapping in plan.accesses:
        key=(mapping.event,mapping.mode,mapping.index)
        if mapping.event not in templates or key in access_maps or mapping.index<0 or mapping.index>=len(getattr(templates[mapping.event],mapping.mode)):
            raise LoopError('Invalid/duplicate access map')
        access_maps[key]=mapping
    seeds={}
    for seed in plan.seeds:
        key=(seed.edge,seed.target_iteration)
        if seed.edge not in edge_keys or key in seeds:raise LoopError('Invalid/duplicate boundary seed')
        seeds[key]=seed
    expanded_instances={};instance_refs=[];instance_keys={}
    for identity in sorted(interface_ids):
        expanded_instances[identity]=original_instances[identity];instance_keys[identity]=(identity,None)
        instance_refs.append(InstanceRef(identity,identity,None));counter.take(1)
    def instance_at(identity,iteration):
        if identity in interface_ids:return identity
        name=_id(identity,iteration)
        if name in expanded_instances:
            if instance_keys[name]!=(identity,iteration):
                raise LoopError('Expanded instance identity collision')
            return name
        template=original_instances[identity];view=views.get(identity)
        logical=template.logical_origin or (0,)*len(values[template.value].shape)
        extents=template.layout.extents;offset=template.offset_bytes
        if view is not None:
            logical=tuple(b+iteration*s for b,s in zip(view.origin,view.step));offset+=iteration*view.offset_step_bytes
            if view.clip_tail:extents=tuple(min(n,d-o) for n,d,o in zip(extents,values[template.value].shape,logical))
        if offset<0 or any(o<0 or n<0 or o+n>d for o,n,d in zip(logical,extents,values[template.value].shape)):
            raise LoopError('Affine view outside physical/logical bounds: '+name)
        slots_for=slot_ids[template.allocation];allocation=slots_for[iteration%len(slots_for)]
        instance=replace(template,id=name,allocation=allocation,offset_bytes=offset,
                         logical_origin=logical,layout=Layout(extents,template.layout.strides_bytes))
        if name in interface_ids:raise LoopError('Expanded instance collides with interface identity')
        expanded_instances[name]=instance;instance_keys[name]=(identity,iteration)
        instance_refs.append(InstanceRef(name,identity,iteration));counter.take(1)
        return name
    expanded_events=[];event_refs=[]
    priority=source.dispatch_priority or tuple(sorted(templates));rank={e:i for i,e in enumerate(priority)}
    active=sorted(((i,e) for e,(start,stop) in domains.items() for i in range(start,stop)),key=lambda x:(x[0],rank[x[1]],x[1]))
    for iteration,identity in active:
        template=templates[identity];mapped={}
        for mode in ('reads','writes'):
            mapped[mode]=[]
            for index,region in enumerate(getattr(template,mode)):
                mapping=access_maps.get((identity,mode,index));shift=mapping.iteration_shift if mapping is not None else 0
                if region.instance in interface_ids and shift:
                    raise LoopError('Invariant public interface view cannot be iteration-shifted')
                target=instance_at(region.instance,iteration+shift);span=region.span_bytes
                if mapping is not None and mapping.span=='instance':
                    instance=expanded_instances[target];span=layout_span_bytes(instance.layout,values[instance.value].dtype)-region.offset_bytes
                mapped[mode].append(replace(region,instance=target,span_bytes=span))
        if plan.cost_policy=='assume_unchanged':
            duration=replace(template.duration,evidence=Evidence('assumed',source.fingerprint()+'#'+identity,'finite expansion','Explicit repeated/tail cost assumption; recalibration required'))
        else:duration=Duration(evidence=Evidence('unknown',source.fingerprint()+'#'+identity,'finite expansion','Prior cost invalidated by expansion'))
        name=_id(identity,iteration)
        expanded_events.append(replace(template,id=name,reads=tuple(mapped['reads']),writes=tuple(mapped['writes']),duration=duration))
        event_refs.append(EventRef(name,identity,iteration))
    if len({e.id for e in expanded_events})!=len(expanded_events):raise LoopError('Expanded event identity collision')
    active_ids={(r.template,r.iteration):r.event for r in event_refs};dependencies=[];boundaries=[];pending=[];exits=[];used_seeds=set()
    for edge in sorted(source.dependencies,key=lambda d:(d.source,d.target,d.kind,d.iteration_distance)):
        key=EdgeKey.of(edge);start,stop=domains[edge.target]
        if key in edge_domains:
            lo,hi=edge_domains[key];start=max(start,lo);stop=min(stop,hi)
        for target_i in range(start,stop):
            source_i=target_i-edge.iteration_distance;target=active_ids[(edge.target,target_i)]
            predecessor=active_ids.get((edge.source,source_i))
            if predecessor is None:
                seed=seeds.get((key,target_i))
                if edge.kind=='data':
                    boundaries.append(Boundary(key,target_i,source_i,'data_history_required'))
                    pending.append('Data history requires a real producer and semantic binding: '+target)
                    continue
                if seed is None or not seed.ready:
                    boundaries.append(Boundary(key,target_i,source_i,'missing'))
                    pending.append('Missing initial control seed: '+target+'/'+edge.source);continue
                if len(expanded_events)>=plan.max_events:raise LoopError('Finite event budget exceeded by boundary seeds')
                seed_id='seed:'+str(len(boundaries));used_seeds.add((key,target_i));counter.take(1)
                expanded_events.append(Event(seed_id,templates[edge.target].actor,'synchronize',duration=Duration(0,0,
                    Evidence('assumed',source.fingerprint(),'initial protocol',seed.evidence.note))))
                predecessor=seed_id;boundaries.append(Boundary(key,target_i,source_i,'seeded',seed_id))
            dependencies.append(replace(edge,source=predecessor,target=target,iteration_distance=0,
                evidence=Evidence('assumed',source.fingerprint(),'finite expansion','Template dependency retained with explicit iteration mapping')))
            counter.take(1)
        for source_i in range(*domains[edge.source]):
            target_i=source_i+edge.iteration_distance
            if (edge.target,target_i) not in active_ids:
                exits.append(active_ids[(edge.source,source_i)]+' -> '+edge.target+'@'+str(target_i))
    if set(seeds)-used_seeds:
        # Unused data seeds are reported as pending, never converted to a token.
        for key,target_i in set(seeds)-used_seeds:
            if not any(b.edge==key and b.target_iteration==target_i for b in boundaries):
                raise LoopError('Seed supplied for a non-boundary or inactive edge')
    minimum_stop=min((stop for _,stop in domains.values()),default=0);maximum_start=max((start for start,_ in domains.values()),default=0)
    prologue=tuple(r.event for r in event_refs if r.iteration<maximum_start)
    epilogue=tuple(r.event for r in event_refs if r.iteration>=maximum_start and r.iteration>=minimum_stop)
    steady=tuple(r.event for r in event_refs if r.event not in set(prologue+epilogue))
    def receipt(architecture,coverage=(),drain=()):
        return LoopReceipt(source,plan,'unknown' if architecture is None else 'expanded_conditionally',architecture,
            tuple(event_refs),tuple(instance_refs),tuple(boundaries),tuple(sorted(exits)),prologue,steady,epilogue,drain,coverage,
            CONDITIONS+tuple(sorted(set(pending))))
    if pending:return receipt(None)
    bindings=[]
    for value in source.semantic.values:
        event_ids=tuple(e.id for e in expanded_events if value.id in e.semantic_values)
        instance_ids=tuple(i.id for i in expanded_instances.values() if i.value==value.id)
        if event_ids or instance_ids:bindings.append(Binding(value.id,event_ids,instance_ids))
    claims=tuple(replace(claim,evidence=Evidence('unknown',claim.evidence.reference,claim.evidence.scope,'Template claim invalidated by finite expansion')) for claim in source.claims)
    expanded=replace(source,id=source.id+':finite:'+str(plan.iterations),allocations=tuple(expanded_allocations),
        instances=tuple(expanded_instances.values()),events=tuple(expanded_events),dependencies=tuple(dependencies),bindings=tuple(bindings),claims=claims,
        dispatch_priority=tuple([e.id for e in expanded_events if e.id.startswith('seed:')]+[r.event for r in event_refs]))
    errors=[x for x in validate(expanded).issues if x.severity=='error']
    if errors:raise LoopError('Invalid expanded architecture: '+str(errors))
    pending.extend(_initialization(expanded,plan.inputs,plan.outputs))
    coverage=_coverage(expanded)
    if plan.require_complete_outputs:
        pending.extend('Output coverage not complete: '+x.value for x in coverage if x.status!='complete')
    outgoing={d.source for d in expanded.dependencies}
    drain=tuple(sorted(e.id for e in expanded.events if e.id not in outgoing))
    if pending:return receipt(None,coverage,drain)
    return receipt(expanded,coverage,drain)


def cli(args):
    from pathlib import Path
    from .architecture import _dump
    import sys
    try:
        if args.selftest:
            if args.plan or args.output:raise LoopError('Selftest does not read a plan or write a receipt')
            from .loops_test import selftest
            return selftest()
        if args.verify:
            if args.plan:raise LoopError('Loop receipt replay cannot override its plan')
            receipt=LoopReceipt.from_json(Path(args.verify).read_text());receipt.verify()
            text=_dump(dict(status='replayed_in_model',expansion_status=receipt.status,
                             scope='Finite supplied model; not hardware or arbitrary recurrence semantics certification'),pretty=True)
        else:
            if not args.plan:raise LoopError('Finite loop expansion requires an explicit bound LoopPlan')
            a=Architecture.from_json(Path(args.input).read_text())
            plan=LoopPlan.from_json(Path(args.plan).read_text());text=expand(a,plan).to_json(pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return 0
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr);return 2


def _nc_no_seed_is_not_ready():
    from .architecture import example
    a=example('matmul');a=replace(a,dependencies=(Dependency(a.events[0].id,a.events[0].id,'program',1),))
    result=expand(a,LoopPlan(a.fingerprint(),1))
    return result.status=='unknown' and result.architecture is None


REFS={'LOOP1_explicit_boundaries_and_initialization':Ref(
    what=['有限迭代/视图/slot/seed/接口初始化与输出排空显式；控制token不替代数据，输入数学身份不初始化副本。'],
    status='derived',evidence=['E104 loops_test staged/tail/seed/physical mapping/overwrite counterexamples; numerical and hardware validity remain conditional.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'LOOP1_explicit_boundaries_and_initialization':_nc_no_seed_is_not_ready}
