"""Explicit observation/model replay and bounded hypothesis comparisons.

No names become semantic mappings, observed order never adds dependencies,
and elapsed intervals never become service costs through this interface.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Literal
from .._refs import Ref
from .architecture import (Architecture,Dependency,Event,Resource,Evidence,Record,VersionedRecord,_dag,_dump,_encode)
from .observation import ObservedExecution,validate_observation
from .schedule import Address,SchedulePlan,ScheduleReceipt,_Model,solve
from .trace import TraceReceipt
from .loops import LoopReceipt,_covered


@dataclass(frozen=True)
class TimingHypothesis(VersionedRecord):
    """A timing/resource projection; never a mathematical program."""
    id: str
    events: tuple[Event,...]
    resources: tuple[Resource,...]
    dependencies: tuple[Dependency,...] = ()
    schema_version: int = 1


class _TimingModel:
    """Validate the restricted projection without fabricating an IO program.

    Full Architecture and all scheduling studies still use schedule._Model.
    This projection cannot produce a ScheduleReceipt/implementation contract.
    """
    def __init__(self,a,plan):
        if a.schema_version!=1 or not a.id or plan.schema_version!=1 or plan.source_fingerprint!=a.fingerprint():
            raise ValueError('Timing hypothesis/plan identity mismatch')
        if plan.tick_ns<=0 or plan.max_states<0 or (plan.horizon_ns is not None and plan.horizon_ns<0):
            raise ValueError('Invalid timing plan budget/grid')
        self.events={e.id:e for e in a.events};self.resources={r.id:r for r in a.resources}
        if len(self.events)!=len(a.events) or len(self.resources)!=len(a.resources) or any(not x for x in self.events) or any(not x for x in self.resources):
            raise ValueError('Duplicate/empty timing event/resource')
        if any(r.capacity_units<=0 or not r.owner for r in a.resources):raise ValueError('Invalid timing resource')
        for event in a.events:
            if not event.actor or event.reads or event.writes or event.semantic_values:
                raise ValueError('Timing projections cannot declare physical accesses or mathematical mappings')
            if len({d.resource for d in event.resources})!=len(event.resources):raise ValueError('Duplicate timing demand')
            for demand in event.resources:
                resource=self.resources.get(demand.resource)
                if resource is None or not 0<demand.units<=resource.capacity_units or resource.owner not in ('global',event.actor):
                    raise ValueError('Invalid timing demand/ownership')
        for edge in a.dependencies:
            if edge.source not in self.events or edge.target not in self.events or edge.iteration_distance<0:
                raise ValueError('Invalid timing dependency')
        cyclic,_=_dag(self.events,[(d.source,d.target) for d in a.dependencies if not d.iteration_distance])
        if cyclic:raise ValueError('Timing dependency cycle')
        if plan.alignments:raise ValueError('Timing projections have no storage alignments')
        if plan.priority and (len(plan.priority)!=len(self.events) or set(plan.priority)!=set(self.events)):
            raise ValueError('Timing priority must identify every event')
        costs={}
        for cost in plan.costs:
            if cost.event not in self.events or cost.event in costs:raise ValueError('Invalid timing cost')
            costs[cost.event]=cost.duration
        for event in a.events:
            duration=costs.get(event.id,event.duration);lo,hi=duration.lower_ns,duration.upper_ns
            if (lo is not None and (lo<0 or lo%plan.tick_ns)) or (hi is not None and (hi<0 or hi%plan.tick_ns)) or (lo is not None and hi is not None and lo>hi):
                raise ValueError('Invalid timing interval/grid')
        self.explicit=defaultdict(list)
        for window in plan.reservations:
            event=self.events.get(window.event);demands={} if event is None else {d.resource:d.units for d in event.resources}
            if window.resource not in demands or min(window.offset_ns,window.duration_ns)<0 or not 0<window.units<=demands[window.resource] or window.offset_ns%plan.tick_ns or window.duration_ns%plan.tick_ns:
                raise ValueError('Invalid timing reservation')
            self.explicit[(window.event,window.resource)].append((window.offset_ns,window.duration_ns,window.units))
        from .schedule import _capacity_ok
        for (event,resource),windows in self.explicit.items():
            peak={resource:next(d.units for d in self.events[event].resources if d.resource==resource)}
            if not _capacity_ok([(resource,o,o+d,u) for o,d,u in windows],peak):raise ValueError('Timing reservation peak exceeds demand')
        self.pending=['Timing-only projection has no mathematical, storage or implementation semantics']
        self.pinned=set();self.users={};self.allocations={};self.capacities={};self.alignments={}


@dataclass(frozen=True)
class Scope(Record):
    clock: str
    launch: str
    cores: tuple[str,...]
    layers: tuple[str,...] = ('X',)


@dataclass(frozen=True)
class EventBinding(Record):
    event: str
    records: tuple[str,...]
    compare_cost: bool = False
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Anchor(Record):
    event: str
    offset_ns: int
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class StorageBase(Record):
    storage: str
    space: str
    address: int
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class ReplayPlan(VersionedRecord):
    source_fingerprint: str
    observation_fingerprint: str
    schedule: SchedulePlan
    scope: Scope
    bindings: tuple[EventBinding,...]
    anchors: tuple[Anchor,...] = ()
    addresses: tuple[Address,...] = ()
    storage_bases: tuple[StorageBase,...] = ()
    # 本仓自定: numerical comparison tolerance, never fitted automatically.
    tolerance_ns: int | float = 0
    schema_version: int = 1


@dataclass(frozen=True)
class Check(Record):
    rule: str
    subject: str
    status: Literal['matched','mismatch','unknown']
    detail: str


@dataclass(frozen=True)
class Window(Record):
    event: str
    start_ns: int | float
    finish_ns: int | float
    records: tuple[str,...]


@dataclass(frozen=True)
class Residual(Record):
    event: str
    elapsed_ns: int | float
    lower_ns: int | None
    upper_ns: int | None
    below_lower_ns: int | float | None
    above_upper_ns: int | float | None
    compared: bool


@dataclass(frozen=True)
class Slack(Record):
    source: str
    target: str
    kind: str
    ns: int | float


@dataclass(frozen=True)
class ReplayReceipt(VersionedRecord):
    architecture: Architecture | TimingHypothesis
    observation: ObservedExecution
    plan: ReplayPlan
    trace: TraceReceipt | None
    loop: LoopReceipt | None
    status: Literal['consistent_conditionally','inconsistent','unknown']
    windows: tuple[Window,...]
    checks: tuple[Check,...]
    residuals: tuple[Residual,...]
    slacks: tuple[Slack,...]
    unmapped_records: tuple[str,...]
    outside_scope: tuple[str,...]
    missing_events: tuple[str,...]
    observed_span_ns: int | float | None
    mapped_span_ns: int | float | None
    conditions: tuple[str,...]
    schema_version: int = 1

    def verify(self):
        if replay(self.architecture,self.observation,self.plan,trace=self.trace,loop=self.loop)!=self:
            raise ValueError('Replay receipt differs from its bound model/observation')
        return True


CONDITIONS=(
    'Bindings, envelope grouping, protocol anchors and physical bases are explicit supplied assertions; not semantic inference',
    'Observed completion spans can contain waits; no elapsed span is copied into a scheduling cost',
    'Whole-window resource holding is the model default, not measured hardware occupancy',
    'Matched time/address constraints do not prove mathematics, signal visibility, or implementation correctness',
    'Without a parent TraceReceipt, provenance is a manifest, not authenticated file reads',
    'Outside-scope records remain excluded evidence; a slice cannot certify the complete kernel',
)


def replay(a:Architecture | TimingHypothesis,o:ObservedExecution,plan:ReplayPlan,*,trace:TraceReceipt | None=None,loop:LoopReceipt | None=None):
    if plan.source_fingerprint!=a.fingerprint() or plan.observation_fingerprint!=o.fingerprint():
        raise ValueError('Replay source/observation identity mismatch')
    if trace is not None:
        trace.verify()
        if trace.observation!=o:raise ValueError('Trace parent does not contain the supplied observation')
    errors=[x for x in validate_observation(o).issues if x.severity=='error']
    if errors:raise ValueError('Invalid observation: '+str(errors))
    if loop is not None:
        loop.verify()
        if loop.architecture!=a:raise ValueError('Loop parent does not contain the supplied architecture')
    model=_Model(a,plan.schedule) if isinstance(a,Architecture) else _TimingModel(a,plan.schedule)
    if plan.schema_version!=1 or plan.tolerance_ns<0:raise ValueError('Invalid replay version/tolerance')
    scope=plan.scope
    if not scope.clock or not scope.launch or not scope.cores or not all(scope.cores) or len(set(scope.cores))!=len(scope.cores):
        raise ValueError('Explicit unique clock/launch/cores are required')
    if len(set(scope.layers))!=len(scope.layers):raise ValueError('Duplicate scope layers')
    observed={e.id:e for e in o.events};eligible={};outside=[]
    for event in o.events:
        attrs={x.name:x.value for x in event.attributes};layer=attrs.get('trace.phase')
        if (event.clock==scope.clock and event.launch==scope.launch and event.core in scope.cores
            and (not scope.layers or layer in scope.layers)):eligible[event.id]=event
        else:outside.append(event.id)
    origin=min((e.start_ns for e in eligible.values()),default=0)
    checks=[];windows={};assigned=set();bindings={};residuals=[];slacks=[]
    def check(rule,subject,status,detail):checks.append(Check(rule,subject,status,detail))
    if not eligible:check('scope','capture','unknown','No execution records in explicit scope')
    for core in scope.cores:
        if not any(e.core==core for e in eligible.values()):check('scope',core,'unknown','Requested core has no eligible records')
    for binding in plan.bindings:
        if binding.event not in model.events or binding.event in bindings or not binding.records:
            raise ValueError('Invalid/duplicate/empty event binding')
        if len(set(binding.records))!=len(binding.records) or assigned.intersection(binding.records):
            raise ValueError('Observed records cannot be duplicated or assigned to multiple events')
        if any(identity not in eligible for identity in binding.records):
            raise ValueError('Binding references an unknown or out-of-scope record')
        rows=[eligible[x] for x in binding.records]
        if any(e.architecture_event and e.architecture_event!=binding.event for e in rows):
            raise ValueError('Observation event mapping contradicts explicit binding')
        bindings[binding.event]=binding;assigned.update(binding.records)
        windows[binding.event]=Window(binding.event,min(e.start_ns for e in rows)-origin,
            max(e.end_ns for e in rows)-origin,binding.records)
    for anchor in plan.anchors:
        event=model.events.get(anchor.event)
        if event is None or event.id in windows or anchor.offset_ns<0:
            raise ValueError('Invalid/duplicate protocol anchor')
        duration=next((c.duration for c in plan.schedule.costs if c.event==event.id),event.duration)
        if event.kind!='synchronize' or event.reads or event.writes or event.resources or duration.lower_ns!=0 or duration.upper_ns!=0:
            raise ValueError('Only explicit zero-time no-access protocol states may be anchored')
        windows[event.id]=Window(event.id,anchor.offset_ns,anchor.offset_ns,())
    if o.architecture_fingerprint and o.architecture_fingerprint!=a.fingerprint():
        raise ValueError('Observation declares another architecture')
    missing=tuple(sorted(set(model.events)-set(windows)));unmapped=tuple(sorted(set(eligible)-assigned))
    if missing:check('event_coverage','model','unknown',str(len(missing))+' model events have no observed window or protocol anchor')
    if unmapped:check('record_coverage','capture','unknown',str(len(unmapped))+' scoped records remain unmapped')
    if o.coverage.records_unsupported:check('record_coverage','capture','unknown','Unsupported source records remain')
    for pending in model.pending:check('model_prerequisite','model','unknown',pending)
    for identity,binding in bindings.items():
        window=windows[identity];duration=next((c.duration for c in plan.schedule.costs if c.event==identity),model.events[identity].duration)
        elapsed=window.finish_ns-window.start_ns;lo=duration.lower_ns;hi=duration.upper_ns
        below=None if lo is None else max(0,lo-elapsed);above=None if hi is None else max(0,elapsed-hi)
        residuals.append(Residual(identity,elapsed,lo,hi,below,above,binding.compare_cost))
        if not binding.compare_cost or lo is None or hi is None:
            check('duration',identity,'unknown','Completion envelope is not an asserted service/latency calibration')
        else:
            status='matched' if lo-plan.tolerance_ns<=elapsed<=hi+plan.tolerance_ns else 'mismatch'
            check('duration',identity,status,'Observed '+str(elapsed)+'ns vs declared ['+str(lo)+','+str(hi)+']ns')
    for edge in a.dependencies:
        if edge.iteration_distance:
            check('dependency',edge.source+' -> '+edge.target,'unknown','Positive distance requires finite expansion');continue
        if edge.source not in windows or edge.target not in windows:continue
        slack=windows[edge.target].start_ns-windows[edge.source].finish_ns
        slacks.append(Slack(edge.source,edge.target,edge.kind,slack))
        check('dependency',edge.source+' -> '+edge.target,'matched' if slack>=-plan.tolerance_ns else 'mismatch',str(slack)+'ns completion slack')
    occupancies=defaultdict(list)
    for identity,window in windows.items():
        for demand in model.events[identity].resources:
            key=(identity,demand.resource)
            if key in model.explicit:
                for offset,duration,units in model.explicit[key]:
                    occupancies[demand.resource].append((window.start_ns+offset,window.start_ns+offset+duration,units))
            else:occupancies[demand.resource].append((window.start_ns,window.finish_ns,demand.units))
    for resource,rows in sorted(occupancies.items()):
        changes=defaultdict(int)
        for start,end,units in rows:
            if end>start:changes[start]+=units;changes[end]-=units
        active=peak=0
        for time in sorted(changes):active+=changes[time];peak=max(peak,active)
        capacity=model.resources[resource].capacity_units
        check('resource',resource,'matched' if peak<=capacity else 'mismatch','Peak '+str(peak)+' vs declared capacity '+str(capacity))
    addresses={};bases={}
    for address in plan.addresses:
        if address.allocation not in model.allocations or address.allocation in addresses:
            raise ValueError('Invalid/duplicate physical allocation binding')
        allocation=model.allocations[address.allocation]
        if address.storage!=allocation.storage or address.offset_bytes<0 or address.offset_bytes%model.alignments[address.storage]:
            raise ValueError('Physical allocation storage/alignment mismatch')
        addresses[address.allocation]=address
        capacity=model.capacities[address.storage]
        if capacity is None:check('memory_capacity',address.storage,'unknown','Storage capacity unknown')
        else:check('memory_capacity',address.allocation,'matched' if address.offset_bytes+allocation.size_bytes<=capacity else 'mismatch','Address extent vs declared arena capacity')
    for base in plan.storage_bases:
        if base.storage not in model.capacities or base.storage in bases or not base.space or base.address<0:
            raise ValueError('Invalid/duplicate storage base')
        bases[base.storage]=base
    required=set(model.pinned)|{x for x,users in model.users.items() if users}
    if required-set(addresses):check('memory_binding','model','unknown','Not all live allocations have explicit offsets')
    maximum=max((w.finish_ns for w in windows.values()),default=0)
    lifetimes={}
    for identity in required.intersection(addresses):
        users=model.users[identity]
        if not users.issubset(windows):continue
        if identity in model.pinned:lifetimes[identity]=(0,maximum,True)
        else:lifetimes[identity]=(min(windows[e].start_ns for e in users),max(windows[e].finish_ns for e in users),False)
    for index,left in enumerate(sorted(lifetimes)):
        for right in sorted(lifetimes)[index+1:]:
            x,y=addresses[left],addresses[right];l,r=lifetimes[left],lifetimes[right]
            if not (l[2] or r[2] or max(l[0],r[0])<min(l[1],r[1])):continue
            if x.storage==y.storage:lx,rx=x.offset_bytes,y.offset_bytes
            else:
                xb,yb=bases.get(x.storage),bases.get(y.storage)
                if xb is None or yb is None or xb.space!=yb.space:continue
                lx,rx=xb.address+x.offset_bytes,yb.address+y.offset_bytes
            overlap=max(lx,rx)<min(lx+model.allocations[left].size_bytes,rx+model.allocations[right].size_bytes)
            if overlap:check('memory_alias',left+'/'+right,'mismatch','Overlapping physical addresses have simultaneous leases')
    instances={x.id:x for x in a.instances} if isinstance(a,Architecture) else {}
    for identity,binding in bindings.items():
        if isinstance(a,TimingHypothesis):continue
        event=model.events[identity];expected=[];seen=[];unresolved_modes=set()
        for mode,regions in (('read',event.reads),('write',event.writes)):
            for region in regions:
                if not region.span_bytes:continue
                instance=instances[region.instance];allocation=model.allocations[instance.allocation]
                address=addresses.get(allocation.id);base=bases.get(allocation.storage)
                if address is None or base is None:
                    check('access',identity,'unknown','No physical address/base for '+instance.id);unresolved_modes.add(mode);continue
                start=base.address+address.offset_bytes+instance.offset_bytes+region.offset_bytes
                expected.append((mode,base.space,start,start+region.span_bytes,instance.id))
        for record in binding.records:
            row=observed[record]
            if row.memory_kind=='unknown':check('access',record,'unknown','Instruction memory behavior unknown')
            for access in row.accesses:
                if access.address is None or access.span_bytes is None:
                    check('access',record,'unknown','Observed address/span unavailable');continue
                for mode in (('read','write') if access.mode=='read_write' else (access.mode,)):
                    intervals=[(s,e) for m,space,s,e,iid in expected if m==mode and space==access.space and (not access.instance or iid==access.instance)]
                    if not _covered(access.address,access.address+access.span_bytes,intervals):
                        check('access',record,'unknown' if mode in unresolved_modes else 'mismatch',
                              'Physical map incomplete' if mode in unresolved_modes else 'Observed access is outside the bound event regions')
                    seen.append((mode,access.space,access.address,access.address+access.span_bytes,access.instance))
        for mode,space,start,end,iid in expected:
            intervals=[(s,e) for m,sp,s,e,instance in seen if m==mode and sp==space and (not instance or instance==iid)]
            check('access_coverage',identity+'/'+iid,'matched' if _covered(start,end,intervals) else 'unknown','Declared access vs recovered byte coverage')
    for field in ('addresses','synchronization'):
        if getattr(o.coverage,field)!='declared_complete':check('coverage',field,'unknown','Observation does not declare complete '+field)
    span=None if not eligible else max(e.end_ns for e in eligible.values())-origin
    mapped_span=None if not windows else max(w.finish_ns for w in windows.values())-min(w.start_ns for w in windows.values())
    status='inconsistent' if any(c.status=='mismatch' for c in checks) else 'unknown' if any(c.status=='unknown' for c in checks) else 'consistent_conditionally'
    return ReplayReceipt(a,o,plan,trace,loop,status,tuple(windows[k] for k in sorted(windows)),tuple(checks),
        tuple(sorted(residuals,key=lambda x:x.event)),tuple(slacks),unmapped,tuple(sorted(outside)),missing,span,mapped_span,CONDITIONS)


@dataclass(frozen=True)
class Scenario(Record):
    id: str
    architecture: Architecture
    plan: SchedulePlan
    hypothesis: str
    loop: LoopReceipt | None = None


@dataclass(frozen=True)
class StudyPlan(VersionedRecord):
    reference: str
    scenarios: tuple[Scenario,...]
    # 本仓自定: finite analysis workload budget, not hardware or benefit rule.
    max_scenarios: int = 32
    schema_version: int = 1


@dataclass(frozen=True)
class Gain(Record):
    scenario: str
    witness_delta_ns: int | None
    optimum_gain_lower_ns: int | None
    optimum_gain_upper_ns: int | None
    classification: Literal['conditional_improvement','conditional_regression','overlapping_bounds','unknown']


@dataclass(frozen=True)
class StudyReceipt(VersionedRecord):
    plan: StudyPlan
    results: tuple[ScheduleReceipt,...]
    gains: tuple[Gain,...]
    conditions: tuple[str,...]
    schema_version: int = 1

    def verify(self):
        if study(self.plan)!=self:raise ValueError('Study differs from deterministic scenario replay')
        return True


def study(plan:StudyPlan):
    ids=[s.id for s in plan.scenarios]
    if plan.schema_version!=1 or not ids or len(ids)>plan.max_scenarios or len(set(ids))!=len(ids) or any(not x for x in ids) or plan.reference not in ids:
        raise ValueError('Invalid/duplicate/reference/scenario budget')
    semantics={_dump(_encode(s.architecture.semantic,canonical=True)) for s in plan.scenarios}
    if len(semantics)!=1:raise ValueError('Scenario semantics differ; supply a separate declared algorithm comparison')
    if any(not s.hypothesis for s in plan.scenarios):raise ValueError('Every scenario needs an explicit hypothesis')
    for scenario in plan.scenarios:
        if scenario.loop is not None:
            scenario.loop.verify()
            if scenario.loop.architecture!=scenario.architecture:raise ValueError('Scenario loop parent mismatch')
    results=tuple(solve(s.architecture,s.plan) for s in plan.scenarios);baseline=results[ids.index(plan.reference)];gains=[]
    for scenario,result in zip(plan.scenarios,results):
        delta=None if baseline.upper_ns is None or result.upper_ns is None else baseline.upper_ns-result.upper_ns
        lower=None if baseline.lower_ns is None or result.upper_ns is None else baseline.lower_ns-result.upper_ns
        upper=None if baseline.upper_ns is None or result.lower_ns is None else baseline.upper_ns-result.lower_ns
        classification='unknown' if lower is None or upper is None else 'conditional_improvement' if lower>0 else 'conditional_regression' if upper<0 else 'overlapping_bounds'
        gains.append(Gain(scenario.id,delta,lower,upper,classification))
    return StudyReceipt(plan,results,tuple(gains),(
        'Each result is conditional on its explicit complete scenario, not calibrated hardware or a local derivative',
        'Witness deltas compare feasible schedules; optimum bounds use reference L minus candidate U and reference U minus candidate L',
        'Different cost/occupancy assumptions are separate hypotheses; no ranking is certified outside their joint domain',
        'Unknown or bounded-search failure does not certify physical impossibility or zero benefit',
    ))


def cli(args):
    from pathlib import Path
    import sys
    try:
        if args.selftest:
            if args.output or args.plan or args.trace_receipt or args.observation or args.loop_parent:raise ValueError('Selftest does not consume inputs or outputs')
            from .analysis_test import selftest
            return selftest()
        if args.verify:
            if args.plan or args.trace_receipt or args.observation or args.loop_parent:raise ValueError('Replay cannot override frozen inputs')
            from .architecture import parse_json
            raw=Path(args.verify).read_text();data=parse_json(raw)
            receipt=ReplayReceipt.from_json(raw) if 'observation' in data else StudyReceipt.from_json(raw)
            receipt.verify();text=_dump(dict(status='replayed_in_model',scope='Conditional supplied model, not hardware certification'),pretty=True)
        elif args.study:
            if args.plan or args.trace_receipt or args.observation or args.loop_parent:raise ValueError('Study embeds its own complete scenarios')
            text=study(StudyPlan.from_json(Path(args.study).read_text())).to_json(pretty=True)
        else:
            if not args.plan or bool(args.trace_receipt)==bool(args.observation):raise ValueError('Replay requires a plan and exactly one observation or trace parent')
            from .architecture import parse_json
            raw=Path(args.input).read_text()
            a=Architecture.from_json(raw) if 'semantic' in parse_json(raw) else TimingHypothesis.from_json(raw)
            plan=ReplayPlan.from_json(Path(args.plan).read_text())
            parent=TraceReceipt.from_json(Path(args.trace_receipt).read_text()) if args.trace_receipt else None
            o=parent.observation if parent is not None else ObservedExecution.from_json(Path(args.observation).read_text())
            loop=LoopReceipt.from_json(Path(args.loop_parent).read_text()) if args.loop_parent else None
            text=replay(a,o,plan,trace=parent,loop=loop).to_json(pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return 0
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr);return 2


def _nc_missing_is_unknown():
    from .architecture import example
    from .observation import Coverage,Provenance
    a=example('matmul');o=ObservedExecution('nc',(),(),(),Provenance('fixture','1',()),Coverage(0,0))
    plan=ReplayPlan(a.fingerprint(),o.fingerprint(),SchedulePlan(a.fingerprint()),Scope('clock','launch',('core',)),())
    return replay(a,o,plan).status=='unknown'


REFS={'ANALYSIS1_observation_not_calibration':Ref(
    what=['观测映射/未覆盖/约束冲突显式；不从等待复制服务成本；方案增益用L-U区间而非两个U差冒充保证。'],
    status='derived',evidence=['E105 analysis_test replay, missing/alias/scope/cost/bounds controls; real trace checks retain unknown semantics.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'ANALYSIS1_observation_not_calibration':_nc_missing_is_unknown}
