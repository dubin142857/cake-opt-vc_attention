"""P3 finite integer-time schedules with resource windows and contiguous storage.

All certificates are conditional on the declared model. No simulator interval
is automatically treated as a service cost, and a bounded failed search is not
a proof that a real kernel or even an unbounded model is infeasible.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal
from .._refs import Ref
from .architecture import (Architecture,Contract,Dependency,Duration,Evidence,Record,
                          VersionedRecord,validate)


@dataclass(frozen=True)
class Cost(Record):
    event: str
    duration: Duration


@dataclass(frozen=True)
class Reservation(Record):
    event: str
    resource: str
    offset_ns: int
    duration_ns: int
    units: int = 1
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Alignment(Record):
    storage: str
    bytes: int
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class SchedulePlan(VersionedRecord):
    source_fingerprint: str
    strategy: Literal['list','exact'] = 'list'
    costs: tuple[Cost,...] = ()
    reservations: tuple[Reservation,...] = ()
    alignments: tuple[Alignment,...] = ()
    priority: tuple[str,...] = ()
    tick_ns: int = 1
    horizon_ns: int | None = None
    # 本仓自定: computational search budget, not a research/benefit gate.
    max_states: int = 100000
    schema_version: int = 1


@dataclass(frozen=True)
class Placement(Record):
    event: str
    start_ns: int
    finish_ns: int


@dataclass(frozen=True)
class ResourceUse(Record):
    event: str
    resource: str
    start_ns: int
    finish_ns: int
    units: int


@dataclass(frozen=True)
class Lease(Record):
    allocation: str
    storage: str
    start_ns: int
    finish_ns: int
    bytes: int
    pinned: bool


@dataclass(frozen=True)
class Address(Record):
    allocation: str
    storage: str
    offset_bytes: int


@dataclass(frozen=True)
class Certificate(Record):
    placements: tuple[Placement,...]
    resource_uses: tuple[ResourceUse,...]
    leases: tuple[Lease,...]
    addresses: tuple[Address,...]
    makespan_ns: int


@dataclass(frozen=True)
class LowerTerm(Record):
    rule: str
    ns: int


@dataclass(frozen=True)
class SearchStats(Record):
    states: int
    horizon_ns: int
    exhausted: bool
    optimal_upper_realization: bool
    reason: str


@dataclass(frozen=True)
class ActorStep(Record):
    event: str
    actor: str
    target_start_ns: int
    target_finish_ns: int
    prerequisites: tuple[Dependency,...]


@dataclass(frozen=True)
class ReuseProtocol(Record):
    released_allocation: str
    acquired_allocation: str
    release_after: tuple[str,...]
    acquire_before: tuple[str,...]


@dataclass(frozen=True)
class ImplementationContract(Record):
    numerical_contract: Contract
    actor_steps: tuple[ActorStep,...]
    resource_admission: tuple[ResourceUse,...]
    storage_leases: tuple[Lease,...]
    storage_addresses: tuple[Address,...]
    reuse_protocols: tuple[ReuseProtocol,...]
    obligations: tuple[str,...]


@dataclass(frozen=True)
class ScheduleReceipt(VersionedRecord):
    architecture: Architecture
    plan: SchedulePlan
    status: Literal['scheduled_conditionally','unknown']
    lower_ns: int | None
    upper_ns: int | None
    lower_terms: tuple[LowerTerm,...]
    certificate: Certificate | None
    search: SearchStats
    implementation: ImplementationContract | None
    conditions: tuple[str,...]
    schema_version: int = 1

    def verify(self):
        if self.certificate is not None:
            check_certificate(self.architecture,self.plan,self.certificate)
        if solve(self.architecture,self.plan)!=self:
            raise ValueError('Schedule receipt differs from deterministic model replay')
        return True


class ScheduleError(ValueError):
    pass


class BudgetExhausted(Exception):
    pass


class _Budget:
    def __init__(self,limit):
        self.limit=limit;self.states=0

    def step(self):
        if self.states>=self.limit:
            raise BudgetExhausted()
        self.states+=1


def _capacity_ok(rows,capacities):
    # Half-open occupancy; a completion at t can release before admission at t.
    deltas=defaultdict(lambda:defaultdict(int))
    for resource,start,end,units in rows:
        if end>start:
            deltas[resource][start]+=units;deltas[resource][end]-=units
    for resource,points in deltas.items():
        active=0
        for time in sorted(points):
            active+=points[time]
            if active>capacities[resource]:return False
    return True


def _overlap(a,b):
    # Pinned allocations are distinct even in a zero-length model execution.
    return a[5] or b[5] or max(a[2],b[2])<min(a[3],b[3])


def pack_memory(leases,capacities,alignments,*,exact,budget):
    """Return explicit contiguous offsets; None is only this search's failure.

    Exact mode enumerates every aligned base in the finite declared capacity.
    It can stop after the first witness because this is a feasibility subproblem.
    """
    leases=sorted(leases,key=lambda x:(x[1],-x[4],x[0]));addresses={}
    # Necessary capacity check only: passing the peak test is not a layout proof.
    for storage,capacity in capacities.items():
        rows=[x for x in leases if x[1]==storage]
        if not rows:continue
        pinned=sum(x[4] for x in rows if x[5])
        if pinned>capacity:return None
        points={x[2] for x in rows if not x[5]}
        if any(pinned+sum(x[4] for x in rows if not x[5] and x[2]<=time<x[3])>capacity for time in points):
            return None
    def visit(index):
        if index==len(leases):return dict(addresses)
        current=leases[index];identity,storage,_,_,size,_=current
        cap=capacities[storage];alignment=alignments[storage]
        boundaries={0}
        for old in leases[:index]:
            if old[1]==storage and _overlap(current,old):
                end=addresses[old[0]]+old[4]
                boundaries.add(((end+alignment-1)//alignment)*alignment)
        preferred=sorted(x for x in boundaries if x+size<=cap)
        def candidates():
            yield from preferred
            if exact:
                for candidate in range(0,cap-size+1,alignment):
                    if candidate not in boundaries:yield candidate
        for offset in candidates():
            budget.step()
            if any(old[1]==storage and _overlap(current,old)
                   and max(offset,addresses[old[0]])<min(offset+size,addresses[old[0]]+old[4])
                   for old in leases[:index]):
                continue
            addresses[identity]=offset
            found=visit(index+1)
            if found is not None:return found
            del addresses[identity]
            if not exact:break
        return None
    return visit(0)


class _Model:
    def __init__(self,a,plan):
        self.a=a;self.plan=plan;self.pending=[]
        if plan.schema_version!=1 or plan.source_fingerprint!=a.fingerprint():
            raise ScheduleError('Plan version/source identity mismatch')
        if plan.tick_ns<=0 or plan.max_states<0 or (plan.horizon_ns is not None and plan.horizon_ns<0):
            raise ScheduleError('Invalid time grid, horizon or search budget')
        validation=validate(a)
        errors=[x for x in validation.issues if x.severity=='error']
        if errors:raise ScheduleError('Invalid architecture: '+str(errors))
        self.events={e.id:e for e in a.events};self.resources={r.id:r for r in a.resources}
        self.capacities={s.id:s.capacity_bytes for s in a.storages};self.alignments={s.id:1 for s in a.storages}
        self.allocations={x.id:x for x in a.allocations};instances={x.id:x for x in a.instances}
        self.pinned={x.id for x in a.allocations if x.persistent}
        io=set(a.semantic.inputs+a.semantic.outputs)
        self.pinned.update(x.allocation for x in a.instances if x.value in io)
        self.users=defaultdict(set)
        for event in a.events:
            for region in event.reads+event.writes:
                if region.span_bytes:self.users[instances[region.instance].allocation].add(event.id)
        for allocation in a.allocations:
            if (allocation.id in self.pinned or self.users[allocation.id]) and self.capacities[allocation.storage] is None:
                self.pending.append('Unknown storage capacity: '+allocation.storage)
        seen=set()
        for alignment in plan.alignments:
            if alignment.storage not in self.capacities or alignment.bytes<=0 or alignment.storage in seen:
                raise ScheduleError('Invalid/duplicate storage alignment')
            seen.add(alignment.storage);self.alignments[alignment.storage]=alignment.bytes
        priority=plan.priority or a.dispatch_priority or tuple(sorted(self.events))
        if len(priority)!=len(self.events) or set(priority)!=set(self.events):
            raise ScheduleError('Priority must identify every event exactly once')
        rank={x:i for i,x in enumerate(priority)}
        self.predecessors={x:set() for x in self.events}
        for edge in a.dependencies:
            if edge.iteration_distance:
                self.pending.append('Positive-distance edge needs finite expansion, seeds and drain handling')
            else:self.predecessors[edge.target].add(edge.source)
        remaining=set(self.events);self.order=[]
        while remaining:
            ready=[x for x in remaining if self.predecessors[x].issubset(self.order)]
            if not ready:raise ScheduleError('Zero-distance dependency cycle')
            current=min(ready,key=lambda x:(rank[x],x));self.order.append(current);remaining.remove(current)
        overrides={}
        for cost in plan.costs:
            if cost.event not in self.events or cost.event in overrides:
                raise ScheduleError('Invalid/duplicate event cost')
            overrides[cost.event]=cost.duration
        self.lower={};self.upper={}
        for event in a.events:
            duration=overrides.get(event.id,event.duration);lo,hi=duration.lower_ns,duration.upper_ns
            if (lo is not None and lo<0) or (hi is not None and hi<0) or (lo is not None and hi is not None and lo>hi):
                raise ScheduleError('Invalid cost interval')
            if lo is None or hi is None:
                self.pending.append('Unknown event cost: '+event.id);continue
            if lo%plan.tick_ns or hi%plan.tick_ns:raise ScheduleError('Cost is not on the declared time grid')
            if hi==0 and any(r.span_bytes for r in event.reads+event.writes):
                self.pending.append('Instantaneous physical access needs an explicit model: '+event.id)
            if event.kind in ('compute','copy') and not event.resources:
                self.pending.append('Unknown execution resource: '+event.id)
            self.lower[event.id]=lo;self.upper[event.id]=hi
        self.explicit=defaultdict(list)
        for window in plan.reservations:
            if window.event not in self.events:raise ScheduleError('Unknown reservation event')
            demands={d.resource:d.units for d in self.events[window.event].resources}
            if window.resource not in demands or min(window.offset_ns,window.duration_ns)<0 or window.units<=0 or window.units>demands[window.resource]:
                raise ScheduleError('Invalid resource reservation')
            if window.offset_ns%plan.tick_ns or window.duration_ns%plan.tick_ns:
                raise ScheduleError('Reservation is not on the declared time grid')
            self.explicit[(window.event,window.resource)].append((window.offset_ns,window.duration_ns,window.units))
        for (event,resource),windows in self.explicit.items():
            peak={resource:next(d.units for d in self.events[event].resources if d.resource==resource)}
            if not _capacity_ok([(resource,offset,offset+duration,units) for offset,duration,units in windows],peak):
                raise ScheduleError('Reservation peak exceeds declared event demand')

    def windows(self,event,upper):
        duration=(self.upper if upper else self.lower)[event]
        rows=[]
        for demand in self.events[event].resources:
            pieces=self.explicit.get((event,demand.resource))
            if pieces is None:pieces=[(0,duration,demand.units)]
            rows.extend((demand.resource,offset,offset+span,units) for offset,span,units in pieces)
        return rows

    def extent(self,event,upper=True):
        return max([(self.upper if upper else self.lower)[event]]+[end for _,_,end,_ in self.windows(event,upper)])

    def raw(self,starts):
        uses=[(event,resource,start+offset,start+end,units)
              for event,start in sorted(starts.items()) for resource,offset,end,units in self.windows(event,True)]
        makespan=max([0]+[start+self.extent(event) for event,start in starts.items()])
        leases=[]
        for identity,allocation in sorted(self.allocations.items()):
            pinned=identity in self.pinned;users=self.users[identity]&set(starts)
            if pinned:begin,end=0,makespan
            elif users:
                begin=min(starts[e] for e in users);end=max(starts[e]+self.upper[e] for e in users)
            else:continue
            leases.append((identity,allocation.storage,begin,end,allocation.size_bytes,pinned))
        return uses,leases,makespan

    def candidates(self,event,starts,earliest,horizon):
        # List search is heuristic: jump to declared admission/release boundaries,
        # rather than spending the entire budget walking nanoseconds of busy time.
        uses,leases,_=self.raw(starts);points={earliest}
        for resource,offset,_,_ in self.windows(event,True):
            points.update(end-offset for _,old_resource,_,end,_ in uses if resource==old_resource)
        points.update(row[3] for row in leases)
        tick=self.plan.tick_ns
        return sorted({((t+tick-1)//tick)*tick for t in points if t>=earliest
                       and ((t+tick-1)//tick)*tick+self.extent(event)<=horizon})

    def feasible(self,starts,budget,exact_memory):
        uses,leases,makespan=self.raw(starts)
        capacities={x:r.capacity_units for x,r in self.resources.items()}
        if not _capacity_ok([(r,s,e,u) for _,r,s,e,u in uses],capacities):return None
        addresses=pack_memory(leases,self.capacities,self.alignments,exact=exact_memory,budget=budget)
        if addresses is None:return None
        return uses,leases,makespan,addresses

    def certificate(self,starts,raw):
        uses,leases,makespan,addresses=raw
        return Certificate(tuple(Placement(e,t,t+self.upper[e]) for e,t in sorted(starts.items())),
            tuple(ResourceUse(*row) for row in sorted(uses)),tuple(Lease(*row) for row in leases),
            tuple(Address(e,self.allocations[e].storage,t) for e,t in sorted(addresses.items())),makespan)


def lower_terms(model):
    earliest={}
    for event in model.order:
        earliest[event]=max([0]+[earliest[p]+model.lower[p] for p in model.predecessors[event]])
    cp=max([0]+[earliest[e]+model.extent(e,False) for e in model.order])
    terms=[LowerTerm('dependency_path_and_reservation_tail',cp)];work=defaultdict(int)
    for event in model.order:
        for resource,start,end,units in model.windows(event,False):work[resource]+=(end-start)*units
    for resource,amount in sorted(work.items()):
        capacity=model.resources[resource].capacity_units
        terms.append(LowerTerm('resource_work:'+resource,(amount+capacity-1)//capacity))
    return tuple(terms)


def check_certificate(architecture,plan,certificate):
    model=_Model(architecture,plan)
    if model.pending:raise ScheduleError('Model prerequisites are unresolved')
    starts={p.event:p.start_ns for p in certificate.placements}
    if len(starts)!=len(certificate.placements) or set(starts)!=set(model.events):
        raise ScheduleError('Certificate event coverage is wrong')
    for placement in certificate.placements:
        if placement.start_ns<0 or placement.start_ns%plan.tick_ns or placement.finish_ns!=placement.start_ns+model.upper[placement.event]:
            raise ScheduleError('Invalid event placement')
    for event in starts:
        if any(starts[p]+model.upper[p]>starts[event] for p in model.predecessors[event]):
            raise ScheduleError('Dependency not satisfied')
    uses,leases,makespan=model.raw(starts)
    if certificate.makespan_ns!=makespan or tuple(ResourceUse(*x) for x in sorted(uses))!=certificate.resource_uses or tuple(Lease(*x) for x in leases)!=certificate.leases:
        raise ScheduleError('Resource/lifetime/makespan witness differs from the declared model')
    if not _capacity_ok([(r,s,e,u) for _,r,s,e,u in uses],{x:r.capacity_units for x,r in model.resources.items()}):
        raise ScheduleError('Shared resource capacity exceeded')
    addresses={x.allocation:x for x in certificate.addresses}
    if len(addresses)!=len(certificate.addresses) or set(addresses)!={x[0] for x in leases}:
        raise ScheduleError('Address witness coverage is wrong')
    for lease in leases:
        address=addresses[lease[0]];capacity=model.capacities[lease[1]]
        if address.storage!=lease[1] or address.offset_bytes<0 or address.offset_bytes%model.alignments[lease[1]] or address.offset_bytes+lease[4]>capacity:
            raise ScheduleError('Address bounds/alignment invalid')
    for i,left in enumerate(leases):
        for right in leases[i+1:]:
            if left[1]==right[1] and _overlap(left,right):
                x,y=addresses[left[0]].offset_bytes,addresses[right[0]].offset_bytes
                if max(x,y)<min(x+left[4],y+right[4]):
                    raise ScheduleError('Concurrent allocation addresses overlap')
    return True


CONDITIONS=(
    'All costs, ownership and resource windows are declared hypotheses, not hardware calibration or simulator wait service times',
    'Default whole-latency resource holding and byte alignment=1 are model choices, not Ascend ABI facts',
    'Persistent and public input/output storage remains pinned; temporary leases require real allocation/release protocols',
    'Offsets propose a backing arena: bind public IO to real interface addresses or model explicit copies before implementation',
    'Integer-time and contiguous storage certificates do not certify numerical semantics, cross-core visibility or implementation',
    'Physical accesses must finish by event completion; a resource reservation tail does not authorize later storage access',
    'Loop prologue, steady state, seeds and drain require finite expansion; positive-distance edges are not ignored',
)


def solve(architecture:Architecture,plan:SchedulePlan):
    model=_Model(architecture,plan)
    if model.pending:
        return ScheduleReceipt(architecture,plan,'unknown',None,None,(),None,
            SearchStats(0,plan.horizon_ns if plan.horizon_ns is not None else 0,False,False,'prerequisites'),None,
            CONDITIONS+tuple(sorted(set(model.pending))))
    terms=lower_terms(model);lower=max(x.ns for x in terms)
    horizon=plan.horizon_ns if plan.horizon_ns is not None else sum(model.extent(e) for e in model.order)
    budget=_Budget(plan.max_states);best=None;best_starts=None;exhausted=False;optimal=False;reason='list'
    def earliest(event,starts):
        return max([0]+[starts[p]+model.upper[p] for p in model.predecessors[event]])
    try:
        starts={};candidate=None
        for event in model.order:
            candidate=None
            for time in model.candidates(event,starts,earliest(event,starts),horizon):
                budget.step();trial=dict(starts,**{event:time})
                found=model.feasible(trial,budget,False)
                if found is not None:
                    starts=trial;candidate=found;break
            if candidate is None:break
        if len(starts)==len(model.order):
            best=model.feasible(starts,budget,True) if not model.order else candidate;best_starts=dict(starts)
        if plan.strategy=='exact':
            reason='complete'
            if best is not None and best[2]==lower:
                optimal=True;reason='bound_closed'
            else:
                def visit(index,starts):
                    nonlocal best,best_starts
                    if index==len(model.order):return
                    event=model.order[index];ceiling=horizon if best is None else min(horizon,best[2]-1)
                    for time in range(earliest(event,starts),ceiling-model.extent(event)+1,plan.tick_ns):
                        budget.step();trial=dict(starts,**{event:time})
                        found=model.feasible(trial,budget,True)
                        if found is None:continue
                        if index+1==len(model.order):
                            if best is None or found[2]<best[2]:best=found;best_starts=trial
                        else:visit(index+1,trial)
                visit(0,{})
                optimal=best is not None
    except BudgetExhausted:
        exhausted=True;reason='budget'
    if best is None:
        return ScheduleReceipt(architecture,plan,'unknown',lower,None,terms,None,
            SearchStats(budget.states,horizon,exhausted,False,reason if exhausted else 'no_witness_in_search'),None,
            CONDITIONS+('No certificate found; bounded search failure is not global infeasibility',))
    certificate=model.certificate(best_starts,best);check_certificate(architecture,plan,certificate)
    # An exact optimum at upper costs is not a lower bound for shorter intervals.
    if optimal and all(model.lower[e]==model.upper[e] for e in model.events):
        terms+= (LowerTerm('complete_fixed_cost_integer_search',certificate.makespan_ns),)
        lower=max(lower,certificate.makespan_ns)
    steps=tuple(ActorStep(p.event,model.events[p.event].actor,p.start_ns,p.finish_ns,
                         tuple(sorted((d for d in architecture.dependencies if d.target==p.event),key=lambda d:(d.source,d.kind))))
                for p in sorted(certificate.placements,key=lambda p:(model.events[p.event].actor,p.start_ns,p.event)))
    reuse=[];addresses={x.allocation:x.offset_bytes for x in certificate.addresses}
    for i,left in enumerate(certificate.leases):
        for right in certificate.leases[i+1:]:
            if left.storage!=right.storage:continue
            x,y=addresses[left.allocation],addresses[right.allocation]
            if max(x,y)>=min(x+left.bytes,y+right.bytes):continue
            old,new=(left,right) if left.finish_ns<=right.start_ns else (right,left)
            reuse.append(ReuseProtocol(old.allocation,new.allocation,
                         tuple(sorted(model.users[old.allocation])),tuple(sorted(model.users[new.allocation]))))
    implementation=ImplementationContract(architecture.semantic.contract,steps,certificate.resource_uses,
        certificate.leases,certificate.addresses,tuple(reuse),CONDITIONS+(
            'Targets are not timer waits: realize prerequisite completion, resource admission and lease release before reuse',))
    return ScheduleReceipt(architecture,plan,'scheduled_conditionally',lower,certificate.makespan_ns,terms,certificate,
        SearchStats(budget.states,horizon,exhausted,optimal,reason),implementation,CONDITIONS)


def cli(args):
    from pathlib import Path
    from .architecture import _dump
    import sys
    try:
        overrides={key:getattr(args,key) for key in ('strategy','max_states','horizon_ns','tick_ns') if getattr(args,key) is not None}
        if args.selftest:
            if args.plan or args.output or overrides:
                raise ScheduleError('Selftest does not consume a plan or write a receipt')
            from .schedule_test import selftest
            return selftest()
        if args.verify:
            if args.plan or overrides:
                raise ScheduleError('Receipt replay cannot override a frozen plan')
            from .architecture import parse_json
            raw_text=Path(args.verify).read_text();raw=parse_json(raw_text)
            if type(raw) is dict and 'loop' in raw:
                from .loops import LoopScheduleReceipt
                receipt=LoopScheduleReceipt.from_json(raw_text)
            else:receipt=ScheduleReceipt.from_json(raw_text)
            receipt.verify()
            text=_dump(dict(status='replayed_in_model',upper_ns=receipt.upper_ns,
                             scope='Conditional integer-time model; not numerical or hardware certification'),pretty=True)
        else:
            if args.plan and overrides:
                raise ScheduleError('Create a new plan instead of overriding a frozen one')
            if args.loop_receipt:
                from .loops import LoopReceipt
                loop=LoopReceipt.from_json(Path(args.loop_receipt).read_text());loop.verify()
                if loop.architecture is None:
                    if args.plan or overrides:raise ScheduleError('Scheduling parameters require a concrete loop expansion')
                    from .loops import schedule_loop
                    text=schedule_loop(loop).to_json(pretty=True)
                    if args.output:
                        with Path(args.output).open('x') as stream:stream.write(text+'\n')
                    else:print(text)
                    return 0
                a=loop.architecture
            else:a=Architecture.from_json(Path(args.input).read_text())
            plan=SchedulePlan.from_json(Path(args.plan).read_text()) if args.plan else SchedulePlan(a.fingerprint(),**overrides)
            if args.loop_receipt:
                from .loops import schedule_loop
                receipt=schedule_loop(loop,plan)
            else:receipt=solve(a,plan)
            text=receipt.to_json(pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return 0
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr)
        return 2


def _nc_unknown_is_not_zero():
    from .architecture import example
    a=example('matmul');result=solve(a,SchedulePlan(a.fingerprint()))
    return result.status=='unknown' and result.upper_ns is None and result.certificate is None


REFS={'SCHED1_conditional_certificates':Ref(
    what=['有限整数时间的依赖/共享资源/连续地址租约证书，区间L/U与有界搜索范围显式；未知不伪造零成本或无解。'],
    status='derived',evidence=['E103 schedule_test independent time/address enumeration and certificate counterexamples; not hardware calibration.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'SCHED1_conditional_certificates':_nc_unknown_is_not_zero}
