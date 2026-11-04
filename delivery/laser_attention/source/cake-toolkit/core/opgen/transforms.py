"""P1 architecture transformations with preconditions and replayable receipts.

Reorder changes dispatch priority, not a dependency. BufferRename changes a
produced physical instance, releasing only demonstrably pure reuse constraints.
These are conditional architecture candidates, not measured kernel gains.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Literal

from .._refs import Ref
from .architecture import (Architecture, Allocation, Claim, Dependency, Duration,
    Evidence, Record, Region, Demand, Event, Binding, _dag, _dump, _encode, decode_versioned,
    layout_span_bytes, parse_json, validate)


REFS = {
    'TRANS1_identity_replay': Ref(
        what=['变换绑定源模型身份；版本与回执可重放，失败不发布部分组合。'],
        status='derived', evidence=['E100 transforms_test source/capacity/tampering counterexamples; docs/architecture-objects.md.'], verified='2026-11-03'),
    'TRANS2_dependency_preservation': Ref(
        what=['优先级不替代依赖；重命名需显式生产者，仅释放无剩余冲突及语义前提的纯复用边。'],
        status='derived', evidence=['E100 RAW/partial-overwrite/protected-edge counterexamples and independent functional byte replay; not a hardware proof.'], verified='2026-11-03'),
    'TRANS4_finite_fusion': Ref(
        what=['同actor有限凸组收缩保留内部依赖及逐字节外部输入；融合成本不能由成员实测继承。'],
        status='derived', evidence=['E109 finite fusion boundary/oracle/nonconvex/recurrence/tampering tests; real attention stage reconstruction.'], verified='2026-11-03'),
    'TRANS3_evidence_invalidation': Ref(
        what=['架构变化使旧时长和结论失效；显式沿用成本只能标为假设，不能继承实测认证。'],
        status='derived', evidence=['E100 cost/claim/dependency invalidation and no-op tests; source version retains prior evidence.'], verified='2026-11-03'),
}


@dataclass(frozen=True)
class Reorder(Record):
    priority: tuple[str, ...]
    kind: Literal['reorder'] = 'reorder'


@dataclass(frozen=True)
class BufferRename(Record):
    instance: str
    allocation: str
    release_reuse: bool = True
    kind: Literal['buffer_rename'] = 'buffer_rename'


@dataclass(frozen=True)
class FuseEvents(Record):
    events: tuple[str,...]
    event: str
    kind: Literal['fuse_events'] = 'fuse_events'


@dataclass(frozen=True)
class InternalRead(Record):
    event: str
    region: Region


@dataclass(frozen=True)
class FusionWitness(Record):
    members: tuple[Event,...]
    internal_dependencies: tuple[Dependency,...]
    internal_reads: tuple[InternalRead,...]
    boundary_reads: tuple[Region,...]
    boundary_writes: tuple[Region,...]


@dataclass(frozen=True)
class Compose(Record):
    operations: tuple[Reorder | BufferRename | FuseEvents | Compose, ...]
    kind: Literal['compose'] = 'compose'


Operation = Reorder | BufferRename | FuseEvents | Compose
CostPolicy = Literal['invalidate', 'assume_unchanged']


@dataclass(frozen=True)
class TransformPlan(Record):
    source_fingerprint: str
    operation: Operation
    cost_policy: CostPolicy = 'invalidate'
    schema_version: int = 1

    def to_json(self, *, pretty=False):
        return _dump(_encode(self), pretty=pretty)

    @classmethod
    def from_json(cls, text):
        raw = parse_json(text)
        if type(raw) is dict:
            _operation_tags(raw.get('operation'))
        return decode_versioned(cls, raw)


@dataclass(frozen=True)
class Obligation(Record):
    rule: str
    status: Literal['satisfied_in_model', 'pending']
    statement: str


@dataclass(frozen=True)
class Invalidation(Record):
    subject: str
    prior: Evidence | Duration
    reason: str


@dataclass(frozen=True)
class ByteDelta(Record):
    storage: str
    declared_before_bytes: int
    declared_after_bytes: int
    delta_bytes: int


@dataclass(frozen=True)
class KeptReuse(Record):
    dependency: Dependency
    reason: str


@dataclass(frozen=True)
class Step(Record):
    operation: Reorder | BufferRename | FuseEvents
    before_fingerprint: str
    after_fingerprint: str
    removed_dependencies: tuple[Dependency, ...]
    kept_reuse: tuple[KeptReuse, ...]
    allocation_delta: tuple[ByteDelta, ...]
    invalidated: tuple[Invalidation, ...]
    obligations: tuple[Obligation, ...]
    fusion: FusionWitness | None = None


@dataclass(frozen=True)
class TransformReceipt(Record):
    plan: TransformPlan
    versions: tuple[Architecture, ...]
    steps: tuple[Step, ...]
    status: Literal['applied_conditionally'] = 'applied_conditionally'
    schema_version: int = 1

    @property
    def architecture(self):
        return self.versions[-1]

    def to_json(self, *, pretty=False):
        return _dump(_encode(self), pretty=pretty)

    @classmethod
    def from_json(cls, text):
        raw = parse_json(text)
        if type(raw) is dict:
            if type(raw.get('plan')) is dict:
                _operation_tags(raw['plan'].get('operation'))
                decode_versioned(TransformPlan, raw['plan'])
            if type(raw.get('versions')) is list:
                for version in raw['versions']:
                    decode_versioned(Architecture, version)
            if type(raw.get('steps')) is list:
                for step in raw['steps']:
                    if type(step) is dict:
                        _operation_tags(step.get('operation'))
        return decode_versioned(cls, raw)

    def verify(self):
        if not self.versions:
            raise TransformError('receipt', 'Receipt has no source version')
        expected = apply_plan(self.versions[0], self.plan)
        if expected != self:
            raise TransformError('receipt', 'Receipt differs from deterministic replay')
        return True


class TransformError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def _operation_tags(raw):
    if type(raw) is not dict or raw.get('kind') not in ('reorder','buffer_rename','fuse_events','compose'):
        raise ValueError('Every operation requires an explicit known kind')
    if raw['kind'] == 'compose':
        if type(raw.get('operations')) is not list or not raw['operations']:
            raise ValueError('Compose requires a nonempty operations array')
        for item in raw['operations']:
            _operation_tags(item)


def _leaves(operation):
    if isinstance(operation, Compose):
        if not operation.operations:
            raise TransformError('empty_composition', 'Compose requires operations')
        for item in operation.operations:
            yield from _leaves(item)
    elif isinstance(operation, (Reorder, BufferRename, FuseEvents)):
        yield operation
    else:
        raise TransformError('operation', 'Unsupported operation')


def _require_consistent(architecture):
    result = validate(architecture)
    errors = [x.code+': '+x.message for x in result.issues if x.severity=='error']
    if errors:
        raise TransformError('model', '; '.join(errors))
    return result


def _accesses(architecture, event):
    """Precondition: consistent model. Allocation identities imply distinct storage."""
    instances = {x.id:x for x in architecture.instances}
    out=[]
    for write, regions in ((False,event.reads),(True,event.writes)):
        for region in regions:
            if region.span_bytes:
                instance=instances[region.instance]
                start=instance.offset_bytes+region.offset_bytes
                out.append((instance.allocation,start,start+region.span_bytes,write,instance.id))
    return tuple(out)


def _conflicts(architecture, left, right):
    return tuple((a,b) for a in _accesses(architecture,left) for b in _accesses(architecture,right)
                 if a[0]==b[0] and (a[3] or b[3]) and max(a[1],b[1])<min(a[2],b[2]))


def _fully_covered(start,end,intervals):
    covered=start
    for lo,hi in sorted(intervals):
        if lo>covered:
            break
        covered=max(covered,hi)
    return covered>=end


def _raw_reaches(architecture,left,right,a,b,ancestors):
    """Is any overlapping byte still supplied by this write at the read?

    A later write kills an earlier one only over its explicit byte interval and
    only when dependency paths guarantee it lies between that write and read.
    """
    start,end=max(a[1],b[1]),min(a[2],b[2]);overwrites=[]
    for event in architecture.events:
        if event.id in (left.id,right.id) or left.id not in ancestors[event.id] or event.id not in ancestors[right.id]:
            continue
        for access in _accesses(architecture,event):
            if access[3] and access[0]==a[0] and access[2]>start and access[1]<end:
                overwrites.append((max(start,access[1]),min(end,access[2])))
    return not _fully_covered(start,end,overwrites)


def _producer_precondition(architecture, instance_id):
    """A mathematical input does not initialize every physical copy of it."""
    events={x.id:x for x in architecture.events}
    _, ancestors=_dag(events,[(x.source,x.target) for x in architecture.dependencies if x.iteration_distance==0])
    writers=[(e.id,r) for e in architecture.events for r in e.writes if r.instance==instance_id and r.span_bytes]
    if not writers:
        raise TransformError('initialization_required','Renamed instance needs an explicit local producer; external data is not copied implicitly')
    for reader in architecture.events:
        for region in reader.reads:
            if region.instance!=instance_id or not region.span_bytes:
                continue
            end=region.offset_bytes+region.span_bytes;covered=region.offset_bytes
            intervals=sorted((max(region.offset_bytes,w.offset_bytes),min(end,w.offset_bytes+w.span_bytes))
                             for eid,w in writers if eid in ancestors[reader.id] and w.offset_bytes+w.span_bytes>region.offset_bytes and w.offset_bytes<end)
            for lo,hi in intervals:
                if lo>covered:
                    break
                covered=max(covered,hi)
            if covered<end:
                raise TransformError('initialization_required','Each renamed read must have a local preceding writer; recurrence seeds need explicit handling')
    # Cross-instance RAW aliases are dataflow, not merely buffer reuse.
    for left in architecture.events:
        for right in architecture.events:
            for a,b in _conflicts(architecture,left,right):
                if (a[4]==instance_id)==(b[4]==instance_id) or a[4]==b[4]:
                    continue
                if left.id==right.id:
                    raise TransformError('intra_event_alias','Renaming an in-place cross-view access needs operation-specific proof')
                if a[3] and not b[3] and left.id in ancestors[right.id] and _raw_reaches(architecture,left,right,a,b,ancestors):
                    raise TransformError('alias_dataflow','Reaching cross-view read-after-write requires explicit copy/remapping, even if an edge is labelled reuse')


def _event_values(architecture,event):
    instances={x.id:x for x in architecture.instances}
    return set(event.semantic_values)|{instances[r.instance].value for r in event.writes}


def _rename(architecture, operation):
    instances={x.id:x for x in architecture.instances};allocations={x.id:x for x in architecture.allocations}
    if operation.instance not in instances:
        raise TransformError('instance','Unknown instance')
    if not operation.allocation or operation.allocation in allocations:
        raise TransformError('allocation','New allocation must have a fresh nonempty identity')
    selected=instances[operation.instance];_producer_precondition(architecture,selected.id)
    old=allocations[selected.allocation];values={x.id:x for x in architecture.semantic.values}
    span=layout_span_bytes(selected.layout,values[selected.value].dtype)
    new=Allocation(operation.allocation,old.storage,span,old.persistent)
    candidate=replace(architecture,allocations=architecture.allocations+(new,),
                      instances=tuple(replace(x,allocation=new.id,offset_bytes=0) if x.id==selected.id else x for x in architecture.instances))
    events={x.id:x for x in architecture.events};removed=[];kept=[];dependencies=[]
    _, sem_ancestors=_dag(values,[(p,x.id) for x in architecture.semantic.values for p in x.inputs])
    for dep in architecture.dependencies:
        if dep.kind!='reuse' or dep.iteration_distance or not operation.release_reuse:
            dependencies.append(dep)
            continue
        left,right=events[dep.source],events[dep.target]
        before=_conflicts(architecture,left,right);after=_conflicts(candidate,left,right)
        reason=''
        source_values=_event_values(architecture,left);target_values=_event_values(architecture,right)
        if not before:
            reason='No original physical conflict proves this is a removable reuse edge'
        elif after:
            reason='Physical conflicts remain'
        elif any(a[3] and not b[3] for a,b in before):
            raise TransformError('raw_reuse','Read-after-write cannot be released as a reuse constraint')
        elif not source_values or not target_values:
            reason='Semantic roles are incomplete'
        elif any(x==y or x in sem_ancestors[y] for x in source_values for y in target_values):
            reason='A semantic predecessor relation remains'
        if reason:
            dependencies.append(dep);kept.append(KeptReuse(dep,reason))
        else:
            removed.append(dep)
    candidate=replace(candidate,dependencies=tuple(dependencies))
    return candidate,tuple(removed),tuple(kept)


def _regions(rows):
    """Canonical union for one logical physical instance; never merge aliases."""
    by_instance={}
    for region in rows:
        if region.span_bytes:by_instance.setdefault(region.instance,[]).append((region.offset_bytes,region.offset_bytes+region.span_bytes))
    out=[]
    for identity,intervals in sorted(by_instance.items()):
        merged=[]
        for lo,hi in sorted(intervals):
            if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
            else:merged.append((lo,hi))
        out.extend(Region(identity,lo,hi-lo) for lo,hi in merged)
    return tuple(out)


def _uncovered(region,written):
    pieces=[(region.offset_bytes,region.offset_bytes+region.span_bytes)] if region.span_bytes else []
    for prior in written:
        if prior.instance!=region.instance:continue
        lo,hi=prior.offset_bytes,prior.offset_bytes+prior.span_bytes;remaining=[]
        for left,right in pieces:
            if left<min(right,lo):remaining.append((left,min(right,lo)))
            if max(left,hi)<right:remaining.append((max(left,hi),right))
        pieces=remaining
    return tuple(Region(region.instance,lo,hi-lo) for lo,hi in pieces)


def _fuse(architecture,operation):
    table={e.id:e for e in architecture.events};ids=operation.events;chosen=set(ids)
    if len(ids)<2 or len(chosen)!=len(ids) or not chosen.issubset(table):
        raise TransformError('fusion_members','Fusion requires at least two distinct known events')
    if not operation.event or operation.event in table:raise TransformError('fusion_identity','Fused event requires a fresh nonempty ID')
    members=tuple(table[e] for e in ids);rank={e:i for i,e in enumerate(ids)}
    if len({e.actor for e in members})!=1:raise TransformError('fusion_actor','Cross-actor fusion needs an ownership/lowering proof')
    internal=[];edges=[]
    for dep in architecture.dependencies:
        source,target=dep.source in chosen,dep.target in chosen
        if dep.iteration_distance and (source or target):raise TransformError('fusion_recurrence','Expand recurrence and seeds before fusion')
        if source and target:
            if rank[dep.source]>=rank[dep.target]:raise TransformError('fusion_order','Internal execution order violates a dependency')
            internal.append(dep)
        else:edges.append(replace(dep,source=operation.event if source else dep.source,target=operation.event if target else dep.target))
    new_ids=(set(table)-chosen)|{operation.event}
    cyclic,_=_dag(new_ids,[(e.source,e.target) for e in edges if not e.iteration_distance])
    if cyclic:raise TransformError('fusion_nonconvex','Contraction would require an external event between fused members')
    # A read is internal only for bytes written earlier in the given member order.
    # Reading then writing the same instance must keep the external read.
    writes=[];reads=[];inside=[]
    for event in members:
        for region in event.reads:
            outside=_uncovered(region,writes);reads.extend(outside)
            inside.extend(InternalRead(event.id,r) for r in _uncovered(region,outside))
        writes.extend(event.writes)
    boundary_reads,boundary_writes=_regions(reads),_regions(writes)
    demands={}
    for event in members:
        for demand in event.resources:demands[demand.resource]=max(demands.get(demand.resource,0),demand.units)
    semantics=tuple(dict.fromkeys(v for e in members for v in e.semantic_values))
    fused=Event(operation.event,members[0].actor,'compute',boundary_reads,boundary_writes,
                tuple(Demand(r,n) for r,n in sorted(demands.items())),semantics,
                Duration(evidence=Evidence('unknown',architecture.fingerprint(),'fused execution','New fused implementation needs a new cost; internal member durations are not its duration')),
                'Ordered macro body: '+', '.join(ids))
    # Keep allocation/instance identities and every exposed write. This operation
    # alone proves no intermediate materialization or memory traffic was removed.
    result=[];added=False
    for event in architecture.events:
        if event.id in chosen:
            if not added:result.append(fused);added=True
        else:result.append(event)
    def mapped(values):return tuple(dict.fromkeys(operation.event if v in chosen else v for v in values))
    candidate=replace(architecture,events=tuple(result),dependencies=tuple(dict.fromkeys(edges)),
        bindings=tuple(replace(b,events=mapped(b.events)) for b in architecture.bindings),
        dispatch_priority=mapped(architecture.dispatch_priority))
    invalid=[]
    for event in members:
        prior=event.duration
        if prior.lower_ns is not None or prior.upper_ns is not None or prior.evidence.kind!='unknown':
            invalid.append(Invalidation('event:'+event.id+'.duration',prior,'Member execution was coalesced; no cost is transferred to the fused event'))
    for dep in internal:
        if dep.evidence.kind in ('measured','derived'):
            invalid.append(Invalidation('dependency:'+dep.source+'->'+dep.target,dep.evidence,'Internal order retained as a lowering obligation, not an observation of the new body'))
    witness=FusionWitness(members,tuple(internal),tuple(inside),boundary_reads,boundary_writes)
    return candidate,witness,tuple(invalid)


def _invalidate(before, candidate, policy):
    if candidate.fingerprint()==before.fingerprint():
        return candidate,()
    invalid=[];events=[];base=before.fingerprint()
    for event in candidate.events:
        prior=event.duration
        if prior.lower_ns is not None or prior.upper_ns is not None or prior.evidence.kind!='unknown':
            invalid.append(Invalidation('event:'+event.id+'.duration',prior,'Architecture changed; previous measurement/estimate is not a new observation'))
            if policy=='assume_unchanged':
                evidence=Evidence('assumed',base+'#event:'+event.id,'transformed model',
                                  'Prior service interval reused conditionally; recalibration and sensitivity required')
                duration=Duration(prior.lower_ns,prior.upper_ns,evidence)
            else:
                duration=Duration(evidence=Evidence('unknown',base+'#event:'+event.id,'transformed model','Prior duration invalidated; original retained in source version'))
            event=replace(event,duration=duration)
        events.append(event)
    claims=[]
    for claim in candidate.claims:
        if claim.evidence.kind!='unknown':
            invalid.append(Invalidation('claim:'+claim.id,claim.evidence,'Prior claim must be revalidated on changed architecture'))
            claim=replace(claim,evidence=Evidence('unknown',claim.evidence.reference,claim.evidence.scope,'Invalidated by architecture transformation; source version retains original'))
        claims.append(claim)
    dependencies=[]
    for dep in candidate.dependencies:
        if dep.evidence.kind in ('measured','derived'):
            subject=f'dependency:{dep.source}->{dep.target}:{dep.kind}:{dep.iteration_distance}'
            invalid.append(Invalidation(subject,dep.evidence,'Constraint retained; old implementation observation is not a new observation'))
            dep=replace(dep,evidence=Evidence('assumed',dep.evidence.reference,dep.evidence.scope,
                                             'Dependency retained as a model constraint; transformed implementation must verify it'))
        dependencies.append(dep)
    return replace(candidate,events=tuple(events),claims=tuple(claims),dependencies=tuple(dependencies)),tuple(invalid)


def _byte_delta(before,after):
    result=[]
    for storage in sorted({x.id for x in before.storages}|{x.id for x in after.storages}):
        old=sum(x.size_bytes for x in before.allocations if x.storage==storage)
        new=sum(x.size_bytes for x in after.allocations if x.storage==storage)
        result.append(ByteDelta(storage,old,new,new-old))
    return tuple(result)


def apply_plan(architecture: Architecture, plan: TransformPlan) -> TransformReceipt:
    if plan.schema_version!=1:
        raise TransformError('schema_version','Unsupported transform plan version')
    if not re.fullmatch('[0-9a-f]{64}',plan.source_fingerprint) or architecture.fingerprint()!=plan.source_fingerprint:
        raise TransformError('source_identity','Plan is not bound to this source model')
    _require_consistent(architecture)
    versions=[architecture];steps=[]
    for operation in _leaves(plan.operation):
        before=versions[-1];fusion=None;coalesced=()
        if isinstance(operation,Reorder):
            if len(operation.priority)!=len(before.events) or set(operation.priority)!={x.id for x in before.events}:
                raise TransformError('priority','Priority must name all events exactly once')
            candidate=replace(before,dispatch_priority=operation.priority);removed=();kept=()
        elif isinstance(operation,FuseEvents):
            candidate,fusion,coalesced=_fuse(before,operation);removed=();kept=()
        else:
            candidate,removed,kept=_rename(before,operation)
        candidate,invalid=_invalidate(before,candidate,plan.cost_policy)
        invalid=coalesced+invalid
        _require_consistent(candidate)
        if candidate.semantic!=before.semantic:
            raise TransformError('semantic_change','These operations must preserve the complete semantic object')
        if any(dep.kind!='reuse' or dep.iteration_distance for dep in removed):
            raise TransformError('dependency_change','Only proved zero-distance reuse edges can be removed')
        obligations=[Obligation('semantic_object','satisfied_in_model','Semantic object and numerical contract are unchanged'),
                     Obligation('structural_constraints','satisfied_in_model','No contradiction in checks covered by the concrete structural model'),
                     Obligation('implementation','pending','Lowering must realize the declared allocation identities, ownership and dependency protocol'),
                     Obligation('numerics','pending','Validate the declared contract on the generated implementation'),
                     Obligation('cost','pending','Conditional costs are not measurements; calibrate before claiming performance')]
        if removed:
            obligations.append(Obligation('reuse_disjointness','satisfied_in_model','Original conflicts are pure WAR/WAW; changed allocations no longer overlap and no semantic prerequisite is removed'))
        if fusion is not None:
            obligations.extend((Obligation('fusion_order','satisfied_in_model','Finite same-actor convex contraction preserves explicit internal order and boundary bytes'),
                                Obligation('fusion_lowering','pending','Implement the retained ordered member semantics; boundary contraction is not code generation or a numerical proof'),
                                Obligation('fusion_materialization','pending','Allocations are retained; any removal or relocation needs its own storage/alias proof')))
        if isinstance(operation,BufferRename):
            obligations.append(Obligation('allocation_distinctness','pending','Implementation must allocate distinct storage and preserve initialization'))
            selected=next(x for x in candidate.instances if x.id==operation.instance)
            if selected.value in candidate.semantic.outputs:
                obligations.append(Obligation('external_output_binding','pending','Reconcile changed output storage with the public interface'))
        if any(dep.iteration_distance for dep in candidate.dependencies):
            obligations.append(Obligation('recurrence','pending','Expand positive-distance edges and prove initialization/drain behavior'))
        steps.append(Step(operation,before.fingerprint(),candidate.fingerprint(),removed,kept,
                          _byte_delta(before,candidate),invalid,tuple(obligations),fusion))
        versions.append(candidate)
    if not steps:
        raise TransformError('empty_composition','No operation to apply')
    return TransformReceipt(plan,tuple(versions),tuple(steps))


def cli_transform(args):
    from pathlib import Path
    import sys
    try:
        if args.verify_transform:
            if args.transform:
                raise ValueError('--verify-transform cannot apply another transformation')
            receipt=TransformReceipt.from_json(Path(args.verify_transform).read_text())
            receipt.verify()
            text=_dump(dict(status='replayed_in_model',
                            model_fingerprint=receipt.architecture.fingerprint(),
                            steps=len(receipt.steps),
                            scope='Consistency of the supplied model and receipt; not external provenance, numerical or hardware certification'),pretty=True)
        else:
            if not args.input or args.example or args.selftest:
                raise ValueError('--transform requires --input only')
            source=Architecture.from_json(Path(args.input).read_text())
            plan=TransformPlan.from_json(Path(args.transform).read_text())
            receipt=apply_plan(source,plan);receipt.verify()
            text=receipt.to_json(pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:
                stream.write(text+'\n')
        else:
            print(text)
        return 0
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr)
        return 2



def _nc_unknown_fusion_member():
    from .architecture import example
    a=example('matmul')
    try:apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('missing.a','missing.b'),'fused')))
    except TransformError as error:return error.code=='fusion_members'
    return False


def _nc_identity_replay():
    from .architecture import example
    a=example('matmul');operation=Reorder(tuple(e.id for e in a.events))
    try:
        apply_plan(a,TransformPlan('0'*64,operation))
        return False
    except TransformError as exc:
        if exc.code!='source_identity':
            return False
    receipt=apply_plan(a,TransformPlan(a.fingerprint(),operation))
    try:
        replace(receipt,steps=()).verify()
        return False
    except TransformError as exc:
        return exc.code=='receipt'


def _nc_dependency_preservation():
    from .architecture import example
    a=example('dense_attention')
    try:
        apply_plan(a,TransformPlan(a.fingerprint(),BufferRename('q','new_q')))
        return False
    except TransformError as exc:
        if exc.code!='initialization_required':
            return False
    result=apply_plan(a,TransformPlan(a.fingerprint(),BufferRename('score','new_score')))
    return result.architecture.dependencies==a.dependencies and result.architecture.semantic==a.semantic


def _nc_evidence_invalidation():
    from .architecture import example
    a=example('matmul');event=a.events[0]
    a=replace(a,events=(replace(event,duration=Duration(10,10,Evidence('measured','fixture://old','synthetic'))),))
    result=apply_plan(a,TransformPlan(a.fingerprint(),BufferRename('out','new_out')))
    return (result.architecture.events[0].duration.upper_ns is None
            and result.architecture.events[0].duration.evidence.kind=='unknown'
            and result.versions[0].events[0].duration.evidence.kind=='measured')


NEGATIVE_CONTROLS = {
    'TRANS4_finite_fusion': _nc_unknown_fusion_member,
    'TRANS1_identity_replay': _nc_identity_replay,
    'TRANS2_dependency_preservation': _nc_dependency_preservation,
    'TRANS3_evidence_invalidation': _nc_evidence_invalidation,
}
