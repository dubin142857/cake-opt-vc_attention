"""Lossless snapshots and explicit refinement mappings for legacy aggregate IR.

The snapshot is not an executable Architecture. Aggregate stages do not provide
addresses, per-core events, synchronization or operator mathematics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .._refs import Ref
from .architecture import Architecture, Inspection, Record, VersionedRecord, validate


@dataclass(frozen=True)
class LegacyBuffer(Record):
    id: str
    space: str
    bytes: int
    double: bool
    lifetime: tuple[str,...] | None
    note: str


@dataclass(frozen=True)
class LegacyStage(Record):
    id: str
    unit: str
    reads: tuple[str,...]
    writes: tuple[str,...]
    macs: int
    elem_ops: int
    instr: int
    tile_bytes: int
    tasks: int
    scalar_ops: int
    note: str
    repeats: int


@dataclass(frozen=True)
class LegacyMove(Record):
    id: str
    src: str
    dst: str
    bytes: int
    times: int
    unique: int
    reuse_window: int | None
    note: str


@dataclass(frozen=True)
class LegacySummary(VersionedRecord):
    id: str
    soc: str | None
    note: str
    requires: tuple[str,...]
    buffers: tuple[LegacyBuffer,...]
    stages: tuple[LegacyStage,...]
    moves: tuple[LegacyMove,...]
    schema_version: int = 1


@dataclass(frozen=True)
class LegacyBinding(Record):
    kind: Literal['buffer','stage','move','capability']
    source: str
    targets: tuple[str,...]


@dataclass(frozen=True)
class LegacyAttachment(VersionedRecord):
    id: str
    summary: LegacySummary
    architecture_fingerprint: str
    bindings: tuple[LegacyBinding,...]
    schema_version: int = 1


def snapshot_legacy(schedule):
    from .ir import Schedule,Buf,Stage,Move
    expected={'name','soc','note','requires','bufs','stages','moves','stage_times'}
    if type(schedule) is not Schedule or set(vars(schedule))!=expected:
        raise ValueError('Unsupported legacy Schedule type/fields; refusing silent field loss')
    def slots(obj,cls,expected):
        if type(obj) is not cls or set(cls.__slots__)!=set(expected):
            raise ValueError('Unsupported legacy '+cls.__name__+' type/slots')
    buffers=[];stages=[];moves=[]
    for key,b in schedule.bufs.items():
        slots(b,Buf,('name','space','bytes','double','lifetime','note'))
        if key!=b.name:
            raise ValueError('Legacy buffer key differs from identity')
        lifetime=None if b.lifetime is None else tuple(sorted(b.lifetime))
        buffers.append(LegacyBuffer(b.name,b.space,b.bytes,b.double,lifetime,b.note))
    if len({s.name for s in schedule.stages})!=len(schedule.stages) or set(schedule.stage_times)!={s.name for s in schedule.stages}:
        raise ValueError('Duplicate legacy stages or mismatched repeat-count keys')
    for s in schedule.stages:
        slots(s,Stage,('name','unit','reads','writes','macs','elem_ops','instr','tile_bytes','tasks','scalar_ops','note'))
        stages.append(LegacyStage(s.name,s.unit,s.reads,s.writes,s.macs,s.elem_ops,s.instr,
                                  s.tile_bytes,s.tasks,s.scalar_ops,s.note,schedule.stage_times[s.name]))
    for m in schedule.moves:
        slots(m,Move,('name','src','dst','bytes','times','unique','reuse_window','note'))
        moves.append(LegacyMove(m.name,m.src,m.dst,m.bytes,m.times,m.unique,m.reuse_window,m.note))
    return LegacySummary(schedule.name,schedule.soc,schedule.note,schedule.requires,tuple(buffers),tuple(stages),tuple(moves))


def validate_legacy(summary: LegacySummary):
    s=summary;c=Inspection();c.version(s)
    buffers=c.index(s.buffers,'legacy buffer');stages=c.index(s.stages,'legacy stage');c.index(s.moves,'legacy move')
    if len(set(s.requires))!=len(s.requires) or any(not x for x in s.requires):
        c.error('legacy_capability',s.id,'Capabilities must be unique and nonempty')
    for b in s.buffers:
        if b.space not in ('GM','UB','L1','L0A','L0B','L0C','REG') or b.bytes<0:
            c.error('legacy_buffer',b.id,'Unknown space or negative bytes')
        if b.lifetime is not None and (len(set(b.lifetime))!=len(b.lifetime) or any(x not in stages for x in b.lifetime)):
            c.error('legacy_lifetime',b.id,'Duplicate/unknown lifetime stage')
    for stage in s.stages:
        if any(x not in buffers for x in stage.reads+stage.writes):
            c.error('legacy_reference',stage.id,'Read/write refers to unknown buffer')
        if min(stage.macs,stage.elem_ops,stage.instr,stage.tile_bytes,stage.tasks,stage.scalar_ops,stage.repeats)<0:
            c.error('legacy_work',stage.id,'Negative work or multiplicity')
        if stage.unit not in ('cube','vector'):
            c.error('legacy_unit',stage.id,'Unknown aggregate compute unit')
        if stage.tasks==0:
            c.unknown('legacy_tasks',stage.id,'Zero task count means unspecified in legacy IR, not measured zero work')
    for move in s.moves:
        if min(move.bytes,move.times,move.unique)<0 or (move.reuse_window is not None and move.reuse_window<0):
            c.error('legacy_move',move.id,'Negative movement fields')
        if not move.src or not move.dst:
            c.error('legacy_move',move.id,'Movement endpoints required')
    c.unknown('legacy_aggregate_only',s.id,'Snapshot preserves aggregates; no executable event graph, per-core mapping, alias protocol or numerical semantics is inferred')
    c.unknown('legacy_cost',s.id,'Work counts and requested/unique bytes are not event durations or performance bounds')
    return c.result()


def restore_legacy(summary: LegacySummary):
    """Explicit rollback to legacy aggregates; unknown semantic gaps remain."""
    from .ir import Schedule
    errors=[x for x in validate_legacy(summary).issues if x.severity=='error']
    if errors:
        raise ValueError('Cannot restore invalid legacy snapshot: '+str(errors))
    s=Schedule(summary.id,soc=summary.soc,note=summary.note,requires=summary.requires)
    for b in summary.buffers:
        restored=s.buf(b.id,b.space,b.bytes,double=b.double,lifetime=b.lifetime,note=b.note)
        restored.lifetime=None if b.lifetime is None else frozenset(b.lifetime)
    for stage in summary.stages:
        s.stage(stage.id,stage.unit,reads=stage.reads,writes=stage.writes,macs=stage.macs,
                elem_ops=stage.elem_ops,instr=stage.instr,tile_bytes=stage.tile_bytes,
                tasks=stage.tasks,scalar_ops=stage.scalar_ops,note=stage.note,times=stage.repeats)
    for move in summary.moves:
        s.move(move.id,move.src,move.dst,move.bytes,move.times,unique=move.unique,reuse_window=move.reuse_window,note=move.note)
    return s


def validate_attachment(attachment: LegacyAttachment, architecture: Architecture):
    c=Inspection();c.version(attachment)
    c.issues.extend(validate_legacy(attachment.summary).issues)
    c.issues.extend(validate(architecture).issues)
    if attachment.architecture_fingerprint!=architecture.fingerprint():
        c.error('attachment_identity',attachment.id,'Attachment is not bound to this architecture')
    s=attachment.summary
    sources={'buffer':{b.id:b for b in s.buffers},'stage':{x.id:x for x in s.stages},
             'move':{x.id:x for x in s.moves},'capability':{x:None for x in s.requires}}
    targets={'buffer':{x.id:x for x in architecture.allocations},'stage':{x.id:x for x in architecture.events},
             'move':{x.id:x for x in architecture.events},'capability':{x.id:x for x in architecture.claims}}
    seen=set()
    for binding in attachment.bindings:
        key=(binding.kind,binding.source)
        if key in seen or binding.source not in sources[binding.kind] or not binding.targets or len(set(binding.targets))!=len(binding.targets) or any(x not in targets[binding.kind] for x in binding.targets):
            c.error('attachment_reference',attachment.id,'Duplicate source, unknown identity or empty/duplicate targets');continue
        seen.add(key)
        if binding.kind=='buffer':
            b=sources['buffer'][binding.source]
            declared=sum(targets['buffer'][x].size_bytes for x in binding.targets)
            if declared<b.bytes*(2 if b.double else 1):
                c.error('attachment_capacity',binding.source,'Mapped declarations do not cover legacy live bytes; changed footprint needs a separate explicit refinement')
        c.unknown('attachment_not_proof',binding.source,'Correspondence is declared, not proven; ownership, lifecycle, multiplicity, dataflow and semantics require refinement checks')
    for kind,items in sources.items():
        for source in items:
            if (kind,source) not in seen:
                c.unknown('unmapped_legacy_item',kind+':'+source,'No correspondence supplied')
    return c.result()


def _nc_no_silent_loss():
    from .ir import Schedule
    s=Schedule('nc');s.extra_field='must not disappear'
    try:
        snapshot_legacy(s)
        return False
    except ValueError:
        return True


REFS={'LEG1_preserve_aggregate_and_gaps':Ref(
    what=['旧IR全部字段冻结；显式映射不自动恢复事件DAG、同步、语义或性能。'],
    status='derived',evidence=['E101 research_test all-field snapshots, unknown-field rejection and incomplete/incorrect attachments.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'LEG1_preserve_aggregate_and_gaps':_nc_no_silent_loss}
