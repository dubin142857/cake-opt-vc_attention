"""Observed execution documents. Observed order is not required dataflow."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Literal
from .._refs import Ref
from .architecture import Attribute, Evidence, Inspection, Record, VersionedRecord, _dump


@dataclass(frozen=True)
class Artifact(Record):
    id: str
    kind: Literal['trace','sim_report','hardware','semantic_spec','layout_spec','kernel_source','binary','other']
    location: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class Provenance(Record):
    tool: str
    version: str
    artifacts: tuple[Artifact,...]
    scope: Literal['trace_only','with_source'] = 'trace_only'
    argv: tuple[str,...] = ()
    notes: tuple[str,...] = ()


@dataclass(frozen=True)
class ObservedAccess(Record):
    space: str
    mode: Literal['read','write','read_write']
    address: int | None = None
    span_bytes: int | None = None
    instance: str = ''


@dataclass(frozen=True)
class ObservedEvent(Record):
    id: str
    core: str
    pipe: str
    clock: str
    start_ns: int | float
    end_ns: int | float
    name: str
    source: str
    record: str
    launch: str | None = None
    architecture_event: str = ''
    accesses: tuple[ObservedAccess,...] = ()
    memory_kind: Literal['none','access','unknown'] = 'unknown'
    sync_kind: Literal['none','signal','wait','barrier','unknown'] = 'unknown'
    attributes: tuple[Attribute,...] = ()


@dataclass(frozen=True)
class ObservedRelation(Record):
    source: str
    target: str
    kind: Literal['observed_order','synchronization_candidate','address_candidate']
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Measurement(Record):
    id: str
    subject: str
    metric: str
    unit: str
    value: int | float | None
    source: str
    evidence: Evidence
    clock: str = ''
    launch: str | None = None


@dataclass(frozen=True)
class Coverage(Record):
    records_seen: int
    records_accepted: int
    records_unsupported: int = 0
    events_expected: int | None = None
    cores_expected: tuple[str,...] | None = None
    launches_expected: tuple[str,...] | None = None
    addresses: Literal['unknown','partial','declared_complete'] = 'unknown'
    synchronization: Literal['unknown','partial','declared_complete'] = 'unknown'
    notes: tuple[str,...] = ()


@dataclass(frozen=True)
class ObservedExecution(VersionedRecord):
    id: str
    events: tuple[ObservedEvent,...]
    relations: tuple[ObservedRelation,...]
    measurements: tuple[Measurement,...]
    provenance: Provenance
    coverage: Coverage
    architecture_fingerprint: str = ''
    schema_version: int = 1

    def canonical(self):
        data=super().canonical()
        for field in ('relations','measurements'):
            data[field]=sorted(data[field],key=_dump)
        data['provenance']['artifacts']=sorted(data['provenance']['artifacts'],key=_dump)
        return data


def validate_observation(o: ObservedExecution):
    c=Inspection();c.version(o)
    events=c.index(o.events,'observed event');artifacts=c.index(o.provenance.artifacts,'artifact')
    c.index(o.measurements,'measurement')
    if not o.provenance.tool or not o.provenance.version:
        c.error('provenance',o.id,'Tool and version are required')
    if not artifacts:
        c.unknown('provenance',o.id,'No input artifacts declared')
    for artifact in o.provenance.artifacts:
        if not artifact.location or not re.fullmatch('[0-9a-f]{64}',artifact.sha256) or artifact.size_bytes<0:
            c.error('artifact',artifact.id,'Location, exact hash and nonnegative byte size are required')
        if o.provenance.scope=='trace_only' and artifact.kind=='kernel_source':
            c.error('source_boundary',artifact.id,'Trace-only mode cannot declare a kernel source input')
    c.unknown('provenance_not_verified',o.id,'Manifest contents and actual reads are not authenticated by structural validation')
    if o.architecture_fingerprint:
        if not re.fullmatch('[0-9a-f]{64}',o.architecture_fingerprint):
            c.error('model_identity',o.id,'Malformed architecture fingerprint')
        c.unknown('mapping_not_verified',o.id,'Declared architecture and event/instance mappings require independent checks')
    for event in o.events:
        if not all((event.core,event.pipe,event.clock,event.name,event.record)) or event.source not in artifacts:
            c.error('event_reference',event.id,'Core, pipe, clock, name, record and valid source are required')
        if event.end_ns<event.start_ns:
            c.error('event_interval',event.id,'End precedes start')
        if event.launch is None or not event.launch:
            c.unknown('launch_boundary',event.id,'Launch is not identified; do not infer it from a time gap alone')
        if len({a.name for a in event.attributes})!=len(event.attributes):
            c.error('attribute',event.id,'Duplicate attribute name')
        if event.architecture_event and not o.architecture_fingerprint:
            c.error('model_identity',event.id,'Event mapping needs architecture identity')
        if event.memory_kind=='none' and event.accesses:
            c.error('memory_classification',event.id,'Non-memory event declares accesses')
        if event.memory_kind=='access' and not event.accesses:
            c.unknown('memory_access',event.id,'Memory instruction has no recovered accesses')
        for access in event.accesses:
            if not access.space or (access.address is not None and access.address<0) or (access.span_bytes is not None and access.span_bytes<0):
                c.error('memory_access',event.id,'Space and nonnegative address/span are required when known')
            if access.instance and not o.architecture_fingerprint:
                c.error('model_identity',event.id,'Instance mapping needs architecture identity')
    if not o.events:
        c.unknown('event_coverage',o.id,'No execution events observed')
    for relation in o.relations:
        if relation.source not in events or relation.target not in events or relation.source==relation.target:
            c.error('relation_reference',o.id,'Unknown or identical relation endpoints');continue
        left,right=events[relation.source],events[relation.target]
        if relation.kind=='observed_order':
            if left.clock!=right.clock:
                c.error('clock_domain',o.id,'Cannot assert temporal order across clock domains without alignment')
            if left.launch is not None and right.launch is not None and left.launch!=right.launch:
                c.error('launch_boundary',o.id,'Observed-order edges are scoped to one launch')
            if left.end_ns>right.start_ns:
                c.error('observed_order',o.id,'Intervals overlap or are reversed')
        else:
            c.unknown('relation_candidate',o.id,'Synchronization/address candidate is not a proven execution dependency')
    cov=o.coverage
    if min(cov.records_seen,cov.records_accepted,cov.records_unsupported)<0 or cov.records_seen!=cov.records_accepted+cov.records_unsupported or cov.records_accepted!=len(o.events):
        c.error('coverage_counts',o.id,'Eligible records must be accounted for; accepted count must equal event count')
    if cov.records_unsupported:
        c.unknown('unsupported_records',o.id,'Unsupported records retained as a coverage gap')
    if cov.events_expected is None:
        c.unknown('event_coverage',o.id,'Expected event count not supplied')
    elif cov.events_expected<0 or len(o.events)>cov.events_expected:
        c.error('coverage_counts',o.id,'Observed count exceeds stated expectation')
    elif len(o.events)<cov.events_expected:
        c.unknown('event_coverage',o.id,'Expected events are missing')
    for field,actual in (('cores_expected',{e.core for e in o.events}),('launches_expected',{e.launch for e in o.events if e.launch is not None})):
        expected=getattr(cov,field)
        if expected is None:
            c.unknown('scope_coverage',field,'Expected scope not supplied')
        elif len(set(expected))!=len(expected) or any(not x for x in expected) or actual-set(expected):
            c.error('scope_coverage',field,'Duplicate/empty expectation or unexpected observed scope')
        elif set(expected)-actual:
            c.unknown('scope_coverage',field,'Expected scopes are missing')
    if cov.addresses=='declared_complete':
        if cov.records_unsupported or any(e.memory_kind=='unknown' or (e.memory_kind=='access' and (not e.accesses or any(a.address is None or a.span_bytes is None for a in e.accesses))) for e in o.events):
            c.error('address_coverage',o.id,'Complete address coverage conflicts with missing classification/address/span')
    else:
        c.unknown('address_coverage',o.id,'Addresses are partial or unknown')
    if cov.synchronization=='declared_complete':
        endpoints={x for r in o.relations if r.kind=='synchronization_candidate' for x in (r.source,r.target)}
        if cov.records_unsupported or any(e.sync_kind=='unknown' or (e.sync_kind in ('signal','wait') and e.id not in endpoints) for e in o.events):
            c.error('synchronization_coverage',o.id,'Complete synchronization coverage conflicts with unknown/unpaired events')
    else:
        c.unknown('synchronization_coverage',o.id,'Synchronization is partial or unknown')
    c.unknown('semantics_unknown',o.id,'Observed timing/address/order alone cannot uniquely recover operator semantics or required dependencies')
    for measurement in o.measurements:
        if measurement.source not in artifacts or not all((measurement.subject,measurement.metric,measurement.unit)):
            c.error('measurement',measurement.id,'Measurement needs scope, metric, unit and a valid source')
        if measurement.value is None:
            c.unknown('measurement',measurement.id,'Value unavailable')
    return c.result()


def _nc_boundary():
    o=ObservedExecution('nc',(),(),(),Provenance('test','1',(Artifact('src','kernel_source','fixture://src','0'*64,0),)),Coverage(0,0))
    return any(x.code=='source_boundary' and x.severity=='error' for x in validate_observation(o).issues)


REFS={'OBS1_observation_not_dependency':Ref(
    what=['观测顺序不自动证明必要依赖；时钟/launch/来源/覆盖边界保持显式。'],
    status='derived',evidence=['E101 research_test observation counterexamples; P2 parsing and source-read audit remain pending.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'OBS1_observation_not_dependency':_nc_boundary}
