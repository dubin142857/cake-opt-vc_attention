"""Versioned architecture objects: explicit identities, regions, events and resources.

P1 scope: immutable concrete scenarios and structural consistency. Validation is
not a numerical proof, a hardware certification or a performance prediction.
Unknown operations require an explicit external reference; unknown observations
remain conditional. No kernel source, simulator or device is accessed here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from collections import defaultdict, deque
import hashlib
import json
import math
import types
import typing
from typing import Literal, Union, get_args, get_origin, get_type_hints

# 来源: type storage definitions, bytes per scalar; not hardware throughput.
DTYPE_BYTES = {"bool": 1, "i8": 1, "u8": 1, "f16": 2, "bf16": 2,
               "i32": 4, "f32": 4, "i64": 8, "f64": 8}
Scalar = str | int | float | bool | None

from .._refs import Ref
REFS = {
    "ARCH1_symbol_identity": Ref(
        what=["独立输入符号必须有独立身份，显示标签不改变数学变量。"],
        status="measured", evidence=["VC E99: old equal-shape q/k collapsed to one G_math node; env/e99-legacy-input-identity.log and architecture_test regression."], verified="2026-11-03"),
    "ARCH2_structural_constraints": Ref(
        what=["具体场景检查引用、DAG、区域边界、别名顺序、所有权与显式常驻容量；不作数值/性能证明。"],
        status="derived", evidence=["Versioned P1 object contract; architecture_test positive/negative scenarios and docs/architecture-objects.md."], verified="2026-11-03"),
    "ARCH4_value_clobber": Ref(
        what=["有序alias访问仍须检查旧值是否被覆盖；有限DAG中clobber后需覆盖刷新，同事件与未展开回环保持unknown。"],
        status="measured", evidence=["VC E114/F507-F508: sorted→acc phase reuse; CPU wrong order corrupts511 values while prior validator reports no errors. E115 architecture_test positive/partial-refresh/input/view/recurrence cases."], verified="2026-11-03"),
    "ARCH3_unknown_not_proof": Ref(
        what=["缺失成本、未核验来源、opaque语义和未展开回环必须保持unknown/conditional。"],
        status="derived", evidence=["P1 Validation/Issue model; architecture_test counterexamples; docs/architecture-evolution.md."], verified="2026-11-03"),
}


def _typed(tp, value, path, *, json_input):
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):
        for candidate in args:
            try:
                return _typed(candidate, value, path, json_input=json_input)
            except (ValueError, TypeError):
                continue
        raise ValueError(f"{path}: value does not match {tp}")
    if origin is Literal:
        if any(type(value) is type(x) and value == x for x in args):
            return value
        raise ValueError(f"{path}: expected one of {args}")
    if origin is tuple:
        expected = list if json_input else tuple
        if type(value) is not expected:
            raise ValueError(f"{path}: expected {expected.__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_typed(args[0], x, f"{path}[{i}]", json_input=json_input) for i, x in enumerate(value))
        if len(args) != len(value):
            raise ValueError(f"{path}: tuple length mismatch")
        return tuple(_typed(t, x, path, json_input=json_input) for t, x in zip(args, value))
    if is_dataclass(tp):
        if not json_input:
            if type(value) is not tp:
                raise ValueError(f"{path}: expected {tp.__name__}")
            return value
        if type(value) is not dict:
            raise ValueError(f"{path}: expected object")
        expected = {f.name for f in fields(tp)}
        if 'schema_version' in expected and ('schema_version' not in value or type(value['schema_version']) is not int or value['schema_version'] != 1):
            raise ValueError(f"{path}: explicit supported schema_version is required")
        extra = set(value) - expected
        if extra:
            raise ValueError(f"{path}: unknown fields {sorted(extra)}")
        hints = get_type_hints(tp)
        return tp(**{k: _typed(hints[k], v, f"{path}.{k}", json_input=True) for k, v in value.items()})
    if type(value) is not tp:
        raise ValueError(f"{path}: expected {tp}")
    if tp is float and not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite number")
    return value


class Record:
    def __post_init__(self):
        for key, tp in get_type_hints(type(self)).items():
            _typed(tp, getattr(self, key), key, json_input=False)


@dataclass(frozen=True)
class Evidence(Record):
    kind: Literal["measured", "derived", "assumed", "unknown"] = "unknown"
    reference: str = ""
    scope: str = ""
    note: str = "Not supplied"


@dataclass(frozen=True)
class Attribute(Record):
    name: str
    value: Scalar


@dataclass(frozen=True)
class Contract(Record):
    id: str
    kind: Literal["exact", "approximate"]
    reference: str
    domain: str
    error_policy: str = ""


@dataclass(frozen=True)
class Value(Record):
    id: str
    op: str
    shape: tuple[int, ...]
    dtype: str
    inputs: tuple[str, ...] = ()
    attributes: tuple[Attribute, ...] = ()
    external_reference: str = ""
    label: str = ""


@dataclass(frozen=True)
class SemanticProgram(Record):
    values: tuple[Value, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    contract: Contract


@dataclass(frozen=True)
class Storage(Record):
    id: str
    kind: Literal["GM", "UB", "L1", "L0A", "L0B", "L0C", "REG"]
    owner: str
    capacity_bytes: int | None = None
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Resource(Record):
    id: str
    owner: str  # Actor identity; reserved global explicitly names a shared pool.
    capacity_units: int = 1
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Allocation(Record):
    id: str
    storage: str
    size_bytes: int
    persistent: bool = True


@dataclass(frozen=True)
class Layout(Record):
    extents: tuple[int, ...]
    strides_bytes: tuple[int, ...]


@dataclass(frozen=True)
class Instance(Record):
    id: str
    value: str
    allocation: str
    layout: Layout
    offset_bytes: int = 0
    logical_origin: tuple[int, ...] = ()


@dataclass(frozen=True)
class Region(Record):
    instance: str
    offset_bytes: int
    span_bytes: int


@dataclass(frozen=True)
class Demand(Record):
    resource: str
    units: int = 1


@dataclass(frozen=True)
class Duration(Record):
    lower_ns: int | None = None
    upper_ns: int | None = None
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Event(Record):
    id: str
    actor: str
    kind: Literal["compute", "copy", "synchronize"]
    reads: tuple[Region, ...] = ()
    writes: tuple[Region, ...] = ()
    resources: tuple[Demand, ...] = ()
    semantic_values: tuple[str, ...] = ()
    duration: Duration = Duration()
    label: str = ""


@dataclass(frozen=True)
class Dependency(Record):
    source: str
    target: str
    kind: Literal["data", "program", "synchronization", "reuse"]
    iteration_distance: int = 0
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Binding(Record):
    value: str
    events: tuple[str, ...]
    instances: tuple[str, ...]


@dataclass(frozen=True)
class Claim(Record):
    id: str
    statement: str
    evidence: Evidence


@dataclass(frozen=True)
class Platform(Record):
    soc: str = "unknown"
    software: tuple[Attribute, ...] = ()
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Architecture(Record):
    id: str
    semantic: SemanticProgram
    storages: tuple[Storage, ...]
    resources: tuple[Resource, ...]
    allocations: tuple[Allocation, ...]
    instances: tuple[Instance, ...]
    events: tuple[Event, ...]
    dependencies: tuple[Dependency, ...]
    bindings: tuple[Binding, ...] = ()
    claims: tuple[Claim, ...] = ()
    dispatch_priority: tuple[str, ...] = ()
    platform: Platform = Platform()
    schema_version: int = 1
    label: str = ""

    def fingerprint(self):
        return hashlib.sha256(_dump(_encode(self, canonical=True)).encode()).hexdigest()

    def to_json(self, *, pretty=False):
        return _dump(_encode(self), pretty=pretty)

    @classmethod
    def from_json(cls, text):
        return decode_versioned(cls, parse_json(text))


def parse_json(text):
    """Read JSON without duplicate-key or nonfinite-number fallback."""
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    def constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=constant)


def decode_versioned(cls, raw):
    if type(raw) is not dict or 'schema_version' not in raw:
        raise ValueError('Explicit schema_version is required')
    obj = _typed(cls, raw, cls.__name__, json_input=True)
    if obj.schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {obj.schema_version}")
    return obj


def layout_span_bytes(layout, dtype):
    """Conservative physical span; holes remain part of the interval."""
    width = DTYPE_BYTES[dtype]
    if any(x == 0 for x in layout.extents):
        return 0
    return width + sum((n - 1) * stride for n, stride in zip(layout.extents, layout.strides_bytes))


def _encode(obj, *, canonical=False):
    if is_dataclass(obj):
        result = {}
        unordered = {"values", "attributes", "software", "storages", "resources", "allocations", "instances",
                     "events", "dependencies", "bindings", "claims"}
        for field in fields(obj):
            if canonical and field.name == "label":
                continue
            value = _encode(getattr(obj, field.name), canonical=canonical)
            if canonical and field.name in unordered and isinstance(value, list):
                value = sorted(value, key=_dump)
            result[field.name] = value
        return result
    if isinstance(obj, tuple):
        return [_encode(x, canonical=canonical) for x in obj]
    return obj


def _dump(obj, *, pretty=False):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      indent=2 if pretty else None, separators=None if pretty else (",", ":"))


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "unknown"]
    code: str
    subject: str
    message: str


@dataclass(frozen=True)
class Validation:
    issues: tuple[Issue, ...]

    @property
    def status(self):
        if any(x.severity == "error" for x in self.issues):
            return "invalid"
        return "conditional" if self.issues else "structurally_consistent"

    def to_dict(self):
        return dict(status=self.status, scope="Structural checks only; no numerical, hardware or performance proof",
                    issues=[_encode(x) for x in self.issues])


class VersionedRecord(Record):
    """Shared strict codec for new research documents; identity includes all claims."""
    def to_json(self, *, pretty=False):
        return _dump(_encode(self), pretty=pretty)

    @classmethod
    def from_json(cls, text):
        return decode_versioned(cls, parse_json(text))

    def canonical(self):
        return _encode(self, canonical=True)

    def fingerprint(self):
        return hashlib.sha256(_dump(self.canonical()).encode()).hexdigest()


class Inspection:
    """Shared structural reporting; declared evidence is never authenticated here."""
    def __init__(self):
        self.issues=[]

    def error(self,code,subject,message):
        self.issues.append(Issue('error',code,subject,message))

    def unknown(self,code,subject,message):
        self.issues.append(Issue('unknown',code,subject,message))

    def index(self,records,kind):
        out={}
        for record in records:
            if not record.id or record.id in out:
                self.error('identity',kind,'Empty or duplicate ID')
            out[record.id]=record
        return out

    def version(self,obj):
        if obj.schema_version!=1:
            self.error('schema_version',obj.id,'Unsupported schema version')
        if not obj.id:
            self.error('identity','document','Identity is required')

    def result(self):
        return Validation(tuple(self.issues))


def _dag(ids, edges):
    """Finite zero-distance DAG; positive recurrence edges remain explicit elsewhere."""
    after = {x: set() for x in ids};before = {x: set() for x in ids}
    for source, target in edges:
        if source in after and target in before:
            after[source].add(target);before[target].add(source)
    pending = {x: len(v) for x, v in before.items()};ready = deque(x for x, n in pending.items() if n == 0)
    ancestors = {x: set() for x in ids};order = []
    while ready:
        node = ready.popleft();order.append(node)
        for other in after[node]:
            ancestors[other].update(ancestors[node]);ancestors[other].add(node)
            pending[other] -= 1
            if pending[other] == 0:
                ready.append(other)
    return set(ids) - set(order), ancestors


def _broadcast(*shapes):
    out = ()
    for shape in shapes:
        width = max(len(out), len(shape));a = (1,) * (width-len(out)) + out;b = (1,) * (width-len(shape)) + shape
        if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
            raise ValueError("incompatible broadcast shapes")
        out = tuple(y if x == 1 else x for x, y in zip(a, b))
    return out


def _shape(value, values):
    """Concrete shape rules, not numerical equivalence. Unknown primitives stay opaque."""
    parents = [values[x] for x in value.inputs];shapes = [x.shape for x in parents]
    attrs = {x.name: x.value for x in value.attributes};op = value.op
    if op == "input":
        if parents:
            raise ValueError("input has parents")
        return value.shape
    if op == "matmul":
        if len(shapes) != 2 or any(len(x) != 2 for x in shapes):
            raise ValueError("legacy matmul reference supports rank two only")
        a, b = shapes
        for key in ("transa", "transb"):
            if key in attrs and type(attrs[key]) is not bool:
                raise ValueError("transpose must be bool")
        a = a[::-1] if attrs.get("transa") else a;b = b[::-1] if attrs.get("transb") else b
        if a[1] != b[0]:
            raise ValueError("matmul contraction mismatch")
        return a[0], b[1]
    if op == "elementwise":
        kind = attrs.get("kind");arity = 1 if kind in ("exp", "abs", "neg", "recip") else 2
        if kind not in ("exp", "abs", "neg", "recip", "add", "sub", "mul", "div", "max", "min") or len(shapes) != arity:
            raise ValueError("elementwise kind/arity")
        return _broadcast(*shapes)
    if op in ("reduce", "broadcast_row", "gather"):
        count = 1 if op == "reduce" else 2
        if len(shapes) != count or not shapes[0]:
            raise ValueError("axis operation arity/rank")
        axis = attrs.get("axis", -1)
        if type(axis) is not int or not -len(shapes[0]) <= axis < len(shapes[0]):
            raise ValueError("invalid axis")
        axis %= len(shapes[0]);reduced = shapes[0][:axis]+shapes[0][axis+1:]
        if op == "reduce":
            if attrs.get("kind") not in ("sum", "max", "min"):
                raise ValueError("reduce kind")
            return reduced
        if op == "broadcast_row":
            if shapes[1] != reduced or attrs.get("kind") not in ("add", "sub", "mul", "div", "max", "min"):
                raise ValueError("row broadcast shape/kind")
            return shapes[0]
        if len(shapes[0]) != len(shapes[1]) or parents[1].dtype not in ("i32", "i64"):
            raise ValueError("gather rank/index type")
        if any(a != b and a != 1 and b != 1 for i, (a, b) in enumerate(zip(*shapes)) if i != axis):
            raise ValueError("gather non-axis dimensions")
        return tuple(b if i == axis else max(a, b) for i, (a, b) in enumerate(zip(*shapes)))
    if op == "cast":
        if len(shapes) != 1:
            raise ValueError("cast arity")
        return shapes[0]
    if op == "select":
        if len(shapes) != 3 or parents[0].dtype != "bool":
            raise ValueError("select condition/arity")
        return _broadcast(*shapes)
    return None


def validate(architecture: Architecture) -> Validation:
    issues = []
    def add(severity, code, subject, message):
        issues.append(Issue(severity, code, subject, message))
    def error(code, subject, message):
        add("error", code, subject, message)
    def unknown(code, subject, message):
        add("unknown", code, subject, message)
    def index(records, kind):
        out = {}
        for record in records:
            if not record.id or record.id in out:
                error("identity", kind, "empty or duplicate ID")
            out[record.id] = record
        return out
    a = architecture;s = a.semantic
    if a.schema_version != 1:
        error("schema_version", a.id, "unsupported version")
    values=index(s.values,"value");storages=index(a.storages,"storage");resources=index(a.resources,"resource")
    allocations=index(a.allocations,"allocation");instances=index(a.instances,"instance");events=index(a.events,"event")
    index(a.claims,"claim")
    if len({x.name for x in a.platform.software})!=len(a.platform.software):
        error("platform",a.id,"duplicate software identity fields")
    if not a.platform.soc or a.platform.soc=='unknown':
        unknown("platform",a.id,"SoC/software profile not supplied")
    if not a.id or not s.contract.id or not s.contract.domain or not s.contract.reference:
        error("contract",a.id,"architecture identity and contract/domain/reference required")
    if s.contract.kind == "approximate" and not s.contract.error_policy:
        error("contract",a.id,"approximate contract requires an error policy")
    unknown("contract_not_executed",s.contract.id,"reference is declared, not executed by structural validation")
    for kind, ids in (("input",s.inputs),("output",s.outputs)):
        if not ids or len(set(ids)) != len(ids) or any(x not in values for x in ids):
            error("semantic_interface",kind,"missing, duplicate or unknown interface values")
    if set(s.inputs) != {v.id for v in s.values if v.op == "input"}:
        error("semantic_interface",a.id,"all and only input symbols must be in the input interface")
    for v in s.values:
        if v.dtype not in DTYPE_BYTES or any(type(x) is not int or x < 0 for x in v.shape):
            error("value_type",v.id,"unsupported dtype or invalid concrete shape")
        if len({x.name for x in v.attributes}) != len(v.attributes):
            error("attribute",v.id,"duplicate attribute keys")
        if any(x not in values for x in v.inputs):
            error("semantic_reference",v.id,"unknown parent");continue
        try:
            expected = _shape(v,values)
            if expected is None:
                if not v.external_reference:
                    error("semantic_operation",v.id,"opaque operation requires an external semantic reference")
                else:
                    unknown("opaque_semantics",v.id,"shape and semantics require external checker: "+v.external_reference)
            elif expected != v.shape:
                error("semantic_shape",v.id,f"expected {expected}, got {v.shape}")
        except ValueError as exc:
            error("semantic_shape",v.id,str(exc))
    cyclic,_ = _dag(values,[(x,v.id) for v in s.values for x in v.inputs])
    if cyclic:
        error("semantic_cycle",a.id,str(sorted(cyclic)))
    for st in a.storages:
        if st.capacity_bytes is not None and st.capacity_bytes < 0:
            error("storage_capacity",st.id,"negative capacity")
        if st.kind != "GM" and not st.owner:
            error("storage_owner",st.id,"private storage requires owner")
        if st.capacity_bytes is None:
            unknown("storage_capacity",st.id,"capacity not supplied")
    for resource in a.resources:
        if resource.capacity_units <= 0 or not resource.owner:
            error("resource",resource.id,"positive capacity and explicit owner required")
    for allocation in a.allocations:
        if allocation.storage not in storages or allocation.size_bytes < 0:
            error("allocation",allocation.id,"unknown storage or negative size")
        storage=storages.get(allocation.storage)
        if storage and storage.capacity_bytes is not None and allocation.size_bytes>storage.capacity_bytes:
            error("allocation_capacity",allocation.id,"single allocation exceeds storage capacity regardless of lifetime")
        if not allocation.persistent:
            unknown("allocation_lifetime",allocation.id,"nonpersistent live interval needs schedule-level checking")
    required={}
    for instance in a.instances:
        if instance.value not in values or instance.allocation not in allocations:
            error("instance_reference",instance.id,"unknown value/allocation");continue
        v=values[instance.value];layout=instance.layout
        if (len(layout.extents)!=len(layout.strides_bytes) or any(x<0 for x in layout.extents+layout.strides_bytes)
                or instance.offset_bytes<0):
            error("layout",instance.id,"rank, extent, stride or offset invalid");continue
        if len(layout.extents)!=len(v.shape):
            error("logical_tile",instance.id,"layout and value rank differ")
        elif instance.logical_origin:
            if len(instance.logical_origin)!=len(v.shape) or any(o<0 or o+n>d for o,n,d in zip(instance.logical_origin,layout.extents,v.shape)):
                error("logical_tile",instance.id,"logical tile outside value")
        elif layout.extents!=v.shape:
            unknown("logical_tile",instance.id,"partial tile origin not supplied")
        width=DTYPE_BYTES.get(v.dtype)
        if width is None:
            continue
        span=layout_span_bytes(layout,v.dtype)
        required[instance.id]=span
        if instance.offset_bytes+span>allocations[instance.allocation].size_bytes:
            error("instance_bounds",instance.id,"layout exceeds allocation")
        packed=width
        for stride,n in sorted(zip(layout.strides_bytes,layout.extents)):
            if n>1 and stride<packed:
                unknown("layout_injectivity",instance.id,"possible internal layout alias; requires exact checker")
            if n>0:
                packed += (n-1)*stride
    for st in a.storages:
        persistent=sum(x.size_bytes for x in a.allocations if x.storage==st.id and x.persistent)
        if st.capacity_bytes is not None and persistent>st.capacity_bytes:
            error("persistent_capacity",st.id,f"persistent bytes {persistent} exceed {st.capacity_bytes}")
    edges=[]
    for dep in a.dependencies:
        if dep.source not in events or dep.target not in events or dep.iteration_distance<0:
            error("event_dependency",a.id,"unknown endpoint or negative distance");continue
        if dep.iteration_distance:
            unknown("recurrence_boundary",a.id,"positive-distance edge preserved; requires finite expansion and initialization")
        else:
            edges.append((dep.source,dep.target))
    cyclic,ancestors=_dag(events,edges)
    if cyclic:
        error("event_cycle",a.id,str(sorted(cyclic)))
    if a.dispatch_priority:
        if len(a.dispatch_priority)!=len(events) or set(a.dispatch_priority)!=set(events):
            error("dispatch_priority",a.id,"priority must name every event exactly once")
        # Priority is a scheduler hint, never an added execution dependency.
    accesses=defaultdict(list)
    for event in a.events:
        if not event.actor:
            error("event_actor",event.id,"actor required")
        bounds=event.duration.lower_ns,event.duration.upper_ns
        if any(x is not None and x<0 for x in bounds) or (None not in bounds and bounds[0]>bounds[1]):
            error("duration",event.id,"invalid duration interval")
        if None in bounds:
            unknown("event_duration",event.id,"duration unknown; cannot predict latency")
        if event.kind in ('compute','copy') and not event.resources:
            unknown("event_resources",event.id,"execution resources not supplied")
        if len({x.resource for x in event.resources})!=len(event.resources):
            error("event_resource",event.id,"duplicate resource demand")
        for demand in event.resources:
            if demand.resource not in resources or demand.units<=0 or demand.units>resources[demand.resource].capacity_units:
                error("event_resource",event.id,"unknown resource or demand exceeds capacity")
            elif resources[demand.resource].owner not in (event.actor,'global'):
                error("event_resource",event.id,"resource belongs to a different actor")
        if any(v not in values for v in event.semantic_values):
            error("event_semantics",event.id,"unknown semantic value")
        for writing,regions in ((False,event.reads),(True,event.writes)):
            for region in regions:
                instance=instances.get(region.instance)
                if instance is None or region.instance not in required:
                    error("access_reference",event.id,"unknown/invalid instance");continue
                if region.offset_bytes<0 or region.span_bytes<0 or region.offset_bytes+region.span_bytes>required[instance.id]:
                    error("access_bounds",event.id,"access outside instance span");continue
                if region.span_bytes:
                    start=instance.offset_bytes+region.offset_bytes
                    accesses[instance.allocation].append((event.id,start,start+region.span_bytes,writing,instance.id))
                allocation=allocations[instance.allocation];storage=storages.get(allocation.storage)
                if storage and storage.kind!='GM' and storage.owner!=event.actor:
                    error("private_access",event.id,"private storage owned by another actor")
    def uncovered(lo,hi,intervals):
        gaps=[];covered=lo
        for left,right in sorted((max(lo,x),min(hi,y)) for x,y in intervals if y>lo and x<hi):
            if left>covered:gaps.append((covered,left))
            covered=max(covered,right)
        if covered<hi:gaps.append((covered,hi))
        return gaps
    def same_content(left,right):
        # Same logical value with the same affine logical→physical map. This
        # includes a partial view at its matching origin, not a shifted copy.
        l,r=instances[left],instances[right]
        if l.value!=r.value or l.layout.strides_bytes!=r.layout.strides_bytes:return False
        def base(i):
            origin=i.logical_origin or (0,)*len(i.layout.extents)
            return i.offset_bytes-sum(o*t for o,t in zip(origin,i.layout.strides_bytes))
        return base(l)==base(r)
    periodic=any(dep.iteration_distance for dep in a.dependencies)
    for allocation,regions in accesses.items():
        for i,left in enumerate(regions):
            for right in regions[i+1:]:
                if not (left[3] or right[3]) or max(left[1],right[1])>=min(left[2],right[2]):
                    continue
                x,y=left[0],right[0]
                if x==y:
                    unknown("intra_event_alias",x,"in-place access order requires operation-level checking")
                elif x not in ancestors[y] and y not in ancestors[x]:
                    if any(dep.iteration_distance for dep in a.dependencies):
                        unknown("periodic_alias",allocation,"alias order requires recurrence expansion")
                    else:
                        error("unordered_alias",allocation,f"overlapping access without dependency: {x}, {y}")
        for reader,start,end,writing,iid in regions:
            if not writing:
                # A preceding producer is insufficient when another aliased
                # logical value overwrites bytes between that producer and read.
                # Overlapping unordered accesses are already rejected above.
                for clobber,left,right,is_write,other in regions:
                    if (not is_write or clobber not in ancestors[reader]
                            or same_content(iid,other) or left>=end or right<=start):continue
                    lo,hi=max(start,left),min(end,right)
                    refresh=[(x[1],x[2]) for x in regions if x[3] and same_content(iid,x[4])
                        and clobber in ancestors[x[0]] and x[0] in ancestors[reader]]
                    gaps=uncovered(lo,hi,refresh)
                    if not gaps:continue
                    if periodic:
                        unknown('periodic_value_clobber',reader,'value refresh after alias overwrite requires finite recurrence expansion')
                        continue
                    # Writes inside one opaque event have no declared internal
                    # order; do not invent an order or reject a possible refresh.
                    within=[(x[1],x[2]) for x in regions if x[3] and x[0]==clobber and same_content(iid,x[4])]
                    definite=[gap for gap in gaps if uncovered(*gap,within)]
                    if definite:
                        error('clobbered_value',reader,f'aliased write {clobber}/{other} overwrites {iid}; no later same-value refresh for bytes {definite}')
                    else:
                        unknown('intra_event_value_clobber',reader,'refresh and overwrite occur in the same opaque event; operation order must be checked')
            if writing or instances[iid].value in s.inputs:
                continue
            intervals=sorted((max(start,x[1]),min(end,x[2])) for x in regions if x[3] and x[4]==iid and x[0] in ancestors[reader] and x[2]>start and x[1]<end)
            covered=start
            for lo,hi in intervals:
                if lo>covered:
                    break
                covered=max(covered,hi)
            if covered<end:
                if any(dep.iteration_distance for dep in a.dependencies):
                    unknown("periodic_producer",reader,"producer must be checked after recurrence expansion")
                else:
                    error("missing_producer",reader,"non-input read not covered by a preceding write")
    for output in s.outputs:
        if output in values and math.prod(values[output].shape)==0:
            continue
        if output not in s.inputs and not any(instances[r.instance].value==output and r.span_bytes for ev in a.events for r in ev.writes if r.instance in instances):
            error("output_producer",output,"output has no physical producer")
    if not a.bindings:
        unknown("binding",a.id,"semantic-to-event mapping not supplied")
    if len({x.value for x in a.bindings})!=len(a.bindings):
        error("binding",a.id,"duplicate semantic binding")
    for binding in a.bindings:
        if binding.value not in values or any(x not in events for x in binding.events) or any(x not in instances or instances[x].value!=binding.value for x in binding.instances):
            error("binding",binding.value,"unknown or inconsistent mapping")
        elif any(binding.value not in events[x].semantic_values for x in binding.events):
            error("binding",binding.value,"mapped event does not declare this semantic value")
    # Evidence labels are supplied claims, not certifications. Check their shape and expose uncertainty.
    def evidence(record,subject):
        if is_dataclass(record):
            if isinstance(record,Evidence):
                if record.kind in ('measured','derived') and not record.reference:
                    error('evidence_reference',subject,'measured/derived claim requires source')
                elif record.kind=='unknown':
                    unknown('evidence_unknown',subject,record.note)
                elif record.kind=='assumed':
                    unknown('evidence_assumed',subject,'assumption: '+record.note)
                else:
                    unknown('evidence_not_verified',subject,'source declared, not verified here: '+record.reference)
            else:
                for f in fields(record):
                    evidence(getattr(record,f.name),subject+'.'+f.name)
        elif isinstance(record,tuple):
            for i,item in enumerate(record):
                evidence(item,subject+f'[{i}]')
    evidence(a,a.id)
    return Validation(tuple(issues))


def semantic_from_legacy(program, outputs, contract):
    """Snapshot the existing G_math without sharing mutable attrs or display labels.

    Legacy node IDs identify the symbols. The schema's own fingerprint includes
    every snapshotted semantic field; it is not the legacy cached program hash.
    """
    nodes=program.nodes();known={node.id() for node in nodes}
    if any(node.id() not in known for node in outputs):
        raise ValueError('Requested output is absent from legacy program')
    values=tuple(Value(node.id(),node.op,tuple(node.shape),node.dtype,
                       tuple(parent.id() for parent in node.args),
                       tuple(Attribute(key,value) for key,value in sorted(node.attrs.items())),
                       label=node.name) for node in nodes)
    return SemanticProgram(values,tuple(node.id() for node in nodes if node.op=='input'),
                           tuple(node.id() for node in outputs),contract)


def _contiguous(shape,width):
    strides=[];step=width
    for n in reversed(shape):
        strides.append(step);step*=n
    return Layout(shape,tuple(reversed(strides)))


def example(name):
    """Small explicit scenarios. Synthetic resources do not claim Ascend capacities."""
    if name not in ('matmul','dense_attention','sparse_attention'):
        raise ValueError('Unknown example')
    values=[Value('q','input',(2,2),'f32'),Value('k','input',(2,2),'f32')]
    if name=='matmul':
        values.append(Value('out','matmul',(2,2),'f32',('q','k')))
    else:
        values.append(Value('v','input',(2,2),'f32'))
        values.append(Value('score','matmul',(2,2),'f32',('q','k'),(Attribute('transb',True),)))
        source='score'
        if name=='sparse_attention':
            values.append(Value('indices','input',(2,1),'i64'))
            values.append(Value('selected','gather',(2,1),'f32',('score','indices')));source='selected'
        shape=(2,1) if name=='sparse_attention' else (2,2)
        values.extend((Value('max','reduce',(2,),'f32',(source,),(Attribute('kind','max'),)),
                       Value('shift','broadcast_row',shape,'f32',(source,'max'),(Attribute('kind','sub'),)),
                       Value('exp','elementwise',shape,'f32',('shift',),(Attribute('kind','exp'),)),
                       Value('sum','reduce',(2,),'f32',('exp',),(Attribute('kind','sum'),)),
                       Value('p','broadcast_row',shape,'f32',('exp','sum'),(Attribute('kind','div'),))))
        if name=='sparse_attention':
            values.append(Value('out','external.sparse_pv',(2,2),'f32',('p','indices','v'),external_reference='vc-attention/scripts/mindie_laser_sparse_reference.py'))
        else:
            values.append(Value('out','matmul',(2,2),'f32',('p','v')))
    inputs=tuple(x.id for x in values if x.op=='input')
    semantic=SemanticProgram(tuple(values),inputs,('out',),Contract(name,'exact','core.opgen.semantics / explicit indexed PV','concrete finite two-row scenario; valid indices'))
    allocations=[];instances=[];events=[];dependencies=[];bindings=[]
    for value in values:
        size=math.prod(value.shape)*DTYPE_BYTES[value.dtype]
        allocations.append(Allocation(value.id,'gm',size));instances.append(Instance(value.id,value.id,value.id,_contiguous(value.shape,DTYPE_BYTES[value.dtype])))
        if value.op=='input':
            bindings.append(Binding(value.id,(),(value.id,)));continue
        resource='cube' if value.op=='matmul' else 'vector';actor='aic' if resource=='cube' else 'aiv'
        reads=tuple(Region(x,0,next(z.size_bytes for z in allocations if z.id==x)) for x in value.inputs)
        events.append(Event(value.id,actor,'compute',reads,(Region(value.id,0,size),),(Demand(resource),),(value.id,)))
        dependencies.extend(Dependency(x,value.id,'data') for x in value.inputs if x not in inputs)
        bindings.append(Binding(value.id,(value.id,),(value.id,)))
    return Architecture(name,semantic,(Storage('gm','GM','global',4096,Evidence('assumed',scope='synthetic',note='Example capacity only')),),
                        (Resource('cube','aic'),Resource('vector','aiv')),tuple(allocations),tuple(instances),tuple(events),tuple(dependencies),tuple(bindings))


def _nc_symbol_identity():
    from .graph import leaf, Program
    q,k=leaf('q',(2,2)),leaf('k',(2,2));program=Program();program.add(q);program.add(k)
    return len(program.nodes())==2 and q.id()!=k.id()


def _nc_constraints():
    a=example('matmul');bad=replace(a,events=a.events+(replace(a.events[0],id='conflicting'),))
    return (not any(x.severity=='error' for x in validate(a).issues)
            and any(x.code=='unordered_alias' and x.severity=='error' for x in validate(bad).issues))


def _nc_unknown():
    result=validate(example('sparse_attention'))
    return result.status=='conditional' and any(x.code=='opaque_semantics' for x in result.issues)


def _nc_value_clobber():
    a=example('matmul');alias=replace(a.instances[-1],id='aliased_result',allocation='q')
    overwrite=replace(a.events[0],id='overwrite_input',writes=(Region(alias.id,0,16),))
    b=replace(a,instances=a.instances+(alias,),events=(overwrite,)+a.events,
        dependencies=(Dependency('overwrite_input',a.events[0].id,'program'),))
    return (not any(x.severity=='error' for x in validate(a).issues)
        and any(x.code=='clobbered_value' and x.severity=='error' for x in validate(b).issues))


NEGATIVE_CONTROLS = {
    'ARCH1_symbol_identity': _nc_symbol_identity,
    'ARCH2_structural_constraints': _nc_constraints,
    'ARCH3_unknown_not_proof': _nc_unknown,
    'ARCH4_value_clobber': _nc_value_clobber,
}


def cli(args):
    if any(getattr(args,key,None) for key in ('trace','trace_plan','verify_trace')):
        from .trace import cli as trace_cli
        return trace_cli(args)
    if getattr(args,'transform',None) or getattr(args,'verify_transform',None):
        from .transforms import cli_transform
        return cli_transform(args)
    from pathlib import Path
    import sys
    try:
        if args.selftest:
            if args.output:
                raise ValueError('--selftest does not write an output file')
            from .architecture_test import selftest
            return selftest()
        if args.example:
            text=example(args.example).to_json(pretty=True);code=0
        elif any(getattr(args,field,None) for field in ('observation','campaign','legacy')):
            from .observation import ObservedExecution,validate_observation
            from .campaign import Campaign,validate_campaign
            from .legacy import LegacySummary,validate_legacy
            for field,cls,check in (('observation',ObservedExecution,validate_observation),
                                    ('campaign',Campaign,validate_campaign),('legacy',LegacySummary,validate_legacy)):
                path=getattr(args,field,None)
                if path:
                    obj=cls.from_json(Path(path).read_text());result=check(obj)
                    text=_dump(dict(document_type=cls.__name__,document_fingerprint=obj.fingerprint(),**result.to_dict()),pretty=True)
                    code=2 if result.status=='invalid' else 0
                    break
        else:
            architecture=Architecture.from_json(Path(args.input).read_text())
            result=validate(architecture);text=_dump(dict(model_fingerprint=architecture.fingerprint(),**result.to_dict()),pretty=True)
            code=2 if result.status=='invalid' else 0
        if args.output:
            # Explicit new outputs only; inspection cannot silently overwrite evidence.
            with Path(args.output).open('x') as stream:
                stream.write(text+'\n')
        else:
            print(text)
        return code
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='invalid',error=str(exc))),file=sys.stderr)
        return 2
