"""Bind a transformed finite model to pinned native code and sampled ABI storage.

This verifies shared tensor/interface constraints for one invocation. It does
not turn declared mathematical semantics, private memory or timing into proofs.
"""
from dataclasses import dataclass
from pathlib import Path
from .._refs import Ref
from .architecture import Record,VersionedRecord,DTYPE_BYTES,_dump,parse_json,layout_span_bytes
from .transforms import TransformReceipt
from .schedule import ScheduleReceipt
from .runner import FileIdentity


@dataclass(frozen=True)
class TensorBoundary(Record):
    instance: str
    tensor: str
    alignment_bytes: int = 1


@dataclass(frozen=True)
class NativeEntry(Record):
    event: str
    library: FileIdentity
    symbol: str
    kernel: str


@dataclass(frozen=True)
class LoweringPlan(VersionedRecord):
    id: str
    transform: TransformReceipt
    schedule: ScheduleReceipt
    boundaries: tuple[TensorBoundary,...]
    entries: tuple[NativeEntry,...]
    sources: tuple[FileIdentity,...]
    scope: str
    device: int
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text);TransformReceipt.from_json(_dump(raw['transform']));ScheduleReceipt.from_json(_dump(raw['schedule']))
        return super().from_json(text)

    @property
    def architecture(self):return self.transform.architecture

    def verify(self):
        if self.schema_version!=1 or not self.id or not self.scope or self.device<=0:raise ValueError('Lowering identity/scope/version/device missing')
        self.transform.verify();self.schedule.verify()
        a=self.architecture
        if self.schedule.architecture!=a or self.schedule.implementation is None or self.schedule.certificate is None:
            raise ValueError('Lowering requires the transformed model and its finite implementation contract')
        ids={x.id for x in a.instances};events={e.id for e in a.events}
        if len(self.boundaries)!=len(ids) or {b.instance for b in self.boundaries}!=ids or len({b.tensor for b in self.boundaries})!=len(ids):
            raise ValueError('Lowering tensor boundaries must cover every instance exactly once')
        if any(not b.tensor or b.alignment_bytes<=0 or b.alignment_bytes&(b.alignment_bytes-1) for b in self.boundaries):raise ValueError('Invalid lowering tensor/alignment')
        if len(self.entries)!=len(events) or {e.event for e in self.entries}!=events:raise ValueError('Every modeled event needs one native entry')
        for entry in self.entries:
            if not entry.symbol or not entry.kernel or entry.library.role!='runtime':raise ValueError('Native entry identity missing')
            entry.library.verify()
        if not self.sources or len({s.path for s in self.sources})!=len(self.sources):raise ValueError('Pinned implementation sources are required')
        for source in self.sources:
            if source.role not in ('source','harness'):raise ValueError('Invalid lowering source role')
            source.verify()
        return True


@dataclass(frozen=True)
class TensorObservation(Record):
    name: str
    shape: tuple[int,...]
    dtype: str
    strides_bytes: tuple[int,...]
    data_address: int
    storage_address: int
    storage_bytes: int
    storage_offset_bytes: int
    device: int


@dataclass(frozen=True)
class PhysicalStorage(Record):
    storage: str
    device: int
    address: int


@dataclass(frozen=True)
class LoweringReceipt(VersionedRecord):
    plan: LoweringPlan
    tensors: tuple[TensorObservation,...]
    mapped_libraries: tuple[FileIdentity,...]
    physical_storages: tuple[PhysicalStorage,...]
    invocation: str
    status: str = 'instantiated_conditionally'
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text);LoweringPlan.from_json(_dump(raw['plan']));return super().from_json(text)

    def verify(self):
        expected=instantiate(self.plan,self.tensors,self.mapped_libraries,self.invocation)
        if expected!=self:raise ValueError('Lowering receipt differs from physical constraint replay')
        return True


def instantiate(plan,tensors,mapped_libraries,invocation):
    if not invocation:raise ValueError('Sampled invocation scope is required')
    plan.verify()
    a=plan.architecture;table={t.name:t for t in tensors}
    if len(table)!=len(tensors) or set(table)!={b.tensor for b in plan.boundaries}:raise ValueError('Missing/duplicate/unexpected sampled tensor')
    libraries={f.path:f for f in mapped_libraries}
    if len(libraries)!=len(mapped_libraries):raise ValueError('Duplicate mapped native library')
    for entry in plan.entries:
        if libraries.get(entry.library.path)!=entry.library:raise ValueError('Loaded native library differs from lowering')
    instances={i.id:i for i in a.instances};values={v.id:v for v in a.semantic.values};allocs={x.id:x for x in a.allocations}
    offsets={x.allocation:x.offset_bytes for x in plan.schedule.certificate.addresses};bases={};ranges={}
    for boundary in plan.boundaries:
        instance=instances[boundary.instance];value=values[instance.value];allocation=allocs[instance.allocation];observed=table[boundary.tensor]
        if (observed.shape,observed.dtype,observed.strides_bytes)!=(instance.layout.extents,value.dtype,instance.layout.strides_bytes):
            raise ValueError('Actual tensor layout/type differs: '+boundary.tensor)
        if observed.device!=plan.device or min(observed.data_address,observed.storage_address,observed.storage_bytes)<=0 or observed.storage_offset_bytes<0:
            raise ValueError('Invalid tensor device/storage: '+boundary.tensor)
        if observed.data_address!=observed.storage_address+observed.storage_offset_bytes or observed.data_address%boundary.alignment_bytes:
            raise ValueError('Tensor base/offset/alignment differs: '+boundary.tensor)
        span=layout_span_bytes(instance.layout,value.dtype)
        if observed.storage_offset_bytes+span>observed.storage_bytes:raise ValueError('Actual tensor backing storage is too small')
        base=observed.data_address-offsets[allocation.id]-instance.offset_bytes
        key=allocation.storage;physical=PhysicalStorage(key,observed.device,base)
        if base<0 or (key in bases and bases[key]!=physical):raise ValueError('Model arena does not map consistently to actual addresses')
        bases[key]=physical;start=base+offsets[allocation.id];finish=start+allocation.size_bytes
        if start<observed.storage_address or finish>observed.storage_address+observed.storage_bytes:raise ValueError('Model allocation exceeds actual backing storage')
        ranges[allocation.id]=(observed.device,start,finish)
    if set(ranges)!=set(allocs):raise ValueError('Every allocation requires an actual sampled backing')
    leases={x.allocation:x for x in plan.schedule.certificate.leases}
    for i,left in enumerate(sorted(ranges)):
        for right in sorted(ranges)[i+1:]:
            x,y=ranges[left],ranges[right];lx,ly=leases[left],leases[right]
            if x[0]==y[0] and max(x[1],y[1])<min(x[2],y[2]) and max(lx.start_ns,ly.start_ns)<min(lx.finish_ns,ly.finish_ns):
                raise ValueError('Distinct simultaneous model allocations overlap in actual memory')
    return LoweringReceipt(plan,tuple(tensors),tuple(mapped_libraries),tuple(bases[k] for k in sorted(bases)),invocation)



def validate_run_binding(run):
    if run.implementation is None:return None
    from .runner import AscendLease
    from .ascend import load_profile
    binding=run.implementation;binding.verify()
    if binding.role!='input' or binding not in run.inputs:raise ValueError('Lowering plan is not a frozen run input')
    plan=LoweringPlan.from_json(Path(binding.path).read_text());plan.verify()
    if (plan.architecture.fingerprint(),plan.architecture.semantic.contract.id)!=(run.model_fingerprint,run.contract):
        raise ValueError('Run model/contract differs from actual lowering plan')
    if not isinstance(run.device,AscendLease) or load_profile(run.device).physical_card!=plan.device:raise ValueError('Lowering/run physical device mismatch')
    required=set(plan.sources)|{e.library for e in plan.entries}
    if not required.issubset(set(run.inputs)):raise ValueError('Lowering dependencies are absent from frozen run inputs')
    return plan


def cli(args):
    import sys
    try:
        if args.selftest:
            from .lowering_test import selftest
            return selftest()
        if args.verify:
            receipt=LoweringReceipt.from_json(Path(args.verify).read_text());receipt.verify()
            value=dict(status='physical_constraints_replayed',scope=receipt.plan.scope,invocation=receipt.invocation)
        else:
            plan=LoweringPlan.from_json(Path(args.plan).read_text());plan.verify()
            value=dict(status='lowering_bound_conditionally',model_fingerprint=plan.architecture.fingerprint(),scope=plan.scope,
                warning='Pinned code and declarations are not a mathematical or private-workspace proof')
        text=_dump(value,pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return 0
    except (OSError,ValueError,TypeError) as error:
        print(_dump(dict(status='rejected',error=str(error))),file=sys.stderr);return 2


def _nc_missing_tensor():
    # A declared model alone never supplies a sampled invocation.
    try:instantiate(None,(),(),'')
    except (ValueError,AttributeError):return True
    return False


REFS={'LOWER1_bound_physical_instance':Ref(
    what=['变换/调度父链和源码/库绑定；采样张量完整性、形状/跨度/arena/设备及同时存活不别名。'],
    status='derived',evidence=['E109 lowering_test counterexamples and original attention target_validation ABI/trace binding; one invocation scope, private UB remains outside.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'LOWER1_bound_physical_instance':_nc_missing_tensor}
