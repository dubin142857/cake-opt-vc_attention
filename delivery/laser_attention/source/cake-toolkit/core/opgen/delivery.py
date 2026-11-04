"""Protected measured-contract promotion and fresh rollback evidence.

This updates a campaign selection document only. Frozen trusted evaluators
supply metrics; a CPU fixture or an unreviewed video never becomes a video win.
"""
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Literal
from decimal import Decimal
from .._refs import Ref
from .architecture import Record,VersionedRecord,_dump,parse_json
from .campaign import Campaign,validate_campaign
from .runner import FileIdentity,RunReceipt,StageReport


@dataclass(frozen=True)
class DeliveryPlan(VersionedRecord):
    id: str
    campaign_fingerprint: str
    operation: Literal['promote','rollback']
    target_model: str
    runs: tuple[FileIdentity,...]
    candidate_metric: str
    reference_metric: str
    unit: str
    reason: str
    # 本仓自定: two independent runs are a reproducibility minimum, not confidence.
    minimum_runs: int = 2
    minimum_gain_fraction: int | float = 0
    incumbent: FileIdentity | None = None
    incumbent_metric: str = ''
    scope: Literal['measured_contract_only'] = 'measured_contract_only'
    schema_version: int = 1


@dataclass(frozen=True)
class DeliveryReading(Record):
    run_fingerprint: str
    run_plan: str
    hardware: str
    candidate: int | float
    reference: int | float | None
    incumbent: int | float | None
    unit: str
    gain_fraction: float | None


@dataclass(frozen=True)
class DeliveryReceipt(VersionedRecord):
    before: Campaign
    plan: DeliveryPlan
    after: Campaign
    status: Literal['promoted','rejected','rolled_back']
    readings: tuple[DeliveryReading,...]
    reasons: tuple[str,...]
    conditions: tuple[str,...]
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text)
        Campaign.from_json(_dump(raw['before']));Campaign.from_json(_dump(raw['after']))
        return super().from_json(text)

    def verify(self):
        if assess(self.before,self.plan)!=self:raise ValueError('Delivery receipt differs from protected evidence replay')
        return True


CONDITIONS=(
    'Only the frozen measured workload and numerical-check contract is validated; video quality remains unreviewed',
    'Every supplied run is checked; no best-of-run filtering or pooled positive/negative cancellation',
    'RunPlan metric bindings are trusted evaluator declarations, not mathematical or binary equivalence proofs',
    'This changes campaign selection only; it does not checkout git, install libraries or alter a shared environment',
    'Changed tools/dependencies make old evidence stale and require fresh validation; history is not silently upgraded',
)


def _read_run(identity):
    if identity.role!='output':raise ValueError('Run evidence must identify an output receipt')
    identity.verify();r=RunReceipt.from_json(Path(identity.path).read_text());r.verify();return r


def _signature(r):
    p=r.plan
    dependencies=tuple(sorted((f.path,f.sha256,f.size_bytes) for f in p.inputs if f.role in ('source','harness','runtime')))
    return (p.device.kind,p.workload,p.contract,p.software,p.env,p.sandbox.sha256,p.tool_fingerprint,dependencies)


def _measure(r):
    row=next(x for x in r.stages if x.kind=='measure')
    stage=next(x for x in r.plan.stages if x.kind=='measure')
    product=next(x for x in row.products if x.relative_path==stage.report)
    return {x.name:x for x in StageReport.from_json(Path(product.snapshot.path).read_text()).metrics}


def _binding(r,metric,model):
    matches=[x for x in r.plan.metric_bindings if x.metric==metric]
    if len(matches)!=1 or matches[0].model_fingerprint!=model:raise ValueError('Metric is not bound to the requested model: '+metric)
    return matches[0].scope


def _parent(identity):
    if identity is None or identity.role!='output':raise ValueError('Protected incumbent delivery receipt is required')
    identity.verify();receipt=DeliveryReceipt.from_json(Path(identity.path).read_text());receipt.verify()
    if receipt.status not in ('promoted','rolled_back'):raise ValueError('Incumbent proof never selected a validated best')
    return receipt


def assess(c,p):
    if p.schema_version!=1 or p.scope!='measured_contract_only' or p.campaign_fingerprint!=c.fingerprint() or not p.id or not p.reason:
        raise ValueError('Delivery source/version/scope/reason differs')
    if any(x.severity=='error' for x in validate_campaign(c).issues):raise ValueError('Invalid delivery campaign')
    models={m.id:m for m in c.models}
    if p.target_model not in models or not p.candidate_metric or not p.unit or not p.runs:raise ValueError('Explicit model/metric/unit/run evidence required')
    if p.operation not in ('promote','rollback'):raise ValueError('Unknown delivery operation')
    if p.minimum_gain_fraction<0 or p.minimum_gain_fraction>=1:raise ValueError('Invalid minimum improvement fraction')
    if p.minimum_runs<(2 if p.operation=='promote' else 1):raise ValueError('Promotion needs at least two independent runs; rollback needs one fresh run')
    if p.operation=='promote' and (not p.reference_metric or p.reference_metric==p.candidate_metric):raise ValueError('Distinct same-run reference metric required')
    parent=None
    if p.operation=='rollback' or c.validated_best is not None:
        parent=_parent(p.incumbent);best=parent.after.validated_best
        expected=p.target_model if p.operation=='rollback' else c.validated_best
        if best!=expected or best not in models or next(m.fingerprint for m in parent.after.models if m.id==best)!=models[best].fingerprint:
            raise ValueError('Protected incumbent model differs')
        if p.operation=='promote' and (not p.incumbent_metric or p.incumbent_metric==p.candidate_metric):raise ValueError('Incumbent requires a distinct same-run metric')
    elif p.incumbent is not None or p.incumbent_metric:raise ValueError('Unexpected incumbent argument')
    runs=tuple(_read_run(x) for x in p.runs)
    if len({r.fingerprint() for r in runs})!=len(runs) or len({r.plan.fingerprint() for r in runs})!=len(runs) or len({r.work_dir for r in runs})!=len(runs):
        raise ValueError('Duplicate execution is not independent evidence')
    signatures={_dump_signature(r) for r in runs}
    if len(signatures)!=1:raise ValueError('Run workload/software/source/evaluator scope differs')
    if parent is not None:
        previous=_read_run(parent.plan.runs[0])
        if (previous.plan.device.kind,previous.plan.workload,previous.plan.software)!=(runs[0].plan.device.kind,runs[0].plan.workload,runs[0].plan.software):
            raise ValueError('Protected incumbent hardware/workload/software domain differs')
    if p.operation=='rollback':
        old=_read_run(parent.plan.runs[0])
        if _signature(old)!=_signature(runs[0]):raise ValueError('Rollback implementation/workload differs from protected target')
    reasons=[];readings=[];measurement_scope=None
    if len(runs)<p.minimum_runs:reasons.append('insufficient_independent_runs')
    for r in runs:
        if r.plan.model_fingerprint!=models[p.target_model].fingerprint:raise ValueError('Actual run model differs from requested target')
        scope=_binding(r,p.candidate_metric,models[p.target_model].fingerprint)
        if measurement_scope is not None and measurement_scope!=scope:raise ValueError('Candidate metric measurement scope differs')
        measurement_scope=scope
        if r.state!='verified':reasons.append('run_not_verified:'+r.plan.id);continue
        metrics=_measure(r)
        names=(p.candidate_metric,)+( (p.reference_metric,) if p.operation=='promote' else ())+( (p.incumbent_metric,) if p.operation=='promote' and parent else ())
        if any(n not in metrics or metrics[n].unit!=p.unit for n in names):raise ValueError('Protected metric name/unit differs')
        candidate=metrics[p.candidate_metric].value;reference=metrics[p.reference_metric].value if p.operation=='promote' else None
        incumbent=metrics[p.incumbent_metric].value if p.operation=='promote' and parent else None;gain=None
        if reference is not None:
            if reference<=0:raise ValueError('Reference metric must be positive')
            exact_gain=(Decimal(str(reference))-Decimal(str(candidate)))/Decimal(str(reference));gain=float(exact_gain)
            if exact_gain<=Decimal(str(p.minimum_gain_fraction)):reasons.append('reference_not_beaten:'+r.plan.id)
        if incumbent is not None:
            incumbent_scope=_binding(r,p.incumbent_metric,models[c.validated_best].fingerprint)
            if incumbent_scope!=scope:raise ValueError('Incumbent metric measurement scope differs')
            if candidate>=incumbent:reasons.append('incumbent_not_beaten:'+r.plan.id)
        readings.append(DeliveryReading(r.fingerprint(),r.plan.fingerprint(),r.plan.device.kind,candidate,reference,incumbent,p.unit,gain))
    after=c;status='rejected'
    if not reasons:
        if p.operation=='promote':after=replace(c,validated_best=p.target_model);status='promoted'
        else:after=replace(c,working_head=p.target_model,validated_best=p.target_model);status='rolled_back'
    return DeliveryReceipt(c,p,after,status,tuple(readings),tuple(reasons),CONDITIONS)


def _dump_signature(r):
    # The strict record codec handles Attribute values without custom coercion.
    from .architecture import _encode
    return _dump(_encode(_signature(r)))


def cli(args):
    import sys
    try:
        if args.selftest:
            if args.campaign or args.output:raise ValueError('Selftest consumes no campaign/output')
            from .delivery_test import selftest
            return selftest()
        if args.verify:
            if args.campaign:raise ValueError('Bound delivery replay cannot override the campaign')
            receipt=DeliveryReceipt.from_json(Path(args.verify).read_text());receipt.verify();value=dict(status='protected_delivery_replayed',decision=receipt.status,scope=receipt.plan.scope)
        else:
            if not args.campaign or not args.plan:raise ValueError('Explicit campaign and delivery plan required')
            receipt=assess(Campaign.from_json(Path(args.campaign).read_text()),DeliveryPlan.from_json(Path(args.plan).read_text()));value=json_value(receipt)
        text=_dump(value,pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return 0
    except (OSError,ValueError,TypeError,KeyError) as error:
        print(_dump(dict(status='rejected',error=str(error))),file=sys.stderr);return 2


def json_value(receipt):return parse_json(receipt.to_json())


def _nc_labels_cannot_promote():
    try:_parent(None)
    except ValueError as error:return 'Protected incumbent' in str(error)
    return False


REFS={'DELIVER1_protected_selection':Ref(what=['真实RunReceipt指标/模型/同条件/重复与原best保护；fresh原实现恢复；未验视频保留。'],status='derived',evidence=['E110 real CPU namespace faults, selection/rollback controls and original attention guarded negative-promotion runs'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'DELIVER1_protected_selection':_nc_labels_cannot_promote}
