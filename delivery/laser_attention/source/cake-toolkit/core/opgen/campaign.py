"""Typed architecture and development graphs, not a continuation policy engine."""
from __future__ import annotations
from dataclasses import dataclass
import re
from decimal import Decimal
from typing import Literal
from .._refs import Ref
from .architecture import Evidence, Inspection, Record, VersionedRecord, _dag, _dump, decode_versioned, parse_json
from .transforms import TransformPlan, _operation_tags


@dataclass(frozen=True)
class ModelRef(Record):
    id: str
    fingerprint: str
    reference: str


@dataclass(frozen=True)
class EvolutionEdge(Record):
    id: str
    source: str
    target: str
    plan: TransformPlan
    receipt_reference: str = ''


@dataclass(frozen=True)
class Hypothesis(Record):
    id: str
    statement: str
    models: tuple[str,...]
    status: Literal['unknown','supported','refuted'] = 'unknown'
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class Decision(Record):
    id: str
    action: str
    models: tuple[str,...]
    depends_on: tuple[str,...]
    hypotheses: tuple[str,...]
    milestone: str
    discriminating_probe: str
    exit_conditions: tuple[str,...]
    capabilities: tuple[str,...] = ()
    status: Literal['planned','in_progress','completed','stopped','suspended'] = 'planned'
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class JointGroup(Record):
    id: str
    members: tuple[str,...]
    criterion: str
    kind: Literal['performance','correctness','commit'] = 'performance'


@dataclass(frozen=True)
class Prediction(Record):
    id: str
    model: str
    decision: str
    metric: str
    unit: str
    lower: int | float
    upper: int | float
    scope: str
    reference_model: str = ''
    evidence: Evidence = Evidence()


@dataclass(frozen=True)
class BudgetLimit(Record):
    id: str
    resource: str
    unit: str
    limit: int | float


@dataclass(frozen=True)
class BudgetUse(Record):
    id: str
    budget: str
    decision: str
    amount: int | float
    unit: str
    evidence: Evidence


@dataclass(frozen=True)
class Campaign(VersionedRecord):
    id: str
    objective: str
    models: tuple[ModelRef,...]
    transitions: tuple[EvolutionEdge,...]
    decisions: tuple[Decision,...]
    hypotheses: tuple[Hypothesis,...]
    groups: tuple[JointGroup,...]
    predictions: tuple[Prediction,...]
    budgets: tuple[BudgetLimit,...]
    uses: tuple[BudgetUse,...]
    working_head: str
    validated_best: str | None = None
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text)
        if type(raw) is dict and type(raw.get('transitions')) is list:
            for edge in raw['transitions']:
                if type(edge) is dict and type(edge.get('plan')) is dict:
                    _operation_tags(edge['plan'].get('operation'))
        return decode_versioned(cls,raw)

    def canonical(self):
        data=super().canonical()
        for field in ('models','transitions','decisions','hypotheses','groups','predictions','budgets','uses'):
            data[field]=sorted(data[field],key=_dump)
        return data


def validate_campaign(campaign: Campaign):
    a=campaign;c=Inspection();c.version(a)
    models=c.index(a.models,'model');transitions=c.index(a.transitions,'transition')
    decisions=c.index(a.decisions,'decision');hypotheses=c.index(a.hypotheses,'hypothesis')
    groups=c.index(a.groups,'joint group');predictions=c.index(a.predictions,'prediction')
    budgets=c.index(a.budgets,'budget');c.index(a.uses,'budget use')
    if not a.objective or a.working_head not in models or (a.validated_best is not None and a.validated_best not in models):
        c.error('campaign_identity',a.id,'Objective and valid working/best model references are required')
    for model in a.models:
        if not re.fullmatch('[0-9a-f]{64}',model.fingerprint) or not model.reference:
            c.error('model_identity',model.id,'Model requires exact content identity and reference')
    for edge in a.transitions:
        if edge.source not in models or edge.target not in models or edge.source==edge.target:
            c.error('transition_reference',edge.id,'Distinct known source/target versions are required');continue
        if edge.plan.source_fingerprint!=models[edge.source].fingerprint or edge.plan.schema_version!=1:
            c.error('transition_identity',edge.id,'Plan is not bound to its declared source/version')
        # Applicability and target identity require the actual Architecture and receipt.
        c.unknown('transition_not_replayed',edge.id,'Referenced model/receipt content must be loaded and replayed before certification')
    cycle,_=_dag(models,[(x.source,x.target) for x in a.transitions])
    if cycle:
        c.error('evolution_cycle',a.id,'Version graph has a cycle; rollback should create a new version node')
    edges=[]
    for decision in a.decisions:
        if not all((decision.action,decision.milestone,decision.discriminating_probe)) or not decision.exit_conditions or any(not x for x in decision.exit_conditions):
            c.error('decision_contract',decision.id,'Action, milestone, distinguishing probe and exit conditions are required')
        for field,known in (('models',models),('depends_on',decisions),('hypotheses',hypotheses)):
            refs=getattr(decision,field)
            if len(set(refs))!=len(refs) or any(x not in known for x in refs):
                c.error('decision_reference',decision.id,'Duplicate or unknown '+field)
        edges.extend((source,decision.id) for source in decision.depends_on)
        if decision.status=='completed':
            c.unknown('decision_not_verified',decision.id,'Completion label does not certify a milestone or authorize delivery')
    cycle,_=_dag(decisions,edges)
    if cycle:
        c.error('decision_cycle',a.id,'Development prerequisite graph has a cycle')
    for hypothesis in a.hypotheses:
        if not hypothesis.statement or not hypothesis.models or any(x not in models for x in hypothesis.models):
            c.error('hypothesis_reference',hypothesis.id,'Statement and model-scoped validity are required')
        c.unknown('hypothesis_not_verified',hypothesis.id,'Status/evidence is a declaration scoped to these model versions')
    for group in a.groups:
        if not group.members or len(set(group.members))!=len(group.members) or any(x not in decisions for x in group.members) or not group.criterion:
            c.error('joint_group',group.id,'Known unique members and explicit joint criterion are required')
    for prediction in a.predictions:
        if prediction.model not in models or prediction.decision not in decisions or (prediction.reference_model and prediction.reference_model not in models):
            c.error('prediction_reference',prediction.id,'Unknown prediction model/decision/comparator')
        if prediction.lower>prediction.upper or not all((prediction.metric,prediction.unit,prediction.scope)):
            c.error('prediction_interval',prediction.id,'Ordered interval, metric, unit and scope are required')
    for budget in a.budgets:
        if budget.limit<0 or not budget.resource or not budget.unit:
            c.error('budget',budget.id,'Nonnegative limit, resource and explicit unit are required')
    totals={x:Decimal(0) for x in budgets}
    for use in a.uses:
        if use.budget not in budgets or use.decision not in decisions or use.amount<0:
            c.error('budget_use',use.id,'Known budget/decision and nonnegative usage are required');continue
        if use.unit!=budgets[use.budget].unit:
            c.error('budget_unit',use.id,'Budget usage unit differs; no implicit conversion')
        else:
            totals[use.budget]+=Decimal(str(use.amount))
        c.unknown('budget_use_not_verified',use.id,'Usage record source requires independent verification')
    for key,total in totals.items():
        if total>Decimal(str(budgets[key].limit)):
            c.unknown('budget_exceeded',key,'Recorded use exceeds declared limit; continuation policy must evaluate this')
    if not a.budgets or not a.predictions:
        c.unknown('campaign_contract',a.id,'No budget or prediction supplied')
    c.unknown('campaign_policy_pending',a.id,'Structural validity does not decide continue/revise/stop or reject an intermediate slowdown')
    if a.validated_best is not None:
        c.unknown('best_not_verified',a.validated_best,'Best marker requires independent evaluation evidence; working_head does not update it')
    return c.result()


def _nc_cycle():
    d=Decision('d','action',(),('d',),(),'milestone','probe',('budget exhaustion',))
    a=Campaign('nc','test',(ModelRef('m','0'*64,'fixture://m'),),(),(d,),(),(),(),(),(),'m')
    return any(x.code=='decision_cycle' and x.severity=='error' for x in validate_campaign(a).issues)


REFS={'CAM1_separate_graphs_and_claims':Ref(
    what=['架构版本与开发依赖分图；预测/预算有单位与作用域，联合验收及工作/最佳水位分开。'],
    status='derived',evidence=['E101 research_test DAG/hash/unit/negative-intermediate examples; P4 policy and evidence authentication are not implemented here.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'CAM1_separate_graphs_and_claims':_nc_cycle}
