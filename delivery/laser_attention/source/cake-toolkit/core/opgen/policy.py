"""P4 conditional campaign reviews and immutable development state changes.

Evidence is replayed. Natural-language scope/necessity and budget accounting
remain declared assumptions; this module neither executes work nor promotes
an unvalidated candidate into the delivery best pointer.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from decimal import Decimal
from typing import Literal
from .._refs import Ref
from .architecture import Record,VersionedRecord,_dump,parse_json,decode_versioned,_dag
from .campaign import Campaign,validate_campaign
from .analysis import ReplayReceipt,StudyReceipt
from .trace import TraceReceipt
from .transforms import TransformReceipt


@dataclass(frozen=True)
class RuleKey(Record):
    rule: str
    subject: str


@dataclass(frozen=True)
class Requirement(Record):
    hypothesis: str
    model: str
    check: RuleKey
    revisable: bool = False


@dataclass(frozen=True)
class MilestoneSpec(Record):
    decision: str
    model: str
    checks: tuple[RuleKey,...]


@dataclass(frozen=True)
class RemainingCost(Record):
    budget: str
    unit: str
    lower: int | float
    upper: int | float


@dataclass(frozen=True)
class ProbePlan(Record):
    uncertainties: tuple[str,...]
    costs: tuple[RemainingCost,...]
    outcomes: tuple[str,...]


@dataclass(frozen=True)
class MetricReading(Record):
    prediction: str
    model_fingerprint: str
    trace: TraceReceipt
    measurement: str


@dataclass(frozen=True)
class TransitionProof(Record):
    edge: str
    receipt: TransformReceipt


@dataclass(frozen=True)
class ReviewEvidence(VersionedRecord):
    replays: tuple[ReplayReceipt,...] = ()
    readings: tuple[MetricReading,...] = ()
    transitions: tuple[TransitionProof,...] = ()
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text)
        if type(raw) is dict:
            for proof in raw.get('transitions',[]):
                if type(proof) is dict:TransformReceipt.from_json(_dump(proof.get('receipt')))
        return decode_versioned(cls,raw)


@dataclass(frozen=True)
class ReviewPlan(VersionedRecord):
    campaign_fingerprint: str
    decision: str
    target_model: str
    reference_model: str
    remaining: tuple[RemainingCost,...]
    group: str = ''
    study: StudyReceipt | None = None
    requirements: tuple[Requirement,...] = ()
    milestones: tuple[MilestoneSpec,...] = ()
    probe_hypotheses: tuple[str,...] = ()
    # 本仓自定: a frozen integer-model target, not a hardware or global research gate.
    minimum_gain_ns: int = 1
    probe: ProbePlan | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class Finding(Record):
    kind: str
    subject: str
    status: str
    detail: str


def _campaign_tags(raw):
    if type(raw) is dict:Campaign.from_json(_dump(raw))


@dataclass(frozen=True)
class ReviewReceipt(VersionedRecord):
    campaign: Campaign
    plan: ReviewPlan
    evidence: ReviewEvidence
    action: Literal['continue','revise','stop','unknown']
    continuation_scope: Literal['implementation','probe']
    findings: tuple[Finding,...]
    next_probe: str
    conditions: tuple[str,...]
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text)
        if type(raw) is dict:
            _campaign_tags(raw.get('campaign'))
            if type(raw.get('evidence')) is dict:ReviewEvidence.from_json(_dump(raw['evidence']))
        return decode_versioned(cls,raw)

    def verify(self):
        if review(self.campaign,self.plan,self.evidence)!=self:
            raise ValueError('Review differs from replay of its frozen inputs')
        return True


CONDITIONS=(
    'Read-only branch advice, not authority to launch, commit, push, or deliver a candidate',
    'Study conclusions hold within complete declared model assumptions; witness deltas alone do not guarantee improvement',
    'Replayed files authenticate their recorded contents, not the natural-language mapping to a hypothesis, binary or workload',
    'Budgets and remaining estimates are declared campaign accounting; sunk cost is not a continuation argument',
    'Metric/source/clock/launch scope must match explicitly; hardware and numerical delivery authority requires P5 run evidence',
)


def review(c:Campaign,p:ReviewPlan,e:ReviewEvidence):
    if p.schema_version!=1 or e.schema_version!=1 or p.campaign_fingerprint!=c.fingerprint():
        raise ValueError('Campaign review identity/version mismatch')
    errors=[x for x in validate_campaign(c).issues if x.severity=='error']
    if errors:raise ValueError('Invalid campaign: '+str(errors))
    models={x.id:x for x in c.models};decisions={x.id:x for x in c.decisions};groups={x.id:x for x in c.groups}
    hypotheses={x.id:x for x in c.hypotheses};budgets={x.id:x for x in c.budgets}
    if p.decision not in decisions or p.target_model not in models or p.reference_model not in models or p.minimum_gain_ns<0:
        raise ValueError('Unknown decision/model or negative improvement target')
    decision=decisions[p.decision];members=(p.decision,)
    if p.group:
        if p.group not in groups or p.decision not in groups[p.group].members:raise ValueError('Review does not belong to declared joint group')
        members=groups[p.group].members
    if p.target_model not in {m for d in members for m in decisions[d].models}:
        raise ValueError('Final target is not part of reviewed decision/group')
    if len(set(p.probe_hypotheses))!=len(p.probe_hypotheses) or any(h not in hypotheses for h in p.probe_hypotheses):
        raise ValueError('Invalid/duplicate probe hypotheses')
    findings=[];blockers=[];revisions=[];uncertainties=[]
    def add(kind,subject,status,detail):findings.append(Finding(kind,subject,status,detail))
    edges={edge.id:edge for edge in c.transitions};proven=set()
    for proof in e.transitions:
        if proof.edge not in edges or proof.edge in proven:raise ValueError('Unknown/duplicate transition proof')
        edge=edges[proof.edge];proof.receipt.verify()
        if (proof.receipt.plan!=edge.plan or proof.receipt.versions[0].fingerprint()!=models[edge.source].fingerprint
            or proof.receipt.architecture.fingerprint()!=models[edge.target].fingerprint):
            raise ValueError('Transition proof source/plan/target mismatch')
        proven.add(proof.edge);add('transition',proof.edge,'replayed','Exact operation chain and version identities')
    relevant={p.reference_model,p.target_model}|{m for d in members for m in decisions[d].models}
    while True:
        parents={edge.source for edge in c.transitions if edge.target in relevant}-relevant
        if not parents:break
        relevant.update(parents)
    for edge in c.transitions:
        if edge.target in relevant and edge.id not in proven:
            uncertainties.append('transition:'+edge.id);add('transition',edge.id,'unknown','Declared version edge needs its actual transform receipt')
    # Replays are keyed by exact model identity, never by mutable labels.
    replay_by={}
    for replay in e.replays:
        replay.verify();fingerprint=replay.architecture.fingerprint()
        if replay in replay_by.get(fingerprint,[]):raise ValueError('Duplicate identical replay')
        if fingerprint not in {m.fingerprint for m in models.values()}:raise ValueError('Evidence model is outside the campaign')
        replay_by.setdefault(fingerprint,[]).append(replay)
    def checked(model,key,*,milestone=False):
        statuses=[]
        for replay in replay_by.get(models[model].fingerprint,[]):
            found=[x for x in replay.checks if x.rule==key.rule and x.subject==key.subject]
            usable=len(found)==1 and replay.trace is not None and (not milestone or replay.status!='inconsistent')
            statuses.append(found[0].status if usable else 'unknown')
        if not statuses or 'unknown' in statuses:return 'unknown'
        return statuses[0] if len(set(statuses))==1 else 'conflict'
    specs={}
    for spec in p.milestones:
        if spec.decision not in decisions or spec.decision in specs or spec.model not in decisions[spec.decision].models or not spec.checks or len(set(spec.checks))!=len(spec.checks):
            raise ValueError('Invalid/duplicate milestone specification')
        specs[spec.decision]=spec
        statuses=[checked(spec.model,key,milestone=True) for key in spec.checks]
        status='matched' if all(x=='matched' for x in statuses) else 'mismatch' if 'mismatch' in statuses else 'unknown'
        add('milestone',spec.decision,status,'Frozen check set; completion labels alone do not certify this scope')
    completed={f.subject for f in findings if f.kind=='milestone' and f.status=='matched'}
    _,ancestors=_dag(decisions,[(source,d.id) for d in c.decisions for source in d.depends_on])
    for prerequisite in sorted(ancestors[p.decision]):
        if prerequisite not in completed or decisions[prerequisite].status!='completed':
            uncertainties.append('prerequisite:'+prerequisite);add('prerequisite',prerequisite,'unknown','Need replayed milestone evidence before this decision')
    seen=set();unprobed=[]
    required={h for d in members for h in decisions[d].hypotheses}
    for req in p.requirements:
        if req.hypothesis not in required or req.hypothesis in seen or req.model not in hypotheses[req.hypothesis].models:
            raise ValueError('Invalid/duplicate/model-mismatched necessary requirement')
        seen.add(req.hypothesis);status=checked(req.model,req.check)
        add('necessary',req.hypothesis,status,'Exact model/check predicate; no general algorithm impossibility claim')
        if status=='conflict':revisions.append('necessary_conflict:'+req.hypothesis)
        elif status=='mismatch':
            (revisions if req.revisable else blockers).append('necessary:'+req.hypothesis)
        elif status!='matched' and req.hypothesis not in p.probe_hypotheses:unprobed.append(req.hypothesis)
    # Every dependency hypothesis requires a frozen predicate or an explicit probe.
    for h in sorted(required-seen):
        add('necessary',h,'unknown','Campaign status label is not replayed evidence')
        if h not in p.probe_hypotheses:unprobed.append(h)
    uncertainties.extend('hypothesis:'+h for h in sorted(set(unprobed)))
    estimates={}
    for cost in p.remaining:
        if cost.budget not in budgets or cost.budget in estimates or cost.unit!=budgets[cost.budget].unit or cost.lower<0 or cost.lower>cost.upper:
            raise ValueError('Invalid/duplicate/units/interval for remaining budget')
        estimates[cost.budget]=cost
    for identity,budget in budgets.items():
        spent=sum((Decimal(str(x.amount)) for x in c.uses if x.budget==identity),Decimal(0))
        remaining=Decimal(str(budget.limit))-spent;estimate=estimates.get(identity)
        if estimate is None:
            uncertainties.append('budget:'+identity);add('budget',identity,'unknown','No estimate for the whole remaining reviewed branch');continue
        if Decimal(str(estimate.lower))>remaining:
            blockers.append('budget:'+identity);status='insufficient'
        elif Decimal(str(estimate.upper))>remaining:
            uncertainties.append('budget:'+identity);status='overlapping'
        else:status='fits'
        add('budget',identity,status,'Limit-spent='+str(remaining)+'; remaining interval=['+str(estimate.lower)+','+str(estimate.upper)+'] '+estimate.unit)
    if not budgets:uncertainties.append('budget:undeclared');add('budget','campaign','unknown','No declared budget')
    predictions={x.id:x for x in c.predictions};seen=set()
    for reading in e.readings:
        key=(reading.prediction,reading.trace.plan.fingerprint(),reading.measurement)
        if reading.prediction not in predictions or key in seen:raise ValueError('Invalid/duplicate prediction observation')
        seen.add(key);prediction=predictions[reading.prediction]
        if reading.model_fingerprint!=models[prediction.model].fingerprint:raise ValueError('Metric reading model identity mismatch')
        reading.trace.verify()
        matches=[m for m in reading.trace.observation.measurements if m.id==reading.measurement]
        if len(matches)!=1:raise ValueError('Unknown/ambiguous measurement')
        measurement=matches[0]
        scope='/'.join((measurement.clock,str(measurement.launch),measurement.subject))
        if (measurement.metric,measurement.unit,scope)!=(prediction.metric,prediction.unit,prediction.scope):
            raise ValueError('Prediction measurement metric/unit/scope differs')
        if measurement.value is None:uncertainties.append('prediction:'+prediction.id);status='unknown'
        else:
            status='matched' if prediction.lower<=measurement.value<=prediction.upper else 'outside'
            if status=='outside' and prediction.decision in members:revisions.append('prediction:'+prediction.id)
        add('prediction',prediction.id,status,'Original interval unchanged; actual='+str(measurement.value)+'; check instrument/source before assigning blame')
    if p.study is None:
        uncertainties.append('model:missing');add('final_gain',p.target_model,'unknown','No complete model study supplied')
    else:
        p.study.verify();ids=[x.id for x in p.study.plan.scenarios]
        if p.study.plan.reference!=p.reference_model or p.target_model not in ids:raise ValueError('Study reference/final target mismatch')
        for scenario in p.study.plan.scenarios:
            if scenario.id not in models or scenario.architecture.fingerprint()!=models[scenario.id].fingerprint:
                raise ValueError('Study model version not bound to campaign')
        gain=p.study.gains[ids.index(p.target_model)]
        if gain.optimum_gain_upper_ns is not None and gain.optimum_gain_upper_ns<p.minimum_gain_ns:
            blockers.append('model:optimistic_insufficient');status='insufficient_within_model'
        elif gain.optimum_gain_lower_ns is not None and gain.optimum_gain_lower_ns>=p.minimum_gain_ns:
            status='supported_within_model'
        else:
            status='unknown';uncertainties.append('model:overlapping_bounds')
        add('final_gain',p.target_model,status,'Optimum gain interval=['+str(gain.optimum_gain_lower_ns)+','+str(gain.optimum_gain_upper_ns)+']ns; witness delta='+str(gain.witness_delta_ns)+'ns')
    if decision.status in ('stopped','suspended','completed'):
        uncertainties.append('decision:inactive');add('state',p.decision,'inactive','Explicit reopen is required; review does not revive a closed decision')
    mode='implementation';probe_fits=False
    if p.probe is not None:
        probe=p.probe
        if not probe.uncertainties or len(set(probe.uncertainties))!=len(probe.uncertainties) or len(probe.outcomes)<2 or len(set(probe.outcomes))!=len(probe.outcomes) or any(not x for x in probe.outcomes):
            raise ValueError('Probe needs explicit unique uncertainty keys and distinguishable outcomes')
        costs={}
        for cost in probe.costs:
            if cost.budget not in budgets or cost.budget in costs or cost.unit!=budgets[cost.budget].unit or cost.lower<0 or cost.lower>cost.upper:
                raise ValueError('Invalid probe budget interval/unit')
            costs[cost.budget]=cost
        probe_fits=bool(budgets) and set(costs)==set(budgets)
        for identity,budget in budgets.items():
            available=Decimal(str(budget.limit))-sum((Decimal(str(x.amount)) for x in c.uses if x.budget==identity),Decimal(0))
            fits=identity in costs and Decimal(str(costs[identity].upper))<=available
            probe_fits=probe_fits and fits
            add('probe_budget',identity,'fits' if fits else 'unknown','Probe cost compared separately from the whole remaining branch')
    if (uncertainties and p.probe is not None and probe_fits and set(uncertainties).issubset(p.probe.uncertainties)
        and not any(x.startswith(('prerequisite:','decision:')) for x in uncertainties)):
        mode='probe'
    action='stop' if blockers else 'revise' if revisions else 'continue' if mode=='probe' else 'unknown' if uncertainties else 'continue'
    for kind,reasons in (('stop_reason',blockers),('revise_reason',revisions),('unknown_reason',uncertainties)):
        for reason in reasons:add(kind,p.decision,'unknown' if kind=='unknown_reason' else action,reason)
    return ReviewReceipt(c,p,e,action,mode,tuple(findings),decision.discriminating_probe,CONDITIONS)


@dataclass(frozen=True)
class StatePlan(VersionedRecord):
    campaign_fingerprint: str
    operation: Literal['advance','complete','stop','suspend','reopen','promote']
    decision: str
    reason: str
    working_model: str = ''
    revival_condition: str = ''
    schema_version: int = 1


@dataclass(frozen=True)
class StateReceipt(VersionedRecord):
    before: Campaign
    plan: StatePlan
    review: ReviewReceipt | None
    after: Campaign
    invalidated_decisions: tuple[str,...]
    conditions: tuple[str,...]
    schema_version: int = 1

    @classmethod
    def from_json(cls,text):
        raw=parse_json(text)
        if type(raw) is dict:
            for field in ('before','after'):_campaign_tags(raw.get(field))
            if type(raw.get('review')) is dict:ReviewReceipt.from_json(_dump(raw['review']))
        return decode_versioned(cls,raw)

    def verify(self):
        if change_state(self.before,self.plan,self.review)!=self:raise ValueError('State receipt differs from replay')
        return True


def change_state(c:Campaign,p:StatePlan,assessment:ReviewReceipt | None=None):
    if p.schema_version!=1 or p.campaign_fingerprint!=c.fingerprint() or not p.reason:raise ValueError('State source/version/reason mismatch')
    errors=[x for x in validate_campaign(c).issues if x.severity=='error']
    if errors:raise ValueError('Invalid campaign: '+str(errors))
    decisions={d.id:d for d in c.decisions};models={m.id for m in c.models}
    if p.decision not in decisions:raise ValueError('Unknown decision')
    if assessment is not None:
        assessment.verify()
        if assessment.campaign!=c or assessment.plan.decision!=p.decision:raise ValueError('State review belongs to another campaign/decision')
    if p.working_model and p.operation not in ('advance','reopen'):raise ValueError('Working pointer argument is inapplicable to this operation')
    if p.revival_condition and p.operation!='reopen':raise ValueError('Revival condition is only valid for reopen')
    if p.operation=='promote':raise ValueError('Delivery best promotion requires protected P5 run evidence, not campaign/model labels')
    invalidated=set();working=c.working_head;best=c.validated_best
    if p.operation=='reopen':
        if not p.revival_condition or decisions[p.decision].status not in ('completed','stopped','suspended'):
            raise ValueError('Reopen needs a closed decision and an explicit changed condition')
        _,ancestors=_dag(decisions,[(a,d.id) for d in c.decisions for a in d.depends_on])
        invalidated={d for d in decisions if d==p.decision or p.decision in ancestors[d]}
        for identity in invalidated:decisions[identity]=replace(decisions[identity],status='planned')
        affected={m for identity in invalidated for m in decisions[identity].models}
        if best in affected:best=None
        if p.working_model:
            if p.working_model not in models:raise ValueError('Unknown reopen working model')
            working=p.working_model
    elif p.operation=='suspend':
        if decisions[p.decision].status=='completed':raise ValueError('Completed decision must be reopened explicitly')
        decisions[p.decision]=replace(decisions[p.decision],status='suspended')
    elif p.operation=='stop':
        if assessment is None or assessment.action!='stop':raise ValueError('Stop requires a replayed stop recommendation')
        decisions[p.decision]=replace(decisions[p.decision],status='stopped')
        _,ancestors=_dag(decisions,[(a,d.id) for d in c.decisions for a in d.depends_on])
        invalidated={d for d in decisions if p.decision in ancestors[d]}
        for identity in invalidated:decisions[identity]=replace(decisions[identity],status='planned')
        affected={m for identity in invalidated|{p.decision} for m in decisions[identity].models}
        if best in affected:best=None
    else:
        if assessment is None or assessment.action!='continue':raise ValueError('Advancement requires a replayed continuation recommendation')
        if p.operation=='complete':
            if assessment.continuation_scope=='probe':raise ValueError('Probe continuation cannot certify implementation completion')
            matched={f.subject for f in assessment.findings if f.kind=='milestone' and f.status=='matched'}
            required={p.decision}
            if assessment.plan.group:required.update(next(g.members for g in c.groups if g.id==assessment.plan.group))
            if not required.issubset(matched):raise ValueError('Completion needs all frozen individual/joint milestone evidence')
            if any(f.kind=='necessary' and f.status!='matched' for f in assessment.findings):raise ValueError('Completion requires matched necessary predicates, not just planned probes')
            decisions[p.decision]=replace(decisions[p.decision],status='completed')
        else:
            if p.working_model not in decisions[p.decision].models:raise ValueError('Working model is outside this decision')
            working=p.working_model;decisions[p.decision]=replace(decisions[p.decision],status='in_progress')
    after=replace(c,working_head=working,validated_best=best,decisions=tuple(decisions[d.id] for d in c.decisions))
    return StateReceipt(c,p,assessment,after,tuple(sorted(invalidated)),CONDITIONS+(
        'This changes an architecture development document only; no git branch, executable, budget or shared state is modified',
        'Reopen invalidates the selected decision and every dependent descendant; affected delivery best is cleared conservatively',
    ))


def cli(args):
    from pathlib import Path
    import sys
    try:
        if args.selftest:
            if args.plan or args.evidence or args.review or args.output:raise ValueError('Selftest consumes no plan/evidence/output')
            from .policy_test import selftest
            return selftest()
        if args.verify:
            if args.plan or args.evidence or args.review:raise ValueError('Frozen replay inputs cannot be overridden')
            raw=Path(args.verify).read_text();data=parse_json(raw)
            receipt=StateReceipt.from_json(raw) if 'before' in data else ReviewReceipt.from_json(raw)
            receipt.verify();text=_dump(dict(status='replayed_conditionally'),pretty=True)
        else:
            if not args.plan:raise ValueError('Explicit frozen plan required')
            c=Campaign.from_json(Path(args.input).read_text())
            if args.review:
                if args.evidence:raise ValueError('State changes consume a frozen review, not replacement evidence')
                receipt=change_state(c,StatePlan.from_json(Path(args.plan).read_text()),ReviewReceipt.from_json(Path(args.review).read_text()))
            else:
                data=parse_json(Path(args.plan).read_text())
                if 'operation' in data:
                    if args.evidence:raise ValueError('State change cannot consume replacement evidence')
                    receipt=change_state(c,StatePlan.from_json(_dump(data)))
                else:
                    e=ReviewEvidence.from_json(Path(args.evidence).read_text()) if args.evidence else ReviewEvidence()
                    receipt=review(c,ReviewPlan.from_json(_dump(data)),e)
            text=receipt.to_json(pretty=True)
        if args.output:
            with Path(args.output).open('x') as stream:stream.write(text+'\n')
        else:print(text)
        return 0
    except (OSError,ValueError,TypeError) as exc:
        print(_dump(dict(status='rejected',error=str(exc))),file=sys.stderr);return 2


def _nc_labels_do_not_complete():
    from .campaign import ModelRef,Decision,BudgetLimit
    c=Campaign('nc','test',(ModelRef('m','0'*64,'fixture://m'),),(),(Decision('d','probe',('m',),(),(),'milestone','probe',('exit',)),),(),(),(),(BudgetLimit('b','cpu','s',1),),(),'m')
    p=ReviewPlan(c.fingerprint(),'d','m','m',(RemainingCost('b','s',0,0),))
    return review(c,p,ReviewEvidence()).action=='unknown'


REFS={'POLICY1_joint_evidence_and_bounds':Ref(
    what=['组合最终价值/里程碑/必要谓词/预算驱动条件建议；中间负收益不自动否决，标签不认证，重开使后代失效。'],
    status='derived',evidence=['E106 policy_test negative intermediates, optimistic-bound failure, wrong instrumentation, budget and reopening controls; P5 delivery evidence remains separate.'],verified='2026-11-03')}
NEGATIVE_CONTROLS={'POLICY1_joint_evidence_and_bounds':_nc_labels_do_not_complete}
