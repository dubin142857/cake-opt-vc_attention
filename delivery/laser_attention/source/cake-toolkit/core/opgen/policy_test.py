"""P4 branch decisions, not synthetic costs presented as kernel speedups."""
from dataclasses import replace
from pathlib import Path
import json,tempfile,unittest
from .architecture import Duration,Evidence,example
from .campaign import (Campaign,ModelRef,Decision,Hypothesis,JointGroup,Prediction,BudgetLimit,BudgetUse)
from .analysis import (EventBinding,Scope,ReplayPlan,Scenario,StudyPlan,replay,study)
from .schedule import SchedulePlan
from .trace import make_plan,ingest
from .policy import (RuleKey,Requirement,MilestoneSpec,RemainingCost,MetricReading,ReviewEvidence,
                     ReviewPlan,ReviewReceipt,StatePlan,StateReceipt,TransitionProof,ProbePlan,review,change_state)


def combination(directory):
    """Declared model costs: both intermediate variants regress; joint wins."""
    models={}
    for identity,duration in (('base',10),('A',12),('B',13),('AB',6)):
        a=example('matmul')
        models[identity]=replace(a,id=identity,events=tuple(replace(e,duration=Duration(duration,duration,Evidence('assumed','fixture://declared-combination'))) for e in a.events))
    scenarios=tuple(Scenario(identity,a,SchedulePlan(a.fingerprint(),strategy='exact'),'Declared joint-cost fixture, not an implemented optimization') for identity,a in models.items())
    proof=study(StudyPlan('base',scenarios))
    decisions=(Decision('stageA','implement A',('A',),(),(),'A timing matches frozen interval','trace stage A',('budget',),status='completed'),
               Decision('stageB','implement B',('B',),('stageA',),('h',),'B timing within frozen interval','distinguish h using stage B',('budget',)),
               Decision('combine','combine A+B',('AB',),('stageA','stageB'),('h',),'joint timing and contract','joint ablation',('model refutation',)))
    c=Campaign('combo','Test joint continuation',tuple(ModelRef(i,a.fingerprint(),'fixture://'+i) for i,a in models.items()),(),decisions,
        (Hypothesis('h','Joint benefit remains possible',('A','B','AB')),),(JointGroup('joint',tuple(d.id for d in decisions),'all milestone predicates then final target'),),
        (Prediction('pa','A','stageA','span','ns',12,12,'clock/launch/core'),Prediction('pab','AB','combine','span','ns',6,6,'clock/launch/core')),
        (BudgetLimit('cpu','development','steps',8),),(BudgetUse('u','cpu','stageA',2,'steps',Evidence('assumed','fixture://accounting')),),'base','base')
    trace,r=timing(directory,'stage-A',models['A'],12)
    p=ReviewPlan(c.fingerprint(),'stageB','AB','base',(RemainingCost('cpu','steps',2,4),),'joint',proof,
        milestones=(MilestoneSpec('stageA','A',(RuleKey('duration',models['A'].events[0].id),)),),probe_hypotheses=('h',))
    e=ReviewEvidence((r,),(MetricReading('pa',models['A'].fingerprint(),trace,'report:0:span'),))
    return c,p,e,models


def timing(directory,name,a,elapsed):
    trace=Path(directory)/(name+'.trace.json');report=Path(directory)/(name+'.report.json')
    assert not trace.exists() and not report.exists()
    trace.write_text(json.dumps(dict(traceEvents=[dict(ph='X',pid='core',tid='pipe',name='work',ts=0,dur=elapsed)])))
    report.write_text(json.dumps({'core':dict(duration_us=elapsed/1000,running_us=elapsed/1000)}))
    parent=ingest(make_plan(str(trace),str(report),timestamp_unit='ns',clock='clock',single_launch='launch'))
    p=ReplayPlan(a.fingerprint(),parent.observation.fingerprint(),SchedulePlan(a.fingerprint()),Scope('clock','launch',('core',)),
        (EventBinding(a.events[0].id,(parent.observation.events[0].id,),True,Evidence('assumed','fixture://scope')),))
    return parent,replay(a,parent.observation,p,trace=parent)


class PolicyTests(unittest.TestCase):
    def test_negative_intermediates_continue_with_final_joint_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,models=combination(directory);r=review(c,p,e)
            self.assertEqual([x.upper_ns for x in p.study.results],[10,12,13,6])
            self.assertEqual(r.action,'continue',r.findings)
            self.assertTrue(any(f.kind=='final_gain' and '[4,4]' in f.detail for f in r.findings))
            self.assertTrue(any(f.kind=='prediction' and f.status=='matched' for f in r.findings))
            self.assertTrue(r.verify());self.assertEqual(ReviewReceipt.from_json(r.to_json()),r)
            self.assertEqual(c.predictions[0].lower,12)

    def test_actual_joint_14_outside_frozen_6_revises_not_rewrites(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);trace,r=timing(directory,'joint-bad',m['AB'],14)
            before=c.to_json();e=replace(e,readings=e.readings+(MetricReading('pab',m['AB'].fingerprint(),trace,'report:0:span'),))
            result=review(c,p,e);self.assertEqual(result.action,'revise')
            self.assertEqual(c.to_json(),before);self.assertTrue(any(f.status=='outside' and '14' in f.detail for f in result.findings))

    def test_refuted_necessary_check_stops_exact_branch_or_revises_model(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);_,bad=timing(directory,'bad-A',m['A'],14)
            p=replace(p,requirements=(Requirement('h','A',RuleKey('duration',m['A'].events[0].id)),))
            # stageA prerequisite still lacks a matching proof, yet necessary failure is decisive for this declared branch.
            evidence=ReviewEvidence((bad,),())
            self.assertEqual(review(c,p,evidence).action,'stop')
            p=replace(p,requirements=(replace(p.requirements[0],revisable=True),))
            self.assertEqual(review(c,p,evidence).action,'revise')
            # A bare observation manifest cannot upgrade the same conflict to verified refutation.
            bare=replace(bad,trace=None)
            self.assertEqual(review(c,p,ReviewEvidence((bare,),())).action,'unknown')

    def test_supported_completed_labels_do_not_replace_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);c=replace(c,hypotheses=(replace(c.hypotheses[0],status='supported'),))
            p=replace(p,campaign_fingerprint=c.fingerprint(),probe_hypotheses=())
            result=review(c,p,ReviewEvidence());self.assertEqual(result.action,'unknown')
            self.assertTrue(any(f.kind=='prerequisite' for f in result.findings))

    def test_unknown_hypothesis_can_be_next_discriminating_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            self.assertEqual(review(c,p,e).action,'continue')
            self.assertEqual(review(c,replace(p,probe_hypotheses=()),e).action,'unknown')

    def test_budget_lower_upper_missing_and_decimal_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            for lower,upper,expected in ((7,8,'stop'),(5,7,'unknown'),(0,6,'continue')):
                self.assertEqual(review(c,replace(p,remaining=(RemainingCost('cpu','steps',lower,upper),)),e).action,expected)
            self.assertEqual(review(c,replace(p,remaining=()),e).action,'unknown')
            c=replace(c,budgets=(replace(c.budgets[0],limit=0.3),),uses=(replace(c.uses[0],amount=0.1),replace(c.uses[0],id='u2',amount=0.2)))
            p=replace(p,campaign_fingerprint=c.fingerprint(),remaining=(RemainingCost('cpu','steps',0,0),))
            self.assertEqual(review(c,p,e).action,'continue')

    def test_optimistic_final_bound_can_stop_but_witness_delta_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            self.assertEqual(review(c,replace(p,minimum_gain_ns=5),e).action,'stop')
            from .schedule import Cost
            scenarios=tuple(replace(s,plan=replace(s.plan,costs=tuple(Cost(ev.id,Duration(1,ev.duration.upper_ns,Evidence('assumed'))) for ev in s.architecture.events))) for s in p.study.plan.scenarios)
            uncertain=study(replace(p.study.plan,scenarios=scenarios))
            self.assertGreater(uncertain.gains[3].witness_delta_ns,0)
            self.assertEqual(review(c,replace(p,study=uncertain),e).action,'unknown')

    def test_bad_units_scope_model_source_and_ambiguity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            invalid=(replace(p,campaign_fingerprint='0'*64),replace(p,remaining=(RemainingCost('cpu','seconds',0,1),)),
                     replace(p,group='missing'),replace(p,target_model='missing'),replace(p,remaining=p.remaining+p.remaining))
            for bad in invalid:
                with self.assertRaises(ValueError):review(c,bad,e)
            for reading in (replace(e.readings[0],model_fingerprint='0'*64),replace(e.readings[0],measurement='missing')):
                with self.assertRaises(ValueError):review(c,p,replace(e,readings=(reading,)))
            c2=replace(c,predictions=(replace(c.predictions[0],scope='wrong'),)+c.predictions[1:])
            with self.assertRaises(ValueError):review(c2,replace(p,campaign_fingerprint=c2.fingerprint()),e)
            with self.assertRaises(ValueError):review(c,p,replace(e,replays=e.replays+e.replays))

    def test_advance_keeps_best_and_joint_completion_needs_every_milestone(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);r=review(c,p,e)
            plan=StatePlan(c.fingerprint(),'advance','stageB','implement the next joint member',working_model='B')
            result=change_state(c,plan,r)
            self.assertEqual(result.after.working_head,'B');self.assertEqual(result.after.validated_best,'base')
            self.assertEqual(c.working_head,'base');self.assertTrue(result.verify())
            self.assertEqual(StateReceipt.from_json(result.to_json()),result)
            with self.assertRaises(ValueError):change_state(c,replace(plan,operation='complete'),r)
            with self.assertRaises(ValueError):change_state(c,replace(plan,operation='promote'),r)

    def test_reopen_invalidates_descendants_and_best_not_unrelated_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            c=replace(c,decisions=tuple(replace(d,status='completed') for d in c.decisions),validated_best='AB')
            plan=StatePlan(c.fingerprint(),'reopen','stageA','model invalidated',revival_condition='new occupancy model and independent capture')
            result=change_state(c,plan)
            self.assertEqual(result.invalidated_decisions,('combine','stageA','stageB'));self.assertIsNone(result.after.validated_best)
            self.assertTrue(all(d.status=='planned' for d in result.after.decisions));self.assertTrue(result.verify())
            with self.assertRaises(ValueError):change_state(c,replace(plan,revival_condition=''))
            c2=replace(c,validated_best='base');result=change_state(c2,replace(plan,campaign_fingerprint=c2.fingerprint()))
            self.assertEqual(result.after.validated_best,'base')

    def test_stop_requires_evidence_and_does_not_stop_other_group_members(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);r=review(c,replace(p,minimum_gain_ns=5),e)
            plan=StatePlan(c.fingerprint(),'stop','stageB','Declared model cannot meet the frozen target')
            result=change_state(c,plan,r)
            self.assertEqual(result.after.decisions[1].status,'stopped');self.assertEqual(result.after.decisions[0].status,'completed')
            with self.assertRaises(ValueError):change_state(c,plan)
            self.assertEqual(result.after.validated_best,'base')

    def test_independent_conflicting_trials_revise_without_discarding_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);trace,bad=timing(directory,'repeat-A',m['A'],14)
            p=replace(p,requirements=(Requirement('h','A',RuleKey('duration',m['A'].events[0].id)),))
            evidence=replace(e,replays=e.replays+(bad,),readings=e.readings+(MetricReading('pa',m['A'].fingerprint(),trace,'report:0:span'),))
            r=review(c,p,evidence);self.assertEqual(r.action,'revise')
            self.assertTrue(any(f.kind=='necessary' and f.status=='conflict' for f in r.findings))
            self.assertEqual(len([f for f in r.findings if f.kind=='prediction']),2)
            self.assertEqual(len(r.evidence.replays),2)

    def test_complete_with_joint_evidence_and_all_necessary_predicates(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);proofs=list(e.replays);specs=list(p.milestones)
            for decision,model,elapsed in (('stageB','B',13),('combine','AB',6)):
                _,proof=timing(directory,decision,m[model],elapsed);proofs.append(proof)
                specs.append(MilestoneSpec(decision,model,(RuleKey('duration',m[model].events[0].id),)))
            p=replace(p,milestones=tuple(specs),requirements=(Requirement('h','A',RuleKey('duration',m['A'].events[0].id)),))
            e=replace(e,replays=tuple(proofs));r=review(c,p,e);self.assertEqual(r.action,'continue')
            plan=StatePlan(c.fingerprint(),'complete','stageB','all frozen joint predicates passed')
            result=change_state(c,plan,r);self.assertEqual(result.after.decisions[1].status,'completed')
            self.assertEqual(result.after.validated_best,'base');self.assertTrue(result.verify())
            unknown=review(c,replace(p,requirements=()),e)
            with self.assertRaises(ValueError):change_state(c,plan,unknown)

    def test_stopped_prerequisite_cannot_be_overridden_by_old_good_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            c=replace(c,decisions=tuple(replace(d,status='stopped') if d.id=='stageA' else d for d in c.decisions))
            r=review(c,replace(p,campaign_fingerprint=c.fingerprint()),e)
            self.assertEqual(r.action,'unknown')

    def test_failed_joint_branch_does_not_refute_members(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            p=replace(p,decision='combine',minimum_gain_ns=5)
            r=review(c,p,e);self.assertEqual(r.action,'stop')
            result=change_state(c,StatePlan(c.fingerprint(),'stop','combine','target not met in this joint model'),r)
            self.assertEqual(result.after.decisions[:2],c.decisions[:2])
            self.assertEqual(result.after.decisions[2].status,'stopped')

    def test_transitive_closed_prerequisite_and_ignored_state_arguments_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            c=replace(c,decisions=tuple(replace(d,status='stopped') if d.id=='stageA' else replace(d,status='completed') if d.id=='stageB' else replace(d,depends_on=('stageB',)) for d in c.decisions))
            p=replace(p,campaign_fingerprint=c.fingerprint(),decision='combine')
            r=review(c,p,e);self.assertEqual(r.action,'unknown')
            self.assertTrue(any(f.kind=='prerequisite' and f.subject=='stageA' for f in r.findings))
            with self.assertRaises(ValueError):change_state(c,StatePlan(c.fingerprint(),'suspend','combine','pause',working_model='AB'))

    def test_milestone_cannot_ignore_other_conflict_in_same_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            # The duration predicate matches, but a declared program edge is contradicted.
            from .architecture import Event,Dependency
            a=m['A'];other=replace(a.events[0],id='other',reads=(),writes=(),semantic_values=())
            a=replace(a,events=a.events+(other,),dependencies=(Dependency(a.events[0].id,'other','program'),))
            trace=Path(directory)/'conflict.trace.json'
            trace.write_text(json.dumps(dict(traceEvents=[dict(ph='X',pid='core',tid='pipe',name=n,ts=0,dur=12) for n in ('a','b')])))
            parent=ingest(make_plan(str(trace),timestamp_unit='ns',clock='clock',single_launch='launch'))
            rp=ReplayPlan(a.fingerprint(),parent.observation.fingerprint(),SchedulePlan(a.fingerprint()),Scope('clock','launch',('core',)),tuple(EventBinding(event.id,(record.id,),True) for event,record in zip(a.events,parent.observation.events)))
            r=replay(a,parent.observation,rp,trace=parent);self.assertEqual(r.status,'inconsistent')
            c=replace(c,models=tuple(replace(x,fingerprint=a.fingerprint()) if x.id=='A' else x for x in c.models))
            scenarios=tuple(replace(s,architecture=a,plan=SchedulePlan(a.fingerprint())) if s.id=='A' else s for s in p.study.plan.scenarios)
            p=replace(p,campaign_fingerprint=c.fingerprint(),study=study(replace(p.study.plan,scenarios=scenarios)))
            result=review(c,p,ReviewEvidence((r,),()));self.assertEqual(result.action,'unknown')
            self.assertTrue(any(f.kind=='milestone' and f.status=='unknown' for f in result.findings))

    def test_transform_chain_binds_source_plan_target_and_exports_contract(self):
        from .transforms_test import pipeline
        from .transforms import Compose,Reorder,BufferRename,TransformPlan,apply_plan
        from .campaign import EvolutionEdge
        source=pipeline();tp=TransformPlan(source.fingerprint(),Compose((Reorder(('p0','p1','c0','c1')),BufferRename('t1','new'))),'assume_unchanged')
        transformed=apply_plan(source,tp);target=transformed.architecture
        studyproof=study(StudyPlan('base',(Scenario('base',source,SchedulePlan(source.fingerprint()),'source'),Scenario('new',target,SchedulePlan(target.fingerprint()),'conditional transforms'))))
        c=Campaign('transforms','test', (ModelRef('base',source.fingerprint(),'fixture://base'),ModelRef('new',target.fingerprint(),'fixture://new')),
            (EvolutionEdge('edge','base','new',tp),),(Decision('d','transform',('new',),(),(),'contract exported','check transformed contract',('exit',)),),(),(),(),
            (BudgetLimit('b','cpu','steps',1),),(),'base','base')
        p=ReviewPlan(c.fingerprint(),'d','new','base',(RemainingCost('b','steps',0,1),),study=studyproof)
        self.assertEqual(review(c,p,ReviewEvidence()).action,'unknown')
        e=ReviewEvidence(transitions=(TransitionProof('edge',transformed),));r=review(c,p,e)
        self.assertEqual(r.action,'continue');self.assertIsNotNone(r.plan.study.results[1].implementation)
        self.assertTrue(r.verify());self.assertEqual(ReviewReceipt.from_json(r.to_json()),r)
        earlier=replace(c,models=c.models+(ModelRef('earlier',source.fingerprint(),'fixture://earlier'),),transitions=(EvolutionEdge('earlier-edge','earlier','base',TransformPlan(source.fingerprint(),Reorder(source.dispatch_priority),'assume_unchanged')),)+c.transitions)
        self.assertEqual(review(earlier,replace(p,campaign_fingerprint=earlier.fingerprint()),e).action,'unknown')
        raw=json.loads(r.to_json());del raw['evidence']['transitions'][0]['receipt']['plan']['operation']['kind']
        with self.assertRaises(ValueError):ReviewReceipt.from_json(json.dumps(raw))
        forged=replace(transformed,versions=transformed.versions[:-1])
        with self.assertRaises(ValueError):review(c,p,replace(e,transitions=(TransitionProof('edge',forged),)))

    def test_unknown_final_bounds_can_continue_a_budgeted_discriminating_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory)
            p=replace(p,study=None,probe=ProbePlan(('model:missing',),(RemainingCost('cpu','steps',0,1),),('cost bound supported','cost model refuted')))
            r=review(c,p,e);self.assertEqual((r.action,r.continuation_scope),('continue','probe'))
            self.assertTrue(any(f.kind=='unknown_reason' and f.status=='unknown' for f in r.findings))
            state=StatePlan(c.fingerprint(),'advance','stageB','run the frozen discriminating experiment',working_model='B')
            self.assertEqual(change_state(c,state,r).after.working_head,'B')
            with self.assertRaises(ValueError):change_state(c,replace(state,operation='complete',working_model=''),r)
            unknown=review(c,replace(p,probe=replace(p.probe,costs=())),e)
            self.assertEqual(unknown.action,'unknown')
            unknown=review(c,replace(p,probe=replace(p.probe,costs=(RemainingCost('cpu','steps',7,8),))),e)
            self.assertEqual(unknown.action,'unknown')
            with self.assertRaises(ValueError):review(c,replace(p,probe=replace(p.probe,costs=(RemainingCost('cpu','seconds',0,1),))),e)
            with self.assertRaises(ValueError):review(c,replace(p,probe=replace(p.probe,outcomes=('one outcome',))),e)

    def test_tampered_results_parents_and_closed_decision_do_not_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            c,p,e,m=combination(directory);r=review(c,p,e)
            with self.assertRaises(ValueError):replace(r,action='stop').verify()
            with self.assertRaises(ValueError):replace(r,findings=()).verify()
            c2=replace(c,decisions=tuple(replace(d,status='stopped') if d.id=='stageB' else d for d in c.decisions))
            self.assertEqual(review(c2,replace(p,campaign_fingerprint=c2.fingerprint()),e).action,'unknown')
            Path(e.readings[0].trace.plan.sim_report.location).write_text('{}')
            with self.assertRaises(ValueError):r.verify()


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
