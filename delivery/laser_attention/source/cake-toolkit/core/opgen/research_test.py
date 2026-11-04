"""Research document contracts and rollback; no device or trace acquisition."""
from dataclasses import replace
import json
import unittest

from .architecture import Evidence,example
from .observation import (Artifact,Provenance,ObservedAccess,ObservedEvent,ObservedRelation,
    Measurement,Coverage,ObservedExecution,validate_observation)
from .campaign import (ModelRef,EvolutionEdge,Hypothesis,Decision,JointGroup,Prediction,
    BudgetLimit,BudgetUse,Campaign,validate_campaign)
from .legacy import (LegacySummary,LegacyAttachment,LegacyBinding,snapshot_legacy,
                     restore_legacy,validate_legacy,validate_attachment)
from .transforms import TransformPlan,BufferRename,Reorder,apply_plan


def observation_fixture():
    artifact=Artifact('trace','trace','fixture://trace','a'*64,100)
    events=(ObservedEvent('load','c0','mte2','trace',0,5.5,'load','trace','row:1','L0',
                          accesses=(ObservedAccess('GM','read',0,4),),memory_kind='access',sync_kind='none'),
            ObservedEvent('work','c0','vec','trace',6,10,'work','trace','row:2','L0',memory_kind='none',sync_kind='none'))
    return ObservedExecution('obs',events,(ObservedRelation('load','work','observed_order'),),(),
                             Provenance('fixture','1',(artifact,)),
                             Coverage(2,2,events_expected=2,cores_expected=('c0',),launches_expected=('L0',),
                                      addresses='declared_complete',synchronization='declared_complete'))


def campaign_fixture():
    a=example('dense_attention');p1=TransformPlan(a.fingerprint(),BufferRename('score','new_score'))
    b=apply_plan(a,p1).architecture;p2=TransformPlan(b.fingerprint(),Reorder(tuple(e.id for e in b.events)[::-1]))
    ab=apply_plan(b,p2).architecture
    models=tuple(ModelRef(key,model.fingerprint(),'fixture://'+key) for key,model in (('base',a),('a',b),('ab',ab)))
    decisions=(Decision('A','allocate another scratch',('a',),(),('capacity',),'new storage realized','allocation probe',('unrealizable capacity',)),
               Decision('B','overlap independent chains',('ab',),('A',),('capacity',),'joint schedule','pipeline replay',('wrong prerequisite','budget exhaustion')))
    predictions=(Prediction('baseline','base','A','full_time','ns',40,40,'synthetic',evidence=Evidence('assumed')),
                 Prediction('intermediate','a','A','full_time','ns',44,48,'synthetic','base',Evidence('assumed')),
                 Prediction('combined','ab','B','full_time','ns',28,32,'synthetic','base',Evidence('assumed')))
    return Campaign('campaign','test combinations, not hardware performance',models,
                    (EvolutionEdge('A','base','a',p1),EvolutionEdge('AB','a','ab',p2)),decisions,
                    (Hypothesis('capacity','extra scratch fits',('a','ab')),),
                    (JointGroup('joint',('A','B'),'joint full time below baseline'),),predictions,
                    (BudgetLimit('cpu','development','minutes',20),),
                    (BudgetUse('use1','cpu','A',2,'minutes',Evidence('assumed')),),'ab','base')


def legacy_fixture():
    from .ir import Schedule
    s=Schedule('legacy',soc='ascend910b2',note='all fields',requires=('cv_direct_path',))
    s.buf('x','GM',64,note='input')
    s.buf('scratch','UB',16,double=True,lifetime=('compute','consume'),note='ping pong')
    s.buf('y','GM',64,note='output')
    s.stage('compute','vector',reads=('x',),writes=('scratch',),macs=1,elem_ops=7,instr=3,
            tile_bytes=16,tasks=2,scalar_ops=5,note='compute note',times=4)
    s.stage('consume','vector',reads=('scratch',),writes=('y',),tasks=3,times=2)
    s.move('read','GM','UB',32,8,unique=64,reuse_window=128,note='move note')
    return s


class ResearchTests(unittest.TestCase):
    def errors(self,result):
        return {x.code for x in result.issues if x.severity=='error'}

    def unknowns(self,result):
        return {x.code for x in result.issues if x.severity=='unknown'}

    def test_package_registry_preserves_all_child_provenance(self):
        from . import REFS,_REF_MODULES
        from ..trust import load_refs
        discovered=load_refs(['opgen'])
        for module in _REF_MODULES:
            for key,ref in getattr(module,'REFS',{}).items():
                qualified=module.__name__.rsplit('.',1)[-1]+'.'+key
                self.assertIs(REFS[qualified],ref)
                self.assertIs(discovered['opgen.'+qualified],ref)

    def test_document_roundtrips_and_content_identity(self):
        for obj in (observation_fixture(),campaign_fixture(),snapshot_legacy(legacy_fixture())):
            with self.subTest(document=type(obj).__name__):
                restored=type(obj).from_json(obj.to_json(pretty=True))
                self.assertEqual(obj,restored);self.assertEqual(obj.fingerprint(),restored.fingerprint())

    def test_common_codec_rejects_unknown_versions_keys_nonfinite_types(self):
        for obj in (observation_fixture(),campaign_fixture(),snapshot_legacy(legacy_fixture())):
            for mutation in ('version','unknown','bool_version','missing_version'):
                raw=json.loads(obj.to_json())
                if mutation=='version':raw['schema_version']=2
                if mutation=='unknown':raw['hidden']=True
                if mutation=='bool_version':raw['schema_version']=True
                if mutation=='missing_version':del raw['schema_version']
                with self.subTest(document=type(obj).__name__,mutation=mutation),self.assertRaises(ValueError):
                    type(obj).from_json(json.dumps(raw))
            with self.assertRaises(ValueError):type(obj).from_json('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(ValueError):type(obj).from_json('{"schema_version":1,"hidden":NaN}')
        with self.assertRaises(ValueError):replace(observation_fixture().events[0],start_ns=float('inf'))
        with self.assertRaises(ValueError):replace(campaign_fixture().budgets[0],limit=True)
        with self.assertRaises(ValueError):replace(observation_fixture(),events=list(observation_fixture().events))

    def test_observation_validity_does_not_infer_dependencies_or_trust(self):
        o=observation_fixture();result=validate_observation(o)
        self.assertFalse(self.errors(result));self.assertEqual(result.status,'conditional')
        self.assertIn('provenance_not_verified',self.unknowns(result));self.assertIn('semantics_unknown',self.unknowns(result))
        self.assertFalse(hasattr(o,'dependencies'))
        self.assertEqual(o.events[0].end_ns,5.5)

    def test_observation_event_declaration_order_does_not_create_order(self):
        o=observation_fixture();reversed_events=replace(o,events=o.events[::-1])
        self.assertEqual(o.fingerprint(),reversed_events.fingerprint())
        self.assertEqual(o.relations,reversed_events.relations)

    def test_observation_reversed_and_overlapping_time_rejected(self):
        o=observation_fixture()
        for event in (replace(o.events[1],end_ns=0),replace(o.events[1],start_ns=3)):
            self.assertTrue(self.errors(validate_observation(replace(o,events=(o.events[0],event)))))

    def test_observation_clock_and_launch_domains_cannot_be_collapsed(self):
        o=observation_fixture()
        for field,value,code in (('clock','unrelated','clock_domain'),('launch','L1','launch_boundary')):
            other=replace(o.events[1],**{field:value})
            modified=replace(o,events=(o.events[0],other),coverage=replace(o.coverage,launches_expected=None))
            self.assertIn(code,self.errors(validate_observation(modified)))
            # Multiple domains/scopes can coexist without a false cross-domain order.
            self.assertFalse(self.errors(validate_observation(replace(modified,relations=()))))

    def test_observation_missing_address_and_launch_remain_unknown(self):
        o=observation_fixture();event=replace(o.events[0],launch=None,accesses=())
        o=replace(o,events=(event,o.events[1]),coverage=replace(o.coverage,addresses='partial',events_expected=None,cores_expected=None,launches_expected=None))
        result=validate_observation(o);self.assertFalse(self.errors(result))
        self.assertTrue({'address_coverage','launch_boundary','scope_coverage','event_coverage'}.issubset(self.unknowns(result)))

    def test_false_complete_address_and_sync_claims_rejected(self):
        o=observation_fixture()
        missing=replace(o.events[0],accesses=(ObservedAccess('GM','read'),))
        self.assertIn('address_coverage',self.errors(validate_observation(replace(o,events=(missing,o.events[1])))))
        unpaired=replace(o.events[0],sync_kind='wait')
        self.assertIn('synchronization_coverage',self.errors(validate_observation(replace(o,events=(unpaired,o.events[1])))))

    def test_observation_candidates_are_not_proven_edges(self):
        o=observation_fixture();relation=ObservedRelation('load','work','synchronization_candidate',Evidence('measured','fixture://candidate'))
        result=validate_observation(replace(o,relations=(relation,)))
        self.assertFalse(self.errors(result));self.assertIn('relation_candidate',self.unknowns(result))

    def test_observation_sources_and_manifest_hashes_required(self):
        o=observation_fixture()
        artifact=replace(o.provenance.artifacts[0],sha256='bad')
        self.assertIn('artifact',self.errors(validate_observation(replace(o,provenance=replace(o.provenance,artifacts=(artifact,))))))
        event=replace(o.events[0],source='not declared')
        self.assertIn('event_reference',self.errors(validate_observation(replace(o,events=(event,o.events[1])))))

    def test_source_free_boundary_is_explicit_but_not_self_certifying(self):
        o=observation_fixture();source=Artifact('kernel','kernel_source','fixture://kernel','b'*64,10)
        o=replace(o,provenance=replace(o.provenance,artifacts=o.provenance.artifacts+(source,)))
        self.assertIn('source_boundary',self.errors(validate_observation(o)))
        with_source=replace(o,provenance=replace(o.provenance,scope='with_source'))
        self.assertFalse(self.errors(validate_observation(with_source)))
        self.assertIn('provenance_not_verified',self.unknowns(validate_observation(with_source)))

    def test_coverage_counts_cannot_drop_unsupported_records(self):
        o=observation_fixture()
        for coverage in (replace(o.coverage,records_seen=3),replace(o.coverage,records_accepted=1),replace(o.coverage,events_expected=1)):
            self.assertIn('coverage_counts',self.errors(validate_observation(replace(o,coverage=coverage))))
        unknown=replace(o,coverage=replace(o.coverage,records_seen=3,records_unsupported=1,events_expected=3,addresses='partial',synchronization='partial'))
        self.assertFalse(self.errors(validate_observation(unknown)))
        self.assertIn('unsupported_records',self.unknowns(validate_observation(unknown)))

    def test_unsupported_execution_records_prevent_complete_coverage_claim(self):
        o=observation_fixture()
        o=replace(o,coverage=replace(o.coverage,records_seen=3,records_unsupported=1,events_expected=3))
        errors=self.errors(validate_observation(o))
        self.assertIn('address_coverage',errors);self.assertIn('synchronization_coverage',errors)

    def test_observation_expected_scope_and_mapping_identity_checked(self):
        o=observation_fixture()
        self.assertIn('scope_coverage',self.errors(validate_observation(replace(o,coverage=replace(o.coverage,cores_expected=('other',))))))
        event=replace(o.events[0],architecture_event='load')
        self.assertIn('model_identity',self.errors(validate_observation(replace(o,events=(event,o.events[1])))))

    def test_independent_report_measurement_stays_unit_and_source_scoped(self):
        o=observation_fixture();m=Measurement('reported','c0','span','cycles',17,'trace',Evidence('measured'),clock='sim',launch='L0')
        o=replace(o,measurements=(m,));self.assertFalse(self.errors(validate_observation(o)))
        self.assertEqual(o.measurements[0].unit,'cycles')
        self.assertIn('measurement',self.errors(validate_observation(replace(o,measurements=(replace(m,source='missing'),)))))

    def test_campaign_allows_negative_intermediate_and_keeps_best_separate(self):
        a=campaign_fixture();result=validate_campaign(a)
        self.assertFalse(self.errors(result));self.assertIn('campaign_policy_pending',self.unknowns(result))
        self.assertGreater(a.predictions[1].lower,a.predictions[0].upper)
        self.assertLess(a.predictions[2].upper,a.predictions[0].lower)
        self.assertEqual(a.working_head,'ab');self.assertEqual(a.validated_best,'base')
        self.assertIn('best_not_verified',self.unknowns(result))

    def test_campaign_graph_declarations_are_order_independent(self):
        a=campaign_fixture()
        self.assertEqual(a.fingerprint(),replace(a,models=a.models[::-1],decisions=a.decisions[::-1],predictions=a.predictions[::-1]).fingerprint())

    def test_campaign_decision_and_evolution_cycles_rejected(self):
        a=campaign_fixture();bad=replace(a.decisions[0],depends_on=('B',))
        self.assertIn('decision_cycle',self.errors(validate_campaign(replace(a,decisions=(bad,a.decisions[1])))))
        edge=a.transitions[1];cycle=replace(edge,target='base')
        self.assertIn('evolution_cycle',self.errors(validate_campaign(replace(a,transitions=(a.transitions[0],cycle)))))

    def test_campaign_source_identity_and_unknown_references_rejected(self):
        a=campaign_fixture();edge=replace(a.transitions[0],plan=replace(a.transitions[0].plan,source_fingerprint='0'*64))
        self.assertIn('transition_identity',self.errors(validate_campaign(replace(a,transitions=(edge,a.transitions[1])))))
        d=replace(a.decisions[0],models=('missing',))
        self.assertIn('decision_reference',self.errors(validate_campaign(replace(a,decisions=(d,a.decisions[1])))))
        self.assertIn('campaign_identity',self.errors(validate_campaign(replace(a,validated_best='missing'))))

    def test_campaign_predictions_budgets_and_units_checked(self):
        a=campaign_fixture()
        bad=replace(a.predictions[0],lower=100,upper=1)
        self.assertIn('prediction_interval',self.errors(validate_campaign(replace(a,predictions=(bad,)))))
        self.assertIn('budget',self.errors(validate_campaign(replace(a,budgets=(replace(a.budgets[0],limit=-1),)))))
        self.assertIn('budget_use',self.errors(validate_campaign(replace(a,uses=(replace(a.uses[0],amount=-1),)))))
        self.assertIn('budget_unit',self.errors(validate_campaign(replace(a,uses=(replace(a.uses[0],unit='seconds'),)))))
        over=replace(a,uses=(replace(a.uses[0],amount=21),));result=validate_campaign(over)
        self.assertFalse(self.errors(result));self.assertIn('budget_exceeded',self.unknowns(result))

    def test_campaign_missing_probe_group_and_hypothesis_scope_rejected(self):
        a=campaign_fixture()
        d=replace(a.decisions[0],discriminating_probe='')
        self.assertIn('decision_contract',self.errors(validate_campaign(replace(a,decisions=(d,a.decisions[1])))))
        self.assertIn('joint_group',self.errors(validate_campaign(replace(a,groups=(replace(a.groups[0],members=('missing',)),)))))
        self.assertIn('hypothesis_reference',self.errors(validate_campaign(replace(a,hypotheses=(replace(a.hypotheses[0],models=()),)))))

    def test_campaign_nested_schema_and_operation_kind_are_explicit(self):
        a=campaign_fixture()
        for target in ('version','kind'):
            raw=json.loads(a.to_json())
            if target=='version':del raw['transitions'][0]['plan']['schema_version']
            else:del raw['transitions'][0]['plan']['operation']['kind']
            with self.subTest(target=target),self.assertRaises(ValueError):Campaign.from_json(json.dumps(raw))

    def test_legacy_all_fields_and_mutation_isolation(self):
        s=legacy_fixture();snap=snapshot_legacy(s)
        self.assertFalse(self.errors(validate_legacy(snap)))
        self.assertEqual((snap.id,snap.soc,snap.note,snap.requires),(s.name,s.soc,s.note,s.requires))
        for b,old in zip(snap.buffers,s.bufs.values()):
            self.assertEqual((b.id,b.space,b.bytes,b.double,b.lifetime,b.note),
                (old.name,old.space,old.bytes,old.double,None if old.lifetime is None else tuple(sorted(old.lifetime)),old.note))
        for stage,old in zip(snap.stages,s.stages):
            for field in old.__slots__:
                self.assertEqual(getattr(stage,'id' if field=='name' else field),getattr(old,field))
            self.assertEqual(stage.repeats,s.stage_times[old.name])
        for move,old in zip(snap.moves,s.moves):
            for field in old.__slots__:
                self.assertEqual(getattr(move,'id' if field=='name' else field),getattr(old,field))
        before=snap.fingerprint();s.bufs['scratch'].bytes=99;s.stages[0].tasks=77;s.moves[0].unique=12
        self.assertEqual(snap.fingerprint(),before);self.assertEqual(snap.buffers[1].bytes,16)
        self.assertIn('legacy_aggregate_only',self.unknowns(validate_legacy(snap)));self.assertFalse(hasattr(snap,'events'))

    def test_legacy_restore_preserves_aggregate_behavior(self):
        s=legacy_fixture();snap=snapshot_legacy(s);restored=restore_legacy(snap)
        self.assertEqual(snapshot_legacy(restored),snap)
        self.assertEqual(s.capacity({'GM':1024,'UB':64}),restored.capacity({'GM':1024,'UB':64}))
        self.assertEqual(s.moves[0].requested(),restored.moves[0].requested())
        self.assertEqual(s.moves[0].dram(1024),restored.moves[0].dram(1024))
        s.bufs['scratch'].lifetime=frozenset()
        self.assertEqual(snapshot_legacy(restore_legacy(snapshot_legacy(s))),snapshot_legacy(s))

    def test_legacy_unknown_fields_and_repeat_keys_not_silently_lost(self):
        s=legacy_fixture();s.new_field=1
        with self.assertRaises(ValueError):snapshot_legacy(s)
        s=legacy_fixture();s.stage_times['hidden']=2
        with self.assertRaises(ValueError):snapshot_legacy(s)
        s=legacy_fixture();s.stages.append(s.stages[0])
        with self.assertRaises(ValueError):snapshot_legacy(s)

    def test_legacy_bad_aggregate_constraints_rejected(self):
        s=snapshot_legacy(legacy_fixture())
        invalid=(replace(s,buffers=(replace(s.buffers[0],bytes=-1),)+s.buffers[1:]),
                 replace(s,stages=(replace(s.stages[0],reads=('missing',)),)+s.stages[1:]),
                 replace(s,buffers=(replace(s.buffers[0],lifetime=('missing',)),)+s.buffers[1:]),
                 replace(s,moves=(replace(s.moves[0],reuse_window=-1),)))
        for item in invalid:
            with self.subTest(item=item):self.assertTrue(self.errors(validate_legacy(item)))
            with self.assertRaises(ValueError):restore_legacy(item)

    def test_legacy_attachment_explicit_coverage_and_identity(self):
        from .ir import Schedule
        a=example('matmul');s=Schedule('small');s.buf('out','GM',4);s.stage('mm','cube',writes=('out',),macs=1,tasks=1)
        snap=snapshot_legacy(s)
        attachment=LegacyAttachment('mapping',snap,a.fingerprint(),(LegacyBinding('buffer','out',('out',)),LegacyBinding('stage','mm',(a.events[0].id,))))
        result=validate_attachment(attachment,a);self.assertFalse(self.errors(result))
        self.assertIn('attachment_not_proof',self.unknowns(result))
        self.assertEqual(LegacyAttachment.from_json(attachment.to_json()),attachment)
        self.assertIn('unmapped_legacy_item',self.unknowns(validate_attachment(replace(attachment,bindings=()),a)))
        self.assertIn('attachment_identity',self.errors(validate_attachment(replace(attachment,architecture_fingerprint='0'*64),a)))
        self.assertIn('attachment_reference',self.errors(validate_attachment(replace(attachment,bindings=(LegacyBinding('buffer','out',('missing',)),)),a)))
        too_large=replace(snap,buffers=(replace(snap.buffers[0],bytes=1000000),))
        self.assertIn('attachment_capacity',self.errors(validate_attachment(replace(attachment,summary=too_large),a)))
        raw=json.loads(attachment.to_json());del raw['summary']['schema_version']
        with self.assertRaises(ValueError):LegacyAttachment.from_json(json.dumps(raw))


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResearchTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
