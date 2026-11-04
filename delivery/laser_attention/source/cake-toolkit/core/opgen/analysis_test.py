"""Independent staged windows, adversarial replay and interval comparisons."""
from dataclasses import replace
import json
import unittest
from .architecture import Attribute,Duration,Evidence,example
from .observation import (Artifact,Coverage,ObservedAccess,ObservedEvent,ObservedExecution,Provenance)
from .schedule import Address,Alignment,Cost,Reservation,SchedulePlan
from .loops import expand
from .loops_test import staged
from .analysis import (Anchor,EventBinding,ReplayPlan,ReplayReceipt,Scenario,Scope,StorageBase,
                       StudyPlan,StudyReceipt,TimingHypothesis,replay,study)


def observed_pipeline():
    template,lp=staged();loop=expand(template,lp);a=loop.architecture
    # Independent closed-form physical layout and double-buffer schedule.
    addresses=(Address('x','gm',0),Address('y','gm',16),Address('slot:0|scratch','gm',32),Address('slot:1|scratch','gm',36))
    offsets={x.allocation:x.offset_bytes for x in addresses};instances={x.id:x for x in a.instances}
    records=[];bindings=[];anchors=[]
    for e in a.events:
        if e.id.startswith('seed:'):anchors.append(Anchor(e.id,0,Evidence('assumed')));continue
        iteration=int(e.id.split('|')[0].split(':')[1]);phase=e.id.split('|')[1]
        start=2*iteration;finish=start+2
        accesses=[]
        for mode,regions in (('read',e.reads),('write',e.writes)):
            for region in regions:
                instance=instances[region.instance]
                accesses.append(ObservedAccess('pool',mode,100+offsets[instance.allocation]+instance.offset_bytes+region.offset_bytes,region.span_bytes,instance.id))
        identity='observed:'+e.id
        records.append(ObservedEvent(identity,e.actor,phase,'clock',1000+start,1000+finish,e.id,'trace',identity,
            launch='launch',architecture_event=e.id,accesses=tuple(accesses),memory_kind='access',sync_kind='none',attributes=(Attribute('trace.phase','X'),)))
        bindings.append(EventBinding(e.id,(identity,),True,Evidence('assumed','fixture://formula')))
    o=ObservedExecution('pipeline',tuple(records),(),(),Provenance('fixture','1',(Artifact('trace','trace','fixture://closed-form','a'*64,0),)),
        Coverage(8,8,events_expected=8,cores_expected=('producer','consumer'),launches_expected=('launch',),addresses='declared_complete',synchronization='declared_complete'),a.fingerprint())
    p=ReplayPlan(a.fingerprint(),o.fingerprint(),SchedulePlan(a.fingerprint(),alignments=(Alignment('gm',4),)),
        Scope('clock','launch',('producer','consumer')),tuple(bindings),tuple(anchors),addresses,(StorageBase('gm','pool',100),))
    return a,o,p,loop


def changed_observation(o,p,**kwargs):
    changed=replace(o,**kwargs)
    return changed,replace(p,observation_fingerprint=changed.fingerprint())


class AnalysisTests(unittest.TestCase):
    def test_independent_windows_addresses_and_loop_parent_replay(self):
        a,o,p,loop=observed_pipeline();result=replay(a,o,p,loop=loop)
        self.assertEqual(result.status,'consistent_conditionally',result.checks)
        self.assertEqual(result.observed_span_ns,10);self.assertEqual(result.mapped_span_ns,10)
        self.assertEqual(result.unmapped_records,());self.assertEqual(result.missing_events,())
        self.assertTrue(all(x.ns>=0 for x in result.slacks));self.assertTrue(result.verify())
        self.assertEqual(ReplayReceipt.from_json(result.to_json()),result)
        with self.assertRaises(ValueError):replace(result,loop=replace(loop,drain=())).verify()
        self.assertEqual(a,loop.architecture)

    def test_early_consumer_and_wait_contamination_are_visible(self):
        a,o,p,_=observed_pipeline();target='observed:loop:1|c'
        oo,pp=changed_observation(o,p,events=tuple(replace(e,start_ns=e.start_ns-1,end_ns=e.end_ns-1) if e.id==target else e for e in o.events))
        result=replay(a,oo,pp);self.assertEqual(result.status,'inconsistent')
        self.assertTrue(any(c.rule=='dependency' and c.status=='mismatch' for c in result.checks))
        oo,pp=changed_observation(o,p,events=tuple(replace(e,end_ns=e.end_ns+5) if e.id==target else e for e in o.events))
        result=replay(a,oo,pp);self.assertEqual(result.status,'inconsistent')
        self.assertTrue(any(r.above_upper_ns==5 for r in result.residuals))
        self.assertEqual(pp.schedule,p.schedule);self.assertEqual(a.events,observed_pipeline()[0].events)

    def test_default_elapsed_comparison_does_not_calibrate_costs(self):
        a,o,p,_=observed_pipeline();p=replace(p,bindings=tuple(replace(b,compare_cost=False) for b in p.bindings))
        result=replay(a,o,p);self.assertEqual(result.status,'unknown')
        self.assertTrue(all(not r.compared for r in result.residuals))
        self.assertEqual(result.architecture,a);self.assertEqual(result.plan.schedule,p.schedule)

    def test_shared_capacity_and_explicit_issue_windows(self):
        a,o,p,_=observed_pipeline()
        # Both stages use one global capacity pool; whole-window assumption fails.
        from .architecture import Demand,Resource
        a=replace(a,resources=(Resource('shared','global'),),events=tuple(replace(e,resources=(Demand('shared'),)) if not e.id.startswith('seed:') else e for e in a.events))
        o=replace(o,architecture_fingerprint=a.fingerprint())
        p=replace(p,source_fingerprint=a.fingerprint(),observation_fingerprint=o.fingerprint(),schedule=SchedulePlan(a.fingerprint()))
        result=replay(a,o,p);self.assertEqual(result.status,'inconsistent')
        self.assertTrue(any(c.rule=='resource' and c.status=='mismatch' for c in result.checks))
        # Zero-duration explicit occupancy is a declared experiment, not inferred hardware.
        reservations=tuple(Reservation(e.id,'shared',0,0,evidence=Evidence('assumed')) for e in a.events if e.resources)
        result=replay(a,o,replace(p,schedule=replace(p.schedule,reservations=reservations)))
        self.assertEqual(result.status,'consistent_conditionally')

    def test_bad_physical_access_alias_capacity_and_missing_address(self):
        a,o,p,_=observed_pipeline();target=o.events[0].id
        altered=tuple(replace(e,accesses=(replace(e.accesses[0],address=9999),)+e.accesses[1:]) if e.id==target else e for e in o.events)
        oo,pp=changed_observation(o,p,events=altered)
        self.assertEqual(replay(a,oo,pp).status,'inconsistent')
        pp=replace(p,addresses=tuple(replace(x,offset_bytes=32) if x.allocation=='slot:1|scratch' else x for x in p.addresses))
        result=replay(a,o,pp);self.assertTrue(any(c.rule=='memory_alias' and c.status=='mismatch' for c in result.checks))
        pp=replace(p,addresses=tuple(replace(x,offset_bytes=400) if x.allocation=='y' else x for x in p.addresses))
        self.assertTrue(any(c.rule=='memory_capacity' and c.status=='mismatch' for c in replay(a,o,pp).checks))
        result=replay(a,o,replace(p,addresses=p.addresses[:-1]));self.assertEqual(result.status,'unknown')
        self.assertTrue(any(c.rule=='memory_binding' and c.status=='unknown' for c in result.checks))

    def test_missing_unknown_addresses_and_sync_do_not_fake_complete(self):
        a,o,p,_=observed_pipeline()
        oo,pp=changed_observation(o,p,coverage=replace(o.coverage,addresses='unknown',synchronization='unknown'))
        self.assertEqual(replay(a,oo,pp).status,'unknown')
        result=replay(a,o,replace(p,bindings=p.bindings[:-1]));self.assertEqual(result.status,'unknown')
        self.assertEqual(len(result.unmapped_records),1);self.assertEqual(len(result.missing_events),1)

    def test_identity_duplicate_scope_and_nonprotocol_anchor_rejected(self):
        a,o,p,_=observed_pipeline()
        invalid=(replace(p,source_fingerprint='0'*64),replace(p,observation_fingerprint='0'*64),
                 replace(p,scope=replace(p.scope,clock='other')),replace(p,scope=replace(p.scope,launch='other')),
                 replace(p,bindings=p.bindings+(p.bindings[0],)),
                 replace(p,bindings=(replace(p.bindings[0],records=(p.bindings[0].records[0],)*2),)+p.bindings[1:]),
                 replace(p,anchors=p.anchors+(Anchor(p.bindings[0].event,0),)),replace(p,tolerance_ns=-1))
        for item in invalid:
            with self.subTest(plan=item),self.assertRaises(ValueError):replay(a,o,item)

    def test_tolerance_preserves_residual_not_silently_fitting(self):
        a,o,p,_=observed_pipeline();identity=o.events[0].id
        oo,pp=changed_observation(o,p,events=tuple(replace(e,end_ns=e.end_ns+0.001) if e.id==identity else e for e in o.events))
        result=replay(a,oo,replace(pp,tolerance_ns=0.002))
        residual=next(r for r in result.residuals if r.event==o.events[0].architecture_event)
        self.assertGreater(residual.above_upper_ns,0)
        self.assertTrue(any(c.rule=='duration' and c.status=='matched' for c in result.checks))
        self.assertEqual(pp.tolerance_ns,0)

    def test_scope_exclusions_are_retained_not_promoted_to_full_capture(self):
        a,o,p,_=observed_pipeline();extra=replace(o.events[0],id='extra',clock='other',architecture_event='')
        oo,pp=changed_observation(o,p,events=o.events+(extra,),coverage=replace(o.coverage,records_seen=9,records_accepted=9,events_expected=9))
        result=replay(a,oo,pp);self.assertEqual(result.outside_scope,('extra',));self.assertEqual(result.status,'consistent_conditionally')
        self.assertTrue(any('slice cannot certify' in x for x in result.conditions))

    def test_zero_time_anchor_cannot_invent_data_initialization(self):
        a,o,p,_=observed_pipeline();identity=p.anchors[0].event
        a=replace(a,events=tuple(replace(e,kind='copy') if e.id==identity else e for e in a.events))
        p=replace(p,source_fingerprint=a.fingerprint(),schedule=replace(p.schedule,source_fingerprint=a.fingerprint()))
        with self.assertRaises(ValueError):replay(a,o,p)

    def test_source_unknown_cost_and_positive_distance_remain_unknown(self):
        a,o,p,_=observed_pipeline()
        p=replace(p,schedule=replace(p.schedule,costs=(Cost(p.bindings[0].event,Duration()),)))
        self.assertEqual(replay(a,o,p).status,'unknown')

    def test_fixed_cost_buffer_and_stage_cost_sensitivity(self):
        cases=[]
        for identity,slots,pcost,ccost in (('one',1,2,2),('two',2,2,2),('producer4',2,4,2),('consumer4',2,2,4)):
            t,lp=staged(slots=slots);loop=expand(t,lp);a=loop.architecture
            overrides=tuple(Cost(e.id,Duration(pcost if e.id.endswith('|p') else ccost,pcost if e.id.endswith('|p') else ccost,Evidence('assumed'))) for e in a.events if not e.id.startswith('seed:'))
            cases.append(Scenario(identity,a,SchedulePlan(a.fingerprint(),strategy='exact',costs=overrides),identity,loop))
        result=study(StudyPlan('one',tuple(cases)))
        self.assertEqual([r.upper_ns for r in result.results],[16,10,18,18])
        gain=result.gains[1];self.assertEqual((gain.optimum_gain_lower_ns,gain.optimum_gain_upper_ns),(6,6))
        self.assertEqual(gain.classification,'conditional_improvement');self.assertTrue(result.verify())
        self.assertEqual(StudyReceipt.from_json(result.to_json()),result)

    def test_witness_difference_is_not_a_proven_gain_when_bounds_overlap(self):
        t,lp=staged(slots=1);a=expand(t,lp).architecture
        t,lp=staged(slots=2);b=expand(t,lp).architecture
        def plan(a):
            costs=tuple(Cost(e.id,Duration(1,2,Evidence('assumed'))) for e in a.events if e.resources)
            return SchedulePlan(a.fingerprint(),costs=costs)
        result=study(StudyPlan('one',(Scenario('one',a,plan(a),'range'),Scenario('two',b,plan(b),'range'))))
        gain=result.gains[1];self.assertEqual(gain.witness_delta_ns,6)
        self.assertEqual(gain.classification,'overlapping_bounds');self.assertLess(gain.optimum_gain_lower_ns,0)

    def test_capacity_failure_unknown_and_wrong_semantics_budget_rejected(self):
        t,lp=staged();a=expand(t,lp).architecture
        a=replace(a,allocations=tuple(replace(x,persistent=False) if x.id.startswith('slot:') else x for x in a.allocations))
        small=replace(a,storages=(replace(a.storages[0],capacity_bytes=32),))
        plan=StudyPlan('base',(Scenario('base',a,SchedulePlan(a.fingerprint()),'base'),Scenario('small',small,SchedulePlan(small.fingerprint()),'capacity experiment')))
        result=study(plan);self.assertIsNone(result.results[1].upper_ns);self.assertEqual(result.gains[1].classification,'unknown')
        with self.assertRaises(ValueError):study(replace(plan,max_scenarios=1))
        other=example('matmul')
        with self.assertRaises(ValueError):study(replace(plan,scenarios=(plan.scenarios[0],Scenario('other',other,SchedulePlan(other.fingerprint()),'different math'))))

    def test_coverage_helper_ignores_unrelated_later_intervals(self):
        from .loops import _covered
        self.assertTrue(_covered(0,4,((0,4),(8,12))))
        self.assertFalse(_covered(0,10,((0,4),(8,12))))

    def test_positive_distance_is_explicitly_unresolved(self):
        from .architecture import Dependency
        a,o,p,_=observed_pipeline()
        a=replace(a,dependencies=a.dependencies+(Dependency('loop:3|p','loop:0|p','program',1),))
        o=replace(o,architecture_fingerprint=a.fingerprint())
        p=replace(p,source_fingerprint=a.fingerprint(),observation_fingerprint=o.fingerprint(),schedule=replace(p.schedule,source_fingerprint=a.fingerprint()))
        result=replay(a,o,p);self.assertEqual(result.status,'unknown')
        self.assertTrue(any(c.rule=='dependency' and c.status=='unknown' for c in result.checks))

    def test_missing_requested_core_and_cross_storage_physical_alias(self):
        a,o,p,_=observed_pipeline()
        pp=replace(p,scope=replace(p.scope,cores=p.scope.cores+('absent',)))
        self.assertEqual(replay(a,o,pp).status,'unknown')
        # Distinct declared arenas can still alias in the same real address space.
        a=replace(a,storages=a.storages+(replace(a.storages[0],id='other'),),
                  allocations=tuple(replace(x,storage='other') if x.id=='y' else x for x in a.allocations))
        o=replace(o,architecture_fingerprint=a.fingerprint())
        p=replace(p,source_fingerprint=a.fingerprint(),observation_fingerprint=o.fingerprint(),schedule=replace(p.schedule,source_fingerprint=a.fingerprint()),
                  addresses=tuple(replace(x,storage='other',offset_bytes=0) if x.allocation=='y' else x for x in p.addresses),
                  storage_bases=p.storage_bases+(StorageBase('other','pool',100),))
        result=replay(a,o,p)
        self.assertTrue(any(c.rule=='memory_alias' and c.status=='mismatch' for c in result.checks))

    def test_trace_parent_is_checked_at_creation_and_replayed(self):
        import tempfile
        from pathlib import Path
        from .trace import make_plan,ingest
        a,o,p,_=observed_pipeline()
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'trace.json'
            source.write_text(json.dumps(dict(traceEvents=[dict(ph='X',pid='core',tid='pipe',name='op',ts=0,dur=1)])))
            parent=ingest(make_plan(str(source),clock='clock',single_launch='launch'))
            # Different observation is rejected even with a genuine parent.
            with self.assertRaises(ValueError):replay(a,o,p,trace=parent)
            source.write_text('{}')
            with self.assertRaises(ValueError):replay(a,o,p,trace=parent)

    def test_timing_projection_has_no_fabricated_mathematics_or_memory(self):
        from .architecture import Demand,Event,Resource
        a,o,p,_=observed_pipeline()
        timing=TimingHypothesis('projection',tuple(replace(e,reads=(),writes=(),semantic_values=()) for e in a.events),a.resources,a.dependencies)
        o=replace(o,architecture_fingerprint='',events=tuple(replace(e,architecture_event='',accesses=tuple(replace(x,instance='') for x in e.accesses)) for e in o.events))
        p=replace(p,source_fingerprint=timing.fingerprint(),observation_fingerprint=o.fingerprint(),schedule=SchedulePlan(timing.fingerprint()),addresses=(),storage_bases=())
        result=replay(timing,o,p);self.assertEqual(result.status,'unknown')
        self.assertFalse(any(c.status=='mismatch' for c in result.checks))
        self.assertTrue(result.verify());self.assertEqual(ReplayReceipt.from_json(result.to_json()),result)
        bad=replace(timing,events=(replace(timing.events[0],semantic_values=('invented',)),)+timing.events[1:])
        with self.assertRaises(ValueError):replay(bad,o,replace(p,source_fingerprint=bad.fingerprint(),schedule=SchedulePlan(bad.fingerprint())))
        # The original complete-Architecture producer requirement is not waived.
        model=example('matmul');model=replace(model,events=tuple(replace(e,writes=()) for e in model.events))
        with self.assertRaises(ValueError):replay(model,o,replace(p,source_fingerprint=model.fingerprint(),schedule=SchedulePlan(model.fingerprint())))

    def test_timing_projection_dependency_and_reservation_validation(self):
        from .architecture import Demand,Dependency,Event,Resource
        t=TimingHypothesis('t',(Event('a','actor','compute',resources=(Demand('r'),)),Event('b','actor','compute',resources=(Demand('r'),))),(Resource('r','actor'),))
        a,o,p,_=observed_pipeline();o=replace(o,architecture_fingerprint='',events=tuple(replace(e,architecture_event='') for e in o.events))
        p=ReplayPlan(t.fingerprint(),o.fingerprint(),SchedulePlan(t.fingerprint()),p.scope,())
        for bad in (replace(t,events=t.events+(t.events[0],)),replace(t,dependencies=(Dependency('a','b','program'),Dependency('b','a','program')))):
            with self.assertRaises(ValueError):replay(bad,o,replace(p,source_fingerprint=bad.fingerprint(),schedule=SchedulePlan(bad.fingerprint())))
        badplan=replace(p.schedule,reservations=(Reservation('a','r',0,2),Reservation('a','r',0,2)))
        with self.assertRaises(ValueError):replay(t,o,replace(p,schedule=badplan))

    def test_receipt_tampering_and_strict_schema(self):
        a,o,p,_=observed_pipeline();result=replay(a,o,p)
        with self.assertRaises(ValueError):replace(result,checks=()).verify()
        with self.assertRaises(ValueError):replace(result,unmapped_records=('fabricated',)).verify()
        raw=json.loads(result.to_json());raw['plan']['tolerance_ns']=float('nan')
        with self.assertRaises(ValueError):ReplayReceipt.from_json(json.dumps(raw))


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
