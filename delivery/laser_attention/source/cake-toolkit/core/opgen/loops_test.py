"""Finite staged loops: global-array reference and physical pool execution."""
from dataclasses import replace
import json
import unittest
from .architecture import (Allocation,Architecture,Attribute,Binding,Contract,Demand,Dependency,
    Duration,Evidence,Event,Instance,Layout,Platform,Region,Resource,SemanticProgram,Storage,Value,example,validate)
from .loops import (AccessMap,Domain,EdgeDomain,EdgeKey,InterfaceRegion,LoopError,LoopPlan,LoopReceipt,
                    LoopScheduleReceipt,Seed,SlotMap,ViewMap,expand,schedule_loop)
from .schedule import Alignment,SchedulePlan,solve


def seeds_for(a,plan):
    domains={e.id:(0,plan.iterations) for e in a.events}
    domains.update({d.event:(d.start,plan.iterations+d.stop_offset) for d in plan.domains})
    seeds=[]
    for edge in a.dependencies:
        for target in range(*domains[edge.target]):
            predecessor=target-edge.iteration_distance;start,stop=domains[edge.source]
            if not start<=predecessor<stop and edge.kind!='data':
                seeds.append(Seed(EdgeKey.of(edge),target,True,Evidence('assumed','fixture://initial-free','synthetic')))
    return tuple(seeds)


def staged(n=4,tile=1,slots=2):
    count=(n+tile-1)//tile;initial_tile=min(n,tile)
    values=(Value('x','input',(n,1),'f32'),
            Value('t','elementwise',(n,1),'f32',('x',),(Attribute('kind','exp'),)),
            Value('y','elementwise',(n,1),'f32',('t',),(Attribute('kind','neg'),)))
    semantic=SemanticProgram(values,('x',),('y',),Contract('full-exp-neg','exact','NumPy global y=-exp(x)','finite f32; synthetic loop'))
    instances=(Instance('xfull','x','x',Layout((n,1),(4,4))),Instance('yfull','y','y',Layout((n,1),(4,4))),
               Instance('xt','x','x',Layout((initial_tile,1),(4,4)),logical_origin=(0,0)),
               Instance('tt','t','scratch',Layout((initial_tile,1),(4,4)),logical_origin=(0,0)),
               Instance('yt','y','y',Layout((initial_tile,1),(4,4)),logical_origin=(0,0)))
    cost=Duration(2,2,Evidence('assumed','fixture://stage-cost','synthetic'))
    events=(Event('p','producer','compute',(Region('xt',0,initial_tile*4),),(Region('tt',0,initial_tile*4),),(Demand('producer'),),('t',),cost),
            Event('c','consumer','compute',(Region('tt',0,initial_tile*4),),(Region('yt',0,initial_tile*4),),(Demand('consumer'),),('y',),cost))
    edges=(Dependency('p','c','data',1),Dependency('c','p','reuse',slots-1),
           Dependency('p','p','program',1),Dependency('c','c','program',1))
    a=Architecture('staged',semantic,(Storage('gm','GM','global',2*n*4+slots*tile*4),),
        (Resource('producer','producer'),Resource('consumer','consumer')),
        (Allocation('x','gm',n*4),Allocation('y','gm',n*4),Allocation('scratch','gm',tile*4)),instances,events,edges,
        (Binding('x',(),('xfull','xt')),Binding('t',('p',),('tt',)),Binding('y',('c',),('yfull','yt'))),
        dispatch_priority=('p','c'),platform=Platform('synthetic'))
    plan=LoopPlan(a.fingerprint(),count,slots=(SlotMap('scratch',slots),),
        views=(ViewMap('xt',(0,0),(tile,0),tile*4,True),ViewMap('tt',(0,0),(tile,0),0,True),ViewMap('yt',(0,0),(tile,0),tile*4,True)),
        domains=(Domain('p'),Domain('c',1,1)),
        accesses=(AccessMap('p','reads',0,0,'instance'),AccessMap('p','writes',0,0,'instance'),
                  AccessMap('c','reads',0,-1,'instance'),AccessMap('c','writes',0,-1,'instance')),
        inputs=(InterfaceRegion('xfull',0,n*4,Evidence('assumed')),),outputs=(InterfaceRegion('yfull',0,n*4,Evidence('assumed')),),
        cost_policy='assume_unchanged')
    return a,replace(plan,seeds=seeds_for(a,plan))


def run_expanded(receipt,x):
    import numpy as np
    a=receipt.architecture
    scheduled=solve(a,SchedulePlan(a.fingerprint(),alignments=(Alignment('gm',4),)))
    if scheduled.certificate is None:raise AssertionError(scheduled.conditions)
    pools={s.id:bytearray(s.capacity_bytes) for s in a.storages};allocs={x.id:x for x in a.allocations}
    inst={x.id:x for x in a.instances};events={e.id:e for e in a.events};offsets={x.allocation:x.offset_bytes for x in scheduled.certificate.addresses}
    values={v.id:v for v in a.semantic.values}
    def locate(instance,offset=0):
        return pools[allocs[instance.allocation].storage],offsets[instance.allocation]+instance.offset_bytes+offset
    for interface in receipt.plan.inputs:
        pool,start=locate(inst[interface.instance],interface.offset_bytes)
        pool[start:start+interface.span_bytes]=np.asarray(x,dtype=np.float32).tobytes()
    for placement in sorted(scheduled.certificate.placements,key=lambda p:(p.start_ns,p.event)):
        event=events[placement.event]
        if event.kind=='synchronize':continue
        read,write=event.reads[0],event.writes[0]
        pool,start=locate(inst[read.instance],read.offset_bytes)
        data=np.frombuffer(pool[start:start+read.span_bytes],dtype=np.float32).astype(np.float64)
        if event.kind=='copy':out=data.astype(np.float32)
        else:
            operation=next(v.value for v in values[inst[write.instance].value].attributes if v.name=='kind')
            out=np.exp(data).astype(np.float32) if operation=='exp' else (-data).astype(np.float32)
        pool,start=locate(inst[write.instance],write.offset_bytes);pool[start:start+write.span_bytes]=out.tobytes()
    interface=receipt.plan.outputs[0];pool,start=locate(inst[interface.instance],interface.offset_bytes)
    output=np.frombuffer(pool[start:start+interface.span_bytes],dtype=np.float32).copy()
    return output,scheduled


class LoopTests(unittest.TestCase):
    def test_one_two_slots_seed_counts_pipeline_and_drain(self):
        for slots,expected,seeds in ((1,16,3),(2,10,4)):
            a,plan=staged(slots=slots);receipt=expand(a,plan)
            with self.subTest(slots=slots):
                self.assertEqual(receipt.status,'expanded_conditionally',receipt.conditions)
                self.assertEqual(sum(b.status=='seeded' for b in receipt.boundaries),seeds)
                self.assertEqual(len(receipt.events),8)
                self.assertEqual([r.event for r in receipt.events if r.template=='c' and r.iteration==4],list(receipt.epilogue))
                self.assertEqual(receipt.drain,receipt.epilogue)
                result=solve(receipt.architecture,SchedulePlan(receipt.architecture.fingerprint(),alignments=(Alignment('gm',4),)))
                self.assertEqual(result.upper_ns,expected)
                self.assertTrue(all(d.iteration_distance==0 for d in receipt.architecture.dependencies))
                self.assertTrue(receipt.verify())

    def test_global_semantics_and_physical_slot_pool_functional_bits(self):
        import numpy as np
        randomizer=np.random.default_rng(104)
        for slots in (1,2):
            a,plan=staged(slots=slots);receipt=expand(a,plan)
            self.assertEqual(receipt.architecture.semantic,a.semantic)
            for x in randomizer.normal(0,5,(8,4)):
                actual,_=run_expanded(receipt,x)
                expected=-np.exp(x.astype(np.float32).astype(np.float64)).astype(np.float32)
                self.assertTrue(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)))

    def test_tail_clip_span_and_complete_output(self):
        import numpy as np
        a,plan=staged(n=5,tile=2);receipt=expand(a,plan)
        self.assertEqual(receipt.status,'expanded_conditionally',receipt.conditions)
        self.assertEqual(receipt.output_coverage[0].covered_elements,5)
        self.assertEqual(receipt.output_coverage[0].status,'complete')
        tail=next(i for i in receipt.architecture.instances if i.id=='loop:2|tt')
        self.assertEqual(tail.layout.extents,(1,1))
        consumer=next(e for e in receipt.architecture.events if e.id=='loop:3|c')
        self.assertEqual(consumer.reads[0].span_bytes,4);self.assertEqual(consumer.writes[0].span_bytes,4)
        x=np.random.default_rng(105).normal(0,5,5);out,schedule=run_expanded(receipt,x)
        self.assertEqual(schedule.upper_ns,8)
        expected=-np.exp(x.astype(np.float32).astype(np.float64)).astype(np.float32)
        self.assertTrue(np.array_equal(out.view(np.uint32),expected.view(np.uint32)))

    def test_small_nonmultiples_and_shorter_than_tile_against_global_reference(self):
        import numpy as np
        randomizer=np.random.default_rng(106)
        for n in (1,2,3,5,7):
            for tile in (1,2,3):
                for slots in (1,2):
                    a,plan=staged(n=n,tile=tile,slots=slots);receipt=expand(a,plan)
                    self.assertEqual(receipt.status,'expanded_conditionally',receipt.conditions)
                    x=randomizer.normal(0,5,n);out,scheduled=run_expanded(receipt,x)
                    expected=-np.exp(x.astype(np.float32).astype(np.float64)).astype(np.float32)
                    with self.subTest(n=n,tile=tile,slots=slots):
                        self.assertTrue(np.array_equal(out.view(np.uint32),expected.view(np.uint32)))
                        count=(n+tile-1)//tile
                        self.assertEqual(scheduled.upper_ns,4*count if slots==1 else 2*(count+1))

    def test_missing_or_false_control_seed_does_not_disappear(self):
        a,plan=staged()
        for seeds in (plan.seeds[1:],(replace(plan.seeds[0],ready=False),)+plan.seeds[1:]):
            receipt=expand(a,replace(plan,seeds=seeds))
            self.assertEqual(receipt.status,'unknown');self.assertIsNone(receipt.architecture)
            self.assertTrue(any(b.status=='missing' for b in receipt.boundaries))

    def test_data_history_cannot_be_replaced_by_ready_token(self):
        a=example('matmul');eid=a.events[0].id;edge=Dependency(eid,eid,'data',1)
        a=replace(a,dependencies=(edge,));plan=LoopPlan(a.fingerprint(),1,seeds=(Seed(EdgeKey.of(edge),0,True),))
        result=expand(a,plan);self.assertEqual(result.status,'unknown')
        self.assertEqual(result.boundaries[0].status,'data_history_required');self.assertIsNone(result.architecture)

    def test_input_math_identity_does_not_initialize_another_physical_copy(self):
        a,plan=staged();copy=Instance('copy','x','copy_alloc',Layout((1,1),(4,4)),logical_origin=(0,0))
        a=replace(a,allocations=a.allocations+(Allocation('copy_alloc','gm',4),),instances=a.instances+(copy,),
            storages=(replace(a.storages[0],capacity_bytes=a.storages[0].capacity_bytes+4),),
            events=tuple(replace(e,reads=(Region('copy',0,4),)) if e.id=='p' else e for e in a.events))
        plan=replace(plan,source_fingerprint=a.fingerprint(),views=plan.views+(ViewMap('copy',(0,0),(1,0)),))
        result=expand(a,plan);self.assertEqual(result.status,'unknown')
        self.assertTrue(any('Uninitialized' in x for x in result.conditions))

    def test_explicit_copy_event_initializes_local_input_slots(self):
        import numpy as np
        a,plan=staged();copy=Instance('copy','x','copy_alloc',Layout((1,1),(4,4)),logical_origin=(0,0))
        loader=Event('load','producer','copy',(Region('xt',0,4),),(Region('copy',0,4),),(Demand('load'),),('x',),Duration(1,1,Evidence('assumed')))
        a=replace(a,allocations=a.allocations+(Allocation('copy_alloc','gm',4),),instances=a.instances+(copy,),
            storages=(replace(a.storages[0],capacity_bytes=a.storages[0].capacity_bytes+8),),resources=a.resources+(Resource('load','producer'),),
            events=(loader,)+tuple(replace(e,reads=(Region('copy',0,4),)) if e.id=='p' else e for e in a.events),
            dependencies=a.dependencies+(Dependency('load','p','data'),Dependency('p','load','reuse',2)),dispatch_priority=('load','p','c'))
        plan=replace(plan,source_fingerprint=a.fingerprint(),slots=plan.slots+(SlotMap('copy_alloc',2),),
                     views=plan.views+(ViewMap('copy',(0,0),(1,0)),))
        plan=replace(plan,seeds=seeds_for(a,plan))
        result=expand(a,plan);self.assertEqual(result.status,'expanded_conditionally',result.conditions)
        x=np.asarray([1,2,3,4],dtype=np.float32);out,scheduled=run_expanded(result,x)
        self.assertEqual(scheduled.upper_ns,11)
        self.assertTrue(np.array_equal(out.view(np.uint32),(-np.exp(x.astype(np.float64)).astype(np.float32)).view(np.uint32)))

    def test_missing_interface_input_or_wrong_byte_mapping_stays_unknown(self):
        a,plan=staged()
        for changed in (replace(plan,inputs=()),replace(plan,views=tuple(replace(v,offset_step_bytes=0) if v.instance=='xt' else v for v in plan.views))):
            result=expand(a,changed);self.assertEqual(result.status,'unknown');self.assertIsNone(result.architecture)

    def test_output_byte_mapping_and_missing_epilogue_cannot_fake_complete(self):
        a,plan=staged()
        wrong=replace(plan,views=tuple(replace(v,offset_step_bytes=0) if v.instance=='yt' else v for v in plan.views))
        result=expand(a,wrong);self.assertEqual(result.status,'unknown')
        self.assertTrue(any('Output physical' in x for x in result.conditions))
        short=replace(plan,domains=tuple(replace(d,stop_offset=0) if d.event=='c' else d for d in plan.domains))
        result=expand(a,short);self.assertEqual(result.status,'unknown')
        self.assertEqual(result.output_coverage[0].status,'partial');self.assertEqual(result.output_coverage[0].covered_elements,3)

    def test_tail_without_legal_access_span_is_rejected(self):
        a,plan=staged(n=5,tile=2)
        wrong=replace(plan,accesses=tuple(replace(m,span='template') if m.event=='c' and m.mode=='writes' else m for m in plan.accesses))
        with self.assertRaises(LoopError):expand(a,wrong)

    def test_incorrect_reuse_distance_is_not_made_safe_by_slot_count(self):
        a,plan=staged()
        a=replace(a,dependencies=tuple(replace(d,iteration_distance=2) if d.kind=='reuse' else d for d in a.dependencies))
        plan=replace(plan,source_fingerprint=a.fingerprint());plan=replace(plan,seeds=seeds_for(a,plan))
        with self.assertRaises(LoopError):expand(a,plan)

    def test_ordered_stale_alias_is_not_a_valid_reaching_definition(self):
        a,plan=staged(n=3)
        a=replace(a,dependencies=tuple(replace(d,iteration_distance=0) if d.kind=='data' else replace(d,iteration_distance=2) if d.kind=='reuse' else d for d in a.dependencies))
        plan=replace(plan,source_fingerprint=a.fingerprint(),domains=(Domain('p'),Domain('c',2,0)),
            accesses=tuple(replace(m,iteration_shift=-2) if m.event=='c' else m for m in plan.accesses),require_complete_outputs=False)
        plan=replace(plan,seeds=seeds_for(a,plan))
        # The expanded finite graph now rejects the concrete overwrite before
        # the old loop-specific initialization checker can downgrade it to unknown.
        with self.assertRaisesRegex(LoopError,'clobbered_value'):expand(a,plan)

    def test_costs_default_to_unknown_and_old_claims_are_not_certified(self):
        a,plan=staged();result=expand(a,replace(plan,cost_policy='invalidate'))
        self.assertEqual(result.status,'expanded_conditionally')
        scheduled=solve(result.architecture,SchedulePlan(result.architecture.fingerprint()))
        self.assertEqual(scheduled.status,'unknown');self.assertIsNone(scheduled.upper_ns)

    def test_bad_domains_slots_interface_and_budgets_rejected(self):
        a,plan=staged()
        invalid=(replace(plan,source_fingerprint='0'*64),replace(plan,iterations=0),replace(plan,max_events=1),replace(plan,max_objects=1),
                 replace(plan,iterations=100000000),replace(plan,slots=(SlotMap('x',2),)),
                 replace(plan,domains=(Domain('missing'),)),replace(plan,views=(ViewMap('xt',(0,),(1,)),)),
                 replace(plan,inputs=(InterfaceRegion('xfull',0,100000),)),
                 replace(plan,seeds=plan.seeds+(plan.seeds[0],)))
        for item in invalid:
            with self.subTest(plan=item),self.assertRaises(LoopError):expand(a,item)

    def test_event_budget_includes_zero_time_boundary_seeds(self):
        a,plan=staged()
        with self.assertRaises(LoopError):expand(a,replace(plan,max_events=8))

    def test_receipts_json_replay_source_immutability_and_tampering(self):
        a,plan=staged();before=a.to_json();receipt=expand(a,plan)
        self.assertEqual(a.to_json(),before);self.assertTrue(receipt.verify())
        self.assertEqual(LoopReceipt.from_json(receipt.to_json()),receipt)
        with self.assertRaises(ValueError):replace(receipt,drain=()).verify()
        with self.assertRaises(ValueError):replace(receipt,boundaries=()).verify()
        with self.assertRaises(ValueError):replace(receipt,conditions=()).verify()
        raw=json.loads(receipt.to_json());del raw['plan']['schema_version']
        with self.assertRaises(ValueError):LoopReceipt.from_json(json.dumps(raw))

    def test_reversed_affine_iteration_preserves_global_array_values(self):
        import numpy as np
        a,plan=staged()
        a=replace(a,instances=tuple(replace(i,offset_bytes=12) if i.id in ('xt','yt') else i for i in a.instances))
        plan=replace(plan,source_fingerprint=a.fingerprint(),views=tuple(replace(v,origin=(3,0),step=(-1,0),
                     offset_step_bytes=-4 if v.instance in ('xt','yt') else 0) for v in plan.views))
        receipt=expand(a,plan);self.assertEqual(receipt.status,'expanded_conditionally',receipt.conditions)
        x=np.asarray([3,-2,1,4],dtype=np.float32);out,_=run_expanded(receipt,x)
        expected=-np.exp(x.astype(np.float64)).astype(np.float32)
        self.assertTrue(np.array_equal(out.view(np.uint32),expected.view(np.uint32)))
        self.assertEqual(receipt.output_coverage[0].covered_elements,4)

    def test_explicit_edge_domain_can_describe_prologue_without_phantom_prior_work(self):
        a,plan=staged();key=next(EdgeKey.of(d) for d in a.dependencies if d.source==d.target=='p')
        plan=replace(plan,edge_domains=(EdgeDomain(key,1,0),),seeds=tuple(s for s in plan.seeds if s.edge!=key))
        receipt=expand(a,plan);self.assertEqual(receipt.status,'expanded_conditionally',receipt.conditions)
        self.assertEqual(sum(b.status=='seeded' for b in receipt.boundaries),3)
        self.assertFalse(any(b.edge==key and b.target_iteration==0 for b in receipt.boundaries))

    def test_combined_receipt_keeps_initialization_and_drain_provenance(self):
        a,plan=staged();loop=expand(a,plan);pipeline=schedule_loop(loop)
        self.assertEqual(pipeline.upper_ns,10);self.assertTrue(pipeline.verify())
        self.assertEqual(LoopScheduleReceipt.from_json(pipeline.to_json()),pipeline)
        forged=replace(loop,plan=replace(plan,inputs=()))
        with self.assertRaises(ValueError):replace(pipeline,loop=forged).verify()
        with self.assertRaises(ValueError):replace(pipeline,upper_ns=0).verify()
        unready=expand(a,replace(plan,seeds=()));pipeline=schedule_loop(unready)
        self.assertEqual(pipeline.status,'unknown');self.assertIsNone(pipeline.schedule);self.assertTrue(pipeline.verify())

    def test_cli_loop_to_schedule_keeps_chain_and_rejects_partial_override(self):
        import tempfile,contextlib,io
        from pathlib import Path
        from .__main__ import main
        a,plan=staged()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.json';parameters=root/'plan.json';output=root/'loop.json';scheduled=root/'schedule.json'
            source.write_text(a.to_json());parameters.write_text(plan.to_json())
            def run(argv):
                stdout=io.StringIO();stderr=io.StringIO()
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    try:code=main(argv)
                    except SystemExit as exc:code=exc.code
                return code,stdout.getvalue(),stderr.getvalue()
            args=['loop','--input',str(source),'--plan',str(parameters),'--output',str(output)]
            self.assertEqual(run(args)[0],0);saved=output.read_bytes()
            self.assertEqual(run(args)[0],2);self.assertEqual(output.read_bytes(),saved)
            self.assertEqual(run(['loop','--verify',str(output)])[0],0)
            self.assertEqual(run(['loop','--verify',str(output),'--plan',str(parameters)])[0],2)
            self.assertEqual(run(['schedule','--loop-receipt',str(output),'--output',str(scheduled)])[0],0)
            self.assertTrue(LoopScheduleReceipt.from_json(scheduled.read_text()).verify())
            self.assertEqual(run(['schedule','--verify',str(scheduled)])[0],0)
            self.assertEqual(run(['loop','--input',str(source)])[0],2)
            parameters.write_text(replace(plan,seeds=()).to_json());unready=root/'unready.json'
            self.assertEqual(run(['loop','--input',str(source),'--plan',str(parameters),'--output',str(unready)])[0],0)
            code,text,_=run(['schedule','--loop-receipt',str(unready)])
            self.assertEqual(code,0);self.assertEqual(json.loads(text)['status'],'unknown')
            self.assertEqual(run(['schedule','--loop-receipt',str(unready),'--max-states','1'])[0],2)

    def test_declaration_order_does_not_change_expansion_identity(self):
        a,plan=staged();b=replace(a,events=a.events[::-1],dependencies=a.dependencies[::-1],instances=a.instances[::-1])
        self.assertEqual(a.fingerprint(),b.fingerprint())
        x,y=expand(a,plan),expand(b,plan)
        self.assertEqual(x.architecture.fingerprint(),y.architecture.fingerprint())
        self.assertEqual(x.boundaries,y.boundaries);self.assertEqual(x.events,y.events)


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(LoopTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
