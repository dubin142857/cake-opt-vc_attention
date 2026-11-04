"""P1 regression tests; synthetic models and old graph counterexamples, no device."""
from dataclasses import replace, FrozenInstanceError
import json
import unittest

from .architecture import (Architecture, Attribute, Allocation, Binding, Claim,
    Contract, Demand, Dependency, Duration, Evidence, Instance, Layout, Region,
    Value, Platform, Storage, Resource, Event, SemanticProgram, example, validate)


def phased_alias_example(order=('sort','gather','zero_acc','consume')):
    # Actual E114 buffer roles/sizes, reduced to a finite source-derived obligation.
    ref='E114 phase replay: arange4096 sort storage; gather resident512; zero acc2048; output resident+acc[:512]'
    values=(Value('q','input',(1,),'f32'),Value('sorted','external.sorted',(4096,),'f32',('q',),external_reference=ref),
        Value('resident','external.select',(512,),'f32',('sorted',),external_reference=ref),
        Value('acc','external.zero',(2048,),'f32',('q',),external_reference=ref),
        Value('out','external.add',(512,),'f32',('resident','acc'),external_reference=ref))
    layout=lambda n:Layout((n,),(4,))
    instances=(Instance('q','q','input',layout(1)),Instance('sort','sorted','scratch',layout(4096)),
        Instance('resident','resident','resident',layout(512)),Instance('acc','acc','scratch',layout(2048)),Instance('out','out','output',layout(512)))
    events=(Event('sort','AIV0','compute',(Region('q',0,4),),(Region('sort',0,16384),),(Demand('V'),),('sorted',)),
        Event('gather','AIV0','compute',(Region('sort',0,16384),),(Region('resident',0,2048),),(Demand('V'),),('resident',)),
        Event('zero_acc','AIV0','compute',(Region('q',0,4),),(Region('acc',0,8192),),(Demand('V'),),('acc',)),
        Event('consume','AIV0','compute',(Region('resident',0,2048),Region('acc',0,2048)),(Region('out',0,2048),),(Demand('V'),),('out',)))
    return Architecture('E114-phased-UB-alias',SemanticProgram(values,('q',),('out',),Contract('phase','exact',ref,'finite alias ordering example')),
        (Storage('UB','UB','AIV0',196608),Storage('GM','GM','global',4096)),(Resource('V','AIV0'),),
        (Allocation('input','GM',4),Allocation('output','GM',2048),Allocation('scratch','UB',16384),Allocation('resident','UB',2048),Allocation('remaining_native_UB','UB',152192)),instances,events,
        tuple(Dependency(x,y,'program') for x,y in zip(order,order[1:])),
        (Binding('q',(),('q',)),Binding('sorted',('sort',),('sort',)),Binding('resident',('gather',),('resident',)),Binding('acc',('zero_acc',),('acc',)),Binding('out',('consume',),('out',))),platform=Platform('Ascend910B2'))


def refreshed_alias_example(span=16384,clobber_span=8192,alias=False):
    a=phased_alias_example();restore=replace(a.events[0],id='restore',writes=(Region('sort_alias' if alias else 'sort',0,span),))
    reset=replace(a.events[2],id='reset');events=tuple(replace(e,writes=(Region('acc',0,clobber_span),)) if e.id=='zero_acc' else e for e in a.events)+(restore,reset)
    order=('sort','zero_acc','restore','gather','reset','consume')
    instances=a.instances+(replace(a.instances[1],id='sort_alias'),) if alias else a.instances
    return replace(a,events=events,instances=instances,dependencies=tuple(Dependency(x,y,'program') for x,y in zip(order,order[1:])))


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.a = example('dense_attention')

    def codes(self, architecture, severity='error'):
        return {x.code for x in validate(architecture).issues if x.severity == severity}

    def test_freshness_e114_good_and_ordered_clobber_have_different_results(self):
        import numpy as np
        good=phased_alias_example();bad=phased_alias_example(('sort','zero_acc','gather','consume'))
        self.assertEqual(self.codes(good),set())
        self.assertIn('clobbered_value',self.codes(bad))
        expected=np.arange(512,dtype=np.float32);memory=np.arange(4096,dtype=np.float32)
        memory[:2048]=0;observed=memory[:512].copy()+memory[:512]
        self.assertEqual(int(np.count_nonzero(observed!=expected)),511)

    def test_freshness_full_refresh_repairs_each_live_value(self):
        self.assertEqual(self.codes(refreshed_alias_example()),set())
        self.assertEqual(self.codes(refreshed_alias_example(alias=True)),set())

    def test_freshness_partial_refresh_covers_only_the_damaged_region(self):
        self.assertEqual(self.codes(refreshed_alias_example(span=2048,clobber_span=2048)),set())
        errors=[x for x in validate(refreshed_alias_example(span=4096)).issues if x.code=='clobbered_value' and x.severity=='error']
        self.assertTrue(any(x.subject=='gather' for x in errors))

    def test_freshness_input_initial_value_is_not_immune_to_alias_writes(self):
        a=phased_alias_example();q=replace(a.instances[0],allocation='scratch')
        b=replace(a,instances=(q,)+a.instances[1:])
        self.assertIn('clobbered_value',self.codes(b))

    def test_freshness_same_value_exact_view_alias_preserves_content(self):
        a=phased_alias_example();alias=replace(a.instances[1],id='same_view')
        mirror=replace(a.events[0],id='mirror',writes=(Region('same_view',0,16384),))
        order=('sort','mirror','gather','zero_acc','consume')
        b=replace(a,instances=a.instances+(alias,),events=a.events+(mirror,),dependencies=tuple(Dependency(x,y,'program') for x,y in zip(order,order[1:])))
        self.assertEqual(self.codes(b),set())
        partial=replace(alias,layout=Layout((512,),(4,)),offset_bytes=4096,logical_origin=(1024,))
        mapped=replace(b,instances=a.instances+(partial,),events=a.events+(replace(mirror,writes=(Region('same_view',0,2048),)),))
        self.assertEqual(self.codes(mapped),set())

    def test_freshness_same_value_with_different_logical_mapping_can_clobber(self):
        a=phased_alias_example();alias=replace(a.instances[1],id='shifted',layout=Layout((512,),(4,)),offset_bytes=4096,logical_origin=(0,))
        mirror=replace(a.events[0],id='mirror',writes=(Region('shifted',0,2048),))
        order=('sort','mirror','gather','zero_acc','consume')
        b=replace(a,instances=a.instances+(alias,),events=a.events+(mirror,),dependencies=tuple(Dependency(x,y,'program') for x,y in zip(order,order[1:])))
        self.assertIn('clobbered_value',self.codes(b))

    def test_freshness_intra_event_refresh_remains_operation_level_unknown(self):
        a=phased_alias_example(('sort','zero_acc','gather','consume'))
        combined=replace(a.events[2],writes=a.events[2].writes+(Region('sort',0,16384),),semantic_values=('acc','sorted'))
        b=replace(a,events=(a.events[0],a.events[1],combined,a.events[3]))
        self.assertEqual(self.codes(b),set())
        self.assertIn('intra_event_value_clobber',self.codes(b,'unknown'))

    def test_freshness_unexpanded_recurrence_stays_unknown(self):
        a=phased_alias_example(('sort','zero_acc','gather','consume'))
        b=replace(a,dependencies=a.dependencies+(Dependency('consume','sort','reuse',1),))
        self.assertNotIn('clobbered_value',self.codes(b))
        self.assertIn('periodic_value_clobber',self.codes(b,'unknown'))

    def test_freshness_bounded_intervals_against_independent_byte_oracle(self):
        import itertools
        spans=[(lo,hi) for lo in range(4) for hi in range(lo+1,5)]
        reads=[(0,4),(0,2),(2,4),(1,3),(0,0)]
        cases=0
        for damage,refresh,read,order in itertools.product(spans,spans+[(0,0)],reads,itertools.permutations(('damage','refresh','read'))):
            lo,hi=read;width=hi-lo;ref='Finite 4-byte oracle: initial/refresh value A=1; clobber B=2; read must retain A'
            vals=(Value('a','input',(4,),'u8'),Value('b','external.constant',(4,),'u8',external_reference=ref),Value('out','external.slice',(width,),'u8',('a',),external_reference=ref))
            inst=(Instance('a','a','scratch',Layout((4,),(1,))),Instance('b','b','scratch',Layout((4,),(1,))),Instance('out','out','output',Layout((width,),(1,))))
            ev=(Event('damage','worker','compute',(),(Region('b',damage[0],damage[1]-damage[0]),),(Demand('pipe'),),('b',)),
                Event('refresh','worker','compute',(),(Region('a',refresh[0],refresh[1]-refresh[0]),),(Demand('pipe'),),('a',)),
                Event('read','worker','copy',(Region('a',lo,width),),(Region('out',0,width),),(Demand('pipe'),),('out',)))
            model=Architecture('byte-oracle',SemanticProgram(vals,('a',),('out',),Contract('byte-oracle','exact',ref,'Four-byte finite declaration')),
                (Storage('GM','GM','global',16),),(Resource('pipe','worker'),),(Allocation('scratch','GM',4),Allocation('output','GM',width)),inst,ev,
                tuple(Dependency(x,y,'program') for x,y in zip(order,order[1:])),platform=Platform('synthetic'))
            memory=bytearray([1]*4);wrong=False
            for event in order:
                if event=='damage':memory[damage[0]:damage[1]]=bytes([2])*(damage[1]-damage[0])
                elif event=='refresh':memory[refresh[0]:refresh[1]]=bytes([1])*(refresh[1]-refresh[0])
                else:wrong=memory[lo:hi]!=bytes([1])*width
            codes=self.codes(model)
            self.assertEqual('clobbered_value' in codes,wrong,(damage,refresh,read,order,codes))
            self.assertFalse(codes-{'clobbered_value'},codes);cases+=1
        self.assertEqual(cases,3300)

    def test_freshness_scheduler_refuses_clobbered_model(self):
        from .schedule import SchedulePlan,Cost,solve,ScheduleError
        for order in (('sort','gather','zero_acc','consume'),('sort','zero_acc','gather','consume')):
            a=phased_alias_example(order)
            p=SchedulePlan(a.fingerprint(),costs=tuple(Cost(e.id,Duration(1,1,Evidence('assumed',note='Synthetic unit costs, not NPU timing'))) for e in a.events))
            if order[1]=='zero_acc':
                with self.assertRaises(ScheduleError):solve(a,p)
            else:self.assertIsNotNone(solve(a,p).certificate)

    def test_examples_have_no_structural_errors(self):
        for name in ('matmul','dense_attention','sparse_attention'):
            with self.subTest(name=name):
                a=example(name);self.assertEqual(self.codes(a),set())
                self.assertEqual(validate(a).status,'conditional')

    def test_platform_identity_and_incomplete_mapping(self):
        platform=Platform('Ascend910B2',(Attribute('cann','9.2.0'),))
        b=replace(self.a,platform=platform)
        changed=replace(b,platform=replace(platform,software=(Attribute('cann','different'),)))
        self.assertNotEqual(b.fingerprint(),changed.fingerprint())
        self.assertIn('binding',self.codes(replace(self.a,bindings=()),'unknown'))
        self.assertIn('event_resources',self.codes(replace(self.a,events=(replace(self.a.events[0],resources=()),)+self.a.events[1:]),'unknown'))
        bad=(Binding('q',('score',),('q',)),)+self.a.bindings[1:]
        self.assertIn('binding',self.codes(replace(self.a,bindings=bad)))

    def test_unknown_does_not_become_hardware_proof(self):
        report=validate(self.a).to_dict()
        self.assertIn('event_duration',self.codes(self.a,'unknown'))
        self.assertIn('no numerical',report['scope'])
        self.assertIn('opaque_semantics',self.codes(example('sparse_attention'),'unknown'))

    def test_example_math_against_independent_dense_and_indexed_reference(self):
        import numpy as np
        from .semantics import PRIM_BY_NAME
        q=np.array([[1.,2.],[-1.,0.5]]);k=np.array([[2.,-1.],[0.,2.]])
        v=np.array([[1.,2.],[-1.,4.]]);indices=np.array([[0],[1]],dtype=np.int64)
        for name in ('matmul','dense_attention','sparse_attention'):
            a=Architecture.from_json(example(name).to_json());out=dict(q=q,k=k,v=v,indices=indices)
            for value in a.semantic.values:
                if value.op=='input':continue
                args=[out[x] for x in value.inputs];attrs={x.name:x.value for x in value.attributes}
                if value.op=='external.sparse_pv':
                    out[value.id]=(args[0][...,None]*args[2][args[1]]).sum(axis=1)
                elif value.op in ('elementwise','reduce','broadcast_row'):
                    kind=attrs.pop('kind');out[value.id]=PRIM_BY_NAME[value.op].ref(kind,*args,**attrs)
                else:out[value.id]=PRIM_BY_NAME[value.op].ref(*args,**attrs)
            if name=='matmul':expected=q@k
            else:
                score=q@k.T
                if name=='sparse_attention':score=np.take_along_axis(score,indices,axis=-1)
                weights=np.exp(score-score.max(axis=-1,keepdims=True));weights/=weights.sum(axis=-1,keepdims=True)
                expected=(weights[...,None]*v[indices]).sum(axis=1) if name=='sparse_attention' else weights@v
            np.testing.assert_allclose(out['out'],expected,rtol=1e-14,atol=1e-14)

    def test_roundtrip_exact(self):
        for name in ('matmul','dense_attention','sparse_attention'):
            a=example(name);b=Architecture.from_json(a.to_json(pretty=True))
            self.assertEqual(a,b);self.assertEqual(a.fingerprint(),b.fingerprint())

    def test_collection_order_does_not_change_identity(self):
        b=replace(self.a,events=self.a.events[::-1],instances=self.a.instances[::-1],
                  semantic=replace(self.a.semantic,values=self.a.semantic.values[::-1]),dependencies=self.a.dependencies[::-1])
        self.assertEqual(self.a.fingerprint(),b.fingerprint())

    def test_labels_are_not_symbols(self):
        b=replace(self.a,label='different title',events=tuple(replace(x,label='different label') for x in self.a.events),
                  semantic=replace(self.a.semantic,values=tuple(replace(x,label='shown') for x in self.a.semantic.values)))
        self.assertEqual(self.a.fingerprint(),b.fingerprint())

    def test_semantic_changes_change_identity(self):
        changes=[replace(self.a,dispatch_priority=tuple(x.id for x in self.a.events)[::-1]),
                 replace(self.a,allocations=(replace(self.a.allocations[0],size_bytes=128),)+self.a.allocations[1:]),
                 replace(self.a,semantic=replace(self.a.semantic,contract=replace(self.a.semantic.contract,kind='approximate',error_policy='separate error'))),
                 replace(self.a,dependencies=self.a.dependencies[:-1])]
        for b in changes:self.assertNotEqual(self.a.fingerprint(),b.fingerprint())

    def test_immutable_and_strict_runtime_types(self):
        with self.assertRaises(FrozenInstanceError):self.a.id='changed'
        with self.assertRaises(ValueError):Value('x','input',[2,2],'f32')
        with self.assertRaises(ValueError):Duration(lower_ns=True)
        with self.assertRaises(ValueError):Attribute('x',float('nan'))
        with self.assertRaises(ValueError):replace(self.a,schema_version=True)

    def test_json_rejects_unrecognized_or_wrong_fields(self):
        base=json.loads(self.a.to_json())
        for mutation in ('extra','version','unversioned','nested','nan','type','missing'):
            obj=json.loads(json.dumps(base))
            if mutation=='extra':obj['future_semantics']='ignored?'
            if mutation=='version':obj['schema_version']=2
            if mutation=='unversioned':del obj['schema_version']
            if mutation=='nested':obj['events'][0]['hidden_source']='not allowed'
            if mutation=='nan':obj['semantic']['values'][0]['attributes']=[dict(name='x',value=float('nan'))]
            if mutation=='type':obj['events'][0]['duration']['lower_ns']=1.5
            if mutation=='missing':del obj['semantic']
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,TypeError)):
                Architecture.from_json(json.dumps(obj))
        with self.assertRaises(ValueError):Architecture.from_json('{"id":"a","id":"b"}')

    def test_duplicate_or_empty_identity(self):
        self.assertIn('identity',self.codes(replace(self.a,events=self.a.events+(self.a.events[0],))))
        self.assertIn('identity',self.codes(replace(self.a,instances=(replace(self.a.instances[0],id=''),)+self.a.instances[1:])))

    def test_unknown_semantic_parent_and_bad_shape(self):
        q=replace(self.a.semantic.values[0],inputs=('missing',))
        self.assertIn('semantic_reference',self.codes(replace(self.a,semantic=replace(self.a.semantic,values=(q,)+self.a.semantic.values[1:]))))
        score=next(v for v in self.a.semantic.values if v.id=='score')
        values=tuple(replace(v,shape=(3,2)) if v==score else v for v in self.a.semantic.values)
        self.assertIn('semantic_shape',self.codes(replace(self.a,semantic=replace(self.a.semantic,values=values))))

    def test_unknown_primitive_needs_reference(self):
        vals=tuple(replace(v,op='new_algorithm') if v.id=='out' else v for v in self.a.semantic.values)
        self.assertIn('semantic_operation',self.codes(replace(self.a,semantic=replace(self.a.semantic,values=vals))))

    def test_math_cycle(self):
        vals=tuple(replace(v,inputs=('out',),op='elementwise',attributes=(Attribute('kind','exp'),)) if v.id=='score' else v for v in self.a.semantic.values)
        self.assertIn('semantic_cycle',self.codes(replace(self.a,semantic=replace(self.a.semantic,values=vals))))

    def test_invalid_contract_and_interface(self):
        b=replace(self.a.semantic,contract=replace(self.a.semantic.contract,kind='approximate'))
        self.assertIn('contract',self.codes(replace(self.a,semantic=b)))
        self.assertIn('semantic_interface',self.codes(replace(self.a,semantic=replace(self.a.semantic,inputs=('q','q')))))

    def test_layout_and_access_bounds(self):
        inst=replace(self.a.instances[0],offset_bytes=1)
        self.assertIn('instance_bounds',self.codes(replace(self.a,instances=(inst,)+self.a.instances[1:])))
        ev=replace(self.a.events[0],reads=(Region('q',0,100),))
        self.assertIn('access_bounds',self.codes(replace(self.a,events=(ev,)+self.a.events[1:])))
        inst=replace(self.a.instances[0],layout=Layout((2,2),(4,)),logical_origin=(0,0))
        self.assertIn('layout',self.codes(replace(self.a,instances=(inst,)+self.a.instances[1:])))

    def test_logical_tile_boundaries(self):
        inst=replace(self.a.instances[0],logical_origin=(1,0))
        self.assertIn('logical_tile',self.codes(replace(self.a,instances=(inst,)+self.a.instances[1:])))
        inst=replace(self.a.instances[0],layout=Layout((1,2),(8,4)))
        self.assertIn('logical_tile',self.codes(replace(self.a,instances=(inst,)+self.a.instances[1:]),'unknown'))

    def test_noninjective_layout_is_not_certified(self):
        inst=replace(self.a.instances[0],layout=Layout((2,2),(0,4)))
        self.assertIn('layout_injectivity',self.codes(replace(self.a,instances=(inst,)+self.a.instances[1:]),'unknown'))

    def test_persistent_capacity_and_unresolved_lifetime(self):
        st=replace(self.a.storages[0],capacity_bytes=1)
        self.assertIn('persistent_capacity',self.codes(replace(self.a,storages=(st,))))
        alloc=replace(self.a.allocations[0],persistent=False)
        self.assertIn('allocation_lifetime',self.codes(replace(self.a,allocations=(alloc,)+self.a.allocations[1:]),'unknown'))
        huge=replace(alloc,size_bytes=8192)
        self.assertIn('allocation_capacity',self.codes(replace(self.a,allocations=(huge,)+self.a.allocations[1:])))

    def test_resource_demands_and_ownership(self):
        ev=replace(self.a.events[0],resources=(Demand('cube',2),))
        self.assertIn('event_resource',self.codes(replace(self.a,events=(ev,)+self.a.events[1:])))
        ev=replace(self.a.events[0],actor='someone_else')
        self.assertIn('event_resource',self.codes(replace(self.a,events=(ev,)+self.a.events[1:])))
        st=replace(self.a.storages[0],kind='UB',owner='aic')
        self.assertIn('private_access',self.codes(replace(self.a,storages=(st,))))

    def test_broken_dependencies_and_cycle(self):
        self.assertIn('event_dependency',self.codes(replace(self.a,dependencies=self.a.dependencies+(Dependency('missing','out','data'),))))
        self.assertIn('event_cycle',self.codes(replace(self.a,dependencies=self.a.dependencies+(Dependency('out','score','program'),))))

    def test_recurrence_is_not_silently_discarded(self):
        dep=Dependency('out','score','reuse',1)
        b=replace(self.a,dependencies=self.a.dependencies+(dep,))
        self.assertIn('recurrence_boundary',self.codes(b,'unknown'))
        self.assertIn(dep,Architecture.from_json(b.to_json()).dependencies)

    def test_missing_producer_and_alias(self):
        b=replace(self.a,dependencies=tuple(d for d in self.a.dependencies if d.target!='max'))
        self.assertIn('missing_producer',self.codes(b));self.assertIn('unordered_alias',self.codes(b))
        a=example('matmul');ev=a.events[0]
        # Two independent writers to the same physical allocation must be ordered.
        extra=replace(ev,id='second')
        self.assertIn('unordered_alias',self.codes(replace(a,events=(ev,extra))))

    def test_disjoint_regions_and_read_read_do_not_conflict(self):
        a=example('matmul');ev=a.events[0]
        allocs=tuple(replace(x,size_bytes=32) if x.id=='out' else x for x in a.allocations)
        insts=tuple(replace(x,layout=Layout((2,2),(8,4))) if x.id=='out' else x for x in a.instances)
        # Same instance, disjoint writes: each covers eight bytes.
        left=replace(ev,writes=(Region('out',0,8),));right=replace(ev,id='second',writes=(Region('out',8,8),))
        b=replace(a,allocations=allocs,instances=insts,events=(left,right))
        self.assertNotIn('unordered_alias',self.codes(b))

    def test_mapping_and_output_must_exist(self):
        self.assertIn('binding',self.codes(replace(self.a,bindings=(Binding('q',('missing',),('k',)),))))
        self.assertIn('output_producer',self.codes(replace(self.a,events=self.a.events[:-1])))

    def test_cost_and_evidence_cannot_be_fabricated(self):
        ev=replace(self.a.events[0],duration=Duration(10,5))
        self.assertIn('duration',self.codes(replace(self.a,events=(ev,)+self.a.events[1:])))
        ev=replace(self.a.events[0],duration=Duration(1,2,Evidence('measured')))
        self.assertIn('evidence_reference',self.codes(replace(self.a,events=(ev,)+self.a.events[1:])))
        ev=replace(self.a.events[0],duration=Duration(1,2,Evidence('measured','claim.json','synthetic')))
        self.assertIn('evidence_not_verified',self.codes(replace(self.a,events=(ev,)+self.a.events[1:]),'unknown'))

    def test_dispatch_priority_is_not_a_dependency(self):
        b=replace(self.a,dispatch_priority=tuple(x.id for x in self.a.events)[::-1])
        self.assertEqual(self.codes(b),set())
        self.assertEqual(b.dependencies,self.a.dependencies)
        self.assertIn('dispatch_priority',self.codes(replace(self.a,dispatch_priority=('out',))))

    def test_legacy_snapshot_has_independent_input_symbols(self):
        from .graph import leaf,Program,MathNode
        from .architecture import semantic_from_legacy
        q,k=leaf('q',(2,2)),leaf('k',(2,2));out=MathNode('matmul',(q,k),{},(2,2),name='out')
        p=Program();p.add(out);snapshot=semantic_from_legacy(p,(out,),self.a.semantic.contract)
        self.assertEqual(len(snapshot.inputs),2)
        self.assertEqual(snapshot.values[-1].inputs,(q.id(),k.id()))
        out.attrs['transa']=True
        self.assertEqual(snapshot.values[-1].attributes,())
        with self.assertRaises(ValueError):semantic_from_legacy(p,(leaf('missing',(2,2)),),self.a.semantic.contract)

    def test_zero_sized_output_needs_no_fake_write(self):
        a=example('matmul');values=tuple(replace(v,shape=(0,2)) if v.id in ('q','out') else v for v in a.semantic.values)
        allocations=tuple(replace(x,size_bytes=0) if x.id in ('q','out') else x for x in a.allocations)
        instances=tuple(replace(x,layout=Layout((0,2),(8,4))) if x.id in ('q','out') else x for x in a.instances)
        event=replace(a.events[0],reads=(Region('q',0,0),Region('k',0,16)),writes=(Region('out',0,0),))
        self.assertEqual(self.codes(replace(a,semantic=replace(a.semantic,values=values),allocations=allocations,instances=instances,events=(event,))),set())

    def test_legacy_input_identity_regression(self):
        from .graph import leaf,Program,MathNode,INPUT
        q=leaf('Q display',(16,16),symbol='q');k=leaf('K display',(16,16),symbol='k')
        p=Program();p.add(q);p.add(k)
        self.assertEqual(len(p.nodes()),2);self.assertNotEqual(q.id(),k.id())
        pretty=leaf('different display',(16,16),symbol='q');self.assertEqual(q.id(),pretty.id())
        unnamed=MathNode(INPUT,shape=(2,2))
        for _ in range(2):
            with self.assertRaises(ValueError):unnamed.id()


def selftest(verbose=True):
    import io,sys
    from .transforms_test import TransformationTests
    from .research_test import ResearchTests
    from .trace_test import TraceTests
    from .schedule_test import ScheduleTests
    from .loops_test import LoopTests
    from .analysis_test import AnalysisTests
    from .policy_test import PolicyTests
    from .runner_test import RunnerTests
    from .ascend_test import AscendTests
    from .ascend_sandbox_test import AscendSandboxTests
    from .lowering_test import LoweringTests
    from .delivery_test import DeliveryTests
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(test)
                             for test in (ArchitectureTests,TransformationTests,ResearchTests,TraceTests,ScheduleTests,LoopTests,AnalysisTests,PolicyTests,RunnerTests,AscendTests,AscendSandboxTests,LoweringTests,DeliveryTests))
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(suite)
    if not verbose and not result.wasSuccessful():
        print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(selftest())
