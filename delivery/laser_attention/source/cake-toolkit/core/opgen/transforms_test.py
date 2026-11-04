"""Transformation tests with functional memory replay, not cycle simulation."""
from dataclasses import replace
import itertools
import json
import unittest

from .architecture import (Allocation, Architecture, Attribute, Binding, Claim,
    Contract, Demand, Dependency, Duration, Evidence, Event, Instance, Layout,
    Platform, Region, Resource, SemanticProgram, Storage, Value, example, validate)
from .transforms import (BufferRename, Compose, Reorder, FuseEvents, TransformError,
                         TransformPlan, TransformReceipt, apply_plan)


def pipeline(capacity=64):
    """Two chains share scratch. The reuse edge serializes otherwise independent work."""
    values=(Value('x0','input',(1,),'f32'),Value('x1','input',(1,),'f32'),
            Value('t0','elementwise',(1,),'f32',('x0',),(Attribute('kind','exp'),)),
            Value('o0','elementwise',(1,),'f32',('t0',),(Attribute('kind','neg'),)),
            Value('t1','elementwise',(1,),'f32',('x1',),(Attribute('kind','exp'),)),
            Value('o1','elementwise',(1,),'f32',('t1',),(Attribute('kind','neg'),)))
    semantic=SemanticProgram(values,('x0','x1'),('o0','o1'),Contract('two-chains','exact','CPU unary operations','finite inputs; functional replay only'))
    allocs=tuple(Allocation(x,'gm',4) for x in ('x0','x1','scratch','o0','o1'))
    instances=tuple(Instance(v.id,v.id,'scratch' if v.id in ('t0','t1') else v.id,Layout((1,),(4,))) for v in values)
    cost=Duration(10,10,Evidence('assumed','fixture://service-cost','synthetic','not a hardware observation'))
    events=[];bindings=[Binding(x,(),(x,)) for x in semantic.inputs]
    for eid,source,dest,actor in (('p0','x0','t0','producer'),('c0','t0','o0','consumer'),
                                 ('p1','x1','t1','producer'),('c1','t1','o1','consumer')):
        events.append(Event(eid,actor,'compute',(Region(source,0,4),),(Region(dest,0,4),),
                            (Demand(actor),),(dest,),cost))
        bindings.append(Binding(dest,(eid,),(dest,)))
    return Architecture('two-chains',semantic,(Storage('gm','GM','global',capacity),),
                        (Resource('producer','producer'),Resource('consumer','consumer')),
                        allocs,instances,tuple(events),
                        (Dependency('p0','c0','data'),Dependency('c0','p1','reuse'),Dependency('p1','c1','data')),
                        tuple(bindings),(Claim('old-latency','Old serialized model schedule is 40 ns',
                            Evidence('derived','fixture://base-schedule','synthetic','not measured')),),
                        ('p0','c0','p1','c1'),Platform('synthetic'))


def fusion_fixture():
    values=(Value('x','input',(1,),'f32'),Value('t','elementwise',(1,),'f32',('x',),(Attribute('kind','neg'),)),
        Value('u','elementwise',(1,),'f32',('t',),(Attribute('kind','exp'),)),Value('out','elementwise',(1,),'f32',('u',),(Attribute('kind','neg'),)))
    events=tuple(Event(e,'worker','compute',(Region(src,0,4),),(Region(dst,0,4),),(Demand('stream'),),(dst,),
        Duration(n,n,Evidence('assumed','fixture://cost','CPU test','not hardware'))) for e,src,dst,n in (('p','x','t',10),('m','t','u',20),('c','u','out',30)))
    return Architecture('fusion-chain',SemanticProgram(values,('x',),('out',),Contract('unary','exact','float32 CPU operations','finite scalar inputs')),
        (Storage('gm','GM','global',64),),(Resource('stream','global'),),tuple(Allocation(v.id,'gm',4) for v in values),
        tuple(Instance(v.id,v.id,v.id,Layout((1,),(4,))) for v in values),events,(Dependency('p','m','data'),Dependency('m','c','data')),
        tuple(Binding(v.id,() if v.id=='x' else (next(e.id for e in events if v.id in e.semantic_values),),(v.id,)) for v in values),
        (Claim('old','Old cost',Evidence('derived','fixture://cost','CPU test','not hardware')),),('p','m','c'),Platform('synthetic'))


def legal_orders(architecture):
    for order in itertools.permutations(e.id for e in architecture.events):
        positions={x:i for i,x in enumerate(order)}
        if all(d.iteration_distance==0 and positions[d.source]<positions[d.target] for d in architecture.dependencies):
            yield order


def memory_replay(architecture,order,inputs):
    """Independent functional executor: resolve actual symbolic allocation/byte locations."""
    import numpy as np
    positions={x:i for i,x in enumerate(order)}
    if set(order)!={e.id for e in architecture.events} or any(d.iteration_distance or positions[d.source]>=positions[d.target] for d in architecture.dependencies):
        raise ValueError('Not a legal finite execution order')
    instances={x.id:x for x in architecture.instances};events={x.id:x for x in architecture.events}
    values={x.id:x for x in architecture.semantic.values};memory={x.id:bytearray(x.size_bytes) for x in architecture.allocations}
    for instance in architecture.instances:
        if instance.value in inputs:
            raw=np.asarray(inputs[instance.value],dtype=np.float32).tobytes()
            memory[instance.allocation][instance.offset_bytes:instance.offset_bytes+len(raw)]=raw
    for eid in order:
        event=events[eid];source=event.reads[0];dest=event.writes[0]
        read=instances[source.instance];write=instances[dest.instance]
        start=read.offset_bytes+source.offset_bytes
        value=np.frombuffer(memory[read.allocation][start:start+source.span_bytes],dtype=np.float32).astype(np.float64)
        kind=next(x.value for x in values[write.value].attributes if x.name=='kind')
        if kind=='exp':out=np.exp(value).astype(np.float32)
        elif kind=='neg':out=(-value).astype(np.float32)
        else:raise AssertionError('Unsupported replay primitive')
        start=write.offset_bytes+dest.offset_bytes;memory[write.allocation][start:start+dest.span_bytes]=out.tobytes()
    outputs=[]
    for output in architecture.semantic.outputs:
        instance=next(x for x in architecture.instances if x.value==output)
        outputs.append(np.frombuffer(memory[instance.allocation][instance.offset_bytes:instance.offset_bytes+4],dtype=np.float32).copy())
    return np.concatenate(outputs)


class TransformationTests(unittest.TestCase):
    def setUp(self):
        self.a=pipeline()

    def plan(self,operation,a=None,policy='invalidate'):
        a=self.a if a is None else a
        return TransformPlan(a.fingerprint(),operation,policy)

    def result(self,operation,a=None,policy='invalidate'):
        a=self.a if a is None else a
        return apply_plan(a,self.plan(operation,a,policy))

    def test_fixture_constraints_and_baseline_order(self):
        self.assertFalse(any(x.severity=='error' for x in validate(self.a).issues))
        self.assertEqual(list(legal_orders(self.a)),[('p0','c0','p1','c1')])

    def test_rename_releases_only_pure_reuse(self):
        result=self.result(BufferRename('t1','scratch1'));after=result.architecture
        self.assertEqual(after.semantic,self.a.semantic)
        self.assertEqual([(x.source,x.target,x.kind) for x in result.steps[0].removed_dependencies],[('c0','p1','reuse')])
        self.assertEqual([(x.source,x.target,x.kind) for x in after.dependencies],[('p0','c0','data'),('p1','c1','data')])
        self.assertEqual(sum(x.delta_bytes for x in result.steps[0].allocation_delta),4)
        self.assertEqual(len(list(legal_orders(after))),6)
        self.assertTrue(result.verify())

    def test_every_legal_functional_order_preserves_bits(self):
        import numpy as np
        after=self.result(BufferRename('t1','scratch1')).architecture
        generator=np.random.default_rng(100)
        for x0,x1 in generator.normal(0,5,(8,2)):
            inputs=dict(x0=[x0],x1=[x1]);expected=memory_replay(self.a,('p0','c0','p1','c1'),inputs)
            independent=-np.exp(np.asarray([x0,x1],dtype=np.float32).astype(np.float64)).astype(np.float32)
            self.assertTrue(np.array_equal(expected.view(np.uint32),independent.view(np.uint32)))
            for order in legal_orders(after):
                got=memory_replay(after,order,inputs)
                self.assertTrue(np.array_equal(expected.view(np.uint32),got.view(np.uint32)))

    def test_partial_overwrite_does_not_erase_reaching_raw_bytes(self):
        from .transforms import _accesses,_raw_reaches
        from .architecture import _dag
        # Byte-level fixture: an old writer fills four bytes, then only two are overwritten.
        a=self.a
        events=tuple(replace(e,writes=(Region('t1',0,2),)) if e.id=='p1' else e for e in a.events)
        a=replace(a,events=events);by_id={e.id:e for e in a.events}
        _,ancestors=_dag(by_id,[(d.source,d.target) for d in a.dependencies])
        left,right=by_id['p0'],by_id['c1']
        old=next(x for x in _accesses(a,left) if x[3]);read=next(x for x in _accesses(a,right) if not x[3])
        self.assertTrue(_raw_reaches(a,left,right,old,read,ancestors))
        full=self.a;by_id={e.id:e for e in full.events}
        self.assertFalse(_raw_reaches(full,by_id['p0'],by_id['c1'],old,read,ancestors))
        # A full write without a proven happens-before path is not an overwrite proof.
        unproved=replace(full,dependencies=tuple(d for d in full.dependencies if d.source!='c0'))
        _,ancestors=_dag(by_id,[(d.source,d.target) for d in unproved.dependencies])
        self.assertTrue(_raw_reaches(unproved,by_id['p0'],by_id['c1'],old,read,ancestors))

    def test_reorder_is_not_a_new_execution_dependency(self):
        operation=Reorder(('p0','p1','c0','c1'));result=self.result(operation)
        self.assertEqual(result.architecture.dispatch_priority,operation.priority)
        self.assertEqual(result.architecture.dependencies,self.a.dependencies)
        self.assertEqual(result.steps[0].removed_dependencies,())
        with self.assertRaises(ValueError):memory_replay(result.architecture,operation.priority,dict(x0=[1],x1=[2]))

    def test_compose_tracks_versions_and_is_replayable(self):
        operation=Compose((Reorder(('p0','p1','c0','c1')),Compose((BufferRename('t1','scratch1'),))))
        result=self.result(operation)
        self.assertEqual(len(result.steps),2);self.assertEqual(len(result.versions),3)
        self.assertEqual(result.steps[0].after_fingerprint,result.steps[1].before_fingerprint)
        self.assertEqual(TransformReceipt.from_json(result.to_json()),result)
        self.assertTrue(result.verify());self.assertEqual(result.versions[0],self.a)
        for x in result.versions:self.assertFalse(any(y.severity=='error' for y in validate(x).issues))

    def test_noop_keeps_existing_evidence(self):
        result=self.result(Reorder(self.a.dispatch_priority))
        self.assertEqual(result.architecture,self.a);self.assertEqual(result.steps[0].invalidated,())

    def test_no_implicit_external_input_copy(self):
        with self.assertRaisesRegex(TransformError,'initialization_required'):self.result(BufferRename('x0','new_input'))

    def test_input_copy_with_explicit_producer_can_be_renamed(self):
        a=self.a;external=Instance('external_x1','x1','external_x1',Layout((1,),(4,)))
        init=Event('init','producer','copy',(Region('external_x1',0,4),),(Region('x1',0,4),),(Demand('producer'),),('x1',))
        bindings=tuple(replace(x,events=('init',),instances=('external_x1','x1')) if x.value=='x1' else x for x in a.bindings)
        a=replace(a,allocations=a.allocations+(Allocation('external_x1','gm',4),),instances=a.instances+(external,),
                  events=(init,)+a.events,dependencies=a.dependencies+(Dependency('init','p1','data'),),bindings=bindings,dispatch_priority=())
        result=self.result(BufferRename('x1','new_x1'),a)
        self.assertEqual(result.architecture.semantic,a.semantic)
        self.assertTrue(result.verify())

    def test_raw_alias_even_mislabelled_reuse_is_rejected(self):
        a=replace(self.a,instances=tuple(replace(x,allocation='scratch') if x.id=='x1' else x for x in self.a.instances),
                  dependencies=self.a.dependencies+(Dependency('p0','p1','reuse'),))
        self.assertTrue(any(x.code=='clobbered_value' and x.severity=='error' for x in validate(a).issues))
        with self.assertRaisesRegex(TransformError,'clobbered_value'):self.result(BufferRename('t0','new_t0'),a)

    def test_data_program_and_sync_edges_are_retained(self):
        for kind in ('data','program','synchronization'):
            a=replace(self.a,dependencies=tuple(replace(d,kind=kind) if d.kind=='reuse' else d for d in self.a.dependencies))
            result=self.result(BufferRename('t1','scratch1'),a)
            self.assertEqual(result.architecture.dependencies,a.dependencies)
            self.assertEqual(result.steps[0].removed_dependencies,())

    def test_positive_distance_edge_is_retained_and_pending(self):
        a=replace(self.a,dependencies=self.a.dependencies+(Dependency('c1','p1','reuse',1),))
        result=self.result(BufferRename('t1','scratch1'),a)
        self.assertIn(a.dependencies[-1],result.architecture.dependencies)
        self.assertTrue(any(x.rule=='recurrence' and x.status=='pending' for x in result.steps[0].obligations))

    def test_reuse_edge_with_remaining_conflict_is_kept(self):
        result=self.result(BufferRename('o0','new_o0'))
        self.assertEqual(result.steps[0].removed_dependencies,())
        self.assertTrue(any(x.reason=='Physical conflicts remain' for x in result.steps[0].kept_reuse))
        self.assertTrue(any(x.rule=='external_output_binding' for x in result.steps[0].obligations))

    def test_unproved_reuse_and_semantic_order_are_kept(self):
        a=replace(self.a,dependencies=self.a.dependencies+(Dependency('c0','c1','reuse'),))
        result=self.result(BufferRename('t1','scratch1'),a)
        self.assertIn(a.dependencies[-1],result.architecture.dependencies)
        self.assertTrue(any('No original physical conflict' in x.reason for x in result.steps[0].kept_reuse))

    def test_release_can_be_disabled(self):
        result=self.result(BufferRename('t1','scratch1',False))
        self.assertEqual(result.architecture.dependencies,self.a.dependencies)

    def test_capacity_failure_does_not_mutate_source_or_publish_partial_result(self):
        a=pipeline(capacity=20);fingerprint=a.fingerprint()
        plan=self.plan(Compose((Reorder(('p1','p0','c1','c0')),BufferRename('t1','scratch1'))),a)
        with self.assertRaisesRegex(TransformError,'persistent_capacity'):apply_plan(a,plan)
        self.assertEqual(a.fingerprint(),fingerprint);self.assertEqual(len(a.allocations),5)

    def test_invalid_identifiers_and_priority_are_rejected(self):
        for operation in (BufferRename('missing','fresh'),BufferRename('t1','scratch'),BufferRename('t1',''),
                          Reorder(('p0',)),Reorder(('p0','p0','c0','c1')),Compose(())):
            with self.subTest(operation=operation),self.assertRaises(TransformError):self.result(operation)

    def test_source_and_schema_identity_enforced(self):
        for fingerprint in ('x','0'*64):
            with self.assertRaisesRegex(TransformError,'source_identity'):
                apply_plan(self.a,TransformPlan(fingerprint,Reorder(self.a.dispatch_priority)))
        with self.assertRaisesRegex(TransformError,'schema_version'):
            apply_plan(self.a,replace(self.plan(Reorder(self.a.dispatch_priority)),schema_version=2))

    def test_unknown_costs_stay_unknown(self):
        a=example('matmul');result=self.result(BufferRename('out','new_out'),a,policy='assume_unchanged')
        self.assertIsNone(result.architecture.events[0].duration.upper_ns)
        self.assertNotEqual(result.architecture.events[0].duration.evidence.kind,'measured')

    def test_cost_and_claim_invalidation_preserves_originals(self):
        a=replace(self.a,events=(replace(self.a.events[0],duration=Duration(10,10,Evidence('measured','fixture://before','synthetic'))),)+self.a.events[1:])
        result=self.result(BufferRename('t1','scratch1'),a)
        self.assertEqual(result.versions[0].events[0].duration.evidence.kind,'measured')
        self.assertTrue(all(x.duration.upper_ns is None for x in result.architecture.events))
        self.assertEqual(result.architecture.claims[0].evidence.kind,'unknown')
        self.assertTrue(any(x.subject=='claim:old-latency' for x in result.steps[0].invalidated))
        assumed=self.result(BufferRename('t1','scratch1'),a,policy='assume_unchanged')
        self.assertEqual(assumed.architecture.events[0].duration.lower_ns,10)
        self.assertEqual(assumed.architecture.events[0].duration.evidence.kind,'assumed')
        self.assertEqual(assumed.architecture.claims[0].evidence.kind,'unknown')

    def test_retained_dependency_observation_is_not_new_measurement(self):
        dep=replace(self.a.dependencies[0],evidence=Evidence('measured','fixture://old-protocol','synthetic'))
        a=replace(self.a,dependencies=(dep,)+self.a.dependencies[1:])
        result=self.result(BufferRename('t1','scratch1'),a)
        retained=next(x for x in result.architecture.dependencies if x.kind=='data' and x.source=='p0')
        self.assertEqual(retained.evidence.kind,'assumed');self.assertEqual(result.versions[0].dependencies[0].evidence.kind,'measured')
        self.assertTrue(result.verify())

    def test_receipt_tampering_is_detected(self):
        result=self.result(BufferRename('t1','scratch1'))
        step=result.steps[0]
        forged=(replace(result,steps=(replace(step,obligations=()),)),
                replace(result,steps=(replace(step,allocation_delta=()),)),
                replace(result,versions=(self.a,self.a)),
                replace(result,steps=()))
        for item in forged:
            with self.assertRaises(TransformError):item.verify()

    def test_strict_plan_and_receipt_json(self):
        plan=self.plan(Compose((BufferRename('t1','scratch1'),Reorder(('p0','p1','c0','c1')))))
        self.assertEqual(TransformPlan.from_json(plan.to_json()),plan)
        data=json.loads(plan.to_json())
        for kind in ('missing_kind','unknown_kind','unknown_field','empty','version','policy'):
            raw=json.loads(json.dumps(data))
            if kind=='missing_kind':del raw['operation']['operations'][0]['kind']
            if kind=='unknown_kind':raw['operation']['kind']='magic'
            if kind=='unknown_field':raw['operation']['hidden']=True
            if kind=='empty':raw['operation']['operations']=[]
            if kind=='version':raw['schema_version']=2
            if kind=='policy':raw['cost_policy']='certify'
            with self.subTest(kind=kind),self.assertRaises((TypeError,ValueError)):TransformPlan.from_json(json.dumps(raw))
        with self.assertRaises(ValueError):TransformPlan.from_json('{"schema_version":1,"schema_version":1}')
        result=self.result(BufferRename('t1','scratch1'))
        raw=json.loads(result.to_json());raw['status']='measured_win'
        with self.assertRaises(ValueError):TransformReceipt.from_json(json.dumps(raw))

    def test_receipt_nested_versions_and_operations_are_explicit(self):
        receipt=self.result(BufferRename('t1','scratch1'))
        for target in ('plan','source','result','step'):
            raw=json.loads(receipt.to_json())
            if target=='plan':del raw['plan']['schema_version']
            if target=='source':del raw['versions'][0]['schema_version']
            if target=='result':raw['versions'][1]['schema_version']=2
            if target=='step':del raw['steps'][0]['operation']['kind']
            with self.subTest(target=target),self.assertRaises(ValueError):
                TransformReceipt.from_json(json.dumps(raw))

    def test_cli_applies_and_replays_receipt_without_overwriting(self):
        import contextlib,io,tempfile
        from pathlib import Path
        from .__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'model.json';plan=root/'plan.json';output=root/'receipt.json'
            source.write_text(self.a.to_json())
            plan.write_text(self.plan(Compose((BufferRename('t1','scratch1'),Reorder(('p0','p1','c0','c1'))))).to_json())
            def run(argv):
                stdout=io.StringIO();stderr=io.StringIO()
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):code=main(['architecture']+argv)
                return code,stdout.getvalue(),stderr.getvalue()
            argv=['--input',str(source),'--transform',str(plan),'--output',str(output)]
            self.assertEqual(run(argv)[0],0)
            receipt=output.read_bytes();original=source.read_bytes()
            self.assertTrue(TransformReceipt.from_json(receipt).verify())
            code,text,error=run(['--verify-transform',str(output)])
            self.assertEqual(code,0);self.assertEqual(json.loads(text)['status'],'replayed_in_model')
            self.assertEqual(run(argv)[0],2);self.assertEqual(output.read_bytes(),receipt)
            self.assertEqual(source.read_bytes(),original)
            for argv in (['--example','matmul','--transform',str(plan)],
                         ['--selftest','--transform',str(plan)],
                         ['--verify-transform',str(output),'--transform',str(plan)]):
                with self.subTest(argv=argv):self.assertEqual(run(argv)[0],2)
            forged=json.loads(receipt);forged['steps'][0]['allocation_delta']=[]
            output.write_text(json.dumps(forged))
            self.assertEqual(run(['--verify-transform',str(output)])[0],2)

    def test_fusion_keeps_semantics_boundary_body_and_allocations(self):
        a=fusion_fixture();r=apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('m','c'),'fused')));b=r.architecture
        self.assertTrue(r.verify());self.assertEqual(b.semantic,a.semantic);self.assertEqual(b.allocations,a.allocations);self.assertEqual(b.instances,a.instances)
        e=next(e for e in b.events if e.id=='fused');self.assertEqual(e.reads,(Region('t',0,4),));self.assertEqual(e.writes,(Region('out',0,4),Region('u',0,4)))
        self.assertEqual(b.dependencies,(Dependency('p','fused','data'),));self.assertEqual(b.dispatch_priority,('p','fused'))
        w=r.steps[0].fusion;self.assertEqual([e.id for e in w.members],['m','c']);self.assertEqual(w.internal_dependencies,(Dependency('m','c','data'),))
        self.assertEqual([(x.event,x.region) for x in w.internal_reads],[('c',Region('u',0,4))])
        self.assertTrue(all(d.delta_bytes==0 for d in r.steps[0].allocation_delta))

    def test_fusion_does_not_sum_or_inherit_member_measurements(self):
        a=fusion_fixture();r=apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('m','c'),'fused'),'assume_unchanged'))
        e=next(e for e in r.architecture.events if e.id=='fused')
        self.assertIsNone(e.duration.lower_ns);self.assertIsNone(e.duration.upper_ns);self.assertEqual(e.duration.evidence.kind,'unknown')
        subjects={x.subject for x in r.steps[0].invalidated};self.assertIn('event:m.duration',subjects);self.assertIn('event:c.duration',subjects)
        self.assertEqual(r.architecture.claims[0].evidence.kind,'unknown')
        self.assertTrue(any(o.rule=='fusion_lowering' and o.status=='pending' for o in r.steps[0].obligations))

    def test_fusion_functional_expansion_matches_independent_memory_executor(self):
        import numpy as np
        a=fusion_fixture();r=apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('m','c'),'fused')))
        for value in (-5.,-1.,0.,1.,5.):
            wanted=memory_replay(a,('p','m','c'),{'x':[value]})
            for order in legal_orders(r.architecture):
                expanded=tuple(y for x in order for y in (('m','c') if x=='fused' else (x,)))
                got=memory_replay(a,expanded,{'x':[value]})
                for x,y in zip(got,wanted):self.assertTrue(np.array_equal(x,y))

    def test_fusion_rejects_wrong_members_order_nonconvex_actor_and_recurrence(self):
        a=fusion_fixture()
        for operation,code in ((FuseEvents((),'new'),'fusion_members'),(FuseEvents(('m',),'new'),'fusion_members'),
                (FuseEvents(('m','m'),'new'),'fusion_members'),(FuseEvents(('m','missing'),'new'),'fusion_members'),
                (FuseEvents(('m','c'),'p'),'fusion_identity'),(FuseEvents(('m','c'),''),'fusion_identity'),
                (FuseEvents(('c','m'),'new'),'fusion_order'),(FuseEvents(('p','c'),'new'),'fusion_nonconvex')):
            with self.subTest(operation=operation),self.assertRaises(TransformError) as cm:apply_plan(a,TransformPlan(a.fingerprint(),operation))
            self.assertEqual(cm.exception.code,code)
        other=replace(a,events=tuple(replace(e,actor='other') if e.id=='c' else e for e in a.events))
        with self.assertRaisesRegex(TransformError,'fusion_actor'):apply_plan(other,TransformPlan(other.fingerprint(),FuseEvents(('m','c'),'new')))
        periodic=replace(a,dependencies=a.dependencies+(Dependency('c','m','data',1),))
        with self.assertRaisesRegex(TransformError,'fusion_recurrence'):apply_plan(periodic,TransformPlan(periodic.fingerprint(),FuseEvents(('m','c'),'new')))

    def test_fusion_partial_written_bytes_keep_external_holes(self):
        a=fusion_fixture();values=tuple(replace(v,shape=(4,)) for v in a.semantic.values if v.id!='u')
        values=tuple(replace(v,inputs=('t',)) if v.id=='out' else v for v in values)
        events=(Event('p','worker','compute',(Region('x',0,16),),(Region('t',0,16),),(Demand('stream'),),('t',)),
            Event('a','worker','compute',(Region('x',0,4),),(Region('t',0,4),),(Demand('stream'),),('t',)),
            Event('b','worker','compute',(Region('t',0,16),),(Region('out',0,16),),(Demand('stream'),),('out',)))
        a=replace(a,semantic=replace(a.semantic,values=values),allocations=tuple(Allocation(v.id,'gm',16) for v in values),
            instances=tuple(Instance(v.id,v.id,v.id,Layout((4,),(4,))) for v in values),events=events,
            dependencies=(Dependency('p','a','program'),Dependency('a','b','data')),bindings=(),dispatch_priority=())
        r=apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('a','b'),'ab')));w=r.steps[0].fusion
        self.assertEqual(w.boundary_reads,(Region('t',4,12),Region('x',0,4)))
        self.assertEqual([(x.event,x.region) for x in w.internal_reads],[('b',Region('t',0,4))])
        # Independent per-byte live-in oracle, not interval subtraction.
        defined=set();live=set()
        for e in w.members:
            for region in e.reads:
                for byte in range(region.offset_bytes,region.offset_bytes+region.span_bytes):
                    if (region.instance,byte) not in defined:live.add((region.instance,byte))
            for region in e.writes:defined.update((region.instance,byte) for byte in range(region.offset_bytes,region.offset_bytes+region.span_bytes))
        actual={(region.instance,byte) for region in w.boundary_reads for byte in range(region.offset_bytes,region.offset_bytes+region.span_bytes)}
        self.assertEqual(actual,live)

    def test_fusion_read_before_later_write_stays_live_in(self):
        a=fusion_fixture();events=(a.events[0],replace(a.events[1],reads=(Region('t',0,4),),writes=(Region('out',0,4),),semantic_values=('out',)),
            replace(a.events[2],reads=(Region('x',0,4),),writes=(Region('t',0,4),),semantic_values=('t',)))
        # This fixture checks physical access order; the supplied semantic bindings
        # are deliberately omitted rather than inventing numerical equivalence.
        a=replace(a,events=events,bindings=())
        r=apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('m','c'),'mc')))
        self.assertEqual(r.steps[0].fusion.boundary_reads,(Region('t',0,4),Region('x',0,4)))
        self.assertFalse(r.steps[0].fusion.internal_reads)

    def test_fusion_codec_tamper_replay_and_composition(self):
        a=fusion_fixture();operation=Compose((FuseEvents(('m','c'),'fused'),Reorder(('fused','p'))))
        plan=TransformPlan.from_json(TransformPlan(a.fingerprint(),operation).to_json());r=apply_plan(a,plan)
        self.assertTrue(TransformReceipt.from_json(r.to_json()).verify())
        raw=json.loads(r.to_json());raw['steps'][0]['fusion']['members']=[]
        with self.assertRaisesRegex(TransformError,'deterministic replay'):TransformReceipt.from_json(json.dumps(raw)).verify()
        raw=json.loads(plan.to_json());raw['operation']['operations'][0].pop('kind')
        with self.assertRaises(ValueError):TransformPlan.from_json(json.dumps(raw))

    def test_existing_architecture_examples_support_composed_operations(self):
        for name,instance in (('dense_attention','score'),('sparse_attention','selected'),('matmul','out')):
            a=example(name);operation=Compose((BufferRename(instance,'new_'+instance),Reorder(tuple(e.id for e in a.events)[::-1])))
            result=self.result(operation,a)
            self.assertEqual(result.architecture.semantic,a.semantic);self.assertTrue(result.verify())
            self.assertEqual(validate(result.architecture).status,'conditional')


def selftest(verbose=True):
    import io,sys
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TransformationTests)
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(suite)
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
