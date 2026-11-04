"""Independent tiny time/address oracles and packed functional execution."""
from dataclasses import replace
import itertools
import json
import random
import unittest
from .architecture import (Allocation,Architecture,Attribute,Binding,Contract,Demand,
    Dependency,Duration,Evidence,Event,Instance,Layout,Platform,Region,Resource,SemanticProgram,Storage,Value,example,validate)
from .transforms import BufferRename,TransformPlan,apply_plan
from .transforms_test import pipeline
from .schedule import (Address,Alignment,Cost,Lease,Placement,Reservation,ScheduleError,
    SchedulePlan,ScheduleReceipt,_Budget,check_certificate,pack_memory,solve)


def jobs(durations=(2,2,3),capacity=2):
    values=[];allocations=[];instances=[];events=[];bindings=[]
    for i,duration in enumerate(durations):
        x,y='x'+str(i),'y'+str(i);eid='e'+str(i)
        values.extend((Value(x,'input',(1,),'f32'),Value(y,'elementwise',(1,),'f32',(x,),(Attribute('kind','neg'),))))
        for name in (x,y):
            allocations.append(Allocation(name,'gm',4));instances.append(Instance(name,name,name,Layout((1,),(4,))))
        events.append(Event(eid,'actor'+str(i),'compute',(Region(x,0,4),),(Region(y,0,4),),(Demand('shared'),),(y,),
                            Duration(duration,duration,Evidence('assumed','fixture://job-cost','synthetic'))))
        bindings.extend((Binding(x,(),(x,)),Binding(y,(eid,),(y,))))
    semantic=SemanticProgram(tuple(values),tuple('x'+str(i) for i in range(len(durations))),tuple('y'+str(i) for i in range(len(durations))),
                             Contract('neg','exact','independent unary negation','synthetic finite f32'))
    return Architecture('jobs',semantic,(Storage('gm','GM','global',8*len(durations)),),(Resource('shared','global',capacity),),
                        tuple(allocations),tuple(instances),tuple(events),(),tuple(bindings),platform=Platform('synthetic'))


def chains(capacity=24,temporary=True):
    a=pipeline(capacity=capacity)
    a=replace(a,events=tuple(replace(e,duration=Duration(2,2,Evidence('assumed','fixture://cost','synthetic'))) for e in a.events),
              allocations=tuple(replace(x,persistent=not temporary) if x.id=='scratch' else x for x in a.allocations))
    return apply_plan(a,TransformPlan(a.fingerprint(),BufferRename('t1','scratch1'),cost_policy='assume_unchanged')).architecture


def simple_oracle(a):
    """Enumerate every integer start vector; resource checker uses direct ticks."""
    ids=tuple(e.id for e in a.events);cost={e.id:e.duration.upper_ns for e in a.events}
    horizon=sum(cost.values());optimum=None
    capacities={r.id:r.capacity_units for r in a.resources}
    for choices in itertools.product(range(horizon+1),repeat=len(ids)):
        starts=dict(zip(ids,choices));finish={e:starts[e]+cost[e] for e in ids}
        makespan=max(finish.values(),default=0)
        if makespan>horizon or (optimum is not None and makespan>=optimum):continue
        if any(finish[d.source]>starts[d.target] for d in a.dependencies):continue
        good=True
        for tick in range(makespan):
            for resource,capacity in capacities.items():
                used=sum(d.units for e in a.events for d in e.resources if d.resource==resource and starts[e.id]<=tick<finish[e.id])
                if used>capacity:good=False;break
            if not good:break
        if good:optimum=makespan
    return optimum


def address_oracle(leases,capacity,alignment):
    """Direct product of aligned offsets; no scheduler packing helpers."""
    domains=[range(0,capacity-row[4]+1,alignment) for row in leases]
    for offsets in itertools.product(*domains):
        good=True
        for i,a in enumerate(leases):
            for j,b in enumerate(leases[:i]):
                simultaneous=a[5] or b[5] or (a[2]<b[3] and b[2]<a[3])
                spatial=offsets[i]<offsets[j]+b[4] and offsets[j]<offsets[i]+a[4]
                if simultaneous and spatial:good=False
        if good:return offsets
    return None


def packed_replay(receipt,inputs):
    """Execute unary fixtures at returned physical offsets in shared byte pools."""
    import numpy as np
    a=receipt.architecture;c=receipt.certificate
    memory={s.id:bytearray(s.capacity_bytes) for s in a.storages}
    allocations={x.id:x for x in a.allocations};instances={x.id:x for x in a.instances}
    offsets={x.allocation:x.offset_bytes for x in c.addresses};events={e.id:e for e in a.events};values={v.id:v for v in a.semantic.values}
    def location(instance,offset=0):
        allocation=allocations[instance.allocation]
        return memory[allocation.storage],offsets[allocation.id]+instance.offset_bytes+offset
    for instance in a.instances:
        if instance.value in inputs:
            pool,start=location(instance);pool[start:start+4]=np.asarray([inputs[instance.value]],dtype=np.float32).tobytes()
    for placement in sorted(c.placements,key=lambda x:(x.start_ns,x.event)):
        e=events[placement.event];src,dst=e.reads[0],e.writes[0]
        pool,start=location(instances[src.instance],src.offset_bytes)
        value=np.frombuffer(pool[start:start+4],dtype=np.float32).astype(np.float64)
        kind=next(x.value for x in values[instances[dst.instance].value].attributes if x.name=='kind')
        out=np.exp(value).astype(np.float32) if kind=='exp' else (-value).astype(np.float32)
        pool,start=location(instances[dst.instance],dst.offset_bytes);pool[start:start+4]=out.tobytes()
    output=[]
    for value in a.semantic.outputs:
        instance=next(i for i in a.instances if i.value==value);pool,start=location(instance)
        output.extend(np.frombuffer(pool[start:start+4],dtype=np.float32))
    return np.asarray(output,dtype=np.float32)


class ScheduleTests(unittest.TestCase):
    def plan(self,a,**kwargs):
        return SchedulePlan(a.fingerprint(),alignments=(Alignment('gm',4),),**kwargs)

    def test_buffer_rename_enables_overlap_under_declared_cost_assumption(self):
        a=pipeline();a=replace(a,events=tuple(replace(e,duration=Duration(2,2,Evidence('assumed'))) for e in a.events))
        before=solve(a,self.plan(a,strategy='exact'))
        b=apply_plan(a,TransformPlan(a.fingerprint(),BufferRename('t1','scratch1'),cost_policy='assume_unchanged')).architecture
        after=solve(b,self.plan(b,strategy='exact'))
        self.assertEqual((before.upper_ns,after.upper_ns),(8,6));self.assertEqual(a.semantic,b.semantic)
        self.assertTrue(after.search.optimal_upper_realization);self.assertEqual(after.lower_ns,6)

    def test_finite_contiguous_storage_changes_schedule(self):
        small=chains(20);large=chains(24)
        results=[solve(a,self.plan(a,strategy='exact')) for a in (small,large)]
        self.assertEqual([r.upper_ns for r in results],[8,6])
        addresses=[{x.allocation:x.offset_bytes for x in r.certificate.addresses} for r in results]
        self.assertEqual(addresses[0]['scratch'],addresses[0]['scratch1'])
        self.assertNotEqual(addresses[1]['scratch'],addresses[1]['scratch1'])
        for result in results:self.assertTrue(check_certificate(result.architecture,result.plan,result.certificate))

    def test_returned_physical_layout_preserves_functional_bits(self):
        import numpy as np
        results=[solve(a,self.plan(a,strategy='exact')) for a in (chains(20),chains(24))]
        generator=np.random.default_rng(103)
        for x0,x1 in generator.normal(0,5,(8,2)):
            expected=-np.exp(np.asarray([x0,x1],dtype=np.float32).astype(np.float64)).astype(np.float32)
            for result in results:
                got=packed_replay(result,dict(x0=x0,x1=x1))
                self.assertTrue(np.array_equal(got.view(np.uint32),expected.view(np.uint32)))

    def test_global_resource_sharing_and_profiled_latency_are_distinct(self):
        a=jobs((4,4),1)
        self.assertFalse(any(x.severity=='error' for x in validate(a).issues))
        ordinary=solve(a,self.plan(a,strategy='exact'))
        windows=tuple(Reservation(e.id,'shared',0,1) for e in a.events)
        profiled=solve(a,self.plan(a,strategy='exact',reservations=windows))
        self.assertEqual((ordinary.upper_ns,profiled.upper_ns),(8,5))
        self.assertEqual([p.finish_ns-p.start_ns for p in profiled.certificate.placements],[4,4])
        private=replace(a,resources=(replace(a.resources[0],owner='actor0'),))
        with self.assertRaises(ScheduleError):solve(private,self.plan(private))

    def test_reservation_tail_is_included_after_event_output_is_ready(self):
        a=jobs((4,),1)
        result=solve(a,self.plan(a,strategy='exact',reservations=(Reservation('e0','shared',5,1),)))
        self.assertEqual(result.certificate.placements[0].finish_ns,4)
        self.assertEqual((result.lower_ns,result.upper_ns),(6,6))

    def test_exact_time_solver_matches_independent_oracle(self):
        randomizer=random.Random(103)
        cases=[((2,2,3),2)]+[(tuple(randomizer.randint(1,3) for _ in range(3)),randomizer.randint(1,2)) for _ in range(12)]
        for durations,capacity in cases:
            a=jobs(durations,capacity)
            edges=tuple(Dependency('e'+str(i),'e'+str(j),'program') for i in range(3) for j in range(i+1,3) if randomizer.random()<.25)
            a=replace(a,dependencies=edges)
            expected=simple_oracle(a);result=solve(a,self.plan(a,strategy='exact'))
            with self.subTest(durations=durations,capacity=capacity,edges=edges):
                self.assertTrue(result.search.optimal_upper_realization,result.search)
                self.assertEqual(result.upper_ns,expected);self.assertLessEqual(result.lower_ns,expected)
                self.assertTrue(check_certificate(a,result.plan,result.certificate))

    def test_list_is_a_witness_not_automatically_optimal(self):
        a=jobs((2,2,3),2)
        greedy=solve(a,self.plan(a));exact=solve(a,self.plan(a,strategy='exact'))
        self.assertEqual((greedy.upper_ns,exact.upper_ns),(5,4))
        self.assertFalse(greedy.search.optimal_upper_realization)

    def test_interval_upper_optimum_does_not_raise_universal_lower_bound(self):
        a=jobs((4,),1);plan=self.plan(a,strategy='exact',costs=(Cost('e0',Duration(1,4,Evidence('assumed'))),))
        result=solve(a,plan)
        self.assertEqual((result.lower_ns,result.upper_ns),(1,4));self.assertTrue(result.search.optimal_upper_realization)
        self.assertFalse(any(x.rule=='complete_fixed_cost_integer_search' for x in result.lower_terms))

    def test_address_search_matches_independent_small_capacity_oracle(self):
        randomizer=random.Random(104)
        for trial in range(30):
            cap=randomizer.randint(3,8);alignment=randomizer.randint(1,2)
            leases=[]
            for i in range(3):
                start=randomizer.randint(0,3);end=start+randomizer.randint(1,3)
                leases.append(('a'+str(i),'gm',start,end,randomizer.randint(1,3),randomizer.random()<.15))
            expected=address_oracle(leases,cap,alignment)
            actual=pack_memory(leases,{'gm':cap},{'gm':alignment},exact=True,budget=_Budget(100000))
            with self.subTest(trial=trial,leases=leases,cap=cap,alignment=alignment):self.assertEqual(actual is None,expected is None)

    def test_fragmentation_defeats_first_fit_but_not_exact_layout(self):
        leases=[('a0','gm',1,5,2,False),('a1','gm',1,4,3,False),('a2','gm',4,6,4,False),
                ('a3','gm',3,6,3,False),('a4','gm',0,4,2,False)]
        self.assertIsNone(pack_memory(leases,{'gm':10},{'gm':1},exact=False,budget=_Budget(100000)))
        exact=pack_memory(leases,{'gm':10},{'gm':1},exact=True,budget=_Budget(100000))
        self.assertIsNotNone(exact);self.assertIsNotNone(address_oracle(leases,10,1))

    def test_large_ns_latencies_do_not_make_list_search_walk_every_tick(self):
        a=jobs((1000000,1000000),1)
        result=solve(a,self.plan(a,max_states=1000))
        self.assertEqual(result.upper_ns,2000000)
        self.assertLess(result.search.states,1000)

    def test_large_byte_buffers_do_not_scan_every_possible_offset(self):
        a=jobs((1,1),1)
        a=replace(a,allocations=tuple(replace(x,size_bytes=1000000) for x in a.allocations),
                  storages=(replace(a.storages[0],capacity_bytes=4000000),))
        result=solve(a,self.plan(a,max_states=1000))
        self.assertEqual(result.upper_ns,2)
        self.assertLess(result.search.states,1000)

    def test_pinned_public_io_is_not_reclaimed_as_temporary_scratch(self):
        a=jobs((2,2),1);a=replace(a,allocations=tuple(replace(x,persistent=False) for x in a.allocations))
        result=solve(a,self.plan(a,strategy='exact'))
        self.assertTrue(all(x.pinned and x.start_ns==0 and x.finish_ns==result.upper_ns for x in result.certificate.leases))
        reduced=replace(a,storages=(replace(a.storages[0],capacity_bytes=12),))
        unknown=solve(reduced,self.plan(reduced,strategy='exact',max_states=10000))
        self.assertEqual(unknown.status,'unknown');self.assertIsNone(unknown.upper_ns)

    def test_budget_and_horizon_limits_do_not_prove_infeasibility_or_optimality(self):
        a=jobs((2,2,3),2)
        no_budget=solve(a,self.plan(a,strategy='exact',max_states=0))
        self.assertEqual(no_budget.status,'unknown');self.assertTrue(no_budget.search.exhausted)
        self.assertFalse(no_budget.search.optimal_upper_realization)
        short=solve(a,self.plan(a,strategy='exact',horizon_ns=1))
        self.assertEqual(short.status,'unknown');self.assertFalse(short.search.optimal_upper_realization)

    def test_budget_exhaustion_retains_a_valid_nonoptimal_witness(self):
        a=jobs((2,2,3),2);listed=solve(a,self.plan(a))
        limited=solve(a,self.plan(a,strategy='exact',max_states=listed.search.states+1))
        self.assertEqual(limited.upper_ns,5);self.assertTrue(limited.search.exhausted)
        self.assertFalse(limited.search.optimal_upper_realization)
        self.assertTrue(check_certificate(a,limited.plan,limited.certificate))

    def test_unknown_cost_capacity_and_recurrence_do_not_become_zero(self):
        a=example('matmul');result=solve(a,SchedulePlan(a.fingerprint()))
        self.assertEqual(result.status,'unknown');self.assertIsNone(result.upper_ns)
        a=jobs((1,1),1)
        for changed in (replace(a,storages=(replace(a.storages[0],capacity_bytes=None),)),
                        replace(a,dependencies=(Dependency('e1','e0','reuse',1),))):
            result=solve(changed,self.plan(changed))
            self.assertEqual(result.status,'unknown');self.assertIsNone(result.certificate)

    def test_unused_unknown_storage_does_not_block_used_known_storage(self):
        a=jobs((2,),1);a=replace(a,storages=a.storages+(Storage('unused','UB','actor0',None),))
        result=solve(a,self.plan(a,strategy='exact'))
        self.assertEqual(result.upper_ns,2)

    def test_bad_plans_and_reservation_profiles_rejected(self):
        a=jobs((2,2),1)
        plans=(replace(self.plan(a),source_fingerprint='0'*64),self.plan(a,tick_ns=0),self.plan(a,priority=('e0',)),
               self.plan(a,costs=(Cost('missing',Duration(1,1)),)),self.plan(a,costs=(Cost('e0',Duration(3,1)),)),
               self.plan(a,reservations=(Reservation('e0','missing',0,1),)),
               self.plan(a,reservations=(Reservation('e0','shared',-1,1),)),
               self.plan(a,reservations=(Reservation('e0','shared',0,2),Reservation('e0','shared',0,2))),
               replace(self.plan(a),alignments=(Alignment('missing',4),)),self.plan(a,max_states=-1))
        for plan in plans:
            with self.subTest(plan=plan),self.assertRaises(ScheduleError):solve(a,plan)

    def test_certificate_tampering_is_detected(self):
        a=chains(24);result=solve(a,self.plan(a,strategy='exact'));c=result.certificate
        first=c.placements[0]
        for forged in (replace(c,placements=()),replace(c,makespan_ns=0),
                       replace(c,placements=(replace(first,finish_ns=first.finish_ns+1),)+c.placements[1:]),
                       replace(c,addresses=tuple(replace(x,offset_bytes=0) for x in c.addresses)),
                       replace(c,addresses=(replace(c.addresses[0],offset_bytes=-1),)+c.addresses[1:]),
                       replace(c,resource_uses=()),replace(c,leases=())):
            with self.subTest(forged=forged),self.assertRaises(ScheduleError):check_certificate(a,result.plan,forged)

    def test_receipts_replay_and_strict_nested_json(self):
        a=jobs((2,2,3),2);result=solve(a,self.plan(a,strategy='exact'))
        self.assertTrue(result.verify());restored=ScheduleReceipt.from_json(result.to_json());self.assertEqual(restored,result)
        with self.assertRaises(ValueError):replace(result,lower_ns=999).verify()
        with self.assertRaises(ValueError):replace(result,conditions=()).verify()
        raw=json.loads(result.to_json());del raw['plan']['schema_version']
        with self.assertRaises(ValueError):ScheduleReceipt.from_json(json.dumps(raw))

    def test_implementation_contract_contains_realization_obligations(self):
        a=chains(20);result=solve(a,self.plan(a,strategy='exact'));contract=result.implementation
        self.assertEqual(contract.numerical_contract,a.semantic.contract)
        self.assertEqual(len(contract.actor_steps),len(a.events));self.assertEqual(contract.storage_addresses,result.certificate.addresses)
        self.assertTrue(any(step.prerequisites for step in contract.actor_steps))
        self.assertTrue(any('not timer waits' in text for text in contract.obligations))
        self.assertEqual(len(contract.reuse_protocols),1)
        protocol=contract.reuse_protocols[0]
        self.assertEqual(protocol.released_allocation,'scratch')
        self.assertEqual(protocol.acquired_allocation,'scratch1')
        self.assertEqual(protocol.release_after,('c0','p0'))
        self.assertEqual(protocol.acquire_before,('c1','p1'))
        a=chains(24);wide=solve(a,self.plan(a,strategy='exact'))
        self.assertFalse(wide.implementation.reuse_protocols)

    def test_cli_preserves_conditional_results_and_frozen_plans(self):
        import tempfile,contextlib,io
        from pathlib import Path
        from .__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=jobs((2,2,3),2)
            source=root/'a.json';plan=root/'plan.json';output=root/'receipt.json'
            source.write_text(a.to_json());plan.write_text(self.plan(a,strategy='exact').to_json())
            def run(argv):
                stdout=io.StringIO();stderr=io.StringIO()
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):code=main(['schedule']+argv)
                return code,stdout.getvalue(),stderr.getvalue()
            argv=['--input',str(source),'--plan',str(plan),'--output',str(output)]
            self.assertEqual(run(argv)[0],0);saved=output.read_bytes()
            self.assertEqual(run(argv)[0],2);self.assertEqual(output.read_bytes(),saved)
            self.assertEqual(run(['--verify',str(output)])[0],0)
            self.assertEqual(run(['--verify',str(output),'--strategy','list'])[0],2)
            self.assertEqual(run(['--input',str(source),'--plan',str(plan),'--max-states','1'])[0],2)
            code,text,_=run(['--input',str(source),'--max-states','0'])
            self.assertEqual(code,0);self.assertEqual(json.loads(text)['status'],'unknown')

    def test_declaration_order_does_not_change_schedule(self):
        a=jobs((2,2,3),2);b=replace(a,events=a.events[::-1],allocations=a.allocations[::-1],instances=a.instances[::-1])
        self.assertEqual(a.fingerprint(),b.fingerprint())
        ra=solve(a,self.plan(a,strategy='exact'));rb=solve(b,self.plan(b,strategy='exact'))
        self.assertEqual(ra.certificate,rb.certificate);self.assertEqual(ra.search,rb.search)


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScheduleTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
