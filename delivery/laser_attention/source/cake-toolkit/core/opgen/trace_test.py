"""Source-free trace ingestion counterexamples; tiny fixtures, no device."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
from .architecture import _dump
from .observation import Artifact
from .trace import TracePlan,TraceReceipt,make_plan,ingest


def event(name='work',start=0,duration=1,**fields):
    return dict(ph='X',pid='c0',tid='VEC',name=name,ts=start,dur=duration,**fields)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def plan(self,rows,report=None,**kwargs):
        path=self.root/'trace.json';path.write_text(json.dumps(dict(displayTimeUnit='ns',traceEvents=rows)))
        reportpath=None
        if report is not None:
            reportpath=self.root/'report.json';reportpath.write_text(json.dumps(report))
        return make_plan(path,reportpath,**kwargs)

    def test_us_conversion_ignores_display_unit_and_keeps_fractional_ns(self):
        receipt=ingest(self.plan([event(start=2,duration=.0005)],single_launch='L0'))
        e=receipt.observation.events[0]
        self.assertEqual((e.start_ns,e.end_ns),(2000,2000.5))
        ns=ingest(replace(receipt.plan,timestamp_unit='ns')).observation.events[0]
        self.assertEqual((ns.start_ns,ns.end_ns),(2,2.0005))

    def test_top_level_array_decimal_epoch_preserves_one_nanosecond(self):
        path=self.root/'array.json';origin=1793707615494510000
        path.write_text(json.dumps([event(start='1793707615494510.001',duration='0.001')]))
        plan=make_plan(path,time_encoding='decimal_ns',origin_ns=origin)
        r=ingest(plan);e=r.observation.events[0]
        self.assertEqual((e.start_ns,e.end_ns),(1,2));self.assertIs(type(e.start_ns),int)
        self.assertEqual(e.record,'[0]');self.assertEqual(json.loads(r.accounting.trace_header),{'source_container':'array'})
        self.assertIn('origin_ns='+str(origin),' '.join(r.observation.provenance.notes));self.assertTrue(r.verify())
        self.assertTrue(TraceReceipt.from_json(r.to_json()).verify())
        with self.assertRaisesRegex(ValueError,'No supported'):ingest(replace(plan,time_encoding='number',origin_ns=0))

    def test_decimal_be_sorts_numerically_with_mixed_timestamp_types(self):
        rows=[dict(ph='E',pid=0,tid=1,ts='10'),dict(ph='B',pid=0,tid=1,name='work',ts=9)]
        r=ingest(self.plan(rows,timestamp_unit='ns',time_encoding='decimal_ns'))
        self.assertEqual((r.observation.events[0].start_ns,r.observation.events[0].end_ns),(9,10))
        self.assertFalse(r.accounting.unsupported)

    def test_decimal_invalid_subnanosecond_and_overflow_are_not_rounded(self):
        invalid=[True,'NaN','Infinity','1e30','0.0005','not a timestamp']
        rows=[event(start='1',duration='0.001')]+[event(name='bad',start=x) for x in invalid]+[event(name='negative',start='2',duration='-1')]
        r=ingest(self.plan(rows,time_encoding='decimal_ns'))
        self.assertEqual(len(r.observation.events),1);self.assertEqual(len(r.accounting.unsupported),7)
        indices=[i for group in r.accounting.normalized for i in group.records]+[x.index for x in r.accounting.unsupported]
        self.assertEqual(sorted(indices),list(range(8)))
        with self.assertRaisesRegex(ValueError,'origin requires'):ingest(replace(r.plan,time_encoding='number',origin_ns=1))

    def test_top_level_numeric_array_keeps_original_indices(self):
        path=self.root/'array.json';path.write_text(json.dumps([dict(ph='M',name='meta',pid=1),event(),dict(ph='C',name='counter',ts=2)]))
        r=ingest(make_plan(path));self.assertEqual(r.accounting.raw_records,3)
        self.assertEqual([x.index for x in r.accounting.metadata],[0]);self.assertEqual([x.index for x in r.accounting.unsupported],[2])
        self.assertEqual(r.observation.events[0].record,'[1]')

    def test_metadata_x_and_be_accounting_is_a_partition(self):
        rows=[dict(ph='M',name='process_name',pid='c0',args={'name':'core'}),event(),
              dict(ph='B',pid='c0',tid='CUBE',id='x',name='SET_FLAG',ts=2,args={}),
              dict(ph='E',pid='c0',tid='CUBE',id='x',name='SET_FLAG',ts=3),
              dict(ph='unsupported',name='unknown')]
        receipt=ingest(self.plan(rows,single_launch='L0'))
        a=receipt.accounting
        self.assertEqual((a.raw_records,len(a.metadata),len(a.normalized),len(a.unsupported)),(5,1,2,1))
        indices=[x.index for x in a.metadata]+[i for g in a.normalized for i in g.records]+[x.index for x in a.unsupported]
        self.assertEqual(sorted(indices),list(range(5)));self.assertEqual(len(receipt.observation.events),2)
        self.assertEqual(receipt.observation.coverage.records_seen,3)
        self.assertTrue(any(e.sync_kind=='signal' for e in receipt.observation.events))

    def test_nested_unidentified_be_and_missing_end_name(self):
        rows=[event(),dict(ph='B',pid=0,tid=1,name='outer',ts=2),dict(ph='B',pid=0,tid=1,name='inner',ts=3),
              dict(ph='E',pid=0,tid=1,ts=4),dict(ph='E',pid=0,tid=1,ts=5)]
        receipt=ingest(self.plan(rows))
        self.assertEqual(len(receipt.observation.events),3);self.assertFalse(receipt.accounting.unsupported)
        ranges={e.name:(e.start_ns,e.end_ns) for e in receipt.observation.events}
        self.assertEqual(ranges['outer'],(2000,5000));self.assertEqual(ranges['inner'],(3000,4000))

    def test_identified_be_order_is_independent_of_row_order(self):
        rows=[event(),dict(ph='B',pid='c0',tid='CUBE',id='x',name='SET_FLAG',ts=2),
              dict(ph='E',pid='c0',tid='CUBE',id='x',name='SET_FLAG',ts=3)]
        expected=None
        for seed in range(8):
            shuffled=rows.copy();random.Random(seed).shuffle(shuffled)
            receipt=ingest(self.plan(shuffled))
            intervals=sorted((e.core,e.pipe,e.name,e.start_ns,e.end_ns) for e in receipt.observation.events)
            if expected is None:expected=intervals
            else:self.assertEqual(intervals,expected)
            self.assertFalse(receipt.accounting.unsupported)

    def test_bad_or_ambiguous_intervals_are_explicit_gaps(self):
        rows=[event(),event(name='bad',duration=-1),dict(ph='B',pid='c0',tid='VEC',id=1,name='a',ts=2),
              dict(ph='B',pid='c0',tid='VEC',id=1,name='a',ts=3),
              dict(ph='E',pid='c0',tid='VEC',id=1,name='a',ts=4),
              dict(ph='E',pid='c0',tid='VEC',id=1,name='a',ts=5)]
        receipt=ingest(self.plan(rows));self.assertEqual(len(receipt.observation.events),1)
        self.assertEqual(len(receipt.accounting.unsupported),5)
        self.assertNotEqual(receipt.observation.coverage.addresses,'declared_complete')

    def test_long_gap_and_multiple_launches_are_retained(self):
        rows=[event(name='a',args={'launch_id':'L0'}),event(name='b',start=100,args={'launch_id':'L1'})]
        report={'format':'opgen.report.v1','rows':[{'core':'c0','launch':x,'duration_us':1} for x in ('L0','L1')]}
        receipt=ingest(self.plan(rows,report,relative_tolerance=0))
        self.assertEqual({e.launch for e in receipt.observation.events},{'L0','L1'})
        self.assertEqual(len(receipt.comparisons),2);self.assertTrue(all(c.status=='matched' for c in receipt.comparisons))
        self.assertFalse(receipt.observation.relations)

    def test_missing_launch_is_unknown_even_when_report_and_gap_exist(self):
        receipt=ingest(self.plan([event(),event(start=100)],{'c0':{'duration_us':1}},relative_tolerance=.05))
        self.assertTrue(all(e.launch is None for e in receipt.observation.events))
        self.assertTrue(all(c.status=='unscoped' and c.difference_ns is None for c in receipt.comparisons))

    def test_be_launch_annotation_can_be_carried_by_begin_only(self):
        rows=[event(args={'launch_id':'L0'}),dict(ph='B',pid='c0',tid='VEC',id=2,name='a',ts=2,args={'launch_id':'L0'}),
              dict(ph='E',pid='c0',tid='VEC',id=2,name='a',ts=3)]
        receipt=ingest(self.plan(rows));self.assertEqual(len(receipt.observation.events),2)
        self.assertFalse(receipt.accounting.unsupported);self.assertTrue(all(e.launch=='L0' for e in receipt.observation.events))

    def test_be_known_conflicting_launches_are_not_merged(self):
        rows=[event(),dict(ph='B',pid='c0',tid='VEC',id=1,name='a',ts=2,args={'launch_id':'L0'}),
              dict(ph='E',pid='c0',tid='VEC',id=1,name='a',ts=3,args={'launch_id':'L1'})]
        receipt=ingest(self.plan(rows));self.assertEqual(len(receipt.observation.events),1)
        self.assertEqual(len(receipt.accounting.unsupported),2)

    def test_unrepresentable_positive_duration_is_not_silently_zero(self):
        receipt=ingest(self.plan([event(),event(start=10**25,duration=.0001)]))
        self.assertEqual(len(receipt.observation.events),1)
        self.assertEqual(len(receipt.accounting.unsupported),1)
        self.assertIn('precision',receipt.accounting.unsupported[0].reason)

    def test_program_counter_and_source_paths_are_metadata_not_reads_or_addresses(self):
        kernel=self.root/'kernel.h';kernel.write_text('do not read this source')
        plan=self.plan([event(args={'code':str(kernel),'pc_addr':'0x1234','detail':'XD:X1=0xbeef,'})])
        opened=[];actual_open=Path.open
        def observed_open(path,*args,**kwargs):
            opened.append(str(path.resolve()));return actual_open(path,*args,**kwargs)
        with mock.patch.object(Path,'open',observed_open):receipt=ingest(plan)
        self.assertEqual(opened,[plan.trace.location]);self.assertFalse(receipt.observation.events[0].accesses)
        self.assertEqual(len(receipt.accounting.input_reads),1)
        self.assertTrue(any(x.name=='native.code' for x in receipt.observation.events[0].attributes))

    def test_explicit_accesses_are_scoped_and_unknown_hex_is_not_decoded(self):
        access={'space':'GM','mode':'read','address':32,'span_bytes':16}
        receipt=ingest(self.plan([event(args={'opgen_accesses':[access]})]))
        self.assertEqual(receipt.observation.events[0].accesses[0].address,32)
        self.assertEqual(receipt.observation.coverage.addresses,'partial')

    def test_explicit_flag_candidates_keep_wait_start_before_signal(self):
        signal=dict(action='signal',channel='ready',generation='7',scope='cross_core')
        wait=dict(signal,action='wait')
        rows=[event(name='signal',start=3,duration=1,args={'launch_id':'L0','opgen_sync':signal}),
              event(name='wait',start=1,duration=4,args={'launch_id':'L0','opgen_sync':wait})]
        receipt=ingest(self.plan(rows))
        candidates=[r for r in receipt.observation.relations if r.kind=='synchronization_candidate']
        self.assertEqual(len(candidates),1);self.assertEqual(candidates[0].evidence.kind,'assumed')
        self.assertFalse(hasattr(receipt.observation,'dependencies'))

    def test_ambiguous_flag_generations_are_not_arbitrarily_paired(self):
        annotation=dict(action='signal',channel='ready',generation='7',scope='cross_core')
        rows=[event(name='signal0',args={'opgen_sync':annotation}),event(name='signal1',start=2,args={'opgen_sync':annotation}),
              event(name='wait',start=4,args={'opgen_sync':dict(annotation,action='wait')})]
        receipt=ingest(self.plan(rows))
        self.assertFalse(any(r.kind=='synchronization_candidate' for r in receipt.observation.relations))
        self.assertTrue(receipt.accounting.notes)

    def test_clock_scope_mismatch_is_not_compared(self):
        report={'format':'opgen.report.v1','rows':[{'core':'c0','launch':'L0','clock':'other','duration_us':1}]}
        receipt=ingest(self.plan([event(args={'launch_id':'L0'})],report,relative_tolerance=0))
        self.assertTrue(all(c.status=='clock_mismatch' and c.difference_ns is None for c in receipt.comparisons))

    def test_missing_report_core_and_missing_trace_core_are_visible(self):
        receipt=ingest(self.plan([event()],{'c1':{'duration_us':1}},single_launch='L0'))
        self.assertEqual({c.status for c in receipt.comparisons},{'missing_report','missing_trace'})
        no_report=ingest(self.plan([event()],single_launch='L0'))
        self.assertEqual(no_report.comparisons[0].status,'missing_report')

    def test_reported_zero_does_not_erase_nonzero_error(self):
        receipt=ingest(self.plan([event()],{'c0':{'duration_us':0}},single_launch='L0',relative_tolerance=.05))
        c=receipt.comparisons[0];self.assertEqual(c.status,'mismatch');self.assertEqual(c.difference_ns,1000)
        self.assertIsNone(c.relative_error)
        zero=ingest(self.plan([event(duration=0)],{'c0':{'duration_us':0}},single_launch='L0',relative_tolerance=0))
        self.assertEqual(zero.comparisons[0].status,'matched')

    def test_no_tolerance_means_residual_without_acceptance(self):
        receipt=ingest(self.plan([event()],{'c0':{'duration_us':2}},single_launch='L0'))
        c=receipt.comparisons[0];self.assertEqual(c.status,'compared');self.assertEqual(c.difference_ns,1000)
        self.assertIsNone(c.allowed_error_ns)

    def test_overlapping_visual_layers_do_not_sum_as_core_time(self):
        rows=[event(duration=3),dict(ph='B',pid='c0',tid='VEC',id=1,name='SET_FLAG',ts=1),
              dict(ph='E',pid='c0',tid='VEC',id=1,name='SET_FLAG',ts=2)]
        receipt=ingest(self.plan(rows,{'c0':{'duration_us':3}},single_launch='L0',relative_tolerance=0))
        self.assertEqual(receipt.comparisons[0].observed_span_ns,3000)
        self.assertEqual(receipt.comparisons[0].status,'matched')

    def test_native_flow_endpoints_are_preserved_as_candidate_markers(self):
        rows=[event(),dict(ph='s',pid='c0',tid='CUBE',name='flow',cat='CUBEToFIXP',id='flag0',ts=100),
              dict(ph='t',pid='c0',tid='FIXP',name='flow',cat='CUBEToFIXP',id='flag0',ts=101)]
        receipt=ingest(self.plan(rows,{'c0':{'duration_us':1}},single_launch='L0',relative_tolerance=0))
        self.assertEqual(len(receipt.observation.events),3);self.assertFalse(receipt.accounting.unsupported)
        self.assertEqual(sum(r.kind=='synchronization_candidate' for r in receipt.observation.relations),1)
        self.assertEqual(receipt.comparisons[0].observed_span_ns,1000)
        self.assertEqual(receipt.comparisons[0].status,'matched')
        wider=ingest(replace(receipt.plan,comparison_layers=('X','FLOW')))
        self.assertEqual(wider.comparisons[0].status,'mismatch')

    def test_ambiguous_or_cross_core_flow_is_not_claimed_as_dataflow(self):
        rows=[event(),dict(ph='s',pid='c0',tid='CUBE',name='flow',id=1,ts=1),
              dict(ph='s',pid='c0',tid='CUBE',name='flow',id=1,ts=2),
              dict(ph='f',pid='c0',tid='FIXP',name='flow',id=1,ts=3)]
        receipt=ingest(self.plan(rows));self.assertTrue(receipt.accounting.notes)
        self.assertFalse(any(r.kind=='synchronization_candidate' for r in receipt.observation.relations))
        rows=[rows[0],rows[1],dict(rows[3],pid='c1')]
        receipt=ingest(self.plan(rows));self.assertTrue(receipt.accounting.notes)
        self.assertFalse(any(r.kind=='synchronization_candidate' for r in receipt.observation.relations))

    def test_identity_and_roles_checked_before_foreign_reads(self):
        plan=self.plan([event()]);bad=replace(plan.trace,kind='kernel_source',location=str(self.root/'missing.h'))
        with mock.patch.object(Path,'open',side_effect=AssertionError('must not open')):
            with self.assertRaises(ValueError):ingest(replace(plan,trace=bad))
        with self.assertRaisesRegex(ValueError,'identity changed'):ingest(replace(plan,trace=replace(plan.trace,sha256='0'*64)))
        Path(plan.trace.location).write_text('changed')
        with self.assertRaisesRegex(ValueError,'identity changed'):ingest(plan)

    def test_receipt_replay_detects_tampering_and_nested_versions(self):
        receipt=ingest(self.plan([event()],single_launch='L0'));self.assertTrue(receipt.verify())
        restored=TraceReceipt.from_json(receipt.to_json());self.assertEqual(restored,receipt)
        with self.assertRaises(ValueError):replace(receipt,comparisons=()).verify()
        with self.assertRaises(ValueError):replace(receipt,accounting=replace(receipt.accounting,raw_records=100)).verify()
        for obj in (receipt.plan,receipt):
            raw=json.loads(obj.to_json());raw['schema_version']=2
            with self.assertRaises(ValueError):type(obj).from_json(json.dumps(raw))
        raw=json.loads(receipt.to_json());del raw['observation']['schema_version']
        with self.assertRaises(ValueError):TraceReceipt.from_json(json.dumps(raw))

    def test_cli_freezes_plans_replays_and_does_not_hide_mismatch(self):
        import contextlib,io
        from .__main__ import main
        plan=self.plan([event()],{'c0':{'duration_us':2}},single_launch='L0',relative_tolerance=0)
        planfile=self.root/'plan.json';planfile.write_text(plan.to_json());output=self.root/'receipt.json'
        def run(argv):
            stdout=io.StringIO();stderr=io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                try:code=main(['architecture']+argv)
                except SystemExit as exc:code=exc.code
            return code,stdout.getvalue(),stderr.getvalue()
        argv=['--trace-plan',str(planfile),'--output',str(output)]
        self.assertEqual(run(argv)[0],1)
        saved=output.read_bytes();self.assertEqual(run(argv)[0],2);self.assertEqual(output.read_bytes(),saved)
        self.assertEqual(run(['--verify-trace',str(output)])[0],0)
        self.assertEqual(run(['--verify-trace',str(output),'--relative-tolerance','1'])[0],2)
        self.assertEqual(run(['--trace-plan',str(planfile),'--sim-report',str(self.root/'report.json')])[0],2)
        self.assertEqual(run(['--example','matmul','--single-launch','L0'])[0],2)
        self.assertEqual(run(['--trace',plan.trace.location,'--transform','bad'])[0],2)
        valid=run(['--trace',plan.trace.location,'--single-launch','L0'])
        self.assertEqual(valid[0],0);self.assertEqual(json.loads(valid[1])['comparisons'][0]['status'],'missing_report')

    def test_empty_aggregate_csv_and_duplicate_json_rejected(self):
        plan=self.plan([])
        with self.assertRaisesRegex(ValueError,'No supported execution'):ingest(plan)
        path=self.root/'summary.csv';path.write_text('instr,call_count,cycles\nBAR,16,10\n')
        with self.assertRaises(ValueError):ingest(make_plan(path))
        path=self.root/'duplicate.json';path.write_text('{"traceEvents":[],"traceEvents":[]}')
        with self.assertRaises(ValueError):ingest(make_plan(path))


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(TraceTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
