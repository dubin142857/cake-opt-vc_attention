"""CPU namespace execution with real subprocess faults; not NPU validation."""
from dataclasses import replace
from pathlib import Path
import json,os,subprocess,sys,tempfile,threading,time,unittest
from .architecture import Attribute
from .runner import (DeviceLease,RunPlan,RunReceipt,Stage,cancel_run,execute,identify,
                     tool_fingerprint,validate_plan)


DRIVER='''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
kind,mode,source=sys.argv[1:4]
print(kind+' start',flush=True)
if kind=='build':
    Path('artifact.bin').write_bytes(Path(source).read_bytes())
    if mode=='prewrite':Path('guard.json').write_text('{}')
    if mode=='symlink':
        Path('artifact.bin').unlink();Path('artifact.bin').symlink_to(source)
if kind=='guard' and mode=='guard_fail':sys.exit(7)
if kind=='guard' and (mode in ('hang','cancel') or mode=='crash_once' and os.environ['CAKE_ATTEMPT']=='0'):
    subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    time.sleep(30)
if kind=='check':
    if mode=='tamper_harness':Path(__file__).write_text('changed')
    Path('checked.bin').write_text(hashlib.sha256(Path('artifact.bin').read_bytes()).hexdigest())
if kind=='measure':
    if mode=='transient' and os.environ['CAKE_ATTEMPT']=='0':sys.exit(9)
    if mode=='tamper_artifact':Path('artifact.bin').write_bytes(b'changed after checking')
    Path('raw.dat').write_text('sample')
passed=not (kind=='check' and mode=='false')
if kind=='verify':passed=Path('checked.bin').read_text()==hashlib.sha256(Path('artifact.bin').read_bytes()).hexdigest()
r=dict(schema_version=1,plan_fingerprint=os.environ['CAKE_PLAN_FP'],build_fingerprint=os.environ['CAKE_BUILD_FP'],kind=kind,passed=passed,nonce=os.environ['CAKE_NONCE'],metrics=[dict(name='core_time',unit='ns',value=12)] if kind=='measure' else [])
if kind=='check':
    if mode=='wrong_plan':r['plan_fingerprint']='0'*64
    if mode=='wrong_build':r['build_fingerprint']='0'*64
    if mode=='stale_nonce':r['nonce']='old challenge'
    if mode=='nan':r['metrics']=[dict(name='bad',unit='ns',value=float('nan'))]
if kind=='check' and mode=='bad_json':Path(kind+'.json').write_text('{bad')
else:Path(kind+'.json').write_text(json.dumps(r))
print(kind+' finished',flush=True)
'''


def sandbox_path():
    return Path(os.environ['CAKE_TEST_BWRAP']) if 'CAKE_TEST_BWRAP' in os.environ else Path(__file__).resolve().parents[3]/'env/e107-tooling/unpack/usr/bin/bwrap'


def fixture(directory,mode='ok',cache=True):
    root=Path(directory);source=root/'source.txt';driver=root/'driver.py';data=root/'input.json';runtime=root/'runtime.txt'
    for p,text in ((source,'deterministic artifact'),(driver,DRIVER),(data,json.dumps(dict(std=5,seed=107,mode=mode))),(runtime,sys.version)):
        assert not p.exists();p.write_text(text)
    files=tuple(identify(p,role) for p,role in ((source,'source'),(driver,'harness'),(data,'input'),(runtime,'runtime'),(sys.executable,'tool')))
    stages=tuple(Stage(kind,kind,(sys.executable,str(driver),kind,mode,str(source)),kind+'.json',outputs,0.3 if kind=='guard' and mode=='hang' else 10)
        for kind,outputs in (('build',('artifact.bin',)),('guard',()),('check',('checked.bin',)),('measure',('raw.dat',)),('verify',())))
    return RunPlan('fixture',str(root/'run'),'a'*64,'fixture bytes and checksum','CPU synthetic failure driver',
        (Attribute('python',sys.version),),files,identify(sandbox_path(),'tool'),tool_fingerprint(),DeviceLease('cpu'),stages,cache_cpu=cache)


@unittest.skipUnless(sandbox_path().is_file(),'private bubblewrap unavailable: CPU execution coverage is unknown')
class RunnerTests(unittest.TestCase):
    def test_complete_chain_and_identity_verified_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory);r=execute(p)
            self.assertEqual(r.state,'verified',r.error)
            self.assertEqual(r.transitions,('planned','built','guards_passed','checked','measured','verified'))
            self.assertEqual([x.kind for x in r.stages],['build','guard','check','measure','verify'])
            self.assertTrue(r.verify());self.assertEqual(RunReceipt.from_json(r.to_json()),r)
            self.assertEqual(len({s.nonce for s in r.stages}),5)
            with self.assertRaises(FileExistsError):execute(p)

    def test_guard_failure_never_checks_or_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            r=execute(fixture(directory,'guard_fail'))
            self.assertEqual(r.state,'failed');self.assertEqual(r.stages[-1].returncode,7)
            self.assertEqual([s.kind for s in r.stages],['build','guard']);self.assertTrue(r.verify())
            self.assertTrue(Path(r.stages[-1].stdout.path).read_text().startswith('guard start'))

    def test_strict_report_errors_fail_before_measurement(self):
        for mode in ('false','bad_json','wrong_plan','wrong_build','nan','stale_nonce'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                r=execute(fixture(directory,mode));self.assertEqual(r.state,'failed')
                self.assertEqual(r.stages[-1].kind,'check');self.assertFalse(any(s.kind=='measure' for s in r.stages))
                self.assertTrue(r.verify())

    def test_harness_is_read_only_and_previous_artifact_cannot_be_replaced(self):
        for mode in ('tamper_harness','tamper_artifact','symlink','prewrite'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                p=fixture(directory,mode);r=execute(p);self.assertEqual(r.state,'failed')
                for item in p.inputs:item.verify()
                self.assertNotEqual(r.transitions[-1],'verified')
                if mode=='tamper_artifact':
                    self.assertIn('Previously verified',r.error)
                    with self.assertRaises(ValueError):execute(p,resume=True)

    def test_resume_reuses_only_opted_in_build_check_and_refreshes_others(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory,'transient');first=execute(p)
            self.assertEqual(first.state,'failed');self.assertEqual(first.stages[-1].kind,'measure')
            second=execute(p,resume=True);self.assertEqual(second.state,'verified',second.error)
            self.assertEqual([s.status for s in second.stages],['reused','passed','reused','passed','passed'])
            self.assertNotEqual(first.stages[1].nonce,second.stages[1].nonce);self.assertTrue(second.verify())
            self.assertTrue(first.verify())

    def test_cache_is_opt_in_and_tampered_snapshot_or_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory,cache=False);first=execute(p);second=execute(p,resume=True)
            self.assertTrue(all(s.status=='passed' for s in second.stages));self.assertTrue(second.verify())
            source=Path(p.inputs[0].path);source.write_text('new source')
            with self.assertRaises(ValueError):execute(p,resume=True)
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory);r=execute(p);snapshot=Path(r.stages[0].products[0].snapshot.path);snapshot.write_text('corrupt snapshot')
            with self.assertRaises(ValueError):r.verify()
            with self.assertRaises(ValueError):execute(p,resume=True)

    def test_input_runtime_tool_and_plan_changes_invalidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory);r=execute(p)
            with self.assertRaises(ValueError):execute(replace(p,contract='changed'),resume=True)
            with self.assertRaises(ValueError):validate_plan(replace(p,tool_fingerprint='0'*64))
            runtime=next(i for i in p.inputs if i.role=='runtime');Path(runtime.path).write_text('runtime changed')
            with self.assertRaises(ValueError):r.verify()

    def test_timeout_kills_owned_namespace_leaves_external_witness(self):
        witness=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'])
        try:
            with tempfile.TemporaryDirectory() as directory:
                r=execute(fixture(directory,'hang'))
                self.assertEqual(r.stages[-1].status,'timeout');self.assertEqual(r.state,'failed')
                self.assertIsNone(witness.poll());self.assertTrue(r.verify())
        finally:witness.terminate();witness.wait()

    def test_cancel_and_active_lease_cannot_be_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory,'cancel');results=[];errors=[]
            def work():
                try:results.append(execute(p))
                except BaseException as exc:errors.append(exc)
            thread=threading.Thread(target=work);thread.start()
            try:
                log=Path(p.output_dir)/'attempt-0/guard/stdout.log';deadline=time.monotonic()+5
                while time.monotonic()<deadline and (not log.exists() or 'guard start' not in log.read_text()):time.sleep(.02)
                self.assertTrue(log.exists())
                with self.assertRaisesRegex(ValueError,'lease is active'):execute(p,resume=True)
                cancel_run(p);thread.join(5);self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
                self.assertEqual(results[0].stages[-1].status,'canceled');self.assertTrue(results[0].verify())
            finally:
                if thread.is_alive():
                    if not (Path(p.output_dir)/'CANCEL').exists():cancel_run(p)
                    thread.join(10)

    def test_bad_order_zero_card_and_unknown_npu_adapter_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory)
            for bad in (replace(p,stages=(p.stages[3],)+p.stages[:3]+p.stages[4:]),replace(p,device=DeviceLease('npu',0)),replace(p,device=DeviceLease('npu',3))):
                with self.assertRaises(ValueError):execute(bad)
            self.assertFalse(Path(p.output_dir).exists())

    def test_early_verified_receipt_and_stale_challenges_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory);r=execute(p)
            with self.assertRaises(ValueError):replace(r,stages=r.stages[:3],transitions=r.transitions[:4]).verify()
            with self.assertRaises(ValueError):replace(r,build_fingerprint='0'*64).verify()
            second=execute(p,resume=True)
            row=replace(second.stages[1],nonce=r.stages[1].nonce)
            with self.assertRaises(ValueError):replace(second,stages=second.stages[:1]+(row,)+second.stages[2:]).verify()

    def test_parent_hard_exit_resumes_from_incomplete_protected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory,'crash_once');plan=Path(directory)/'plan.json';plan.write_text(p.to_json())
            code='from pathlib import Path;from core.opgen.runner import RunPlan,execute;import sys;execute(RunPlan.from_json(Path(sys.argv[1]).read_text()))'
            process=subprocess.Popen([sys.executable,'-c',code,str(plan)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                log=Path(p.output_dir)/'attempt-0/guard/stdout.log';deadline=time.monotonic()+5
                while time.monotonic()<deadline and (not log.exists() or 'guard start' not in log.read_text()):time.sleep(.02)
                self.assertTrue(log.exists())
                process.kill();process.wait(timeout=5)
                checkpoint=RunReceipt.from_json((Path(p.output_dir)/'latest.json').read_text())
                self.assertEqual(checkpoint.state,'incomplete');self.assertEqual(checkpoint.transitions[-1],'built')
                with self.assertRaises(ValueError):checkpoint.verify()
                self.assertTrue(checkpoint.verify(allow_incomplete=True))
                resumed=execute(p,resume=True);self.assertEqual(resumed.state,'verified',resumed.error)
                self.assertEqual(resumed.stages[0].status,'reused');self.assertTrue(resumed.verify())
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stderr.close()

    def test_stale_atomic_temp_does_not_block_resume_or_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory);r=execute(p)
            stale=Path(p.output_dir)/'latest.json.tmp';stale.write_text('interrupted write evidence')
            second=execute(p,resume=True);self.assertEqual(second.state,'verified')
            self.assertEqual(stale.read_text(),'interrupted write evidence');self.assertTrue(second.verify())

    def test_bad_relative_paths_and_duplicate_snapshot_entries_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory)
            for name in ('.','../escape','a/../b','/absolute'):
                bad=replace(p,stages=(replace(p.stages[0],report=name),)+p.stages[1:])
                with self.assertRaises(ValueError):validate_plan(bad)
            r=execute(p);row=replace(r.stages[0],products=r.stages[0].products+r.stages[0].products[:1])
            with self.assertRaises(ValueError):replace(r,stages=(row,)+r.stages[1:]).verify()

    def test_frozen_shell_script_accepts_clean_argument_without_command_templates(self):
        import shlex
        with tempfile.TemporaryDirectory() as directory:
            p=fixture(directory);script=Path(directory)/'build.sh'
            argv=(sys.executable,str(Path(directory)/'driver.py'),'build','ok',str(Path(directory)/'source.txt'))
            script.write_text('test "$1" = --clean || exit 4\n'+'exec '+shlex.join(argv)+'\n')
            shell='/bin/bash';p=replace(p,inputs=p.inputs+(identify(script,'harness'),identify(shell,'tool')),
                stages=(replace(p.stages[0],argv=(shell,str(script),'--clean','--install')),)+p.stages[1:])
            self.assertEqual(execute(p).state,'verified')
            for argv in ((shell,'-c','echo unsafe'),(shell,'/missing/script','--clean')):
                with self.assertRaises(ValueError):validate_plan(replace(p,stages=(replace(p.stages[0],argv=argv),)+p.stages[1:]))

    def test_legacy_decimal_budget_no_false_overflow_but_real_overflow_retained(self):
        from .campaign import Campaign,ModelRef,Decision,BudgetLimit,BudgetUse,validate_campaign
        from .architecture import Evidence
        c=Campaign('budget','test',(ModelRef('m','0'*64,'fixture://m'),),(),(Decision('d','probe',('m',),(),(),'milestone','probe',('exit',)),),(),(),(),
            (BudgetLimit('b','cpu','steps',.3),),(BudgetUse('a','b','d',.1,'steps',Evidence('assumed')),BudgetUse('b','b','d',.2,'steps',Evidence('assumed'))),'m')
        self.assertFalse(any(x.code=='budget_exceeded' for x in validate_campaign(c).issues))
        bad=replace(c,uses=(c.uses[0],replace(c.uses[1],amount=.21)))
        self.assertTrue(any(x.code=='budget_exceeded' for x in validate_campaign(bad).issues))


def selftest(verbose=True):
    import io
    stream=sys.stdout if verbose else io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests))
    if not verbose and not result.wasSuccessful():print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(selftest())
