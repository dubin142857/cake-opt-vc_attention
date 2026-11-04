"""Real CPU execution receipts stress delivery/rollback, never NPU claims."""
from dataclasses import replace
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
from .campaign import Campaign,ModelRef
from .runner import MetricBinding,execute,identify,RunReceipt
from .runner_test import fixture,sandbox_path
from .delivery import DeliveryPlan,DeliveryReceipt,assess


class Experiment:
    def __init__(self,root):
        self.root=Path(root);self.base=fixture(root,cache=False);self.counter=0
        driver=next(x for x in self.base.inputs if x.role=='harness');p=Path(driver.path);s=p.read_text()
        old="metrics=[dict(name='core_time',unit='ns',value=12)] if kind=='measure' else []";assert s.count(old)==1
        p.write_text(s.replace(old,"metrics=[dict(name='candidate',unit='ns',value=int(sys.argv[4])),dict(name='laser',unit='ns',value=10),dict(name='incumbent',unit='ns',value=int(sys.argv[5]))] if kind=='measure' else []"))
        self.base=replace(self.base,inputs=tuple(identify(x.path,x.role) for x in self.base.inputs))
        self.campaign=Campaign('CPU-selection-fixture','CPU failure/selection protocol only',
            (ModelRef('fast','a'*64,'fixture://a'),ModelRef('experiment','b'*64,'fixture://b')),(),(),(),(),(),(),(),'experiment')

    def run(self,value=8,incumbent=9,model='fast',mode='ok',bind=True,workload=None,binding_scope='CPU synthetic core',runtime=None):
        self.counter+=1;name='run'+str(self.counter);fingerprint=next(m.fingerprint for m in self.campaign.models if m.id==model)
        stages=tuple(replace(s,argv=s.argv[:3]+(mode,)+s.argv[4:]+(str(value),str(incumbent))) for s in self.base.stages)
        p=replace(self.base,id=name,output_dir=str(self.root/name),model_fingerprint=fingerprint,stages=stages,
            metric_bindings=(MetricBinding('candidate',fingerprint,'CPU synthetic core'),MetricBinding('incumbent','a'*64,'CPU synthetic core')) if bind else ())
        if workload is not None:p=replace(p,workload=workload)
        if binding_scope!='CPU synthetic core':p=replace(p,metric_bindings=tuple(replace(b,scope=binding_scope) for b in p.metric_bindings))
        if runtime is not None:p=replace(p,inputs=tuple(identify(runtime,'runtime') if i.role=='runtime' else i for i in p.inputs))
        r=execute(p);path=self.root/(name+'.json');path.write_text(r.to_json(pretty=True));return identify(path,'output'),r

    def plan(self,runs,*,campaign=None,**changes):
        c=campaign or self.campaign
        p=DeliveryPlan('fixture',c.fingerprint(),'promote','fast',tuple(runs),'candidate','laser','ns','CPU protocol experiment')
        return replace(p,**changes)

    def promoted(self):
        runs=(self.run()[0],self.run()[0]);receipt=assess(self.campaign,self.plan(runs));path=self.root/'best.json';path.write_text(receipt.to_json(pretty=True));return receipt,identify(path,'output')


@unittest.skipUnless(sandbox_path().is_file(),'private bubblewrap unavailable; delivery execution coverage unknown')
class DeliveryTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.e=Experiment(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_real_verified_runs_promote_without_overwriting_research_head(self):
        receipt,_=self.e.promoted();self.assertEqual(receipt.status,'promoted');self.assertEqual(receipt.after.validated_best,'fast')
        self.assertEqual(receipt.after.working_head,'experiment');self.assertEqual({r.hardware for r in receipt.readings},{'cpu'})
        self.assertTrue(DeliveryReceipt.from_json(receipt.to_json()).verify());self.assertEqual(receipt.plan.scope,'measured_contract_only')

    def test_ranking_reversal_cannot_be_averaged_into_a_win(self):
        runs=(self.e.run(1)[0],self.e.run(11)[0]);r=assess(self.e.campaign,self.e.plan(runs))
        self.assertEqual(r.status,'rejected');self.assertEqual(r.after,self.e.campaign);self.assertTrue(any('reference_not_beaten' in x for x in r.reasons))
        self.assertLess(sum(x.candidate for x in r.readings)/2,10)

    def test_insufficient_duplicate_failed_and_incomplete_runs(self):
        first,_=self.e.run();r=assess(self.e.campaign,self.e.plan((first,)));self.assertIn('insufficient_independent_runs',r.reasons)
        with self.assertRaisesRegex(ValueError,'Duplicate'):assess(self.e.campaign,self.e.plan((first,first)))
        failed,raw=self.e.run(mode='guard_fail');self.assertEqual(raw.state,'failed')
        r=assess(self.e.campaign,self.e.plan((first,failed)));self.assertEqual(r.status,'rejected');self.assertIsNone(r.after.validated_best)
        # A forged terminal label cannot replace the missing stages.
        p=Path(failed.path);p.write_text(replace(raw,state='verified',error='').to_json())
        with self.assertRaises(ValueError):assess(self.e.campaign,self.e.plan((first,identify(p,'output'))))

    def test_metric_model_unit_and_scope_must_be_bound(self):
        runs=(self.e.run()[0],self.e.run()[0])
        for change in (dict(candidate_metric='unmeasured'),dict(unit='ms'),dict(target_model='experiment'),dict(reference_metric='candidate')):
            with self.subTest(change=change),self.assertRaises(ValueError):assess(self.e.campaign,self.e.plan(runs,**change))
        unbound=self.e.run(bind=False)[0]
        with self.assertRaisesRegex(ValueError,'not bound'):assess(self.e.campaign,self.e.plan((unbound,)))

    def test_valid_runs_cannot_mix_measurement_domains_or_source_closures(self):
        first,_=self.e.run();other,_=self.e.run(workload='other input shape')
        with self.assertRaisesRegex(ValueError,'scope differs'):assess(self.e.campaign,self.e.plan((first,other)))
        scoped,_=self.e.run(binding_scope='CPU preparation rather than core')
        with self.assertRaisesRegex(ValueError,'measurement scope differs'):assess(self.e.campaign,self.e.plan((first,scoped)))
        runtime=self.e.root/'other-runtime.txt';runtime.write_text('different runtime identity')
        changed,_=self.e.run(runtime=runtime)
        with self.assertRaisesRegex(ValueError,'scope differs'):assess(self.e.campaign,self.e.plan((first,changed)))
        parent,proof=self.e.promoted();p=self.e.plan((changed,),campaign=parent.after,operation='rollback',reference_metric='',minimum_runs=1,incumbent=proof)
        with self.assertRaisesRegex(ValueError,'implementation/workload differs'):assess(parent.after,p)

    def test_actual_summary_tamper_and_receipt_readings_are_rejected(self):
        receipt,_=self.e.promoted();bad=replace(receipt,readings=(replace(receipt.readings[0],candidate=0),)+receipt.readings[1:])
        with self.assertRaisesRegex(ValueError,'differs'):bad.verify()
        raw=RunReceipt.from_json(Path(receipt.plan.runs[0].path).read_text());stage=next(s for s in raw.stages if s.kind=='measure')
        report=next(p for p in stage.products if p.relative_path=='measure.json');Path(report.snapshot.path).write_text('{}')
        with self.assertRaisesRegex(ValueError,'identity changed'):receipt.verify()

    def test_existing_best_is_protected_by_same_run_incumbent(self):
        parent,proof=self.e.promoted();c=parent.after
        runs=(self.e.run(8,7,model='experiment')[0],self.e.run(8,7,model='experiment')[0])
        p=self.e.plan(runs,campaign=c,target_model='experiment',incumbent=proof,incumbent_metric='incumbent')
        r=assess(c,p);self.assertEqual(r.status,'rejected');self.assertEqual(r.after.validated_best,'fast');self.assertTrue(any('incumbent_not_beaten' in x for x in r.reasons))
        good=(self.e.run(6,7,model='experiment')[0],self.e.run(6,7,model='experiment')[0]);r=assess(c,replace(p,runs=good));self.assertEqual(r.status,'promoted');self.assertEqual(r.after.validated_best,'experiment')

    def test_missing_incumbent_proof_or_model_labels_never_suffice(self):
        parent,proof=self.e.promoted();c=parent.after;runs=(self.e.run(6,7,model='experiment')[0],self.e.run(6,7,model='experiment')[0])
        with self.assertRaisesRegex(ValueError,'Protected incumbent'):assess(c,self.e.plan(runs,campaign=c,target_model='experiment'))
        with self.assertRaisesRegex(ValueError,'distinct same-run'):assess(c,self.e.plan(runs,campaign=c,target_model='experiment',incumbent=proof))

    def test_fresh_rollback_restores_actual_checked_artifact_and_selection(self):
        parent,proof=self.e.promoted();c=replace(parent.after,working_head='experiment')
        failure,_=self.e.run(1,model='experiment',mode='guard_fail');failed=assess(c,self.e.plan((failure,),campaign=c,target_model='experiment',incumbent=proof,incumbent_metric='incumbent'))
        self.assertEqual(failed.after.validated_best,'fast')
        restored,raw=self.e.run();p=self.e.plan((restored,),campaign=c,operation='rollback',minimum_runs=1,reference_metric='',incumbent=proof)
        receipt=assess(c,p);self.assertEqual(receipt.status,'rolled_back');self.assertEqual((receipt.after.working_head,receipt.after.validated_best),('fast','fast'));self.assertTrue(receipt.verify())
        old=RunReceipt.from_json(Path(parent.plan.runs[0].path).read_text())
        before=next(x.snapshot.sha256 for x in old.stages[0].products if x.relative_path=='artifact.bin');after=next(x.snapshot.sha256 for x in raw.stages[0].products if x.relative_path=='artifact.bin');self.assertEqual(before,after)

    def test_rollback_needs_fresh_validated_target_and_source_closure(self):
        parent,proof=self.e.promoted();c=parent.after;wrong,_=self.e.run(model='experiment')
        p=self.e.plan((wrong,),campaign=c,operation='rollback',minimum_runs=1,reference_metric='',incumbent=proof)
        with self.assertRaisesRegex(ValueError,'model differs'):assess(c,p)
        failed,_=self.e.run(mode='guard_fail');r=assess(c,replace(p,runs=(failed,)));self.assertEqual(r.status,'rejected');self.assertEqual(r.after,c)
        bad=replace(parent,status='rejected');path=self.e.root/'not-best.json';path.write_text(bad.to_json())
        with self.assertRaises(ValueError):assess(c,replace(p,incumbent=identify(path,'output')))

    def test_relaxed_reproducibility_and_unreviewed_video_scope_rejected(self):
        first,_=self.e.run()
        for changes in (dict(minimum_runs=1),dict(scope='video_ready'),dict(operation='invented'),dict(minimum_gain_fraction=-1)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):assess(self.e.campaign,self.e.plan((first,),**changes))

    def test_workload_and_current_tool_identity_cannot_drift(self):
        first,_=self.e.run();second,raw=self.e.run();p=Path(second.path)
        # The stage report keys must still match the plan if its workload changes.
        p.write_text(replace(raw,plan=replace(raw.plan,workload='different shape')).to_json())
        with self.assertRaises(ValueError):assess(self.e.campaign,self.e.plan((first,identify(p,'output'))))
        p.write_text(replace(raw,plan=replace(raw.plan,tool_fingerprint='0'*64)).to_json())
        with self.assertRaisesRegex(ValueError,'identity changed'):assess(self.e.campaign,self.e.plan((first,identify(p,'output'))))

    def test_real_cli_json_roundtrip_and_no_overwrite(self):
        runs=(self.e.run()[0],self.e.run()[0]);p=self.e.plan(runs);root=self.e.root
        (root/'campaign.json').write_text(self.e.campaign.to_json());(root/'plan.json').write_text(p.to_json());out=root/'delivery.json'
        args=[sys.executable,'-m','core.opgen','delivery','--campaign',str(root/'campaign.json'),'--plan',str(root/'plan.json'),'--output',str(out)]
        result=subprocess.run(args,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(out.read_text())['status'],'promoted')
        old=out.read_bytes();self.assertEqual(subprocess.run(args,capture_output=True).returncode,2);self.assertEqual(out.read_bytes(),old)
        result=subprocess.run([sys.executable,'-m','core.opgen','delivery','--verify',str(out)],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)


def selftest(verbose=True):
    import io
    stream=sys.stdout if verbose else io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(DeliveryTests))
    if not verbose and not r.wasSuccessful():print(stream.getvalue())
    return 0 if r.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(selftest())
