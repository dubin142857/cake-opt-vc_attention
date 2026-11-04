"""Device binding parser controls; no NPU operations in unit tests."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json,tempfile,unittest
from .architecture import Attribute
from .ascend import (AscendProfile,Occupancy,PROBES,REGEXES,guard_argv,parse_occupancy,
                     load_profile,probe_argv,probe_launcher_argv,native_binding,bootstrap_probe,capture_probe,validate_binding,verify_guard_audit,verify_occupancy)
from .runner import AscendLease,DeviceLease,identify,validate_plan
from .ascend_sandbox import filter_bytes
from .runner_test import fixture,sandbox_path


INFO='''| 5     910B2 | OK | 98 |
| NPU Chip | Process id | Process name | Process memory |
| No running processes found in NPU 5 |
'''
MAPPING='5 0 5 Ascend 910B2\n'


class AscendTests(unittest.TestCase):
    def test_explicit_healthy_empty_process_table_and_mapping(self):
        self.assertTrue(parse_occupancy(INFO,MAPPING,5))
        for info,mapping in ((INFO.replace('OK','Alarm'),MAPPING),(INFO.replace('No running processes found in NPU 5','5 0 | 123 | python | 5'),MAPPING),
                             (INFO.replace('No running processes found in NPU 5','unknown'),MAPPING),(INFO,MAPPING.replace('5 0 5','5 0 0')),
                             (INFO,MAPPING+MAPPING),('query failed',MAPPING),(INFO, '')):
            with self.subTest(info=info,mapping=mapping),self.assertRaises(ValueError):parse_occupancy(info,mapping,5)

    def test_malformed_or_contradictory_rows_never_become_idle(self):
        for row in ('| 5 0 | unknown | python |','| 5 0 | 999 | python |'):
            with self.assertRaises(ValueError):parse_occupancy(INFO+row,MAPPING,5)

    def test_guard_audit_exact_order_final_rc_and_regex(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry=root/'probes.json'
            scripts=('smoke_one.py','card_stability_probe.py','determinism_sweep.py')
            registry.write_text(json.dumps({label:dict(cmd='cd '+str(root)+' && python3 scripts/'+script+' --device {card}',expect_rc=0,expect_regex=rx) for label,script,rx in zip(PROBES,scripts,REGEXES)}))
            p=AscendProfile(5,5,'/usr/bin/python',str(root),str(root),str(root),'9.2.0','Ascend910B2',identify(registry,'harness'),identify(registry,'tool'),(),identify(registry,'harness'),identify(registry,'runtime'))
            audit=dict(physical_card=5,profile_fingerprint=p.fingerprint(),probes=[dict(label=label,argv=list(probe_argv(p,label)),launch_argv=list(probe_launcher_argv(p,'profile.json',label)),native=native_binding(p),returncode=0,stdout=text,stderr='') for label,text in zip(PROBES,('RMSE 0.000000e+00','不稳定 0 次','1 个形状里 0 个不确定'))])
            self.assertTrue(verify_guard_audit(p,audit,'profile.json'))
            for bad in (dict(audit,probes=audit['probes'][:2]),dict(audit,probes=audit['probes'][::-1]),dict(audit,physical_card=0)):
                with self.assertRaises(ValueError):verify_guard_audit(p,bad,'profile.json')
            audit['probes'][1]['returncode']=1
            with self.assertRaises(ValueError):verify_guard_audit(p,audit,'profile.json')

    def test_bootstrap_selects_only_bound_card_before_native_load(self):
        from types import SimpleNamespace
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'__init__.py';package.write_text('fixture');native=root/'libvc_ascend_ops.so';native.write_text('native')
            identity=identify(package,'harness')
            p=AscendProfile(5,5,'python',directory,directory,directory,'9.2.0','Ascend910B2',identity,identity,(),identity,identify(native,'runtime'))
            calls=[];fake_torch=SimpleNamespace(npu=SimpleNamespace(set_device=lambda card:calls.append(('select',card))))
            fake_vc=SimpleNamespace(__file__=str(package),load_native=lambda:calls.append(('load',None)))
            real_read=Path.read_text
            def read(path,*args,**kwargs):
                if str(path)=='/proc/self/maps':return '1000-2000 r-xp 0 00:00 1 '+str(native)+'\n'
                return real_read(path,*args,**kwargs)
            with patch.dict(sys.modules,{'torch':fake_torch,'torch_npu':SimpleNamespace(),'vc_attention_ascend':fake_vc}),patch.object(Path,'read_text',read):
                self.assertEqual(bootstrap_probe(p),native_binding(p));self.assertEqual(calls,[('select',5),('load',None)])
                fake_vc.__file__=str(root/'other.py')
                with self.assertRaisesRegex(ValueError,'different installed'):bootstrap_probe(p)
                self.assertEqual(len(calls),2)
                with self.assertRaisesRegex(ValueError,'NPU0'):bootstrap_probe(replace(p,physical_card=0,logical_card=0))
                self.assertEqual(len(calls),2)

    def test_probe_timeout_keeps_partial_output_and_cannot_pass(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry=root/'probes.json'
            registry.write_text(json.dumps({PROBES[0]:dict(cmd='cd '+str(root)+' && python3 scripts/smoke_one.py --device {card}',expect_rc=0,expect_regex=REGEXES[0])}))
            identity=identify(registry,'harness');p=AscendProfile(5,5,'python',directory,directory,directory,'9.2.0','Ascend910B2',identity,identity,(),identity,identity)
            with patch('core.opgen.ascend.subprocess.run',side_effect=subprocess.TimeoutExpired(['probe'],180,output=b'partial stdout',stderr=b'partial stderr')):
                row=capture_probe(p,PROBES[0],'profile.json')
            self.assertIsNone(row['returncode']);self.assertTrue(row['timed_out'])
            self.assertEqual((row['stdout'],row['stderr']),('partial stdout','partial stderr'));self.assertIsNone(row['native'])

    def test_actual_cann_version_and_guard_provider_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'opp').mkdir();version=root/'opp/version.info';version.write_text('Version=9.2.0\n')
            package=root/'__init__.py';package.write_text('fixture');native=root/'libvc_ascend_ops.so';native.write_text('native')
            identity=identify(package,'harness');filterpath=root/'sandbox.bpf';filterpath.write_bytes(filter_bytes())
            p=AscendProfile(5,5,'python',directory,directory,directory,'9.2.0','Ascend910B2',identity,identity,(),identity,identify(native,'runtime'),identify(filterpath,'tool'))
            profile=root/'profile.json';profile.write_text(p.to_json());binding=AscendLease(identify(profile,'input'))
            with patch('core.opgen.ascend.freeze_nodes',return_value=()):
                self.assertEqual(load_profile(binding),p)
                version.write_text('Version=8.5.0\n')
                with self.assertRaisesRegex(ValueError,'Actual CANN'):load_profile(binding)
                version.write_text('Version=9.2.0\n');native.write_text('changed provider')
                with self.assertRaisesRegex(ValueError,'identity changed'):load_profile(binding)

    def test_source_queries_must_succeed_and_bind_physical_card(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'x';file.write_text('fixture');identity=identify(file,'tool')
            p=AscendProfile(5,5,'python',directory,directory,directory,'9.2.0','Ascend910B2',identity,identity,(),identity,identity)
            o=Occupancy(5,1,0,INFO,'',0,MAPPING,'');self.assertTrue(verify_occupancy(p,o))
            for bad in (replace(o,physical_card=0),replace(o,info_returncode=1),replace(o,mapping_returncode=1)):
                with self.assertRaises(ValueError):verify_occupancy(p,bad)

    @unittest.skipUnless(sandbox_path().is_file(),'bubblewrap unavailable; binding envelope unknown')
    def test_guard_replacement_visibility_and_missing_input_bindings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=fixture(root);registry=root/'registry';registry.write_text('{}')
            profile=AscendProfile(5,5,p.stages[0].argv[0],directory,directory,directory,'9.2.0','Ascend910B2',identify(registry,'harness'),identify(registry,'tool'),(),identify(registry,'harness'),identify(registry,'runtime'))
            profile=replace(profile,seccomp=identify(registry,'tool'))
            profilefile=root/'profile.json';profilefile.write_text(profile.to_json())
            p=replace(p,device=AscendLease(identify(profilefile,'input')),cache_cpu=False,
                env=(Attribute('ASCEND_HOME_PATH',directory),Attribute('PYTHONPATH',directory)),
                inputs=p.inputs+(identify(registry,'harness'),identify(profilefile,'input')))
            with patch('core.opgen.ascend.load_profile',return_value=profile):
                with self.assertRaisesRegex(ValueError,'guard cannot'):validate_binding(p)
                guard=replace(p.stages[1],argv=guard_argv(profile,str(profilefile),p.stages[1].report),outputs=('guard.audit.json',))
                p=replace(p,stages=p.stages[:1]+(guard,)+p.stages[2:])
                with self.assertRaisesRegex(ValueError,'visibility'):validate_binding(replace(p,env=p.env+(Attribute('ASCEND_RT_VISIBLE_DEVICES','5'),)))
                with self.assertRaisesRegex(ValueError,'not frozen'):validate_binding(p)


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(AscendTests))
    if not verbose and not r.wasSuccessful():print(stream.getvalue())
    return 0 if r.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(selftest())
