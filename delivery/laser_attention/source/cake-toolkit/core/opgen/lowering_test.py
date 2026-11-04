"""Lowering/source/physical ABI controls; synthetic pointers, no NPU calls."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,tempfile,unittest
from .architecture import Storage,Duration,Evidence,_dump
from .transforms import TransformPlan,FuseEvents,apply_plan
from .transforms_test import fusion_fixture
from .schedule import SchedulePlan,Cost,solve
from .runner import identify,AscendLease
from .lowering import LoweringPlan,LoweringReceipt,TensorBoundary,NativeEntry,TensorObservation,instantiate,validate_run_binding


def fixture(root,separate=False):
    root.mkdir(parents=True,exist_ok=True);source=root/'source.cpp';source.write_text('CPU fixture declaration')
    library=root/'native.so';library.write_text('CPU fixture bytes; never loaded')
    a=fusion_fixture()
    if separate:
        a=replace(a,storages=tuple(Storage(v.id,'GM','global',4) for v in a.allocations),allocations=tuple(replace(v,storage=v.id) for v in a.allocations))
    transform=apply_plan(a,TransformPlan(a.fingerprint(),FuseEvents(('p','m','c'),'native')))
    target=transform.architecture
    schedule=solve(target,SchedulePlan(target.fingerprint(),costs=(Cost('native',Duration(1,3,Evidence('assumed','fixture://cost','CPU only','not hardware'))),)))
    plan=LoweringPlan('fixture',transform,schedule,tuple(TensorBoundary(i.id,i.id,4) for i in target.instances),
        (NativeEntry('native',identify(library,'runtime'),'fixture.native','fixture_kernel'),),(identify(source,'source'),),'CPU synthetic ABI sample only',5)
    offsets={v.allocation:v.offset_bytes for v in schedule.certificate.addresses};allocs={v.id:v for v in target.allocations}
    base={s.id:4096+index*256 for index,s in enumerate(target.storages)}
    observations=[]
    for i in target.instances:
        address=base[allocs[i.allocation].storage];offset=offsets[i.allocation]
        observations.append(TensorObservation(i.id,i.layout.extents,'f32',i.layout.strides_bytes,address+offset,address,64,offset,5))
    return plan,tuple(observations)


class LoweringTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_bound_parent_and_sample_roundtrip(self):
        p,t=fixture(self.root);self.assertTrue(p.verify());receipt=instantiate(p,t,tuple(e.library for e in p.entries),'sample0')
        self.assertEqual(receipt.status,'instantiated_conditionally');self.assertTrue(receipt.verify());self.assertTrue(LoweringReceipt.from_json(receipt.to_json()).verify())
        self.assertEqual(receipt.plan.schedule.implementation.numerical_contract,p.architecture.semantic.contract)

    def test_invalid_layout_device_storage_alignment_and_missing_tensor(self):
        p,t=fixture(self.root)
        for bad in (replace(t[0],shape=(2,)),replace(t[0],dtype='f16'),replace(t[0],strides_bytes=(8,)),
                    replace(t[0],device=0),replace(t[0],device=6),replace(t[0],storage_bytes=1),
                    replace(t[0],storage_offset_bytes=t[0].storage_offset_bytes+1),
                    replace(t[0],data_address=t[0].data_address+1,storage_offset_bytes=t[0].storage_offset_bytes+1)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):instantiate(p,(bad,)+t[1:],tuple(e.library for e in p.entries),'sample')
        for bad in (t[:-1],t+(t[0],)):
            with self.assertRaises(ValueError):instantiate(p,bad,tuple(e.library for e in p.entries),'sample')
        with self.assertRaises(ValueError):instantiate(p,t,tuple(e.library for e in p.entries),'')

    def test_wrong_native_source_parent_or_unbound_instances(self):
        p,t=fixture(self.root)
        with self.assertRaisesRegex(ValueError,'Loaded native'):instantiate(p,t,(),'sample')
        for bad in (replace(p,boundaries=p.boundaries[:-1]),replace(p,entries=()),replace(p,device=0),
                    replace(p,schedule=replace(p.schedule,architecture=p.transform.versions[0]))):
            with self.assertRaises(ValueError):bad.verify()
        Path(p.sources[0].path).write_text('changed')
        with self.assertRaisesRegex(ValueError,'identity changed'):p.verify()

    def test_actual_same_arena_mapping_must_be_consistent(self):
        p,t=fixture(self.root);bad=replace(t[0],data_address=t[0].data_address+256,storage_address=t[0].storage_address+256)
        with self.assertRaisesRegex(ValueError,'arena'):instantiate(p,(bad,)+t[1:],tuple(e.library for e in p.entries),'sample')

    def test_distinct_arenas_cannot_hide_live_physical_alias(self):
        p,t=fixture(self.root,separate=True);bad=replace(t[1],data_address=t[0].data_address,storage_address=t[0].storage_address,storage_offset_bytes=t[0].storage_offset_bytes)
        with self.assertRaisesRegex(ValueError,'overlap'):instantiate(p,(t[0],bad)+t[2:],tuple(e.library for e in p.entries),'sample')

    def test_json_requires_explicit_transform_kind_and_receipt_not_forged(self):
        p,t=fixture(self.root);raw=json.loads(p.to_json());raw['transform']['plan']['operation'].pop('kind')
        with self.assertRaises(ValueError):LoweringPlan.from_json(json.dumps(raw))
        receipt=instantiate(p,t,tuple(e.library for e in p.entries),'sample');bad=replace(receipt,physical_storages=())
        with self.assertRaisesRegex(ValueError,'physical constraint replay'):bad.verify()

    def test_run_uses_real_model_contract_device_and_declared_sources(self):
        p,t=fixture(self.root);file=self.root/'lowering.json';file.write_text(p.to_json());identity=identify(file,'input')
        run=SimpleNamespace(implementation=identity,inputs=(identity,)+p.sources+tuple(e.library for e in p.entries),
            model_fingerprint=p.architecture.fingerprint(),contract=p.architecture.semantic.contract.id,device=AscendLease(identity))
        with patch('core.opgen.ascend.load_profile',return_value=SimpleNamespace(physical_card=5)):
            self.assertEqual(validate_run_binding(run),p)
            for key,value in (('model_fingerprint','0'*64),('contract','other'),('inputs',(identity,))):
                args=run.__dict__.copy();args[key]=value
                with self.assertRaises(ValueError):validate_run_binding(SimpleNamespace(**args))
        with patch('core.opgen.ascend.load_profile',return_value=SimpleNamespace(physical_card=6)):
            with self.assertRaisesRegex(ValueError,'physical device'):validate_run_binding(run)


def selftest(verbose=True):
    import io,sys
    stream=sys.stdout if verbose else io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2 if verbose else 0).run(unittest.defaultTestLoader.loadTestsFromTestCase(LoweringTests))
    if not verbose and not r.wasSuccessful():print(stream.getvalue())
    return 0 if r.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(selftest())
