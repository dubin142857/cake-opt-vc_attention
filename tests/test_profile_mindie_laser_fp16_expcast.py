"""Host orchestration and failure injection; no NPU timing assertions."""
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import profile_mindie_laser_fp16_expcast as profile


def test_defaults_and_no_probe_or_trace_switch():
    args = profile.parser().parse_args([])
    assert args.shape == [1,28,37741,128]
    assert (args.warmup,args.repeats,args.trials)==(5,10,9)
    assert "vscale_pv_prefetch" in args.production_build
    assert "vscale_pv_expcast" in args.expcast_build
    assert "probe" not in repr(vars(args)) and not hasattr(args,"trace")


def test_parent_module_import_does_not_import_torch():
    script = "import sys;sys.path.insert(0,'scripts');import profile_mindie_laser_fp16_expcast;assert 'torch' not in sys.modules;assert 'torch_npu' not in sys.modules"
    subprocess.run([sys.executable,"-c",script],cwd=ROOT,check=True,timeout=30)


@pytest.mark.parametrize("failure_phase",["call","sync"])
def test_first_device_failure_preserved_and_later_access_refused(failure_phase):
    report,actions,snapshots = {},[],[]
    def synchronize():
        actions.append("sync")
        if failure_phase=="sync":
            raise RuntimeError("first507014")
    def launch():
        actions.append("launch")
        if failure_phase=="call":
            raise RuntimeError("first507014")
        return "pending"
    guard = profile.DeviceGuard(report,lambda:snapshots.append(json.dumps(report)),synchronize)
    with pytest.raises(RuntimeError,match="first507014"):
        guard("candidate",launch)
    first = dict(report["first_device_failure"])
    with pytest.raises(RuntimeError,match="Further device access"):
        guard("shouldnotrun",lambda:actions.append("bad"))
    assert actions == (["launch"] if failure_phase=="call" else ["launch","sync"])
    assert report["first_device_failure"] == first
    assert first["further_device_access_aborted"]
    assert snapshots


def test_four_rows_have_shared_fixed_int8_and_fresh_full(monkeypatch):
    import mindie_laser_fp16_trace_worker
    monkeypatch.setattr(mindie_laser_fp16_trace_worker,"native_call",lambda a,p:a.run_native(p))
    class Adapter:
        def __init__(self,name):self.name=name
        def __call__(self,*inputs):return self.name,"full",inputs
        def run_native(self,payload):return self.name,"native",payload
        def run_prepared(self,payload):return self.name,"prepared",payload
    adapters = {name:Adapter(name) for name in profile.NAMES}
    own,shared = object(),object()
    payloads = {profile.ORIGINAL:own,**{name:shared for name in profile.NAMES[1:]}}
    calls = profile.make_calls(adapters,payloads,(1,2,3))
    for name in profile.NAMES:
        assert calls["native"][name]()[2] is payloads[name]
        assert calls["prepared"][name]()[2] is payloads[name]
        assert calls["full"][name]()[2] == (1,2,3)
    payloads[profile.CONTROL] = object()
    with pytest.raises(ValueError,match="identical tensor objects"):
        profile.make_calls(adapters,payloads,(1,2,3))


def test_three_provider_environment_allows_new_controls_one_root(tmp_path,monkeypatch):
    providers = {name:SimpleNamespace(vendor_root=str(tmp_path/name),marker_env="VC_"+name+"_MARKER_FILE")
                 for name in profile.PROVIDERS}
    monkeypatch.setattr(profile.quant.baseline,"worker_environment",lambda base,*args:dict(base))
    env = profile.worker_environment({"VC_OLD_MARKER_FILE":"old","LD_LIBRARY_PATH":"/system"},providers,tmp_path)
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)==[str((tmp_path/name).resolve()) for name in profile.PROVIDERS]
    assert "VC_OLD_MARKER_FILE" not in env
    assert len([name for name in env if name.endswith("MARKER_FILE")])==3
    providers[profile.EXPCAST].vendor_root=providers[profile.STABLE].vendor_root
    with pytest.raises(ValueError,match="distinct"):
        profile.worker_environment({},providers,tmp_path)


def test_schema_rejects_trace_abi_and_accepts_original_eight_arguments():
    def native(arguments,returns):
        return SimpleNamespace(_schema=SimpleNamespace(arguments=[SimpleNamespace(type=x) for x in arguments],
            returns=[SimpleNamespace(type=x) for x in returns]))
    arguments=["Tensor"]*5+["float","int","bool"]
    profile.check_schema(native(arguments,["Tensor"]))
    for args,returns in ((arguments+["int"],["Tensor"]),(arguments,["Tensor","Tensor"])):
        with pytest.raises(RuntimeError,match="schema"):
            profile.check_schema(native(args,returns))


def test_parent_bad_provenance_never_launches_worker(tmp_path,monkeypatch):
    args=profile.parser().parse_args(["--output",str(tmp_path/"report.json")])
    monkeypatch.setattr(profile,"resolve_providers",lambda args:{})
    def failed(*args):raise ValueError("bad source SHA")
    monkeypatch.setattr(profile,"verify_builds",failed)
    monkeypatch.setattr(profile,"launch",lambda *args:pytest.fail("worker must not start"))
    assert profile.parent(args)==1
    report=json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))
    assert report["status"]=="failed" and "bad source SHA" in report["error"]
    assert report["workers"]=={}


def test_pairings_keep_algorithm_and_same_provider_contrasts_separate():
    timing={"rows":{name:dict(trials_ms=[30-i,29-i]) for i,name in enumerate(profile.NAMES)},
            "orders":[list(profile.NAMES),list(reversed(profile.NAMES))]}
    result=profile.comparisons(timing)
    assert set(result["vs_original"])==set(profile.NAMES[1:])
    assert set(result["vs_stable"])=={profile.CONTROL,profile.EXPCAST}
    assert result["feature_toggle"]["baseline"]==profile.CONTROL
    assert result["feature_toggle"]["candidate"]==profile.EXPCAST


def test_original_native_postprocess_preserves_statistics_output_abi():
    seen=[]
    adapter=SimpleNamespace(postprocess=lambda *args:seen.append(args) or args)
    payload=SimpleNamespace(dtype="bf16",shape=(1,28,37741,128))
    profile.postprocess_output(profile.ORIGINAL,adapter,("statistics","raw_fp16"),payload)
    assert seen==[("raw_fp16","bf16",37741,128)]
    profile.postprocess_output(profile.EXPCAST,adapter,"raw_fp32",payload)
    assert seen[-1]==("raw_fp32",payload)
    with pytest.raises(RuntimeError,match="statistics/output"):
        profile.postprocess_output(profile.ORIGINAL,adapter,"bad",payload)


def test_inspection_accepts_exact_three_provider_libraries_and_rejects_mutation(tmp_path,monkeypatch):
    providers,evidence,paths={},{},[]
    for name in profile.PROVIDERS:
        vendor=tmp_path/name
        vendor.mkdir()
        library=vendor/"libcust_opapi.so"
        bridge=tmp_path/(name+"_bridge.so")
        library.write_bytes(name.encode())
        bridge.write_bytes(b"bridge")
        providers[name]=SimpleNamespace(vendor_root=str(vendor),bridge_path=str(bridge))
        evidence[name]=dict(host_library_sha256={str(library):profile.common.sha256_file(library)},
                            bridge_sha256=profile.common.sha256_file(bridge))
        paths.extend([str(library),str(bridge)])
    original_read=Path.read_text
    monkeypatch.setattr(Path,"read_text",lambda p,*a,**kw:"maps" if str(p).replace("\\","/")=="/proc/self/maps" else original_read(p,*a,**kw))
    monkeypatch.setattr(profile.common,"mapped_libraries",lambda text:paths)
    monkeypatch.setattr(profile.quant.baseline,"inspect_batch_invariant",lambda *args:dict(libraries=[],metadata={}))
    import profile_mindie_laser_vector_scale_bulk as common_profile
    seen=[]
    monkeypatch.setattr(common_profile,"verify_provider_paths",lambda path,provider,**kw:seen.append(tuple(provider)) or paths)
    report={}
    profile.inspect_providers(providers,evidence,report,tmp_path,cann_root="CANN",soc="Ascend910_9382")
    assert seen==[profile.PROVIDERS]
    assert set(report["loaded_library_sha256"])==set(paths)
    Path(paths[-1]).write_bytes(b"mutated")
    with pytest.raises(RuntimeError,match="differs from audited manifest"):
        profile.inspect_providers(providers,evidence,report,tmp_path,cann_root="CANN",soc="Ascend910_9382")
