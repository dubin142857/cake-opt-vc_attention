"""No-device provenance, ABI, provider isolation and first-failure coverage."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import mindie_laser_msopprof_worker as worker


def test_import_does_not_load_torch_or_npu():
    subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0,'scripts'); import mindie_laser_msopprof_worker; "
        "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"],
        cwd=ROOT, check=True, timeout=30)


@pytest.fixture
def host_provider(tmp_path, monkeypatch):
    calls = []
    provider = SimpleNamespace(vendor_root=str(tmp_path/"vendor"), bridge_path=str(tmp_path/"bridge.so"),
        build_manifest=str(tmp_path/"build.json"), source_manifest=str(tmp_path/"source.json"),
        preprocess_build=str(tmp_path/"preprocessor.json"), bridge_manifest=str(tmp_path/"bridge.json"))
    def arguments(args, name):
        calls.append(("arguments", name, vars(args)))
        return provider
    evidence = {"build_manifest":{"cann_root":str(tmp_path/"cann"), "soc":"ascend910_93"},
                "build_manifest_sha256":"verified-build", "source_manifest_sha256":"verified-source",
                "device_binary_sha256":{"kernel.o":"checked-device"}}
    monkeypatch.setattr(worker.quant, "build_arguments", arguments)
    monkeypatch.setattr(worker.quant.baseline, "provenance", lambda arg:dict(evidence))
    def provenance(arg, *, identity, source_validator):
        calls.append(("provenance", identity, source_validator))
        return dict(evidence)
    monkeypatch.setattr(worker.quant, "quant_provenance", provenance)
    import mindie_laser_quant_fused
    monkeypatch.setattr(mindie_laser_quant_fused, "validate_preprocess_manifest",
        lambda path:dict(build_manifest_sha256="verified-preprocessor"))
    return provider, evidence, calls


@pytest.mark.parametrize("variant", worker.VARIANTS)
def test_selected_provider_only_host_evidence(variant, host_provider, tmp_path):
    provider, evidence, calls = host_provider
    info = worker.provider_info(variant, device=2, shape=(1,2,17,64), output=tmp_path/"report.json")
    json.dumps(info)  # Contract with the collection parent is plain JSON.
    assert info["device"] == 2 and info["shape_bnsd"] == [1,2,17,64]
    assert info["native_attention_calls"] == 1
    assert info["expcast"] is (variant == "fp16_expcast")
    assert info["provenance"]["build_manifest_sha256"] == "verified-build"
    assert info["provider"]["vendor_root"] == provider.vendor_root
    assert calls[0][1] == ("baseline" if variant == "original_laser" else "qk_int8_fused")
    if variant == "original_laser":
        assert info["kernel_prefix"] == "VcLaserAttention_"
        assert "preprocessor" not in info["provenance"]
    else:
        spec = calls[1]
        assert spec[0] == "provenance" and callable(spec[2])
        assert info["kernel_prefix"] == spec[1]["op_type"]+"_"
        assert info["provenance"]["preprocessor"]["build_manifest_sha256"] == "verified-preprocessor"
        expected = "pv_prefetch" if variant == "vector_pv_prefetch" else "pv_expcast"
        assert expected in calls[0][2]["quant_build"]


def test_bad_provenance_stops_before_runtime_import(host_provider, tmp_path, monkeypatch):
    def bad(*args, **kwargs):
        raise ValueError("device_binaries SHA256 mismatch")
    monkeypatch.setattr(worker.quant, "quant_provenance", bad)
    args = SimpleNamespace(variant="vector_pv_prefetch", device=0, shape=[1,2,17,64],
                           output=tmp_path/"failed.json")
    assert worker.collect(args) == 1
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed" and "SHA256" in report["error"]
    assert report["native_attention_calls"] == 0
    assert "active_device_operation" not in report


def test_environment_one_vendor_markers_and_visibility(host_provider, tmp_path):
    info = worker.provider_info("vector_pv_prefetch", output=tmp_path/"report.json")
    env = worker.worker_environment({"ASCEND_CUSTOM_OPP_PATH":"/old/vendor",
        "VC_OLD_MARKER_FILE":"old", "LD_PRELOAD":"bad.so"}, info)
    worker.check_environment(info, env)
    assert env["ASCEND_CUSTOM_OPP_PATH"] == info["provider"]["vendor_root"]
    assert env[info["marker_env"]] == info["marker_path"]
    assert "VC_OLD_MARKER_FILE" not in env and "LD_PRELOAD" not in env
    assert env["ACL_OP_COMPILER_CACHE_MODE"] == "disable"
    for changed in ({"ASCEND_RT_VISIBLE_DEVICES":"0"},
                    {"ASCEND_CUSTOM_OPP_PATH":env["ASCEND_CUSTOM_OPP_PATH"]+os.pathsep+"/other"},
                    {info["marker_env"]:"foreign"}, {"ASCEND_HOME_PATH":"/other"}):
        with pytest.raises(ValueError):
            worker.check_environment(info, dict(env, **changed))
    with pytest.raises(ValueError, match="physical"):
        worker.worker_environment({"ASCEND_RT_VISIBLE_DEVICES":"2"}, info)


@pytest.mark.parametrize("when", ["launch", "sync"])
def test_failure_never_calls_device_again(when):
    report, actions, snapshots = {}, [], []
    def operation():
        actions.append("launch")
        if when == "launch": raise RuntimeError("broken")
        return "pending"
    def sync():
        actions.append("sync")
        if when == "sync": raise RuntimeError("broken")
    stages = worker.DeviceStages(report, lambda:snapshots.append(json.dumps(report)), sync)
    with pytest.raises(RuntimeError, match="broken"):
        stages.call("native_attention", operation)
    failure = dict(report["first_device_failure"])
    with pytest.raises(RuntimeError, match="stopped"):
        stages.call("forbidden", lambda:actions.append("bad"))
    assert actions == (["launch"] if when == "launch" else ["launch", "sync"])
    assert report["first_device_failure"] == failure
    assert failure["operation"]["phase"] == ("about_to_call" if when == "launch" else "awaiting_sync")
    assert snapshots


@pytest.mark.parametrize("variant", worker.VARIANTS)
def test_exactly_one_native_abi_call(variant):
    calls, report = [], {}
    payload = SimpleNamespace(tensors=("q","k","v"), scale=.125, shape=(1,2,17,64), pre_tokens=127)
    adapter = SimpleNamespace(native=lambda *args:calls.append(args) or "out",
        run_native=lambda value:calls.append(value) or "out")
    stages = worker.DeviceStages(report, lambda:None, lambda:calls.append("sync"))
    assert worker.native_once(adapter, payload, variant, stages) == "out"
    assert len(calls) == 2 and calls[-1] == "sync"
    if variant == "original_laser":
        assert calls[0] == ("q","k","v",None,None,None,.125,2,"BNSD",1.,127,1,True)
    else:
        assert calls[0] is payload


def test_original_raw_output_is_second_fp32_tensor():
    torch = pytest.importorskip("torch")
    payload = SimpleNamespace(tensors=(torch.empty(1,2,256,128),))
    statistics, output = torch.empty(1,2,256), torch.empty(1,2,256,128)
    assert worker.output_tensor((statistics,output),payload,"original_laser",torch) is output
    with pytest.raises(ValueError, match="dtype"):
        worker.output_tensor((statistics,output.half()),payload,"original_laser",torch)
    with pytest.raises(ValueError, match="statistics"):
        worker.output_tensor(output,payload,"original_laser",torch)


def test_quant_cpu_postprocess_does_not_read_device_scales():
    torch = pytest.importorskip("torch")
    from mindie_laser_quant_backend import MindIEQuantLaser
    payload = SimpleNamespace(shape=(1,2,17,64),dtype=torch.bfloat16,v_quantize=False)
    # No q, v_scale or other NPU payload is required for this CPU-only path.
    adapter = MindIEQuantLaser(lambda *args:pytest.fail("unexpected native call"),v_quantize=False)
    result = worker.cpu_postprocess(torch.randn(1,2,256,128),adapter,payload,"vector_pv_prefetch")
    assert result.device.type == "cpu" and result.dtype == torch.bfloat16
    assert result.shape == payload.shape
    payload.v_quantize = True
    with pytest.raises(ValueError, match="unquantized"):
        worker.cpu_postprocess(torch.zeros(1,2,256,128),adapter,payload,"fp16_expcast")


@pytest.mark.parametrize("options", [{"variant":"probe"}, {"variant":"original_laser","device":-1},
    {"variant":"original_laser","shape":(1,2,17,32)}, {"variant":"original_laser","shape":(1,2,0,64)}])
def test_bad_input_contract_rejected_without_manifests(options):
    with pytest.raises(ValueError):
        worker.provider_info(**options)
