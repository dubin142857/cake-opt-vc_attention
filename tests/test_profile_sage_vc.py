"""Profiling contracts: stage coverage/parity, identical core inputs, timing units."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
try:
    import profile_sage_vc as profile
finally:
    sys.path.remove(str(ROOT / "scripts"))


class Recorder:
    def __init__(self):
        self.names = []

    def call(self,name,fn):
        self.names.append(name)
        return fn()


def fake_ops():
    def group(v,clusters,iters):
        assert (clusters,iters) == (8,2)
        return torch.arange(v.shape[2]-1,-1,-1).expand(*v.shape[:2],v.shape[2])

    def qk(q,k,km,**kwargs):
        assert kwargs == dict(BLKQ=64,WARPQ=16,BLKK=128,WARPK=128,tensor_layout="HND")
        shape = q.shape[:2]
        return q.round().to(torch.int8), torch.ones(*shape,((q.shape[2]+63)//64)*32), \
            (k-km).round().to(torch.int8), torch.ones(*shape,((k.shape[2]+127)//128)*4)

    def global_v(v,**kwargs):
        assert kwargs == dict(tensor_layout="HND",smooth_v=False)
        return v.transpose(-1,-2).contiguous().to(torch.float8_e4m3fn), torch.ones(*v.shape[:2],v.shape[-1]),None

    def block_v(v, channel_scale=True):
        b,h,n,d = v.shape
        # Deliberately identifiable data; these mocks test orchestration, not CUDA quantization.
        return v.transpose(-1,-2).contiguous().to(torch.float8_e4m3fn), \
            torch.full((b,h,d) if channel_scale else (b,h,(n+127)//128,d),2.), torch.full((b,h,(n+127)//128,d),3.,dtype=torch.float16)

    return SimpleNamespace(qk=qk,v_global=global_v,ext=SimpleNamespace(group_values=group,prepare_values=block_v,sage_vc_kernel_version=4))


def load_installed_prepare(monkeypatch,ops):
    # Execute the actual vendored Python prepare with injectable CPU primitive doubles.
    spec = importlib.util.spec_from_file_location("sageattention_local.vc",ROOT / "third_party/sageattention_local/sageattention_local/vc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,"_extension",lambda: ops.ext)
    monkeypatch.setitem(sys.modules,"sageattention_local.triton.quant_per_thread",SimpleNamespace(per_thread_int8=ops.qk))
    monkeypatch.setitem(sys.modules,"sageattention_local.quant",SimpleNamespace(per_channel_fp8=ops.v_global))
    return module


@pytest.mark.parametrize("smooth",[False,True])
@pytest.mark.parametrize("cast",[False,True])
@pytest.mark.parametrize("n",[1,128,129])
def test_staged_mirror_matches_production_preparation(monkeypatch,smooth,cast,n):
    torch.manual_seed(407)
    qkv = [torch.randn(2,3,t,64).half() for t in (65,n,n)]
    ops = fake_ops()
    vc = load_installed_prepare(monkeypatch,ops)
    expected = vc.prepare(*qkv,expcast=cast,v_smooth=smooth)
    recorder = Recorder()
    actual = profile.staged_prepare(*qkv,ops,recorder,expcast=cast,smooth=smooth)
    profile.assert_payload_equal(actual,expected)
    names = (["cluster_sort","reorder_kv"] if smooth else []) + ["k_mean","qk_int8"]
    if not smooth and n % 128:
        names.append("v_pad")
    assert recorder.names == names + ["v_quantize"]
    if smooth:
        assert torch.equal(actual["permutation"], torch.arange(n-1,-1,-1).expand(2,3,n))
    # Input tensors must not be modified, and the attention dispatch receives the staged state.
    ops.vc = SimpleNamespace(forward_prepared=lambda state: state)
    recorder = Recorder()
    output = profile.staged_forward(qkv,ops,recorder,expcast=cast,smooth=smooth)
    profile.assert_payload_equal(output,expected)
    assert recorder.names[-1] == "attention"


def test_frozen_permutation_skips_clustering_only_in_parity_check():
    qkv = [torch.randn(1,2,5,64).half() for _ in range(3)]
    ops = fake_ops()
    permutation = torch.tensor([2,4,0,3,1]).expand(1,2,5)
    ops.ext.group_values = lambda *args: pytest.fail("Parity check must reuse the exact permutation")
    recorder = Recorder()
    state = profile.staged_prepare(*qkv,ops,recorder,expcast=False,smooth=True,fixed_permutation=permutation)
    assert "cluster_sort" not in recorder.names
    assert state["permutation"] is permutation
    expected_k = qkv[1].gather(2,permutation.unsqueeze(-1).expand_as(qkv[1]))
    assert torch.equal(state["k"],(expected_k-expected_k.mean(2,keepdim=True)).round().to(torch.int8))


def test_kernel_comparison_reuses_identical_state_and_selects_both_modes():
    calls = []
    def forward(state, *, kernel_mode):
        calls.append((state,kernel_mode))
    a, b = {"q":object()}, {"q":object()}
    ops = SimpleNamespace(vc=SimpleNamespace(forward_prepared=forward))
    functions = profile.kernel_calls(ops,{"a":a,"b":b})
    for fn in functions.values():
        fn()
    assert [mode for _,mode in calls] == ["vc4","optimized","vc4","optimized"]
    assert calls[0][0] is calls[1][0] is a
    assert calls[2][0] is calls[3][0] is b


def test_output_difference_checks_tail_and_all_heads_and_rejects_nonfinite():
    ref = torch.ones(2,3,1025,4)
    actual = ref.clone()
    actual[-1,-1,-1,-1] += 2
    result = profile.output_difference(actual,ref,chunk=512)
    assert result["checked_queries"] == 1025
    assert result["max_abs"] == 2
    assert result["rel_rmse"] == pytest.approx(2/ref.numel()**.5)
    assert not result["exact_match"]
    assert profile.output_difference(ref,ref)["exact_match"]
    actual[-1,-1,-1,-1] = float("nan")
    with pytest.raises(RuntimeError,match="Nonfinite"):
        profile.output_difference(actual,ref)


def test_kernel_comparison_cli_and_missing_native_support():
    args = profile.parse_args(["--compare-kernels"])
    assert args.compare_kernels
    with pytest.raises(RuntimeError,match="needs vc5"):
        profile.compare_kernels(SimpleNamespace(ext=SimpleNamespace()),{},args,"cpu")
    with pytest.raises(SystemExit):
        profile.parse_args(["--compare-kernels","--ncu-variant","sage_vc"])


def test_v_scale_comparison_holds_grouping_and_qk_payload_fixed():
    qkv = [torch.randn(2,3,n,64).half() for n in (65,129,129)]
    ops = fake_ops()
    base = profile.staged_prepare(*qkv,ops,Recorder(),expcast=False,smooth=True)
    seen = []
    quantize = ops.ext.prepare_values
    def block(v,channel_scale):
        seen.append((v,channel_scale))
        return quantize(v,channel_scale)
    ops.ext.prepare_values = block
    ops.ext.group_values = lambda *args: pytest.fail("Must reuse exact grouping")
    states = profile.value_scale_states(ops,qkv,base)
    assert states["channel"] is base
    for key in ("q","k","qs","ks","permutation"):
        assert states["block"][key] is base[key]
    assert states["channel"]["vs"].shape == (2,3,64)
    assert states["block"]["vs"].shape == (2,3,2,64)
    assert seen[0][1] is False
    torch.testing.assert_close(seen[0][0],qkv[2].gather(2,base["permutation"].unsqueeze(-1).expand_as(qkv[2])))


def test_v_scale_comparison_timing_and_accuracy_are_separate(monkeypatch):
    qkv = [torch.randn(1,1,n,64).half() for n in (5,129,129)]
    ops = fake_ops()
    base = profile.staged_prepare(*qkv,ops,Recorder(),expcast=False,smooth=True)
    calls = []
    def forward(state):
        # Deliberately large quantization difference: this is NOT a same-payload
        # regression and must not be rejected using the 0.1% implementation gate.
        return torch.full_like(qkv[0],2. if state["vs"].ndim == 3 else 1.)
    def api(*args,expcast,v_smooth,v_scale_mode):
        calls.append(v_scale_mode)
        return torch.ones_like(args[0])
    ops.vc = SimpleNamespace(forward_prepared=forward,sageattn_vc=api)
    monkeypatch.setattr(profile,"measure_trial",lambda fn,repeats,device:(fn(),1.))
    monkeypatch.setattr(profile.torch.cuda,"synchronize",lambda *_:None)
    args = SimpleNamespace(check_queries=3,reference_chunk=2,warmup=1,repeats=1,trials=2,seed=4)
    result = profile.compare_v_scales(ops,qkv,base,args,"cpu")
    assert result["query_indices"] == [0,2,4]
    assert set(calls) == {"block","channel"}
    assert len(result["trial_order"]) == 2
    for row in result["variants"].values():
        assert row["channel_vs_block_all_outputs"]["checked_queries"] == 5
        assert row["channel_vs_block_all_outputs"]["rel_rmse"] == 1
        assert row["channel"]["full_api"]["median_ms"] == 1
        assert row["block"]["vs_dense_fp32"]["relative_rmse_vs_fp32"] > 0


def test_v_scale_cli_guards():
    assert profile.parse_args(["--compare-v-scales"]).compare_v_scales
    with pytest.raises(SystemExit): profile.parse_args(["--check-queries","-1"])
    with pytest.raises(SystemExit): profile.parse_args(["--reference-chunk","0"])
    with pytest.raises(SystemExit): profile.parse_args(["--ncu-variant","sage_vc","--compare-v-scales"])
    with pytest.raises(RuntimeError,match="needs vc6"):
        profile.compare_v_scales(SimpleNamespace(ext=SimpleNamespace()),None,None,None,None)


def test_kernel_comparison_reports_interleaved_trials_and_full_output_errors(monkeypatch):
    seen = []
    def forward(state, *, kernel_mode):
        seen.append(kernel_mode)
        return state[kernel_mode]
    def measure(call,repeats,device):
        for _ in range(repeats):
            out = call()
        return out, (2. if seen[-1] == "vc4" else 1.)
    monkeypatch.setattr(profile,"measure_trial",measure)
    monkeypatch.setattr(profile.torch.cuda,"synchronize",lambda *_: None)
    ops = SimpleNamespace(ext=SimpleNamespace(sage_vc_kernel_version=3),
                          vc=SimpleNamespace(forward_prepared=forward))
    state = {"vc4":torch.ones(1,2,7,4),"optimized":torch.ones(1,2,7,4)+1e-5}
    args = SimpleNamespace(warmup=1,trials=3,repeats=2,seed=17)
    report = profile.compare_kernels(ops,{"test":state},args,"cpu")
    row = report["variants"]["test"]
    assert row["speedup"] == 2
    assert row["output_difference"]["checked_queries"] == 7
    assert 0 < row["output_difference"]["rel_rmse"] < .001
    assert len(report["trial_order"]) == 3
    for order in report["trial_order"]:
        assert set(order) == {("test","vc4"),("test","optimized")}
    state["optimized"].fill_(2)
    with pytest.raises(AssertionError,match="exceeds"):
        profile.compare_kernels(ops,{"test":state},args,"cpu")


def test_vc7_comparison_selects_vc6_and_keeps_explicit_vc4_available(monkeypatch):
    seen = []
    def forward(state,*,kernel_mode):
        seen.append((state,kernel_mode))
        return state[kernel_mode]
    ops = SimpleNamespace(ext=SimpleNamespace(sage_vc_kernel_version=5),
                          vc=SimpleNamespace(forward_prepared=forward))
    args = SimpleNamespace(warmup=1,trials=2,repeats=1,seed=17,kernel_baseline=None)
    assert profile.kernel_baseline(ops,args) == "vc6"
    monkeypatch.setattr(profile,"measure_trial",lambda call,repeats,device:(call(),1.))
    monkeypatch.setattr(profile.torch.cuda,"synchronize",lambda *_:None)
    state = {mode:torch.ones(1,1,7,4) for mode in ("vc4","vc6","optimized")}
    report = profile.compare_kernels(ops,{"test":state},args,"cpu")
    assert report["baseline_mode"] == "vc6"
    assert report["variants"]["test"]["output_difference"]["exact_match"]
    assert set(mode for _,mode in seen) == {"vc6","optimized"}
    assert all(x is state for x,_ in seen)
    args.kernel_baseline = "vc4"
    assert profile.kernel_baseline(ops,args) == "vc4"
    args.kernel_baseline = "vc6"
    ops.ext.sage_vc_kernel_version = 4
    with pytest.raises(RuntimeError,match="needs vc7"): profile.kernel_baseline(ops,args)
    with pytest.raises(SystemExit): profile.parse_args(["--kernel-baseline","vc6"])


def test_payload_parity_checks_used_scales_and_bytes_but_ignores_unused_padding():
    qkv = [torch.randn(1,1,1,64).half() for _ in range(3)]
    state = profile.staged_prepare(*qkv,fake_ops(),Recorder(),expcast=False,smooth=False)
    clone = {key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in state.items()}
    clone["qs"][...,1:] = float("nan")
    clone["ks"][...,1:] = 14
    profile.assert_payload_equal(clone,state)
    clone["qs"][...,0] += 1
    with pytest.raises(AssertionError,match="qs"):
        profile.assert_payload_equal(clone,state)
    clone["qs"] = state["qs"]
    clone["v"].view(torch.uint8)[...,0] = 127
    with pytest.raises(AssertionError,match="v"):
        profile.assert_payload_equal(clone,state)


def test_fixed_core_pairs_share_all_tensors_without_repreparing():
    bases = {False:{"q":torch.tensor([1]),"v":torch.tensor([2]),"expcast":False,"block_values":False},
             True:{"q":torch.tensor([3]),"v":torch.tensor([4]),"expcast":False,"block_values":True}}
    fixed = profile.fixed_variants(bases)
    for a,b in (("sage_vc_control","sage_vc_expcast"),("sage_vc_smooth","sage_vc")):
        assert fixed[a]["q"] is fixed[b]["q"]
        assert fixed[a]["v"] is fixed[b]["v"]
        assert fixed[a]["expcast"] is False and fixed[b]["expcast"] is True
    assert bases[False]["expcast"] is False and bases[True]["expcast"] is False


def test_stage_summary_uses_trial_means_and_keeps_cpu_gpu_separate():
    def sample(x):
        return {"wall_ms":x+3,"stages":{"k_mean":{"cuda_event_ms":x,"cpu_enqueue_ms":x/10},
                                        "attention":{"cuda_event_ms":2*x,"cpu_enqueue_ms":x/20}}}
    report = profile.summarize_stages([[sample(1),sample(3)],[sample(10),sample(14)]])
    assert report["stages"]["k_mean"]["cuda_event_ms"]["trial_mean_ms"] == [2,12]
    assert report["stages"]["k_mean"]["cpu_enqueue_ms"]["median_ms"] == pytest.approx(0.7)
    assert report["event_sum"]["median_ms"] == 21
    assert report["instrumented_wall"]["median_ms"] == 10


def test_trace_summary_excludes_nested_cpu_and_async_markers():
    rows = [{"cat":cat,"ph":"X","name":name,"dur":dur}
            for cat,name,dur in (("kernel","void sage_vc::sage_vc_attention_kernel<...>",12000),
                                 ("kernel","gather",1500),("kernel","gather",2500),
                                 ("cuda_runtime","cudaDeviceSynchronize",18000),
                                 ("cpu_op","aten::gather",99999),("gpu_memcpy","copy",100))]
    rows.append({"cat":"kernel","ph":"s","name":"flow","dur":5000})
    result = profile.trace_summary(rows)
    assert result["core_kernel_ms"] == 12
    assert result["other_kernel_sum_ms"] == 4
    assert result["kernel_sum_ms"] == 16
    assert result["kernel_launches"] == 3
    assert all(row["category"] != "cpu_op" for row in result["events"])
    with pytest.raises(RuntimeError,match="No GPU"):
        profile.trace_summary([])
    with pytest.raises(RuntimeError,match="native core"):
        profile.trace_summary([{"cat":"kernel","ph":"X","name":"gather","dur":3}])


def test_timer_does_not_synchronize_between_stages(monkeypatch):
    events = []
    class Event:
        def __init__(self,**kwargs): pass
        def record(self): events.append("record")
        def elapsed_time(self,other):
            assert events[-1] == "sync"
            return 2.0
    monkeypatch.setattr(torch.cuda,"Event",Event)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda device:events.append("sync"))
    ticks = iter([1.,1.005,2.,2.009])
    monkeypatch.setattr(profile.time,"perf_counter",lambda:next(ticks))
    recorder = profile.EventStages("cuda:0")
    events.clear()
    recorder.call("k_mean",lambda:events.append("mean"))
    recorder.call("attention",lambda:events.append("core"))
    assert events == ["record","mean","record","record","core","record"]
    result = recorder.finish()
    assert events[-1] == "sync"
    assert result["k_mean"]["cuda_event_ms"] == 2
    assert result["k_mean"]["cpu_enqueue_ms"] == pytest.approx(5)
    assert result["attention"]["cpu_enqueue_ms"] == pytest.approx(9)


def test_target_shape_and_argument_validation():
    args = profile.parse_args([])
    assert profile.input_shapes(args)["BSHD"]["q"] == [1,37741,28,128]
    assert args.trace_dir is None and args.ncu_variant is None
    with pytest.raises(SystemExit):
        profile.parse_args(["--repeats","0"])
    with pytest.raises(SystemExit):
        profile.parse_args(["--ncu-variant","sage_vc","--trace-dir","results/x"])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA event execution requires a GPU")
def test_real_cuda_stage_events_and_output():
    device = torch.device("cuda:0")
    with torch.cuda.device(device):
        x = torch.arange(4096,device=device,dtype=torch.float32)
        timer = profile.EventStages(device)
        y = timer.call("k_mean",lambda:x * 2)
        out = timer.call("attention",lambda:y + 1)
        values = timer.finish()
        torch.testing.assert_close(out,x * 2 + 1,rtol=0,atol=0)
        assert set(values) == {"k_mean","attention"}
        assert all(row["cuda_event_ms"] > 0 and row["cpu_enqueue_ms"] >= 0 for row in values.values())
