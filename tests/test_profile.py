import importlib.util
import json
from pathlib import Path
import sys


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location("vc_profile_cuda", scripts / "profile_cuda.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
finally:
    sys.path.remove(str(scripts))


def test_trace_summary_does_not_double_count_cpu_operators_or_mix_cpu_gpu_time(tmp_path):
    events = [
        {"cat": "kernel", "ph": "X", "name": "gemm", "dur": 5},
        {"cat": "kernel", "ph": "X", "name": "gemm", "dur": 7},
        {"cat": "cuda_runtime", "ph": "X", "name": "cudaLaunchKernel", "dur": 3},
        {"cat": "cpu_op", "ph": "X", "name": "aten::matmul", "dur": 100},
        {"cat": "gpu_memcpy", "ph": "X", "name": "Memcpy DtoH", "dur": 2},
        {"cat": "kernel", "ph": "s", "name": "gemm", "dur": 999},
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    rows = profile.summarize_trace(path)
    assert rows == [
        {"category": "kernel", "name": "gemm", "calls": 2, "total_us": 12},
        {"category": "cuda_runtime", "name": "cudaLaunchKernel", "calls": 1, "total_us": 3},
        {"category": "gpu_memcpy", "name": "Memcpy DtoH", "calls": 1, "total_us": 2},
    ]


def test_gpu_summary_sums_main_and_tail_core_and_excludes_host_waits():
    rows = [
        {"category": "kernel", "name": "fused_attention_interleaved", "calls": 3, "total_us": 210000},
        {"category": "kernel", "name": "fused_attention_d128<true,false,true>", "calls": 3, "total_us": 3000},
        {"category": "kernel", "name": "prepare_values", "calls": 3, "total_us": 12000},
        {"category": "cuda_runtime", "name": "cudaDeviceSynchronize", "calls": 1, "total_us": 225000},
    ]
    assert profile.summarize_gpu(rows, 3) == {
        "core_ms_per_call": 71, "other_kernels_ms_per_call": 4, "kernel_sum_ms_per_call": 75,
        "core_percent_of_kernel_sum": 100 * 213000 / 225000, "kernel_launches_per_call": 3}
