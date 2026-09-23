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
