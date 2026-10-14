"""Adapter contract tests with CPU doubles; NOT MindIE/NPU execution tests."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_backend as lb


def make_adapter(monkeypatch):
    monkeypatch.setattr(lb, "_require_npu", lambda q: None)
    calls = []

    def preprocess(q, k, v):
        return tuple(F.pad(x.half(), (0, 128 - x.shape[-1], 0, (-x.shape[2]) % 256)).contiguous()
                     for x in (q, k, v))

    def native(q, k, v, *args):
        calls.append(args)
        _, _, _, scale, heads, layout, keep, tail, next_tokens, precise = args
        assert heads == q.shape[1] and layout == "BNSD" and keep == 1
        assert next_tokens == 1 and precise is True
        assert args[:3] == (None, None, None)
        valid_keys = k.shape[2] - (0 if tail == 2147483647 else tail)
        # Independent dense, unpadded computation, not an implementation of Laser.
        out = ((q.float() @ k[:, :, :valid_keys].float().transpose(-1, -2)) * scale).softmax(-1) @ v[:, :, :valid_keys].float()
        return torch.full(q.shape[:-1], float("nan")), out

    def postprocess(out, dtype, nq, d):
        return out.half().to(dtype)[:, :, :nq, :d]

    return lb.MindIELaser(preprocess, postprocess, native, {}), calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nq,nk,d,tail", [(17, 256, 128, 2147483647), (33, 257, 64, 255)])
def test_padding_scale_output_index_and_fresh_preprocessing(monkeypatch, dtype, nq, nk, d, tail):
    adapter, calls = make_adapter(monkeypatch)
    generator = torch.Generator().manual_seed(37)
    q, k, v = [torch.randn((1, 2, n, d), generator=generator).to(dtype) for n in (nq, nk, nk)]
    v[..., 0] = 2
    p = adapter.prepare(q, k, v)
    assert p.pre_tokens == tail
    assert p.scale == pytest.approx(d**-.5, rel=1e-15)  # original D, not padded D
    out = adapter.run_prepared(p)
    expected = ((q.half().float() @ k.half().float().transpose(-1, -2)) * d**-.5).softmax(-1) @ v.half().float()
    torch.testing.assert_close(out, expected.half().to(dtype), rtol=.004, atol=.001)
    torch.testing.assert_close(out[..., 0], torch.full_like(out[..., 0], 2), rtol=0, atol=0)
    assert out.shape == q.shape and out.dtype == dtype and out.is_contiguous()
    torch.testing.assert_close(adapter(q, k, v), out, rtol=0, atol=0)
    assert len(calls) == 2
    # Full mode reads fresh V, rather than secretly timing the cached payload.
    torch.testing.assert_close(adapter(q, k, torch.zeros_like(v)), torch.zeros_like(out), rtol=0, atol=0)
    assert adapter.describe_payload(p)["native_dtype"] == "torch.float16"


def test_no_cpu_fallback():
    x = torch.zeros(1, 2, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="NPU"):
        lb.MindIELaser(None, None, None, {}).prepare(x, x, x)


def test_changed_preprocessing_is_rejected(monkeypatch):
    adapter, _ = make_adapter(monkeypatch)
    adapter.preprocess = lambda q, k, v: (q, k, v)
    x = torch.zeros(1, 2, 33, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="preprocessing changed"):
        adapter.prepare(x, x, x)


def test_wrong_native_output_is_rejected(monkeypatch):
    adapter, _ = make_adapter(monkeypatch)
    x = torch.zeros(1, 2, 33, 64, dtype=torch.float16)
    p = adapter.prepare(x, x, x)
    adapter.native = lambda *args: torch.empty_like(x)
    with pytest.raises(RuntimeError, match="must return"):
        adapter.run_prepared(p)


def test_missing_mindiesd_message(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'mindiesd'")
    monkeypatch.setattr(lb.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="MindIE-LLM alone is insufficient"):
        lb.load_mindie_laser()


def test_required_laser_failure_writes_report_before_device_work(monkeypatch, tmp_path):
    import benchmark_ascend910b as benchmark
    path = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_ascend910b.py", "--backends", "mindie_laser",
                                    "--require-laser", "--output", str(path)])
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="test-only"))

    def unavailable():
        raise RuntimeError("mindiesd unavailable (test double)")
    monkeypatch.setattr(benchmark, "load_mindie_laser", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        benchmark.main()
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert "mindie_laser" in report["skipped"]
    assert report["timings"] == {}  # no fabricated timing or silent CANN substitute
