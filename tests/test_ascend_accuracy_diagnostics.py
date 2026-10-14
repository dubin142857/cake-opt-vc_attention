"""CPU tests of diagnostic payload preservation and failure gates, not NPU tests."""
from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ascend"))
import ascend_accuracy_diagnostics as diag
from vc_attention_ascend.api import prepare
from vc_attention_ascend.reference import payload_reference


def prepared():
    generator = torch.Generator().manual_seed(19)
    q, k, v = [torch.randn(2, 3, n, 64, generator=generator).bfloat16() for n in (65, 257, 257)]
    return prepare(q, k, v, layout="HND", _reference_only=True)


@pytest.mark.parametrize("query", [0, 16, 31, 32, 64])
def test_isolation_retains_original_block_scales_and_tail(query):
    p = prepared()
    isolated = diag.isolate_query_tile(p, 1, 2, query)
    block = query // 32
    assert isolated["q"].shape == (1, 1, 32, 64)
    assert isolated["queries"] == min(32, 65 - 32 * block)
    torch.testing.assert_close(isolated["q_scale"], p["q_scale"][1:2, 2:3, block:block + 1], rtol=0, atol=0)
    for cast in (False, True):
        expected = payload_reference(p, expcast=cast)[1:2, 2:3, block * 32:block * 32 + isolated["queries"]]
        torch.testing.assert_close(payload_reference(isolated, expcast=cast), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("keys", [1, 128, 129, 257])
def test_prefix_keeps_global_scales_and_masks_only_valid_keys(keys):
    p = diag.isolate_query_tile(prepared(), 0, 1, 32)
    prefix = diag.key_prefix(p, keys)
    assert prefix["k"].shape[2] == (keys + 127) // 128 * 128
    torch.testing.assert_close(prefix["v_scale"], p["v_scale"], rtol=0, atol=0)
    torch.testing.assert_close(prefix["q_scale"], p["q_scale"], rtol=0, atol=0)
    prefix["q"] = torch.zeros_like(prefix["q"])
    # Uniform probabilities expose the exact prefix of decoded V.
    expected = (p["v"][:, :, :keys].float().mean(2, keepdim=True)
                * p["v_scale"].unsqueeze(2)).expand(-1, -1, 32, -1)
    torch.testing.assert_close(payload_reference(prefix), expected, rtol=1e-5, atol=1e-6)


def test_worst_row_uses_original_query_id_and_batch_head_task_index():
    ref = torch.ones(2, 3, 3, 8)
    out = ref.clone()
    out[1, 2, 1, 0] += 2
    result = diag.row_errors(out, ref, torch.tensor([0, 49, 64]), 65)
    row = result["worst_rows"][0]
    assert (row["batch"], row["head"], row["query"], row["task"], row["aiv_half"]) == (1, 2, 49, 16, 1)
    assert row["squared_error_fraction"] == 1


@pytest.mark.parametrize("drift,passed", [(0.0, True), (.02, False)])
def test_diagnostic_does_not_relax_decoded_gate(monkeypatch, drift, passed):
    p = diag.isolate_query_tile(prepared(), 0, 0, 0)
    oracle = payload_reference(p)
    monkeypatch.setattr(diag, "raw_forward", lambda vc, payload, expcast: oracle * (1 + drift))
    row, _ = diag.check_payload(None, p, True)
    assert row["passed"] is passed
    assert row["repeat_all_rows"]["relative_rmse"] == 0


def test_payload_differences_identify_changed_quantized_data():
    p = prepared()
    other = {**p, "k": p["k"].clone()}
    other["k"][0, 0, 0, 0] = 0 if other["k"][0, 0, 0, 0] else 1
    report = diag.payload_difference(p, other)
    assert not report["k"]["exact_match"]
    assert all(report[name]["exact_match"] for name in report if name != "k")
