"""Regression: constant V need not survive Sage's rounded/unrounded P ratio."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_sage_vc31 as validation
from validate_sage_vc31 import check_constant_channels

spec = importlib.util.spec_from_file_location(
    "vc31_validation_oracle", ROOT / "third_party/sageattention_local/sageattention_local/vc_reference.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def constant_payload(dtype, expcast, smooth, keys):
    # One logit is 0; all others are -0.3. After Sage's rescaling, the
    # smaller probabilities round to 320, creating a deliberate row-sum bias.
    dim, queries, tiles = 64, 3, (keys + 127) // 128
    q = torch.zeros(1, 1, queries, dim, dtype=torch.int8)
    q[..., 0] = 1
    k = torch.zeros(1, 1, keys, dim, dtype=torch.int8)
    k[..., 0] = -3
    k[:, :, 0, 0] = 0
    values = torch.zeros(1, 1, keys, dim, dtype=dtype)
    values[..., 0], values[..., 1] = 1.5, -2.
    packed = torch.zeros(1, 1, dim, tiles*128)
    means = torch.zeros(1, 1, tiles, dim, dtype=torch.float16)
    if smooth:
        means[..., 0], means[..., 1] = 3., -4.
    else:
        # Constant rows make the V token permutation immaterial, including padding.
        packed[:, :, 0], packed[:, :, 1] = 3., -4.
    state = dict(q=q, k=k, v=packed.to(torch.float8_e4m3fn),
                 qs=torch.ones(1, 1, 8), ks=torch.ones(1, 1, tiles*4),
                 vs=torch.full((1, 1, dim), .5), means=means,
                 expcast=expcast, block_values=smooth, scale=.1)
    return state, values


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("keys", [128, 129, 257])
@pytest.mark.parametrize("expcast,v_smooth", [(False, False), (True, False), (False, True), (True, True)])
def test_constant_contract_against_independent_two_logit_formula(dtype, keys, expcast, v_smooth):
    state, values = constant_payload(dtype, expcast, v_smooth, keys)
    decoded = oracle.reference(state)
    constant = values[:, :, :1].expand_as(decoded)
    if not expcast and not v_smooth:
        # Independent scalar expression, not another call to the tiled oracle.
        peak = 2. ** float(torch.tensor(8.807, dtype=torch.float32))
        small = peak * torch.exp(torch.tensor(-.3, dtype=torch.float64)).item()
        p = torch.tensor([peak, small]).to(torch.float8_e4m3fn).double()
        ratio = (p[0] + (keys-1)*p[1]) / (peak + (keys-1)*small)
        expected = constant.float() * ratio.float()
        torch.testing.assert_close(decoded, expected, rtol=2e-6, atol=2e-6)
        # The old validator falsely rejected this correct result by >3%.
        with pytest.raises(AssertionError):
            torch.testing.assert_close(decoded.to(dtype)[..., :2], constant[..., :2], rtol=.004, atol=.003)
    else:
        torch.testing.assert_close(decoded, constant.float(), rtol=2e-6, atol=2e-6)
    result = check_constant_channels(decoded.to(dtype), decoded.to(dtype), values,
                                     expcast=expcast, v_smooth=v_smooth)
    assert result["contract"] == ("constant_preservation" if expcast or v_smooth else "rounded_P_over_unrounded_mass")


@pytest.mark.parametrize("expcast,v_smooth", [(False, False), (True, False), (False, True), (True, True)])
def test_wrong_output_still_fails_elementwise_gate(expcast, v_smooth):
    state, values = constant_payload(torch.float16, expcast, v_smooth, 128)
    decoded = oracle.reference(state).half()
    wrong = decoded.clone()
    wrong[0, 0, 0, 0] += .1
    with pytest.raises(AssertionError, match="decoded quantized reference"):
        check_constant_channels(wrong, decoded, values, expcast=expcast, v_smooth=v_smooth)


def test_using_rounded_mass_in_plain_sage_is_not_accepted():
    state, values = constant_payload(torch.float16, False, False, 128)
    decoded = oracle.reference(state).half()
    wrong = values[:, :, :1].expand_as(decoded)
    with pytest.raises(AssertionError, match="decoded quantized reference"):
        check_constant_channels(wrong, decoded, values, expcast=False, v_smooth=False)


@pytest.mark.parametrize("expcast,v_smooth", [(True, False), (False, True), (True, True)])
def test_constant_preservation_remains_an_independent_gate(expcast, v_smooth):
    state, values = constant_payload(torch.float16, expcast, v_smooth, 128)
    bad = oracle.reference(state).half() * .98
    # A matching but erroneous reference must not disable the known-answer gate.
    with pytest.raises(AssertionError, match="preserve constant"):
        check_constant_channels(bad, bad, values, expcast=expcast, v_smooth=v_smooth)


def test_random_bf16_pointwise_difference_is_logged_with_unchanged_global_gates():
    values = torch.full((1, 1, 128, 64), -2., dtype=torch.bfloat16)
    decoded = torch.full((1, 1, 65, 64), -2., dtype=torch.bfloat16)
    previous = decoded.clone()
    # Adjacent BF16 outputs can fail the old local .004/.003 check even
    # when vc31 equals vc28 and the original whole-output oracle gate passes.
    decoded[0, 0, 3, 1] = -2.015625
    previous[0, 0, 3, 1] = -2.03125
    actual = previous.clone()
    validation.check_random_output(actual, previous, decoded, v_smooth=False)
    diagnostic = validation.constant_channel_diagnostics(actual, decoded, values, previous)
    assert diagnostic["previous_exact_match"]
    assert not diagnostic["pointwise_reference_within_legacy_tolerance"]
    assert diagnostic["worst_reference_element"] == {
        "index": [0, 0, 3, 1], "actual": -2.03125, "decoded": -2.015625,
        "previous": -2.03125, "abs_error": .015625,
    }
    assert diagnostic["vs_decoded_reference"] == diagnostic["previous_vs_decoded_reference"]


@pytest.mark.parametrize("failure", ["previous", "decoded", "bitwise", "nan"])
def test_random_regression_and_reference_gates_still_fail(failure):
    actual = torch.ones(1, 1, 65, 64)
    previous, decoded = actual.clone(), actual.clone()
    if failure == "previous":
        previous.mul_(1.002)  # Exceeds the original 0.1% regression gate.
    elif failure == "decoded":
        decoded.mul_(1.02)  # Exceeds the original 1% oracle gate.
    elif failure == "bitwise":
        previous[0, 0, 3, 1] += 1e-5  # Tiny errors still fail the unsmoothed branch.
    else:
        actual[0, 0, 3, 1] = float("nan")
    with pytest.raises((AssertionError, ValueError)):
        validation.check_random_output(actual, previous, decoded, v_smooth=failure == "previous")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("keys", [1, 128, 129, 513])
@pytest.mark.parametrize("expcast,v_smooth", [(False, False), (True, False), (False, True), (True, True)])
def test_hopper_known_answer_fixtures_have_independent_analytic_answers(dtype, dim, keys, expcast, v_smooth):
    state, values, answer = validation.known_answer_payload(dtype, dim, keys, expcast, v_smooth)
    # Verify the same fixtures that the remote script passes directly to native.
    assert state["qs"].shape[-1] == ((state["q"].shape[2] + 63) // 64) * 32
    assert state["output_type"] == (0 if dtype == torch.float16 else 1)
    assert (state["means"].numel() > 0) == v_smooth
    decoded = oracle.reference(state, chunk=32)
    torch.testing.assert_close(decoded, answer, rtol=2e-6, atol=2e-6)
    check_constant_channels(answer.to(dtype), decoded.to(dtype), values,
                            expcast=expcast, v_smooth=v_smooth)
    unpacked = state["v"].float().index_select(-1, oracle.value_permutation(state["v"].shape[-1], "cpu"))
    assert torch.all(unpacked[..., keys:] == 64.)  # Masked padding cannot masquerade as valid V.


def test_constant_failure_includes_pytorch_numeric_details():
    state, values = constant_payload(torch.bfloat16, False, False, 128)
    decoded = oracle.reference(state).bfloat16()
    with pytest.raises(AssertionError) as error:
        check_constant_channels(decoded + .5, decoded, values, expcast=False, v_smooth=False)
    assert "Greatest absolute difference" in str(error.value)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("mode", ["vc31_unified", "vc32_full_pv", "vc32_packed_mean"])
def test_main_saves_results_including_partial_failed_case(tmp_path, monkeypatch, fail, mode):
    output = tmp_path / "nested" / "validation.json"
    monkeypatch.setattr(sys, "argv", ["validate_sage_vc31.py", "--output", str(output), "--kernel-mode", mode])
    monkeypatch.setitem(sys.modules, "sageattention_local", SimpleNamespace(vc=object(), vc_reference=object()))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))

    def fake_validate(vc, reference, device, report, kernel_mode="vc31_unified"):
        assert kernel_mode == mode
        report.update(cases=[{"status": "completed"}], active_case={"dtype": "torch.bfloat16", "max_abs": .015625})
        if fail:
            raise AssertionError("native diagnostic detail")
        report.pop("active_case")
        report["status"] = "passed"

    monkeypatch.setattr(validation, "validate", fake_validate)
    if fail:
        with pytest.raises(AssertionError, match="native diagnostic detail"):
            validation.main()
    else:
        validation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == ("failed" if fail else "passed")
    assert report["cases"] == [{"status": "completed"}]
    if fail:
        assert report["active_case"]["max_abs"] == .015625
        assert "native diagnostic detail" in report["error"]
