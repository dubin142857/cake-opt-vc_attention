"""Regression: constant V need not survive Sage's rounded/unrounded P ratio."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
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
