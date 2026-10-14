"""CPU fake-native dispatch tests; no NPU kernel compilation or execution."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ascend"))
from vc_attention_ascend import api


def fixture_payload(dtype, layout):
    # Only the device tag is mocked. Native output formatting still uses real
    # CPU tensors, including NaNs in padding which must never reach the API.
    payload = {name: object() for name in ("k", "v", "q_scale", "k_scale", "v_scale", "lut")}
    payload.update(q=SimpleNamespace(device=SimpleNamespace(type="npu")),
                   queries=65, keys=129, scale=.125, dtype=dtype, layout=layout)
    raw = torch.arange(2 * 3 * 128 * 64, dtype=torch.float32).reshape(2, 3, 128, 64) / 7000
    raw[:, :, 65:] = float("nan")
    return payload, raw


def assert_payload_args(args, payload, cast, tile=None):
    for actual, name in zip(args, ("q", "k", "v", "q_scale", "k_scale", "v_scale", "lut")):
        assert actual is payload[name]
    expected = (65, 129, .125, cast) + (() if tile is None else (tile,))
    assert args[7:] == expected


def assert_formatted(actual, raw, payload):
    expected = raw[:, :, :65].to(payload["dtype"])
    if payload["layout"] == "NHD":
        expected = expected.transpose(1, 2)
    assert actual.dtype == payload["dtype"]
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,layout", [(torch.float16, "HND"), (torch.bfloat16, "NHD")])
@pytest.mark.parametrize("tile", [None, 64])
def test_default_legacy_dispatch_and_m64_dispatch_preserve_payload_and_format(monkeypatch, dtype, layout, tile):
    payload, raw = fixture_payload(dtype, layout)
    calls = []

    class FakeNative:
        def forward(self, *args):
            calls.append(("legacy", args))
            return raw[:, :, :96]  # M32's native padded shape.

        def forward_tiled(self, *args):
            calls.append(("tiled", args))
            return raw  # M64 pads the same Q65 payload to Q128.

    monkeypatch.setattr(api, "load_native", lambda: FakeNative())
    kwargs = {} if tile is None else {"tile_m": tile}
    actual = api.run_prepared(payload, expcast=False, **kwargs)
    assert len(calls) == 1
    assert calls[0][0] == ("legacy" if tile is None else "tiled")
    assert_payload_args(calls[0][1], payload, False, tile)
    assert_formatted(actual, raw, payload)


@pytest.mark.parametrize("tile,dtype,layout", [(32, torch.float16, "NHD"), (64, torch.bfloat16, "HND")])
def test_probe_dispatch_preserves_raw_records_and_formats_only_output(monkeypatch, tile, dtype, layout):
    payload, raw = fixture_payload(dtype, layout)
    records = torch.arange(2 * 3 * 4 * 16, dtype=torch.int64).reshape(2, 3, 4, 16)
    before = records.clone()
    calls = []

    class FakeNative:
        def profile_tiled(self, *args):
            calls.append(args)
            return raw, records

    monkeypatch.setattr(api, "load_native", lambda: FakeNative())
    actual, actual_records = api.profile_prepared(payload, expcast=True, tile_m=tile)
    assert len(calls) == 1
    assert_payload_args(calls[0], payload, True, tile)
    assert_formatted(actual, raw, payload)
    assert actual_records is records
    assert torch.equal(records, before)


@pytest.mark.parametrize("entry", [api.run_prepared, api.profile_prepared])
@pytest.mark.parametrize("tile", [0, 16, 33, 128])
def test_invalid_execution_tile_is_rejected_before_native_loading(monkeypatch, entry, tile):
    payload, _ = fixture_payload(torch.float16, "HND")

    def unexpected_native_load():
        raise AssertionError("Invalid tile must not load or invoke the native extension")

    monkeypatch.setattr(api, "load_native", unexpected_native_load)
    with pytest.raises(ValueError, match="tile_m must be 32 or 64"):
        entry(payload, tile_m=tile)


def test_full_api_forwards_tile_and_preprocessing_options(monkeypatch):
    payload, raw = fixture_payload(torch.bfloat16, "NHD")
    q, k, v = object(), object(), object()
    calls = []

    def fake_prepare(q_arg, k_arg, v_arg, **kwargs):
        assert q_arg is q and k_arg is k and v_arg is v
        calls.append(("prepare", kwargs))
        return payload

    def fake_run(p, **kwargs):
        assert p is payload
        calls.append(("run", kwargs))
        return raw

    monkeypatch.setattr(api, "prepare", fake_prepare)
    monkeypatch.setattr(api, "run_prepared", fake_run)
    assert api.attention(q, k, v, tile_m=64, expcast=False, check_finite=False) is raw
    assert calls == [("prepare", {"layout": "NHD", "v_quantize": True, "check_finite": False}),
                     ("run", {"expcast": False, "tile_m": 64})]
