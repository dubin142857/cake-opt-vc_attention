"""Keep the H200 main workflow on INT8 QK and honest full/core comparisons."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_int8_h200 as profile


def state(smooth=False):
    return {
        "q": torch.zeros(1, 1, 1, 64, dtype=torch.int8),
        "k": torch.zeros(1, 1, 1, 64, dtype=torch.int8),
        "v": torch.zeros(1, 1, 64, 128).to(torch.float8_e4m3fn),
        "qs": torch.ones(1), "ks": torch.ones(1), "vs": torch.ones(1),
        "block_values": smooth, "output_type": 1, "scale": .125,
    }


@pytest.mark.parametrize("key", ("q", "k"))
def test_fp8_qk_rejected_before_dispatch(key):
    payload = state()
    payload[key] = payload[key].to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="torch.int8"):
        profile.make_calls(None, None, None, (), {False: payload, True: state(True)})


def test_wrong_quantization_contract_rejected():
    payload = state()
    profile.require_int8_state(payload)
    with pytest.raises(ValueError, match="INT8 QK"):
        profile.require_int8_state({**payload, "qk_format": "fp8_block"})
    with pytest.raises(ValueError, match="E4M3"):
        profile.require_int8_state({**payload, "v": payload["v"].half()})
    with pytest.raises(ValueError, match="FP32"):
        profile.require_int8_state({**payload, "qs": payload["qs"].half()})


def test_feature_pairs_share_payloads_but_full_calls_preprocess_fresh_inputs():
    seen = []
    def forward(payload, *, kernel_mode):
        seen.append(("core", payload, kernel_mode))
    def full(*qkv, **kwargs):
        seen.append(("full", qkv, kwargs))
    vc = SimpleNamespace(forward_prepared=forward, sageattn_vc=full)
    bases = {smooth: state(smooth) for smooth in (False, True)}
    original = tuple(object() for _ in range(3))
    core, apis, states = profile.make_calls(vc, full, None, original, bases)
    for name, cast, smooth, mode in profile.VARIANTS:
        core[name]()
        assert seen[-1] == ("core", states[name], mode)
        for key in ("q", "k", "v", "qs", "ks", "vs"):
            assert states[name][key] is bases[smooth][key]
        apis[name](); apis[name]()
        assert seen[-1][0] == seen[-2][0] == "full"
        assert seen[-1][1] == original
        assert seen[-1][2]["v_smooth"] is smooth
        assert seen[-1][2]["expcast"] is cast
        assert seen[-1][2]["kernel_mode"] == mode
    assert all("expcast" not in base for base in bases.values())
    assert states["int8_vc"]["q"] is not states["int8_expcast"]["q"]
    assert {mode for _, _, _, mode in profile.VARIANTS[:4]} == {"vc15"}
    assert profile.VARIANTS[-1][1:] == (True, False, "vc22_both")
    assert "sage_cpp" in core and "sage_cpp" in apis


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_nonfinite_outputs_fail_even_without_quality_gate(bad):
    expected = torch.ones(2, 3)
    with pytest.raises(AssertionError, match="Nonfinite"):
        profile.metrics(torch.full_like(expected, bad), expected)


def test_regression_gate_and_metrics():
    expected = torch.ones(2, 3)
    assert profile.metrics(expected, expected, limit=.001)["relative_rmse"] == 0
    # Dense-attention quantization error is reported, not given a model-quality gate.
    assert profile.metrics(expected * 2, expected)["relative_rmse"] == 1
    with pytest.raises(AssertionError, match="exceeds"):
        profile.metrics(expected * 2, expected, limit=.001)
    with pytest.raises(AssertionError, match="shapes"):
        profile.metrics(expected[:1], expected)


def test_cli_workload_and_measurement_validation():
    args = profile.parse_args([])
    assert (args.batch, args.queries, args.heads, args.dim) == (1, 37741, 28, 128)
    assert args.tokens == 37741 and args.dtype == "bfloat16"
    assert args.trace_dir is None
    assert profile.parse_args(["--check-queries", "0"]).check_queries == 0
    for argv in (["--trials", "0"], ["--tokens", "0"], ["--check-queries", "-1"]):
        with pytest.raises(SystemExit):
            profile.parse_args(argv)


def run_shell(args=(), fail=""):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    # Run the actual orchestrator with stubbed build/validation commands.
    stub = r'''
python3() { echo "python3 $*"; if [[ "$1" == "$FAIL_SCRIPT" ]]; then return 9; fi; }
bash() { echo "build $*"; }
export -f python3 bash
/usr/bin/bash scripts/validate_int8_h200.sh "$@"
'''
    return subprocess.run([bash, "-c", stub, "test", *args], cwd=ROOT,
                          env={**os.environ, "FAIL_SCRIPT": fail}, capture_output=True, text=True)


@pytest.mark.parametrize("args", ((), ("--skip-build",)))
def test_shell_never_runs_fp8_qk_workflow(args):
    result = run_shell(args)
    assert result.returncode == 0, result.stderr
    assert "profile_int8_h200.py" in result.stdout
    assert "fp8_qk" not in result.stdout
    if args:
        assert "build scripts/" not in result.stdout
        assert result.stdout.index("validate_sage_vc.py") < result.stdout.index("profile_int8_h200.py")
    else:
        assert "build scripts/build_sage_local.sh" in result.stdout


def test_shell_stops_before_timing_on_validation_failure():
    result = run_shell(("--skip-build",), "scripts/validate_sage_vc.py")
    assert result.returncode == 9
    assert "profile_int8_h200.py" not in result.stdout
    result = run_shell(("--unknown",))
    assert result.returncode == 2 and "Usage:" in result.stderr
