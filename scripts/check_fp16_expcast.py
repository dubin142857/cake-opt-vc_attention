"""CPU accuracy experiment comparing FP8 and an FP16 adaptation of ExpCast.

Run: bash scripts/check_fp16_expcast.sh
Requires NumPy only. This measures numerical error, not GPU performance.
The corrected variant is an experimental extension, not a paper FP16 result.
"""

import argparse
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GAIN = np.float32(1024.0 * math.log2(math.e))
BASE_CODE = 1024 * (8 + 15)
CORRECTION = -0.35 * (1024 / 8)
FP8_GAIN = np.float32(8.0 * math.log2(math.e))
FP8_CODES = np.arange(121, dtype=np.uint16)
FP8_EXPONENT = (FP8_CODES >> 3).astype(np.int32)
FP8_MANTISSA = (FP8_CODES & 7).astype(np.float64)
FP8_VALUES = np.where(
    FP8_EXPONENT == 0,
    np.ldexp(FP8_MANTISSA, -9),
    np.ldexp(1.0 + FP8_MANTISSA / 8.0, FP8_EXPONENT - 7),
)


def reference_weights(x):
    """Exact-function reference in FP64, evaluated at the same FP32 inputs."""
    return 256.0 * np.exp(np.asarray(x, dtype=np.float32).astype(np.float64))


def expcast(x, correction=0.0):
    """FP32 affine -> RNE -> uint16 -> FP16 bitcast, then exact FP64 decode.

    FP64 intermediate emulates a single FP32 FMA over the tested input range;
    both the gain and bias are first rounded to FP32. This is deliberately not
    a separately rounded FP32 multiplication followed by FP32 addition.
    """
    x = np.asarray(x, dtype=np.float32)
    bias = np.float32(BASE_CODE + correction)
    affine = (x.astype(np.float64) * float(GAIN) + float(bias)).astype(np.float32)
    codes = np.clip(np.rint(affine), 0, BASE_CODE).astype(np.uint16)
    return codes.view(np.float16).astype(np.float64)


def round_to_fp8(ref):
    """Nearest E4M3FN value, ties to an even code, for weights in [0,256]."""
    high = np.clip(np.searchsorted(FP8_VALUES, ref), 1, 120)
    low = high - 1
    low_distance = ref - FP8_VALUES[low]
    high_distance = FP8_VALUES[high] - ref
    choose_high = (high_distance < low_distance) | (
        (high_distance == low_distance) & ((high & 1) == 0)
    )
    return FP8_VALUES[np.where(choose_high, high, low)]


def expcast_fp8(x, correction=-0.35):
    """FP8 ExpCast with the same FP32 FMA/RNE contract as the FP16 variant."""
    x = np.asarray(x, dtype=np.float32)
    bias = np.float32(120 + correction)
    affine = (x.astype(np.float64) * float(FP8_GAIN) + float(bias)).astype(np.float32)
    codes = np.clip(np.rint(affine), 0, 120).astype(np.uint8)
    return FP8_VALUES[codes]


def evaluate(x, include_fp8=False):
    ref = reference_weights(x)
    values = {
        "exp_then_fp16": ref.astype(np.float16).astype(np.float64),
        "expcast_raw": expcast(x),
        "expcast_corrected": expcast(x, CORRECTION),
    }
    if include_fp8:
        values.update({
            "exp_then_fp8": round_to_fp8(ref),
            "expcast_fp8_raw": expcast_fp8(x, 0),
            "expcast_fp8_corrected": expcast_fp8(x),
        })
    return ref, values


def error_metrics(actual, ref):
    delta = actual - ref
    relative = np.abs(delta) / ref
    return {
        "max_relative_error_percent": float(100 * relative.max()),
        "mean_relative_error_percent": float(100 * relative.mean()),
        "p99_relative_error_percent": float(100 * np.quantile(relative, 0.99)),
        "signed_mean_relative_error_percent": float(100 * (delta / ref).mean()),
        "relative_l2_error_percent": float(100 * np.linalg.norm(delta) / np.linalg.norm(ref)),
        "zero_count": int(np.count_nonzero(actual == 0)),
        "count": int(ref.size),
    }


def validate_encoding():
    """Check every emitted bit pattern against an independent IEEE decoder."""
    codes = np.arange(BASE_CODE + 1, dtype=np.uint16)
    exponent = (codes >> 10).astype(np.int32)
    mantissa = (codes & 1023).astype(np.float64)
    decoded = np.where(
        exponent == 0,
        np.ldexp(mantissa, -24),
        np.ldexp(1.0 + mantissa / 1024.0, exponent - 15),
    )
    np.testing.assert_array_equal(codes.view(np.float16).astype(np.float64), decoded)
    x = np.array([0.0, -math.log(2), -2 * math.log(2)], dtype=np.float32)
    np.testing.assert_array_equal(expcast(x), [256.0, 128.0, 64.0])
    # Verify RNE tie handling and saturation used by the encoder.
    np.testing.assert_array_equal(np.rint([0.5, 1.5, 2.5, 3.5]), [0, 2, 2, 4])
    assert expcast(np.array([-np.inf]))[0] == 0.0
    # Independent exact relation: these FP8 values / 256 have half bits c<<7.
    np.testing.assert_array_equal(FP8_VALUES, (FP8_CODES << 7).view(np.float16).astype(np.float64) * 256)
    np.testing.assert_array_equal(round_to_fp8(FP8_VALUES), FP8_VALUES)
    midpoints = (FP8_VALUES[:-1] + FP8_VALUES[1:]) / 2
    expected_even_codes = np.arange(120) + (np.arange(120) & 1)
    np.testing.assert_array_equal(round_to_fp8(midpoints), FP8_VALUES[expected_even_codes])
    np.testing.assert_array_equal(expcast_fp8(x, 0), [256.0, 128.0, 64.0])
    return {"status": "passed", "independent_ieee_bit_patterns": int(codes.size), "fp8_codes": 121, "fp8_rounding_midpoints": 120}


def compare_fp8(grid_points):
    # This shared interval keeps target AND every candidate output normal in
    # both formats. It isolates the approximation/rounding from underflow.
    x = np.linspace(-9, 0, grid_points, dtype=np.float32)
    ref, values = evaluate(x, include_fp8=True)
    assert ref.min() >= 2 ** -6
    assert all(actual.min() >= 2 ** -6 for actual in values.values())
    return {
        "x_range": [-9.0, 0.0],
        "scope": "Identical FP32 x; targets and outputs normal in both FP8 E4M3FN and FP16. No subnormal or underflow effects in this comparison.",
        "variants": {name: error_metrics(actual, ref) for name, actual in values.items()},
    }


def sweep(grid_points):
    # 256*exp(x) is normal FP16 when x >= -22*ln(2). Filtering handles the
    # FP32 rounding of the endpoint without silently including subnormals.
    x = np.linspace(-22 * math.log(2), 0, grid_points, dtype=np.float32)
    ref, values = evaluate(x)
    normal = ref >= float(np.finfo(np.float16).tiny)
    normal_results = {}
    for name, actual in values.items():
        metrics = error_metrics(actual[normal], ref[normal])
        both_normal = normal & (actual >= float(np.finfo(np.float16).tiny))
        metrics["target_and_output_both_normal"] = error_metrics(actual[both_normal], ref[both_normal])
        worst_index = np.flatnonzero(normal)[np.argmax(np.abs(actual[normal] / ref[normal] - 1))]
        metrics["worst_x"] = float(x[worst_index])
        metrics["worst_reference_weight"] = float(ref[worst_index])
        metrics["worst_approximate_weight"] = float(actual[worst_index])
        normal_results[name] = metrics

    tail_x = np.linspace(-30, -22 * math.log(2), grid_points // 4, dtype=np.float32)
    tail_ref, tail_values = evaluate(tail_x)
    tail = tail_ref < float(np.finfo(np.float16).tiny)
    return {
        "normal_target_range": [float(np.finfo(np.float16).tiny), 256.0],
        "normal": normal_results,
        "subnormal_and_underflow": {
            name: error_metrics(actual[tail], tail_ref[tail])
            for name, actual in tail_values.items()
        },
    }


def scalar_examples():
    x = np.array([0, -math.log(2), -1, -5, -10, -15, -16, -20, -24], dtype=np.float32)
    ref, values = evaluate(x)
    return [
        {
            "x_fp32": float(x[i]),
            "reference_weight": float(ref[i]),
            **{
                name: {
                    "weight": float(actual[i]),
                    "relative_error_percent": float(100 * abs(actual[i] / ref[i] - 1)),
                }
                for name, actual in values.items()
            },
        }
        for i in range(len(x))
    ]


def softmax_case(name, logits, seed):
    # The FP32 max subtraction models the input of the proposed ExpCast step.
    logits = np.asarray(logits, dtype=np.float32)
    x = logits - logits.max(axis=-1, keepdims=True)
    ref_weights, weights = evaluate(x, include_fp8=True)
    ref = ref_weights / ref_weights.sum(axis=-1, keepdims=True)
    rng = np.random.default_rng(seed)
    # Same FP32 V for every variant; FP64 PV isolates the probability error.
    v = rng.standard_normal((x.shape[1], 32), dtype=np.float32).astype(np.float64)
    ref_output = ref @ v
    candidates = {}
    for variant, weight in weights.items():
        actual = weight / weight.sum(axis=-1, keepdims=True)
        delta = actual - ref
        tv = 0.5 * np.abs(delta).sum(axis=-1)
        output_delta = actual @ v - ref_output
        candidates[variant] = {
            "mean_total_variation_percent": float(100 * tv.mean()),
            "max_total_variation_percent": float(100 * tv.max()),
            "probability_relative_l2_percent": float(100 * np.linalg.norm(delta) / np.linalg.norm(ref)),
            "max_absolute_probability_error": float(np.abs(delta).max()),
            "output_relative_l2_percent": float(100 * np.linalg.norm(output_delta) / np.linalg.norm(ref_output)),
            "reference_probability_mass_dropped_percent": float(100 * np.sum(ref * (weight == 0), axis=-1).mean()),
        }
        if name == "one_peak_many_small_weights":
            # V[0]=0, every tail V=1: output equals all tail probability mass.
            candidates[variant]["tail_indicator_v_reference_output"] = float(ref[0, 1:].sum())
            candidates[variant]["tail_indicator_v_actual_output"] = float(actual[0, 1:].sum())
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual.sum(axis=-1), 1.0, rtol=0, atol=1e-14)
    return {"name": name, "rows": int(x.shape[0]), "tokens": int(x.shape[1]), "value_dim": 32, "variants": candidates}


def softmax_experiments(seed):
    results = []
    for i, sigma in enumerate((1, 2, 4, 8)):
        rng = np.random.default_rng(seed + i)
        logits = rng.standard_normal((256, 1024), dtype=np.float32) * np.float32(sigma)
        results.append(softmax_case(f"gaussian_sigma{sigma}", logits, seed + 100 + i))
    rng = np.random.default_rng(seed + 4)
    results.append(softmax_case("long_gaussian_sigma4", rng.standard_normal((32, 37741), dtype=np.float32) * 4, seed + 104))
    results.append(softmax_case("uniform", np.zeros((4, 1024), dtype=np.float32), seed + 105))
    tail = np.full((1, 37741), -16.0, dtype=np.float32)
    tail[:, 0] = 0.0
    results.append(softmax_case("one_peak_many_small_weights", tail, seed + 106))
    return results


def print_report(report):
    print("FP8 / FP16 ExpCast accuracy; CPU numerical experiment, NOT a GPU speed benchmark")
    print(f"FP32 gain={float(GAIN):.10f}; raw bias={BASE_CODE}; corrected bias={float(np.float32(BASE_CODE + CORRECTION)):.10f}")
    print("Reference: 256*exp(FP32 x), evaluated and normalized in FP64.")
    print("Normal target weights [2^-14, 256]; dense-grid errors (%):")
    print(f"{'variant':<22} {'max relative':>14} {'mean relative':>15} {'p99 relative':>14}")
    for name, stats in report["weight_sweep"]["normal"].items():
        print(f"{name:<22} {stats['max_relative_error_percent']:>14.6f} {stats['mean_relative_error_percent']:>15.6f} {stats['p99_relative_error_percent']:>14.6f}")
    print("Max relative error when BOTH target and output are normal FP16 (%):")
    for name, stats in report["weight_sweep"]["normal"].items():
        print(f"  {name}: {stats['target_and_output_both_normal']['max_relative_error_percent']:.6f}")
    print("\nFP8 vs FP16 on IDENTICAL x in [-9,0], all targets/outputs normal; errors (%):")
    print(f"{'variant':<24} {'max relative':>14} {'mean relative':>15}")
    for name, stats in report["fp8_comparison"]["variants"].items():
        print(f"{name:<24} {stats['max_relative_error_percent']:>14.6f} {stats['mean_relative_error_percent']:>15.6f}")
    print("\nSoftmax and random-V output errors (%); TV is half the probability L1 distance:")
    print(f"{'case':<29} {'variant':<22} {'mean TV':>10} {'max TV':>10} {'P rel L2':>10} {'PV rel L2':>11}")
    for case in report["softmax"]:
        for name, stats in case["variants"].items():
            print(f"{case['name']:<29} {name:<22} {stats['mean_total_variation_percent']:>10.5f} {stats['max_total_variation_percent']:>10.5f} {stats['probability_relative_l2_percent']:>10.5f} {stats['output_relative_l2_percent']:>11.5f}")
    print("\nScalar weights (tiny values can underflow; see JSON for their relative errors):")
    print(f"{'x':>10} {'reference':>14} {'exp->FP16':>14} {'raw ExpCast':>14} {'corrected':>14}")
    for row in report["scalar_examples"]:
        print(f"{row['x_fp32']:>10.5f} {row['reference_weight']:>14.7g} {row['exp_then_fp16']['weight']:>14.7g} {row['expcast_raw']['weight']:>14.7g} {row['expcast_corrected']['weight']:>14.7g}")
    print("\nTail diagnostic: logits=[0, -16 repeated 37740 times], V=[0, 1, ..., 1]:")
    for name, stats in report["softmax"][-1]["variants"].items():
        print(f"  {name}: output={stats['tail_indicator_v_actual_output']:.10g}; reference={stats['tail_indicator_v_reference_output']:.10g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "fp16_expcast_accuracy.json")
    parser.add_argument("--grid-points", type=int, default=2_000_001)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.grid_points < 100:
        parser.error("--grid-points must be >= 100")
    report = {
        "schema": "fp8_fp16_expcast_cpu_accuracy_v2",
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "configuration": {"seed": args.seed, "grid_points": args.grid_points, "gain_fp32": float(GAIN), "base_code": BASE_CODE, "experimental_code_correction": CORRECTION},
        "scope": "CPU accuracy only. No GPU timing, QK quantization, V quantization, or online-softmax accumulation; not an end-to-end attention/model quality measurement.",
        "reference": "FP64 256*exp(x) at identical max-shifted FP32 inputs; FP64 normalization and PV with identical FP32 random V.",
        "variants": {
            "exp_then_fp16": "FP64 exact-function reference weights rounded to FP16 (ordinary quantization baseline).",
            "expcast_raw": "RNE(FMA_FP32(x,1024*log2(e),23552)), clamp to [0,23552], uint16 bitcast to FP16.",
            "expcast_corrected": "Experimental same encoder with bias 23552-44.8. Correction scales FP8's -0.35 by 1024/8; not claimed optimal or a published FP16 implementation.",
            "exp_then_fp8": "FP64 exact-function reference rounded to nearest FP8 E4M3FN, ties to even code.",
            "expcast_fp8_raw": "RNE(FMA_FP32(x,8*log2(e),120)), clamp to [0,120], E4M3FN decode.",
            "expcast_fp8_corrected": "FP8 ExpCast with FP32 bias 119.65, same 256*exp(x) target and identical inputs as FP16.",
        },
        "validation": validate_encoding(),
        "weight_sweep": sweep(args.grid_points),
        "fp8_comparison": compare_fp8(args.grid_points),
        "scalar_examples": scalar_examples(),
        "softmax": softmax_experiments(args.seed),
    }
    print_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
