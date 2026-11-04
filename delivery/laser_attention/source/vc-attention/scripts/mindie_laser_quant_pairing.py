"""CPU-import-safe, synchronized paired wall-time measurements.

Prepare payloads and check outputs before calling this module. The supplied
zero-argument callables define the entire timed scope: this module does not
preprocess, fingerprint, inspect or retain their outputs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import random
import statistics
import time
from typing import Any


DESCRIPTIVE_CAVEAT = (
    "Descriptive paired wall-time observations only; ranges are observed trial "
    "bounds, not confidence intervals or evidence of statistical significance. "
    "Position balancing does not remove drift, cache effects or carryover."
)


def _count(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _orders(names: list[str], trials: int, seed: int) -> list[list[str]]:
    """Randomized cyclic blocks balance positions, including a partial block."""
    rng = random.Random(seed)
    result = []
    while len(result) < trials:
        base = list(names)
        rng.shuffle(base)
        offsets = list(range(len(base)))
        rng.shuffle(offsets)
        for offset in offsets:
            result.append(base[offset:] + base[:offset])
            if len(result) == trials:
                break
    return result


def run_paired(
    calls: Mapping[str, Callable[[], Any]],
    *,
    synchronize: Callable[[], Any],
    warmup: int = 5,
    repeats: int = 10,
    trials: int = 9,
    seed: int = 2026,
    clock: Callable[[], float] = time.perf_counter,
    on_trial: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Measure each call once per trial, storing milliseconds per invocation.

    Each measurement synchronizes before the start timestamp and after its
    ``repeats`` invocations. Every return value is immediately discarded, before
    the next call or final synchronization. Warmup and ``on_trial(report)`` run
    outside measured intervals. The callback receives the live report after
    every complete trial and should treat it as read-only.

    A seeded random permutation and shuffled cyclic offsets determine each
    block of ``len(calls)`` trial orders. Every call occupies each position once
    per full block; position counts differ by at most one for a partial block.
    This does not balance all possible predecessor/successor pairs.
    """
    _count("warmup", warmup, minimum=0)
    _count("repeats", repeats, minimum=1)
    _count("trials", trials, minimum=1)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(calls, Mapping) or not calls:
        raise ValueError("calls must be a nonempty mapping of names to callables")
    functions = dict(calls)
    for name, call in functions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("call names must be nonempty strings")
        if not callable(call):
            raise ValueError(f"call {name!r} must be callable")
    for name, call in (("synchronize", synchronize), ("clock", clock)):
        if not callable(call):
            raise ValueError(f"{name} must be callable")
    if on_trial is not None and not callable(on_trial):
        raise ValueError("on_trial must be callable or None")

    planned_orders = _orders(list(functions), trials, seed)
    report = {
        "rows": {name: {"trials_ms": []} for name in functions},
        "orders": [],
        "warmup": warmup,
        "repeats": repeats,
        "trials": trials,
        "trial_count": trials,
        "completed_trials": 0,
        "seed": seed,
        "order_strategy": "seeded shuffled cyclic blocks; balanced trial positions",
        "timing_scope": (
            "Synchronized wall time per invocation, including supplied callable "
            "and output release; payload preparation and correctness checks must "
            "be performed separately. Warmup and trial callbacks excluded."
        ),
        "caveat": DESCRIPTIVE_CAVEAT,
    }
    for call in functions.values():
        for _ in range(warmup):
            call()
    synchronize()

    for order in planned_orders:
        for name in order:
            call = functions[name]
            synchronize()
            started = clock()
            for _ in range(repeats):
                call()
            synchronize()
            elapsed_ms = (clock() - started) * 1000 / repeats
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise ValueError(f"{name!r} elapsed milliseconds must be finite and positive")
            row = report["rows"][name]
            row["trials_ms"].append(elapsed_ms)
            row.update(
                median_ms=statistics.median(row["trials_ms"]),
                min_ms=min(row["trials_ms"]),
                max_ms=max(row["trials_ms"]),
            )
        report["orders"].append(order)
        report["completed_trials"] += 1
        if on_trial is not None:
            on_trial(report)
    return report


def paired_comparison(report: Mapping[str, Any], baseline: str, candidate: str) -> dict[str, Any]:
    """Compare aligned trials: positive saved time and ratios > 1 favor candidate."""
    if baseline == candidate:
        raise ValueError("baseline and candidate must name different calls")
    rows = report["rows"]
    old = list(rows[baseline]["trials_ms"])
    new = list(rows[candidate]["trials_ms"])
    if not old or len(old) != len(new):
        raise ValueError("paired rows must contain the same nonzero number of trials")
    orders = report.get("orders", [])
    if len(orders) != len(old) or any(
        order.count(baseline) != 1 or order.count(candidate) != 1 for order in orders
    ):
        raise ValueError("each paired trial must record both calls exactly once in its order")
    if any(not math.isfinite(value) or value <= 0 for value in old + new):
        raise ValueError("paired trial milliseconds must be finite and positive")
    saved = [left - right for left, right in zip(old, new)]
    speedups = [left / right for left, right in zip(old, new)]
    if not all(math.isfinite(value) for value in speedups):
        raise ValueError("paired speedups must be finite")
    return {
        "baseline": baseline,
        "candidate": candidate,
        "trials": len(saved),
        "trial_count": len(saved),
        "baseline_median_ms": statistics.median(old),
        "candidate_median_ms": statistics.median(new),
        "paired_saved_ms": saved,
        "median_saved_ms": statistics.median(saved),
        "min_saved_ms": min(saved),
        "max_saved_ms": max(saved),
        "paired_speedups": speedups,
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "faster_trials": sum(value > 0 for value in saved),
        "slower_trials": sum(value < 0 for value in saved),
        "tied_trials": sum(value == 0 for value in saved),
        "caveat": DESCRIPTIVE_CAVEAT,
    }
