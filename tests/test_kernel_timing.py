"""Timing attribution must retain auxiliary work and reject incomplete CUPTI traces."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kernel_timing import summarize_trace


def span(i, start, duration=1000):
    return dict(cat="user_annotation", ph="X", name=f"vc31_call_{i}", ts=start, dur=duration)


def kernel(name, start, duration, category="kernel"):
    return dict(cat=category, ph="X", name=name, ts=start, dur=duration)


def test_fa4_auxiliary_kernel_and_copy_are_retained_separately():
    events = [span(0, 0), span(1, 1000)]
    for start in (0, 1000):
        events += [kernel("FlashAttentionForward", start+100, 200),
                   kernel("combine", start+400, 30),
                   kernel("copy", start+500, 10, "gpu_memcpy"),
                   kernel("cudaLaunchKernel", start+20, 50, "cuda_runtime")]
    row = summarize_trace(events, 2, ("flash",))
    assert row["attention_median_ms"] == .2
    assert row["all_kernel_median_ms"] == .23
    assert row["all_gpu_work_median_ms"] == .24
    assert row["single_attention_launch"]
    assert [c["kernel_launches"] for c in row["calls"]] == [2, 2]


@pytest.mark.parametrize("events,match", [
    ([span(0, 0)], "Missing attention"),
    ([span(0, 0), kernel("attention", 900, 200)], "Missing attention"),
    ([span(0, 0), kernel("attention", 1, 100), kernel("stray", 1100, 10)], "outside"),
    ([span(0, 0), kernel("attention", 1, 0)], "Nonpositive"),
    ([span(0, 0), kernel("unrecognized", 1, 10)], "Missing attention"),
])
def test_missing_misattributed_or_invalid_gpu_events_fail(events, match):
    with pytest.raises(RuntimeError, match=match):
        summarize_trace(events, 1, ("attention",))


def test_split_attention_is_not_described_as_single_kernel():
    row = summarize_trace([span(0, 0), kernel("attention_split", 10, 100),
                           kernel("attention_combine", 200, 50)], 1, ("attention",))
    assert not row["single_attention_launch"]
    assert row["attention_median_ms"] == .15
