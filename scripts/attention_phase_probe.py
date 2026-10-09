"""Build a diagnostic copy of the actual vc28 loop; never edit the installed kernel.

clock64 intervals are sampled thread elapsed cycles, including scheduling and
instrumentation. They are NOT tensor/SFU utilization or additive kernel ms.
The non-instrumented copy is compiled alongside it to expose compiler drift.
"""
import hashlib
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VC = ROOT / "third_party/sageattention_local/csrc/vc"
PHASES = (
    "k_scale_and_wait", "qk_mma_issue_wait", "next_k_prefetch",
    "probability", "v_wait", "pv_issue", "history_during_pv",
    "pv_wait_remaining", "accumulate_next_v",
)


def replace_exact(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f"Kernel source changed: expected {count} phase anchors for {old!r}")
    return text.replace(old, new)


def instrument_loop(source):
    source = source.replace("\r\n", "\n")
    source = replace_exact(source, "bool PreloadMeans=false>", "bool PreloadMeans=false, bool PhaseTrace=false>")
    source = replace_exact(source, "  static_assert(NUM_THREADS == 128);",
        "  static_assert(NUM_THREADS == 128);\n"
        "  static_assert(!BlockValues && !FP8QK && !return_lse && !CompactPV);\n"
        "  PhaseRecorder<PhaseTrace> probe(auxiliary, kv_len);")
    source = replace_exact(source, "    p ^= 1;", "    probe.start(iter - 1);\n    p ^= 1;", 2)
    # The final masked block is outside the iter loop.
    tail = source.index("(num_iterations - 1) * k_scale_advance_offset")
    before, after = source[:tail], source[tail:]
    pos = before.rfind("probe.start(iter - 1)")
    before = before[:pos] + before[pos:].replace("probe.start(iter - 1)", "probe.start(num_iterations - 1)", 1)
    source = before + after
    source = replace_exact(source, "    wait(&barrier_K, p);", "    wait(&barrier_K, p);\n    probe.mark(1);", 2)
    source = replace_exact(source, "    wgmma::warpgroup_wait<0>();", "    wgmma::warpgroup_wait<0>();\n    probe.mark(2);", 4)
    # Distinguish the two QK waits from the two PV waits.
    source = replace_exact(source, "    // Integer-max candidates", "    probe.mark(3);\n    // Integer-max candidates")
    source = replace_exact(source, "    // convert RS to float", "    probe.mark(3);\n    // convert RS to float")
    source = replace_exact(source, "    // wait for V\n    wait(&barrier_V, p);",
        "    probe.mark(4);\n    // wait for V\n    wait(&barrier_V, p);\n    probe.mark(5);", 2)
    source = replace_exact(source, "    wgmma::warpgroup_commit_batch();",
        "    wgmma::warpgroup_commit_batch();\n    probe.mark(6);", 4)
    # QK commits must not record the PV marker.
    source = replace_exact(source, "    probe.mark(6);\n    if constexpr (WorkReduction==8)",
                           "    if constexpr (WorkReduction==8)", 2)
    for end in ("iter-1", "num_iterations-1"):
        anchor = f"head_id,{end},div_ceil(kv_len,CTA_K),head_dim,lane_id);\n    wgmma::warpgroup_wait<0>();\n    probe.mark(2);"
        source = replace_exact(source, anchor,
            f"head_id,{end},div_ceil(kv_len,CTA_K),head_dim,lane_id);\n"
            "    probe.mark(7);\n    wgmma::warpgroup_wait<0>();\n    probe.mark(8);")
    source = replace_exact(source,
        "      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);\n    }\n  }",
        "      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);\n    }\n    probe.mark(9);\n  }")
    source = replace_exact(source, "    if constexpr (carry_mass) vc_probability_drain_mass(RS_f8,probability_alpha,d);",
        "    if constexpr (carry_mass) vc_probability_drain_mass(RS_f8,probability_alpha,d);\n    probe.mark(9);")
    return source.replace("#pragma once", "")


RECORDER = r'''
// Only one lane per sampled CTA records; no barriers or waits are added.
template<bool Enabled> struct PhaseRecorder {
  uint64_t* base;
  uint64_t* tile;
  uint32_t last;
  __device__ __forceinline__ PhaseRecorder(void* ptr, uint32_t keys) {
    if constexpr (Enabled) {
      last=(keys+127)/128-1;
      const uint32_t slots=(last+PHASE_TILE_STRIDE-1)/PHASE_TILE_STRIDE+1;
      const uint32_t cta=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
      base=(threadIdx.x==0 && cta%PHASE_CTA_STRIDE==0) ?
        static_cast<uint64_t*>(ptr)+(cta/PHASE_CTA_STRIDE)*slots*10 : nullptr;
    }
  }
  __device__ __forceinline__ void start(uint32_t k) {
    if constexpr (Enabled) {
      tile=(base && (k%PHASE_TILE_STRIDE==0 || k==last)) ?
        base+((k+PHASE_TILE_STRIDE-1)/PHASE_TILE_STRIDE)*10 : nullptr;
      mark(0);
    }
  }
  __device__ __forceinline__ void mark(int index) {
    if constexpr (Enabled) {
      if (tile) {
        uint64_t tick;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(tick) :: "memory");
        // The store/check cost appears in the following interval.
        reinterpret_cast<volatile uint64_t*>(tile)[index]=tick;
      }
    }
  }
};
'''


def generate_source(cta_stride=64, tile_stride=8):
    if min(cta_stride, tile_stride) < 1:
        raise ValueError("Sampling strides must be positive")
    native = (VC / "vc_sm90.cu").read_text(encoding="utf-8")
    anchor = "template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS"
    prefix = native[:native.index(anchor)]  # Original headers, namespace and TMA helpers.
    kernel = instrument_loop((VC / "sage_layout.cuh").read_text(encoding="utf-8"))
    wrapper = Path(__file__).with_name("attention_phase_probe.cu").read_text(encoding="utf-8")
    return (f"#define PHASE_CTA_STRIDE {cta_stride}\n#define PHASE_TILE_STRIDE {tile_stride}\n" +
            prefix + '#include "fp8_qk_mma.cuh"\n' + RECORDER + kernel +
            "\n} // namespace sage_vc\n" + wrapper)


def load_probe(cta_stride=64, tile_stride=8):
    from torch.utils.cpp_extension import load, _get_build_directory
    source = generate_source(cta_stride, tile_stride)
    # Hash transitive local headers too; torch's JIT cache need not notice them.
    digest = hashlib.sha256(source.encode())
    for path in sorted((VC.parent).rglob("*.cuh")):
        digest.update(path.read_bytes())
    key = digest.hexdigest()[:16]
    name = "_vc_expcast_phases_" + key
    directory = Path(_get_build_directory(name, verbose=False))
    path = directory / "probe.cu"
    path.write_text(source, encoding="utf-8")
    os.environ.setdefault("MAX_JOBS", "2")
    ext = load(name=name, sources=[str(path)], extra_include_paths=[str(VC)],
               extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                                  "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
               extra_ldflags=["cuda.lib"] if os.name == "nt" else ["-lcuda"], verbose=True)
    return ext, {"source_hash": key, "generated_source": str(path),
                 "cta_stride": cta_stride, "tile_stride": tile_stride}


def summarize_clocks(clocks, keys, tile_stride):
    """Aggregate sampled thread intervals; never convert cycles into global ms."""
    if clocks.ndim != 3 or clocks.shape[2] != 10 or clocks.shape[0] < 1:
        raise ValueError("Expected [sampled CTAs, sampled tiles, 10] timestamps")
    tiles = sorted(set(range(0, (keys+127)//128, tile_stride)) | {((keys+127)//128)-1})
    if clocks.shape[1] != len(tiles) or (clocks <= 0).any():
        raise RuntimeError("Missing phase timestamps; refusing partial diagnosis")
    delta = clocks[..., 1:] - clocks[..., :-1]
    if (delta <= 0).any():
        raise RuntimeError("Nonmonotonic phase timestamps")
    def group(values):
        if not values.numel():
            return None
        totals = values.double().sum(dim=(0, 1))
        flat = values.reshape(-1, len(PHASES)).double()
        return {"samples": len(flat), "phases": {
            phase: {"median_cycles": flat[:, i].quantile(.5).item(),
                    "p10_cycles": flat[:, i].quantile(.1).item(),
                    "p90_cycles": flat[:, i].quantile(.9).item(),
                    "sampled_elapsed_share_percent": (100*totals[i]/totals.sum()).item()}
            for i, phase in enumerate(PHASES)}}
    return {"main_tiles": group(delta[:, :-1]), "final_masked_tile": group(delta[:, -1:]),
            "sampled_ctas": clocks.shape[0], "sampled_tile_indices": tiles,
            "scope": "Thread-0 elapsed cycles in an instrumented kernel, NOT SM utilization or "
                     "global kernel-time percentages. Stores/scheduling/compiler motion perturb intervals. "
                     "PV execution overlaps history; pv_wait_remaining is only the exposed remainder. "
                     "Prologue/epilogue are excluded; never multiply shares by production kernel ms."}
