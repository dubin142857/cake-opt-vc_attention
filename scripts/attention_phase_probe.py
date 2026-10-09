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
DETAIL_PHASES = ("max_and_alpha", "gain_range_check", "bias_setup",
                 "encode_pack", "fp8_decode", "fp32_mass_sum")
DETAIL_WIDTH = 28  # 0..9 coarse clocks, 10..26 detail clocks, 27 route tag.


def replace_exact(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f"Kernel source changed: expected {count} phase anchors for {old!r}")
    return text.replace(old, new)


def instrument_loop(source, probability_detail=False):
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
    if probability_detail:
        for tile, tag in (("iter - 1", "0"), ("num_iterations - 1", "4")):
            source = replace_exact(source, f"    probe.start({tile});",
                f"    probe.start({tile});\n    probe.value(27, ExpCast ? {tag} : 3);")
        old = ("        vc_probability_prepare_integer<(WorkReduction==6),carry_mass,(WorkReduction>=9),(WorkReduction==10)>"
               "(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass);")
        source = replace_exact(source, old,
            "        {\n          if constexpr (PhaseTrace && WorkReduction==9)\n"
            "            vc_probability_prepare_integer_detail<false,false,true,false>"
            "(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass,probe);\n"
            "          else\n" + old + "\n        }")
    return source.replace("#pragma once", "")


def instrument_probability_helpers():
    """Copy exact helpers, retaining encode/decode/sum interleaving per 2-fragment group."""
    tile = (VC / "streamed_probability.cuh").read_text(encoding="utf-8")
    tile = tile[tile.index("template<bool InterleaveRows=false, uint32_t NK>"):]
    tile = replace_exact(tile, "template<bool InterleaveRows=false, uint32_t NK>",
                         "template<bool InterleaveRows=false, uint32_t NK, typename Probe>")
    tile = tile.replace("vc_probability_int_affine_tile(", "vc_probability_int_affine_tile_detail(")
    tile = replace_exact(tile, "float (&mass)[2]) {", "float (&mass)[2], Probe& probe) {")
    tile = replace_exact(tile, "  float sums[2][2]={};", "  float sums[2][2]={};\n  probe.mark(13);")
    anchor = "    #pragma unroll\n    for (int k=0;k<2;++k)\n"
    if tile.count(anchor) != 2:
        raise RuntimeError("Probability decode/sum anchors changed")
    tile = tile.replace(anchor, "    probe.mark(14+(base/2)*3);\n"+anchor, 1)
    # The second k loop consumes decoded half2 values and updates FP32 sums.
    pos = tile.rindex(anchor)
    tile = tile[:pos] + "    probe.mark(15+(base/2)*3);\n" + tile[pos:]
    tile = replace_exact(tile, "  }\n  mass[0]=", "    probe.mark(16+(base/2)*3);\n  }\n  mass[0]=")
    tile = replace_exact(tile, "  mass[1]=__fadd_rn(sums[1][0],sums[1][1]);",
                         "  mass[1]=__fadd_rn(sums[1][0],sums[1][1]);\n  probe.mark(26);")

    work = (VC / "work_reduction.cuh").read_text(encoding="utf-8")
    start = work.index("template<bool BitEncode=false")
    work = work[start:work.index("template<bool SkipIdentity", start)]
    work = replace_exact(work, "uint32_t NQ, uint32_t NK>", "uint32_t NQ, uint32_t NK, typename Probe>")
    work = work.replace("vc_probability_prepare_integer(", "vc_probability_prepare_integer_detail(")
    work = replace_exact(work, "float (&mass)[NQ][2]) {",
        "float (&mass)[NQ][2], Probe& probe) {\n"
        "  static_assert(NQ==1 && NK==8 && StreamConvert && !InterleaveRows);\n  probe.mark(10);")
    work = replace_exact(work, "    if constexpr (StreamConvert) {", "    probe.mark(11);\n    if constexpr (StreamConvert) {")
    work = replace_exact(work, "      if (__all_sync(0xffffffff,safe)) {",
        "      bool regular=__all_sync(0xffffffff,safe);\n      probe.mark(12);\n"
        "      if (regular) {\n        probe.value(27,1);")
    work = replace_exact(work,
        "        vc_probability_int_affine_tile<InterleaveRows>(scores[q],maximum[q],gain,packed[q],mass[q]);",
        "        vc_probability_int_affine_tile_detail<InterleaveRows>(scores[q],maximum[q],gain,packed[q],mass[q],probe);")
    work = replace_exact(work, "      } else {\n        float converted",
        "      } else {\n        probe.value(27,2);\n        float converted")
    work = replace_exact(work,
        "        vc_probability_variant_tile<2,2,false,false>(converted,maximum[q],scale,packed[q],mass[q]);\n      }",
        "        vc_probability_variant_tile<2,2,false,false>(converted,maximum[q],scale,packed[q],mass[q]);\n"
        "        probe.mark(26);\n      }")
    return tile + "\n" + work


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
        static_cast<uint64_t*>(ptr)+(cta/PHASE_CTA_STRIDE)*slots*PHASE_RECORD_WIDTH : nullptr;
    }
  }
  __device__ __forceinline__ void start(uint32_t k) {
    if constexpr (Enabled) {
      tile=(base && (k%PHASE_TILE_STRIDE==0 || k==last)) ?
        base+((k+PHASE_TILE_STRIDE-1)/PHASE_TILE_STRIDE)*PHASE_RECORD_WIDTH : nullptr;
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
  __device__ __forceinline__ void value(int index, uint64_t v) {
    if constexpr (Enabled) {
      if (tile) reinterpret_cast<volatile uint64_t*>(tile)[index]=v;
    }
  }
};
'''


def generate_source(cta_stride=64, tile_stride=8, probability_detail=False):
    if min(cta_stride, tile_stride) < 1:
        raise ValueError("Sampling strides must be positive")
    native = (VC / "vc_sm90.cu").read_text(encoding="utf-8")
    anchor = "template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS"
    prefix = native[:native.index(anchor)]  # Original headers, namespace and TMA helpers.
    kernel = instrument_loop((VC / "sage_layout.cuh").read_text(encoding="utf-8"), probability_detail)
    wrapper = Path(__file__).with_name("attention_phase_probe.cu").read_text(encoding="utf-8")
    return (f"#define PHASE_CTA_STRIDE {cta_stride}\n#define PHASE_TILE_STRIDE {tile_stride}\n" +
            f"#define PHASE_RECORD_WIDTH {DETAIL_WIDTH if probability_detail else 10}\n" +
            prefix + '#include "fp8_qk_mma.cuh"\n' + RECORDER +
            (instrument_probability_helpers() if probability_detail else "") + kernel +
            "\n} // namespace sage_vc\n" + wrapper)


def load_probe(cta_stride=64, tile_stride=8, probability_detail=False):
    from torch.utils.cpp_extension import load, _get_build_directory
    source = generate_source(cta_stride, tile_stride, probability_detail)
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
                 "cta_stride": cta_stride, "tile_stride": tile_stride,
                 "probability_detail": probability_detail}


def summarize_probability_clocks(clocks, keys, tile_stride, expcast):
    """Sum interleaved groups PER SAMPLE before any median; tag fallback and tail."""
    if clocks.ndim != 3 or clocks.shape[2] != DETAIL_WIDTH:
        raise ValueError("Expected 28-field probability probe records")
    coarse = summarize_clocks(clocks[..., :10], keys, tile_stride)
    tags = clocks[..., 27]
    if not expcast:
        if not (tags==3).all() or (clocks[..., 10:27]!=0).any():
            raise RuntimeError("Control unexpectedly entered detailed ExpCast helper")
        coarse["probability_detail"] = {"status": "not_applicable_control"}
        return coarse
    if not (tags[:, -1]==4).all() or (clocks[:, -1, 10:27]!=0).any():
        raise RuntimeError("Final masked tile must stay on its original probability path")
    main_tags = tags[:, :-1]
    if not ((main_tags==1) | (main_tags==2)).all():
        raise RuntimeError("Missing/unknown detailed probability route tag")
    main = clocks[:, :-1]
    regular, fallback = main[main_tags==1], main[main_tags==2]
    # Fallback has max/range/end clocks only; never silently count it as regular.
    if fallback.numel():
        f = fallback[:, [3,10,11,12,26,4]]
        if ((f[:, 1:]-f[:, :-1])<=0).any() or (fallback[:,13:26]!=0).any():
            raise RuntimeError("Incomplete fallback timing record")
    detail = {"regular_samples": len(regular), "fallback_samples": len(fallback),
              "masked_tail_samples": len(tags), "phases": {},
              "scope": "Sampled thread-0 main tiles only. Encode/decode/sum stay interleaved in four groups. "
                       "Intervals include clock/store/scheduling overhead; percentages are within this "
                       "instrumented helper, never shares of production kernel ms. Max includes alpha exp. "
                       "Masked tail and guarded fallback are not attributed to regular encoding."}
    if len(regular):
        ticks = regular[:, 10:27]
        diffs = ticks[:, 1:]-ticks[:, :-1]
        if (diffs<=0).any() or (ticks[:,0]<=regular[:,3]).any() or (ticks[:,-1]>=regular[:,4]).any():
            raise RuntimeError("Missing/nonmonotonic probability detail timestamps")
        values = torch.stack((diffs[:,0], diffs[:,1], diffs[:,2],
                              diffs[:,[3,6,9,12]].sum(1), diffs[:,[4,7,10,13]].sum(1),
                              diffs[:,[5,8,11,14,15]].sum(1)), dim=1).double()
        total = (ticks[:,-1]-ticks[:,0]).double()
        if not torch.equal(values.sum(1),total):
            raise AssertionError("Probability interval accounting mismatch")
        detail["phases"] = {name: {
            "median_cycles": values[:,i].quantile(.5).item(),
            "p10_cycles": values[:,i].quantile(.1).item(),
            "p90_cycles": values[:,i].quantile(.9).item(),
            "sampled_helper_share_percent": (100*values[:,i].sum()/total.sum()).item()}
            for i,name in enumerate(DETAIL_PHASES)}
        detail["helper_median_cycles"] = total.quantile(.5).item()
    coarse["probability_detail"] = detail
    return coarse


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
