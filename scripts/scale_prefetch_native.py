"""Carry the next K scale across tiles; original vc28 math and kernel baseline."""
import hashlib
from pathlib import Path
from attention_phase_probe import ROOT, VC, replace_exact
import split_mass_native as base

EXACT_OUTPUT = True


def candidate_kernel():
    source = (VC / "sage_layout.cuh").read_text(encoding="utf-8")
    source = replace_exact(source, "sage_vc_attention_kernel_sage_layout(",
                           "sage_vc_attention_kernel_scale_prefetch(")
    source = replace_exact(source, "  static_assert(NUM_THREADS == 128);",
        "  static_assert(NUM_THREADS == 128);\n"
        "  static_assert(ExpCast && WorkReduction==9 && !FP8QK && !BlockValues && !CompactPV);\n"
        "  static_assert(mask_mode==MaskMode::kNone && !return_lse);")
    source = replace_exact(source, "  float q_scale = Q_scale[q_scale_idx];",
        "  float q_scale = Q_scale[q_scale_idx];\n"
        "  float current_k_scale = K_scale[k_scale_idx];")
    source = replace_exact(source, "q_scale * K_scale[k_scale_idx + (iter - 1) * k_scale_advance_offset]",
                           "q_scale * current_k_scale")
    source = replace_exact(source, "q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset]",
                           "q_scale * current_k_scale")
    # The MAIN loop always has a following tile, including the final masked
    # tile. iter is in [1, num_iterations-1], so the load cannot run past ks.
    # No writes/reads of pending WGMMA P operands or accumulators are added.
    source = replace_exact(source,
        "    wgmma::warpgroup_commit_batch();\n    // Scale history while PV is live;",
        "    wgmma::warpgroup_commit_batch();\n"
        "    // Read the NEXT tile's scale while this PV group is pending.\n"
        "    current_k_scale = K_scale[k_scale_idx + iter * k_scale_advance_offset];\n"
        "    // Scale history while PV is live;")
    return source.replace("#pragma once", "")


def generate_source():
    installed = (VC / "vc_sm90.cu").read_text(encoding="utf-8")
    prefix = installed[:installed.index("template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS")]
    # Retain the EXACT production template signature for the baseline (no
    # extra experiment boolean). Both kernels use the same native wrapper.
    original = (VC / "sage_layout.cuh").read_text(encoding="utf-8").replace("#pragma once", "")
    wrapper = base.generate_source().rsplit("\nnamespace {", 1)[1]
    wrapper = "namespace {" + wrapper
    old = """  return sage_vc::sage_vc_attention_kernel_sage_layout<64,128,128,D,
    QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
    MaskMode::kNone,false,true,Cast,false,false,Cast,0,168,false,
    (Cast ? 9 : 0),false,false,false,Trace>;"""
    args = """<64,128,128,D,
    QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
    MaskMode::kNone,false,true,true,false,false,true,0,168,false,9>;"""
    wrapper = replace_exact(wrapper, old,
        "  static_assert(Cast);\n"
        "  if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scale_prefetch" + args +
        "\n  else return sage_vc::sage_vc_attention_kernel_sage_layout" + args)
    # This experiment exposes only ExpCast. Avoid instantiating unsupported
    # control arms, even though the shared public wrapper accepts a cast flag.
    start = wrapper.index("  if (cast && trace)")
    end = wrapper.index("\n}", start)
    wrapper = wrapper[:start] + """  if (trace) launch_phase<D,Out,true,true>(q,k,v,qs,ks,vs,out,clocks,scale);
  else launch_phase<D,Out,true,false>(q,k,v,qs,ks,vs,out,clocks,scale);""" + wrapper[end:]
    wrapper = replace_exact(wrapper,
        '  return {{"control_plain",resources_for<D,Out,false,false>()},\n'
        '          {"control_probe",resources_for<D,Out,false,true>()},\n'
        '          {"expcast_plain",resources_for<D,Out,true,false>()},',
        '  return {{"expcast_plain",resources_for<D,Out,true,false>()},')
    wrapper = wrapper.replace("Split mass experiment", "Scale prefetch experiment")
    return prefix + '#include "fp8_qk_mma.cuh"\n' + original + candidate_kernel() + "\n}\n" + wrapper


def load_native():
    return base.load_source(generate_source(), "_vc_scale_prefetch_")


def build_audit(ext, vc, build, dim, output_type):
    """Identify actual loaded binaries; source flags do not prove binary parity."""
    def binary(module):
        path = Path(module.__file__).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ninja = Path(build["generated_source"]).with_name("build.ninja")
    flags = [line for line in ninja.read_text(encoding="utf-8").splitlines()
             if line.startswith(("nvcc =", "cuda_cflags =", "cuda_post_cflags =", "cflags ="))]
    result = {"jit_binary": binary(ext), "installed_binary": binary(vc._extension()),
              "jit_build_flags": flags,
              "baseline_source": "Unmodified sage_layout.cuh with original production template signature",
              "baseline_source_sha256": hashlib.sha256((VC / "sage_layout.cuh").read_bytes()).hexdigest(),
              "launch": {"query_tile": 64, "key_tile": 128, "threads": 128,
                         "dynamic_shared_bytes": 320 * dim, "register_limit": 168},
              "limitation": "Installed build flags/source cannot be inferred from its filename/version. "
                            "Use binary hashes, resource reports, paired timings and output checks together."}
    # The installed interface reports D128 resources only; do not mislabel
    # those values as a D64 measurement.
    if dim == 128:
        result["installed_resources"] = vc.kernel_resources(output_type)["expcast_sage_layout_35"]
    return result


forward = base.forward
