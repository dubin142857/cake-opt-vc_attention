#!/usr/bin/env python3
"""Audited incremental direct-launch builds for quantization, packing and merge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

try:
    from .build_cann_fas_local import read_cache, run_visible
except ImportError:
    from build_cann_fas_local import read_cache, run_visible

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ascend/laser_quant_preprocess"
SOURCE_NAMES = ("CMakeLists.txt", "quantize_contract.h", "quantize_kernel_impl.h",
                "quantize_kernel.cpp", "quantize_bf16_kernel.cpp", "quantize_torch.cpp")
LIBRARY_NAME = "libvc_laser_quant_preprocess.so"
CONTRACT_VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_details():
    if sys.platform != "linux":
        raise ValueError("Build on the remote Linux machine with CANN, torch and torch_npu installed")
    import torch
    import torch_npu
    return {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "python_executable": sys.executable, "python": sys.version,
            "torch_root": str(Path(torch.__file__).resolve().parent),
            "torch_npu_root": str(Path(torch_npu.__file__).resolve().parent),
            "torch_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI)}


def check_cache(path, definitions):
    path = Path(path)
    if not path.exists():
        return
    cache = read_cache(path)
    for key, value in definitions.items():
        if key not in cache or cache[key] != str(value):
            raise ValueError(f"Existing cache mismatch for {key}; preserve it and use another --build-dir")
    for key, value in cache.items():
        if value and ("TOOLCHAIN_FILE" in key or "COMPILER_LAUNCHER" in key or
                      (key.startswith("CMAKE_PROJECT") and "INCLUDE" in key)):
            raise ValueError(f"Unsupported CMake toolchain override {key}; build directory preserved")



def refresh_inplace_linked_objects(build_dir, source_files):
    """CANN 9.2 links device .o in place to ET_EXEC; regenerate ET_REL first.

    Touch only declared entry sources, never headers, shared CANN or host cache.
    File bytes and provenance hashes stay unchanged. Validate all paths first.
    """
    build_dir = Path(build_dir).resolve()
    records, sources = [], set()
    for record in source_files:
        source = Path(record["path"]).resolve()
        if source.suffix != ".cpp":
            continue
        for obj in build_dir.rglob(source.name + ".o"):
            if not any(f"/CMakeFiles/{role}_obj.dir/" in str(obj) for role in ("aic", "aiv")):
                continue
            if not str(obj).endswith(str(source) + ".o"):
                continue
            if obj.is_symlink() or not obj.resolve().is_relative_to(build_dir):
                raise ValueError("Unsafe device object path")
            with obj.open("rb") as stream:
                header = stream.read(20)
            elf_type = (int.from_bytes(header[16:18], "little" if header[5] == 1 else "big")
                        if len(header) == 20 and header[:4] == b"\x7fELF" else None)
            if elf_type != 1:
                records.append(dict(source=str(source), object=str(obj), elf_type=elf_type,
                                    action="refresh declared source timestamp for ET_REL recompilation"))
                sources.add(source)
    for source in sources:
        source.touch()
    return records


def build(cann_root, soc, build_dir, jobs, output, *, runtime=None, runner=None, operation="quantize"):
    report = {"status": "running", "contract_version": CONTRACT_VERSION,
              "torch_api": "torch.ops.vc_laser_quant_preprocess.quantize",
              "scope": "Pure AIV fused per-128-token quantization; no attention or OPP package rebuild"}
    output = Path(output).resolve()
    source, names, library_name = SOURCE, SOURCE_NAMES, LIBRARY_NAME
    if operation in ("sparse_delay", "sparse_pair", "sparse_fma_float", "sparse_fma_half"):
        source = ROOT / ("ascend/laser_"+operation)
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "online_top8.h", "inline_pv.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp", "qk_tile.cpp", "fma_probe.cpp") + (("pair_probe.cpp",) if operation in ("sparse_pair","sparse_delay") else ())
        library_name = "libvc_laser_"+operation+".so"
        report.update(torch_api="torch.ops.vc_laser_"+operation+".produce",scope="Postmax Top8 + FP32 vmla PV; exact half-factor premise independently checked")
    elif operation == "sparse_postmax_scale":
        source = ROOT / "ascend/laser_sparse_postmax_scale"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "online_top8.h", "inline_pv.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp", "qk_tile.cpp")
        library_name = "libvc_laser_sparse_postmax_scale.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_postmax_scale.produce", scope="Raw block max then half scale, window top8 and inline FP32 PV; new tie contract")
    elif operation == "max_scale_probe":
        source = ROOT / "ascend/laser_max_scale_probe"
        names = ("CMakeLists.txt", "max_scale_probe_kernel.cpp", "max_scale_probe_torch.cpp")
        library_name = "libvc_laser_max_scale_probe.so"
        report.update(torch_api="torch.ops.vc_laser_max_scale_probe.run", scope="AIV FP16 scale table and scale/max composition, not attention")
    elif operation == "sparse_inline_pv":
        source = ROOT / "ascend/laser_sparse_inline_pv"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "online_top8.h", "inline_pv.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_inline_pv.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_inline_pv.produce",
                      scope="Block1 window32 resident Top8 + same-AIV FP32 normalization/PV; diagnostic source/pair/packet outputs retained")
    elif operation == "sparse_window_top8":
        source = ROOT / "ascend/laser_sparse_window_top8"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "online_top8.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_window_top8.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_window_top8.produce",
                      scope="Block1 score8 producer: window32 local Top8 then UB-resident merge; full QK remains")
    elif operation == "sparse_online_top8":
        source = ROOT / "ascend/laser_sparse_online_top8"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "online_top8.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_online_top8.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_online_top8.produce",
                      scope="Block1 score8 producer with UB-resident Top8 and direct row-major scores; full QK still required")
    elif operation == "sort32_probe":
        source = ROOT / "ascend/laser_sort32_probe"
        names = ("CMakeLists.txt", "sort32_probe_kernel.cpp", "sort32_probe_torch.cpp")
        library_name = "libvc_laser_sort32_probe.so"
        report.update(torch_api="torch.ops.vc_laser_sort32_probe.sort32",scope="AIV Sort32 float32/uint32 interleaved-data mechanism probe; not attention")
    elif operation == "merge":
        source = ROOT / "ascend/laser_merge"
        names = ("CMakeLists.txt", "merge_kernel_impl.h", "merge_kernel.cpp", "merge_bf16_kernel.cpp", "merge_torch.cpp")
        library_name = "libvc_laser_merge.so"
        report.update(contract_version=2, torch_api="torch.ops.vc_laser_merge.merge",
                      scope="Pure AIV split2 merge + FP16/BF16 conversion + trim; no attention rebuild")
    elif operation == "pack":
        source = ROOT / "ascend/laser_pack"
        names = ("CMakeLists.txt", "pack_kernel_impl.h", "pack_kernel.cpp", "pack_bf16_kernel.cpp", "pack_torch.cpp")
        library_name = "libvc_laser_pack.so"
        report.update(torch_api="torch.ops.vc_laser_pack.pack",
                      scope="Pure AIV split2 FP16 operand packing from BF16/FP16; no attention rebuild")
    elif operation == "sparse_candidates_bulk_output":
        source = ROOT / "ascend/laser_sparse_candidates_bulk_output"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_bulk_output.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_bulk_output.produce",
                      scope="Same block1 per-score handoff; eight candidate columns per two output DMA calls")
    elif operation == "sparse_candidates_score_chunk":
        source = ROOT / "ascend/laser_sparse_candidates_score_chunk"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_score_chunk.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_score_chunk.produce",
                      scope="Block1 eight-score groups, two GM ring slots, MTE2 group readfree and bulk8 output")
    elif operation == "sparse_candidates_score_chunk32":
        source = ROOT / "ascend/laser_sparse_candidates_score_chunk32"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_score_chunk32.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_score_chunk32.produce",
                      scope="Block1 32-score groups, two GM ring slots, MTE2 group readfree and bulk8 output")
    elif operation == "sparse_candidates_read_ack":
        source = ROOT / "ascend/laser_sparse_candidates_read_ack"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_read_ack.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_read_ack.produce",
                      scope="Same block1 attention; score slot released after both MTE2 reads")
    elif operation == "sparse_candidates_block1":
        source = ROOT / "ascend/laser_sparse_candidates_block1"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_block1.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_block1.produce",
                      scope="Full QK, one maximum per K128, global top8; refinement not implemented")
    elif operation == "sparse_candidates_dma_overlap":
        source = ROOT / "ascend/laser_sparse_candidates_dma_overlap"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_dma_overlap.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_dma_overlap.produce",
                      scope="Same block2 lookup attention, first candidate MTE3 overlapped with second reduction")
    elif operation == "sparse_candidates_lookup":
        source = ROOT / "ascend/laser_sparse_candidates_lookup"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates_lookup.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates_lookup.produce",
                      scope="Same block2/top8 contract, exact-position mask lookup instead of full-width broadcast/compare")
    elif operation == "sparse_candidates":
        source = ROOT / "ascend/laser_sparse_candidates"
        names = ("CMakeLists.txt", "candidate_contract.h", "candidate_cube.h", "candidate_selector.h", "candidate_vector.h", "candidate_kernel.cpp", "candidate_probe.cpp", "candidate_diagnostic.cpp", "candidate_torch.cpp")
        library_name = "libvc_laser_sparse_candidates.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_candidates.produce",
                      scope="MIX full QK with two candidates per K128; independent selector probe")
    elif operation == "sparse_pv":
        source = ROOT / "ascend/laser_sparse_pv"
        names = ("CMakeLists.txt", "sparse_pv_kernel_impl.h", "sparse_pv_kernel.cpp", "sparse_pv_torch.cpp")
        library_name = "libvc_laser_sparse_pv.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_pv.weighted",
                      scope="AIV fused sparse V gather and FP32 weighted sum; no attention rebuild")
    elif operation == "sparse_finalize":
        source = ROOT / "ascend/laser_sparse_finalize"
        names = ("CMakeLists.txt", "sparse_finalize_kernel_impl.h", "sparse_finalize_kernel.cpp", "sparse_finalize_torch.cpp")
        library_name = "libvc_laser_sparse_finalize.so"
        report.update(torch_api="torch.ops.vc_laser_sparse_finalize.finalize",
                      scope="AIV fused sparse V gather and FP32 weighted sum; no attention rebuild")
    elif operation == "global":
        source = ROOT / "ascend/laser_global"
        names = ("CMakeLists.txt", "global_contract.h", "global_kernel_impl.h", "global_kernel.cpp", "global_bf16_kernel.cpp", "global_torch.cpp")
        library_name = "libvc_laser_global.so"
        report.update(torch_api="torch.ops.vc_laser_global.attention",
                      scope="MIX Q32 full-K scores/global FP16 softmax/resident FP32 PV; optional exact-zero PV skip")
    elif operation == "global_probe":
        try:
            from . import prepare_mindie_laser_global_probe_source as global_probe
        except ImportError:
            import prepare_mindie_laser_global_probe_source as global_probe
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = (*global_probe.NAMES,"vc_phase_probe.h")
        library_name = "libvc_laser_global_profile.so"
        report.update(torch_api="torch.ops.vc_laser_global_profile.attention",
                      scope="Independent global diagnostic: same arithmetic, sparse serial phase samples; no production edit")
    elif operation == "global_half_sum":
        try:
            from . import prepare_mindie_laser_global_half_sum_source as half_sum
        except ImportError:
            import prepare_mindie_laser_global_half_sum_source as half_sum
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = half_sum.NAMES
        library_name = "libvc_laser_global_half_sum.so"
        report.update(contract_version=2,torch_api="torch.ops.vc_laser_global_half_sum.attention",
                      scope="Independent FP16 chunk rowsum / cross-chunk FP32 attention variant")
    elif operation == "global_pipe":
        try:
            from . import prepare_mindie_laser_global_pipe_source as global_pipe
        except ImportError:
            import prepare_mindie_laser_global_pipe_source as global_pipe
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = global_pipe.NAMES
        library_name = "libvc_laser_global_pipe.so"
        report.update(contract_version=2,torch_api="torch.ops.vc_laser_global_pipe.attention",
                      scope="Independent Cube double-buffer pipeline; half-sum arithmetic unchanged")
    elif operation in ("global_q64", "global_q128"):
        try:
            from . import prepare_mindie_laser_global_geometry_source as global_geometry
        except ImportError:
            import prepare_mindie_laser_global_geometry_source as global_geometry
        q_rows=int(operation.removeprefix("global_q"))
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = global_geometry.NAMES
        library_name = "libvc_laser_" + operation + ".so"
        report.update(contract_version=3,torch_api="torch.ops.vc_laser_"+operation+".attention",
                      geometry=global_geometry.geometry(q_rows),
                      scope="Independent larger-Q Cube pipeline; same per-row half-sum arithmetic")
    elif operation == "global_stream":
        try:
            from . import prepare_mindie_laser_global_stream_source as global_stream
        except ImportError:
            import prepare_mindie_laser_global_stream_source as global_stream
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = global_stream.NAMES
        library_name = "libvc_laser_global_stream.so"
        report.update(contract_version=4,torch_api="torch.ops.vc_laser_global_stream.attention",
                      scope="Q128 bounded K2048 ring; FP32 online output/rowsum recurrence")
    elif operation in ("global_stream_q64", "global_stream_onepass"):
        try:
            from . import prepare_mindie_laser_stream_onepass_source as stream_onepass
        except ImportError:
            import prepare_mindie_laser_stream_onepass_source as stream_onepass
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = stream_onepass.NAMES
        library_name = "libvc_laser_"+operation+".so"
        report.update(contract_version=5,torch_api="torch.ops.vc_laser_"+operation+".attention",
                      geometry=stream_onepass.geometry(operation),scope="Q64 online control/UB resident single pass")
    elif operation == "global_stream_overlap":
        try:
            from . import prepare_mindie_laser_stream_overlap_source as stream_overlap
        except ImportError:
            import prepare_mindie_laser_stream_overlap_source as stream_overlap
        source = Path(build_dir).resolve().with_name(Path(build_dir).name + "_source")
        names = stream_overlap.NAMES
        library_name = "libvc_laser_global_stream_overlap.so"
        report.update(contract_version=6,torch_api="torch.ops.vc_laser_global_stream_overlap.attention",
                      scope="Q128 K2048 two-slot score/P lookahead; unchanged E81 arithmetic")
    report["operation"] = operation
    write_json(output, report)
    try:
        if operation not in ("quantize", "merge", "pack", "global", "global_probe", "global_half_sum", "global_pipe", "global_q64", "global_q128", "global_stream", "global_stream_q64", "global_stream_onepass", "global_stream_overlap", "sparse_pv", "sparse_candidates", "sparse_candidates_lookup", "sparse_candidates_dma_overlap", "sparse_candidates_block1", "sparse_candidates_read_ack", "sparse_candidates_bulk_output", "sparse_candidates_score_chunk", "sparse_candidates_score_chunk32", "sparse_finalize", "sparse_delay", "sparse_pair", "sparse_fma_float", "sparse_fma_half", "sparse_postmax_scale", "max_scale_probe", "sort32_probe", "sparse_online_top8", "sparse_window_top8", "sparse_inline_pv"):
            raise ValueError("Unsupported direct-launch operation")
        if operation in ("global", "global_probe", "global_half_sum", "global_pipe", "global_q64", "global_q128", "global_stream", "global_stream_q64", "global_stream_onepass", "global_stream_overlap", "sparse_pv", "sparse_candidates", "sparse_candidates_lookup", "sparse_candidates_dma_overlap", "sparse_candidates_block1", "sparse_candidates_read_ack", "sparse_candidates_bulk_output", "sparse_candidates_score_chunk", "sparse_candidates_score_chunk32", "sparse_finalize", "sparse_delay", "sparse_pair", "sparse_fma_float", "sparse_fma_half", "sparse_postmax_scale", "max_scale_probe", "sort32_probe", "sparse_online_top8", "sparse_window_top8", "sparse_inline_pv") and soc != "Ascend910B2":
            raise ValueError("Global attention currently targets exact Ascend910B2")
        if not re.fullmatch(r"Ascend910(?:B(?:[1-4]|2C)|_93[0-9]{2})", soc):
            raise ValueError("--soc requires the exact A2/A3 runtime SoC, e.g. Ascend910_9382")
        if jobs < 1:
            raise ValueError("--jobs must be positive")
        for key in ("CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
                    "CMAKE_CXX_COMPILER_LAUNCHER"):
            if os.environ.get(key):
                raise ValueError(f"Unsupported inherited CMake override {key}")
        runtime = runtime_details() if runtime is None else runtime
        runner = run_visible if runner is None else runner
        cann = Path(cann_root).resolve()
        build_dir = Path(build_dir).resolve()
        if operation == "global_probe":
            report["generator_inputs"] = global_probe.prepare(source)
        elif operation == "global_half_sum":
            report["generator_inputs"] = half_sum.prepare(source)
        elif operation == "global_pipe":
            report["generator_inputs"] = global_pipe.prepare(source)
        elif operation in ("global_q64", "global_q128"):
            report["generator_inputs"] = global_geometry.prepare(source,q_rows)
        elif operation == "global_stream":
            report["generator_inputs"] = global_stream.prepare(source)
        elif operation in ("global_stream_q64", "global_stream_onepass"):
            report["generator_inputs"] = stream_onepass.prepare(source,operation)
        elif operation == "global_stream_overlap":
            report["generator_inputs"] = stream_overlap.prepare(source)
        definitions = {
            "ASCEND_CANN_PACKAGE_PATH": str(cann), "SOC_VERSION": soc,
            "Python3_EXECUTABLE": runtime["python_executable"],
            "CMAKE_HOME_DIRECTORY": str(source), "CMAKE_GENERATOR": "Unix Makefiles",
            "CMAKE_BUILD_TYPE": "Release", "VC_TORCH_ROOT": runtime["torch_root"],
            "VC_TORCH_NPU_ROOT": runtime["torch_npu_root"],
            "VC_TORCH_VERSION": runtime["torch"], "VC_TORCH_NPU_VERSION": runtime["torch_npu"],
            "VC_TORCH_ABI": str(runtime["torch_abi"]),
        }
        check_cache(build_dir / "CMakeCache.txt", definitions)
        source_files = [{"path": str(source / name), "sha256": sha(source / name)} for name in names]
        report.update(soc=soc, cann_root=str(cann), build_dir=str(build_dir),
                      runtime=runtime, definitions=definitions, source_files=source_files)
        write_json(output, report)
        build_dir.mkdir(parents=True, exist_ok=True)
        configure_definitions = {k: v for k, v in definitions.items()
                                 if k not in ("CMAKE_HOME_DIRECTORY", "CMAKE_GENERATOR")}
        configure = ["cmake", "-S", str(source), "-B", str(build_dir), "-G", "Unix Makefiles",
                     *[f"-D{k}={v}" for k, v in configure_definitions.items()]]
        compile_ = ["cmake", "--build", str(build_dir), "--parallel", str(jobs)]
        report["commands"] = [configure, compile_]
        write_json(output, report)
        phase = "quant_preprocess" if operation == "quantize" else operation
        runner(configure, ROOT, os.environ.copy(), phase + "_configure")
        report["device_object_refresh"] = refresh_inplace_linked_objects(build_dir, source_files)
        write_json(output, report)
        runner(compile_, ROOT, os.environ.copy(), phase + "_build")
        library = build_dir / library_name
        if not library.is_file():
            raise RuntimeError("Build did not produce fused preprocessing library")
        if any(sha(record["path"]) != record["sha256"] for record in source_files):
            raise RuntimeError("Native sources changed during compilation; rerun the incremental build")
        if operation == "global_probe":
            global_probe.validate(source,report["generator_inputs"])
        elif operation == "global_half_sum":
            half_sum.validate(source,report["generator_inputs"])
        elif operation == "global_pipe":
            global_pipe.validate(source,report["generator_inputs"])
        elif operation in ("global_q64", "global_q128"):
            global_geometry.validate(source,report["generator_inputs"],q_rows)
        elif operation == "global_stream":
            global_stream.validate(source,report["generator_inputs"])
        elif operation in ("global_stream_q64", "global_stream_onepass"):
            stream_onepass.validate(source,report["generator_inputs"],operation)
        elif operation == "global_stream_overlap":
            stream_overlap.validate(source,report["generator_inputs"])
        report.update(status="passed", library_path=str(library), library_sha256=sha(library))
        write_json(output, report)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cann-root", "--toolkit", dest="cann_root", type=Path, required=True)
    result.add_argument("--soc", required=True)
    result.add_argument("--operation", choices=("quantize","merge","pack","global","global_probe","global_half_sum","global_pipe","global_q64","global_q128","global_stream","global_stream_q64","global_stream_onepass","global_stream_overlap","sparse_pv","sparse_candidates","sparse_candidates_lookup","sparse_candidates_dma_overlap", "sparse_candidates_block1", "sparse_candidates_read_ack", "sparse_candidates_bulk_output", "sparse_candidates_score_chunk", "sparse_candidates_score_chunk32", "sparse_finalize", "sparse_delay", "sparse_pair", "sparse_fma_float", "sparse_fma_half", "sparse_postmax_scale", "max_scale_probe", "sort32_probe", "sparse_online_top8", "sparse_window_top8", "sparse_inline_pv"), default="quantize")
    result.add_argument("--build-dir", type=Path, default=ROOT / "build/mindie_laser_quant_preprocess")
    result.add_argument("--jobs", type=int, default=4)
    result.add_argument("--output", type=Path, default=ROOT / "results/mindie_laser_quant_preprocess_build.json")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = build(args.cann_root, args.soc, args.build_dir, args.jobs, args.output, operation=args.operation)
        print(f"Direct-launch {args.operation}:", report["library_path"], flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
