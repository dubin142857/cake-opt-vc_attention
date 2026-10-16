"""Host checks only: no claim of CANN compilation or device timing."""
import json
from pathlib import Path
import re

import pytest

from scripts import build_mindie_laser_quant_preprocess as builder
from scripts.mindie_laser_quant_fused import validate_preprocess_manifest
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax
from tests import cann901_host_stub_fixture as cann_stub


def runtime(tmp_path):
    return {"python_executable": str(tmp_path / "python3"), "python": "fixture",
            "torch": "2.10.0", "torch_npu": "2.10.0.post2",
            "torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
            "torch_abi": 1}


def test_build_manifest_proves_small_native_library_and_sources(tmp_path):
    directory = tmp_path / "build"
    report_path = tmp_path / "manifest.json"
    calls = []
    def run(command, cwd, env, phase):
        calls.append((command, phase))
        if phase == "quant_preprocess_build":
            (directory / builder.LIBRARY_NAME).write_bytes(b"fixture library")
    report = builder.build(tmp_path / "cann", "Ascend910_9382", directory, 4,
                           report_path, runtime=runtime(tmp_path), runner=run)
    assert report["status"] == "passed"
    assert report["contract_version"] == 1
    assert report["soc"] == "Ascend910_9382"
    assert report["library_sha256"] == builder.sha(report["library_path"])
    assert {Path(r["path"]).name for r in report["source_files"]} == set(builder.SOURCE_NAMES)
    assert all(r["sha256"] == builder.sha(r["path"]) for r in report["source_files"])
    assert all("package" not in call[0] and "--clean-first" not in call[0] for call in calls)
    assert calls[1][0][-2:] == ["--parallel", "4"]
    assert json.loads(report_path.read_text())["status"] == "passed"
    evidence = validate_preprocess_manifest(report_path)
    assert evidence["library_sha256"] == report["library_sha256"]
    assert evidence["soc"] == report["soc"]


def test_cache_reuse_and_different_soc_are_explicit(tmp_path):
    path = tmp_path / "CMakeCache.txt"
    expected = {"SOC_VERSION": "Ascend910_9382", "VC_TORCH_VERSION": "2.10.0"}
    builder.check_cache(path, expected)
    path.write_text("".join(f"{key}:STRING={value}\n" for key, value in expected.items()))
    builder.check_cache(path, expected)
    with pytest.raises(ValueError, match="SOC_VERSION"):
        builder.check_cache(path, {**expected, "SOC_VERSION": "Ascend910B2"})
    with pytest.raises(ValueError, match="VC_TORCH_ABI"):
        builder.check_cache(path, {**expected, "VC_TORCH_ABI": "0"})
    assert path.exists()


@pytest.mark.parametrize("soc", ["ascend910_93", "910B_23", "Ascend950PR", "Ascend910B"])
def test_wrong_soc_fails_before_compiler_and_records_failure(tmp_path, soc):
    report = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="exact A2/A3"):
        builder.build(tmp_path, soc, tmp_path / "build", 4, report,
                      runtime=runtime(tmp_path), runner=lambda *args: pytest.fail("Compiler started"))
    assert json.loads(report.read_text())["status"] == "failed"


def test_missing_library_never_reuses_passed_manifest(tmp_path):
    report = tmp_path / "manifest.json"
    report.write_text('{"status":"passed"}')
    with pytest.raises(RuntimeError, match="did not produce"):
        builder.build(tmp_path, "Ascend910B2", tmp_path / "build", 1, report,
                      runtime=runtime(tmp_path), runner=lambda *args: None)
    assert json.loads(report.read_text())["status"] == "failed"


def test_failed_compiler_preserves_existing_build_files(tmp_path):
    directory = tmp_path / "build"
    directory.mkdir()
    old = directory / "keep_this.o"
    old.write_bytes(b"existing cache")
    def run(*args):
        raise RuntimeError("compiler fixture error")
    with pytest.raises(RuntimeError, match="fixture"):
        builder.build(tmp_path, "Ascend910_9382", directory, 4, tmp_path / "manifest.json",
                      runtime=runtime(tmp_path), runner=run)
    assert old.read_bytes() == b"existing cache"


def test_cli_defaults_and_toolkit_alias():
    args = builder.parser().parse_args(["--toolkit", "/toolkit", "--soc", "Ascend910_9382"])
    assert args.build_dir.name == "mindie_laser_quant_preprocess"
    assert args.output.name == "mindie_laser_quant_preprocess_build.json"
    assert args.jobs == 4 and args.cann_root == Path("/toolkit")


@pytest.mark.parametrize("sequence,heads", [(1,1),(129,3),(4097,2),(37741,28)])
@pytest.mark.parametrize("cores", [1,24,48])
def test_grouped_scale_writes_have_no_shared_dma_block_or_missing_quant_block(sequence, heads, cores):
    padded = (sequence + 255) // 256 * 256
    tasks = heads * (padded // 128)
    groups = (tasks + 7) // 8
    writers = {}
    quant_blocks = set()
    for core in range(min(cores, groups)):
        for group in range(core, groups, min(cores, groups)):
            for task in range(group * 8, min((group + 1) * 8, tasks)):
                assert task not in quant_blocks
                quant_blocks.add(task)
                # A physical 32-byte scale line belongs to exactly one core.
                assert writers.setdefault(task // 8, core) == core
                head, tile = divmod(task, padded // 128)
                assert 0 <= head < heads and 0 <= tile * 128 < padded
    assert quant_blocks == set(range(tasks))


STUBS = r"""
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
struct half { uint16_t bits; explicit half(float); };
struct bfloat16_t { uint16_t bits; };
#define __aicore__
#define __global__
#define __gm__
#define GM_ADDR uint8_t*
#define KERNEL_TASK_TYPE_DEFAULT(type) static_assert(type == 1, "Pure AIV required")
#define KERNEL_TYPE_AIV_ONLY 1
namespace AscendC {
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class ReduceOrder { ORDER_ONLY_VALUE };
enum class HardEvent { V_MTE2, MTE2_V, V_S, S_V, V_MTE3, MTE3_S, S_MTE3 };
enum { PIPE_V };
template<class T> struct LocalTensor {
  LocalTensor operator[](uint64_t) const;
  T GetValue(uint32_t) const;
  void SetValue(uint32_t, T) const;
};
template<class T> struct GlobalTensor {
  void SetGlobalBuffer(T*);
  GlobalTensor operator[](uint64_t) const;
};
template<TPosition P> struct TBuf { template<class T> LocalTensor<T> Get(); };
struct TPipe {
  template<TPosition P> void InitBuffer(TBuf<P>&, uint32_t);
  int FetchEventID(HardEvent);
};
struct DataCopyParams { uint16_t blockCount, blockLen, srcStride, dstStride; };
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, reserved; };
template<class T> void Duplicate(LocalTensor<T>, T, uint32_t);
template<class T> void DataCopy(LocalTensor<T>, GlobalTensor<T>, DataCopyParams);
template<class T> void DataCopy(GlobalTensor<T>, LocalTensor<T>, uint32_t);
template<class T> void DataCopyPad(GlobalTensor<T>, LocalTensor<T>, DataCopyExtParams);
template<class T,class U> void Cast(LocalTensor<T>, LocalTensor<U>, RoundMode, uint32_t);
template<class T> void Abs(LocalTensor<T>, LocalTensor<T>, uint32_t);
template<class T> void WholeReduceMax(LocalTensor<T>, LocalTensor<T>, int, int, int, int, int, ReduceOrder);
template<class T> void Div(LocalTensor<T>, LocalTensor<T>, LocalTensor<T>, uint32_t);
template<class T> void Maxs(LocalTensor<T>, LocalTensor<T>, T, uint32_t);
template<class T> void Mins(LocalTensor<T>, LocalTensor<T>, T, uint32_t);
template<HardEvent> void SetFlag(int);
template<HardEvent> void WaitFlag(int);
template<int> void PipeBarrier();
uint32_t GetBlockIdx();
}
"""


@pytest.mark.parametrize("entry_file", ["quantize_kernel.cpp", "quantize_bf16_kernel.cpp"])
def test_actual_kernel_instantiates_both_dtypes_with_host_cpp_declarations(cpp_compiler, tmp_path, entry_file):
    # Compiles the actual source, not a separately maintained imitation. Stubs
    # catch C++ typing/templates but cannot verify Ascend API availability.
    contract = (builder.SOURCE / "quantize_contract.h").read_text()
    implementation = (builder.SOURCE / "quantize_kernel_impl.h").read_text()
    source = (builder.SOURCE / entry_file).read_text()
    stripped = "\n".join(line for line in (contract + "\n" + implementation + "\n" + source).splitlines()
                         if not line.startswith(("#include", "#pragma")))
    code, diagnostic = compile_syntax(cpp_compiler, tmp_path, STUBS + stripped)
    assert code == 0, diagnostic


def legacy_signature_input(source):
    # Model only the entry tokens seen by legacy extract_host_stub.py. Do not
    # erase attributes as the host C++ semantic check does: that hid this bug.
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    for token, expansion in {
        "__global__": "__attribute__((cce_kernel))",
        "__aicore__": "[aicore]",
        "__vector__": "__attribute__((cce_vector))",
    }.items():
        source = source.replace(token, expansion)
    return source


@pytest.mark.parametrize("entry_file,entry_name", [
    ("quantize_kernel.cpp", "vc_laser_quantize_fp16"),
    ("quantize_bf16_kernel.cpp", "vc_laser_quantize_bf16"),
])
def test_cann901_parser_reads_single_entry_per_translation_unit(entry_file, entry_name):
    implementation = (builder.SOURCE / "quantize_kernel_impl.h").read_text()
    source = implementation + "\n" + (builder.SOURCE / entry_file).read_text()
    signatures = cann_stub.FUNC_SIGNATURE.findall(legacy_signature_input(source))
    parsed = [cann_stub.parse_func_signature_parts(signature) for _, signature in signatures]
    # Explicit-AIV metadata in the 9.0.1 generator uses the source's base key
    # for every function, while aliases increment the key per function. A
    # second entry in this TU would therefore point at a nonexistent symbol.
    assert len(parsed) == 1
    assert parsed[0][1] == entry_name
    expected = ["GM_ADDR input", "GM_ADDR output", "GM_ADDR scales", "uint32_t sequence",
                "uint32_t dimension", "uint32_t paddedSequence", "uint64_t tasks", "uint32_t cores"]
    for result_type, _, params in parsed:
        assert result_type == "void"
        assert [" ".join(param.split()) for param in params.split(",")] == expected
    # Request AIV explicitly without changing the persistent block mapping.
    entries = re.findall(r"void vc_laser_quantize_\w+\([^)]*\)\s*\{([^}]*)\}", source)
    assert len(entries) == 1
    for body in entries:
        assert re.search(r"KERNEL_TASK_TYPE_DEFAULT\(KERNEL_TYPE_AIV_ONLY\)\s*;", body)


@pytest.mark.parametrize("entry_file", ["quantize_kernel.cpp", "quantize_bf16_kernel.cpp"])
def test_cann901_parser_reproduces_reported_failure_for_old_vector_attribute(entry_file):
    source = (builder.SOURCE / entry_file).read_text()
    old_source = source.replace('__global__ __aicore__ void vc_laser_quantize_',
                                '__global__ __aicore__ __vector__ void vc_laser_quantize_')
    signatures = cann_stub.FUNC_SIGNATURE.findall(legacy_signature_input(old_source))
    assert len(signatures) == 1
    for _, signature in signatures:
        with pytest.raises(cann_stub.FuncNameNotFound):
            cann_stub.parse_func_signature_parts(signature)


def test_both_entry_units_are_separate_cmake_sources_and_manifest_includes_shared_impl():
    cmake = (builder.SOURCE / "CMakeLists.txt").read_text()
    sources = re.search(r"ascendc_library\(vc_quantize_kernel\s+STATIC\s+([^)]*)\)", cmake)
    assert sources is not None
    assert sources.group(1).split() == ["quantize_kernel.cpp", "quantize_bf16_kernel.cpp"]
    assert set(sources.group(1).split()) <= set(builder.SOURCE_NAMES)
    assert "quantize_kernel_impl.h" in builder.SOURCE_NAMES
