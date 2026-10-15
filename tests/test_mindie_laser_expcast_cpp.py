"""Host C++ semantic regression, NOT a CCE build or device API validation.

Compile the actual replacement bytes inside a template with upstream's mutable
SIZE_FP16 instance member. Intrinsics are declarations only: these checks catch
ordinary C++ constant-expression, template and name-resolution mistakes. They
cannot establish device instruction support, address spaces, synchronization or
numerical behavior. No compiler is installed; unavailable compilers skip tests.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import mindie_laser_expcast_patch as patch
from scripts import mindie_laser_probe_patch as probe


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cpp_compiler():
    for name in ("clang++", "g++", "c++", "cl"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    if os.name == "nt":
        # Existing Visual Studio installation; do not require a developer shell
        # or standard-library headers for the deliberately small translation unit.
        vs = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio"
        candidates = sorted(vs.glob("*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"), reverse=True)
        if candidates:
            return candidates[0]
        clang = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "LLVM/bin/clang++.exe"
        if clang.is_file():
            return clang
    pytest.skip("An existing C++ compiler is required for host semantic checks; no tools installed")


PREAMBLE = r"""
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
struct half { uint16_t bits; };
#define __aicore__
#define __ubuf__
#define __gm__
#define __DAV_C220_VEC__ 1
enum { PIPE_V, PIPE_S, EVENT_ID7 };
void set_vector_mask(uint64_t, uint64_t);
void pipe_barrier(int);
void set_flag(int, int, int);
void wait_flag(int, int, int);
void vconv_f162f32(float*, half*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
void vconv_f322s16r(int16_t*, float*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
void vmuls(float*, float*, float, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
void vadds(float*, float*, float, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
void vmaxs(float*, float*, float, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
void vmins(float*, float*, float, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
void vshl(int16_t*, int16_t*, uint32_t, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
constexpr int32_t BASE_BLOCK_SIDE_LEN = 128;
constexpr int MAX_LENG_PER_UB_PROC = 16384;
"""


def compile_syntax(compiler, directory, source):
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / "semantic_check.cpp"
    filename.write_text(source, encoding="utf-8")
    if compiler.name.lower() in ("cl", "cl.exe"):
        command = [str(compiler), "/nologo", "/std:c++17", "/utf-8", "/Zs", "/TP", str(filename)]
    else:
        command = [str(compiler), "-std=c++17", "-fsyntax-only", "-x", "c++", str(filename)]
    result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
    diagnostic = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    return result.returncode, diagnostic


def replacement_translation_unit(fragment, input_type="half"):
    header = (ROOT / "ascend/laser_expcast/vc_laser_probability.h").read_text(encoding="utf-8")
    # SIZE_FP16 MUST stay mutable/non-static; making it constexpr masks the bug
    # that failed on the remote CCE compiler. The method is actually instantiated.
    return PREAMBLE + header + r"""
template<class INPUT_T> struct VectorContext {
    int32_t SIZE_FP16 = 2;
    int32_t rowPerTime = 4;
    struct Params { int32_t section_block_num; };
    struct Scratch { INPUT_T *buf_for_cacl_rowmax_fp16; };
    void transform(INPUT_T *cur_buf_for_attn_score_fp16, Scratch ub_attn, Params param) {
""" + fragment.decode("utf-8") + r"""
    }
};
""" + f"""
void instantiate(VectorContext<{input_type}>& context, {input_type}* values,
                 VectorContext<{input_type}>::Scratch scratch,
                 VectorContext<{input_type}>::Params params) {{
    context.transform(values, scratch, params);
}}
"""


def test_actual_fixed_fragment_compiles_with_mutable_upstream_size_member(cpp_compiler, tmp_path):
    result, diagnostic = compile_syntax(cpp_compiler, tmp_path, replacement_translation_unit(patch.EXPERIMENTAL_PROBABILITY))
    assert result == 0, diagnostic


def test_legacy_fragment_fails_for_nonconstant_this_member_not_missing_headers(cpp_compiler, tmp_path):
    result, diagnostic = compile_syntax(cpp_compiler, tmp_path,
                                        replacement_translation_unit(patch.LEGACY_EXPERIMENTAL_PROBABILITY))
    assert result != 0, "The negative control must reproduce the remote static_assert error"
    assert ("C2131" in diagnostic or "constant expression" in diagnostic
            or "non-constant condition" in diagnostic), diagnostic
    assert not any(token in diagnostic for token in ("C1083", "file not found", "No such file")), diagnostic


def test_actual_fixed_fragment_rejects_non_16_bit_input(cpp_compiler, tmp_path):
    result, diagnostic = compile_syntax(cpp_compiler, tmp_path,
                                        replacement_translation_unit(patch.EXPERIMENTAL_PROBABILITY, "float"))
    assert result != 0
    assert "ExpCast requires 16-bit Laser input" in diagnostic, diagnostic


def test_probe_header_has_no_ordinary_cpp_name_or_template_errors(cpp_compiler, tmp_path):
    # Keep the real header include; it resolves to a local declaration-only stub.
    # This does NOT verify these declarations against CANN headers or hardware.
    (tmp_path / "kernel_operator.h").write_text("// Host syntax fixture only.\n", encoding="utf-8")
    declarations = r"""
namespace AscendC {
template<class T> class GlobalTensor {
public:
    void SetGlobalBuffer(T*, uint64_t);
    void SetValue(uint32_t, T);
    GlobalTensor<T> operator[](uint32_t);
};
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T, CacheLine L, DcciDst D> void DataCacheCleanAndInvalid(const GlobalTensor<T>&);
uint64_t GetSystemCycle();
}
"""
    result, diagnostic = compile_syntax(cpp_compiler, tmp_path,
                                        PREAMBLE + declarations + probe.HEADER.decode("utf-8"))
    assert result == 0, diagnostic
