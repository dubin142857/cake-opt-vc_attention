"""Host syntax and UB sizing checks; NOT CCE compilation/device validation."""
from pathlib import Path
import re

import pytest

from test_mindie_laser_expcast_cpp import PREAMBLE, compile_syntax, cpp_compiler
from scripts import mindie_laser_expcast_patch as patch


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ascend/laser_expcast_fp16/vc_laser_probability.h"


@pytest.mark.parametrize("operation,constant", [("vmuls", "GAIN"), ("vadds", "BIAS"), ("vmins", "CODE_MAX")])
def test_legacy_cce_scalar_operands_keep_explicit_float_values(operation, constant):
    # Regression for the actual CANN 9.0.1 builtin matcher: a constexpr float
    # lvalue failed remotely although an ordinary host declaration accepts it.
    text = HEADER.read_text(encoding="utf-8")
    assert re.search(rf"{operation}\(scratch,\s*scratch,\s*static_cast<float>\({constant}\),", text)


def test_binary16_helper_compiles_with_existing_laser_callsite(cpp_compiler, tmp_path):
    # The real callsite keeps SIZE_FP16 as an ordinary instance member. Half
    # intrinsic declarations below are syntax stubs, not a device simulator.
    preamble = PREAMBLE.replace("struct half { uint16_t bits; };",
        "struct half { uint16_t bits; explicit half(float); };")
    preamble += "void vmuls(half*, half*, half, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);\n"
    source = preamble + HEADER.read_text(encoding="utf-8") + """
template<class INPUT_T> struct VectorContext {
    int32_t SIZE_FP16 = 2;
    int32_t rowPerTime = 4;
    struct Params { int32_t section_block_num; };
    struct Scratch { INPUT_T *buf_for_cacl_rowmax_fp16; };
    void transform(INPUT_T *cur_buf_for_attn_score_fp16, Scratch ub_attn, Params param) {
""" + patch.EXPERIMENTAL_PROBABILITY.decode("utf-8") + """
    }
};
void instantiate(VectorContext<half>& context, half* values,
                 VectorContext<half>::Scratch scratch, VectorContext<half>::Params params) {
    context.transform(values, scratch, params);
}
"""
    status, diagnostic = compile_syntax(cpp_compiler, tmp_path, source)
    assert status == 0, diagnostic


@pytest.mark.parametrize("count", [128, 512, 4096, 8192, 8704, 16384])
def test_existing_scratch_and_conversion_strides_cover_nonoverlapping_chunks(count):
    # 64 FP32 lanes and 128 FP16 lanes; each repeat stride uses 32 B blocks.
    touched = []
    for offset in range(0, count, 8192):
        chunk = min(count - offset, 8192)
        repeat32, repeat16 = chunk // 64, chunk // 128
        assert 1 <= repeat32 <= 128 and 1 <= repeat16 <= 64
        assert ((repeat32 - 1) * 8 * 32 + 64 * 4) <= 32768
        half_write_end = (repeat32 - 1) * 4 * 32 + 64 * 2
        half_scale_end = (repeat16 - 1) * 8 * 32 + 128 * 2
        assert half_write_end == half_scale_end == chunk * 2
        touched.extend(range(offset, offset + chunk))
    assert touched == list(range(count))
