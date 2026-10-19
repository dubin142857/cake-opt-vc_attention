"""Pinned source delta, C++ signatures and asynchronous slot-lifetime checks.

Host checks cannot establish CCE support or device timing. The independent
queue model explores producer/consumer order, including a slow Vector row sum
after a fast P store, which is unsafe with only the inherited MTE3 fence.
"""
from pathlib import Path
import random

import pytest

from scripts import mindie_laser_quant_prefetch_patch as patch
from scripts import prepare_mindie_laser_quant_reuse_source as reuse
from scripts.prepare_cann_fas_source import git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reuse_files():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned source cache unavailable; patch tests do not fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == reuse.SOURCE_LOCK.commit
    return reuse.expected_files(reference)[0]


@pytest.fixture(scope="module")
def vector(reuse_files):
    return reuse_files[f"{reuse.CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"]


def test_only_vector_changes_and_exact_source_inverse(reuse_files, vector):
    changed = []
    for name, data in reuse_files.items():
        result = patch.apply(name.replace("reuse", "prefetch"), data)
        if result != data:
            changed.append(Path(name).name)
            assert patch.unpatch_vector(result, data) == data
    assert changed == ["VectorForward.h"]
    new = patch.patch_vector(vector)
    # Max, exponentiation, padding and row-sum arithmetic retain byte order.
    for start, end in (
        (b"    vmuls(cur_buf_for_attn_score_fp16", b"template <typename INPUT_T"),
        (b"    if ((is_tail_section) && (windowLen != 0))", b"    // "),
        (b"    copy_ubuf_to_gm(((__gm__ half *)", b"\n}"),
    ):
        begin = vector.index(start)
        finish = vector.index(end, begin + len(start))
        assert vector[begin:finish] in new
    # Actual mask copies and all related pointer arithmetic remain one copy.
    begin = vector.index(b"    bool sparse_tail_mask_flag")
    end = vector.index(patch.READY_OLD, begin)
    masks = vector[begin:end]
    assert new.count(masks) == 1
    # No Cube-to-Vector scheduling delta is hidden in Run.
    begin = vector.index(b"::Run()")
    end = vector.index(b"::allocate_ubuf_for_short_seq_attn_score", begin)
    assert vector[begin:end] in new


def test_fail_closed_on_duplicate_patch_or_changed_protocol(vector):
    with pytest.raises(ValueError, match="already"):
        patch.patch_vector(patch.patch_vector(vector))
    for old, new in (
        (b"rowPerTime = 4;", b"rowPerTime = 8;"),
        (b"MAX_LENG_PER_UB_PROC * SIZE_FP16 * 2", b"MAX_LENG_PER_UB_PROC * SIZE_FP16"),
        (patch.CALL_OLD, b"changed_call();"),
        (b"BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN", b"BASE_BLOCK_DATA_NUM - 7"),
    ):
        with pytest.raises(ValueError):
            patch.patch_vector(vector.replace(old, new))
    with pytest.raises(ValueError, match="unused"):
        patch.patch_vector(vector + b"\nset_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);\n")


def test_crlf_is_preserved_and_windows_paths_dispatch(vector):
    data = vector.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    new = patch.apply(r"csrc\ops\vc_laser_quant_prefetch_attention\op_kernel\VectorForward.h", data)
    assert patch.unpatch_vector(new, data) == data
    assert b"\n" not in new.replace(b"\r\n", b"")


def test_event_protocol_and_no_prefetch_past_end(vector):
    new = patch.patch_vector(vector)
    assert patch.PROLOGUE in new
    assert patch.DRIVER_NEW in new
    assert patch.RELEASE_NEW in new
    assert patch.DRAIN_NEW in new
    # Row sum is enqueued before the new Vector release. This is necessary:
    # completion of P's MTE3 copy alone does not release later Vector reads.
    begin = new.index(b"        process_line_phase_one_for_short_seq_exp_and_rowsum(qk_triangle,")
    release = new.index(patch.RELEASE_NEW, begin)
    assert begin < release
    helper = new[new.index(b"    __aicore__ __inline__ void vc_prefetch_score_batch"):new.index(patch.HELPER_ANCHOR)]
    assert helper.index(b"wait_flag(PIPE_V, PIPE_MTE2") < helper.index(b"copy_gm_to_ubuf")
    assert helper.index(b"wait_flag(PIPE_MTE3, PIPE_MTE2") < helper.index(b"copy_gm_to_ubuf")
    assert helper.index(b"copy_gm_to_ubuf") < helper.index(b"set_flag(PIPE_MTE2, PIPE_V")


@pytest.mark.parametrize("blocks", [1, 2, 3, 7, 8, 9, 16, 17, 31, 32])
@pytest.mark.parametrize("subblock", [0, 1])
def test_dma_addresses_cover_original_rows_without_reading_written_p(blocks, subblock):
    # GM layout is [K128 tile][128 query rows][128 keys], independently of
    # the two AIVs. Compare original row addressing to opposite-slot lookahead.
    start = 391 * 16384 + subblock * 64 * 128
    offsets = [start, start + 4 * 128]
    loaded, written = [], set()
    for line in range(0, 64, 4):
        slot = line // 4 % 2
        assert offsets[slot] == start + line * 128
        current = {
            (offsets[slot] + tile * 16384 + cell)
            for tile in range(blocks) for cell in range(4 * 128)
        }
        assert not current & written
        loaded.append(current)
        if line + 4 < 64:
            other = 1 - slot
            assert offsets[other] == start + (line + 4) * 128
            future = {
                offsets[other] + tile * 16384 + cell
                for tile in range(blocks) for cell in range(4 * 128)
            }
            assert not future & (written | current)
        written |= current
        offsets[slot] += 256 * 4
    expected = {
        start + tile * 16384 + cell
        for tile in range(blocks) for cell in range(64 * 128)
    }
    assert set().union(*loaded) == expected
    assert len(written) == blocks * 64 * 128


def _queue_protocol(vector_fence=True):
    """Model only actual buffer users and event directions, as separate queues."""
    queues = {p: [] for p in ("V", "MTE2", "MTE3")}

    def emit(pipe, kind, *args):
        queues[pipe].append((kind, *args))

    def signal(src, dst, slot):
        emit(src, "set", src, dst, slot)

    def wait(src, dst, slot):
        emit(dst, "wait", src, dst, slot)

    def load(batch):
        slot = batch % 2
        wait("MTE3", "MTE2", slot)
        if vector_fence:
            wait("V", "MTE2", slot)
        emit("MTE2", "load", batch)
        signal("MTE2", "V", slot)

    for slot in (0, 1):
        signal("MTE3", "MTE2", slot)
        if vector_fence:
            signal("V", "MTE2", slot)
    load(0)
    for batch in range(16):
        slot = batch % 2
        wait("MTE2", "V", slot)
        if batch < 15:
            load(batch + 1)
        emit("V", "probability", batch)
        signal("V", "MTE3", slot)
        wait("V", "MTE3", slot)
        emit("MTE3", "store", batch)
        emit("V", "sum", batch)
        if vector_fence:
            signal("V", "MTE2", slot)
        signal("MTE3", "MTE2", slot)
    for slot in (0, 1):
        if vector_fence:
            wait("V", "MTE2", slot)
        wait("MTE3", "MTE2", slot)
    return queues


def _run_interleaving(seed, vector_fence=True):
    queues = _queue_protocol(vector_fence)
    tokens, contents, summed, stored = set(), {}, set(), set()
    positions = {p: 0 for p in queues}
    rng = random.Random(seed)
    while any(positions[p] < len(q) for p, q in queues.items()):
        runnable = []
        for p, q in queues.items():
            if positions[p] == len(q):
                continue
            op = q[positions[p]]
            if op[0] != "wait" or tuple(op[1:]) in tokens:
                runnable.append(p)
        assert runnable, "Deadlocked queue dependency graph"
        pipe = rng.choice(runnable)
        op = queues[pipe][positions[pipe]]
        positions[pipe] += 1
        kind, args = op[0], op[1:]
        if kind == "set":
            assert args not in tokens, "Double signal without a consuming wait"
            tokens.add(args)
        elif kind == "wait":
            tokens.remove(args)
        else:
            batch, = args
            slot = batch % 2
            if kind == "load":
                if batch >= 2:
                    assert batch - 2 in summed, "Slot overwritten before Vector row sum"
                    assert batch - 2 in stored, "Slot overwritten before P store"
                contents[slot] = batch
            else:
                assert contents[slot] == batch, "Wrong batch consumed from ping/pong slot"
                if kind == "sum":
                    summed.add(batch)
                elif kind == "store":
                    stored.add(batch)
    assert not tokens, "Undrained events leak into the next section"
    assert summed == stored == set(range(16))


def test_async_pipeline_orders_keep_both_slot_users_alive():
    for seed in range(250):
        _run_interleaving(seed)


def test_model_detects_missing_vector_free_fence():
    # Regression relevance: some legal queue interleavings must reject the
    # superficially plausible version protected only by the old P-store fence.
    for seed in range(250):
        try:
            _run_interleaving(seed, vector_fence=False)
        except AssertionError as exc:
            assert "row sum" in str(exc) or "Wrong batch" in str(exc)
            break
    else:
        pytest.fail("Model did not expose the score-slot lifetime hazard")


@pytest.mark.parametrize("valid_keys", [1, 127, 128, 129, 255, 256, 4097, 8193, 37741])
def test_dense_padding_guard_does_not_exclude_non_aligned_keys(vector, valid_keys):
    new = patch.patch_vector(vector)
    # Padding is a later Vector operation, not a mask DMA or a guard input.
    begin = new.index(b"    const bool vc_score_prefetch")
    end = new.index(b";", begin)
    assert b"windowLen" not in new[begin:end]
    padded = (valid_keys + 255) // 256 * 256
    padding = padded - valid_keys
    assert 0 <= padding < 256
    # Tail vector_dup keeps the original windowLen path exactly. K37741 has
    # one whole masked K128 block and 19 masked entries in the preceding one.
    if valid_keys == 37741:
        assert padding == 147
    begin = vector.index(b"    if ((is_tail_section) && (windowLen != 0))")
    end = vector.index(b"    process_cacl_max(", begin)
    assert vector[begin:end] in new


def test_actual_helper_and_section_driver_have_valid_cpp_signatures(vector, cpp_compiler, tmp_path):
    new = patch.patch_vector(vector).decode()
    # Instantiate the actual patched section method and helper. Other methods
    # are declarations only: this is host language validation, not an NPU test.
    class_start = new.index("class VectorForward")
    declaration_start = new.index("    __aicore__ __inline__ void process_line_phase_one_for_short_seq_max", class_start)
    declaration_end = new.index(";", declaration_start) + 1
    max_declaration = new[declaration_start:declaration_end]
    driver_decl_start = new.index("    __aicore__ __inline__ void attention_score_short_double_line_one", class_start)
    driver_decl_end = new.index(";", driver_decl_start) + 1
    driver_declaration = new[driver_decl_start:driver_decl_end]
    struct_start = new.index("    struct UB_FOR_SHORT_LEN_ATTN_SCORE")
    struct_end = new.index("    struct UB_FOR_NORMALIZE", struct_start)
    scratch = new[struct_start:struct_end]
    param_start = new.index("    struct PARAM_SHORT_SEQ_MAX")
    param_end = new.index("    struct PARAM_LONG_SEQ_EXP", param_start)
    params = new[param_start:param_end]
    method_marker = "IF_BF16, WORKSPACE_T>::attention_score_short_double_line_one("
    body_start = new.rfind("template <typename INPUT_T", 0, new.index(method_marker))
    body_end = new.index("template <typename INPUT_T", new.index(method_marker))
    method = new[body_start:body_end]
    source = r"""
typedef int int32_t; typedef unsigned long long uint64_t; typedef int event_t;
typedef unsigned long long uintptr_t;
#define __aicore__
#define __inline__ inline
#define __ubuf__
#define __gm__
#define __restrict__
enum {PIPE_V,PIPE_S,PIPE_MTE2,PIPE_MTE3,EVENT_ID0,EVENT_ID1,EVENT_ID2,EVENT_ID3,EVENT_ID4,EVENT_ID5,EVENT_ID6};
enum {TRI_MATRIX_NONE,TRI_MATRIX_TAIL,TRI_MATRIX_HEAD_AND_TAIL};
constexpr int BASE_BLOCK_SIDE_LEN=128, BASE_BLOCK_SIZE_DOUBLE=256, BASE_BLOCK_DATA_NUM=16384;
void set_flag(int,int,event_t); void wait_flag(int,int,event_t);
void copy_gm_to_ubuf(float*,float*,int,int,int,int,int);
void vector_dup(...); void pipe_barrier(int); int get_subblockid();
void __set_mask(int); void set_vector_mask(uint64_t,uint64_t); void vconv_f162f32(...);
template<class INPUT_T,bool IF_BF16,class WORKSPACE_T> class VectorForward {
public:
    int rowPerTime=4,maskSeqLength=256;
    float *diagExpMaxJMatUbAddr,*diag_rowmax_gm;
""" + scratch + params + patch.HELPER.decode() + max_declaration + driver_declaration + r"""
    void get_padding_info_for_row_max(...);
    void process_line_phase_one_for_short_seq_exp_and_rowsum(...);
    void form_diagmat(...); void updateRowsum(...); void backupMaxandSum(...);
};
""" + method + r"""
void instantiate(VectorForward<float,false,float>& op,
                 VectorForward<float,false,float>::UB_FOR_SHORT_LEN_ATTN_SCORE scratch,
                 float* score,float* mask) {
    op.attention_score_short_double_line_one(0,2,0,32,0,64,0,true,false,0,score,mask,scratch,false);
}
"""
    result, diagnostic = compile_syntax(cpp_compiler, tmp_path, source)
    assert result == 0, diagnostic
