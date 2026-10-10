"""Optional Hopper lowering checks for the isolated TileLang candidates."""
from pathlib import Path
import re
import sys

import pytest

tilelang = pytest.importorskip("tilelang")
from tvm.target import Target

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tilelang_expcast_opt import NO_WS_VARIANTS, compiler_pass_configs, make_primfunc


def lower_source(dim, stages, variant, dtype, queries=65, keys=129):
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with target, tilelang.transform.PassContext(config=compiler_pass_configs(variant)):
        return tilelang.lower(make_primfunc(1, 2, queries, keys, dim, dtype, stages, variant),
                              target=target).kernel_source


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("variant,stages", [
    ("cached", 1), ("cached", 2), ("mma_qk", 1), ("mma_qk", 2),
    ("cached_no_ws", 1), ("cached_no_ws_overlap", 1), ("cached_no_ws_int_mass", 1),
    ("cached_no_ws_vec_decode", 1), ("cached_no_ws_reduce4", 1),
])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cached_scales_and_native_low_precision_gemms(dim, stages, variant, dtype):
    # Three K tiles exercise a refill, as well as the masked key tail, in the
    # cp.async pipeline. Existing warp-specialized sources retain their shape.
    source = lower_source(dim, stages, variant, dtype,
                          keys=257 if variant in NO_WS_VARIANTS else 129)
    outdir = Path(__file__).resolve().parents[1] / "results" / "tilelang_opt_local_sources"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{variant}_{dtype}_d{dim}_s{stages}.cu").write_text(source, encoding="utf-8")
    # Cache QS outside the key loop. The compiler must not rematerialize its
    # global reads for every score, which is the point of this optimization.
    qs = source.index("QS[")
    loop = source.index("for (int", qs)
    assert "QS[" not in source[loop:]
    assert "query_scale[" in source and "key_scale[" in source
    assert "vc_paper_code(" in source
    assert "kFloat8_e4m3, tl::DataType::kFloat8_e4m3" in source
    if variant == "mma_qk":
        assert "mma_sync" in source
        assert "wgmma_ss<tl::DataType::kInt8" not in source
    else:
        assert "kInt8, tl::DataType::kInt8" in source
    if variant in NO_WS_VARIANTS:
        assert "__launch_bounds__(128, 1)" in source
        # The operand fence must cover only the physical per-thread fragment.
        # A Q128/256-thread probe exposed an out-of-bounds default fence.
        for name in ("dots", "pv"):
            size = int(re.search(rf"\b(?:int|float) {name}\[(\d+)\]", source)[1])
            fences = re.findall(
                rf"warpgroup_fence_operand\(reinterpret_cast<[^>]+>\({name} \+ 0\), (\d+)\)",
                source)
            assert fences and all(int(count) <= size for count in fences)
        assert "warpgroup_reg_alloc" not in source
        assert "warpgroup_reg_dealloc" not in source
        assert "if (((int)threadIdx.x) < 128)" not in source
        assert "tma_load(" not in source
        assert "tl::cp_async_gs<16>" in source
        assert "tl::cp_async_wait<" in source
        qk = source.index("kInt8, tl::DataType::kInt8")
        ready = source.rindex("tl::cp_async_wait<", 0, qk)
        assert "__syncthreads()" in source[ready:qk]
        qk_done = source.index("warpgroup_wait<0>()", qk)
        refill = qk_done + re.search(
            r"tl::cp_async_gs(?:_conditional)?<16>", source[qk_done:]).start()
        assert "__syncthreads()" in source[qk_done:refill]
        pv = source.index("kFloat8_e4m3, tl::DataType::kFloat8_e4m3")
        if variant == "cached_no_ws_overlap":
            commit = source.index("warpgroup_commit_batch()", pv)
            mass = source.index("AllReduce<tl::SumOp", pv)
            scale_history = re.search(r"accum\[.+?\] = vc_paper_mul", source[mass:])
            assert scale_history
            pv_done = source.index("wait_wgmma<0>()", pv)
            assert commit < mass < mass + scale_history.start() < pv_done
            fence = source.index("warpgroup_fence_operand", pv_done)
            assert re.match(
                rf"warpgroup_fence_operand\(reinterpret_cast<float\*>\(pv \+ 0\), {dim // 2}\)",
                source[fence:])
            shared_live = source.index("vc_opt_keep_shared_live(", pv_done)
            add_history = re.search(r"accum\[.+?\] = vc_paper_add", source[pv_done:])
            assert add_history
            assert pv_done < fence < shared_live < pv_done + add_history.start()
            assert "warpgroup_wait<0>()" not in source[pv:pv_done]
        else:
            pv_done = source.index("warpgroup_wait<0>()", pv)
        refill = source.index("tl::cp_async_gs<16>", pv_done)
        assert "__syncthreads()" in source[pv_done:refill]
        assert "AllReduce<tl::MaxOp" in source
        assert "AllReduce<tl::SumOp" in source
        if variant == "cached_no_ws_int_mass":
            assert "vc_opt_prob_units(code)" in source
            assert "int mass_units[2]" in source
            assert "mass_units" in source[source.index("AllReduce<tl::SumOp"):pv]
            mass_region = source[source.index("AllReduce<tl::SumOp"):pv]
            assert "(float)(v_.x)" in mass_region and "(float)(v_.y)" in mass_region
            assert "0x1p-9f" in mass_region
            assert not re.search(r"logits\[.+?\] = \(\(float\)prob\[", source)
        elif variant == "cached_no_ws_vec_decode":
            assert "__tl_cvt_fp8x2_to_float2" in source
            assert not re.search(r"logits\[.+?\] = \(\(float\)prob\[", source)
        elif variant == "cached_no_ws_reduce4":
            assert "float grouped_values[64]" in source
            assert "float grouped_partial[8]" in source
            assert "grouped_partial" in source[source.index("AllReduce<tl::MaxOp"):]
        return
    # The compiler adds a producer warpgroup. Explicit lifetimes must prevent
    # either DMA buffer from being overwritten while a GEMM still uses it.
    from test_tilelang_expcast_codegen import branch_body
    split = re.search(r"if \(\(\(int\)threadIdx.x\) < 128\)", source)
    assert split
    producer, end = branch_body(source, split.start())
    consumer, _ = branch_body(source, source.index("else", end))
    assert "tma_load(V_desc" in producer
    pv = consumer.index("kFloat8_e4m3, tl::DataType::kFloat8_e4m3")
    wait = consumer.index("warpgroup_wait<0>()", pv)
    releases = list(re.finditer(r"mbarrier\[.+?\]\.arrive\(\);", consumer))
    assert len(releases) == 2
    assert releases[0].start() < pv < wait < releases[1].start()
    if variant == "cached":
        qk = consumer.index("kInt8, tl::DataType::kInt8")
        assert consumer.index("warpgroup_wait<0>()", qk) < releases[0].start()
    else:
        assert consumer.index("k_register +") < releases[0].start()
        # The shared K lifetime extends through every synchronous MMA and the
        # explicit 128-consumer rendezvous, not just through the register copy.
        finish = consumer.index("vc_mma_finish_k(")
        assert consumer.rindex("tl::mma_sync", 0, finish) < finish < releases[0].start()
        assert "((signed char*)k)" not in consumer[releases[0].end():]
        assert "k_register" not in consumer[pv:]
        assert "vc_mma_finish_k(" not in producer
        assert 'asm volatile("bar.sync 15, 128;" : : "l"(k) : "memory")' in source
    assert "AllReduce<tl::MaxOp" in consumer
    assert "AllReduce<tl::SumOp" in consumer


@pytest.mark.parametrize("variant", NO_WS_VARIANTS)
def test_no_ws_two_stages_rejected(variant):
    with pytest.raises(ValueError, match="two stages spill registers"):
        make_primfunc(1, 2, 65, 257, 128, stages=2, variant=variant)
