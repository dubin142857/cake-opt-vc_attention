"""Optional Hopper lowering checks for the isolated TileLang candidates."""
from pathlib import Path
import re
import sys

import pytest

tilelang = pytest.importorskip("tilelang")
from tvm.target import Target

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tilelang_expcast_opt import make_primfunc


def lower_source(dim, stages, variant, dtype, queries=65, keys=129):
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with target, tilelang.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}):
        return tilelang.lower(make_primfunc(1, 2, queries, keys, dim, dtype, stages, variant),
                              target=target).kernel_source


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("variant", ["cached", "mma_qk"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cached_scales_and_native_low_precision_gemms(dim, stages, variant, dtype):
    source = lower_source(dim, stages, variant, dtype)
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
    assert "tma_load(V_desc" in source
    if variant == "mma_qk":
        assert "mma_sync" in source
        assert "wgmma_ss<tl::DataType::kInt8" not in source
    else:
        assert "kInt8, tl::DataType::kInt8" in source
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
        assert "((signed char*)k)" not in consumer[releases[0].end():]
        assert "k_register" not in consumer[pv:]
    assert "AllReduce<tl::MaxOp" in consumer
    assert "AllReduce<tl::SumOp" in consumer
