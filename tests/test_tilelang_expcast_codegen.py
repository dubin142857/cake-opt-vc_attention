"""Optional TileLang lowering checks for direct natural-V DMA and lifetime.

These inspect generated Hopper code, not GPU runtime correctness. TileLang is
optional for the existing test suite; remote attention tests remain mandatory.
"""
from pathlib import Path
import re
import sys

import pytest

tilelang=pytest.importorskip("tilelang")
from tvm.target import Target
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from tilelang_expcast import make_primfunc


def branch_body(source,start):
    begin=source.index("{",start)
    depth=1
    for end in range(begin+1,len(source)):
        depth += (source[end]=="{")-(source[end]=="}")
        if depth==0:return source[begin+1:end],end+1
    raise AssertionError("Unclosed generated branch")


@pytest.mark.parametrize("dim",[64,128])
@pytest.mark.parametrize("stages",[1,2])
@pytest.mark.parametrize("schedule",["baseline","overlap"])
def test_natural_v_has_no_unpack_and_releases_only_after_pv(dim,stages,schedule):
    target=Target({"kind":"cuda","arch":"sm_90a"})
    with target,tilelang.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True}):
        source=tilelang.lower(make_primfunc(1,2,65,129,dim,"float16",stages,schedule),target=target).kernel_source
    match=re.search(r"if \(\(\(int\)threadIdx.x\) < 128\)",source)
    assert match,"Expected the pinned compiler's producer/consumer split"
    producer,end=branch_body(source,match.start())
    consumer,_=branch_body(source,source.index("else",end))
    assert "tma_load(V_desc" in producer
    assert "fp8_e4_t*)v)" in producer
    assert "v_packed" not in source
    # No scalar/shared reordering stores into V on the consumer side.
    assert "= *(fp8_e4_2_t*)(((fp8_e4_t*)v)" not in consumer
    pv=consumer.index("kFloat8_e4m3, tl::DataType::kFloat8_e4m3")
    wait=consumer.index("warpgroup_wait<0>()" if schedule=="baseline" else "tl::wait_wgmma<0>()",pv)
    release=list(re.finditer(r"mbarrier\[.+?\]\.arrive\(\);",consumer))
    assert len(release)==2 and release[0].start()<pv and release[1].start()>wait
    if schedule=="overlap":
        assert pv<consumer.index("AllReduce<tl::SumOp")<wait
        assert "vc_paper_mul(accum[" in consumer[pv:wait]
        assert wait<consumer.index("vc_paper_keep_shared_live(")<release[1].start()
        assert "vc_paper_add(accum[" in consumer[wait:]
    else:
        assert consumer.index("AllReduce<tl::SumOp")<pv
    # The old unpack loop had 32/64 thread barriers. None may be loop-local.
    for loop in re.finditer(r"for \(int i_\d+ = 0;",consumer):
        body,_=branch_body(consumer,loop.start())
        assert "sync_thread_partial" not in body
