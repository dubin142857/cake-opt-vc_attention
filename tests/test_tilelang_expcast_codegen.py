"""Optional TileLang lowering checks for the V producer/consumer race.

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
def test_v_is_copied_by_producer_and_unpacked_only_by_consumer(dim,stages):
    target=Target({"kind":"cuda","arch":"sm_90a"})
    with target,tilelang.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True}):
        source=tilelang.lower(make_primfunc(1,2,65,129,dim,"float16",stages),target=target).kernel_source
    match=re.search(r"if \(\(\(int\)threadIdx.x\) < 128\)",source)
    assert match,"Expected the pinned compiler's producer/consumer split"
    producer,end=branch_body(source,match.start())
    consumer,_=branch_body(source,source.index("else",end))
    assert "tma_load(V_desc" in producer
    assert "fp8_e4_t*)v)" not in producer,"Producer overwrites the live PV operand"
    assert "fp8_e4_t*)v_packed)" in producer
    assert "fp8_e4_t*)v)" in consumer and "fp8_e4_t*)v_packed)" in consumer
    assert consumer.index("fp8_e4_t*)v)") < consumer.index("kFloat8_e4m3, tl::DataType::kFloat8_e4m3")
    # Packed DMA staging and consumer PV input must have disjoint storage.
    offsets={name:int(re.search(r"void\* "+name+r" = .*buf_dyn_shmem \+ (\d+)",source)[1])
             for name in ("v","v_packed")}
    assert abs(offsets["v"]-offsets["v_packed"])>=dim*128
    assert "warpgroup_wait<0>()" in consumer
