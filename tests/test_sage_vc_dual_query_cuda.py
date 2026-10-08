"""Production barrier/address helpers on CUDA; full attention needs H200 checks."""
import pytest
import torch

probe=pytest.importorskip("_sage_vc_dual_query_probe")


@pytest.mark.parametrize("queries",[1,63,64,65,127,128,129,191,192,193,255,256,257,37741])
@pytest.mark.parametrize("stagger",[False,True])
def test_q64_scales_and_cta_handoffs(queries,stagger):
    batch,heads=2,3
    count=(queries+63)//64
    steps=7 if queries!=37741 else 2
    scales=torch.arange(batch*heads*count*32,device="cuda",dtype=torch.int32).reshape(batch,heads,-1)
    actual=probe.run(scales,queries,steps,17,stagger).cpu()
    tid=torch.arange(256)
    group,lane=tid//128,tid%128
    for block in range((queries+127)//128):
        qblock=block*2+group
        indices=qblock*32+lane//32*8+lane%32//4
        for b in range(batch):
            for h in range(heads):
                expected=torch.where(qblock<count,(b*heads+h)*count*32+indices,-1).int()
                torch.testing.assert_close(actual[b,h,block,:,0],expected,rtol=0,atol=0)
    for step in range(steps):
        expected_k=(step*257+(lane*3+group)%128).int().expand_as(actual[...,2*step+1])
        expected_v=(step*509+(lane*5+group)%128+17).int().expand_as(actual[...,2*step+2])
        torch.testing.assert_close(actual[...,2*step+1],expected_k,rtol=0,atol=0)
        torch.testing.assert_close(actual[...,2*step+2],expected_v,rtol=0,atol=0)


@pytest.mark.parametrize("stagger",[False,True])
def test_long_handoff_reuse_on_nondefault_stream(stagger):
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scales=torch.arange(160,device="cuda",dtype=torch.int32).view(1,1,-1)
        outputs=[probe.run(scales,257,296,3,stagger) for _ in range(3)]
    stream.synchronize()
    lane=torch.arange(256,device="cuda")%128
    group=torch.arange(256,device="cuda")//128
    for out in outputs:
        torch.testing.assert_close(out[...,-2],(295*257+(lane*3+group)%128).int().expand_as(out[...,-2]),rtol=0,atol=0)
        torch.testing.assert_close(out[...,-1],(295*509+(lane*5+group)%128+17).int().expand_as(out[...,-1]),rtol=0,atol=0)
