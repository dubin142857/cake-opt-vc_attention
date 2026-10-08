"""Real shared-address descriptor equality; not Hopper MMA execution."""
import pytest
import torch

probe=pytest.importorskip("_sage_vc_descriptor_probe")


@pytest.mark.parametrize("stride",[64,128])
@pytest.mark.parametrize("side_stream",[False,True])
def test_aligned_anchor_offsets_match_upstream_descriptor_bits(stride,side_stream):
    stream=torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        result=probe.run(torch.empty(0,device="cuda"),stride)
    stream.synchronize()
    old,new=result[...,0],result[...,1]
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    # Start address advances by 32 bytes; all other descriptor fields stay fixed.
    assert (new[:,0].bitwise_and(7)==0).all().item()
    address=new.bitwise_and(0x3fff)
    expected=address[:,:1]+torch.arange(4,device="cuda")[None,:]*2
    torch.testing.assert_close(address,expected,rtol=0,atol=0)
    torch.testing.assert_close(new.bitwise_and(~0x3fff),
                               new[:,:1].bitwise_and(~0x3fff).expand_as(new),rtol=0,atol=0)
