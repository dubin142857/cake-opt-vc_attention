"""Optional native-helper test. Build tests/cuda/sage_vc_tile_probe.cu as
_sage_vc_tile_probe with torch's CUDAExtension, then put it on PYTHONPATH.
The installed production attention does not contain this probe.
"""
import importlib.util
from pathlib import Path
import pytest
import torch

probe = pytest.importorskip("_sage_vc_tile_probe")
spec = importlib.util.spec_from_file_location("vc_oracle", Path(__file__).resolve().parents[1] /
    "third_party/sageattention_local/sageattention_local/vc_reference.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_real_expcast_register_packing_online_rescale_and_mean_restore(seed):
    torch.manual_seed(seed)
    device = "cuda"
    scores = torch.randn(64,128,device=device) * 5
    scores[:,117:] = -float("inf")
    out, product = (torch.randn_like(scores) for _ in range(2))
    m = torch.randn(64,device=device) * 2 + 1
    d = torch.rand(64,device=device) * 200
    scale = torch.rand(128,device=device) + 0.01
    mean = (torch.randn(128,device=device) * 8).half()
    codes, actual, maximum, denom = probe.run(scores,out,product,m,d,scale,mean)
    expected_max = torch.maximum(m,(scores * 0.25).amax(-1))
    alpha = (m-expected_max).exp()
    expected_codes = oracle.expcast_codes(scores * 0.25 - expected_max[:,None])
    mass = expected_codes.view(torch.float8_e4m3fn).float().sum(-1)
    expected = out * alpha[:,None] + (product + mass[:,None] * mean.float()) * scale
    torch.testing.assert_close(codes,expected_codes,rtol=0,atol=0)
    torch.testing.assert_close(maximum,expected_max,rtol=0,atol=0)
    torch.testing.assert_close(denom,d * alpha + mass,rtol=1e-6,atol=1e-5)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-5)
