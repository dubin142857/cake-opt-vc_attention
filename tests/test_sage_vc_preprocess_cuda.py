"""Optional native preprocessing checks; run on any CUDA GPU, without Hopper MMA."""
import importlib.util
from pathlib import Path
import pytest
import torch

ext = pytest.importorskip("_vc_sm90")
ROOT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load("vc_preprocess_oracle",ROOT / "third_party/sageattention_local/sageattention_local/vc_reference.py")


def test_compiled_resource_query_validates_dtype_and_device():
    # The local MSVC/PyTorch 2.3 cross-build can translate c10 errors without
    # their message (also true of existing guards); require the exception type.
    with pytest.raises(RuntimeError):
        ext.kernel_resources(2)
    if torch.cuda.get_device_capability() != (9,0):
        with pytest.raises(RuntimeError):
            ext.kernel_resources(1)
        return
    for output_type in (0,1):
        resources = ext.kernel_resources(output_type)
        assert set(resources) == {"smooth_vc7","full_vc7","smooth_optimized","full_optimized"}
        for name, row in resources.items():
            assert row["dynamic_shared_bytes"] == 40960
            assert row["threads_per_block"] == 128
            assert row["predicted_active_blocks_per_sm"] > 0
            if "optimized" in name:
                assert row["registers_per_thread"] <= 168


def test_both_native_value_quantizers_against_independent_statistics():
    validator = load("vc_preprocess_validator",ROOT / "scripts/validate_sage_vc.py")
    result = validator.validate_preprocessing(ext,oracle)
    assert result["value_cases"] == 60


@pytest.mark.parametrize("n",[1,129,4097])
def test_channel_scale_constant_and_large_means_no_overflow_and_stream(n):
    # Different nonzero constants in every block/channel/head; padded rows must
    # not enter statistics. Means above FP16 range exercise the protective scale.
    v = torch.zeros(2,3,n,64,device="cuda",dtype=torch.float32)
    for t,start in enumerate(range(0,n,128)):
        v[:,:,start:start+128] = (torch.arange(64,device="cuda")-31) * (t+1) * 10000.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        packed,scales,means = ext.prepare_values(v)
        decoded = packed.float().index_select(-1,oracle.value_permutation(packed.shape[-1],"cuda"))
        restored = means.float() * scales.unsqueeze(2)
    stream.synchronize()
    assert scales.shape == (2,3,64)
    assert torch.isfinite(scales).all() and torch.isfinite(means).all()
    assert torch.count_nonzero(decoded) == 0
    torch.testing.assert_close(restored,v[:,:,::128],rtol=5e-4,atol=1e-6)


def test_channel_scale_reduction_includes_tiles_beyond_one_warp():
    torch.manual_seed(919)
    v = torch.randn(2,3,4353,128,device="cuda") * .01
    v[:,:,4224:4352] *= 1000  # Largest residual in tile 33, after the first warp pass.
    v[:,:,-1] = 90000        # One valid row: zero residual, nonzero mean constraint.
    packed,scales,means = ext.prepare_values(v)
    blocks = [v[:,:,i:i+128] for i in range(0,v.shape[2],128)]
    raw_means = torch.stack([x.mean(2) for x in blocks],2)
    peaks = torch.stack([(x-x.mean(2,keepdim=True)).abs().amax(2) for x in blocks],2).amax(2)
    expected = torch.maximum(peaks/448,raw_means.abs().amax(2)/65504)
    torch.testing.assert_close(scales,expected,rtol=2e-5,atol=1e-7)
    assert torch.isfinite(packed.float()).all() and torch.isfinite(means).all()
    torch.testing.assert_close(means.float()*scales.unsqueeze(2),raw_means,rtol=1e-3,atol=1e-5)
