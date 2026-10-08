"""Mean restoration must sum four lanes once and address every head/tile/channel."""
import pytest
import torch

probe = pytest.importorskip("_sage_vc_mean_probe")


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("mean_only", [False, True])
@pytest.mark.parametrize("side_stream", [False, True])
def test_mean_restoration_matches_independent_fp64_outer_product(dim, mean_only, side_stream):
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        torch.manual_seed(3131 + dim)
        history = torch.randn(2, 3, 128, dim//16, 8, device="cuda")
        products = torch.zeros_like(history) if mean_only else torch.randn_like(history)
        mass = torch.randint(0, 100, (2, 3, 128, 2), device="cuda").float() / 8
        original_mass = mass.clone()
        means = torch.randn(2, 3, dim, device="cuda", dtype=torch.float16)
        result = probe.run(history, products, mass, means)
        row_mass = mass.reshape(2, 3, 32, 4, 2).double().sum(3).repeat_interleave(4, 2)
        lane = torch.arange(128, device="cuda")[:, None, None]
        fragment = torch.arange(dim//16, device="cuda")[None, :, None]
        slot = torch.arange(8, device="cuda")[None, None, :]
        channels = fragment*16 + (lane % 4)*2 + (slot//4)*8 + slot%2
        indexed_means = means.double()[..., channels]
        indexed_mass = row_mass[..., (slot.flatten() % 4)//2].unsqueeze(-2)
        # One rounded FMA into the product, then a separate FP32 history add.
        expected = history + (products.double() + indexed_mass*indexed_means).float()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(mass, original_mass, rtol=0, atol=0)
    stream.synchronize()
