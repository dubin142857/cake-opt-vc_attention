"""Native fused Lloyd grouping, independently checked on any CUDA GPU."""
import pytest
import torch

ext = pytest.importorskip("_vc_sm90")


def oracle(v, clusters, iterations):
    # CUDA FP32 linspace is part of the initialization contract. All distance
    # and centroid math below is independent FP64, with direct squared deltas.
    count = min(clusters, v.shape[2])
    ids = torch.linspace(0, v.shape[2]-1, count, device=v.device).long()
    x = v.double()
    centers = x.index_select(2, ids)
    for step in range(iterations):
        distances = (x.unsqueeze(-2)-centers.unsqueeze(2)).square().sum(-1)
        labels = distances.argmin(-1)
        if step+1 < iterations:
            centers = torch.stack([
                torch.where((labels == c).sum(-1, keepdim=True) > 0,
                    (x*(labels == c).unsqueeze(-1)).sum(2)/(labels == c).sum(-1, keepdim=True).clamp_min(1),
                    centers[:,:,c]) for c in range(count)], 2)
    return labels, centers


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("n,clusters,iterations", [(1,8,2),(7,8,1),(127,3,3),(128,1,2),(129,8,2),(2049,8,2)])
def test_fused_matches_independent_lloyd_and_stable_sort(dtype, dim, n, clusters, iterations):
    torch.manual_seed(211+n)
    v = torch.randn(2,2,n,dim,device="cuda",dtype=dtype)
    permutation, labels, centers = ext.group_values_debug(v,clusters,iterations)
    expected, expected_centers = oracle(v,clusters,iterations)
    # Fixed seeds here are away from decision boundaries. This is a correctness
    # test, not a claim of bitwise agreement for arbitrary floating-point inputs.
    assert torch.equal(labels,expected)
    torch.testing.assert_close(centers.double(),expected_centers,rtol=2e-5,atol=2e-6)
    assert torch.equal(permutation,expected.argsort(dim=-1,stable=True))
    assert torch.equal(permutation,ext.group_values(v,clusters,iterations))


@pytest.mark.parametrize("iterations", [1,2,3])
def test_empty_clusters_ties_and_current_stream(iterations):
    v = torch.full((2,3,257,128), 2., device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        permutation,labels,centers = ext.group_values_debug(v,8,iterations)
        result = labels.sum()
    stream.synchronize()
    assert result.item() == 0
    assert torch.equal(permutation,torch.arange(257,device="cuda").expand(2,3,257))
    assert (centers == 2).all()  # Empty groups keep their initialized center.


def test_duplicate_centers_lowest_id_and_legacy_fallback():
    x = torch.tensor([2,0,1,2,1,0],device="cuda",dtype=torch.float32)
    v = x.view(1,1,6,1).expand(1,1,6,64).contiguous()
    expected = torch.tensor([0,3,2,4,1,5],device="cuda").view(1,1,6)
    assert torch.equal(ext.group_values(v,3,2),expected)
    assert torch.equal(ext.group_values_legacy(v,3,2),expected)
    v = torch.arange(17,device="cuda",dtype=torch.float32).view(1,1,17,1).expand(1,1,17,128).contiguous()
    assert torch.equal(ext.group_values(v,9,2),ext.group_values_legacy(v,9,2))


@pytest.mark.parametrize("kind", ["cpu","dtype","dim","empty","strided","clusters0","clusters257","effective9","iterations0","iterations101"])
def test_fused_rejects_unsupported_inputs(kind):
    v = torch.zeros(1,1,17,64,device="cuda")
    clusters,iterations = 8,2
    if kind == "cpu": v = v.cpu()
    if kind == "dtype": v = v.to(torch.int8)
    if kind == "dim": v = v[:,:,:,:32].contiguous()
    if kind == "empty": v = v[:,:,:0]
    if kind == "strided": v = v[:,:,::2]
    if kind == "clusters0": clusters = 0
    if kind == "clusters257": clusters = 257
    if kind == "effective9": clusters = 9
    if kind == "iterations0": iterations = 0
    if kind == "iterations101": iterations = 101
    with pytest.raises(RuntimeError):
        ext.group_values_debug(v,clusters,iterations)
