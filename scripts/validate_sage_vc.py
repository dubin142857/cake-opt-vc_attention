"""Validate the Sage+VC experiment, separately from the unchanged Sage baseline.

Implementation gates use decoded quantized payloads. Dense FP32 errors are
reported separately and are not a model-quality acceptance criterion.
"""
import argparse
import json
from pathlib import Path

import torch


def metrics(actual, expected):
    a, e = actual.double(), expected.double()
    return {"rel_rmse": ((a-e).square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-12)).item(),
            "max_abs": (a-e).abs().max().item()}


@torch.no_grad()
def validate_preprocessing(ext, oracle, device="cuda:0"):
    """Runs also on a local non-Hopper diagnostic build; never calls attention."""
    shifted = torch.cat((torch.linspace(-15, 1, 65537, device=device),
                         torch.tensor([-float("inf"), -20., 0.], device=device)))
    torch.testing.assert_close(ext.expcast_codes(shifted), oracle.expcast_codes(shifted), rtol=0, atol=0)
    count = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for dim in (64, 128):
            for n in (1, 127, 128, 129, 257):
                torch.manual_seed(339)
                v = torch.randn(2, 2, n, dim, device=device, dtype=dtype)
                # Unequal block means and scales expose metadata/tail/index errors.
                for t, begin in enumerate(range(0, n, 128)):
                    v[:, :, begin:begin+128] = v[:, :, begin:begin+128] * (t+1) + t * 3
                v[..., 0] = 1.5
                v[..., 1] = -2.0
                packed, scales, means = ext.prepare_values(v)
                unpacked = packed.float().index_select(-1, oracle.value_permutation(packed.shape[-1], device)).transpose(-1, -2)
                for t, begin in enumerate(range(0, n, 128)):
                    rows = v[:, :, begin:begin+128].float()
                    mean = rows.mean(2)
                    residual = rows - mean.unsqueeze(2)
                    peak = residual.abs().amax(2)
                    scale = torch.where(peak > 0, peak / 448., 1.)
                    scale = torch.maximum(scale, mean.abs() / 65504.)
                    torch.testing.assert_close(scales[:, :, t], scale, rtol=2e-5, atol=1e-7)
                    torch.testing.assert_close(means[:, :, t].float() * scales[:, :, t], mean,
                                               rtol=8e-4, atol=2e-6)
                    decoded = unpacked[:, :, begin:begin+rows.shape[2]]
                    # Quantizer has ~6.25% maximum relative rounding error in normal FP8 range.
                    err = (decoded * scales[:, :, t].unsqueeze(2) - residual).abs()
                    bound = residual.abs() * 0.063 + scales[:, :, t].unsqueeze(2) / 512 + 3e-6
                    if not torch.all(err <= bound).item():
                        raise AssertionError(f"V residual/packing mismatch: {dtype}, D={dim}, N={n}")
                if n % 128:
                    assert torch.count_nonzero(unpacked[:, :, n:]).item() == 0, "Nonzero padded V"
                count += 1
    # Stable grouping of separated clusters, plus a bijection check for every head.
    labels = torch.tensor([2, 0, 1, 2, 1, 0], device=device)
    values = labels.float().reshape(1,1,6,1).expand(2,2,6,64).contiguous()
    permutation = ext.group_values(values, 3, 2)
    expected_ids = torch.arange(6, device=device).expand(2,2,6)
    torch.testing.assert_close(permutation.sort(-1).values, expected_ids, rtol=0, atol=0)
    grouped = labels[permutation]
    for rows, ids in zip(grouped.reshape(-1,6), permutation.reshape(-1,6)):
        # Initialization chooses labels 2,1,0. Each group must remain contiguous and stable.
        assert rows.tolist() == [2,2,1,1,0,0]
        assert ids.tolist() == [0,3,2,4,1,5]
    torch.cuda.synchronize(device)
    return {"value_cases": count, "expcast_codes": shifted.numel(), "stable_clustering": "passed"}


@torch.no_grad()
def validate(device="cuda:0"):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("Full Sage VC validation requires Hopper")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    from sageattention_local import sageattn, sageattn_vc, sage_vc_build_info
    from sageattention_local import vc, vc_reference as oracle
    info = sage_vc_build_info()
    ext = vc._extension()
    report = {"build": info, "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device)}, "preprocessing": validate_preprocessing(ext, oracle, device),
              "cases": [], "implementation_rel_rmse_limit": 0.01,
              "scope": "Synthetic implementation regression only; no model quality or speed acceptance."}
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for nq, nk in ((1,1), (63,127), (64,128), (65,129), (129,257), (257,385)):
                torch.manual_seed(337)
                batch, heads = (2,3) if (nq,nk) == (65,129) else (1,2)
                q, k, v = (torch.randn(batch,heads,n,dim,device=device,dtype=dtype) for n in (nq,nk,nk))
                dense = torch.softmax((q.float() @ k.float().transpose(-1,-2)) * dim**-0.5, -1) @ v.float()
                baseline = sageattn(q,k,v,tensor_layout="HND",is_causal=False)
                for cast, smooth in ((False,False),(True,False),(False,True),(True,True)):
                    state = vc.prepare(q,k,v,expcast=cast,v_smooth=smooth)
                    actual = vc.forward_prepared(state)
                    expected = oracle.reference(state).to(dtype)
                    torch.cuda.synchronize(device)
                    if not torch.isfinite(actual).all().item():
                        raise AssertionError("Native Sage VC output is not finite")
                    agreement = metrics(actual, expected)
                    if agreement["rel_rmse"] > 0.01:
                        raise AssertionError(f"Native/payload-reference mismatch: {dtype}, {dim}, {nq}/{nk}, "
                                             f"expcast={cast}, smooth={smooth}, {agreement}")
                    row = {"dtype": str(dtype), "dim": dim, "batch": batch, "heads": heads, "queries": nq, "keys": nk,
                           "expcast": cast, "v_smooth": smooth, "native_vs_payload_reference": agreement,
                           "native_vs_dense_fp32": metrics(actual,dense)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 3:
                        old = vc.forward_prepared(state,kernel_mode="vc4")
                        delta = metrics(actual,old)
                        if not torch.isfinite(old).all().item() or delta["rel_rmse"] > 0.001:
                            raise AssertionError(f"Optimized/vc4 regression: {row}, delta={delta}")
                        row["optimized_vs_vc4"] = delta
                    if not cast and not smooth:
                        row["control_vs_sage_local"] = metrics(actual,baseline)
                        if row["control_vs_sage_local"]["rel_rmse"] > 0.001:
                            raise AssertionError(f"Derived control differs from original Sage: {row}")
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
    # A contiguous view can still start at an odd element. Reject misaligned
    # metadata before dispatching the paired-load kernel (vc4 and later).
    if getattr(ext, "sage_vc_kernel_version", 0) >= 2:
        for field in ("vs", "means"):
            original = state[field]
            view = torch.empty(original.numel()+1,device=device,dtype=original.dtype)[1:].view_as(original)
            view.copy_(original)
            bad_state = {**state,field:view}
            try:
                vc.forward_prepared(bad_state)
            except RuntimeError as exc:
                if "pointer alignment" not in str(exc):
                    raise
            else:
                raise AssertionError(f"Unaligned {field} was not rejected")
        report["unaligned_metadata_cases"] = 2
    # Uniform attention over three unequal blocks: catches missing mean correction,
    # tail inclusion, incorrect denominator, and applying just the last V scale.
    for dtype in (torch.float16, torch.bfloat16):
        q = torch.zeros(1,2,65,128,device=device,dtype=dtype)
        k = torch.zeros(1,2,257,128,device=device,dtype=dtype)
        channels = torch.linspace(-2,2,128,device=device)
        v = torch.empty_like(k)
        v[:,:,:128] = channels
        v[:,:,128:256] = channels * 2 + 3
        v[:,:,256:] = channels * 0.25 - 6
        expected = v.float().mean(2,keepdim=True).expand_as(q).to(dtype)
        for cast in (False,True):
            out = sageattn_vc(q,k,v,expcast=cast,v_smooth=True)
            torch.testing.assert_close(out,expected,rtol=0.004,atol=0.004)
        # Both layout paths and public API must agree for random, nonzero data too.
        torch.manual_seed(338)
        q,k,v = (torch.randn_like(x) for x in (q,k,v))
        hnd = sageattn_vc(q,k,v)
        nhd = sageattn_vc(*(x.transpose(1,2).contiguous() for x in (q,k,v)),tensor_layout="NHD")
        torch.testing.assert_close(hnd,nhd.transpose(1,2),rtol=0,atol=0)
    torch.cuda.synchronize(device)
    report.update(status="passed", known_answer_cases=4, layout_cases=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/sage_vc_validation.json")
    args = parser.parse_args()
    report = validate(args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Sage VC checks passed; saved to {path}")
