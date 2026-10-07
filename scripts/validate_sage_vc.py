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
                blocks = []
                for begin in range(0,n,128):
                    rows = v[:, :, begin:begin+128].float()
                    mean = rows.mean(2)
                    residual = rows - mean.unsqueeze(2)
                    peak = residual.abs().amax(2)
                    blocks.append((mean,residual,peak))
                for channel in (False,True):
                    packed, scales, means = ext.prepare_values(v,channel_scale=channel)
                    unpacked = packed.float().index_select(-1, oracle.value_permutation(packed.shape[-1], device)).transpose(-1, -2)
                    if channel:
                        peak = torch.stack([b[2] for b in blocks]).amax(0)
                        mean_peak = torch.stack([b[0].abs() for b in blocks]).amax(0)
                        expected_scale = torch.maximum(torch.where(peak > 0, peak / 448., 1.),mean_peak/65504.)
                        assert scales.shape == (2,2,dim)
                        torch.testing.assert_close(scales,expected_scale,rtol=2e-5,atol=1e-7)
                    for t,(mean,residual,peak) in enumerate(blocks):
                        scale = scales if channel else scales[:,:,t]
                        if not channel:
                            expected_scale = torch.maximum(torch.where(peak > 0,peak/448.,1.),mean.abs()/65504.)
                            torch.testing.assert_close(scale,expected_scale,rtol=2e-5,atol=1e-7)
                        torch.testing.assert_close(means[:,:,t].float()*scale,mean,rtol=8e-4,atol=2e-6)
                        decoded = unpacked[:,:,t*128:t*128+residual.shape[2]]
                        err = (decoded * scale.unsqueeze(2) - residual).abs()
                        bound = residual.abs() * .063 + scale.unsqueeze(2) / 512 + 3e-6
                        if not torch.all(err <= bound).item():
                            raise AssertionError(f"V residual/packing mismatch: {dtype}, D={dim}, N={n}, channel={channel}")
                    if n % 128:
                        assert torch.count_nonzero(unpacked[:,:,n:]).item() == 0, "Nonzero padded V"
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
def validate_grouping(ext, device="cuda:0"):
    """Well-separated Lloyd cases checked with independent FP64 distances.

    Includes unaligned tails and empty centers. Exact labels here do not imply
    universal equality at FP32 decision boundaries on arbitrary model inputs.
    """
    cases = 0
    for dtype in (torch.float16,torch.bfloat16,torch.float32):
        for dim in (64,128):
            for n,count,iterations in ((1,8,2),(127,3,3),(129,8,2),(2049,8,2)):
                torch.manual_seed(883)
                ids = torch.linspace(0,n-1,min(n,count),device=device).long()
                # Give the initialized centers distinct, separated values.
                labels = torch.arange(n,device=device) % min(n,count)
                labels[ids] = torch.arange(min(n,count),device=device)
                v = (labels.view(1,1,n,1)*4 + torch.randn(2,2,n,dim,device=device)*.01).to(dtype)
                permutation,actual,centers = ext.group_values_debug(v,count,iterations)
                x = v.double()
                expected_centers = x.index_select(2,ids)
                for step in range(iterations):
                    distance = (x.unsqueeze(-2)-expected_centers.unsqueeze(2)).square().sum(-1)
                    expected = distance.argmin(-1)
                    if step+1 < iterations:
                        expected_centers = torch.stack([
                            torch.where((expected == c).sum(-1,keepdim=True) > 0,
                                (x*(expected == c).unsqueeze(-1)).sum(2)/(expected == c).sum(-1,keepdim=True).clamp_min(1),
                                expected_centers[:,:,c]) for c in range(min(n,count))],2)
                assert torch.equal(actual,expected)
                assert torch.equal(permutation,expected.argsort(dim=-1,stable=True))
                torch.testing.assert_close(centers.double(),expected_centers,rtol=2e-5,atol=2e-6)
                assert torch.equal(permutation,ext.group_values_legacy(v,count,iterations))
                cases += 1
    v = torch.full((2,2,257,128),2.,device=device)
    permutation,labels,centers = ext.group_values_debug(v,8,3)
    assert torch.count_nonzero(labels).item() == 0
    assert torch.equal(permutation,torch.arange(257,device=device).expand(2,2,257))
    assert (centers == 2).all().item()
    return {"fp64_lloyd_cases":cases,"empty_clusters_and_stable_ties":"passed"}


@torch.no_grad()
def validate_pipeline(vc, device, *, compute=False, history=False, work=False):
    """Exercise both slot parities, reuse, partial tails and long lookahead loops."""
    from sageattention_local.vc_reference import value_permutation
    minimum=19 if work else 18 if history else 17 if compute else 16
    if vc._extension().sage_vc_kernel_version < minimum:
        raise RuntimeError("Rebuild sageattention-local for the requested pipeline validation")
    modes=vc.WORK_CANDIDATES if work else vc.HISTORY_CANDIDATES if history else vc.COMPUTE_CANDIDATES if compute else vc.PIPELINE_CANDIDATES
    label="Work reduction" if work else "Shared history" if history else "Compute" if compute else "TMA"
    rows = []
    known_answers = 0
    recurrence_cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for dim in ((128,) if history else (64,128)):
            for nk in (1,127,128,129,255,256,257,383,384,385,511,512,513,1025,37741):
                torch.manual_seed(1900+nk)
                batch, heads = (2,3) if nk==513 else (1,1)
                q = torch.randn(batch,heads,65,dim,device=device,dtype=dtype)
                k,v = (torch.randn(batch,heads,nk,dim,device=device,dtype=dtype) for _ in range(2))
                tile = (torch.arange(nk,device=device)//128).view(1,1,nk,1)
                # Unequal adjacent payloads expose reading a previous/next slot.
                k.mul_((1+(tile % 3)*.2).to(dtype))
                v.add_(((tile % 7)-3).to(dtype)*.25)
                state = vc.prepare(q,k,v,expcast=False,v_smooth=False)
                for cast in (False,True):
                    payload = {**state,"expcast":cast}
                    expected = vc.forward_prepared(payload,kernel_mode="vc16")
                    if not torch.isfinite(expected).all().item():
                        raise AssertionError("Nonfinite pipeline baseline")
                    for mode in modes:
                        deltas=[]
                        for repeat in range(3):
                            actual = vc.forward_prepared(payload,kernel_mode=mode)
                            delta = metrics(actual,expected)
                            if not torch.isfinite(actual).all().item() or delta["rel_rmse"] > .001:
                                raise AssertionError(f"Pipeline regression: {dtype}, D{dim}, K{nk}, {cast}, {mode}, repeat {repeat}: {delta}")
                            deltas.append({**delta,"exact_match":torch.equal(actual,expected)})
                        rows.append(dict(dtype=str(dtype),dim=dim,keys=nk,queries=65,batch=batch,heads=heads,
                                         expcast=cast,mode=mode,repeats=deltas))
            # Independent uniform-attention oracle over decoded, unpadded V.
            # Five unequal tiles make stale slots or included padding visible.
            q = torch.zeros(1,1,65,dim,device=device,dtype=dtype)
            k = torch.zeros(1,1,513,dim,device=device,dtype=dtype)
            v = torch.arange(513,device=device).view(1,1,513,1).expand_as(k).to(dtype)/128-2
            if history:
                # Distinct channels/halves expose slot, float4 and epilogue errors.
                v = (v * torch.linspace(.5,1.5,dim,device=device).to(dtype)
                     + torch.arange(dim,device=device).remainder(11).to(dtype)*.125).contiguous()
            payload = vc.prepare(q,k,v,expcast=False,v_smooth=False)
            decoded = payload["v"].float()
            decoded = decoded.index_select(-1,value_permutation(decoded.shape[-1],decoded.device))
            mean = (decoded[...,:513].mean(-1)*payload["vs"]).unsqueeze(2).expand_as(q)
            for cast in (False,True):
                # Sage divides rounded FP8 P by unrounded exp2(P) mass;
                # ExpCast divides by the same decoded P used in its numerator.
                weight = 1. if cast else 448. / (2**float(torch.tensor(8.807)))
                expected = (mean*weight).to(dtype)
                for mode in modes:
                    actual = vc.forward_prepared({**payload,"expcast":cast},kernel_mode=mode)
                    torch.testing.assert_close(actual,expected,rtol=.004,atol=.004)
                    known_answers += 1
            if work:
                # Fixed native operands force tied, falling and increasing
                # maxima across tiles. Different lane scales/row signs expose
                # moving the scale after a shuffle or skipping a partial warp.
                q = torch.zeros(1,2,65,dim,device=device,dtype=dtype)
                k = torch.zeros(1,2,1025,dim,device=device,dtype=dtype)
                state = vc.prepare(q,k,torch.randn_like(k),expcast=True,v_smooth=False)
                signs=(torch.arange(65,device=device).remainder(3)-1).to(torch.int8)
                state["q"].copy_(signs.view(1,1,65,1).expand_as(state["q"]))
                levels=torch.tensor([3,3,-7,7,-2,7,8,5,9],device=device,dtype=torch.int8)
                values=levels[(torch.arange(1025,device=device)//128)]
                state["k"].copy_(values.view(1,1,1025,1).expand_as(state["k"]))
                state["qs"].fill_(1.)
                state["ks"].copy_((torch.arange(state["ks"].shape[-1],device=device).remainder(4)+1).float())
                for scale in (1e-4,1.,1e3):
                    payload={**state,"scale":scale}
                    expected=vc.forward_prepared(payload,kernel_mode="vc16")
                    if not torch.isfinite(expected).all().item():
                        raise AssertionError("Nonfinite max-update baseline")
                    for mode in modes:
                        actual=vc.forward_prepared(payload,kernel_mode=mode)
                        delta=metrics(actual,expected)
                        if not torch.isfinite(actual).all().item() or delta["rel_rmse"]>.001:
                            raise AssertionError(f"Max-update regression: {dtype}, D{dim}, scale={scale}, {mode}: {delta}")
                        recurrence_cases.append(dict(dtype=str(dtype),dim=dim,scale=scale,mode=mode,
                            exact_match=torch.equal(actual,expected),**delta))
            print(f"{label} pipeline slot/tail checks passed: {dtype}, D{dim}",flush=True)
    return {"cases":rows,"known_answer_cases":known_answers,"max_update_cases":recurrence_cases,
            "status":"passed","implementation_rel_rmse_limit":.001,
            "scope":"All rows versus vc16, both algorithms separately, 3 repeats; no model-quality or speed claim."}


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
    if getattr(ext,"sage_vc_preprocess_version",0) >= 1:
        report["grouping"] = validate_grouping(ext,device)
    if getattr(ext,"sage_vc_kernel_version",0) >= 16:
        report["pipeline"] = validate_pipeline(vc,device)
    if getattr(ext,"sage_vc_kernel_version",0) >= 17:
        report["compute_pipeline"] = validate_pipeline(vc,device,compute=True)
    if getattr(ext,"sage_vc_kernel_version",0) >= 18:
        report["shared_history"] = validate_pipeline(vc,device,history=True)
    if getattr(ext,"sage_vc_kernel_version",0) >= 19:
        report["work_reduction"] = validate_pipeline(vc,device,work=True)
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for nq, nk in ((1,1), (63,127), (64,128), (65,129), (129,257), (257,385)):
                torch.manual_seed(337)
                batch, heads = (2,3) if (nq,nk) == (65,129) else (1,2)
                q, k, v = (torch.randn(batch,heads,n,dim,device=device,dtype=dtype) for n in (nq,nk,nk))
                dense = torch.softmax((q.float() @ k.float().transpose(-1,-2)) * dim**-0.5, -1) @ v.float()
                baseline = sageattn(q,k,v,tensor_layout="HND",is_causal=False)
                for cast, smooth, scale_mode in ((False,False,"channel"),(True,False,"channel"),
                                                (False,True,"channel"),(True,True,"channel"),
                                                (False,True,"block"),(True,True,"block")):
                    state = vc.prepare(q,k,v,expcast=cast,v_smooth=smooth,v_scale_mode=scale_mode)
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
                           "expcast": cast, "v_smooth": smooth, "v_scale_mode":scale_mode, "native_vs_payload_reference": agreement,
                           "native_vs_dense_fp32": metrics(actual,dense)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 3:
                        old = vc.forward_prepared(state,kernel_mode="vc4")
                        delta = metrics(actual,old)
                        if not torch.isfinite(old).all().item() or delta["rel_rmse"] > 0.001:
                            raise AssertionError(f"Optimized/vc4 regression: {row}, delta={delta}")
                        row["optimized_vs_vc4"] = delta
                    if getattr(ext,"sage_vc_kernel_version",0) >= 5:
                        previous = vc.forward_prepared(state,kernel_mode="vc6")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > 0.001:
                            raise AssertionError(f"Optimized/vc6 regression: {row}, delta={delta}")
                        row["optimized_vs_vc6"] = {**delta,"exact_match":torch.equal(actual,previous)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 6:
                        previous = vc.forward_prepared(state,kernel_mode="vc7")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > 0.001:
                            raise AssertionError(f"Optimized/vc7 regression: {row}, delta={delta}")
                        row["optimized_vs_vc7"] = {**delta,"exact_match":torch.equal(actual,previous)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 12:
                        previous = vc.forward_prepared(state,kernel_mode="vc14")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > .001:
                            raise AssertionError(f"Optimized/vc14 regression: {row}, delta={delta}")
                        row["optimized_vs_vc14"] = {**delta,"exact_match":torch.equal(actual,previous)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 13 and not smooth:
                        modes = ("vc16","vc16_early_v","vc16_serial") if cast else (
                            "vc16","vc16_early_v","vc16_pack_first","vc16_pack_early")
                        if cast and getattr(ext,"sage_vc_kernel_version",0) >= 14:
                            modes += ("vc17_g1","vc17_g4","vc17_c4","vc17_direct")
                        row["sage_layout_vs_vc15"] = {}
                        for mode in modes:
                            candidate = vc.forward_prepared(state,kernel_mode=mode)
                            delta = metrics(candidate,actual)
                            if not torch.isfinite(candidate).all().item() or delta["rel_rmse"] > .001:
                                raise AssertionError(f"{mode}/vc15 regression: {row}, delta={delta}")
                            row["sage_layout_vs_vc15"][mode] = {**delta,"exact_match":torch.equal(candidate,actual)}
                        if getattr(ext,"sage_vc_kernel_version",0) >= 15:
                            # Compare caps to the SAME algorithm/body at 168 registers.
                            baseline_vc16 = vc.forward_prepared(state,kernel_mode="vc16")
                            row["register_candidates_vs_vc16"] = {}
                            for mode in vc.REGISTER_CANDIDATES:
                                candidate = vc.forward_prepared(state,kernel_mode=mode)
                                delta = metrics(candidate,baseline_vc16)
                                if not torch.isfinite(candidate).all().item() or delta["rel_rmse"] > .001:
                                    raise AssertionError(f"{mode}/vc16 regression: {row}, delta={delta}")
                                row["register_candidates_vs_vc16"][mode] = {**delta,"exact_match":torch.equal(candidate,baseline_vc16)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 10:
                        previous = vc.forward_prepared(state,kernel_mode="vc12")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > .001:
                            raise AssertionError(f"Optimized/vc12 regression: {row}, delta={delta}")
                        row["optimized_vs_vc12"] = {**delta,"exact_match":torch.equal(actual,previous)}
                        serial = vc.forward_prepared(state,kernel_mode="vc13_serial")
                        delta = metrics(serial,previous)
                        if not torch.isfinite(serial).all().item() or delta["rel_rmse"] > .001:
                            raise AssertionError(f"Serial/vc12 regression: {row}, delta={delta}")
                        row["serial_vs_vc12"] = {**delta,"exact_match":torch.equal(serial,previous)}
                        if getattr(vc,"DEFAULT_EXPCAST_KERNEL",None) in ("vc12", "vc14", "vc15"):
                            overlap = vc.forward_prepared(state,kernel_mode="vc13_overlap")
                            delta = metrics(overlap,previous)
                            if not torch.isfinite(overlap).all().item() or delta["rel_rmse"] > .001:
                                raise AssertionError(f"Overlap/vc12 regression: {row}, delta={delta}")
                            row["overlap_vs_vc12"] = {**delta,"exact_match":torch.equal(overlap,previous)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 9:
                        previous = vc.forward_prepared(state,kernel_mode="vc11")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > .001:
                            raise AssertionError(f"Optimized/vc11 regression: {row}, delta={delta}")
                        row["optimized_vs_vc11"] = {**delta,"exact_match":torch.equal(actual,previous)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 8:
                        previous = vc.forward_prepared(state,kernel_mode="vc10")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > .001:
                            raise AssertionError(f"Optimized/vc10 regression: {row}, delta={delta}")
                        row["optimized_vs_vc10"] = {**delta,"exact_match":torch.equal(actual,previous)}
                    if getattr(ext,"sage_vc_kernel_version",0) >= 7:
                        previous = vc.forward_prepared(state,kernel_mode="vc9")
                        delta = metrics(actual,previous)
                        if not torch.isfinite(previous).all().item() or delta["rel_rmse"] > 0.001:
                            raise AssertionError(f"Optimized/vc9 regression: {row}, delta={delta}")
                        row["optimized_vs_vc9"] = {**delta,"exact_match":torch.equal(actual,previous)}
                    if not cast and not smooth:
                        row["control_vs_sage_local"] = metrics(actual,baseline)
                        if row["control_vs_sage_local"]["rel_rmse"] > 0.001:
                            raise AssertionError(f"Derived control differs from original Sage: {row}")
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
    # A contiguous view can still start at an odd element. Reject misaligned
    # metadata before dispatching the paired-load kernel (vc4 and later).
    if getattr(ext, "sage_vc_kernel_version", 0) >= 2:
        for scale_mode in ("block","channel"):
            state = vc.prepare(q,k,v,v_scale_mode=scale_mode)
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
        report["unaligned_metadata_cases"] = 4
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
    parser.add_argument("--pipeline-only", action="store_true",
                        help="vc19 slot/tail/reuse checks only, also suitable for optional compute-sanitizer")
    parser.add_argument("--compute-only", action="store_true", help="vc20 compute carry/tail checks only")
    parser.add_argument("--history-only", action="store_true", help="vc21 FP32 shared history and tail checks only")
    parser.add_argument("--work-only", action="store_true", help="vc22 max/identity rescale and tail checks only")
    args = parser.parse_args()
    if sum((args.pipeline_only,args.compute_only,args.history_only,args.work_only))>1:
        parser.error("Choose --pipeline-only, --compute-only, --history-only or --work-only")
    if args.pipeline_only or args.compute_only or args.history_only or args.work_only:
        torch.cuda.set_device(args.device)
        from sageattention_local import vc
        report = validate_pipeline(vc,args.device,compute=args.compute_only,history=args.history_only,work=args.work_only)
    else:
        report = validate(args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Sage VC checks passed; saved to {path}")
