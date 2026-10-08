"""vc27 correctness gates. Run before performance; no ncu or network needed."""
import argparse
import json
from pathlib import Path
import torch


def quant_reference(x, rows, mean=None):
    x = x.float()
    if mean is not None:
        x = x - mean.unsqueeze(2)
    packed, scales = [], []
    for start in range(0, x.shape[2], rows):
        block = x[:, :, start:start+rows]
        # Emulate __fdiv_rn, not TensorIterator's scalar reciprocal-multiply.
        scale = (block.abs().amax((-2, -1)).double() / 448).float().clamp_min(1e-12)
        normalized = (block.double() / scale[..., None, None].double()).float()
        packed.append(normalized.clamp(-448, 448).to(torch.float8_e4m3fn))
        scales.append(scale)
    return torch.cat([x.view(torch.uint8) for x in packed], 2).view(torch.float8_e4m3fn), torch.stack(scales, 2)


@torch.no_grad()
def validate_quantization(ext):
    torch.manual_seed(2701)
    checked = 0
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for nq, nk in ((1,1),(63,127),(64,128),(65,129),(193,513),(129,4097)):
                q = torch.randn(2,3,nq,dim,device="cuda",dtype=dtype)
                k = torch.randn(2,3,nk,dim,device="cuda",dtype=dtype)
                # Distinct channel means and per-block scales catch wrong indices.
                k.add_(torch.arange(dim,device="cuda",dtype=dtype).view(1,1,1,-1) / 4)
                q[:,:,64:] *= 3
                current = torch.cuda.current_stream()
                side = torch.cuda.Stream()
                side.wait_stream(current)
                with torch.cuda.stream(side):
                    q8,qs,k8,ks,mean = ext.prepare_qk_fp8(q,k)
                current.wait_stream(side)
                for t in (q8,qs,k8,ks,mean): t.record_stream(current)
                torch.testing.assert_close(mean, k.double().mean(2).float(), rtol=2e-5, atol=1e-5)
                for actual,scales,x,rows,center in ((q8,qs,q,64,None),(k8,ks,k,128,mean)):
                    expected, expected_scales = quant_reference(x,rows,center)
                    torch.testing.assert_close(scales,expected_scales,rtol=0,atol=0)
                    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)), "FP8 QK byte mismatch"
                checked += 1
            for value in (0., 8., 10000.):
                q=torch.full((1,2,65,dim),value,device="cuda",dtype=dtype)
                k=torch.full((1,2,129,dim),value,device="cuda",dtype=dtype)
                q8,qs,k8,ks,mean=ext.prepare_qk_fp8(q,k)
                assert torch.isfinite(q8.float()).all() and torch.isfinite(qs).all()
                assert torch.count_nonzero(k8.float()) == 0
                torch.testing.assert_close(mean,k.float().mean(2),rtol=0,atol=0)
                assert torch.all(ks == 1e-12)
                checked += 1
    return {"cases":checked,"qk_bytes_and_scales":"exact against PyTorch cast with native K mean",
            "key_mean":"checked independently against FP64 statistics", "stream":"nondefault with explicit dependencies"}


def metrics(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Nonfinite or incorrectly shaped FP8 attention output")
    delta=actual.float()-expected.float()
    return {"relative_rmse":(delta.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12)).item(),
            "max_abs":delta.abs().max().item()}


def check_implementation(actual, expected):
    error=metrics(actual,expected)
    # Decoded FP32 matmul is an oracle, not a bit-level Hopper FP8 accumulator
    # emulator. The 1% gate includes QK/PV accumulation and P byte-boundary effects.
    # It says nothing about error vs dense attention or a model's quality.
    if error["relative_rmse"] > .01:
        raise AssertionError(f"FP8 implementation/decoded-oracle error exceeds 1%: {error}")
    return error


@torch.no_grad()
def validate_attention(api, oracle):
    torch.manual_seed(2702)
    rows=[]
    side=torch.cuda.Stream()
    for dtype in (torch.float16,torch.bfloat16):
        for dim in (64,128):
            for nq,nk in ((1,1),(17,63),(63,127),(64,128),(65,129),(127,255),(128,256),(129,257),(193,513)):
                q=torch.randn(2,2,nq,dim,device="cuda",dtype=dtype)
                k=torch.randn(2,2,nk,dim,device="cuda",dtype=dtype)
                v=torch.randn_like(k)
                state=api.prepare(q,k,v)
                for cast in (False,True):
                    state["expcast"]=cast
                    current=torch.cuda.current_stream()
                    side.wait_stream(current)
                    with torch.cuda.stream(side): out=api.forward_prepared(state)
                    current.wait_stream(side)
                    out.record_stream(current)
                    expected=oracle.reference(state,chunk=32).to(dtype)
                    error=check_implementation(out,expected)
                    # All output rows; public HND/NHD include independent preprocessing.
                    public=api.sageattn_vc_fp8(q,k,v,expcast=cast)
                    torch.testing.assert_close(public,out,rtol=0,atol=0)
                    nhd=api.sageattn_vc_fp8(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                          tensor_layout="NHD",expcast=cast)
                    torch.testing.assert_close(nhd.transpose(1,2),out,rtol=0,atol=0)
                    row=dict(dtype=str(dtype),dim=dim,queries=nq,keys=nk,expcast=cast,**error)
                    rows.append(row)
                    print(json.dumps(row),flush=True)
            # Known constant output catches scale, tail masks, and V layout bugs.
            q=torch.zeros(1,2,65,dim,device="cuda",dtype=dtype)
            k=torch.zeros(1,2,257,dim,device="cuda",dtype=dtype)
            v=torch.arange(dim,device="cuda",dtype=dtype).remainder(8).view(1,1,1,dim).expand_as(k).contiguous()/8
            for cast in (False,True):
                out=api.sageattn_vc_fp8(q,k,v,expcast=cast)
                torch.testing.assert_close(out,v[:,:,:1].expand_as(out),rtol=.004,atol=.002)
            with torch.cuda.stream(side):
                try: api.sageattn_vc_fp8(q,k,v)
                except ValueError as e:
                    assert "default CUDA stream" in str(e)
                else: raise AssertionError("Public default-stream guard missing")
    return {"all_row_cases":rows,"known_constant_cases":8,"public_stream_guard_cases":4,
            "scope":"Native/decoded FP8 oracle, not dense-attention quality", "implementation_gate_relative_rmse":.01}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/h200_vc27_validation.json"))
    args=parser.parse_args()
    from sageattention_local import vc_fp8, vc_reference
    ext=vc_fp8._extension()
    if torch.cuda.get_device_capability() != (9,0): raise RuntimeError("Hopper required for attention validation")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report={"version":27,"native_kernel_version":ext.sage_vc_kernel_version,
            "device":torch.cuda.get_device_name(),"quantization":validate_quantization(ext),
            "attention":validate_attention(vc_fp8,vc_reference),"status":"passed"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("FP8 QK validation passed:",args.output,flush=True)


if __name__ == "__main__": main()
