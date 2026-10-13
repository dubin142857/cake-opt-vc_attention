"""Exact FP8 partial-denominator gate, including subnormal and lane isolation."""
import torch


def byte_cases(device):
    # Each row holds one thread's 16 words (64 bytes): four K32 groups,
    # two K16 fragments, two query rows, four bytes per word.
    shape=(128,4,2,2,4)
    for code in range(121):
        yield f"constant_{code}",torch.full(shape,code,dtype=torch.uint8,device=device)
    # Every byte location independently lit across all lanes/warps: catches
    # wrong MMA fragment mapping and cross-thread mixing in the selector B.
    for location in range(64):
        p=torch.zeros((128,64),dtype=torch.uint8,device=device)
        p[:,location]=(torch.arange(128,device=device)%121).byte()
        yield f"impulse_{location}",p.reshape(shape)
    for tiny in range(1,8):
        p=torch.full(shape,120,dtype=torch.uint8,device=device)
        p.reshape(128,64)[:,::2]=tiny
        yield f"max_and_subnormal_{tiny}",p
    gen=torch.Generator(device=device).manual_seed(4300)
    yield "random_lane_warp",torch.randint(0,121,(4096,4,2,2,4),device=device,dtype=torch.uint8,generator=gen)


@torch.no_grad()
def check_mass_case(ext,name,codes,side_stream=False):
    packed=codes.contiguous().view(torch.int32).reshape(-1,16)
    stream=torch.cuda.Stream(device=codes.device) if side_stream else torch.cuda.current_stream(codes.device)
    stream.wait_stream(torch.cuda.current_stream(codes.device))
    with torch.cuda.stream(stream):out=ext.check_mass(packed)
    stream.synchronize()
    # Independent exact integer-unit decoder: subnormals are not flushed.
    c=codes.to(torch.int64)
    units=torch.where(c<8,c,((c&7)+8)<<((c>>3)-1).clamp_min(0))
    expected=units.sum((1,2,4)).double().mul(2**-9).float()
    if not torch.equal(out.view(torch.uint8),expected.view(torch.uint8)):
        error=(out-expected).abs().max().item()
        raise AssertionError(f"Tensor mass exact gate failed: {name}, max_abs={error}; do not benchmark")
    return {"case":name,"rows":codes.shape[0],"side_stream":side_stream,"exact":True}


@torch.no_grad()
def validate(ext,device,progress=lambda rows:None):
    rows=[]
    for name,codes in byte_cases(device):
        row=check_mass_case(ext,name,codes,side_stream=(name=="random_lane_warp"))
        rows.append(row);progress(rows)
    native=torch.cuda.get_device_capability(device)>=(8,9)
    print(f"Tensor mass: {len(rows)} byte/mapping cases passed; native FP8 decode={native}; FP16 MMA/FP32 accumulation",flush=True)
    return {"status":"passed","native_fp8_decode":native,"cases":rows,
            "scope":"Each thread's partial denominator, including nonzero FP8 subnormal bytes; exact integer oracle. FP8->FP16 decode is exact; all nonzero decoded values are normal FP16. The reduction uses FP16 MMA with FP32 accumulation on both SM86 and Hopper; SM86 uses software FP8 decoding."}
