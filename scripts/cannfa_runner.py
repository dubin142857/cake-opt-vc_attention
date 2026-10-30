import argparse, math, sys, time
import torch, torch_npu
ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=3)
ap.add_argument("--queries", type=int, default=37741)
ap.add_argument("--keys", type=int, default=37741)
ap.add_argument("--heads", type=int, default=28)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--launches", type=int, default=3)
ap.add_argument("--time", action="store_true")
a = ap.parse_args()
torch.npu.set_device(a.device); dev = f"npu:{a.device}"
g = torch.Generator().manual_seed(0)
q, k, v = [torch.randn((1, a.heads, n, a.dim), generator=g).bfloat16().to(dev)
           for n in (a.queries, a.keys, a.keys)]
def run():
    return torch_npu.npu_fusion_attention(q, k, v, a.heads, "BNSD",
        scale=1.0/math.sqrt(a.dim), keep_prob=1.0,
        pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0]
run(); torch.npu.synchronize()
if a.time:
    for _ in range(3):
        t0 = time.perf_counter(); run(); torch.npu.synchronize()
        print(f"  CANN FA: {(time.perf_counter()-t0)*1000:.2f} ms")
else:
    for _ in range(a.launches): run()
    torch.npu.synchronize()
    print(f"launched {a.launches}")
