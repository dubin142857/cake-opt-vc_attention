#!/usr/bin/env python3
"""Fixpipe Probe 的单臂 driver: 只跑 control, 不跑 expcast (两者共用核名, 混采无法分离)。
用法: driver.py <device> <tile_m> <reps>   —— device 是**物理**卡号 (msOpProf 按物理归属)。
它把 so 的 sha256 打到 stdout, 由外层写进采集目录 (E11)。
"""
import hashlib, os, sys
import torch, torch_npu  # noqa: F401
import vc_attention_ascend as vc

dev_i, tile_m, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
so = os.path.join(os.path.dirname(vc.__file__), "libvc_ascend_ops.so")
print("SO_SHA256", hashlib.sha256(open(so, "rb").read()).hexdigest(), flush=True)

torch.npu.set_device(f"npu:{dev_i}")
torch.manual_seed(0)
q = torch.randn(1, 37741, 28, 128, device=f"npu:{dev_i}", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
payload = vc.prepare(q, k, v, layout="NHD")
del q, k, v
torch.npu.synchronize()
#  ★prepare() 在采集之外没法做 —— msprof 从进程起点就采。靠 --kernel-name 过滤内核。
for _ in range(reps):
    vc.run_prepared(payload, expcast=False, tile_m=tile_m)
    torch.npu.synchronize()
print("DRIVER_OK", tile_m, reps, flush=True)
