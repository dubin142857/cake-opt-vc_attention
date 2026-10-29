"""测一个**已安装**的构建: 固定 payload, 打印一行 JSON。

交付口径: 只计 run_prepared() 的原生核心调用, prepare() 在计时之外;
长形状 [1, 37741, 28, 128] BF16, tile_m=64; 每次 warmup 2 遍、计时 5 遍取中位。
每行都记下所用 .so 的 sha256 前 16 位 —— 事后可核对"这一趟跑的到底是哪个构建"。

单跑:  python3 scripts/bench_one.py --device 1 --label h26
成对跑: 见 scripts/bench_pairwise.sh
"""
import argparse, hashlib, json, os, statistics, time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="0", help="NPU 卡号 (逻辑号)")
    ap.add_argument("--label", default="?", help="这一趟的标签, 原样写进输出")
    ap.add_argument("--queries", type=int, default=37741)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--dim", type=int, default=128)
    #  tile_m 现在只支持 64 (手写 cube 的几何)。
    ap.add_argument("--tile-m", type=int, default=64, choices=(64,))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401
    import vc_attention_ascend as vc

    so = os.path.join(os.path.dirname(vc.__file__), "libvc_ascend_ops.so")
    digest = hashlib.sha256(open(so, "rb").read()).hexdigest()

    dev = f"npu:{a.device}"
    torch.npu.set_device(dev)
    torch.manual_seed(0)
    q = torch.randn(1, a.queries, a.heads, a.dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    payload = vc.prepare(q, k, v, layout="NHD")
    del q, k, v
    torch.npu.synchronize()

    out = {
        "label": a.label,
        "so_sha256": digest,
        "so_sha256_16": digest[:16],
        #  口径一起写进每一行 —— 两个口径不同的数不可相减。
        "config": {"queries": a.queries, "keys": a.queries, "heads": a.heads,
                   "dim": a.dim, "dtype": "bfloat16", "tile_m": a.tile_m,
                   "layout": "NHD", "warmup": a.warmup, "reps": a.reps,
                   "timed": "run_prepared() only; prepare() excluded"},
        "t": time.time(),
    }
    #  ★指标名必须由实际的 tile_m 生成。此前写死成 _m64, 传 --tile-m 32 时
    #  输出的键仍叫 control_m64 —— 标签与口径不符, 日后无法归属。
    for mode, cast in ((f"control_m{a.tile_m}", False), (f"expcast_m{a.tile_m}", True)):
        for _ in range(a.warmup):
            vc.run_prepared(payload, expcast=cast, tile_m=a.tile_m)
            torch.npu.synchronize()
        xs = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            r = vc.run_prepared(payload, expcast=cast, tile_m=a.tile_m)
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
            del r
        out[mode] = statistics.median(xs)
        out[mode + "_samples"] = xs
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
