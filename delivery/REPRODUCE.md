# 复现步骤

从甲方基线 `9e3a550` 复现本交付与其数字。全部命令在
Ascend 910B2 / CANN 9.0.0 / torch 2.9.0 + torch_npu 2.9.0 / Python 3.10 / aarch64 上实跑过。

## 1. 构建

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPU_DEVICE=<空闲卡号> bash scripts/run_ascend910b.sh      # preflight + 构建 + 141 用例验收
```

★**不要直接 `pip install ./ascend`** —— 缺 `SOC_VERSION` 会以
`RuntimeError: Set exact SOC_VERSION from preflight` 失败。`run_ascend910b.sh` 会从
preflight 的 `compiler_soc` 导出它。

★**构建后先核对产物 `sha256` 再进入验收**：

```bash
sha256sum "$(python3 -c 'import vc_attention_ascend,pathlib;print(pathlib.Path(vc_attention_ascend.__file__).with_name("libvc_ascend_ops.so"))')"
# 本交付应为 89386a98f399247d93b0...
```

我们两次被误导过：一次是 `pip install ... | tail -5` 后取 `$?`（拿到的是 `tail` 的退出码，
构建失败仍报 0），一次是改动未提交、`git checkout` 把工作区改动带到了基线分支上。
两次都靠哈希核对才发现，否则会拿旧二进制跑出一份看起来正常的验收。

## 2. 正确性

两道闸门都要跑，它们抓的不是同一类问题：前者覆盖形状组合，后者覆盖 37741 keys 下
148 个 tile 的在线 softmax 累积效应。

```bash
python3 scripts/validate_ascend910b.py --device <卡号>
#  期望 status=passed, 141 个用例, 0 个超 1% 闸门

python3 scripts/profile_ascend910b.py --device <卡号> --warmup 2 --repeats 2 --trials 2
#  期望 accuracy_status=passed
#    decoded_reference  control 0.000068 / expcast 0.000043
#    dense_fp32         control 0.029799 / expcast 0.043645
```

**逐位重复性**——本交付改动了跨核同步时序，这一项务必跑，且用 **8 次而非 2 次**
（2 次正是漏掉同类竞态的原因）：

```bash
python3 - <<'PY'
import torch, torch_npu
from vc_attention_ascend.api import prepare, run_prepared
torch.npu.set_device(1)
g = torch.Generator().manual_seed(7)
for nq, nk, d in ((33,129,64), (65,257,128), (1024,2049,128)):
    for cast in (False, True):
        q,k,v = [torch.randn((1,2,n,d), generator=g).to(torch.float16) for n in (nq,nk,nk)]
        pl = prepare(*(x.to('npu:1') for x in (q,k,v)), layout='HND')
        first = run_prepared(pl, expcast=cast, tile_m=64).cpu().view(torch.int16)
        same = sum(int(torch.equal(run_prepared(pl, expcast=cast, tile_m=64).cpu().view(torch.int16), first))
                   for _ in range(8))
        print(f"q{nq} k{nk} d{d} cast={cast}: 8 次中逐位相同 {same}/8")
PY
```

期望全部 8/8。

## 3. 性能

用贵方自有的测量工具即可。若自行搭配对测量，三条建议都是我们实撞出来的：

1. **整包交换而非只换 `.so`**。本交付同时动了 Python 侧的 `K_TILE`，只换 `.so`
   会用错 payload 而静默跑出一份看起来正常的结果。
2. **臂序用 ABBA 而非 ABAB**。我们用同一个产物挂两个标签跑同一协议（真效应恒为 0），
   测出每轮内后跑的那一臂系统性快约 0.16%——即「N/N 配对同号」在零效应下同样成立，
   不构成效应证据。两臂产物不同时还会出现反相的周期 2 振荡，使效应估计随
   「丢弃前几轮」摆动约 1.9 个百分点。
3. **先用空载跑标定分辨率下限**。把同一个产物挂成两个臂跑一遍，得到的配对差分布
   就是本机本协议下「零效应」的样子；效应要超过它才算可分辨。

计时口径：形状 `[B=1, N=37741, H=28, D=128]` NHD、BF16、自注意力、`tile_m=64`；
payload 由 `prepare()` 一次构造并**排除在计时之外**，只测 `run_prepared()`；
warmup 2，每次 5 遍取中位。

期望：`control_m64` 约 −70%，`expcast_m64` 约 −71%（均相对基线）。
绝对毫秒值会随机器与温度漂移，配对百分比才是可比的。

## 4. 机理复核（可选）

```bash
msprof op --kernel-name=vc910_forward_m64 --launch-count=3 --output=<目录> python3 <驱动脚本>
```

★`msprof op` 默认采**第一个**算子；进程内若有其他算子先于内核启动，必须用
`--kernel-name` 过滤，否则采到的是无关算子（我们被这一条误导过一次）。

★由 pipe 占比推下界只能用 `max(单项 pipe)`，**不能用 `Σ`**——各 pipe 之间本身可以重叠，
用 `Σ` 会系统性低估可改进空间。我们曾据 `Σ` 推出「最多还能快 9%」，而实际改动拿到了 29%。
