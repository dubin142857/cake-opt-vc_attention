# vc16：沿用 Sage 主体调度的独立候选

目标是在用户的 H200、BSHD `[1,37741,28,128]`、BF16、noncausal 场景继续降低耗时。
用户上一轮同场核心 API：原版 Sage `24.5821 ms`，VC control `25.5126 ms`，
vc15 ExpCast `25.4846 ms`。vc15 与 control 已接近，但仍比原版慢约 3.7%。
单次 trace 的原版为 `22.2435 ms`，这是 profiler 下的 GPU kernel 时间，不能当成
完整 API 或多轮核心 API 的结果。**22 ms 是目标；vc16 尚无 H200 结果。**

## 本轮改动

版本 `sageattention-local==2.2.0+vc16`，native kernel version 13。
从固定版本的原版 Sage kernel 主体派生独立 `sage_layout.cuh`，保留 QK/PV 的
TMA、WGMMA、两级 FP32 累加结构，将 vc15 ExpCast 接到该主体上。
候选保持相同的 INT8 QK、FP8 V、P 编码及解码后 FP32 分母；不增加共享概率副本。
尾块使用 VC 的负无穷遮罩，寄存器上限为 168。

| `kernel_mode` | 普通 Softmax control | ExpCast-only |
|---|---|---|
| `vc15` / `optimized` | 保留默认内核 | 保留默认内核 |
| `vc16` | 原版先求和、后打包，V 在输出累加后预取 | vc15 编码，历史更新与 PV 重叠，V 较晚预取 |
| `vc16_early_v` | 仅将 V 预取移到输出累加前 | 仅将 V 预取移到输出累加前 |
| `vc16_pack_first` | 先打包、后求和，V 较晚预取 | 不支持 |
| `vc16_pack_early` | 先打包、后求和，V 较早预取 | 不支持 |
| `vc16_serial` | 与 `vc16` 相同 | 历史更新在 PV 发射前完成 |

新候选仅支持 `v_smooth=False`。**默认保持 vc15**，收到 H200 同场结果后才决定
是否提升某个候选为默认。原版 `sageattn` 内核未修改。
普通 Softmax 的两个调度开关形成四个组合，可辨别打包顺序和 V 预取各自的影响。
ExpCast 则比较历史更新串行/重叠、V 预取早/晚；减少指令或改变顺序并不保证提速。

## 同场测量

在远端仓库根目录执行，全部使用 `python3`：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --expcast-only --compare-sage-layout --warmup 3 --repeats 10 --trials 9 --kernel-samples 20 --kernel-trials 3 --trace-dir results/h200_vc16_traces --output results/h200_vc16_layout.json
```

默认 shape 就是上述目标。构建脚本先验证原版与 VC，再测性能。
`--compare-sage-layout` 新增十行：原版 Sage、一组 control、一组 ExpCast。

- `core API`：所有候选与原版共用同一份量化输入，包含每次输出分配及原生下发，排除预处理。
- `full API`：每次从原始 Q/K/V 重新进行全部预处理。原版行调用公开 `sageattn`。
- `vc15/core`：对应算法的 vc15 核心耗时 / 候选核心耗时，大于 1 表示候选更快。
- `CUPTI kernel durations`：随机顺序的三轮，每轮每个候选采集 20 次 kernel；
  先取每轮中位数，再取轮间中位数。它是 profiler 开启时的 GPU 时间，单独列出。

完整 API 和固定核心均采用随机顺序、多轮同步 wall time。新候选分别与同算法 vc15
比较**全部输出行**，相对 RMSE 门槛 0.1%；完整 API 输出也与固定输入调用核对。
不将 ExpCast 与普通 Softmax 的算法差异套入实现回归门槛。
JSON 保留原始各轮数据、全部 kernel 时长、调用顺序、编译资源与 trace 路径。
不需要 ncu。若 CUPTI 不可用，去掉 `--trace-dir`；正常计时先写入 JSON。

先看新 control 是否接近原版，再看新 ExpCast 是否快于同场原版；最后确认完整 API
也有收益。只有 GPU kernel 接近 22 ms，不能算完整 API 达到 22 ms。

可显式试用候选：

```python
from sageattention_local import sageattn_vc
output = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True,
                     v_smooth=False, kernel_mode="vc16")
```

## 本地验证及限制

- 346 项测试通过，另有 16 组历史 CUDA helper golden 数据精确一致。
- CUDA 12.4 编译并链接全部 116 个 `sm_90a` 内核：28 个新候选，88 个历史内核。
- 新候选 D128 每线程 168 寄存器，D64 为 128；ptxas 栈/溢出均为零，无 ptxas 警告。
- 88 个历史内核的机器指令文本与 vc15 一致；新候选确实生成了不同的指令主体。
- Hopper 验证脚本增加 168 组同输入候选对照，覆盖 FP16/BF16、D64/D128、
  多 batch/head，以及 1、127、128、129、257、385 个 key 的尾块；待远端执行。
- 打包验证包含新增头文件及来源说明，原版 36 份 CUDA/Triton kernel 源码不变。

本地运行设备为 RTX 3060 SM86，不能执行 Hopper WGMMA attention。
本地测试覆盖实际概率/聚类/量化 helper、Python 接口与计时编排；**不能替代新完整内核的
H200 数值验证**。CUDA 13 的编译资源也可能不同；不作新候选速度或模型质量通过的声明。
证据：[编译资源](results/sage_vc16_codegen_local.json)、[本地验证](results/sage_vc16_local_validation.json)。
