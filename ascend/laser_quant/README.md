# 在 MindIE Laser 中执行 INT8 QK

当前主线为原生 INT8 QK + 原 Laser softmax + FP16 V/PV。QK 使用 INT8 Cube、INT32 累加，随后回到原 Laser 的 FP16 score、Vector softmax 和 FP16 PV 路径，V 直接由原输入转为 FP16。E4M3 V 解码和 ExpCast 保留为显式启用的实验。该实现从固定版本 MindIE-SD Laser 内核派生，没有调用旧的 `vc_ascend` attention 内核；上游固定为 [MindIE-SD `71098c20`](https://github.com/Ascend/MindIE-SD/tree/71098c20ff90838d1fd1ca0a40515c751a3869b0)。

此前目标长形状的 INT8 QK 版本已跑通，但预处理约 16.94 ms、完整调用约 107.18 ms，慢于 Laser 源码基线约 85.98 ms。这一轮增加融合 Q/K 量化预处理，保持同一个 INT8 attention 内核作对照，并分别测量原生调用和输出处理。融合候选仅完成本地代码和 CPU 检查，仍需远端 CANN 编译及 NPU 正确性、性能验证；不能提前认定获得提速。

## 一条命令

在已配置 CANN 9.0.1、匹配的 PyTorch/torch_npu 和 MindIE-SD Laser 的 A2/A3 Linux 主机上，从仓库目录运行：

```bash
bash scripts/run_mindie_laser_quant.sh
```

默认输入为 BF16、BNSD `[1,28,37741,128]`。脚本先预检设备，再复用通过来源及文件哈希检查的原始 Laser 构建；缺少有效原始构建时才准备源码、构建算子和桥接库。接着增量构建独立量化算子和融合预处理库，默认分别启动四个 Python 进程比较：

| 变体 | QK | V 输入和 PV | 概率计算 |
| --- | --- | --- | --- |
| `installed` | 已安装 Laser FP16 | FP16 | 原始 exp |
| `baseline` | 固定源码 Laser FP16 | FP16 | 原始 exp |
| `qk_int8` | INT8 → INT32 | 原始 V 转 FP16 | 原始 exp |
| `qk_int8_fused` | 同一 INT8 → INT32 内核，Q/K 由融合 Ascend C 算子量化 | 原始 V 转 FP16 | 原始 exp |

`qk_int8` 保留多次 PyTorch NPU 调用的量化预处理作为对照；`qk_int8_fused` 融合 Q/K 的转换、padding、block absmax、scale 和 INT8 舍入输出。两组都保留原 softmax 和 FP16 PV，不执行单位 V scale 乘法。构建融合库使用预检报告中的完整运行时 SoC（如 `Ascend910_9382`），Laser attention 包继续使用原有构建族（如 `ascend910_93`），不会因此清空或重建原 attention 缓存。

以下两组仅在显式启用实验时追加，组成六组对照：

| 实验变体 | QK | V 输入和 PV | 概率计算 |
| --- | --- | --- | --- |
| `qkv_int8_e4m3` | INT8 → INT32 | 按通道量化 E4M3、解码为 FP16 | 原始 exp |
| `qkv_int8_e4m3_expcast` | INT8 → INT32 | 按通道量化 E4M3、解码为 FP16 | ExpCast |

```bash
LASER_QUANT_EXPERIMENTS=1 bash scripts/run_mindie_laser_quant.sh
```

`LASER_QUANT_EXPERIMENTS` 默认为 `0`，仅接受 `0` 或 `1`。直接运行 `benchmark_mindie_laser_quant.py` 也默认使用四组主线；其 `--variants` 接受全部六个变体。

每个进程先执行小形状、尾块、零输入和 scale 变化检查；通过后才运行长形状和计时。任一构建或正确性检查失败会保存 JSON/log 并退出，不继续输出通过的速度结论。

执行环境和构建并行度也可通过环境变量指定：

```bash
PYTHON_BIN=python3 NPU_DEVICE=0 LASER_BUILD_JOBS=4 \
  bash scripts/run_mindie_laser_quant.sh
```

主要结果为 `results/mindie_laser_quant_compare.json`、同名 `.log`、各进程的原始报告和采样输出；attention 构建清单为 `results/mindie_laser_quant_build.json`，融合预处理构建清单为 `results/mindie_laser_quant_preprocess_build.json`。任一构建失败都会退出，不回退到多算子路径冒充融合结果。脚本不安装或升级 CANN、PyTorch、torch_npu、MindIE-SD。

## 数值和原生接口

- Q/K 均按每个 batch、head、128 token block 的绝对最大值量化，使用对称 INT8 `[-127,127]` 和 FP32 scale。全零块的 scale 为 1。**Q128/K128 与 H200 路径的 Q32 量化粒度不同**，不能作逐位等价比较。
- 输入按 S256/D128 补零，D64 输入仍使用原始 `1/sqrt(64)` attention scale。有效 key 数传给内核，padding 不参与概率和 PV。
- `sQ*sK` 先按 FP32 计算；DAV-2201 FIX 的 DEQF16 scale 编码仅保留 FP32 bits 31:13，低 13 bits 截断。INT32 QK 乘该编码 scale 转 FP16 后，继续执行原 Laser 的 FP16 attention scale。CPU decoded oracle 显式包含这些舍入步骤。
- `qk_int8` 和 `qk_int8_fused` 均直接采用原始 V 的 FP16 转换，不执行 V token 重排、聚类、V 平滑或 K 均值平滑，也不乘值为 1 的 V scale。
- E4M3 V 实验按 batch、head、channel 求 scale，数值量化后立即解码为 FP16，原生 PV 使用 FP16 Cube。原生输出后再乘 V scale；该步骤包含在 full 和 prepared 两种调用计时中。**这里没有原生 FP8 matmul，也没有 FP8 V 存储节省或带宽加速声明。**
- ExpCast 实验保留 Laser 实验的两次 FP32 `Muls`/`Adds` 舍入、gain `11.541560173034668`、bias `119.6500015258789`、RNE code `[0,120]` 和 `decode(E4M3)/256`。输入 score 为 FP16；在线历史 rescale 仍使用原 exp。普通 exp 和 ExpCast 的分母均读取供 PV 使用的 FP16 概率，再以 FP32 求和。

FIX scale 位布局依据 [CANN 随路量化说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_key_features/on_the_fly_quantization.md)，设置后保留 [官方示例中的 PIPE_FIX barrier](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0254.html)。具体 CCE intrinsic 的编译兼容性仍由远端 CANN 9.0.1 验证。

桥接接口独立注册为：

```text
torch.ops.vc_laser_quant.la(
    q_int8, k_int8, v_half, q_scale, k_scale,
    scale: float, keys: int, expcast: bool
) -> FP32 padded BNSD tensor
```

Q/K scale 布局为连续 `[B,H,S/128]`，V 原生布局为连续 BNSD。唯一 ACLNN API 为 `aclnnVcLaserQuantAttention`，算子为 `VcLaserQuantAttention`。Python 适配器只接受 NPU；CPU 函数仅用于显式数值参考，没有自动 fallback。

融合预处理通过 `torch.ops.vc_laser_quant_preprocess.quantize(x)` 直调 Vector 内核，Q/K 各调用一次，返回 INT8 tensor 和 FP32 scale。每个 128×128 块在约 177 KiB UB 内完成计算；先将 FP32 归一化值按 RNE 转 INT32，再精确转为 INT8，避免先转 FP16 导致二次舍入。scale 按相邻 8 项分组写出，避免不同核心写同一 32-byte 区域。实现参考 [CANNBot 直调模板](https://github.com/hicann/cannbot-skills/blob/dbf4019cae5d757820f03c9cd21e0f0b277e1701/ops/ascendc-direct-invoke-template/SKILL.md)，具体来源随扩展记录。

## 准确性和计时口径

小形状包含 FP16/BF16、D64/D128、Q/K 不等长、S256 padding、K4097 分段边界及全零输入。结构化 Q/K 在 feature 31/32/63/64/95/96/127 附近放置非对称非零数据，不同 Q/K block 使用不同 scale，V 各通道幅值也不同。每个 128-key block 保留零 score 最大值，使在线历史因子为 1，便于独立检查 INT8 QK、FIX scale、概率和 V dequant。常量 V 通道另作检查，避免其掩盖非恒定输出错误。

CPU/NPU 预处理分开校验：scale 必须为正、有限，且与独立 CPU 结果相差至多 1 个 FP32 ULP；全零块/通道的 scale 必须精确为 1。INT8 使用实际 NPU scale 在 CPU 重算，只允许归一化值及其相邻一个 FP32 数所能产生的 RNE 结果，整数差至多为 1；偏离舍入边界、错误邻居、错误 scale、padding 非零均失败。二次幂 scale 在量化边界处要求精确舍入，另在设备执行 scale=1 的正负半整数 RNE 探针。V 解码值仍精确比较，不套用 INT8 例外。这是有界的跨设备检查策略，不是把所有 INT8 误差放宽到 1。

例如 K4097 的 FP16 用例中，CPU 比值 `-63.499996185302734` 与 `-63.5` 只差 1 ULP，分别可能量化为 `-63` 和 `-64`。报告在 `small_accuracy[].preprocessing_cpu_comparison` 中保存差异数、坐标、实际/CPU scale、归一化值及允许的整数范围，失败前也写盘。attention decoded reference 始终读取实际 NPU payload；以上预处理检查全部在计时外执行。

融合路径另检查 B2/H3、D64 非连续输入及连续修改复用同一输入存储的场景，覆盖跨 head 的 scale 分组、尾块和零输入。报告保存为 `fused_preprocessing_reuse`。新旧量化的除法可能在舍入边界生成不同 INT8 值，因此分别校验原始输入、实际 payload 和 decoded 输出，不强制将两条路径的 payload 哈希当作逐位一致门限。

decoded 实现检查的相对 RMSE 门限为 1%，full 与固定 prepared 输出的一致性门限为 0.1%。这些是实现检查；与原始 FP32 dense softmax 的误差单独报告，不把近似算法误差包装成模型质量通过。长形状抽取至多 32 个 query、覆盖所有 key/head/feature；长形状 decoded 参考使用全局最大值，没有声称逐步模拟原生在线更新和归约舍入，也不施加同算法等价门限。

报告直接测量三种同步 wall time：

| 字段 | 范围 |
| --- | --- |
| `preprocessing` | 每次重新转换、padding、Q/K 量化及可选 V 量化/解码 |
| `full` | 每次重新预处理 + 原生分配/调用 + 输出后处理；V 输出 scale 仅用于 V 量化实验 |
| `prepared_core` | 固定预处理结果 + 原生分配/调用 + 相同输出后处理 |

`prepared_core` 不是 kernel-only 时间；预处理时间也不是用 `full - prepared` 推算。输入有限性在计时前检查一次。各变体按独立进程分组，默认 warmup 5、每 trial 10 次、3 个 trial，报告中位数；它不是变体交错的配对试验。

量化变体另报告 Q/K 量化、原生 attention API 和输出后处理的独立同步 wall time，帮助区分 16.94 ms 预处理和 prepared 开销中的大项。原生 API 包含分配及 dispatch，Q/K 量化仍包含输出分配；各项在独立调用中采样，缓存、同步和对象存活期可能不同。**这些时间不可相加恢复 full/prepared，也不能换算成设备内核各阶段占比。** 只有远端正确性检查通过后，才能据此评价融合预处理是否有效。

每个量化进程在计时外保存实际 Q/K/V 和三组 scale 的 SHA256。启用对应实验变体时，父进程要求 QK-only 与 E4M3-V 变体的 Q/K 及其 scale 完全一致，要求 E4M3-V 普通 exp 与 ExpCast 的全部预处理输入完全一致，支持消融使用相同输入的结论。

来源检查包含固定 Git commit、prepared source 重验、source/build/bridge/device binary 哈希、当前进程中加载的 provider 路径，以及首次长形状成功 tiling 的新 marker。marker 在计时前关闭。这些是源码构建和 host dispatch 证据，不能独立证明设备实际执行二进制的身份。

## 开发时单独构建和小形状检查

一键流程已经先执行小形状检查。已有原始 baseline 和 bridge 时，也可以单独迭代量化构建：

```bash
python3 scripts/build_mindie_laser_quant.py \
  --cann-root "$ASCEND_HOME_PATH" --soc ascend910_93 --jobs 4 \
  --output results/mindie_laser_quant_build.json

python3 scripts/build_mindie_laser_quant_preprocess.py \
  --cann-root "$ASCEND_HOME_PATH" --soc Ascend910_9382 --jobs 4 \
  --output results/mindie_laser_quant_preprocess_build.json

python3 scripts/benchmark_mindie_laser_quant.py \
  --preprocess-build results/mindie_laser_quant_preprocess_build.json \
  --variants qk_int8 qk_int8_fused \
  --shape 1 2 257 128 --warmup 1 --repeats 1 --trials 3 \
  --output results/mindie_laser_quant_smoke.json
```

SoC 参数应与预检报告一致：Laser attention 构建在 A2 使用 `ascend910b` 构建族；融合预处理编译必须传完整运行时型号（如 A3 的 `Ascend910_9382` 或 A2 的实际 `Ascend910B*` 型号）。不能把 A3 设备伪装成 A2。实验开发时可在 `--variants` 后追加 `qkv_int8_e4m3 qkv_int8_e4m3_expcast`。小形状完成后，应再运行默认长形状比较，才能评价目标 workload。

CPU 检查可以独立执行；它们不代替 CANN 编译和 NPU 验证：

```bash
python3 -m pytest -q tests/test_mindie_laser_quant_backend.py \
  tests/test_mindie_laser_quant_fused.py \
  tests/test_mindie_laser_quant_preprocess_build.py \
  tests/test_mindie_laser_quant_preprocessing.py \
  tests/test_mindie_laser_quant_benchmark.py \
  tests/test_mindie_laser_quant_orchestration.py
```
