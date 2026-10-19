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

CANN 9.0.1 的 legacy `extract_host_stub.py` 不接受 `__global__ __aicore__` 后再插入 `__vector__`，会在生成主机调用接口时抛出 `FuncNameNotFound`。融合量化入口采用标准 `__global__ __aicore__ void` 声明，并显式选择 `KERNEL_TYPE_AIV_ONLY`。若此前卡在该错误，`git pull` 后重新执行同一个脚本即可沿用构建缓存。此修复有同版本解析器规则的离线回归检查；完整 CANN 编译及 NPU 运行仍需远端验证。

FP16、BF16 入口分别编译，共用 `quantize_kernel_impl.h`。同版本 legacy 生成器对同一文件的入口符号按 `base_key + index` 编号，纯 AIV 的 KTYPE 元数据却都使用 `base_key`；两个入口放在一起会造成第二个符号与元数据名称不一致。这是在 `RegisterAscendBinary aiv ret 107000`、随后启动返回 `507000` 的排查中发现的确定代码生成问题；拆分绕过该问题，实际注册与设备执行由远端检查确认。

一键脚本在构建融合库后、长序列基线计时前，先单独运行 FP16/BF16 的小量化检查，验证库注册、两个入口启动、RNE 舍入和 padding。检查结果写入 `results/mindie_laser_quant_preprocess_smoke.json`，失败会立即停止，不必再等待整轮基线。已有 attention 构建缓存继续复用。

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

## 融合量化通过之后：固定输入配对诊断

用户回传的这轮长形状结果中，融合 Q/K 量化从 15.8247 ms 降至 1.3348 ms，完整 API 从 106.6155 ms 降至 93.0462 ms；原始 Laser 为 85.8781 ms。原生 INT8 attention 仍约 88–90 ms。此前两条量化路径在不同进程运行，不能从这两个 native 时间认定融合预处理拖慢了同一个 attention 内核。

已有成功构建时直接运行：

```bash
bash scripts/run_mindie_laser_quant_profile.sh
```

该入口复用 `results/mindie_laser_quant_build.json` 和 `results/mindie_laser_quant_preprocess_build.json` 指向的产物，不编译、不安装包。默认 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮。仍使用原始 exp、INT8 QK、FP16 PV，关闭 V 量化、重排和平滑。可沿用 `NPU_DEVICE` 指定卡。结果为 `results/mindie_laser_quant_profile.json`，日志同名 `.log`；原始 trace 在报告记录的独立运行目录中。

单个隔离进程加载同一个 `vc_laser_quant::la`，每轮随机且平衡两条调用的先后位置，记录原始顺序和耗时：

| 行名 | 比较范围 | 能说明什么 |
| --- | --- | --- |
| `shared_native` | 两个适配器使用**同一个 fused payload 对象和同一个 native operator 对象** | 两个名称并不是不同核心；这是同输入重复计时对照 |
| `own_native` | 相同 native 算子，各自固定的 torch/fused 量化结果 | 独立 payload 下的差异；也包含地址、缓存状态及舍入边界差异 |
| `prepared` | 各自固定 payload + 原生 API + 输出后处理 | 固定输入的调用开销 |
| `full` | 每次重新预处理 + 原生 API + 输出后处理 | 融合预处理对完整调用的收益 |
| `preprocessing` | 单独重新生成 Q/K/V 和 scale | 直接测预处理，不能用 full 减 prepared 代替 |

表中 `old/fused` 为两行中位数之比；`paired saved` 为同轮 torch 减 fused 的差值中位数，正值表示 fused 更快。快轮次数、差值范围和原始数据均写入 JSON。这是描述性配对数据，不是显著性检验。各 scope 独立测量，不能相加，也不能据此声称 QK/softmax/PV 的设备时间占比。两份固定 payload 在整轮中均保持存活；新结果与之前分进程结果的内存/缓存条件不同，应以本轮内部比较为准。

计时前，两条预处理路径都重新跑结构化小形状和实际 payload 检查，并检查同 payload 的**全部 padded 输出**完全一致。长形状 dense FP32 误差另用至多 32 个 query、所有 key/head/feature 计算，仅作算法诊断。源码、build、桥接、库映射、SoC/CANN 以及首次长形状 tiling marker 继续检查；marker 在计时前关闭。

正常 API 计时写盘之后，再用已安装的 `torch_npu.profiler` 分别采集两份固定 payload 的 native 调用及 fused 预处理。支持时请求 Level1/PipeUtilization，保留 `kernel_details.csv`、trace 和实际导出的计数器。**请求指标不等于成功采集**；不支持 profiler 或缺少可解析文件时会明确报告 trace 失败/仅有原始文件，保留已经完成的 wall-time 数据。设备调用或同步失败仍会使整个任务失败。

设备表按真实 task 名称/类型汇总；AIC/AIV 可能重叠，累计 device task 时间不是 API 延迟，也不是 QK/PV/softmax 各阶段归因。没有把 Python 调用事件冒充设备 kernel 时间。需要只测 wall time 时，可直接运行 `python3 scripts/profile_mindie_laser_quant.py`，省略 `--trace`。

已有报告可以直接离线打印计数器，无需重新编译、初始化 NPU 或重复测速：

```bash
python3 scripts/summarize_mindie_laser_quant_profile.py
```

也可用 `--input <报告路径>` 指定主报告或单独的 `capture_report.json`。摘要保留计数器原始列名、数值、样本数和范围，不猜单位或把各 pipe 相加成总耗时；缺失计数器或失败状态会明确显示。新采集完成后也会自动打印这份摘要。

早期采集器未显式设置 schedule，退出时曾出现 `Stop profiler while current state is RECORD`。这符合[官方调度状态机的未完成窗口警告](https://gitee.com/ascend/pytorch/blob/47f1f92999deda63f627141111cd6ffb736deb97/torch_npu/profiler/_profiler_action_controller.py)，不影响此前已完成的无 profiler API 测速；旧 trace 的完整性不能因此认定已验证。当前采集器使用 `wait=0,warmup=0,active=1,repeat=1,skip_first=0`，把整批 3 次调用放在一个 step 内，最后设备同步后显式 `step()` 完成保存，再退出上下文。没有给每个调用增加同步。报告记录 schedule 和 `capture_finalized`，旧报告摘要会标明缺少完成调度证据。

当前源码审计发现：每个 128×128 QK tile 都读取两次 scale，包含缓存失效与 GM 标量读取，并配置 FIX 随路反量化；目标 shape padding 后有 2,453,248 个逻辑 QK tile。这是重复次数推算，尚非瓶颈测量。用户回传同进程配对结果显示融合预处理由 16.8814 ms 降至 2.4468 ms，full 由 106.0615 ms 降至 90.9480 ms；相同 native API、相同 payload 的重复计时约 87 ms。应继续定位 attention 核心，而不是再把两个适配器名称当成不同内核。

## 独立 scale 复用候选与原始 FP16 计数器

```bash
bash scripts/run_mindie_laser_scale_reuse.sh
```

默认 BF16 BNSD `[1,28,37741,128]`，设备 0，warmup 5、每轮 10 次、9 轮。保持 **INT8 QK + 原版 exp + FP16 PV**，不加入 V 量化、重排、平滑或 ExpCast。入口先核对已有成功构建，再只构建 `VcLaserQuantReuseAttention`；源码、安装、bridge、vendor 都使用单独的 `mindie_laser_quant_reuse_*` 目录，不清空或重新构建旧基线。重跑保留候选构建缓存。

源码准备强制使用本地 Git 对象：一键入口从已校验的 INT8 build manifest 读取 `source_dir`，独立运行时也会查找已有成功构建的源码目录。无需重新下载 MindIE 或安装包。旧版网络超时留下的空 `.git` 会从本地锁定 commit 恢复；非空未提交目录、错误 HEAD 或其他 origin 不会被覆盖。没有本地锁定源码时立即说明缺失，不等待 GitHub 网络超时。安装好的二进制包供运行基线使用，修改候选内核仍使用此前构建留下的源码。

候选在每个工作描述符内读取两项 Q scale，每个 K128 分块读取一次 K scale，供两个 Q128 复用。完整 `ky=2,kx=32` 描述符的源代码级读取数由 128 降至 34。保留 `ReadQuantScale` 的 DCCI、FP32 乘法、低 13 位截断、`set_quant_pre`、FIX barrier、DEQF16 和原 Cube/Vector 流水。读取数不能直接换算加速比；用户回传的设备验证和同轮配对结果见下节。

固定输入 A/B 同时加载两个独立命名的原生算子，在相同进程中共享**同一个量化 payload 对象和 scale 地址**。表中 `baseline` 是当前 INT8 内核，`reuse` 是 scale 复用候选：

| scope | 范围 |
| --- | --- |
| `native` | 相同 payload，原生 attention 分配和调用，不含预处理及输出后处理 |
| `prepared` | 相同 payload，原生调用和输出后处理 |
| `full` | 两组使用相同融合量化器，每次从原始 QKV 重新预处理、调用和后处理 |

计时前跑 FP16/BF16 的短形状、尾块和 K4097 section 边界检查；新增长序列全 padded 输出逐值一致性检查。另在多 batch/head、Q257/K4097 输入上反复原地改写同一份 Q/K/V/scale 存储，并立即调用两版内核，交替顺序，验证不会读到上一轮 scale。输出不一致会在计时前失败，不放宽精度阈值来通过候选。

正常计时完成并落盘后，分别采集两个 INT8 native 调用的 3-call profiler 窗口；再在隔离进程中，用同 seed/shape 的原始输入和原 FP16 Laser 采集 native 计数器。FP16 行有自己的非量化 payload，不属于相同输入字节的 INT8 A/B，也不计算跨进程配对加速比。它用于对照 MAC、Scalar、MTE、FIX、Vector 等实际导出指标；pipe 时间可能重叠，**AIV Vector 时间不等于 exp 时间**，MAC 也包含 Laser 的对角矩阵计算，不能用计数器比例直接分摊 QK/softmax/PV 耗时。

报告为 `results/mindie_laser_scale_reuse.json`，日志同名 `.log`；每个 worker 的原始报告、CSV/trace 路径均保留。JSON `workers.paired` 包含配对耗时、全输出校验和 INT8 计数器，`workers.fp16` 包含原版计数器。新候选暂不替换默认主线；以同轮 `native`、`full` 的配对节省、快轮次数和正确性决定是否采用。已有候选构建时，可直接运行 `python3 scripts/profile_mindie_laser_scale_reuse.py --trace`，不触发编译。

## scale 复用实测与 gain 提前准备候选

用户回传 Ascend910_9382、BF16 BNSD `[1,28,37741,128]` 的结果：

| 同进程配对范围 | 旧 INT8 / ms | scale 复用 / ms | 旧/新 | 配对节省中位数 / ms | 更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 89.5797 | 87.5177 | 1.0236× | 2.0421 | 9/9 |
| prepared | 91.5683 | 89.4735 | 1.0234× | 2.0839 | 9/9 |
| full | 93.0581 | 89.4787 | 1.0400× | 3.5713 | 9/9 |

独立 profiler 窗口中，设备 task 时间由 89.0043 降至 86.6291 ms/call；AIC scalar 活跃时间明显降低，而 FIX 活跃时间仍约 43 ms。后者是可能重叠的流水活动指标，不能解释为有 43 ms 可以直接消除。原 FP16 Laser 的 native wall time 为 80.7590 ms，来自另一个进程，不能当成与这两版交错配对的结果。full 和 prepared 接近也不能说明预处理免费。

下一轮只验证一个调度变化：先算出每个 K128 tile 对应的两个 `sQ*sK`，按原规则清除低 13 位，把位模式传给 QK 输出函数。硬件 `set_quant_pre → PIPE_FIX barrier → DEQF16` 仍在原来的受保护输出位置。实际 scale 读取次数、量化输入、矩阵计算、原版 exp、FP16 PV 均保持不变；不加入 V 量化/平滑或 ExpCast。

```bash
bash scripts/run_mindie_laser_gain_precompute.sh
```

入口固定同样的 shape、设备 0、warmup 5、每轮 10 次、9 轮，沿用 `NPU_DEVICE` 和 `LASER_BUILD_JOBS` 可选环境变量。它核验已经成功的 scale-reuse、融合预处理和 FP16 构建，从 reuse build 的 `source_dir` 读取本地锁定 Git 对象，仅构建 `VcLaserQuantGainAttention`；所有新文件使用独立 `mindie_laser_quant_gain_*` 缓存，不访问 GitHub。旧 baseline/reuse 产物保持可复测。

同进程表中的 `scale reuse` 是上一版，`gain ready` 是这版。JSON 为复用配对框架保留 `baseline`/`reuse` 内部键，`experiment=gain-precompute` 和 `variant_labels` 明确实际对应关系。两者同享融合量化器，native/prepared 使用同一个 payload；计时前检查短形状、尾块、全 padded 输出逐值一致、原地更新 scale，以及相邻 Q/K 分块交替采用不同 scale 的情况。任一精度检查失败就停止，不放宽门槛继续计时。

结果：`results/mindie_laser_gain_precompute.json`，日志同名 `.log`；正常 API 配对之后才采集 profiler，FP16 仍是隔离进程对照。已有新构建时可只重测：

```bash
python3 scripts/profile_mindie_laser_scale_reuse.py --experiment gain-precompute --trace --output results/mindie_laser_gain_precompute.json
```

这次只提前准备普通标量计算，不宣称消除了 FIX 代价或证明了 FIX 是唯一瓶颈。编译器也可能已做类似调度，因此要看同轮 native/full 的实际收益。更激进的硬件配置前移缺少足够的寄存器生命周期依据，未在本候选实施；见[官方文档与同步审计材料](../../reports/ascend-gain-sync-audit/README.md)。本地 C++/Python 检查不替代远端 CANN 编译和设备测试。


## gain 结果与下一批 score 提前搬入实验

用户回传的 gain-precompute 同轮配对结果没有显示收益：native/prepared/full 的配对节省中位数分别为 −0.0221/−0.0501/−0.0545 ms，仅 2/9、3/9、3/9 轮更快。保留 scale-reuse 基线，不把 profiler 下约 0.1% 的变化解释为有效加速。

```bash
bash scripts/run_mindie_laser_score_prefetch.sh
```

这版只改 Vector 的分数搬入调度：原来每批四行先搬入、等待，再做 max/exp/sum；新候选利用原有两个独立 32 KiB score 缓冲区，提前发起下一批四行的 GM→UB 传输，尝试与当前计算重叠。它不是重复计算 gain 的标量微调，也不减少 attention 的数学运算量。源码上存在串行等待不等于已经测得它占主要耗时，收益必须由本轮配对结果决定。

候选仅用于无三角/稀疏 mask、每个 Vector section 64 行、每批 4 行、1–32 个 K128 block 的路径。目标 K37741 的 padding 仍走原有 tail/windowLen 处理；其他路径保留串行执行。mask 缓冲区存在重叠，所以没有提前搬入 mask。每次复用 score slot 前，同时等待该 slot 的 P 写回和 Vector 行求和读取结束。Cube、INT8 QK/scale、FIX DEQF16、原版 exp、FP16 PV、跨核交接顺序均保留；不增加 UB 容量，不加入 V 量化/平滑或 ExpCast。

入口核对已有 scale-reuse、融合量化和 FP16 构建，读取本地固定 commit 的 Git 对象，只编译独立的 `VcLaserQuantPrefetchAttention`。vendor、bridge、安装路径和源码使用独立 `mindie_laser_quant_prefetch_*` 缓存，不访问 GitHub、不清空或重编旧基线。固定 BF16 BNSD `[1,28,37741,128]`，设备 0，warmup 5、每轮 10 次、9 轮。可沿用 `NPU_DEVICE` 选卡。

计时前检查 FP16/BF16、K4096/4097/8193、Q 尾块、zero/constant/outlier、同一 payload 反复调用及原地更新，逐值比较全部 padded 原生 FP32 输出；长形状也要求全部原生输出精确一致。dense FP32 参考仍抽取最多 32 行，仅用于算法误差诊断。

表中 `scale reuse` 为已有基线，`score ahead` 为新候选；JSON 为复用框架保留 `baseline`/`reuse` 内部键，实际身份看 `experiment=score-prefetch` 与 `variant_labels`。native/prepared 使用同一个 payload；full 每次重新预处理。主要判断 native/full 的配对节省和快轮次数，各 scope 不相加。API 测速落盘后再采 profiler，FP16 是独立进程参照，不能当成相同 payload 的配对结果，也不从各 pipe 活跃时间推算 QK/softmax/PV 百分比。

报告 `results/mindie_laser_score_prefetch.json`，日志同名 `.log`。已有候选构建时可只重测：

```bash
python3 scripts/profile_mindie_laser_scale_reuse.py --experiment score-prefetch --trace --output results/mindie_laser_score_prefetch.json
```

[同步审核记录](../../reports/ascend-score-prefetch-sync-audit/README.md)保留源码差异、工具输出与缓冲区生命周期检查。本地测试不代替远端 CANN 编译、NPU 正确性和性能验证。候选暂不替换基线；在结果返回前不宣称提速。

## score 预取结果与独立阶段诊断

用户回传 score ahead 的 native 配对节省为 −0.0227 ms（0/9 更快），full 为 +0.0622 ms（9/9 更快），整体差距不足 0.1%。这不足以替换 scale-reuse 基线。下一步先采样跨核交接等待，以及 Vector 的细分执行区间。

```bash
git pull origin master
bash scripts/run_mindie_laser_phases.sh
```

默认仍是 BF16 BNSD `[1,28,37741,128]`、设备 0；warmup 5、每轮 10 次、9 轮配对校准，另采集 5 次记录。入口只要求已有 scale-reuse 与融合量化器成功构建，从本地固定 Git 对象准备和编译独立的 `VcLaserQuantProbeAttention`，不访问 GitHub，也不重编或替换现有基线。`NPU_DEVICE`、`LASER_BUILD_JOBS` 可选；普通使用直接运行上面的 bash 命令。

该版本保持 INT8 QK、原版 exp、FP16 PV，不增加 V 量化、重排、平滑或 ExpCast。同一个进程、相同 payload 对象比较以下调用：

| 名称 | 行为 |
| --- | --- |
| `reuse` | 已有 scale-reuse 原生 API |
| `plain` / mode 0 | 编译了诊断代码，但不读计数器、不写记录 |
| `waits` / mode 1 | 在原有 Cube 等 Vector、Vector 等 Cube 的 `wait_flag_dev` 前后读取计数器 |
| `phases` / mode 2 | 额外记录分数读取、最大值与 online history、减最大值、主体 exp、求和的完成区间 |

诊断 ABI 保留原 statistics 和 attention 输出，另分配并清零 INT64 `[4,3,8,24]` 记录区。mode 0 仍有诊断类状态、分支和 trace 分配/清零，因此不能叫“与生产完全相同的内核”。`reuse → plain` 校准整个诊断框架的开销，`plain → waits/phases` 观察新增计时的扰动。表中全部是同步 native API wall time，包含分配和调用，不含预处理、后处理或 profiler；额外 trace 清零也在范围内。校准不是加速结果。

只采 core 0–3，各自 Cube、AIV0、AIV1 的稀疏样本。等待取逻辑 round 0、1、中间、末尾，重复位置只保留首次；Cube round 标识等待的 P tile，而不是提前计算的 QK tile。Vector 各取 head/steady/tail/head+tail 首个实际四行批次，记录 round、section、key block 数。长形状要求实际存在等待和 Vector 阶段记录；短形状允许未用核、子核和 section 类别为空。不是覆盖所有任务的总体采样。

Vector 完成边界使用相邻 `V→S EVENT_ID0` set/wait，原有 V_S1–6 和跨核信号保持原样。`score_load` 包含标量地址准备、可能的 slot 依赖、分数 DMA 和完成等待；`max_history` 包含 scale、mask、max reduction、历史小向量 exp 和 broadcast；`subtract`、`exp`、`sum` 分开记录。sum 区间从 P 写回发起后开始，写回可能与归约重叠。阶段记录还保留空完成边界的开销，但不会自动相减或宣称得到纯指令延迟。

先验证 FP16/BF16、D64/D128、短序列、尾块及 K4097/8193，并对长形状检查全部 padded 输出与 scale-reuse 精确一致；任一模式不一致则停止。无 profiler 的校准结束后再单独读取 trace；所有数值保留为原始 **system ticks**，不是 AIC/AIV 核心周期。探针增加的同步会改变重叠，等待时间也不等于另一侧的总计算时间，不能将这些片段相加或外推成生产内核百分比。

结果在 `results/mindie_laser_phases.json`，终端日志为同名 `.log`。`worker.calibration` 是配对扰动，`worker.traces` 是带位置的原始记录，`worker.phase_summary` 是按角色、等待位置或 section 类别汇总的样本。来源、库映射和构建哈希继续校验；计时前关闭 tiling marker。已有诊断构建时可直接运行 `python3 scripts/profile_mindie_laser_phases.py`。

[同步审计](../../reports/ascend-phase-probe-sync-audit/README.md)保留工具原始输出和人工边界检查。本地没有 CANN/NPU，普通 C++ 语法桩、Python 和 Bash 检查不能代替远端设备编译及执行；此轮不宣称获得性能收益。

## Cube 的 QK/output 与 PV/history 分段诊断

上一轮用户回传：scale-reuse / plain / waits / phases 的 native API 中位数分别为 87.2881 / 87.2181 / 87.2089 / 87.1877 ms；配对扰动很小，负值不能解释为探针提速。稀疏采样中，Cube 中间轮的跨核等待接近 0，AIV 中间轮仍有约 350–393 system ticks 等待。Vector 四行、32 个 K block 样本中，主体 exp 约 16 ticks，load 约 31–37 ticks。它们来自不同位置和范围，不能相加、换算占比，也不能单凭这些值断言 Cube 的某条指令是瓶颈。

当前 Cube 调度会先做前两轮的 PV，再做当前轮 QK，且有异步 FIX 输出和跨核交接。因此 AIV 等待可能同时受 PV、QK、输出搬运和调度影响。下一步先区分完整的 **QK 连同 score 写出/反量化** 与 **PV 连同历史输出更新/尾段归一化**，并让原版 FP16 Laser 使用同样的采样方法。

```bash
git pull origin master
bash scripts/run_mindie_laser_cube_phases.sh
```

固定 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮校准、5 次采样。可沿用 `NPU_DEVICE`、`LASER_BUILD_JOBS`。脚本先核验已有 FP16、scale-reuse、融合量化器的来源、哈希和设备/CANN 匹配，再从本地 pinned Git 对象准备两套独立探针：`VcLaserCubeInt8ProbeAttention` 和 `VcLaserCubeFp16ProbeAttention`。没有 GitHub 下载；首次需编译这两个新算子，已有生产算子不重编或替换。FP16 探针直接派生原版源码，不通过撤销量化补丁生成。

每种算法分别使用自己的生产原生 API 作为 baseline，在同进程、同一个 prepared payload 上随机交错校准三个模式：

| 模式 | 测量范围 |
| --- | --- |
| `plain` / 0 | 不读时间、不写记录，保留诊断代码、分支与 trace 分配/清零 |
| `issue` / 1 | 完整 cube1/cube2 调用及地址映射的标量发出窗口，不添加完成等待 |
| `completed` / 2 | 采样前排空已有工作，随后记录发出窗口与末尾完成等待，并单独记录空完成边界的开销 |

模式 2 仅在完整函数边界调用 `PipeBarrier<PIPE_ALL>()`，不在 MAD 与配对 FIX copy 之间插入屏障，也不拆开历史缩放 MAD 与后续 PV 累加。保留原运算、buffer 映射、unitFlag 与跨核信号；两个版本的 Vector 代码各自与其基线逐字节一致。新增屏障会改变流水重叠，校准必须结合记录一起看。模式 1 的 `issue` 也不是纯标量计算量，其中可能有原函数内部等待。

trace 是独立清零的 INT64 `[4,2,3,24]`，仅采 Cube core 0–3 的首轮、中间轮和末轮，重复位置只保留首次。记录保留 descriptor 数、K block 数、逻辑 tile 数、历史更新和尾段归一化调用数；这些不是底层 MAD 指令数。不同工作量分别汇总。`issue/tail/elapsed/predrain/empty` 都是原始 system ticks，`tail` 是尚未完成的队列工作等待，**不能叫纯 FIX 时间**；`empty` 也不会自动相减。模式 1 未测的字段显示 `--`。

每种模式先检查 FP16/BF16 输入、短序列、D64/D128、K4097/8193、尾块，再检查长形状的全部 padded 原生输出与其对应 baseline 精确一致。任何不一致或损坏/缺失记录会停止，保留失败报告。长形状 dense FP32 误差另抽取 32 行、所有 keys，仅描述各算法误差；不是模型质量验证。

INT8 和 FP16 使用同一组原始输入，但各自预处理、在不同进程中运行。两者的 ticks 需按相同工作量审视，不输出跨算法配对提速比。API 校准包含分配、调用和 trace 清零，不含预处理、后处理或 profiler；采样值不代表生产阶段百分比，也不相加外推内核毫秒数。

结果为 `results/mindie_laser_cube_phases.json`，终端日志同名 `.log`。查看 `workers.int8` / `workers.fp16` 下的 `calibration`、`traces`、`phase_summary`；先看探针扰动，再对比 QK/PV 同类样本。若第二个 worker 失败，第一个的结果仍保留。已有探针构建后可直接重测：

```bash
python3 scripts/profile_mindie_laser_cube_phases.py
```

[本轮同步审计](../../reports/ascend-cube-phase-sync-audit/README.md)记录官方计时/屏障说明、完整本地头文件树审计、人工 unitFlag 生命周期核对和原始工具输出。CANNBot 的正则审计不理解全部 Laser 原始 intrinsic，同步计数为 0 不等于内核没有同步；本地 C++ 桩检查也不替代远端 CANN 编译和 NPU 执行。本轮交付用于定位下一项优化，尚无设备侧性能收益结论。

## Cube 诊断结果与 scale 缓存行复用候选

用户回传的同工作量中间轮样本：INT8 / FP16 的 QK issue 为 1152 / 954 ticks，QK completed 为 1244 / 983.5 ticks，PV completed 为 1335 / 1344.5 ticks。INT8 QK 的完整诊断区间约高 26.5%，PV 接近；因此先沿 QK 新增的 scale 读取和反量化输出链排查。完整 API 的 baseline→completed 配对扰动分别为 +0.0345 / +0.0578 ms，较小，但局部 drain 仍会改变重叠。这些结果不能证明差额全属于 DCCI、FIX 或 INT8 矩阵乘，也不能累加外推成生产耗时占比。

```bash
git pull origin master
bash scripts/run_mindie_laser_scale_cache.sh
```

本候选基于已测的 scale-reuse，只改变 Q/K scale 的缓存维护方式。旧 helper 每读取一个 FP32 scale 都执行 DCCI；新 helper 分别记住 Q、K 最近访问的**绝对 64B 缓存行地址**，同一行的后续读取跳过重复刷新，切换缓存行时仍刷新。返回先前离开的行也会重新刷新，故不维护全序列缓存表。每次 kernel 的 `SetQuantScales` 重置两个地址记录，防止新调用复用内存地址却读到旧 scale。scale 本身仍按原地址逐值读取。

32 个连续 K scale 共 128B，覆盖 2 个对齐缓存行或 3 个非对齐缓存行；这个局部循环的 DCCI 次数可从 32 次减少到访问行数。实际减少的指令数不等于等比例速度提升，新增判断也有开销。地址判定不依赖逻辑索引模 16，所以 descriptor/head 从行中间开始和非零 storage offset 都应正确。

量化 payload、Q/K scale 值、gain FP32 乘法与低位截断、`set_quant_pre → PIPE_FIX → DEQF16`、矩阵运算及跨核信号顺序都保留。Vector、原版 exp 和 FP16 PV 不变，没有增加 V 量化、平滑、重排或 ExpCast。此版是普通候选内核，不带阶段计时探针或额外诊断输出。

入口仍固定 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮，沿用 `NPU_DEVICE` 选卡。脚本核验已有 scale-reuse、融合量化器和 FP16 基线，只使用本地 pinned Git 对象构建独立 `VcLaserQuantCacheAttention`；其源码、vendor、bridge、安装目录和 manifest 都使用 `mindie_laser_quant_cache_*` 路径。不会访问 GitHub，也不重编旧基线。

计时前除原有结构化、尾块、原地更新检查外，增加非零 scale storage offset、跨缓存行、多个 batch/head、K4097 分段、相邻 scale 不同值及同一指针连续修改的测试。逐值比较全部 padded 原生 FP32 输出，候选必须和 scale-reuse 精确一致；失败即停止并保存报告。长形状也检查全部输出，dense FP32 误差另抽最多 32 行，不能作为模型质量结论。

主表 `scale reuse` 为旧基线，`scale cache` 为本候选，仍是同进程随机交错计时：`native` 和 `prepared` 使用同一个 payload，`full` 每次重新预处理。优先看 native/full 的配对节省中位数和更快轮数；各 scope 不相加。普通 API 计时保存后再采 profiler；原版 FP16 在独立进程中作参照，不输出跨进程配对加速比。

结果与日志分别为 `results/mindie_laser_scale_cache.json` 和同名 `.log`。已有候选构建后可只重测：

```bash
python3 scripts/profile_mindie_laser_scale_reuse.py --experiment scale-cache --trace --output results/mindie_laser_scale_cache.json
```

[缓存一致性及同步审计](../../reports/ascend-scale-cache-sync-audit/README.md)保留官方 API 依据、源码差异、工具原始输出和人工检查。本地没有 CANN/NPU；CPU 测试、语法桩和离线源码校验不能替代远端编译、正确性与性能验证。此候选尚无实测收益，不替换已测基线。
