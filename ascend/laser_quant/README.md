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

[缓存一致性及同步审计](../../reports/ascend-scale-cache-sync-audit/README.md)保留官方 API 依据、源码差异、工具原始输出和人工检查。本地没有 CANN/NPU；CPU 测试、语法桩和离线源码校验不能替代远端编译、正确性与性能验证。后续远端回测结果见下一节，保留独立比较路径。

## scale-cache 回测与 QK 的 K 块预取

2026-10-20 用户回传，scale-reuse → scale-cache 的 native 中位数为 87.7820 → 87.6747 ms，配对节省 +0.0999 ms（9/9 更快）；full 为 90.2839 → 90.2167 ms，配对节省 +0.0635 ms（8/9 更快）。这是约 0.1% 的小收益。profiler 中 `aic_scalar_time(us)` 中位数从 22227.215 降至 16424.347，但设备任务均值仅从 87.7200 降至 87.5802 ms。标量活跃计数下降没有转化成同等总耗时下降；各流水可重叠，不能把差额解释成可直接节省的独占时间。FP16 native 81.2073 ms 来自独立进程，仅作参考。

接下来测试 QK 主循环的数据搬运重叠：

```bash
git pull origin master
bash scripts/run_mindie_laser_k_prefetch.sh
```

旧版每个 K128 块在 GM→L1 搬运后立即等待，再执行 L1→L0 和 QK；新候选先搬入首块，在消费当前块时向另一个 slot 提前搬入下一块。使用原有 128 KiB / 192 KiB 位置的 L1 B 缓冲区、原有 ready/release 信号，不增加 L1/L0 容量。一个 INT8 K128×D128 块占 16 KiB。预取不跨 descriptor 或 QK/PV 调用边界，不读取末块之后的 K；descriptor 结束时保持原来的 slot 奇偶状态和 release token，供 PV 接续使用。

候选在已经验证的 scale-cache 上仅改变 QK 的 K 搬运发出时序；Q/K scale、INT8 payload、MAD、DEQF16、FIX 屏障、Vector softmax、FP16 PV 和跨核交接保持原样。下一块 DMA 与当前计算存在重叠机会，但实际收益可能受 FIX、MTE 竞争或其他流水限制，不能从源码直接推算速度。

脚本只编译独立 `VcLaserQuantKpipeAttention`，使用 `mindie_laser_quant_kpipe_*` 源码、vendor、bridge、安装和构建记录；复用已有 scale-cache、融合量化器和 FP16 基线，从本地 pinned Git objects 准备源码，不访问 GitHub。默认 BF16 BNSD `[1,28,37741,128]`，设备 0，warmup 5、每轮 10 次、9 轮；沿用 `NPU_DEVICE` 选卡。

计时前检查 structured/decoded 参考、scale 非零偏移和同指针重写，并增加跨 K128/4096 边界、多个 batch/head、尾块及反复原地更新的 K 预取检查。native 输出包括 padding，要求与 scale-cache 全部元素精确一致。长形状也先做全输出一致性检查；任何失败立即停止并保存证据。主表 `scale cache` / `K ahead` 为同进程随机交错、相同 prepared tensor 的 native/prepared 对比，full 每次重新预处理；profiler 在普通计时之后单独运行，FP16 仍为独立进程参照。

结果为 `results/mindie_laser_k_prefetch.json`，日志同名 `.log`；已有候选构建后可只重测：

```bash
python3 scripts/profile_mindie_laser_scale_reuse.py --experiment k-prefetch --trace --output results/mindie_laser_k_prefetch.json
```

[K 预取同步审计](../../reports/ascend-k-prefetch-sync-audit/README.md)保留完整工具输出、源代码差异、buffer/flag 生命周期分析和本地调度模型检查。本地未运行 CANN 编译或 NPU 执行；本轮是待远端验证的性能候选。

## K 预取回测与 Q 在 L0A 内复用

K 预取回传结果：native 88.0115 → 87.9880 ms，配对节省 +0.0249 ms（7/9 更快）；prepared 配对节省 −0.0569 ms（4/9 更快），full −0.0512 ms（3/9 更快）。没有稳定的全 API 收益，因此不将 K 预取叠加到下一版。profiler 的任务均值 88.2993 → 88.2120 ms 是另一种采集口径，不能代替上述配对结果，也不能据此断言所有 K 搬运都已被隐藏。

下一版仍从已测的 scale-cache 出发，减少 QK 主循环内 **Q 的重复 L1→L0A 搬运**：

```bash
git pull origin master
bash scripts/run_mindie_laser_q_resident.sh
```

固定 `ky=2` 的一个 descriptor 内，两块 Q128 会与每个 K128 计算。原版每次都重新搬入这两块 Q，每块需要 8 条 Load2D；`kx=32` 时一共 512 条调用。两块 INT8 Q 各占 16 KiB，已有两个 L0A 槽位分别从 0 / 32 KiB 开始。每个 K 块计算后，L0A ping/pong 翻转两次，后续 K 仍对应相同 Q 数据；这个循环内没有其他 L0A 写入者。

候选仅在第一个 K 块加载 Q，后续 K 直接复用现有 L0A 内容，因此普通 `ky=2,kx=32` descriptor 降为 16 条 Q Load2D 调用。不增加片上内存。每个新 descriptor 从 `n_index=0` 重新加载，因此不会把 Q 的有效性跨过 PV 或其他 descriptor；`ky!=2` 保留逐 K 加载，原上三角尾块的 skip 语义保留。原有 L0A ready/free 信号、MAD、FIX、DEQF16、K 搬运、scale 和跨核顺序保持，唯一删减的动态工作是后续 K 对应的 Q 搬入。这是源代码级搬运量减少，不能等比例推算耗时。

一键入口固定 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮；`NPU_DEVICE` 可选。只构建独立 `VcLaserQuantQresidentAttention`，路径使用 `mindie_laser_quant_qresident_*`；复用已有 scale-cache、融合量化器和 FP16 基线，源码来自本地固定 Git objects，不访问 GitHub。表中 `scale cache` / `Q resident` 使用同进程随机交错计时；native/prepared 使用相同 tensor，full 每次重新预处理。

计时前保留 structured/decoded、scale 偏移/原地更新、长形状全输出一致性门禁，并针对 Q0/Q1 不同内容、多个 K section、切换 Q 组/batch/head、同地址刷新增加检查。全部 padded native FP32 输出必须精确一致。公开接口会将 Q pad 到 256，所以 `ky!=2` 的兼容分支另由本地实际 C++ 循环模型检查，不把它误称为已执行的 NPU 分支覆盖。

结果在 `results/mindie_laser_q_resident.json`，日志同名 `.log`。已有候选构建后可只重测：

```bash
python3 scripts/profile_mindie_laser_scale_reuse.py --experiment q-resident --trace --output results/mindie_laser_q_resident.json
```

[Q 常驻同步审计](../../reports/ascend-q-resident-sync-audit/README.md)记录未变的事件/计算序列、Q 数据及 slot 生命周期、原始 CANNBot 输出与 host 模型限制。候选未在本地运行 CANN/NPU；是否能改善约 88 ms 的 native 延迟，必须以远端正确性和配对测速为准。

## 固定 Cube 输出系数，Vector 融合反量化

Q 常驻的远端 native 配对收益约 0.10 ms；它减少了搬运，但没有解决 INT8 主核仍慢于原版 FP16 Laser 的问题。这次独立测试 QK 的结果路径：Cube 不再逐 Q128/K128 tile 配置反量化系数，固定用 `1/64` 将 INT32 点积输出为 FP16；Vector 将分块反量化与原有 attention scale 乘法融合。Q 常驻、FP16 PV、普通指数以及输入量化方式保持一致。

```bash
git pull origin master
bash scripts/run_mindie_laser_vector_scale.sh
```

默认 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮。需要上一轮 Q-resident 构建及已有 FP16/融合量化器的 manifest；脚本先验证这些缓存和实际 SoC，只编译独立 `VcLaserQuantVscaleAttention`，复用本地 Git objects，不访问 GitHub。结果为 `results/mindie_laser_vector_scale.json`，构建日志为 `results/mindie_laser_quant_vscale_build.log`。候选保留在独立 vendor/source/build 路径，没有替换已有实现。

这次改变了舍入顺序，不能要求逐位等价。令 `I` 为 INT32 点积、`h` 为 FP16 舍入、`g` 为清除低 13 位后的 FP32 `sQ*sK`：

```text
旧路径：h(h(I*g) * h(attention_scale))
新路径：transport = h(I/64)
        coefficient = FP32(FP32(64*g) * FP32(h(attention_scale)))
        score = h(transport * h(coefficient))
```

`1/64` 对包含 `-128` 的完整 INT8 范围安全：D128 最大点积除以 64 为 32768。系数无法安全表示为 normal FP16 时使用 FP32 乘法回退，避免先舍入系数造成下溢或溢出。分块系数采用紧凑 UB 广播布局，供一个 section 的多批行复用；没有创建 Q×K 大小的系数矩阵。

验收包含独立候选算术参考、尾块/多 batch/head、同地址输入和 scale 更新。长形状对**全部补齐后的 native 输出**记录 old/new RMSE，超过 1% 停止；该门限用于拒绝本次数值改动漂移，并非模型质量标准。相对 dense FP32 的误差仍单独报告。旧有调度候选的逐位相等要求没有放宽。此候选关闭 V 平滑、V 重排和 ExpCast，先验证 INT8 QK 结果路径本身是否值得保留。

同进程 `Q resident` / `Vector scale` 随机交错配对比较 native、prepared 和 full。FP16 Laser 在独立进程报告，profiler 时间也单列，不能把它们混成同一配对加速比。只有实机通过精度门限且稳定改善 native/full 才考虑保留；本地 CPU/源码检查不代表已完成 CANN 编译或 NPU 验证。

实现、缓冲区生命周期与 CANNBot 检查见[Vector 反量化审计](../../reports/ascend-vector-scale-sync-audit/README.md)。

## Vector scale 实机收益与原版 Laser 基线

2026-10-20 用户回传：Vector scale 已在远端通过编译和脚本检查。同进程、同输入随机交错的 Q-resident → Vector-scale 结果如下；native/prepared 复用相同量化 payload，full 每次重新预处理。

| 范围 | Q resident ms | Vector scale ms | 中位数之比 | 配对节省 ms | 更快轮数 |
|---|---:|---:|---:|---:|---:|
| native | 86.7829 | 79.2512 | 1.0950× | +7.5238 | 9/9 |
| prepared | 88.9755 | 81.0111 | 1.0983× | +7.9796 | 9/9 |
| full | 89.3135 | 82.8106 | 1.0785× | +6.4887 | 9/9 |

全部 padded native 输出相对旧版漂移为 0.118562% RMSE，max abs 为 0.0004175640642642975；这是两种舍入路径之间的差异，不是相对 dense FP32 的误差或模型质量指标。各 scope 独立计时，配对节省也不是两列中位数之差。

同次报告的原版 FP16 Laser 为 full 85.0981 ms、prepared 82.1948 ms、native 80.5880 ms。相对于它，Vector scale 的 full 中位数约为 1.0276×；原版与 INT8 分进程运行，不能由此宣布稳定领先。后续以原版 FP16 Laser 为主基线，同进程交错测量三种实现的完整 API，并单独报告各自 prepared/native 范围。原版指前面已经复现、源码反向重命名等价且带 provider 证据的 pinned Laser 构建，不走自动 attention 分发。

profiler 导出的 AIC scalar/Fixpipe 活动下降、AIV scalar/Vector 活动增加，支持继续检查系数准备开销。计数器存在流水重叠，不能相加成 kernel 时间，也不能解释为 QK、softmax、PV 的阶段百分比。

## 批量准备 Vector 系数，以原版 Laser 为主基线

```bash
git pull origin master
bash scripts/run_mindie_laser_vector_scale_bulk.sh
```

默认 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮。脚本只构建独立 `VcLaserQuantVscaleBulkAttention`，复用已通过的 Vector-scale、原版 FP16 Laser、融合量化器及本地固定 Git objects，不下载上游源码。`NPU_DEVICE` 可选。报告为 `results/mindie_laser_vector_scale_bulk.json`，完整日志为同名 `.log`。

新候选在 8～32 个 K128 block 的 section 中，将逐 block 的系数广播改为四条 `vbrcb` 批量广播；1～7 个 block 保留旧路径。系数计算、FP16 舍入、慢路径选择和最终乘法都保持一致，额外占用 64 B UB，并为标量写入与 Vector 读取增加明确的同步。QK、普通 softmax、FP16 PV 及输入量化不变，仍关闭 V 平滑、重排和 ExpCast。这减少的是系数展开指令，新增同步与调度变化可能抵消收益，需要实测。

计时前执行原版 FP16 正确性检查、INT8 结构化数值参考、同地址 scale 更新和边界检查；旧/新 Vector-scale 的 native 输出必须**全部精确一致**，包括长形状的补齐区域。追加的精确比较在连续入队检查完成后执行，避免 CPU 同步干扰原址更新场景。相对 dense FP32 的误差另按最多 32 个 query、全部 key/head/feature 采样报告。

主表同进程随机交错比较三行：

| 行名 | 实现 |
|---|---|
| `original_laser` | 已复现的 pinned 原版 FP16 Laser，普通 softmax |
| `vector_scale` | 当前通过实测的 INT8 QK、Vector 反量化版本 |
| `vector_scale_bulk` | 相同数值计算，批量展开系数的新候选 |

**以 full API 为主要指标**：每次重新预处理输入，`Laser/full > 1` 表示比原版完整调用更快；同时看配对节省和更快轮数。prepared/native 各自独立测量，不相加、不混用。两种 INT8 的固定输入使用同一 payload 对象；原版使用相同原始输入生成自己的 FP16 payload。原版是固定源码复现，不能冒称与安装包二进制相同。来源、实际加载库哈希和首次长形状调用的 provider 标记均在计时前检查；profiler 在主计时之后单独运行。

本地只完成源码生成、主机模型和脚本验证，尚无本候选的 CANN 编译及 NPU 加速结论。[批量系数同步审计](../../reports/ascend-vector-scale-bulk-audit/README.md)记录缓冲区、事件生命周期和工具覆盖限制。

## 与原版同进程回测：保留 Vector scale，停止叠加 bulk

2026-10-21 用户回传的同进程随机交错结果如下。每种实现每轮调用 10 次，共 9 轮；full 每次重新预处理，原版为前述固定源码 FP16 Laser 复现。

| 实现 | full ms | prepared ms | native ms | Laser/full | dense FP32 RMSE |
|---|---:|---:|---:|---:|---:|
| original_laser | 87.2905 | 83.5870 | 81.5745 | 1.0000× | 0.190947% |
| vector_scale | 85.9907 | 82.3968 | 79.5296 | 1.0151× | 1.365174% |
| vector_scale_bulk | 86.1240 | 82.5851 | 79.8424 | 1.0135× | 1.365174% |

Vector scale 相对原版的 full 配对节省为 +1.2094 ms，9/9 更快。这是本次形状、设备及数值设定下的小幅优势，不是等精度比较或跨场景结论。bulk 相对 Vector scale 的 full 配对节省为 −0.0615 ms（3/9 更快），native 为 −0.2991 ms（0/9 更快）；全部 padded native 输出精确一致。因此后续从 Vector scale 开始，不叠加 bulk。profiler 活跃时三者设备任务平均值为 80.9972 / 79.6046 / 79.9154 ms，单独记录，不混入 API 比率。

## 分两级归约行最大值

**2026-10-21 实机结论：该候选退化，不采用；保留 `vector_scale`。** 下方入口仅用于复现归约实验，完整结果见后面的回测记录。

```bash
git pull origin master
bash scripts/run_mindie_laser_vector_rowmax.sh
```

本候选只改变普通 dense 路径中的行最大值归约。旧路径先对完整的 128 元素行向量跨 K 块逐级取最大值，再做行内归约。新路径先按 K128 块求局部行最大值，再归约各块的小结果；保持原有 score 缩放、mask、padding、在线历史更新、指数、求和及 FP16 PV。目标是减少宽中间向量在 UB 内的读写和归约树层数，不是减少数学上必要的 max 比较总数。

仅四行处理、实际 K 块数 8～32、含 padding 后不超过 32 块的 dense section 走新路径，其他情况保留旧路径。新归约复用已有 rowmax scratch，不增加 UB 或跨流水 event。额外注意源/目标对齐、归约 stride 单位、mask 恢复和四行结果布局；CANNBot 扫描、手工生命周期检查及主机模型的边界见[行最大值归约审计](../../reports/ascend-vector-rowmax-audit/README.md)。

一键入口仍是 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、10 次/轮、9 轮，可用 `NPU_DEVICE` 指定设备。只编译独立 `VcLaserQuantVscaleRowmaxAttention`，复用已有 Vector scale、原版 Laser 和融合量化器；源码使用本地固定 Git objects。结果为 `results/mindie_laser_vector_rowmax.json`，日志为同名 `.log`。同进程主表三行为 `original_laser`、`vector_scale`、`vector_rowmax`，主要比较 fresh full API 的 `Laser/full`、配对节省和更快轮数，prepared/native 另外报告。

计时前保留原版精度检查、Vector scale 自身数值参考、scale 原址更新测试，并增加 K 块及 section 边界检查。新旧 INT8 版本要求全部 native 输出精确一致，长形状也检查补齐区域；dense FP32 误差另行采样报告。原版与 INT8 的固定 payload 各自生成，两种 INT8 的固定比较共享同一 payload。候选尚无本地 CCE/NPU 验证或性能结论，只有远端精度通过且 full API 有收益才考虑采用。

## 行最大值候选回测：正确，但明确变慢

以下为用户回传的 2026-10-21 远端结果，来源为 `results/mindie_laser_vector_rowmax.json` 的终端摘要；本地未取得该完整 JSON。测试仍为 BF16 BNSD `[1,28,37741,128]`，同进程随机交错，9 轮；原版 FP16 Laser 是主基线。

| 实现 | full ms | prepared ms | native ms | Laser/full | dense FP32 RMSE |
|---|---:|---:|---:|---:|---:|
| original_laser | 85.9412 | 82.9925 | 81.6012 | 1.0000× | 0.190947% |
| vector_scale | 84.0519 | 81.5878 | 79.5306 | 1.0225× | 1.365174% |
| vector_rowmax | 97.5193 | 95.0846 | 93.0113 | 0.8813× | 1.365174% |

Vector scale 对原版 full 的配对节省为 +1.8843 ms，9/9 更快；本次速度比为 1.0225×，但 dense 误差不同，并非等精度比较。rowmax 相对 Vector scale 的 full/prepared/native 配对节省分别为 −13.4784 / −13.5035 / −13.4772 ms，均为 0/9 更快；全部 padded native 输出精确一致。因此这是性能退化，不是数值回归。

独立 profiler 设备任务总时长均值为 original 81.5435、Vector scale 79.3816、rowmax 92.9811 ms/call；各自 3 次调用、3 个设备任务。这支持退化发生在 attention 设备执行内部，而不是新增 Python 调用或量化预处理；这些数据不能定位到某一硬件流水，也不与 API wall time 混算。profiler 的无 warmup 提示不作用于已完成预热、未开启 profiler 的主计时。

源码层面，32 个 K128 块、四行时，新路径执行 4×32 个局部水平归约，再做四个合并归约；旧路径主要用逐元素 `vmax` 折叠 K 块，最后执行四个水平归约。新路径减少了宽中间向量的写入，却显著增加了水平归约的重复执行。归约吞吐、stride 布局或编译调度抵消搬运收益，是与现有证据一致的解释，**尚非硬件计数器证实的具体瓶颈**。不能由源码调用条数更少推断执行更快。

后续以 Vector scale 为起点重新检查 QK、PV 和 Cube/Vector 等待链路。原有 phase probe 基于更早的 scale-reuse 实现，不能直接当作移动反量化之后的当前瓶颈证据；应先适配和校准当前版本的诊断，再决定分块或流水改动。不将本次 rowmax、上次 bulk 叠加进后续优化。

## 当前 Vector scale 的 Cube 阶段诊断

```bash
git pull origin master
bash scripts/run_mindie_laser_vscale_phases.sh
```

固定 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、每轮 10 次、9 轮校准和 5 次稀疏采样；`NPU_DEVICE` 可选。入口读取已有 `mindie_laser_quant_vscale_build.json`、原版 Laser 和融合量化器的构建记录，从本地 Git objects 生成独立 `vc_laser_cube_vscale_probe`。原版 FP16 Cube probe 走已有增量构建和来源校验；旧 scale-reuse 的 probe 仍保留其原身份，不能混充当前版本。报告为 `results/mindie_laser_vscale_phases.json`，完整日志为同名 `.log`。

这次交付是诊断适配，不是新的加速候选。当前 `vector_scale` 的计算和 Vector 代码保持原样；只在 Cube 的完整 QK/PV 调用边界增加可关闭的稀疏采样。QK 组包含地址映射、输入搬运、矩阵乘及 score 输出；PV 组包含概率/值搬运、矩阵乘、历史缩放和存在时的末尾归一化。当前反量化在 Vector 执行，**不计入这里的 QK Cube 组**。采样也不覆盖调用前的跨核等待，不能凭它把全部 attention 延迟分摊为 QK、softmax、PV 百分比。

先验证每个诊断模式相对自身生产版本的全部 padded native 输出精确一致，再比较 production、plain、issue、completed 的配对 native API 时间：plain 包含额外 trace 分配及编译足迹；issue 记录发射窗口；completed 增加完整调用前后的流水排空，能观察完成窗口，但会改变重叠。四种模式共享本算法的同一 payload，结果用于量化探针扰动。

当前 INT8 与原版 FP16 在独立进程中分别校准，使用相同原始输入生成各自的 payload；跨算法的这些 native 时间不作为新的配对加速比。原版 Laser 的主性能对比继续以已完成的同进程 full API 结果为准。稀疏记录仅覆盖 Cube 核 0～3 的 first/middle/last 逻辑组，按 descriptor、K 块、history/normalize 次数分组，保留原始 system ticks，不换算核心周期或外推整核毫秒。

将两组校准表和阶段表回传后，先判断当前版本的主要变化在 QK 输出路径还是 PV/history；若仍需区分 Vector/跨核等待，再增加相应诊断。实现和来源/同步检查见[当前 Vector scale Cube 诊断审计](../../reports/ascend-vscale-cube-phase-audit/README.md)。本地没有 CCE/NPU，远端编译、精确输出门限及扰动校准通过后才解释阶段数据。

## 当前 Cube 采样回测：QK 路径变短，优先检查 PV 流水

2026-10-21 用户回传 `results/mindie_laser_vscale_phases.json` 的终端摘要；本地未取得完整 JSON。本次脚本完成两个 worker，经过自身基线输出门限、9 轮校准及 5 轮稀疏采样。原版与当前 INT8 仍在独立进程运行，不能用以下 calibration baseline 的 native 时间制造新的配对加速结论。

| 算法 | baseline native ms | baseline → completed 配对增加 ms | 相对 baseline 中位数的量级 |
|---|---:|---:|---:|
| vector_scale | 79.5117 | +0.0226 | 约 0.028% |
| original FP16 | 81.2538 | +0.2918 | 约 0.359% |

这里的 paired added 是同轮差值的中位数，不是表中两个中位数之差。整体扰动较小，支持继续使用稀疏采样作方向判断；这不代表 sampled completed 窗口没有被 drain 改变，不能拿它求生产流水占比。

下表为 completed elapsed 的中位数，单位是 **raw system ticks**。各行 descriptor=1、Kblocks=32、logical tiles=64、samples=20；middle PV 含一次 history，first PV 不含 history，两者均无末尾 normalize。

| 阶段 / 位置 | original FP16 | vector_scale | 当前窗口相对原版缩短 |
|---|---:|---:|---:|
| QK first | 1024.0 | 640.0 | 37.5% |
| QK middle | 985.5 | 807.0 | 18.1% |
| PV first | 1426.5 | 1352.0 | 5.2% |
| PV middle | 1369.5 | 1345.5 | 1.8% |

尾组只有 8 个 K block，单独观察：QK 为 289.5 → 182.0 ticks；PV 为 756.0 → 759.0 ticks，均含一次 history 和一次 normalize。issue 模式也显示当前 QK first/middle 比原版短；没有把 issue 与 completed 两种调度的数值相减来推算生产阶段。

本次支持的结论是**当前整条 QK Cube 路径已经有效缩短，而 PV 组没有相应幅度的缩短**。不能把它全归功于 INT8 乘法：当前版本还包含 Q 驻留、固定 FIX transport 和反量化搬到 Vector。尤其 Vector 反量化已经移出本次 QK 计时区间，仍要支付执行成本。PV middle 的完成窗口约为当前 QK 的 1.67 倍，是优先检查 PV 的依据；它不等于 PV 占整个 attention 的 62.5%，也不能证明 FP16 MAD 是唯一瓶颈。PV first 没有 history 却仍较长，说明只盯住 history 或尾部归一化不足以解释长窗口。

下一候选优先保持原版 FP16 P/V 与累加顺序，检查 PV 的数据搬运和流水：现有每个 K128 块先搬两个 P 块与一个 V 块，再等待 L1 数据并装载 L0；在 D128、Ky=2 时，源码包含 8 次 V 和 2×8 次 P 的 L1→L0 load 调用。需要判断是否能利用既有双缓冲更早发射下一块搬运、减少装载调度开销，并核对与 QK 共用的 L1/L0 槽和事件生命周期。现有代码本身已有异步重叠，不能仅凭这些调用顺序断言搬运完全串行，也不能把 load 调用次数当作硬件耗时。

继续以 `vector_scale` 为基线，不叠加 rowmax/bulk，不立刻改变 PV 量化或重新加入 ExpCast。候选的数值门限保持全部 padded native 输出精确一致；收益仍以原版 FP16 Laser 的同进程随机交错 full API 为主要验收指标。若 PV 候选无收益，再补当前 Vector 反量化及跨核等待诊断，不能将旧 scale-reuse 的 Vector 数据套用到现在。

## PV 内下一块 V 的预取候选

**2026-10-21 实机回测：同 payload 全部 native 输出精确一致；相对 Vector scale 的 full/prepared/native 配对计时均为 9/9 更快。保留该候选作为后续实验基线，Vector scale 继续作对照。详见下方回测。**

```bash
git pull origin master
bash scripts/run_mindie_laser_pv_prefetch.sh
```

这轮只改变 PV 的 V 搬运发射位置。每个 descriptor 在 history 处理后搬入 V0；当前 V 完成 L1→L0B 装载后，提前释放其 L1B 槽，并把下一块 V 的 GM→L1 搬运发到另一槽，使其有机会与当前 Ky=2 的矩阵乘重叠。最后一块不多搬；不跨 descriptor 投机；原 P 搬运、全部 MAD 累加、FIX 输出、Vector 反量化及 softmax 不变。使用既有双缓冲和事件，不增加 L1/L0/UB 容量。当前 `vector_scale` 继续保留，候选是否更快以远端结果为准。

一键脚本固定 BF16 BNSD `[1,28,37741,128]`、warmup 5、10 次/轮、9 轮，设备默认为 0（可选 `NPU_DEVICE`）。只构建独立 `VcLaserQuantVscalePvPrefetchAttention`，复用已验证的当前 Vector scale、原版 FP16 Laser 及融合量化器，源码从本地 Git objects 重建。主表三行为 `original_laser`、`vector_scale`、`vector_pv_prefetch`，同进程随机交错，full API 每次包含新预处理；prepared/native 与 profiler 另列。

计时前检查原版数值、当前 Vector scale 自身参考、原址 Q/K scale 连续更新，以及 K128/section 边界、多头映射、A/B/A/B 连续排队输入；新旧 INT8 要求全部 padded native 输出精确一致。公开 payload 按 256 token 补齐，逻辑单数 K128 块边界不冒充真实 Cube 奇数循环；循环及事件主机模型另外覆盖。结果保存在 `results/mindie_laser_pv_prefetch.json`，日志为同名 `.log`。实现边界和同步审计见[PV V 预取记录](../../reports/ascend-pv-prefetch-sync-audit/README.md)。

## PV V 预取回测：保留小幅收益，继续检查 PV 数据路径

以下为用户回传的 2026-10-21 `results/mindie_laser_pv_prefetch.json` 终端摘要。本地未取得完整 JSON；本次粘贴没有 full API 主表和 dense RMSE 列，不能补造本轮相对原版 Laser 的 full 加速比或精度数字。脚本默认 BF16 BNSD `[1,28,37741,128]`，同进程随机交错，9 轮。

| 实现 | prepared ms | Laser/prepared | native ms | Laser/native |
|---|---:|---:|---:|---:|
| original_laser | 82.3129 | 1.0000× | 80.6752 | 1.0000× |
| vector_scale | 81.4107 | 1.0111× | 79.0780 | 1.0202× |
| vector_pv_prefetch | 80.9052 | 1.0174× | 78.5039 | 1.0277× |

相对原版，候选的 prepared/native 配对节省为 +1.3978 / +2.1759 ms，均为 9/9 更快。native 包含分配和派发，不是纯 kernel 时长；原版 FP16 与 INT8 使用各自 payload，输出 dtype 也不同。

相对上一版 Vector scale，固定范围共享同一量化 payload；full 每次重新预处理：

| 范围 | old/new 中位数比 | 配对节省 ms | 更快轮数 |
|---|---:|---:|---:|
| full | 1.0035× | +0.4356 | 9/9 |
| prepared | 1.0062× | +0.4904 | 9/9 |
| native | 1.0073× | +0.5541 | 9/9 |

old/new 为两个中位数之比，配对节省为逐轮差值的中位数，两种统计量不能互相替代。三个范围独立测量、不相加。全部 padded native 输出精确一致：这是相同算法的调度优化通过了回归门限，不表示 INT8 与原版 FP16 Laser 等精度。

profiler 活跃时，三者累计设备任务平均时长为 original 80.5560、Vector scale 79.0523、PV prefetch 78.4636 ms/call，各自 3 次调用、3 个设备任务。相对旧 INT8 减少 0.5887 ms，方向与未开 profiler 的 native 配对结果一致，支持收益来自 attention 设备执行内部。它没有测得单独的 V DMA 或 PV 阶段时间，不能把该差额全部归因到某个硬件流水，也不与 API wall time 混算。profiler 无 warmup 提示不作用于已完成的主计时。

保留 `vector_pv_prefetch` 作为后续单变量实验起点，继续保留原版 Laser 和 Vector scale 对照。不叠加此前退化的 bulk/rowmax。下一步可检查 PV 的 P 搬运及 L1→L0 装载链，再单独尝试下一块 P 的 GM→L1 预取：先确认 Vector 已发布当前 descriptor 的 P，两个 P 行块所在的 L1 槽在最后一次 L1→L0A 读取完成后才释放，不跨 descriptor，也不同时调整 MAD 或 L1→L0 顺序。现有结果支持该方向，但不证明整个 kernel 是带宽瓶颈，也不保证同时预取 P/V 会更快。
