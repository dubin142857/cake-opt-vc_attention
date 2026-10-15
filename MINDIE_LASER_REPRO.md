# MindIE Laser 源码复现与性能对照

Ascend 主路线改为先复现 MindIE Laser，再考虑接入 VC 的 ExpCast。此前 CANN FAS 的完整源码构建保留为历史实验，不再作为当前默认操作步骤。

这轮保留 Laser 原有分块、Cube/Vector 流水和普通 softmax，没有加入量化、ExpCast 或 V 聚类。目标是确认自编译版本能够接近已安装 Laser 的精度和性能；尚无本次源码版本的远端 NPU 测速结果。

## 一键运行

在已安装 CANN、PyTorch、torch_npu 和 MindIE-SD 的原 Ascend 机器中执行：

```bash
git pull origin master
bash scripts/run_mindie_laser_local.sh
```

如果旧 FAS 构建仍在当前终端运行，先用 Ctrl+C 结束它。新脚本不会删除旧源码、编译目录或结果，也不会替换已安装的 MindIE 包。

默认使用 `python3`、逻辑 NPU 0、4 个编译任务，运行 BF16、NHD `[1,37741,28,128]`（内部 BNSD `[1,28,37741,128]`）。脚本根据设备选择 `ascend910b` 或 `ascend910_93`；用户当前回传设备为 `Ascend910_9382`，按后者构建。无需手填命令参数。

需要已有 Git、CMake、G++、Make 及 CANN 编译工具。首次准备会获取固定版本的官方源码，离线环境需要预先提供对应 checkout；这不是首次无网络安装器。后续运行复用构建目录，工具链不匹配会明确报错，不会静默清空缓存。4 个任务是外层构建和上游 prepare 子步骤的并发上限，不是编译时长承诺。

### 首轮 host 成功、设备二进制缺失的修复

首轮远端日志中，`.run` 打包和 host 库安装成功，但 `ops_kernel` 没有实际编译任务，最后报 `No installed Laser device binary was produced`。这是构建入口漏传 `CHECK_COMPATIBLE=TRUE`：上游 CMake 未引用该参数的空值，导致 `prepare.sh` 把后面的 SoC 选项当成它的值，丢失了芯片型号与算子名。准备阶段因此还编译了无关算子的 host 定义。

入口现已补齐参数，并在配置完成后检查指定 SoC 的 Laser 编译脚本和对应设备任务。脚本已生成但 CMake 任务仍缺失的旧缓存，也会原地重新配置；保留主构建目录已有 host 对象。配置后的设备任务仍为空时，会提前停止，不再继续打包。原有最终设备二进制检查继续保留。

遇到上述首轮失败，拉取修复后重跑同一个一键入口即可，无需手动删除 `build`。这次修复已通过 CPU 参数传递复现和缓存恢复回归；后续远端已完成设备和接口编译，进展见下。

### 已有 batch_invariant 库触发来源检查

后续远端已完成算子与 bridge 编译，原版 full API / fixed core 中位数为 **85.8195 / 83.2261 ms**。自编译版的小形状验收与首次长形状调用已完成，但来源检查把 CANN 目录下额外映射的 `batch_invariant` tiling 库判为污染，尚未输出自编译版计时和长形状精度对照。上述原版延迟不能作为自编译版的结果。

Laser 使用独立的 `VcLaserAttention`，额外库被加载本身不代表 Laser 被替换。新的 Laser 专用检查只识别实际构建所选 CANN 根目录下 `batch_invariant` 的指定 tiling 库，核验对应 SoC 的 `ops-info` 注册清单不含 Laser，记录库和配置哈希后允许共存。缺失配置、Laser 注册冲突、同名库出现在其他位置、其他未知 vendor 仍会停止；本地 Laser API/tiling/bridge 映射、成功 tiling 标记及正确性门限继续保留。原 CANN FAS 的同名算子检查不作放宽。

`batch_invariant` 有 [vLLM-Ascend 附加算子包安装流程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/batch_invariance.md)，不能仅凭安装在 CANN 目录就称其为原厂内置组件。本次只调整测速来源核验，不卸载已有包，不修改系统 OPP。拉取代码后重跑原一键入口，复用已有编译产物。

## 源码与构建范围

真正的 Laser 实现在 [Ascend/MindIE-SD](https://github.com/Ascend/MindIE-SD/tree/dev/csrc/ops/laser_attention)，不是此前查找的 MindIE-LLM。固定提交为 `71098c20ff90838d1fd1ca0a40515c751a3869b0`。

- `op_kernel/laser_attention.cpp`：混合内核入口，调用 Cube/Vector 实现。
- `op_kernel/CubeForward.h`：矩阵乘相关计算与流水。
- `op_kernel/VectorForward.h`：在线 softmax、归一化等 Vector 计算。
- `op_host/laser_attention_def.cpp`、`laser_attention_tiling.cpp`：算子注册与 tiling。
- `csrc/plugin/la.cpp` 及其调用辅助文件：沿用原版 ACLNN 调度和队列/stream 处理。

源码保留上游 Mulan PSL v2 许可。脚本只构建独立 Laser 算子和一个小型 PyTorch 调用接口，不构建完整 MindIE wheel、其他 MindIE 算子或之前的 CANN FAS 工程。设备代码仍需要 CANN 编译。

为避免误调用安装包，生成独立名称：

| 层级 | 自编译版本 |
|---|---|
| GE 算子 | `VcLaserAttention` |
| ACLNN API | `aclnnVcLaserAttention` |
| PyTorch 接口 | `torch.ops.vc_laser.la` |
| OPP vendor | `vc_laser` |

设备源码只做标识符重命名，并检查逆向重命名后与固定上游版本一致。host tiling 增加可关闭的成功调用标记；计时前关闭标记。接口复用原版插件辅助文件，只有 ACLNN 算子名变化。这里的 `Vc` 仅用于隔离名称，不代表已经加入 VC 算法。

构建目录为 `build/mindie_laser_source` 和 `build/mindie_laser_bridge`；算子安装到项目内 `build/mindie_laser_install/packages/vendors/vc_laser`，不写系统 OPP。保留构建日志和产物哈希，避免“编译成功但实际调用原版”的误判。

## 验证和计时口径

已安装版和自编译版在独立 Python 进程运行。自编译进程不导入 `mindiesd`，避免其初始化覆盖 OPP 路径；原版进程使用已安装 MindIE 的显式 Laser 入口。验证独立符号、实际加载的动态库、首次长形状成功 tiling 标记，并归档设备二进制哈希。这些是源码构建与 host 调度证据，不单独构成设备执行文件身份的完整证明。

沿用 MindIE 的预后处理：BF16 输入先转 FP16，序列补齐到 256、head dimension 补齐到 128，输出按原路径转换并裁剪。它不是 BF16 内核复现，也不是量化 attention。固定源码提交与已安装的 `mindiesd 3.1.0` 不保证字节相同，必须比较实际结果。

先验证 FP16/BF16、小形状对齐与尾块、常数 V 通道、完整 API 与 prepared API 一致性，再检查长形状的 32 个 query 行、所有 37741 个 key。dense FP32 参考在 CPU 计算，不计入 NPU 延迟。dense 相对 RMSE 门限为 1%，源码版/已安装版抽样输出的一致性门限为 0.1%；这些门限是实现验收，不代表模型质量验证或长形状全输出验证。

- **full API**：每次新做 dtype 转换和 padding，再调用原生 Laser，最后转换、裁剪并输出连续张量。
- **fixed core**：复用已转换和补齐的输入，仍包含原生分配/调度与输出后处理；不是纯内核时间。
- 默认 warmup 5 次，每轮 10 次、3 轮，同步测量 API 墙钟时间。两个后端按进程分组运行，不是同一轮随机交错的配对测试。
- 加速比为“已安装 Laser API / 自编译 Laser API”，大于 1 表示自编译更快，不与历史 CANN 或 GPU kernel 时间混算。

## 输出与当前验证边界

- `results/mindie_laser_build.log`：算子构建日志。
- `results/mindie_laser_bridge.log`：调用接口构建日志。
- `results/mindie_laser_build.json`、`mindie_laser_bridge.json`：构建配置与产物记录。
- `results/mindie_laser_local_compare.json`：最终验收、误差和性能对照。
- 同目录中的独立 worker 目录：输出抽样、进程库映射、tiling 标记和逐轮日志。

本地仅具备 Windows/CPU 环境，检查覆盖源码变换、构建计划、预后处理等价性、隔离与失败检测、脚本语法。CANN 原生编译和 NPU 正确性/速度需要在远端执行上述一键脚本后确认。

这次调整优先解决构建范围与基线可复现性：沿用原版调度、隔离算子名、限制嵌套构建并发，再以实测决定是否进入 ExpCast 改造。
