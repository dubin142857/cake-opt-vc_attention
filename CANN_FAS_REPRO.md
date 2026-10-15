# 先复现 CANN FlashAttentionScore，再加入 VC

当前 Ascend 自研内核的长形状性能还没有接近 CANN。用户在 A3 `Ascend910_9382` / CANN 9.0.1 上回传：M64 control **1169.2007 ms**、M64 ExpCast **1290.1728 ms**，同轮 CANN **128.7198 ms**、MindIE Laser **83.9254 ms**。这些都是同步 full API 时间。M64 改善了原 M32，但仍有约 9–10 倍的 CANN 差距，继续局部修改 ExpCast 不能解决整个执行结构的问题。

新路线先构建官方 FAS 源码，保留原有分块、流水、数据类型和 softmax；确认基线后，才在这个实现上接入 ExpCast。本次没有向官方内核加入量化、V 重排或 VC 逻辑。

## 找到的源码与调用链

当前测试调用 `torch_npu.npu_fusion_attention`，不是 `npu_fusion_attention_v2` 或推理专用的 `npu_fused_infer_attention_score`。对于 A3、BNSD、无 sink、无 actual sequence length 的配置，包装层调用 **`aclnnFlashAttentionScore`**。

- 官方算子仓库：[cann/ops-transformer](https://gitcode.com/cann/ops-transformer)。固定标签 **v9.0.1**，提交 **8038339a99bae113a7ae07f4547306d6d15bbddf**。
- [FlashAttentionScore 源码目录](https://gitcode.com/cann/ops-transformer/tree/v9.0.1/attention/flash_attention_score)：`op_api` 是 ACLNN 入口，`op_host` 负责 tiling，`op_kernel` 是设备实现。
- [对应 op-plugin 调用实现](https://github.com/Ascend/op-plugin/blob/3e9fc56dc6e642bd8c03481d71868999906cbed1/op_plugin/ops/opapi/FlashAttentionKernelNpuOpApi.cpp)：来自 PyTorch-NPU 2.10.0.post2 对应源码依赖；远端安装包的实际版本也会记录。
- `opbase` 依赖固定为上游 `cmake/third_party/opbase.cmake` 中的 **929902e577bb809a377a6ab17cb7b00a4dfddc1a**，不是任取它自己的同名发布标签。
- [官方自定义包编译说明](https://gitcode.com/cann/ops-transformer/blob/v9.0.1/docs/zh/invocation/quick_op_invocation.md)；[CANN Open Software License Agreement 2.0](https://gitcode.com/cann/ops-transformer/blob/v9.0.1/LICENSE) 保留在下载的源码中。

同名发布标签不是系统预编译包逐位相同的证明。先固定源码版本，再以远端实测判断两者差距。

## 一键运行

在原 Ascend 机器的项目目录执行：

```bash
git pull origin master
bash scripts/run_cann_fas_local.sh
```

默认 `python3`、逻辑 NPU 0、8 个编译任务，使用 Release / `-O3`。沿用已经安装的 CANN、驱动、torch 和 torch_npu，不自动安装或升级这些基础环境。需要现成的 Git、CMake、G++、Make 和上游要求的编译依赖。首次构建会下载官方源码、固定版本的 opbase 及第三方依赖；离线机器必须提前缓存它们，脚本不承诺首次离线构建。

2026-10-15 起，入口直接调用官方 CMake 构建目标，沿用之前的 Release / `-O3` 和单算子配置，绕过上游 `build.sh` 的无条件清理步骤。`gawk` 不再位于当前构建调用链。工具链环境仍读取 CANN 的 `bin/setenv.bash`。

如果前一轮已经中断或被 kill，**拉取后重跑同一条命令即可尝试续编**，不要手动删除 `build/cann_fas_source/build`。脚本检查已有缓存的源码目录、SoC、算子、vendor、CANN 路径和构建类型；发现不匹配时保留目录并报错，不静默清空重来。首次加入新配置或上次配置未完成时会重新 configure；后续配置完整时直接执行 `cmake --build ... --target package`，由构建系统复用已有产物。

新的 `CMAKE_PROJECT_INCLUDE` 模块只将自定义包中未被 ACLNN 链接、也未安装的 ONNX OBJECT 目标移出默认 ALL。Protobuf/Abseil 外部目标在上游已经排除于 ALL；没有其他目标需要它们时便不会被连带构建。显式依赖仍保留，设备内核、ACLNN、proto 和 tiling 目标不裁剪。这不改变 attention 数学，也不证明推理速度有任何提升。

**续编仍有边界：** 首次切换配置会触发官方 prepare 子构建；它会重建嵌套的 `build/prepare_build`，不等于主构建目录被清空。该子构建可能长时间不输出日志，其并行策略由上游控制。主构建每 30 秒打印 heartbeat，表示进程仍在运行而不是完成百分比。构建步骤与失败状态写入 `results/cann_fas_build_steps.json`；构建日志采用追加模式，保留上一轮信息。已经完成的依赖和对象能否复用还取决于对应目标的依赖和时间戳，不能承诺恢复到某个固定百分比。

脚本依次执行：

1. 预检设备与环境；A3 使用官方构建族名 `ascend910_93`，A2 使用 `ascend910b`。
2. 在 `build/cann_fas_source` 准备固定源码，检查未知改动和设备源码完整性。
3. 首次配置或复用已有 CMake 缓存，构建 `flash_attention_score` 自定义包，vendor 为 `vc_fas`；跳过无关 ONNX 默认构建目标。
4. 安装到项目内 `build/cann_fas_install/vendors/vc_fas_transformer`。不会替换系统 CANN 安装。
5. 分进程运行系统版和自编译版，检查正确性与加载证据，然后比较同步 API 时间。

编译官方 FAS 会产生多个 shape/dtype/tiling 特化，耗时可能较长；这是编译时间，不是 attention 推理延迟。构建日志保存在 `results/cann_fas_build.log`。

## 测量口径与验收

固定 BF16，NHD 形状 **`[1,37741,28,128]`**；传入两个后端的张量均为 contiguous BNSD **`[1,28,37741,128]`**。非因果、无 mask、无 dropout、scale 为 `1/sqrt(128)`，`pre_tokens`/`next_tokens` 为最大 INT32，`sparse_mode=0`。两边都使用普通浮点 FAS，没有 QK INT8 或 ExpCast。

先跑小形状正确性检查，再测长形状。长形状参考取 32 个 query 行、使用全部 37741 个 key，在 CPU 上计算 dense FP32；其误差门限为 1%。两后端同算法的采样输出比较采用 0.1% relative RMSE 门限。这些属于实现验收，不是模型质量认证，也不是所有长形状输出的完整验证。

只比较 **同步 full API**，包括该 API 的内部处理、调度和分配。没有人为构造的 CANN fixed-core 时间，不与 kernel-only 时间混用。两个后端在独立进程中分组计时，以避免 torch_npu 的动态库与函数指针缓存；因此不是同一轮交错的配对试验，需要留意跨进程的设备状态变化。

## 怎样确认走了自编译路径

源码准备脚本只向官方 host API 和 tiling 入口加入可关闭的来源标记；所有 `op_kernel` 源文件保持原样。它记录源码哈希、提交、依赖锁和编译设备二进制哈希。

自编译 worker 在首次长形状调用时要求出现两个标记：`api` 和 `tiling`，并记录 tiling key、进程实际映射的自定义 API/tiling 动态库及哈希。缺少标记则停止，不把自动回退到系统版当成复现成功。正式计时前关闭标记文件写入。

这些证据能确认自定义 **host API 与 tiling** 被执行；包内存在设备二进制，仍不单独证明某个 `.o` 就是运行时实际选中的设备程序。性能与结果一致也不能代替这一证明。若后续出现库混用或结果异常，需要结合设备侧加载日志/trace 进一步核查，报告会保留这个证据边界。

## 输出文件与下一步

| 文件 | 用途 |
| --- | --- |
| `results/cann_fas_environment.json` | 设备、CANN 和 Python 环境 |
| `results/cann_fas_prepare.log` | 固定源码和依赖准备 |
| `results/cann_fas_build.log` | 官方编译输出 |
| `results/cann_fas_build_steps.json` | 续编配置、执行步骤、耗时与失败状态 |
| `results/cann_fas_install.log` | 项目内自定义包安装 |
| `results/cann_fas_build_manifest.json` | 源码、依赖、host 修改与设备二进制哈希 |
| `results/cann_fas_local_compare.json` | 正确性、时间、来源证据与失败信息 |
| `results/cann_fas_compare.log` | 完整对照日志 |

当前本地是 Windows，没有 CANN/NPU，尚未验证官方内核编译或性能。远端先完成原版对照；若自编译版明显慢于系统版，先查版本、编译选项、tiling key 和加载路径。基线成立后再做普通 softmax / ExpCast 的同流水对照；QK INT8 与 V 量化作为后续独立变量，不把不同算法、不同流水的差距都归因于 ExpCast。
