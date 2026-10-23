# Laser 的 msOpProf 采集

目标是对原版 FP16 Laser 和已跑通的 INT8 `vector_pv_prefetch` 收集同口径硬件指标，判断下一步应该优化 Cube 计算、Vector 概率计算、数据搬运还是调度。使用已有算子二进制，不构建、不下载，也不启用此前失败的内部 PV 探针。

在远端仓库执行：

```bash
git pull origin master
bash scripts/run_mindie_laser_msopprof.sh
```

默认设备 0，BF16 输入，BNSD `[1,28,37741,128]`。其他物理设备用 `NPU_DEVICE=1 bash scripts/run_mindie_laser_msopprof.sh`。如果设置过 `ASCEND_RT_VISIBLE_DEVICES`，先取消该映射，再通过 `NPU_DEVICE` 选择物理卡；脚本遇到非空映射会在设备工作之前停止。

脚本从稳定构建清单读取 CANN 环境，在 PATH 和该 CANN 的工具目录查找 `msopprof`，其次检查 `msprof op` 包装入口。读取本机 `--help` 确认所需参数。如果工具安装在其他路径，可设置 `MSOPPROF_BIN=/absolute/path/to/msopprof`。缺少工具时保存检测结果并停止，不自动安装或联网。

每个版本先以独立进程直接调用一次，核对来源、运行和抽样输出；成功后再由 msOpProf 直接启动单进程 worker。worker 只准备一次输入并调用一次目标 native API；不在被采集程序内派生其他 worker，也不启动 torch_npu profiler。采集后检查输入哈希和相同 query 行的输出哈希与直接调用一致；这是抽样检查，不代表所有输出逐元素验证。

## 采集范围

msOpProf 使用精确的算子名前缀、`--launch-count=1`、`--warm-up=5` 和 `--aic-metrics=Default`。前缀来自已核验构建身份，避免采到输入转换或量化辅助算子。`launch-count` 是匹配算子的采集上限，不代表跑五轮或取多次中位数。若本机支持，显式使用 `--replay-mode=kernel`，并关闭提前终止 application 的 `--kill`。

依据 [msOpProf 官方用户指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)，Default 包括算术、流水、内存、缓存和冲突指标。具体导出字段随设备及工具版本变化；摘要保留原始名称和单位，缺失项标为缺失。

| 数据 | 需要观察的问题 |
| --- | --- |
| `OpBasicInfo` | 是否匹配目标 kernel，任务时长、核数和频率是否可比 |
| `PipeUtilization` | AIC/AIV 各核的 Cube、Vector、Scalar、MTE、FIXP 活跃情况与负载差异 |
| `ArithmeticUtilization` | 计算量、指令类型和计算单元利用情况 |
| `Memory` / `MemoryL0` / `MemoryUB` | 搬运量、带宽、片上存储访问和相关停顿证据 |
| `L2Cache` / `ResourceConflictRatio` | 缓存命中及资源冲突是否与时间变化一致 |

这些是整核诊断。某个 pipe 活跃占比高不独自证明它是关键路径；不同通路可以重叠，不能把占比相加，或把 Cube 占比直接解释成 QK/PV 各自耗时。工具不会提供“寄存器值实时变化”的通用视图。逐指令时间线或源码热点属于更深入、依赖硬件/工具支持的采集，本入口不启用 `TimelineDetail`、`Source`、`KernelScale` 或修改 kernel。

## 把什么结果发回来

完成后执行：

```bash
cat results/mindie_laser_msopprof_summary.txt
```

先提供这份摘要。主报告为 `results/mindie_laser_msopprof.json`，其中 `run_root` 指向本轮独立目录。该目录保存工具 help、命令、构建来源、直接运行/采集日志、worker 输出检查，以及原始 OPPROF 文件和 CSV。采集命令完成后冻结该目录的 `manifest.json`，供摘要核对哈希；最终解析状态写入外层主报告。摘要逐任务、逐文件保留身份与字段，不把不同 kernel 或不同 counter pass 混为一个统计。

如果只看到 `partial`，需要看缺失项及本轮日志，不能把工具退出 0 等同于成功采到目标 kernel。任何直接运行、采集或数值检查失败都会停止后续设备任务，不自动重试。每次主机调用有 600 秒上限；超时保留证据并停止后续采集，主机进程退出不证明设备任务已经结束。

真正的提速验收继续用 `bash scripts/run_mindie_laser_stable.sh` 的无 profiler 配对 API 测试。msOpProf 的 kernel replay 可能清理 L2，重放和采集也会改变运行条件，因此本报告不与之前的 full/native API 毫秒直接混算加速比。

## 可选：采 FP16 ExpCast

先成功完成 `bash scripts/run_mindie_laser_fp16_expcast.sh`，再运行：

```bash
LASER_MSOPPROF_EXPCAST=1 bash scripts/run_mindie_laser_msopprof.sh
```

这会追加新 provider 的关闭/开启两组。采集器要求此前 ExpCast 验证报告通过，且源码、构建清单、bridge、设备二进制与预处理器哈希和当前产物一致。默认只采已验证稳定路线。

实现参考 CANNBot [ops-profiling](build/cannbot-skills/ops/ops-profiling/SKILL.md) 的 msprof-op 工作流，并按官方参数和本机帮助能力选择命令。本地只能验证编排、数据解析和失败处理，真实指标采集由远端完成。
