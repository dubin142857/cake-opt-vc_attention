# Q/K、P、V 分阶段误差诊断

这份工具用于定位当前约 5% 的输出误差，正式 `Config`、量化算法、C++/CUDA
源码和 0.2.0 扩展接口均保持原样。局部量化组在 Python 中将 E4M3 解码后使用
FP32 矩阵乘，不测速度；完整 CUDA 对照单独调用已安装的原生算子。

## H200 上直接运行

已有 0.2.0 的安装可以直接拉取运行，不需要重新编译。如果服务器还使用原来的
GitHub 地址，第一行切换为现在的 Gitee；显式指定 `master` 也兼容本地仍叫 `main` 的分支。

```bash
git remote set-url origin https://gitee.com/czf1234/vc-attention.git
git pull --ff-only origin master
python3 scripts/diagnose_cuda_accuracy.py --output results/h200_vc_stages.json
```

默认输入 `[1,2,1024,128]`、BF16、seed 1234，与原测速脚本使用同一种生成方式，
检查所有 1024 个 query 与所有 key。不同 PyTorch 版本或设备不保证生成完全相同的
随机张量；报告记录实际输入的 SHA256、形状与 dtype。TF32 关闭。

## 两组表格怎么看

`v_smooth_off` 和 `v_smooth_on` 分别关闭、开启 V-Smooth。每组内只计算一次
token 排列，各阶段共用该排列、K 去均值、Q/K Hadamard、分块和缩放规则。

| stage | 发生低精度近似的位置 |
| --- | --- |
| `none` | 不量化 Q/K/P/V，检查预处理及流式计算是否保持 dense attention |
| `qk` | 只对 Q、K 做 E4M3 量化 |
| `p` | 只把普通 exp 得到的 P 编码成 E4M3 后用于 PV |
| `v` | 只量化 V；平滑开启时包含残差 E4M3 与缩放均值的存储精度 |
| `qk+p`、`qk+v`、`p+v` | 对应两处同时量化 |
| `qk+p+v` | 完整普通 exp 量化参考，对应 control 或 V-Smooth-only |
| `expcast` 及其组合 | 用 ExpCast 替代 P 路径，共四组；`qk+expcast+v` 是完整 ExpCast 量化参考 |

普通 exp 路径的分母和均值恢复使用编码前的行和；ExpCast 路径使用解码后的行和。
这些规则与现有实现一致。每组的 12 个阶段都会计算完整 Q 后再取检查行，
`--check-queries` 仅控制误差采样，不会先裁剪 Q 而改变 Q 的量化 scale。

| 列 | 含义 |
| --- | --- |
| `FP32-out %` | 保留 FP32 累加输出时，相对独立 dense FP32 attention 的 relative RMSE |
| `cast-out %` | 再转回输入 dtype 后的 relative RMSE，可与原 BF16/FP16 API 的误差对应 |
| `cast max abs` | 转回输入 dtype 后的最大绝对误差 |

表格的 relative RMSE 使用百分数，JSON 使用比例，`0.05` 对应 `5%`。
`none` 的 FP32 输出应接近 dense；其 cast 输出仍包含 BF16/FP16 舍入误差。
先比较 `qk`、`p`、`v` 与对应组合，再比较两个平滑组。误差之间不能直接相加，
也不能用两个 relative RMSE 的差值解释某阶段独立贡献。

## 完整参考与原生 CUDA 检查

每组附带两项交叉检查：

1. `Diagnostic/full Python reference check`：局部诊断全部开启时，与原有
   `attention(..., backend="reference")` 对拍。共享排列应排除再次聚类造成的差异；
   若这里有明显差异，先检查诊断实现，不能直接据此归因量化误差。
2. `CUDA/FP32 %`、`REF/FP32 %`、`CUDA/REF %`：通过同一冻结排列调用正式 CUDA
   与 Python API，分别独立完成后续预处理和量化。它隔离了聚类差异，
   与原 benchmark 中“双方各自聚类”的口径不同，不能把两组 CUDA/REF 数字混用。

默认必须有 Hopper 与可加载的 0.2.0 原生扩展；缺失会报错。显式传入
`--reference-only` 可只测数值模拟，在普通 CUDA 卡或 `--device cpu` 上运行。
报告明确记录是否真的调用了原生 CUDA。此工具不设通用质量通过阈值。

## 真实模型张量

如果已有抓取文件，可保留完整上下文直接回放：

```bash
python3 scripts/diagnose_cuda_accuracy.py --capture /path/to/post_rope_qkv.pt --check-queries 128 --output results/h200_real_stages.json
```

文件为含 `q`、`k`、`v` Tensor 的字典，布局 `[B,H,N,D]`，Q/K 已完成 RoPE。
支持现有 H3 capture 的 `metadata.scale`，也可以用 `--scale` 明确覆盖；没有配置时
按原始 head dimension 缩放。保留张量 dtype，不自动转为 CLI 的 `--dtype`。
诊断始终对比第 0 步的平滑开启/关闭情形，不重放 capture 所在的完整去噪调度。
长序列的 Python 流式诊断仍可能很慢，采样 query 不代表裁剪了 K/V。

## 本地验证范围

当前完整测试为 **137 passed, 31 skipped**；跳过的是 29 项 Hopper 和 2 项 NPU 测试。
CPU 检查包含每个阶段对独立单 tile 公式、未量化流式输出对 dense、尾块、
输出 dtype、采样前保留全 Q scale、capture 自定义 scale，以及已知扰动的
CUDA 对比报告测试。另有 Hopper 上的真实原生常量 V 集成测试。
本地 RTX 3060 已跑通默认形状的全部 24 组数值诊断，完整组与现有 Python 参考
逐元素一致；没有把本地数值模拟报告当作 H200 原生运行结果。
