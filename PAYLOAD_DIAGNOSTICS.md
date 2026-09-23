# 0.2.1：固定量化字节的 CUDA 对拍与 Q/K 粒度审计

上一轮 H200 结果：只量化 Q/K 时约 3.76%，完整量化约 5.1%～5.3%；
固定聚类排列后，CUDA 与 Python 参考仍有约 0.5%～0.6% 差异。
本轮进一步固定 E4M3 字节、scale、存储均值，逐层查找差异来源。

## 已修正的参考舍入差异

在本地 PyTorch 2.3.1 / CUDA 12.1 上，标量表达式 `amax / 448.0` 与 CUDA
FP32 除法有时相差 1 ULP。这个差异会让 FP8 边界附近的元素变成相邻编码。
同一份默认 BF16 随机输入中，修正前有 357/262144 个 V 编码不同；用加宽运算
核对发现 native scale 等于正确舍入的 FP32 商，而标量表达式的 1225/2048 个
V scale 与该商不同。

0.2.1 的参考改用设备上的 FP32 Tensor 除数，显式保留除法。
这修正了 Python 参考的量化舍入，**生产 CUDA forward 的数值路径、FP8 格式与默认粒度未改**。
修正后，本地默认输入在平滑开/关两组的 Q/K/V 编码、scale、均值均与原生预处理一致。
此结果不等于已经解释 H200 上所有 0.5% 差异，也不预示整体约 5% 的量化误差会消失。

## 在 H200 运行

新增诊断导出了生产路径使用的 CUDA 内核，因此这次须重建 0.2.1：

```bash
git pull --ff-only origin master
bash scripts/validate_h800.sh
python3 scripts/diagnose_cuda_payloads.py --output results/h200_vc_payloads.json
```

脚本默认 `[1,2,1024,128]`、BF16、seed 1234，所有 query/key。
继续支持 `--capture`、`--scale` 等输入选项，含义与阶段诊断相同。
没有 Hopper 或新诊断接口会明确报错，CLI 不会用 FP32 替代 FP8 GEMM。
这不是性能测试：Python 逐 tile 调用、检查与同步开销都不具有内核测速意义。

## 三层证据

第一层 `preprocessing` 比较变换后的 Q/K、量化字节、scale 与存储均值。
`Q/K_*_same_transform` 进一步让双方编码完全相同的变换结果，区分变换误差与编码误差。
这里包含零 padding；报告有实际元素数。字节行应看 `different_elements/fraction`，
字节编号的 RMSE 不是解码值误差。

第二层按 tile 检查生产 CUDA 内核：

| 行 | 双方共同的输入 | 比较对象 |
| --- | --- | --- |
| `QK_same_bytes` | Q/K E4M3 字节 | cuBLASLt FP8 与解码后 FP32 的原始矩阵乘结果 |
| `P_bytes_same_scores_state` | native QK 结果、scale、上一轮 max/denom | Softmax/ExpCast 编码字节 |
| `Softmax_*_same_inputs` | 同上 | alpha、行和、running max、denom |
| `PV_same_bytes` | native P 字节和相同 V 字节 | PV 原始矩阵乘结果 |
| `accum_same_inputs` | native PV、行和、alpha、旧累加器、V 元数据 | 在线累加结果 |
| `normalize_same_state` | 相同累加器、denom | 输出归一化 |

这些检查会给下一阶段喂相同输入，避免把前一阶段传播过来的误差误判为本阶段错误。
JSON 保存跨 tile 按元素汇总的误差、首个差异所在的 batch/head/tile。

第三层比较完整输出，打印时双方都转为 API 的输出 dtype：

| 行 | 用途 |
| --- | --- |
| `shared_payload_CUDA_vs_FP32_replay` | 固定 native 字节/scale/均值，观察后续算术差异 |
| `payload_encoding_effect` | 两边都用 FP32 replay，仅切换 native/reference 量化数据 |
| `reference_scale_order_and_output_cast` | 比较 GEMM 后恢复 Q/K scale 与原参考在 GEMM 前反量化的差异 |
| `replay_vs_production_CUDA` | 检查逐 tile 重放与正式 CUDA API 的输出是否一致 |
| `production_CUDA_vs_reference` | 重新测量完整 CUDA/参考差异，双方仍固定相同排列 |

JSON 也保留 FP32 中间输出比较；与 BF16/FP16 API 比时该原始项含输出舍入影响，
要与两边 dtype 对齐后的 `cast_output` 区分。误差不是可加的贡献，不能把表格行相加。

## Q/K 粒度审计的边界

当前原生实现：Q 每 128 tokens、K 每 256 tokens 共享一个 scale，每个 batch/head 独立，
支持的 head dimension 不超过 128；K 的两块 128-token attention tile 共用其 256-token scale。

[论文 §4.1 和附录 B](https://arxiv.org/html/2609.15810v1) 描述了不同 host kernel
与预处理链，但不足以确认这里的 Q/K 块尺寸、scale 与完整格式设置就是作者在 H200 的配置。
论文 §4.5 还单独讨论了 QK/PV 都使用 FP8 的 B300 实现。
因此本轮没有据此把 H200 QK 擅自替换成 INT8，也不声称已逐位复现作者代码。

脚本额外比较 E4M3 的 `(Q,K)` token 块 `(128,256)`、`(128,128)`、`(64,64)`、`(32,32)`。
只改变参考中的量化分组，attention 的 Q/KV tile 都固定为 128，V/P 规则和输入相同。
报告 QK-only 与完整 ExpCast 的误差，属于数值敏感性分析，**不是自动选参或修改生产默认值**。
即使某个块尺寸误差更低，也不能直接当作已经达到论文配置或证明更快。

## 本地验证范围

扩展已在 Windows / CUDA 12.1 编译 `sm_86`、`sm_90` 并安装。
完整测试 **154 passed, 32 skipped**，跳过的是 30 项 Hopper 与 2 项 NPU 测试。
RTX 3060 上实际运行了量化、Softmax、累加与归一化诊断内核；
针对整个默认形状的连通性检查显式用 FP32 GEMM 测试替身，未冒充 Hopper FP8 GEMM。
Hopper 专用测试额外验证已知有符号 E4M3 GEMM 与完整重放，需要在 H200 执行。
