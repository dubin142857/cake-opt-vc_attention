# SageAttention 2.2.0 独立源码编译版

此目录复用 [thu-ml/SageAttention v2.2.0](https://github.com/thu-ml/SageAttention/tree/v2.2.0)，
固定 commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`，遵循 Apache-2.0。
原作者与许可证见 `NOTICE`、`LICENSE`；每份源文件及其哈希见 `UPSTREAM.json`。
`sageattn` 是原版内核的独立源码构建。`2.2.0+vc3` 新增独立的
`sageattn_vc` 实验入口：Sage INT8 QK + VC 的 V 聚类/分块平滑、ExpCast，
不能称为完整 VC 论文复现。使用与对照见仓库根目录 [SAGE_VC.md](../../SAGE_VC.md)。
当前版本 `2.2.0+vc7` 优化 ExpCast 的饱和整数转换、字节打包与概率求和调度。
保留 vc6 的共享通道 V scale 和两级 FP32 PV 累加；`kernel_mode="vc6"` 可执行上一版路径，
`kernel_mode="vc4"` 和 `v_scale_mode="block"` 也保留作对照。
同一量化数据下，本地辅助函数逐位一致；完整 H200 attention 与速度仍需远端验证。

## 构建与导入

目标：Linux H100/H800/H200，CUDA Toolkit >=12.3，PyTorch >=2.4、Triton >=3.0。
用户当前 H200 / PyTorch 2.13 + CUDA 13 环境需要实际编译验证。
安装依赖沿用当前环境，脚本不会升级 torch 或覆盖已安装的 `sageattention`。
所有编译源文件都已放入本仓库；远端无需再次从 GitHub 下载。

从 vc-attention 仓库根目录运行：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
```

脚本编译 `sageattention_local._qattn_sm90`、`sageattention_local._fused`、
`sageattention_local._vc_sm90`，目标固定 `sm_90a`。
先执行原版的 16 组随机输入与 2 组已知答案检查，再验证新入口的四种开关组合。
若当前环境安装了原版，还会同进程导入并输出双方误差。
随机测试的 relative RMSE <=0.10 仅作执行回归门槛，不能代表模型质量。

也可以直接安装本目录：

```bash
MAX_JOBS=4 python3 -m pip install ./third_party/sageattention_local --no-build-isolation --no-deps
```

```python
from sageattention_local import sageattn

# q/k/v: CUDA BF16 或 FP16；HND = [batch, heads, tokens, dim]
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```

你的 BSHD `[1,37741,28,128]` 输入也可直接设置 `tensor_layout="NHD"`。
本次支持/验证目标为 dense、noncausal、相同 Q/K/V head 数、head_dim 64/128 的前向推理；
其余原版入口源码虽保留，但此构建仅包含 Hopper 扩展。没有训练反向支持承诺。

## 和已安装的原版比较

先跑小形状，再跑完整目标；两条命令都包含预处理耗时、FP32 参考误差和双方输出差异：

```bash
python3 scripts/benchmark_cuda.py --backends sage sage_local --baseline sage --queries 1024 --tokens 1024 --heads 2 --dim 128 --check-queries 0 --output results/sage_local_1024.json
bash scripts/benchmark_37741.sh --backends flash3 sage sage_local --baseline sage --output results/h200_sage_local_compare.json
```

- `sage`：已经安装的原版 `sageattention.sageattn`。
- `sage_local`：本目录独立编译的 `sageattention_local.sageattn`。
- `local_speedup_vs_installed = 原版耗时 / 自编译耗时`：1.00 为相当，0.95 为约 95% 的原版吞吐，1.05 为约 105%。

报告记录双方 Python 模块位置、本地原生扩展位置、固定源码 commit。
长序列默认只抽查 128 个 Q 行的误差，每行使用全部 keys；两边测速均计算完整 attention。
两个包分别做量化和预处理，没有调用同一份已安装原版二进制。
在相同源码、配置和编译器下，预期同一性能档次；具体差距必须看同场实测。
仅同名的 `2.2.0` 版本号不能证明你的已安装包对应同一 Git commit。

## 原理与改动范围

Hopper 自动分派使用 `sageattn_qk_int8_pv_fp8_cuda_sm90`：

- K 沿序列方向减均值；默认 Q/K 做 per-thread INT8 量化，预处理是原版 Triton。
- QK 用 INT8 Tensor Core；Softmax 的权重 P 与按通道量化的 V 用 FP8 E4M3 PV。
- PV 采用原版 `fp32+fp32` 分级累加。此 Hopper 默认路径不做 VC 的 V 聚类/ExpCast。
- 原版 `sageattn` 的 CUDA/C++ 内核计算代码和 Triton 源码保持原样。
  新增 `csrc/vc/` 只供 `sageattn_vc` 使用。

本地修改包括：独立包名、Torch 算子注册命名空间、构建脚本、一个 Python 包装层修复和一处头文件依赖修复。
固定的上游标签在 `sm90_compile.py` 将第二个 fake implementation 定义为与真实 callable
相同的 Python 名称；此处给 fake 函数加 `_fake_impl` 后缀，防止覆盖真实入口。
这不意味着你的现有安装一定有此问题；你的既有精度结果应与本轮实测一起判断。
原版 setup/manifest/README 留在 `upstream/` 方便审阅。

`2.2.0+vc2` 去掉 `csrc/fused/fused.cu` 未使用的 `ATen/cuda/CUDAContext.h`，
避免它间接引入 `cusparse.h` 等本内核不使用的库头文件。此修复不改变计算代码；
来源校验会逆向还原该 include 并检查上游哈希，确保只有这处构建改动。

本地验证范围与远端待验证项见仓库根目录 `SAGE_LOCAL.md`。
