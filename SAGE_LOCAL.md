# 独立编译 SageAttention 2.2.0

[源码、编译和对比指南](third_party/sageattention_local/README.md)

源自官方 `thu-ml/SageAttention` 的 v2.2.0，commit
`eb615cf6cf4d221338033340ee2de1c37fbdba4a`。保留 Apache-2.0 许可证及来源，
以 `sageattention-local==2.2.0+vc1` 安装、`sageattention_local` 导入。

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
bash scripts/benchmark_37741.sh --backends flash3 sage sage_local --baseline sage --output results/h200_sage_local_compare.json
```

`sage` 是现有安装，`sage_local` 是本地源码编译版。没有改动 VC 的 CUDA 内核和包版本。
预期同类内核能达到同一性能档次，但没有远端结果前不报告速度或精度结论。

本地是 Windows / RTX 3060 / PyTorch 2.3，无法执行此 Hopper 路径；
本轮验证来源哈希、Python 入口命名隔离、fake 函数覆盖回归、benchmark 适配和源码打包。
相关测试 41 项通过；49 份上游文件来源哈希通过，其中 37 份 CUDA/C++ 或 Triton 文件保持原样。
独立源码包的内核文件、许可证与来源记录已检查，Shell 编译入口通过语法检查。
Linux / CUDA 13 原生构建、Triton JIT、H200 正确性与速度由远端脚本验证。
构建失败或测试失败会中止，不会替换成已安装 Sage、SDPA 或 VC。
