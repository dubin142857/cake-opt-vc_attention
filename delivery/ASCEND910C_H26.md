# H26 on Ascend 910C / A3

This branch starts at the H26 delivery top `8eb8273` (`0979605` is the last kernel change). It keeps the delivered **`v_smooth=False`** kernel and its K tile of 256. No N3/V-Smooth code is merged.

## Core-count decision

No A2 core count is hardcoded in the launch path. `ascend/csrc/vc_torch.cpp` queries `GetCoreNumAic()` on the selected device and launches `min(AIC cores, B × H × ceil(Q / tile_m))` blocks. It also requires at least two AIV cores per AIC and checks UB/L1/L0C resources for the selected SoC. Changing a constant from a presumed 910B count to a presumed 910C count would bypass this runtime contract. The selected A3 SKU must pass these checks on the actual device.

## A3 entrypoint

`NPU_DEVICE=<physical davinci ID> bash scripts/run_ascend910c.sh --long-shape` requires an explicit device and rejects an A2/910B SoC before build. The existing preflight obtains the exact `Ascend910_93xx` compiler target from the selected device; the same script then builds the package, runs the 141-case v1 suite and measures the long shape. Source the container's CANN `set_env.sh` first. Use only an idle, coordinated device and the personal container. No host drivers, shared services or other containers need changing.

The 910B2 binary SHA256 in `delivery/README.md` does **not** apply to A3: CANN and compiler target differ. Record the newly built `.so` SHA256 and exact SoC before interpreting any result. The 910B2 74% core gain is a reference, not an A3 measurement. Run 8 same-payload repetitions for both control and ExpCast, the long-shape precision check, and an ABBA whole-package comparison against `9e3a550` before claiming an A3 gain. The repository's pairwise script swaps complete packages because its Python `K_TILE` changes along the H26 ancestry.

When all devices are occupied, `SOC_VERSION=Ascend910_9382 PYTHON_BIN=<existing torch_npu Python> MAX_JOBS=2 bash scripts/build_ascend910c_offline.sh` compiles to `results/ascend910c-h26-build/` **without installing or loading on a device**. Use the exact SoC previously obtained from this machine's selected-device preflight; the offline build cannot establish runtime correctness.

## Current status

On the shared A3 machine, previous personal-container preflight identified `Ascend910_9382` with CANN 9.1. In the personal container, the offline H26 build for this exact target completed with two build jobs; its `.so` SHA256 is `79ee44cd8d508d052f046aae2269fc465fcd6d6654371cf6a31e55abb987b502`. It was built into a private project directory without package installation or NPU compute. The latest read-only occupancy check showed all 16 physical davinci devices in use, so this H26 build has **not** undergone A3 NPU correctness, determinism, or performance validation. It is a **buildable port candidate**, not an accepted 910C performance result.
