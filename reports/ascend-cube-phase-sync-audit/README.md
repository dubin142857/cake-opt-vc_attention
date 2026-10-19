# Laser Cube phase diagnostic: source and synchronization audit

This diagnostic follows the user-provided Vector/wait result. That result
motivates investigating Cube work; it does not prove a whole-kernel bottleneck.
Two independent providers instrument the same call boundaries: pinned original
FP16 Laser and the measured INT8 scale-reuse Laser. Neither provider replaces
its production baseline. ExpCast, V quantization, V reordering and V smoothing
are disabled. The pinned MindIE-SD revision is
`71098c20ff90838d1fd1ca0a40515c751a3869b0`.

## Source evidence and tool output

`regenerate.py` reconstructs four audit trees from local pinned Git objects
under `build/ascend_cube_phase_sync_audit`: FP16/INT8 baseline and probe. It
never changes the baseline source checkout. All local C++ source/header files
are scanned: respectively 12/13 files for FP16 baseline/probe and 13/14 for
INT8 baseline/probe. Each tool's raw JSON, stderr, command, return code,
tool hash and input hashes are retained. `summary.json` records all eight runs.

Both CANNBot tools exited zero for every tree. Baselines report zero findings;
each probe reports the following **unfiltered performance finding** at line 71
of `op_kernel/vc_laser_cube_probe.h`:

> SYNC-09, 性能: PipeBarrier<PIPE_ALL> 粒度过粗
>
> 若仅需 V 排序可用 PipeBarrier<PIPE_V>；确认无跨 PIPE 依赖后再收窄

There are no red/high findings. The performance warning is material: mode 2
deliberately drains multiple Cube pipelines and changes overlap. It is a
diagnostic, not a production synchronization optimization. Do not replace its
completion with PIPE_V: the observed work includes MTE, matrix and FIX work.

The sync analyzer recognizes only the one newly added `PipeBarrier`; the
flow analyzer recognizes zero operations in every tree. These regex tools
do not model Laser's raw MAD, DMA, `set_flag`, `wait_flag`, `wait_flag_dev`
and `ffts_cross_core_sync` dependencies. Empty results are not proof of safety.
CANN headers, macro expansion, compiler scheduling and device execution are
unavailable locally; the manual review below and remote gates remain necessary.

`source_comparison.json` verifies the following independently of the tools:

- Removing the diagnostic additions restores each Cube header byte-for-byte.
- Vector and both address mappers remain byte-identical within each family.
- Raw same-core/cross-core synchronization, MAD and output-copy call sequences
  are identical: INT8 Cube 78/78, FP16 Cube 77/77, Vector 91/91.
- `int8_CubeForward.h.diff` and `fp16_CubeForward.h.diff` retain line-numbered,
  zero-context patches. No matrix or diagonal helper body is modified.

Host OpDef/proto/tiling/entry changes add an explicit third INT64 output and
runtime mode; operator, ACLNN API, vendor and bridge identities remain separate
for FP16 and INT8 diagnostics. The original statistics/output allocation is
retained. These ABI changes are covered by the preparation and bridge tests.

## Completion boundaries and buffer lifetime

| Expected producer | Actual data path / consumer | Existing dependency and diagnostic boundary |
| --- | --- | --- |
| Q/K GM blocks | Existing QK MTE2/MTE1 loads and MAD | Original ping-pong event sequence is unchanged; no drain is inserted inside `cube1_matmul_op` |
| QK L0C result | Original FIX copy to score GM | INT8 uses MAD unitFlag 3 and matching DEQF16 COPY unitFlag 3; FP16 uses the corresponding F322F16 copy. The optional end drain occurs only after the entire call returns |
| Completed score GM | Original Vector score load | Original FIX cross-core notification and Vector wait remain unchanged; diagnostic trace writing before the notification can delay it and is calibrated |
| Vector probability and diagonal GM | Existing Cube wait, PV input loads and diagonal history update | Original cross-core wait precedes the PV call; the diagnostic begin occurs after that wait and before address mapping |
| Historical output / rescale diagonal | `diag_matmul(copy_out=false)` followed by PV accumulation | The history MAD uses unitFlag 2 without output copy. No diagnostic drain is placed between this retained accumulation and the following PV terminal MAD/copy |
| Completed PV L0C accumulation | Existing terminal MAD unitFlag 3 and FP32 FIX output | Original matching copy is issued before whole `cube2_matmul_op` returns; its optional final normalization diagonal uses matching MAD/COPY as well |
| Probe trace record | `((core*2+stage)*3+position)*24` INT64 words | One Cube owns each 192-byte record; all 24 fields are written after timing and all three disjoint 64-byte lines are cleaned out. The bridge checks base alignment and zeros trace each call |

The diagnostic does not allocate event IDs or introduce a cross-core signal.
It does not use unsupported M-to-S or FIX-to-S event directions. Its sole new
completion primitive is `AscendC::PipeBarrier<PIPE_ALL>()`, called only before
address mapping and after complete QK/PV invocations. All matching unitFlag
instructions have been issued at these boundaries. In particular, inserting
the drain immediately after history `matmul_fp32(copy_out=false)` would be a
different and unaudited change; this implementation does not do so.

QK logical rounds are `roundId`; PV rounds are `roundId`, `roundId-2`, and
`roundId+Z-2` for single, steady and drain paths. The original active-work
condition remains outside the hooks. Inactive cores therefore leave zero
records and do not gain new handshake participation. Only cores 0 through 3
are sampled, at first, middle and last logical rounds, with duplicate targets
assigned to the first slot. No QK/PV work or warmup/drain branch is skipped.

## Measurement meaning

- Mode 0 performs no timestamps, trace writes or PIPE_ALL drains. It still has
  compiled probe members/branches and fresh zeroed trace allocation. Compare
  this mode with the same-family production baseline to assess that footprint.
- Mode 1 records issue elapsed time across address mapping plus the complete
  QK/PV C++ call. Existing queue backpressure and waits remain in that call, so
  this is not pure scalar instruction time and not completed hardware duration.
  Its completion, empty and pre-drain fields are zero/unmeasured.
- Mode 2 first records a pre-drain and an empty completion bracket, outside the
  measured call. The call begins after those drains. `issue_ticks` ends before
  the final drain; `completion_ticks` records the remaining completion tail;
  `elapsed_ticks` spans the entire completed diagnostic call. The tail is
  neither pure FIX cost nor total matrix cost: much hardware work can finish
  during issue. Pre-drain cost belongs to earlier pending work, not the new call.

QK includes address mapping, scales for INT8, input movement, MAD and score
output conversion/copy. PV includes address mapping, probability/value movement,
MAD, history rescaling and final normalization where present. Descriptor count,
sum of K blocks, logical tile pairs, history-call and normalization-call counts
are recorded after timing. Logical tile pairs are `sum(kx*ky)`, not instruction
counts or FLOPs. History/normalization counts count whole diagonal-helper calls.
The reporting groups different work sizes separately.

Raw system-counter intervals from sparse, drained samples must not be added
or extrapolated into production milliseconds, bottleneck percentages or a
predicted gain. Empty-bracket subtraction does not restore lost overlap. The
paired native API calibration uses identical payloads within each family;
FP16 and INT8 are different algorithms/payloads and their outputs are never
required to match each other exactly. All diagnostic modes must exactly match
their own uninstrumented native output before any performance result is accepted.

## Official references and local checks

[CANN 9.0 GetSystemCycle](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0282.html)
documents that the counter read uses the scalar pipeline and demonstrates
PIPE_ALL before and after a measured operation to observe completion.
[CANN 9.0 PipeBarrier](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html)
lists A2/A3 support and PIPE_ALL for blocking all pipelines. The report retains
raw ticks; these are not Cube clock cycles. No unsupported single-pipeline
completion event is inferred from enum availability.

The 12 focused patch/header tests passed locally: pinned FP16/INT8 inverse
equality, all six whole-call hook scopes, round identities, failure on stale
source, mode/ABI contracts and real header instantiation with separate QK/PV
descriptor types. Host C++ compilation checks syntax and member access only;
it is not a CANN compilation, instruction validation or device synchronization
test. Remote CANN9.0.1 / Ascend910_9382 compilation, output gates and calibration
are required before interpreting the new measurements.
