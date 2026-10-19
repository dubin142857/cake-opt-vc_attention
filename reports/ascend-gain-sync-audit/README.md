# Laser INT8 QK gain precomputation: synchronization evidence

This experiment prepares the two Q128/K128 dequantization bit patterns earlier,
before the K tile's GM-to-L1 transfer. It retains the existing hardware
configuration and synchronization at the QK output site. Its performance is a
hypothesis to test on Ascend910_9382; the local audit is not a speed measurement
or an NPU correctness result.

## Decision and boundaries

The proposed broader move of `set_quant_pre` and `pipe_barrier(PIPE_FIX)` ahead
of the current tile's matrix multiplication was rejected for this candidate.
The available official documentation does not establish the lifetime of a
changing `QUANT_PRE` value relative to a preceding in-flight FIX copy. No claim
is made that the broader move is definitely unsafe; its correctness has not
been established here.

Only the ordinary FP32 multiplication, bit reinterpretation and low-bit mask
are moved. The multiplication order, mask `0xffffe000u`, per-tile coefficients,
`DEQF16`, L0C ping-pong addresses, `unitFlag=3`, and hardware configuration sites
are retained. No event IDs, cache invalidations, barriers, Cube-to-Vector flags,
or Vector instructions are removed or changed.

The intended effect is to make the scale bits available before the QK output
path needs them. This neither eliminates FIX dequantization work nor reduces
the number of QK/PV matrix operations. The compiler may already schedule these
calculations effectively, so an improvement is not assumed.

## Authoritative API evidence

- [CANN 9.0 SetFixpipePreQuantFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0254.html)
  describes the scalar quantization configuration and shows a following FIX
  barrier. A2 and A3 are supported.
- [CANN 9.0 DataCopy complete example](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00130.html)
  explains that the barrier ensures subsequent data movement waits for the
  quantization configuration. This candidate preserves that sequence.
- [CANN 9.1 API pipeline table](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/appendix/api_pipeline_type_summary.md)
  classifies the configuration API as `PIPE_S`, matrix computation as `PIPE_M`,
  and L0C-to-GM output as `PIPE_FIX`. This newer reference is explanatory; it
  does not replace validation with the installed CANN 9.0.1 compiler/runtime.
- [CANN UnitFlag documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_key_features/UnitFlag.md)
  describes fine-grained L0C producer/consumer synchronization. It is not a
  documented guarantee about quantization-register lifetime. UnitFlag settings
  and matrix/copy order remain unchanged.

## Static tool execution

The [CANNBot synchronization-audit skill](https://github.com/hicann/cannbot-skills/tree/dbf4019cae5d757820f03c9cd21e0f0b277e1701/ops/ascendc-sync-audit)
was read with its full-audit workflow and detection-scope reference. Both tools
were run over the complete generated operator directory, including Cube,
Vector, address mapping, constants, entry, probability helper and host files.
The CANN SDK headers (`kernel_operator.h`, `lib/matmul_intf.h`, and registration
headers) are not installed locally and are outside this scan. Templates and
compiler expansion were not evaluated.

The baseline is the scale-reuse source at
`build/mindie_laser_quant_reuse_source/csrc/ops/vc_laser_quant_reuse_attention`.
The candidate is the generated source at
`build/mindie_laser_quant_gain_source/csrc/ops/vc_laser_quant_gain_attention`.
It was generated with `prepare_mindie_laser_quant_gain_source.py --offline`
from local pinned Git objects, with network access disabled by the preparer.
The unfiltered reports and stderr are retained beside this document:

- `baseline_sync_audit.json`
- `baseline_ascendc_flow_analyzer.json`
- `baseline_run.json`
- `candidate_sync_audit.json`
- `candidate_ascendc_flow_analyzer.json`
- `candidate_run.json`
- the corresponding `.stderr.txt` files

Both tools exited successfully for both versions and scanned 13 source files
per version. They
reported **zero findings, zero recognized synchronization events/edges, and
zero recognized data-flow operations**. There are therefore no red/high
candidate messages or details to number or omit from the raw output.

**Zero findings is not synchronization clearance.** These regex tools recognize
Ascend C API forms, whereas this pinned Laser implementation primarily uses
lower-level instructions such as `set_flag`, `wait_flag`,
`ffts_cross_core_sync`, `mad`, and `copy_matrix_cc_to_gm`. The empty event and
operation collections expose this coverage limitation. It would be incorrect
to advertise these scans as proving that the kernel has no races or deadlocks.

No synchronization correction is applied on the basis of these scans. The
candidate's targeted check is source comparison plus remote correctness gates:
all output rows on identical payloads, strongly varying adjacent scale blocks,
scale updates at reused tensor addresses, and the long benchmark shape before
timing. Hardware compilation, NPU execution, latency and counters must be
collected remotely.

## Targeted source comparison

`low_level_call_comparison.json` contains the baseline/candidate source hashes
and complete ordered call expressions with line numbers for a separate lexical
comparison. Comments and whitespace were removed for comparison only; the
analyzer JSON files above remain untouched. The inspected instruction names
are recorded in the JSON.

| Kernel source | Compared low-level calls, each version | Ordered expressions identical |
| --- | ---: | --- |
| `CubeForward.h` | 84 | yes |
| `VectorForward.h` | 91 | yes |
| `vc_laser_probability.h` | 7 | yes |

The other kernel files are unchanged after normalizing the independent
operator name. Only `CubeForward.h` has an implementation change. The lexical
comparison includes scalar-cache invalidation, QK/PV matrix instructions,
L0C-to-GM copies, configuration writes, barriers, flag pairs and cross-core
notifications. It establishes that those call expressions and their source
order were retained; it cannot establish instruction issue timing after
compilation or buffer correctness across all runtime paths.

The relevant dependencies remain:

| Producer | Consumer | Existing synchronization retained |
| --- | --- | --- |
| GM Q/K load to L1 | L1-to-L0 load | `PIPE_MTE2` to `PIPE_MTE1` flag pairs |
| L1 Q/K load to L0 | QK MAD | `PIPE_MTE1` to `PIPE_M` flag pairs |
| QK MAD writes L0C | DEQF16 writes score workspace | MAD/copy `unitFlag=3`; hardware-config-then-FIX-barrier sequence |
| Score workspace output | Vector score processing | `ffts_cross_core_sync(PIPE_FIX, ...)` and `wait_flag_dev(AIC2AIVFLAGID)` |
| Vector probabilities/history output | Cube PV/history work | `ffts_cross_core_sync(PIPE_MTE3, ...)` and `wait_flag_dev(AIV2AICFLAGID)` |

This is a bounded review of the candidate's changes, not a complete proof of
the inherited Laser pipeline. No SYNC-14 candidate was produced by either
tool, so no expected-root/actual-root graph evidence exists to report.

## Reproduction

From the repository root, after generating the source:

```bash
python3 build/cannbot-skills/ops/ascendc-sync-audit/scripts/sync_audit.py build/mindie_laser_quant_reuse_source/csrc/ops/vc_laser_quant_reuse_attention --format json
python3 build/cannbot-skills/ops/ascendc-sync-audit/scripts/ascendc_flow_analyzer.py build/mindie_laser_quant_reuse_source/csrc/ops/vc_laser_quant_reuse_attention --format json
python3 build/cannbot-skills/ops/ascendc-sync-audit/scripts/sync_audit.py build/mindie_laser_quant_gain_source/csrc/ops/vc_laser_quant_gain_attention --format json
python3 build/cannbot-skills/ops/ascendc-sync-audit/scripts/ascendc_flow_analyzer.py build/mindie_laser_quant_gain_source/csrc/ops/vc_laser_quant_gain_attention --format json
```

The JSON outputs are kept unfiltered. Paths inside them reflect the scanned
repository-relative Windows paths; the output is not rewritten to manufacture
recognized low-level events.
