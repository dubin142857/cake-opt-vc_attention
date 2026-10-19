# Laser score prefetch: source and synchronization audit

This candidate derives from the measured scale-reuse version, not the
gain-precompute experiment. User-supplied gain results showed no benefit:
paired native/prepared/full savings were -0.0221/-0.0501/-0.0545 ms. The new
hypothesis is to overlap AIV score movement with max/exp/sum computation.
It is not a measured attribution of the production bottleneck.

## Source boundary

Pinned MindIE-SD commit: `71098c20ff90838d1fd1ca0a40515c751a3869b0`.
The independent operator is `VcLaserQuantPrefetchAttention`. Its preparation
uses local Git objects only. `source_comparison.json` records the input and
prepared hashes for all 15 prepared files. After operator/marker identity
normalization, only `VectorForward.h` differs. Reversing the scheduling patch
restores that file byte-for-byte. Cube QK/PV, DEQF16 gain configuration, tiling,
cross-core notifications, and all original arithmetic remain unchanged.

The complete line-numbered change is retained in `vector_prefetch.diff`:

- At the former declaration around line 211: add a score-only DMA helper and
  a `score_preloaded` argument to the max helper.
- At the original score DMA around line 995: bypass this DMA and its ready
  fence only when the driver has already loaded/waited for that score slot.
- At the section loop around line 1523: prime batch zero, wait for the current
  batch, then enqueue the next batch before current arithmetic.
- After exp/rowsum around line 1567: add a Vector-read-complete release token.
- At section drain around line 1621: consume the final two Vector tokens in
  addition to the existing MTE3 tokens.

## Producer, consumer and lifetime review

The guard is dense/no triangle/no sparse mask, `rowPerTime == 4`, 64 Vector
rows per section, and 1 through 32 key blocks. Other paths use the original
serialized code. `windowLen` tail padding remains in the original Vector
computation, so K37741 is eligible without prefetching masks.

| Buffer or dependency | Producer | Consumer / last use | Completion before reuse |
| --- | --- | --- | --- |
| Score slot `s`, UB GM-load destination | MTE2 copies the four rows for slot `s` | Vector max, exp and row sum | MTE2→V `s` is set after load and consumed before processing |
| Score slot `s`, probability values | Vector exp | MTE3 P writeback reads this slot | Existing V→MTE3 and MTE3→MTE2 `s` retained |
| Score slot `s`, post-writeback read | Vector row sum reads the same slot after the P DMA is issued | Later MTE2 would overwrite the slot | New V→MTE2 `s` set after exp/rowsum, consumed before overwrite |
| GM next four-row score source | Cube completes QK and original cross-core signal | MTE2 next-batch copy | Original Cube→Vector handshake unchanged; different row range from current P store |
| Shared max/sum scratch | Current Vector batch | Subsequent Vector batch | Original Vector instruction order and barriers unchanged |
| Mask buffers | Original serialized mask DMA | Original Vector masking | Prefetch guard excludes triangle/sparse mask paths; mask DMA is never moved |

`s` is always derived from `ping_pong_flag` (current slot) or `1 -
ping_pong_flag` (next slot), for both address and event ID. The two score slots
start at UB byte offsets 66048 and 98816, each 32768 bytes. The maximum four-row
transfer is `32 * 4 * 128 * 2 = 32768` bytes. Reduction scratch occupies a
separate earlier UB region. P writeback uses the same strided four-row GM
region as the current score batch, whereas the prefetched batch advances by
four rows; the same slot's offset advances by eight rows after consumption.

The existing mask-slot bases are only 1024 bytes apart; mask operations can
extend beyond that interval. They must not be treated as independently
prefetchable buffers. This candidate deliberately excludes those paths.

For each direction, two initial free tokens are produced. Every loaded batch
consumes its slot's MTE3-free and V-free tokens; each processed batch returns
both. Each DMA produces exactly one MTE2→V ready token, consumed by that
batch. The last iteration does not load past the section. Final drain consumes
both remaining free tokens in both directions. The added V→MTE2 direction
was unused in the pinned Vector source; only numeric IDs 0/1 are used. This
operator uses raw fixed-address/intrinsic scheduling, not TPipe allocation.

The V-read dependency is essential: the P DMA is issued *before* row sum.
Waiting for MTE3 alone does not establish that the later Vector reads ended.
The additional V→MTE2 fence addresses this newly exposed overwrite hazard.

## Official API references

[CANN 9.0 SetFlag/WaitFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)
describes matched producer/consumer events, including V→MTE2 and MTE2→V,
and distinguishes manually managed events from TPipe/TQue allocation.
The added events use IDs 0/1, avoiding the documented static-tensor
reservations on IDs 6/7. Existing upstream event choices are not changed.

[CANN 9.0 static tensor programming](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00019.html)
explains the backward V→MTE2 dependency that prevents overwriting a buffer
while Vector still computes from it. These references support the scheduling
design; the actual low-level code still needs the remote CANN 9.0.1 compiler
and Ascend910_9382 execution test.

## Static tool coverage and validation limits

The CANNBot `ascendc-sync-audit` full-audit workflow was used on all 13 C++
source/header files in each generated operator tree. Raw, unfiltered
`sync_audit` and `ascendc_flow_analyzer` JSON, stderr and commands are retained
here for baseline and candidate. Both tools exited 0 with zero findings,
but they also recognized zero synchronization events and zero data-flow
operations. These regex tools do not cover the raw Laser intrinsics used here.
**Their empty findings do not prove synchronization correctness.** There were
no script red/high candidates to omit. Missing local CANN SDK headers and
compiler expansion further limit this static scan.

The review above supplies the manual buffer/index/lifetime analysis. The 123
selected local tests passed in 82.33 seconds; their selection and environment
are recorded in `local_verification.json`. Host tests check preparation
isolation, inverse source changes, event scheduling,
and regression-gate behavior; they do not simulate NPU arithmetic or physical
pipeline timings. The mandatory remote checks compare all padded FP32 native
outputs exactly, including FP16/BF16, aligned/tail/section boundaries, repeated
payloads and in-place changes. Default long-shape validation also compares
all native output rows before timing. No relaxed numerical threshold is used
to accept this scheduling change.

The runner records same-process randomized paired native/prepared/full API
wall times. Profiler windows are collected separately after those results are
saved. Activity counters overlap and are not QK/softmax/PV phase percentages.
There is no local NPU compilation, correctness or speed claim; the candidate
does not replace the scale-reuse baseline until remote results support it.
