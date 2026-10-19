# INT8 Laser scale cache-line candidate audit

This candidate follows the user-provided Cube diagnostic: sampled INT8 QK
intervals were longer than FP16, while PV samples were similar. That observation
does not isolate DCCI, FIX, DMA or MAD as the cause. The tested hypothesis is
that invalidating the scalar cache for every scale value adds avoidable QK
overhead. No speedup is claimed before remote measurements.

The baseline is the existing INT8 scale-reuse Laser from pinned MindIE-SD
`71098c20ff90838d1fd1ca0a40515c751a3869b0`. The candidate remains INT8 QK/INT32
accumulation, DEQF16 score output, original Vector softmax and FP16 PV. It does
not enable ExpCast, V quantization, V smoothing or reordering. Separate source,
vendor, ACLNN and bridge identities preserve the existing baseline.

## Exact delta and lifetime contract

Each Cube instance gains two uint64 tags: the most recently visited physical
64-byte Q-scale line and K-scale line. `SetQuantScales` resets both tags before
every `Run`, even when the allocator returns the same addresses. Before a scale
read, the original pointer is converted to an integer and masked only for tag
comparison. First access or a line transition executes the original DCCI, then
the original `GetValue(0)`. A line revisit invalidates again. Interleaved Q/K
reads retain separate tags; no persistent array of visited lines is introduced.

The DCCI tensor still starts at the original in-bounds float pointer. The
aligned integer tag is never cast back to a pointer or dereferenced. Therefore
an input scale view beginning at offset 1 or 15 is not rounded down into memory
preceding that view. A physical cache line may of course contain neighboring
allocations; that is the documented behavior of the unchanged DCCI operation.

Scales must be read-only until device kernel completion. Their producing
quantizer finishes before attention through the existing same-stream ordering.
Rewriting the same scale tensor on that stream after the queued native call is
allowed: the next invocation resets tags and invalidates before reading. An
unsynchronized concurrent writer during attention is outside both the baseline
and candidate contract. Cache eviction during an invocation is safe under this
contract because a subsequent read reloads the same immutable GM values.

## CANNBot evidence and limits

`regenerate.py` reconstructs the baseline and candidate trees from local pinned
Git objects under `build/ascend_scale_cache_sync_audit`. It scans all 13 C++
source/header files from each operator, including Cube, Vector, both address
mappers, kernel entry, probability helper and host headers. Raw JSON, stderr,
commands, return codes, source hashes and analyzer hashes are retained for all
four runs. No tool output is filtered.

Both `sync_audit.py` and `ascendc_flow_analyzer.py` returned zero findings for
both trees. They also recognized **zero events/operations**: these regex tools
do not understand the raw Laser MAD/DMA/event/cross-core intrinsics or prove
scalar-cache coherence. This result is not a proof of synchronization safety.
External CANN headers, macro expansion, CCE scheduling and device execution are
unavailable locally. The manual dependency review and remote gates below remain
necessary. There are no script red/high findings to suppress or rewrite.

`source_comparison.json` and `CubeForward.h.diff` retain independent evidence:

- Reversing the helper/tag/read-site edits restores the entire Cube header
  byte-for-byte. The diff includes exact line numbers and zero context.
- Vector and both address mappers remain byte-identical.
- The raw Cube synchronization/MAD/FIX-copy sequence is identical (78 calls);
  the corresponding Vector sequence is identical (91 calls).
- The source-level number and placement of scale reads are unchanged; only
  whether DCCI runs before a read changes. Static DCCI-site counts cannot tell
  the number of dynamically executed invalidations or predict speedup.

## Manual producer, consumer and synchronization review

The table supplements the analyzer, which emitted no buffer dependency graph
or `expected_root`/`actual_root` fields for these raw operations.

| Producer / expected root | Actual consumer root | Existing ordering and candidate check |
| --- | --- | --- |
| Prior quantizer writes Q/K scale tensors | `q_scales_ + q_scale_base` and `k_scales_ + k_scale_index` | Same-stream device ordering remains unchanged; `SetQuantScales` resets both tags before `Run`; first read of each current line executes DCCI before GetValue |
| Valid scalar-cache Q or K line | Subsequent GetValue in that same physical line | Separate absolute-address tags, read-only lifetime; different Q/K allocation addresses cannot match merely because their scale indices match |
| Q/K scale scalars | Existing FP32 `q_scale * k_scale` | Same GetValue sites, multiplication, low-13-bit truncation and `set_quant_pre` remain unchanged |
| Quantized Q/K GM payloads | MTE2/MTE1 inputs and INT32 MAD L0C | Input addresses, ping-pong state, event IDs and raw call order remain unchanged |
| QK L0C and quantization scalar register | Existing DEQF16 FIX output to score GM | Existing `set_quant_pre`, `pipe_barrier(PIPE_FIX)`, unitFlag and COPY sequence remains unchanged |
| Completed score GM | Vector score load and softmax | Existing `ffts_cross_core_sync(PIPE_FIX, ...)` and Vector wait remain unchanged |
| Vector P and diagonal GM | PV/history/normalization consumers | No PV function, address, event, buffer lifetime or cross-core participation changes |

No new event, fence or completion barrier is added. In particular, this change
does not move DCCI past its scalar read, remove the first invalidation, cache
values across calls or alter any unitFlag MAD/COPY pairing. It does not change
the preceding quantizer's implementation or how stream dependencies are set.

## API verification and tests

The local CANNBot docs-search index was consulted; no local ASC_DEVKIT_DIR or
matching full API documentation is installed. Official CANN 9.0 pages were
then checked for the actual A3 target:

- [DataCacheCleanAndInvalid](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0177.html)
  supports A3, the three-template-parameter GlobalTensor overload, and
  SINGLE_CACHE_LINE on the physical line containing a nonaligned input address.
  Its nonaligned example requires a second call only when accessing a second
  physical line. The candidate continues passing the actual scalar pointer.
- [GetValue](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00026.html)
  requires cache maintenance when external writes can make its GM value stale.
  Invocation-local reset and first-line DCCI retain that requirement after each
  producing quantizer; within-call reuse relies on the read-only contract.

Eight focused local tests passed, including strict inverse/source-anchor
checks and execution of the actual new C++ helper in a compiled host DLL. The
host model first creates stale cache data, then checks physical offsets 1/15/16,
interleaved distinct Q/K addresses, line revisits, same-address in-place rewrite
followed by invocation reset, and cache eviction. It rejects pointer reads or
DCCI tensor addresses outside the scale views. This model validates ordinary
C++ control/address logic against the stated cache model, not CCE address spaces
or hardware semantics.

Remote CANN9.0.1 / Ascend910_9382 compilation and NPU validation remain required.
The benchmark must compare all native output elements against scale-reuse using
identical prepared tensors, including in-place scale rewrites at unchanged
addresses before subsequent queued calls. Native, prepared and fresh full-API
timings must be randomized paired measurements without profiler instrumentation.
Do not reinterpret this diagnostic's old Cube sample ticks as a promised
production gain or use separate scope timings as additive phase percentages.
