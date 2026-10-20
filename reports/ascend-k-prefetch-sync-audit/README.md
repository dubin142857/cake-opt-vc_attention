# Laser QK K-prefetch source and synchronization audit

This experiment changes the schedule of K transfers in the INT8 QK path.
The previous scale-cache measurement reduced reported scalar activity while
native latency changed only slightly. It does not prove that FIX alone is the
critical path: exported pipeline counters may overlap. The new hypothesis is
that issuing K for the next tile into the existing alternate L1 slot can hide
part of its GM-to-L1 latency behind current QK work.

The baseline is the separately built **scale-cache** candidate, derived from
MindIE-SD `71098c20ff90838d1fd1ca0a40515c751a3869b0`. The new operator has a
separate source tree, vendor, ACLNN API and bridge. Both retain INT8 QK, INT32
accumulation, DEQF16 scores, original Vector softmax and FP16 PV. ExpCast,
V quantization, smoothing and reordering are not enabled.

## Buffer budget and exact source scope

QK descriptors use `ky=2` in the pinned address mapper and up to 32 K128 tiles.
Each INT8 K tile is 128 x 128 bytes = 16 KiB. The existing K L1 slots are at
byte addresses 128 KiB and 192 KiB, **64 KiB apart**; the 32 KiB separation
elsewhere in the source belongs to L0A/L0B. The candidate adds no buffer and
does not change the L1/L0 budget, tiling, launch dimensions or physical layout.

The only algorithm-source delta is inside `cube1_matmul_op`:

1. After the original Q copy and readiness wait, a nonempty descriptor acquires
   its current K slot, issues K0's original DMA and publishes its readiness.
2. Each iteration issues the readiness wait for its current K slot.
3. When a next K tile exists, it acquires the alternate slot and queues the
   next tile's original DMA/readiness publication **before** the current
   `cube1_base_matmul` call.
4. The current base matmul, slot release, ping-pong toggle and score pointer
   advance remain in their original order.

The first/next DMA uses the same sizes, strides and ND-to-NZ parameters as the
baseline. Its source address changes only from the current logical index to
zero in the prologue or `n_index+1` in lookahead. `n_loop>0` protects the prologue;
`n_index+1<n_loop` protects the tail. No next descriptor is prefetched.

`source_comparison.json` and `CubeForward.h.diff` verify the source scope:

- Reversing the prologue and replacement load block restores the full Cube
  header byte-for-byte.
- The prefix containing PV/diagonal routines is identical.
- `cube1_base_matmul` and everything after it are identical, including MAD,
  gain multiplication/truncation, quantization register setup, FIX barrier,
  DEQF16 output, Run, PresetFlag and ClearFlag.
- Vector and both address mappers are byte-identical.
- Static source-site counts differ because the prologue duplicates a DMA and
  its acquire/publication sites. This is not a claim that raw static call order
  is identical. Dynamically, each real K tile has one acquire, one DMA, one
  ready publication/wait and one release.

## Flag ownership and producer/consumer review

The original K/V L1 free events use MTE1 -> MTE2 IDs 0/1. K data readiness
uses MTE2 -> MTE1 IDs 0/1. Q/P L1 retains IDs 2/3. Opposite directions are
different event channels. The candidate allocates no new event IDs.

| Producer / expected buffer | Consumer / actual buffer | Dependency and reuse rule |
| --- | --- | --- |
| PresetFlag or prior QK/PV MTE1 release for slot `s` | K DMA to `s ? l1_b_pong_ : l1_b_ping_` | Wait MTE1 -> MTE2 event `s` before every write; consume the free token |
| K DMA into current slot `s` | Current QK L1 -> L0B load from that same slot | Publish MTE2 -> MTE1 `s`; issue its wait before the unchanged base matmul |
| Prior MTE1 reader of alternate slot `1-s` | Next K DMA into `1-s` | Acquire the alternate free token before overwrite; current slot and next slot addresses differ |
| Current base matmul's L1 -> L0B read | Future DMA reuse of current slot `s` | Original MTE1 -> MTE2 publication remains after the complete base call; it protects L1 reads, not overall MAD/FIX completion |
| Current Q copy | Both Q128 sub-tile reads in the base call | Original Q acquire, DMA/readiness wait, release and toggle remain untouched |
| QK L0C / quantization register | Existing DEQF16 FIX score output | All MAD/unitFlag/COPY and quantization register ordering remain byte-identical |
| Score GM and Vector probability GM | Existing Cube/Vector consumers | Cross-core notifications, waits, score slots and PV history/output handling remain untouched |

The invariant at the beginning of iteration `i` is that slot `s` contains
exactly K[i] with one readiness token. The alternate slot is either initially
free or has the free token published by the preceding iteration's MTE1 work.
If the next tile exists, that token is consumed to produce exactly K[i+1].
The current readiness is consumed once, current K is read once, and its free
token is restored once. The final iteration does not acquire an unused slot.

At descriptor exit there is no pending K readiness token or acquired K slot;
both free tokens have been restored. The retained slot toggle leaves
`final_slot = initial_slot XOR (kx & 1)`. Consequently, the next descriptor or
unchanged PV routine begins with the same flag ownership and slot parity as
the baseline. A zero-K descriptor performs no K acquire/load/publication and
leaves K parity unchanged, while retaining the original Q path. Upper-right
tail handling still loads the last K tile because the original base method
controls which Q sub-tile consumes it.

These statements describe event dependencies and instruction **issue order**.
They do not assert that scalar issue time equals device completion time or that
the next DMA actually overlaps computation for a particular compiler/shape.
No full-pipeline completion drain was introduced. In particular, restored L1
free tokens do not mean that previous MAD or FIX output has completed; their
unchanged L0 and cross-core dependencies still apply.

## CANNBot audit and official API references

The `ascendc-perf-optimize`, `ascendc-performance-best-practices`,
`ascendc-docs-search` and `ascendc-sync-audit` guidance was consulted. The
performance library's detailed MatMul templates target DAV3510, so they were
not substituted into this A3/DAV2201 kernel. Its FlashAttention and cross-core
optimization sections do not supply a directly usable A3 implementation.
There is no inter-card communication in this operator.

`regenerate.py` constructs baseline/candidate audit trees from local Git objects
and scans all 13 operator C++ source/header files in each tree, including Cube,
Vector, address mapping, kernel entry and local host headers. All four tool
runs have their raw JSON, stderr, command, return code, source hashes and
analyzer hashes preserved.

Both analyzers report zero findings in both trees, but also recognize **zero
events/operations** in the raw Laser intrinsics. They do not prove the modified
DMA schedule safe. The manual dependency table above supplements this missing
graph; the tools emitted no expected_root/actual_root or buffer-lifetime graph
for these raw operations. No red/high finding was filtered out.

[CANN9.0 SetFlag/WaitFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)
documents A3 support, source/destination pipeline directions, the
MTE2_MTE1/MTE1_MTE2 directions and paired event usage. This candidate retains
the pinned raw-intrinsic event API and IDs 0/1 instead of introducing a
TPipe/TQue allocator or mixing those frameworks with the existing raw events.
External CANN headers, compiler lowering and device timing are unavailable
locally. The underlying original event 7 uses in other unchanged routines are
not newly introduced or reinterpreted here.

The per-channel FIX alternative was considered but not implemented.
[CANN9.0 Fixpipe](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0251.html)
supports A3 V220 tensor dequantization with an L1 uint64 parameter tensor.
That route would require additional scale layout/loading and changed output
tiling; it is not an equivalent removal of the scalar DEQF16 barrier. Current
Q128 sub-tiles have distinct Q scales, so simply joining them under one scalar
coefficient would change the algorithm.

## Local tests and remote acceptance

Thirteen focused tests passed. The actual transformed C++ outer QK method is
compiled into a host DLL and executed with an ownership/event/address model.
It covers every `kx` from 0 through 32, both initial K and Q ping-pong states,
`ky=1/2`, upper-right false/true, multiple descriptors and odd/even PV ownership
handoffs. It checks input/output addresses, every DMA parameter, scale values,
exact tile order/count, no descriptor overread, released tokens and final
parity. The model verifies that a next-slot load is issued while the current
slot remains occupied; it does not simulate cycle-level hardware overlap.

Negative controls deliberately remove readiness or release, overwrite the
current slot, load an empty descriptor or an extra tail tile, and break parity.
All are rejected by the executed model. Static inverse tests independently
protect the untouched base matmul/FIX, PV and cross-core paths.

The model does not run CANN or reproduce actual asynchronous queues, so remote
CANN9.0.1 / Ascend910_9382 compilation and NPU checks are mandatory. The same
payload must produce identical complete padded FP32 outputs before timing.
The existing scale storage-offset/in-place-rewrite gates remain enabled because
this candidate still includes the scale-cache helper. Randomized paired native,
prepared and full-API wall timings determine whether this schedule helps.
Optional profiled device counters are separate observations, not additive
phase durations or promised end-to-end savings.
