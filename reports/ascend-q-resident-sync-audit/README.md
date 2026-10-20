# Laser QK Q-residency source and synchronization audit

The candidate retains both Q128 INT8 sub-tiles in their existing L0A slots
while processing the K tiles of one descriptor. It starts from the separate
**scale-cache** baseline, derived from MindIE-SD commit
`71098c20ff90838d1fd1ca0a40515c751a3869b0`. The K-prefetch experiment is not
stacked into this candidate: its latest full-API measurement did not improve.

Both variants retain INT8 QK, INT32 accumulation, scalar DEQF16 FP16 scores,
the original Vector softmax and FP16 PV. There is no ExpCast, V quantization,
smoothing or reordering in this comparison. Separate operator, vendor,
ACLNN API and bridge identities isolate the candidate.

## Work removed and unchanged arithmetic

The pinned address mapper uses `ky=2`, representing two Q128 sub-tiles. Each
K128 iteration previously loaded both Q sub-tiles from L1 to L0A again.
Each Q sub-tile requires eight `load_cbuf_to_ca` instructions, each moving
four 512-byte INT8 fractals: 16 KiB per sub-tile.

For a dense descriptor with 32 K128 tiles, this changes Q loading from
512 load instructions / 1 MiB of L1-to-L0A traffic to 16 instructions /
32 KiB. This is a 96.875% reduction **in this Q load traffic**, not a kernel
speedup estimate. K loading, matrix products, score output and softmax/PV
still perform their previous work.

The patch changes exactly four source locations:

1. Add a final `bool load_q` parameter to the declaration.
2. Pass `n_index == 0 || ky != 2` in the QK outer loop.
3. Add that parameter to the base method definition.
4. Guard only the original eight-instruction Q load loop with
   `!l0_skip_flag && load_q`.

Every existing wait, set, toggle, MAD, gain multiplication, coefficient bit
mask, quantization-register operation, FIX barrier and DEQF16 output remains
in the same order with the same operands. Removing `PIPE_FIX` is **not**
part of this candidate. Its scalar coefficient changes between Q/K
sub-tiles, and removing an ordering operation without proving the register
and output consumers safe would not be an equivalent optimization.

`source_comparison.json` and `CubeForward.h.diff` record an inverse transform
that restores the entire baseline Cube header byte-for-byte. The complete
raw event/MAD/FIX/quantization-register call sequences are identical. Vector
and both address mapper headers are byte-identical. The prefix containing
PV/diagonal routines is identical after restoring the declaration; Run,
PresetFlag, ClearFlag and everything following them are byte-identical.

## L0A identity and lifetime proof

The two existing L0A slots begin at byte offsets **0 and 32 KiB**. A
128x128 INT8 Q sub-tile occupies only 16 KiB. No new allocation, offset,
tiling size, register limit, event ID or shared-memory region is introduced.

Let `s` be the L0A ping-pong flag at the first K tile of a descriptor.
The first Q128 row loads into slot `s`, the second into slot `1-s`.
The original loop toggles the flag after **each** Q row, including skipped
rows. Therefore a complete `ky=2` K iteration toggles twice and restores `s`.
Every subsequent K iteration uses the same Q-row-to-L0A mapping.

MAD reads L0A and L0B and writes L0C. FIX consumes L0C and writes the score
workspace. K uses L0B, which is a different physical memory. There are no
other L0A writers between QK base calls inside this outer descriptor.
Consequently, suppressing later writes leaves precisely the originally
loaded Q bytes available to the same MAD operands.

The lifetime is intentionally confined to one descriptor:

- A new descriptor restarts `n_index` at zero and reloads both rows, regardless
  of pointer equality or surviving SRAM contents.
- PV and diagonal operations may overwrite these L0A slots between calls.
  The candidate makes no residency claim across those operations.
- For `ky!=2`, every K iteration reloads Q. In particular, `ky=1` changes
  slot parity once per K tile and is not eligible for the two-slot invariant.
- `kx=0` executes no base call or Q-to-L0A load; the unchanged Q-to-L1 path
  still releases its original token.
- Upper-right skipping applies only to the first Q row of the last K tile.
  If `kx=1`, that row is never loaded and has no MAD/FIX consumer. If `kx>1`,
  it was loaded on the first K tile and its final skipped use remains skipped.
  The second row is loaded on the first iteration in both cases.

## Producer/consumer and event ownership

| Producer / buffer | Consumer / buffer | Dependency and lifetime |
| --- | --- | --- |
| Q GM-to-L1 copy | First eligible Q L1-to-L0A loads | Original MTE2-to-MTE1 ready publication/wait retained |
| Prior MAD reading L0A slot `s` | Possible Q load writing slot `s` | Original M-to-MTE1 free wait retained before every row, including reuse |
| First Q load into L0A slot `s` | MAD reading Q from slot `s` | Original MTE1-to-M ready publication/wait retained |
| No new Q write on later eligible K tiles | MAD reading the same L0A slot | The prior Q contents persist; the same ready publication/wait still executes |
| Current MAD read of slot `s` | A later Q overwrite or PV handoff | Original M-to-MTE1 release and slot toggle retained |
| K GM/L1/L0B movement | Current QK MAD | All K copies, events and address arithmetic unchanged |
| MAD L0C result | DEQF16 FIX output | MAD/COPY unitFlag and scalar quantization-register/FIX ordering unchanged |
| Score/probability workspaces | Vector/PV consumers | All cross-core synchronization, workspace slots and timing of notifications unchanged |

L0A uses the original M-to-MTE1 and MTE1-to-M event IDs 0/1; L0B retains
IDs 2/3 in those directions. L1 Q/P and K/V ownership is also unchanged.
There is no outstanding extra ready token, producer or buffer acquisition
at descriptor or function exit. After every row the same free token is
restored; final L0A/L0C parity is `initial XOR ((kx*ky)&1)`, and L0B parity
is `initial XOR (kx&1)`. PresetFlag/ClearFlag require no modification.

Keeping a readiness event even when no new copy was needed preserves the
original event protocol. It does not imply a new transfer occurred. The
manual review distinguishes instruction issue/order from hardware completion;
no timing overlap or pipeline utilization is inferred from this proof.

## CANNBot workflow and official references

The local `ascendc-docs-search` and `ascendc-sync-audit` workflows were used,
with the previously read performance guidance. The local devkit API
documentation tree is unavailable, so public official documentation was
consulted for the existing transfer and computation semantics.

[CANN9.0 Load2D](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00169.html)
lists A3 support and the A1-to-A2 and B1-to-B2 transfer paths.
[Official Mmad reference](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0249.html)
describes A/B as inputs and C as the matrix-add output; this older reference
is used only for that established operand role, not to infer A3 availability.
The actual A3 INT8 instruction and all its parameters are already present
in the pinned, remotely exercised baseline and are unchanged here.
[CANN9.0 SetFlag/WaitFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)
documents directional pipeline events and paired use. The candidate
retains the baseline raw intrinsic API rather than mixing a new TPipe/TQue
allocator into its fixed event ownership.

`regenerate.py` scans all 13 local C++ source/header files in each complete
baseline and candidate operator tree. Both `sync_audit.py` and
`ascendc_flow_analyzer.py` raw JSON, stderr, command, return code, analyzer
hash and source hashes are saved. All four scans report zero findings, but
recognize **zero raw intrinsic events/operations**. The analyzers therefore
do not prove the unchanged event protocol or extended data lifetime safe.
They emit no expected_root/actual_root or lifetime graph for these raw
instructions. The manual table and executable model supply that review;
no red/high finding was filtered or suppressed.

## Local validation and remote acceptance

Sixteen focused local tests passed. They compile and execute the actual
baseline and transformed **outer and base C++ methods**, replacing only
hardware intrinsics with a host ownership/address model. The model checks:

- Every `kx` from 0 through 32, `ky=1/2`, upper-right false/true, and every
  combination of initial L0A/L0B/L0C ping-pong flags.
- Q fragment source/destination offsets and all Load2D parameters; all
  eight fragments of each MAD's Q operand must have the expected descriptor
  and row identity.
- K tile identity, MAD dimensions/flags, L0C identity, scale coefficient
  bit pattern, score output address and FIX parameters.
- Exact baseline/candidate sequence fingerprints of every MAD operand
  identity and every event/FIX barrier, plus the expected Q load count.
- Multiple descriptors, zero and single-K tails, and odd/even intervening
  PV-like slot changes. Logical Q contents are poisoned at every new
  descriptor and at PV handoff so stale reuse cannot pass accidentally.
- No unreturned free token or pending readiness at handoff/exit.

Negative controls break the first load, incorrectly reuse at `ky=1`, select
the wrong L0A slot or Q fragment, remove readiness/release, or break slot
parity. Every mutation fails the executed model. Static tests reject
unsupported layout/tiling drift, stale anchors, duplicate application and
stacking on K-prefetch.

This model does not simulate asynchronous CANN queues or run an NPU.
Remote CANN9.0.1 / Ascend910_9382 compilation and exact all-output
same-payload comparisons remain mandatory before timing. The broader
scale-offset/in-place-rewrite gates remain enabled because the candidate
still includes the scale cache. Randomized paired native/prepared/full
wall timings decide whether reduced Q traffic actually improves latency;
profiled device counters remain separate, potentially overlapping evidence.
