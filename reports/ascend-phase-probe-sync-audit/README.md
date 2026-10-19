# Laser phase diagnostic: synchronization and measurement audit

This is an independent diagnostic of the measured INT8 scale-reuse Laser,
not a faster candidate and not the score-prefetch experiment. It preserves
native INT8 QK, FIX DEQF16 output, original Vector exp and FP16 PV. The user
authorized proceeding with phase localization after score prefetch did not
improve native latency.

The pinned MindIE-SD revision is
`71098c20ff90838d1fd1ca0a40515c751a3869b0`. Generated audit copies are under
`build/ascend_phase_probe_sync_audit/{baseline,candidate}`; the original
checkouts are not modified. The independently named operator is
`VcLaserQuantProbeAttention`. The preparer uses local Git objects only.

## Evidence and static-tool limits

The CANNBot `ascendc-sync-audit` full-audit workflow was applied to all 13
baseline and 14 diagnostic C++ source/header files. This includes Cube,
Vector, both address mappers, probability helper, kernel entry, local
constants, tiling, OpDef, proto and source-marker headers. Commands, tool
hashes, exit codes, source hashes, raw JSON and stderr are retained here.
The tools ran with Python UTF-8 mode so their raw JSON remains parseable
despite the Chinese workspace path.

Both `sync_audit.py` and `ascendc_flow_analyzer.py` exited 0 for both trees.
Both returned zero findings; the sync tool recognized zero events and the
flow tool recognized zero operations. There are **no red/high script
candidates to omit**. These regex analyzers do not model Laser's raw
`set_flag`, `wait_flag`, `wait_flag_dev`, `ffts_cross_core_sync`, DMA and MAD
intrinsics. Empty findings therefore do not establish synchronization
correctness. CANN SDK headers, macro expansion, instruction scheduling and
template instantiation are not available in this local Windows environment.

`low_level_call_comparison.json` independently compares raw synchronization
call sequences: Cube 72/72, Vector 91/91 and probability helper 7/7, exactly
unchanged. Both address mappers are byte-identical. The diagnostic patch's
inverse test restores every original Cube/Vector byte. The new adjacent
V-to-S pair lives only in `vc_laser_phase_probe.h`; it is not hidden in the
original-call comparison. `CubeForward.h.diff` and `VectorForward.h.diff`
retain complete line-numbered changes.

## Added dependency and buffer lifetime review

| Expected producer root | Actual producer/consumer root | Dependency / buffer use | Completion coverage |
| --- | --- | --- | --- |
| Original QK score GM | Original `gm_S` / `attn_score_gm`, unchanged address mapping | Cube FIX writes scores; Vector later reads them | Original FIX cross-core signal and Vector `wait_flag_dev` remain unchanged; the wait is timestamped without a new Cube drain |
| Original probability GM | Same original score region reused for P | Vector exp followed by MTE3 P store; Cube PV reads it | Original V-to-MTE3 event, MTE3 cross-core signal and Cube wait remain unchanged |
| Current score UB slot | `pp_buf_for_attn_score_fp16[ping_pong_flag]`, unchanged | MTE2 loads scores; Vector max/exp/sum reads and updates that slot | Original MTE3-to-MTE2 slot-release and MTE2-to-V ready events retained |
| Vector phase completion | Original Vector queue, no new arithmetic buffer | Scalar system-counter read observes completion of the sampled Vector work | New adjacent V-to-S EVENT_ID0 set/wait at each sampled boundary |
| Max/history scalar consumer | Original `subMaxValueResultVectorUbAddr`, unchanged | Original V-to-S events expose history values for scalar diagonal construction | Original event IDs 1 through 6 remain untouched, including outstanding flags |
| Diagnostic trace slot | `((core*3+role)*8+slot)*24`, same expression for every store | One role writes one private 192-byte record, host later reads it | Stores occur after timing; all three 64-byte cache lines are cleaned out; host synchronizes before copying/decoding |

The new Vector event is numeric **ID 0 in the V-to-S direction only**.
Within the max, exp and sum helpers, original V-to-S flags use IDs 1 through
6. The existing V-to-S ID 0 pair occurs later in `Run`'s tail normalization,
after the sampled helper has returned; that pair is adjacent and consumed
before the next section. Each new pair is also adjacent and consumed before
another is set. It cannot consume the original pending V-to-S 1..6 tokens.
There is no new PIPE_ALL, Cube event, cross-core notification or buffer
overwrite. `VectorComplete` uses the same `__DAV_C220_VEC__` guard as the
pinned Vector class; the entry's Vector branch uses that macro as well.
Actual CANN 9.0.1/A3 compilation and execution remain remote validation.

The original Cube probability wait occurs at three source sites: the
single-round path, steady-state loop and final two-round drain. Trace logical
rounds are respectively `roundId`, `roundId-2`, and `roundId+Z-2`: the P tile
being awaited, not the lookahead QK tile. Vector waits use their current QK
round. All original active/inactive-core handshake participation remains.

The trace is a separate INT64 output `[4,3,8,24]`, zeroed for every call.
Original stats and attention output allocation/ordering are retained. It
samples only the first four physical core indices, Cube role 0 and AIV roles
1/2. Distinct 192-byte slots do not share 64-byte cache lines. The wait slots
choose logical rounds `[0,1,floor(total/2),total-1]`, with duplicate targets
assigned to the lowest slot. Phase slots 4..7 record the first actual four-row
batch of each `(head,tail)` section class. No reduction scratch is reused for
trace storage and no trace stores occur inside the measured interval.
Every written record overwrites all 24 words before cleaning all three cache
lines. No partially updated line can flush old zeros over another role's
record. The bridge explicitly checks that the trace base is 64-byte aligned,
in addition to the slot/role/core strides of 192/1536/4608 bytes. Previously written records are
cleaned out before a later call can reuse their allocation. Trace stores and
cache operations outside the measured interval can still delay subsequent
handoffs, so recording remains part of the calibrated perturbation.

## Meaning of the measurements

- Mode 0: no timestamps, trace writes or additional V-to-S fences. It still
  compiles the probe class, runtime branches and a 192-byte local record
  array. Additional register/local-memory pressure or spills are possible;
  they have not been measured locally. Its native API also allocates and
  zeros the trace. Comparison with production scale reuse measures this
  combined diagnostic-framework perturbation.
- Mode 1: elapsed system-counter ticks around sampled original cross-core
  waits. This does not measure the other core's full work or total queue
  occupancy. Trace-write overhead after a wait may affect subsequent work.
- Mode 2: additionally forces sampled Vector work to complete before scalar
  timestamps. These fences alter overlap. Compare against both mode 0 and
  mode 1 before interpreting any phase values.

`score_load_ticks` includes scalar address/branch work, score/mask DMA and
completion. Although the original MTE3-to-MTE2 wait is issued before
PhaseBegin, it is a queue dependency, not scalar completion: any remaining
slot-readiness delay can still be exposed in this load window. It is not a
pure MTE2 duration.

`max_history_ticks` includes score scaling, tail masking, padding, max
reduction, old-max combination, the small historical rescaling exp and
max-value broadcast. `subtract_ticks` covers score minus max. `exp_ticks`
covers the bulk exp branch (the driver disables ExpCast). `sum_ticks` covers
the reduction after its own completed Vector start boundary; probability
writeback may run concurrently. The V-to-MTE3 dependency and DMA issue are
outside the sum interval. An adjacent empty completed window records fence
and timestamp overhead; subtracting it does not reconstruct production cost.

All values are raw GetSystemCycle ticks, **not** AIC/AIV core cycles, calibrated
nanoseconds or production percentages. Phase windows include scalar issue
time and instrumentation. Samples from different cores/classes/rounds are
not additive. Cross-core waiting and other core work overlap. The trace API
and production API wall times are calibrated separately with identical
payloads and randomized paired trials; profiler is not active.

## Verification boundary

Ten focused local tests pass, including strict inverse byte checks and host
C++ semantic compilation of the actual header for Cube and Vector guards.
This checks ordinary C++ names/signatures, not the CCE compiler or device
intrinsic availability. The remote runner must pass exact all-padded-output
comparison for each mode on FP16/BF16 structured shapes and the full long
shape, plus strict trace decoding, before any timing result is accepted.
No local NPU correctness, resource, deadlock-freedom or performance result is
claimed. Existing source/provider/SOC provenance gates remain required.

The new event follows the completion semantics documented in
[CANN 9.0 SetFlag/WaitFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html).
This reference supports the boundary design; it does not substitute for
remote validation of the pinned low-level intrinsic code.

[CANN 9.0 GetSystemCycle](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0282.html)
documents A3 support and identifies the API as a scalar-pipeline system
counter. It is not a Vector-only intrinsic. Its documented frequency is not
used to extrapolate sampled timings to production phases in this report.
[CANN 9.0 DataCacheCleanAndInvalid](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0177.html)
documents A3 support, 64-byte cache lines and cleaning scalar uint64 stores
out to GM, including the danger of different cores sharing one cache line.
