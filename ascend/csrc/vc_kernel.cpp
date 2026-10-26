// Ascend A2/A3 DAV-2201. Original VC adaptation using documented Ascend C APIs.
// Design sources and API versions are recorded in ../SOURCES.json.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "vc_tiling.h"
#include "vc_phase_probe.h"

using namespace AscendC;
using namespace matmul;
using namespace vc910;

template <typename T>
__aicore__ inline void ReadControl(GM_ADDR gm, T& value) {
    auto src = reinterpret_cast<__gm__ uint32_t*>(gm);
    auto dst = reinterpret_cast<uint32_t*>(&value);
    static_assert(sizeof(T) % 4 == 0, "Control structure alignment");
    for (uint32_t i = 0; i < sizeof(T) / 4; ++i) dst[i] = src[i];
}


constexpr auto MM_CONFIG = GetMDLConfig(false, false, 0, false, false, false, true);
using QType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using KType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
using IType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using HType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using FType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// Local Cube implementation: no KFC Matmul client and no REGIST_MATMUL_OBJ.
// Explicit mode-2 flags belong exclusively to this kernel.
template <uint32_t M, bool Profile>
class Cube {
public:
    __aicore__ inline void RunCube(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR ws,
                              GM_ADDR qt, GM_ADDR vt, GM_ADDR trace, const Tiling& t) {
        TPipe pipe;
        TCubeTiling qkt, pvt;
        ReadControl(qt, qkt);
        ReadControl(vt, pvt);
        mmq.SetSubBlockIdx(0);
        mmv.SetSubBlockIdx(0);
        mmq.Init(&qkt, &pipe);
        mmv.Init(&pvt, &pipe);
        GlobalTensor<int8_t> qg, kg;
        GlobalTensor<half> vg, pg;
        GlobalTensor<int32_t> sg;
        GlobalTensor<float> og;
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q));
        kg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(k));
        vg.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v));
        const uint32_t core = GetBlockIdx();
        PhaseProbe<Profile> probe;
        probe.Init(trace, core, 0);
        auto localWs = ws + uint64_t(core) * t.workspace_bytes;
        sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs), M * N);
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + M * N * 4), M * N);
        og.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + M * N * 6), M * t.dim);
        // H6a: O_USED is awaited just before the PV that would overwrite `og`,
        // not at the end of the tile that produced it. The next tile's QK then
        // runs while the Vector halves are still reading `og` and accumulating.
        // Flag counts are unchanged (one wait per O_READY), so the Set/Wait
        // pairing that detsync_lint H7 checks still balances; the outstanding
        // wait is drained after both loops.
        bool o_outstanding = false;
        for (uint64_t task = core; task < t.tasks; task += t.cores) {
            const uint64_t head = task / (t.qpad / M);
            const uint32_t qr = (task % (t.qpad / M)) * M;
            for (uint32_t key = 0; key < t.kpad; key += N) {
                probe.Begin(task, key, t);
                mmq.SetOrgShape(M, N, t.dim, t.dim);
                mmq.SetSingleShape(M, N, t.dim);
                mmq.SetTensorA(qg[head * t.qpad * t.dim + uint64_t(qr) * t.dim], false);
                mmq.SetTensorB(kg[head * t.kpad * t.dim + uint64_t(key) * t.dim], true);
                mmq.template IterateAll<true>(sg, 0);
                mmq.End();
                probe.MarkAll(0); // QK API, including completed GM writeback
                CrossCoreSetFlag<2, PIPE_FIX>(S_READY);
                CrossCoreWaitFlag(P_READY);  // both AIV halves published P
                probe.Mark(1); // exposed wait, not the duration of all Vector work
                if (o_outstanding) {
                    // Deferred from the previous tile. `og` must be free before PV
                    // writes it again. By this point the Vector halves have already
                    // published P for this tile, so the flag is normally already set.
                    CrossCoreWaitFlag(O_USED);
                    o_outstanding = false;
                }

                // A[P] stride=N; B[V] stride=D. Four-argument form is intentional.
                mmv.SetOrgShape(M, t.dim, N, t.dim);
                mmv.SetSingleShape(M, t.dim, N);
                mmv.SetTensorA(pg, false);
                mmv.SetTensorB(vg[head * t.kpad * t.dim + uint64_t(key) * t.dim], false);
                mmv.template IterateAll<true>(og, 0);
                mmv.End();
                probe.MarkAll(2); // PV API, including completed GM writeback
                CrossCoreSetFlag<2, PIPE_FIX>(O_READY);
                o_outstanding = true;  // awaited before the next PV, see H6a note
                probe.Mark(3);
                probe.Finish();
            }
        }
        if (o_outstanding) {
            CrossCoreWaitFlag(O_USED);  // drain the last tile before the kernel ends
        }
    }
private:
    MatmulImpl<QType, KType, IType, IType, MM_CONFIG> mmq;
    MatmulImpl<HType, HType, FType, FType, MM_CONFIG> mmv;
};

template <uint32_t M, bool Profile>
class Vector {
    static constexpr uint32_t ROWS = M / 2;
    static_assert(M == 32 || M == 64, "Each AIV must stay within one Q32 quantization block");
public:
    __aicore__ inline void RunVector(GM_ADDR qs, GM_ADDR ks, GM_ADDR vs, GM_ADDR lut,
                              GM_ADDR output, GM_ADDR ws, GM_ADDR trace, const Tiling& t) {
        TPipe pipe;
        pipe.InitBuffer(rawBuf, ROWS * N * 4);
        pipe.InitBuffer(scoreBuf, ROWS * N * 4);
        pipe.InitBuffer(pBuf, ROWS * N * 2);
        pipe.InitBuffer(accBuf, ROWS * MAX_D * 4);
        pipe.InitBuffer(pvBuf, ROWS * MAX_D * 4);
        pipe.InitBuffer(vsBuf, MAX_D * 4);
        pipe.InitBuffer(stateBuf, 8 * ROWS * 4);
        pipe.InitBuffer(lutBuf, 128 * 4);
        // H8: two per-row broadcast buffers. Brcb expands one value into a full
        // 32B block, so ROWS values occupy ROWS blocks = ROWS * 8 floats each.
        pipe.InitBuffer(brcbBuf, 2 * ROWS * 8 * 4);
        pipe.InitBuffer(tmpBuf, t.reduce_bytes);
        const int32_t taskDoneEvent = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        auto raw = rawBuf.Get<int32_t>();
        auto score = scoreBuf.Get<float>();
        auto ph = pBuf.Get<half>();
        auto acc = accBuf.Get<float>();
        auto pv = pvBuf.Get<float>();
        auto vscale = vsBuf.Get<float>();
        auto states = stateBuf.Get<float>();
        auto m = states[0 * ROWS], l = states[1 * ROWS], alpha = states[2 * ROWS];
        auto rowmax = states[3 * ROWS], mass = states[4 * ROWS];
        auto qscale = states[5 * ROWS], kscale = states[6 * ROWS];
        auto table = lutBuf.Get<float>();
        auto brcb = brcbBuf.Get<float>();
        auto bmax = brcb[0], balpha = brcb[ROWS * 8];
        auto tmp = tmpBuf.Get<uint8_t>();
        static_assert(ROWS % 8 == 0, "Brcb consumes 8 source values per repeat");
        static_assert(N % 64 == 0, "One fp32 repeat covers 64 elements");
        // src1BlkStride = 0 makes all eight blocks of a repeat read the same
        // broadcast block; src1RepStride = 1 advances one block (one row) per
        // repeat. dst/src0 walk whole rows, hence a repeat stride of cols/8.
        const BinaryRepeatParams rowBrc(1, 1, 0, N / 8, N / 8, 1);
        const BinaryRepeatParams accBrc(1, 1, 0, uint8_t(t.dim / 8), uint8_t(t.dim / 8), 1);
        GlobalTensor<float> qsg, ksg, vsg, lg, out;
        qsg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qs));
        ksg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ks));
        vsg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vs));
        lg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lut), 128);
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        const uint32_t core = GetBlockIdx() / GetSubBlockNum();
        const uint32_t sub = GetSubBlockIdx();
        PhaseProbe<Profile> probe;
        probe.Init(trace, core, 1 + sub);
        auto localWs = ws + uint64_t(core) * t.workspace_bytes;
        GlobalTensor<int32_t> sg;
        GlobalTensor<half> pg;
        GlobalTensor<float> og;
        sg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs) + sub * ROWS * N, ROWS * N);
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + M * N * 4) + sub * ROWS * N, ROWS * N);
        og.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + M * N * 6) + sub * ROWS * t.dim, ROWS * t.dim);
        DataCopy(table, lg, 128);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        uint32_t shape[2] = {ROWS, N};
        for (uint64_t task = core; task < t.tasks; task += t.cores) {
            const uint64_t head = task / (t.qpad / M);
            const uint32_t qb = task % (t.qpad / M);
            const uint32_t qr = qb * M + sub * ROWS;
            // Computational tile M64 has two independently quantized Q32 halves.
            DataCopy(qscale, qsg[(head * (t.qpad / Q_SCALE_BLOCK) + qr / Q_SCALE_BLOCK) * 8], 8);
            DataCopy(vscale, vsg[head * t.dim], t.dim);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Duplicate(m, -3.402823466e38f, ROWS);
            Duplicate(l, 0.0f, ROWS);
            Duplicate(acc, 0.0f, ROWS * t.dim);
            PipeBarrier<PIPE_V>();
            for (uint32_t key = 0; key < t.kpad; key += N) {
                probe.Begin(task, key, t);
                CrossCoreWaitFlag(S_READY);
                probe.Mark(0);
                DataCopy(raw, sg, ROWS * N);
                //  一个 tile 覆盖 KSB_PER_TILE 个 K 量化块, 各有自己的 scale, 连续取。
                DataCopy(kscale, ksg[(head * (t.kpad / K_SCALE_BLOCK) + key / K_SCALE_BLOCK) * 8],
                         8 * KSB_PER_TILE);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                float seg_scale[KSB_PER_TILE];
                for (uint32_t sb = 0; sb < KSB_PER_TILE; ++sb)
                    seg_scale[sb] = (qscale.GetValue(0) * kscale.GetValue(sb * 8)) * t.scale;
                probe.Mark(1); // MTE2_S above completed raw scores and scale loads
                Cast(score, raw, RoundMode::CAST_NONE, ROWS * N);
                PipeBarrier<PIPE_V>();
                //  按列半区施加 scale: 一个 repeat 覆盖 64 个 fp32, K_SCALE_BLOCK 是 64 的整数倍,
                //  故量化块边界落在 repeat 边界上, 每块用 K_SCALE_BLOCK/64 次调用,
                //  repeatTimes = ROWS 走完所有行, 行间步长 N/8 个 block。不回到逐行短调用。
                static_assert(K_SCALE_BLOCK % 64 == 0, "量化块边界必须落在 repeat 边界上");
                static_assert(N / 8 <= 255, "行间 repeat 步长超出 uint8");
                {
                    const UnaryRepeatParams rp(1, 1, uint8_t(N / 8), uint8_t(N / 8));
                    for (uint32_t sb = 0; sb < KSB_PER_TILE; ++sb)
                        for (uint32_t off = 0; off < K_SCALE_BLOCK; off += 64)
                            Muls(score[sb * K_SCALE_BLOCK + off], score[sb * K_SCALE_BLOCK + off],
                                 seg_scale[sb], uint64_t(64), ROWS, rp);
                }
                PipeBarrier<PIPE_V>();
                // Tail masks live in UB. No unaligned GM access or all-masked row.
                //  ★key 可能整块越过真实 key 数 (当 K 的填充粒度宽于 N 时)。
                //  t.keys - key 是无符号减法, 越界时会下溢成巨大值而绕过掩码, 故先判。
                const uint32_t remain = key >= t.keys ? 0u : t.keys - key;
                const uint32_t valid = remain < N ? remain : N;
                if (valid < N) {
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    for (uint32_t r = 0; r < ROWS; ++r)
                        for (uint32_t j = valid; j < N; ++j) score.SetValue(r * N + j, -3.402823466e38f);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
                ReduceMax<float, Pattern::Reduce::AR, false>(rowmax, score, tmp, shape, true);
                PipeBarrier<PIPE_V>();
                Max(rowmax, rowmax, m, ROWS);
                PipeBarrier<PIPE_V>();
                Sub(alpha, m, rowmax, ROWS);
                PipeBarrier<PIPE_V>();
                Exp(alpha, alpha, ROWS);  // one history exponential per query row
                // H8: broadcast rowmax per row instead of reading ROWS scalars back
                // and issuing ROWS short Adds. x + (-m) and x - m agree bit for bit
                // on every finite value and on both signed zeros, so the numerical
                // contract is unchanged. The V_S round trip disappears with it.
                PipeBarrier<PIPE_V>();
                Brcb(bmax, rowmax, ROWS / 8, BrcbRepeatParams(1, 8));
                PipeBarrier<PIPE_V>();
                for (uint32_t off = 0; off < N; off += 64)
                    Sub(score[off], score[off], bmax, uint64_t(64), ROWS, rowBrc);
                PipeBarrier<PIPE_V>();
                if (t.expcast) {
                    // Separate FP32 multiply/add is part of the numerical contract.
                    Muls(score, score, EXPCAST_GAIN, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Adds(score, score, 120.0f, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Maxs(score, score, 0.0f, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Mins(score, score, 120.0f, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Cast(raw, score, RoundMode::CAST_RINT, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    // Gather offsets are BYTES. E4M3 decode is exact in FP32/FP16.
                    Muls(raw, raw, int32_t(4), ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Gather(score, table, raw.ReinterpretCast<uint32_t>(), uint32_t(0), ROWS * N);
                } else {
                    Exp(score, score, ROWS * N);
                }
                PipeBarrier<PIPE_V>();
                ReduceSum<float, Pattern::Reduce::AR, false>(mass, score, tmp, shape, true);
                PipeBarrier<PIPE_V>();
                Cast(ph, score, RoundMode::CAST_NONE, ROWS * N);
                probe.template MarkEvent<HardEvent::V_S>(2, pipe);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(pg, ph, ROWS * N);
                CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);
                probe.template MarkEvent<HardEvent::MTE3_S>(3, pipe);
                // History update uses only private UB, while Cube computes PV.
                Mul(l, l, alpha, ROWS);
                PipeBarrier<PIPE_V>();
                Add(l, l, mass, ROWS);
                PipeBarrier<PIPE_V>();
                Adds(m, rowmax, 0.0f, ROWS);
                // H8: same broadcast treatment for the history rescale. Muls by a
                // scalar and Mul by a vector holding that scalar are the same FP32
                // multiply, so the result is bit identical.
                PipeBarrier<PIPE_V>();
                Brcb(balpha, alpha, ROWS / 8, BrcbRepeatParams(1, 8));
                PipeBarrier<PIPE_V>();
                for (uint32_t off = 0; off < t.dim; off += 64)
                    Mul(acc[off], acc[off], balpha, uint64_t(64), ROWS, accBrc);
                probe.template MarkEvent<HardEvent::V_S>(4, pipe);
                CrossCoreWaitFlag(O_READY);
                probe.Mark(5); // only PV wait still exposed after history work
                // ph cannot be overwritten until its MTE3 completes.
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                DataCopy(pv, og, ROWS * t.dim);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                probe.template MarkEvent<HardEvent::MTE2_S>(6, pipe);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, pv, ROWS * t.dim);
                probe.template MarkEvent<HardEvent::V_S>(7, pipe);
                CrossCoreSetFlag<2, PIPE_V>(O_USED);
                probe.Finish();
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float denominator[ROWS];
            for (uint32_t r = 0; r < ROWS; ++r) denominator[r] = l.GetValue(r);
            for (uint32_t r = 0; r < ROWS; ++r)
                Muls(acc[r * t.dim], acc[r * t.dim], 1.0f / denominator[r], t.dim);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < ROWS; ++r)
                Mul(acc[r * t.dim], acc[r * t.dim], vscale, t.dim);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(out[head * t.qpad * t.dim + uint64_t(qr) * t.dim], acc, ROWS * t.dim);
            // Drain this task before Scalar can issue the next task's metadata
            // DMA. MTE3_V only gates Vector: MTE2 could overwrite vsBuf while
            // the epilogue above still reads the previous head's V scale.
            // V -> MTE3 -> S orders both the next MTE2 load and Vector reset.
            SetFlag<HardEvent::MTE3_S>(taskDoneEvent);
            WaitFlag<HardEvent::MTE3_S>(taskDoneEvent);
        }
    }
private:
    TBuf<TPosition::VECCALC> rawBuf, scoreBuf, pBuf, accBuf, pvBuf;
    TBuf<TPosition::VECCALC> vsBuf, stateBuf, lutBuf, tmpBuf, brcbBuf;
};

template <uint32_t TileM, bool Profile>
__aicore__ inline void RunAttention(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    // Legacy ascendc.cmake emits MIX for __aicore__ kernels using both pipelines.
    // No KERNEL_TASK_TYPE_DEFAULT (arch35 only), no KFC workspace parameter.
    Tiling t;
    ReadControl(tiling, t);
    if ASCEND_IS_AIC {
        Cube<TileM, Profile> cube;
        cube.RunCube(q, k, v, ws, qt, vt, trace, t);
    }
    if ASCEND_IS_AIV {
        Vector<TileM, Profile> vector;
        vector.RunVector(qs, ks, vs, lut, output, ws, trace, t);
    }
}

extern "C" __global__ __aicore__ void vc910_forward(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<32, false>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

extern "C" __global__ __aicore__ void vc910_forward_m64(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<64, false>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

extern "C" __global__ __aicore__ void vc910_profile(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<32, true>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

extern "C" __global__ __aicore__ void vc910_profile_m64(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<64, true>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}
