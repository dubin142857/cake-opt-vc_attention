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
//  H16: PV 的 B (即 V) 改由调用方自管 L1。TSCM 只接受 NZ ——
//  CANN 安装包里 TSCM 的全部内建用例均为 CubeFormat::NZ。
using VType = MatmulType<TPosition::TSCM, CubeFormat::NZ, half>;
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
        //  H16: V 的两个 L1 槽。B 位于 TSCM 时 matmul 自己不再为 B 分配 L1
        //  (GetCopyCubeInType 走 FROM_L1 分支, 其 Init/Reset 为空体), 故这 128 KB 是**搬家**不是新增。
        //  每槽 N*MAX_D*2 = 65536 B; dim=64 时只用前半, 不另设尺寸免得两条路径。
        TSCM<TPosition::GM, VSCM_SLOTS> vScm;
        pipe.InitBuffer(vScm, VSCM_SLOTS, VSCM_SLOT_BYTES);
        GlobalTensor<int8_t> qg, kg;
        GlobalTensor<half> vg;
        //  H11: P 双槽 —— PV(i) 读 pgs[i & 1], 与 AIV 正在写的 pgs[(i+1) & 1] 错开。
        GlobalTensor<half> pgs[2];
        //  H15: O 双缓冲 —— AIC 等 O_USED 的唯一原因是 og 只有一个槽。两槽之后
        //  PV(i-1) 写 ogs[(i-1) & 1] 与 AIV 读另一槽互不相干, AIC 可以先跑一步。
        GlobalTensor<float> ogs[2];
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q));
        kg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(k));
        vg.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v));
        const uint32_t core = GetBlockIdx();
        PhaseProbe<Profile> probe;
        probe.Init(trace, core, 0);
        auto localWs = ws + uint64_t(core) * t.workspace_bytes;
        //  H11 布局: sg0 在 0, sg1 在 M*N*4, pg0 在 M*N*8, pg1 在 M*N*10, og 在 M*N*12。
        //  S 与 P 双缓冲, O 单槽, 理由见 vc_tiling_host.h 的 WorkspaceBytes 注释。
        GlobalTensor<int32_t> sgs[2];
        sgs[0].SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs), M * N);
        sgs[1].SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs + M * N * 4), M * N);
        pgs[0].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + M * N * 8), M * N);
        pgs[1].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + M * N * 10), M * N);
        ogs[0].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + M * N * 12), M * t.dim);
        ogs[1].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + M * N * 12) + M * t.dim, M * t.dim);
        // H6a: O_USED is awaited just before the PV that would overwrite `og`,
        // not at the end of the tile that produced it. The next tile's QK then
        // runs while the Vector halves are still reading `og` and accumulating.
        // Flag counts are unchanged (one wait per O_READY), so the Set/Wait
        // pairing that detsync_lint H7 checks still balances; the outstanding
        // wait is drained after both loops.
        //  H15: 最多两个在途, 对应 og 的两个槽。标志计数不变 (每个 O_READY 仍对应
        //  恰好一次 O_USED 等待), 在途的那些在两层循环结束后排空。
        uint32_t o_outstanding = 0;
        for (uint64_t task = core; task < t.tasks; task += t.cores) {
            const uint64_t head = task / (t.qpad / M);
            const uint32_t qr = (task % (t.qpad / M)) * M;
            //  H10: 把循环移位一拍 —— 第 i 轮先**发出**第 i 块的 QK, 再**消费**第 i-1 块。
            //  于是 QK(i) 与 AIV 处理第 i-1 块重叠, 这就是收益来源。
            //  移位写法使 QK 与 PV 的代码各只出现一次; 设备侧不能用 lambda 包装
            //  (__aicore__ 上下文里设备重载对 lambda 体不可见, 本次实撞编译失败)。
            //  sgs[i&1] 在发 QK(i) 时必定空闲: 该槽上次用于第 i-2 块, AIV 在置 P_READY(i-2)
            //  之前已读完它, 而 AIC 在第 i-1 轮消费了该标志。
            const uint32_t tiles = t.kpad / N;
            for (uint32_t i = 0; i <= tiles; ++i) {
                probe.Begin(task, i ? (i - 1) * N : 0, t);
                if (i < tiles) {
                    mmq.SetOrgShape(M, N, t.dim, t.dim);
                    mmq.SetSingleShape(M, N, t.dim);
                    mmq.SetTensorA(qg[head * t.qpad * t.dim + uint64_t(qr) * t.dim], false);
                    mmq.SetTensorB(kg[head * t.kpad * t.dim + uint64_t(i) * N * t.dim], true);
                    mmq.template IterateAll<true>(sgs[i & 1], 0);
                    mmq.End();
                    CrossCoreSetFlag<2, PIPE_FIX>(S_READY);
                }
                //  H16b: V(i) 的装入与 P 毫无依赖 —— 它只由 (head, i) 决定, 放在
                //  CrossCoreWaitFlag(P_READY) **之前**即可; 本轮消费的是另一槽里上一轮
                //  装好的 V(i-1), 预装有整整一轮时间落地。
                //  ★位置必须在 QK 与 S_READY **之后**: H16 把它放在 QK 之前, 与 QK 自己的
                //  MTE2 争用, 实测 S_READY 等待 25.4% -> 40.1% (AIV 等 QK 等得更久),
                //  抵掉了 O_READY 的 47.9% -> 34.3%, 端到端净亏 +0.35%。放到这里,
                //  它占用的是 AIC 等 P_READY 的那段空闲 (h15 实测 50.6%), 不再推迟 S_READY。
                //  深度 2 的 TQue 自带 WAR 互锁: 两槽都在途时 AllocTensor 自己会挡住,
                //  不必手写 MTE1->MTE2 的序 (F89/F96 那类先写后读在这里由队列负责)。
                if (i < tiles) {
                    LocalTensor<half> vl1 = vScm.AllocTensor<half>();
                    Nd2NzParams nd;
                    nd.ndNum = 1;
                    nd.nValue = N;                 // K 方向: 本 tile 的 key 数
                    nd.dValue = t.dim;
                    nd.srcNdMatrixStride = 0;
                    nd.srcDValue = t.dim;          // 源 ND 相邻行起始地址的元素步长
                    nd.dstNzC0Stride = N;          // CeilDiv(N,16)*16; N 已是 16 的倍数
                    nd.dstNzNStride = 1;
                    nd.dstNzMatrixStride = 0;
                    DataCopy(vl1, vg[head * t.kpad * t.dim + uint64_t(i) * N * t.dim], nd);
                    vScm.EnQue(vl1);
                }
                probe.MarkAll(0); // ★H10 后这一段量的是**下一块**的 QK, 不是被消费的那块
                if (i == 0) {
                    probe.Finish();
                    continue;      // 第 0 轮只发不消费
                }
                CrossCoreWaitFlag(P_READY);  // both AIV halves published P
                probe.Mark(1); // exposed wait, not the duration of all Vector work
                if (o_outstanding >= 2) {
                    //  H15: 两个槽都在途时才必须等 —— 要覆写的那个槽 (ogs[(i-1)&1])
                    //  上一次装的是第 i-3 块的 O, AIV 早已读完。常见情形下这里不阻塞。
                    CrossCoreWaitFlag(O_USED);
                    --o_outstanding;
                }

                // A[P] stride=N; B[V] stride=D. Four-argument form is intentional.
                mmv.SetOrgShape(M, t.dim, N, t.dim);
                mmv.SetSingleShape(M, t.dim, N);
                //  H11: 被消费的是第 i-1 块, 其 P 在 pgs[(i-1) & 1]。
                //  H16: V(i-1) 已在上一轮预装进 L1; DeQue 只做等待与取槽, 不再有 GM 搬运。
                LocalTensor<half> vl1prev = vScm.DeQue<half>();
                mmv.SetTensorA(pgs[(i - 1) & 1], false);
                mmv.SetTensorB(vl1prev, false);
                mmv.template IterateAll<true>(ogs[(i - 1) & 1], 0);
                mmv.End();
                vScm.FreeTensor(vl1prev);
                probe.MarkAll(2); // PV API, including completed GM writeback
                CrossCoreSetFlag<2, PIPE_FIX>(O_READY);
                ++o_outstanding;  // awaited before the PV that reuses this slot
                probe.Mark(3);
                probe.Finish();
            }
        }
        while (o_outstanding) {
            CrossCoreWaitFlag(O_USED);  // drain the outstanding tiles before the kernel ends
            --o_outstanding;
        }
    }
private:
    MatmulImpl<QType, KType, IType, IType, MM_CONFIG> mmq;
    MatmulImpl<HType, VType, FType, FType, MM_CONFIG> mmv;
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
        //  H11: 由 8 槽扩到 10 槽 —— 第 8/9 槽存 alpha 的两份暂存。放进同一缓冲而不是
        //  另开一个 ROWS*4 的小 TBuf: 小缓冲既容易被向量指令的读取粒度越界, 又会漏进
        //  VectorBytes 的账 (主机的 UB 预算检查就对不上)。
        pipe.InitBuffer(stateBuf, 10 * ROWS * 4);
        pipe.InitBuffer(lutBuf, 128 * 4);
        //  H26: 位运算解码用的两块。subBuf 存 e=0 支的 c/512, cmpBuf 存逐元素掩码 (1 bit/元素)。
        pipe.InitBuffer(subBuf, ROWS * N * 4);
        pipe.InitBuffer(cmpBuf, ROWS * N / 8);
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
        //  H11: acc 的 rescale 推迟到拿到对应 pv 的那一刻, 故 alpha 必须按 tile 暂存两槽。
        LocalTensor<float> alphaHold[2] = {states[8 * ROWS], states[9 * ROWS]};
        auto table = lutBuf.Get<float>();
        auto brcb = brcbBuf.Get<float>();
        auto bmax = brcb[0], balpha = brcb[ROWS * 8];
        auto tmp = tmpBuf.Get<uint8_t>();
        auto e0val = subBuf.Get<float>();
        auto cmp = cmpBuf.Get<uint8_t>();
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
        GlobalTensor<float> ogs[2];
        //  H10/H11: 与 AIC 侧同一布局; AIV 按本块序号的奇偶选 S 槽与 P 槽。
        GlobalTensor<int32_t> sgs[2];
        GlobalTensor<half> pgs[2];
        sgs[0].SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs) + sub * ROWS * N, ROWS * N);
        sgs[1].SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs + M * N * 4) + sub * ROWS * N, ROWS * N);
        pgs[0].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + M * N * 8) + sub * ROWS * N, ROWS * N);
        pgs[1].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + M * N * 10) + sub * ROWS * N, ROWS * N);
        ogs[0].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + M * N * 12) + sub * ROWS * t.dim, ROWS * t.dim);
        ogs[1].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + M * N * 12) + M * t.dim + sub * ROWS * t.dim, ROWS * t.dim);
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
            //  H11: 移位循环 —— 第 j 轮先算第 j 块的概率并发布, 再消费第 j-1 块的 PV。
            //  这样 AIV 在等 O_READY(j-1) 之前已经把下一块的概率算完,
            //  原本大头的等待被真实工作填上。设备侧不能用 lambda 做结构抽象 (见 F54),
            //  故与 AIC 侧同样采用移位而非复制代码。
            const uint32_t tiles = t.kpad / N;
            for (uint32_t j = 0; j <= tiles; ++j) {
                const uint32_t key = j < tiles ? j * N : (tiles - 1) * N;
                probe.Begin(task, j ? (j - 1) * N : 0, t);
                if (j < tiles) {
                const uint32_t tile = j;
                CrossCoreWaitFlag(S_READY);
                probe.Mark(0);
                //  ★raw 身兼两职: 这里由 MTE2 接收 S, 到 ExpCast 段又存 gather 的字节偏移
                //  供向量单元读。移位循环把上一轮的 Gather 与本轮这次 DataCopy 拉得很近,
                //  缺这条 V->MTE2 的序就会在 expcast 且 tiles>=2 时触发 aicore exception
                //  (plog: fftsplus aivector error)。此前是用一块独立缓冲绕开的 ——
                //  那是拿 ROWS*N*4 的 UB 换一条同步, 这里把它换回来。
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopy(raw, sgs[tile & 1], ROWS * N);
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
                    //  Maxs(0) 是承担越界防护的那一条: 负下标会真的读到表外。
                    Maxs(score, score, 0.0f, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    //  H25: 上界夹持 Mins(score, 120) **恒不生效**, 已删。
                    //  不变式: 本段入口处 score 已是 `Sub(score, score, bmax)` 的结果,
                    //  而 bmax 是 rowmax = Max(ReduceMax(本 tile), m) 的广播, 故 bmax >= 本 tile
                    //  每一个元素; IEEE754 的 a-b 在 b>=a 时**精确** <= 0, 不存在舍入成正数。于是
                    //      score <= 0 -> score*EXPCAST_GAIN <= 0 (GAIN=11.5415... 为正) -> +120 <= 120
                    //  上界永远不会被触到。全掩码行给出 score-rowmax == 0, 结果恰为 120, 仍在界内。
                    //  ★若将来改动 rowmax 的定义使 score 可能大于 rowmax, 这条删除就不再成立 ——
                    //  不变式的来源就是上面那次 Sub, 改它时请一并复核此处。
                    Cast(raw, score, RoundMode::CAST_RINT, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    //  H26: 用位运算直接构造 E4M3 的解码值, 替掉查表。
                    //  码 c = 8e + f 且 f < 8, 故 `c << 20` 本身就等于 `e<<23 | f<<20`;
                    //  再加常量 `120 << 23` (= 127-7 的指数偏置) 得到的 int32, 按 fp32 读出
                    //  恰好是 (1 + f/8) * 2^(e-7) —— 与表里 e>=1 的那 113 项**逐位相同**。
                    //  ★e == 0 (即 c < 8) 那一支例外: E4M3 在指数域为 0 时没有隐含的前导 1,
                    //  值是 f/512, 与位模式不符, 必须单独算再按掩码选。这一支**不能按出现频率
                    //  省掉** —— 尾块掩码的无效列必然落在 c = 0, 而行内跨度超过 9.7041 的
                    //  分布 (实测 q 尺度放大 1.5 倍即 20/20 命中) 在正常列上也会落进来。
                    Cast(e0val, raw, RoundMode::CAST_NONE, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    //  ★比较走 **fp32** 而不是 int32: 向量比较单元在本机只对浮点给出
                    //  正确结果, 用 int32 实测使 c<8 这一支选错 (逐位 16/32 不符,
                    //  相对 RMSE 0.0432% —— 小而弥散, 正是少数元素错经 softmax 归一化
                    //  传播的形态)。c 在 fp32 里是精确整数 0..120, 故 `c < 8.0f` 精确。
                    CompareScalar(cmp, e0val, 8.0f, CMPMODE::LT, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Muls(e0val, e0val, 1.0f / 512.0f, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    ShiftLeft(raw, raw, int32_t(20), ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Adds(raw, raw, int32_t(120) << 23, ROWS * N);
                    PipeBarrier<PIPE_V>();
                    Select(score, cmp, e0val, raw.ReinterpretCast<float>(),
                           SELMODE::VSEL_TENSOR_TENSOR_MODE, ROWS * N);
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
                DataCopy(pgs[tile & 1], ph, ROWS * N);
                CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);
                probe.template MarkEvent<HardEvent::MTE3_S>(3, pipe);
                //  l 与 m 只依赖 rowmax 与 mass, 与 pv 无关, 故立即更新。
                Mul(l, l, alpha, ROWS);
                PipeBarrier<PIPE_V>();
                Add(l, l, mass, ROWS);
                PipeBarrier<PIPE_V>();
                Adds(m, rowmax, 0.0f, ROWS);
                PipeBarrier<PIPE_V>();
                //  ★acc 的 rescale **不在此处施加** —— acc 此刻仍停在第 j-1 块的尺度,
                //  它要等拿到 pv(j-1) 之后才能前进一格。把 alpha(j) 存起来, 到第 j+1 轮再用。
                Adds(alphaHold[tile & 1], alpha, 0.0f, ROWS);
                probe.template MarkEvent<HardEvent::V_S>(4, pipe);
                }
                //  ★ph 的 MTE3 必须在**下一轮重写 ph 之前**排空, 故这一对 Flag 要无条件执行 ——
                //  原写法把它放在 j==0 的 continue 之后, 于是第 0 轮发起的 DataCopy(pgs[0], ph)
                //  还在飞行中, 第 1 轮的 Cast(ph, ...) 就把它覆写了 (先写后读冲突)。
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                if (j == 0) {
                    probe.Finish();
                    continue;      // 第 0 轮只发布概率, 没有可消费的 PV
                }
                const uint32_t prev = j - 1;
                CrossCoreWaitFlag(O_READY);
                probe.Mark(5); // only PV wait still exposed after the next tile's probability
                DataCopy(pv, ogs[prev & 1], ROWS * t.dim);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                probe.template MarkEvent<HardEvent::MTE2_S>(6, pipe);
                PipeBarrier<PIPE_V>();
                //  acc = acc * alpha(prev) + pv(prev)。两项独立, 与即时累加**逐位相同** ——
                //  每一项的浮点运算顺序未变, 只是施加时刻推迟。
                // H8: Muls by a scalar and Mul by a vector holding that scalar are the
                // same FP32 multiply, so the result is bit identical.
                Brcb(balpha, alphaHold[prev & 1], ROWS / 8, BrcbRepeatParams(1, 8));
                PipeBarrier<PIPE_V>();
                for (uint32_t off = 0; off < t.dim; off += 64)
                    Mul(acc[off], acc[off], balpha, uint64_t(64), ROWS, accBrc);
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
    TBuf<TPosition::VECCALC> subBuf, cmpBuf;
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
