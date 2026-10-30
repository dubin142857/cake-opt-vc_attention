// Ascend A2/A3 DAV-2201. Original VC adaptation using documented Ascend C APIs.
// Design sources and API versions are recorded in ../SOURCES.json.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "vc_tiling.h"
#include "vc_phase_probe.h"
#include "vc_cube_v2.h"

using namespace AscendC;
using namespace matmul;
using namespace vc910;

//  ---- 测量变体: 把硬件性能计数器门控到 AIV 的**工作**区间 -------------------
//  ★依据 F207: `AscendC::MetricsProfStart/Stop` 在本机 (Atlas A2 / dav_c220) 可用,
//    公开接口在 kernel_prof_trace_intf.h, 已由 kernel_operator.h 拉进来, 无需新增 include。
//    实现是 pipe_barrier(PIPE_ALL) + set_ctrl(sbitset1(get_ctrl(), 0)) + pipe_barrier(PIPE_ALL),
//    即用 CTRL 第 0 位开关计数器; 官方文档说明它正是给 msOpProf 标代码段边界用的。
//  ★为什么需要它 (F205): msprof 的 `aiv_scalar_time` 是整核聚合且**混入了阻塞等待**
//    (CrossCoreWaitFlag 由标量发射并阻塞标量单元), 不能用来判 AIV 的真实工作量。
//    把两处跨核等待排除在计数区间外, 就能拿到硬件计数的干净值, 与相位探针互为旁证。
//  ★它两侧各插一道 pipe_barrier(PIPE_ALL), **是扰动**: 只做测量变体, 默认关, 不进生产。
//  ★默认 false: 置位后 MetricsOff()/MetricsOn() 之间的代码被排除在硬件计数器之外。
//  ★已校准 (F208): 门控掉整个趟 2 循环时 aiv_scalar_time 降 47.4% 而 aiv_vec_time 只变 +0.1%
//    —— 即它**只门控 aiv_scalar_time**, 对 vec / mte3 无效。用它判 vec 是无效的。
constexpr bool VC_METRICS_REGION = false;
__aicore__ inline void MetricsOff() { if constexpr (VC_METRICS_REGION) MetricsProfStop(); }
__aicore__ inline void MetricsOn()  { if constexpr (VC_METRICS_REGION) MetricsProfStart(); }

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
//  ★旧 Cube 类已删 (2026-10-29)。它用两个 MatmulImpl 逐 tile 交替, 而那正是
//    F188 查实的瓶颈成因: L0B 需 2x(32+64)=192 KB > 64 KB -> 两相不能共存 ->
//    dbL0A/B/C 全关 -> 流水不重叠 -> 标量卡在矩乘内部 flag 上空转, scalar 成 binding
//    (AIC 238019 us, 窗口/binding 1.164; 单 tile 252.5 ticks 里 matmul API 占 251.5,
//     而 cube 真算只要 49.2 —— API 开销 80%)。
//  ★留它当对照没有意义且有害: 跨核旗标会与 CubeV2 的 chunk 级置位混在一起,
//    detsync_lint 的 H6 永远红, 看不出真正的不平衡。对照用 git tag
//    delivery-h26-bitdecode (e2fc535, 产物 3494c48d), 那才是可复现的基线。

template <uint32_t M, bool Profile>
class Vector {
    static constexpr uint32_t ROWS = M / 2;
    static_assert(M == 32 || M == 64, "Each AIV must stay within one Q32 quantization block");
public:
    __aicore__ inline void RunVector(GM_ADDR qs, GM_ADDR ks, GM_ADDR vs, GM_ADDR lut,
                              GM_ADDR output, GM_ADDR ws, GM_ADDR trace, const Tiling& t) {
        TPipe pipe;
        //  ★raw 双缓冲: 趟 1 的 S 搬运与本块计算重叠。UB 精算 (ROWS=32,N=128,MAX_D=128,CHUNK=4):
        //    raw 2x16384 + score 65536 + p 8192 + acc 16384 + pv 16384 + vs 512 + state 1536
        //    + lut 512 + sub 16384 + cmp 512 + brcb 2048 = 160768, UB 196608, 余 35840 - reduce_bytes。
        pipe.InitBuffer(rawBuf, VEC_RAW_SLOTS * ROWS * N * 4);
        //  ★score 由单块扩到**整个 chunk**: 两趟法要求全 chunk 用同一归一化基准,
        //    故第一趟算出的分数必须留着给第二趟用 (重读 GM 会让 AIV 读由 186 涨到 335 GB)。
        pipe.InitBuffer(scoreBuf, map::CHUNK * ROWS * N * 4);
        pipe.InitBuffer(pBuf, ROWS * N * 2);
        pipe.InitBuffer(accBuf, ROWS * MAX_D * 4);
        pipe.InitBuffer(pvBuf, ROWS * MAX_D * 4);
        pipe.InitBuffer(vsBuf, MAX_D * 4);
        //  H11: 由 8 槽扩到 10 槽 —— 第 8/9 槽存 alpha 的两份暂存。放进同一缓冲而不是
        //  另开一个 ROWS*4 的小 TBuf: 小缓冲既容易被向量指令的读取粒度越界, 又会漏进
        //  VectorBytes 的账 (主机的 UB 预算检查就对不上)。
        //  10 -> 12 槽: 第 10 槽存 chunk 内逐块的 ReduceMax 中间量 tmax,
        //  第 11 槽存 chunk 的质量累加 massAcc (原来每 tile 一次, 现在每 chunk 一次)。
        pipe.InitBuffer(stateBuf, 12 * ROWS * 4);
        pipe.InitBuffer(lutBuf, 128 * 4);
        //  H26: 位运算解码用的两块。subBuf 存 e=0 支的 c/512, cmpBuf 存逐元素掩码 (1 bit/元素)。
        pipe.InitBuffer(subBuf, ROWS * N * 4);
        pipe.InitBuffer(cmpBuf, ROWS * N / 8);
        // H8: two per-row broadcast buffers. Brcb expands one value into a full
        // 32B block, so ROWS values occupy ROWS blocks = ROWS * 8 floats each.
        pipe.InitBuffer(brcbBuf, 2 * ROWS * 8 * 4);
        pipe.InitBuffer(tmpBuf, t.reduce_bytes);
        const int32_t taskDoneEvent = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        auto rawAll = rawBuf.Get<int32_t>();
        //  趟 2 / ExpCast 段仍只用槽 0 —— 那时趟 1 已全部消费完, 槽 0 空闲。
        auto raw = rawAll;
        auto chunkScore = scoreBuf.Get<float>();   // CHUNK 块连续排布, 第 j 块在 [j*ROWS*NB]
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
        auto tmax = states[10 * ROWS], massAcc = states[11 * ROWS];
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
        //  ★V2 布局 (与 CubeV2::RunCube 和 WorkspaceBytesV2 是同一个量的三处写法,
        //    改一处必须改另外两处)。S/P 各 2*CHUNK 槽供错位, O 双槽。
        //  ★原来是 sgs[2]/pgs[2] 的旧布局, 而新循环按 sgs[slot + j] 索引 (0..2*CHUNK-1) ——
        //    数组越界读野指针, 实测表现为输出近乎为零 (relative_rmse 1.0)。
        constexpr uint32_t NSLOT = 2 * map::CHUNK;
        GlobalTensor<int32_t> sgs[NSLOT];
        GlobalTensor<half> pgs[NSLOT];
        const uint64_t S_BYTES = uint64_t(NSLOT) * M * N * 4;
        const uint64_t P_BYTES = uint64_t(NSLOT) * M * N * 2;
        for (uint32_t z = 0; z < NSLOT; ++z) {
            sgs[z].SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWs)
                                   + uint64_t(z) * M * N + sub * ROWS * N, ROWS * N);
            pgs[z].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(localWs + S_BYTES)
                                   + uint64_t(z) * M * N + sub * ROWS * N, ROWS * N);
        }
        ogs[0].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + S_BYTES + P_BYTES)
                               + sub * ROWS * t.dim, ROWS * t.dim);
        ogs[1].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localWs + S_BYTES + P_BYTES)
                               + M * t.dim + sub * ROWS * t.dim, ROWS * t.dim);
        DataCopy(table, lg, 128);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        constexpr uint32_t NB = N;        // 向量侧的键块宽度, 与 map::NBLK 同值 (有断言守)
        constexpr uint32_t CHUNK = map::CHUNK;
        uint32_t shape[2] = {ROWS, NB};
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
            //  ---- 按 chunk 分阶段 (与 CubeV2 配对) --------------------------------
            //  ★为什么要两趟: chunk 内跨 key 块在 L0C 上累加 O, 要求整个 chunk 的 P
            //    用**同一个**归一化基准。单趟只能用运行最大值, 基准逐块变, 累加就不成立。
            //    不把 chunk 的 S 留在 UB 而改成重读 GM 的话, AIV 读由 186 GB 涨到 335 GB,
            //    **比基线 224.4 还差** —— 所以留 UB (+48 KB, 预算内 141.1/192 KB)。
            //  ★基准取 newm = max(chunk 最大, m), 与原来的 max(tile 最大, m) 同形,
            //    故 acc = acc*alpha + O_chunk 的形式不变, 只是频次由每 tile 一次变每 chunk 一次,
            //    也不需要额外的 beta 因子。
            //  ★精度: newm >= chunk 最大 >= 各 tile 最大, 与原来一样是上界, 不会溢出;
            //    但对 chunk 内靠前、自身最大值偏小的块, exp 会比原来更小 —— 下溢风险按闸实测,
            //    decoded_reference 当前余量 143 倍 (7e-5 对 1% 闸)。
            const uint32_t chunks = (tiles + CHUNK - 1) / CHUNK;
            for (uint32_t c = 0; c <= chunks; ++c) {
                probe.Begin(task, c ? (c - 1) * CHUNK * NB : 0, t);
                if (c < chunks) {
                const uint32_t slot = (c & 1) * CHUNK;
                const uint32_t base_i = c * CHUNK;
                const uint32_t n_in = (tiles - base_i) < CHUNK ? (tiles - base_i) : CHUNK;
                CrossCoreWaitFlag(S_READY);
                //  ---- 相位布点 (仅 PhaseProbe<true> 的 vc910_profile_m64 生效) --------
                //  ★一律用**不插屏障**的 Mark: 它只读系统计数, 量的是**发射**时间,
                //    正是 aiv_scalar_busy 要分解的东西。MarkEvent 会插 Set/Wait, 是扰动。
                //  ★相位 0/1/3/4 是**每块**的, 同一 chunk 内逐块覆写, 故读到的是**末块**的值;
                //    相位 2/5/6/7 是每 chunk 的。
                probe.Mark(7);   // wait_s: 跨核等待 + 序幕的 S 搬运发射
                //  ---- 趟 1: 读本 chunk 的全部 S, 施加 scale, 求 chunk 最大 ----
                Adds(rowmax, m, 0.0f, ROWS);       // 以历史最大起算
                PipeBarrier<PIPE_V>();
                //  ---- 趟 1 的软件流水: 第 j+1 块的 S 搬运发在第 j 块的计算**之前** ----
                //  ★依据 F202: AIV 的 vec 有 32.9% 的时间在等, 其中一处正是本循环里
                //    `WaitFlag<MTE2_V>` —— 每块都要停下来等 S 从 GM 搬到 UB。
                //    raw 双缓冲后, 第 j 块算的时候第 j+1 块已在搬, 这道等待就被藏掉。
                //  ★V_MTE2 是**相邻自配对** (一次性把 V 定序在 MTE2 之前), 故各处都用
                //    EVENT_ID0 即可; 只有 MTE2_V 要按槽分两个号, 否则第 j 块会吃掉第 j+1 块的置位。
                //  ★序幕的那对 V_MTE2 仍然必需: 上一 chunk 的 ExpCast 段用 raw 槽 0 存
                //    gather 偏移, 缺这条序会在 expcast 且 tiles>=2 时触发 aicore exception
                //    (plog: fftsplus aivector error)。
                //  ---- 本 chunk 的全部 k-scale 一次搬完 (F210 之后的 t1_setup 48.2%) ----
                //  ★原先每块搬一次 8 个 float, 而 MTE2 **按序**: 第 j 块的 kscale 排在
                //    上一轮发的 16 KB 预取**之后**, 于是"等 kscale"隐含等掉了那次大搬运,
                //    预取只买到一个块的提前量。合并到 chunk 头并排在预取之前,
                //    标量就只等这 128 B。
                //  ★相邻分块的 scale 在 ksg 里连续 (KSB_PER_TILE=1 且 NB == K_SCALE_BLOCK,
                //    故块号 = 块下标)。**按 n_in 截断**: 末 chunk 若按 CHUNK 搬会读到
                //    下一个 head 的数据 (最后一个 head 则越界)。
                //  ★容量: kscale 是 stateBuf 的第 6 槽, ROWS=32 个 float; 需 8*KSB_PER_TILE*CHUNK
                //    = 32 个, 正好装满。
                static_assert(8 * KSB_PER_TILE * map::CHUNK <= M / 2,
                              "★kscale 槽 (ROWS 个 float) 装不下整个 chunk 的 k-scale");
                DataCopy(kscale,
                         ksg[(head * (t.kpad / K_SCALE_BLOCK) + base_i * KSB_PER_TILE) * 8],
                         8 * KSB_PER_TILE * n_in);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopy(rawAll[0], sgs[slot], ROWS * NB);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                //  只等 kscale 那次小搬运: 它排在 MTE2 队列里大块预取**之前**, 故先完成。
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t j = 0; j < n_in; ++j) {
                    const uint32_t key = (base_i + j) * NB;
                    auto sc = chunkScore[j * ROWS * NB];
                    const uint32_t cur = j & 1, nxt = cur ^ 1u;
                if (j + 1 < n_in) {
                    //  此处的 V_MTE2 守的是 raw[nxt] —— 它上一次被读是第 j-1 块的 Cast,
                    //  而本语句排在第 j 块的 Cast 之前, 故正好定序在那次读之后。
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    DataCopy(rawAll[nxt * ROWS * N], sgs[slot + j + 1], ROWS * NB);
                    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(nxt));
                }
                //  本块的 scale 在合并缓冲里的下标: (j*KSB_PER_TILE + sb) * 8
                float seg_scale[KSB_PER_TILE];
                for (uint32_t sb = 0; sb < KSB_PER_TILE; ++sb)
                    seg_scale[sb] = (qscale.GetValue(0)
                                     * kscale.GetValue((j * KSB_PER_TILE + sb) * 8)) * t.scale;
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(cur));
                probe.Mark(0);   // t1_setup: kscale 搬运 + 旗标 + 预取发射 + GetValue
                Cast(sc, rawAll[cur * ROWS * N], RoundMode::CAST_NONE, ROWS * NB);
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
                            Muls(sc[sb * K_SCALE_BLOCK + off], sc[sb * K_SCALE_BLOCK + off],
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
                        for (uint32_t j = valid; j < N; ++j) sc.SetValue(r * N + j, -3.402823466e38f);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
                    ReduceMax<float, Pattern::Reduce::AR, false>(tmax, sc, tmp, shape, true);
                    PipeBarrier<PIPE_V>();
                    Max(rowmax, rowmax, tmax, ROWS);
                    PipeBarrier<PIPE_V>();
                    probe.Mark(1);   // t1_vec: Cast + Muls + ReduceMax + Max 的发射
                }
                Sub(alpha, m, rowmax, ROWS);
                PipeBarrier<PIPE_V>();
                Exp(alpha, alpha, ROWS);
                PipeBarrier<PIPE_V>();
                // H8: broadcast rowmax per row instead of reading ROWS scalars back.
                Brcb(bmax, rowmax, ROWS / 8, BrcbRepeatParams(1, 8));
                PipeBarrier<PIPE_V>();
                //  ---- 趟 2: 归一化 + exp + 写 P ----
                Duplicate(massAcc, 0.0f, ROWS);
                PipeBarrier<PIPE_V>();
                //  ★对照 (E24): 把整个趟 2 循环排除在计数区间外。趟 2 含每块的
                //    Sub/Exp/ReduceSum/Cast, 是 aiv_vec_time 的大头 —— 若门控有效,
                //    aiv_vec_time 必然明显下降; 若纹丝不动, 说明门控在本采集模式下无效。
                MetricsOff();
                for (uint32_t j = 0; j < n_in; ++j) {
                    auto sc = chunkScore[j * ROWS * NB];
                    for (uint32_t off = 0; off < NB; off += 64)
                        Sub(sc[off], sc[off], bmax, uint64_t(64), ROWS, rowBrc);
                    PipeBarrier<PIPE_V>();
                if (t.expcast) {
                    // Separate FP32 multiply/add is part of the numerical contract.
                    Muls(sc, sc, EXPCAST_GAIN, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    Adds(sc, sc, 120.0f, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    //  Maxs(0) 是承担越界防护的那一条: 负下标会真的读到表外。
                    Maxs(sc, sc, 0.0f, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    //  H25: 上界夹持 Mins(sc, 120) **恒不生效**, 已删。
                    //  不变式: 本段入口处 sc 已是 `Sub(sc, sc, bmax)` 的结果,
                    //  而 bmax 是 rowmax = Max(ReduceMax(本 tile), m) 的广播, 故 bmax >= 本 tile
                    //  每一个元素; IEEE754 的 a-b 在 b>=a 时**精确** <= 0, 不存在舍入成正数。于是
                    //      sc <= 0 -> sc*EXPCAST_GAIN <= 0 (GAIN=11.5415... 为正) -> +120 <= 120
                    //  上界永远不会被触到。全掩码行给出 sc-rowmax == 0, 结果恰为 120, 仍在界内。
                    //  ★若将来改动 rowmax 的定义使 sc 可能大于 rowmax, 这条删除就不再成立 ——
                    //  不变式的来源就是上面那次 Sub, 改它时请一并复核此处。
                    Cast(raw, sc, RoundMode::CAST_RINT, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    //  H26: 用位运算直接构造 E4M3 的解码值, 替掉查表。
                    //  码 c = 8e + f 且 f < 8, 故 `c << 20` 本身就等于 `e<<23 | f<<20`;
                    //  再加常量 `120 << 23` (= 127-7 的指数偏置) 得到的 int32, 按 fp32 读出
                    //  恰好是 (1 + f/8) * 2^(e-7) —— 与表里 e>=1 的那 113 项**逐位相同**。
                    //  ★e == 0 (即 c < 8) 那一支例外: E4M3 在指数域为 0 时没有隐含的前导 1,
                    //  值是 f/512, 与位模式不符, 必须单独算再按掩码选。这一支**不能按出现频率
                    //  省掉** —— 尾块掩码的无效列必然落在 c = 0, 而行内跨度超过 9.7041 的
                    //  分布 (实测 q 尺度放大 1.5 倍即 20/20 命中) 在正常列上也会落进来。
                    Cast(e0val, raw, RoundMode::CAST_NONE, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    //  ★比较走 **fp32** 而不是 int32: 向量比较单元在本机只对浮点给出
                    //  正确结果, 用 int32 实测使 c<8 这一支选错 (逐位 16/32 不符,
                    //  相对 RMSE 0.0432% —— 小而弥散, 正是少数元素错经 softmax 归一化
                    //  传播的形态)。c 在 fp32 里是精确整数 0..120, 故 `c < 8.0f` 精确。
                    CompareScalar(cmp, e0val, 8.0f, CMPMODE::LT, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    Muls(e0val, e0val, 1.0f / 512.0f, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    ShiftLeft(raw, raw, int32_t(20), ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    Adds(raw, raw, int32_t(120) << 23, ROWS * NB);
                    PipeBarrier<PIPE_V>();
                    Select(sc, cmp, e0val, raw.ReinterpretCast<float>(),
                           SELMODE::VSEL_TENSOR_TENSOR_MODE, ROWS * NB);
                } else {
                    Exp(sc, sc, ROWS * NB);
                }
                PipeBarrier<PIPE_V>();
                //  ★先发布 P, 再求行和 —— 与原先的次序相反。理由: 下面的折叠会**原地破坏** sc,
                //    故必须先把 P 取走。`mass` 仍是**取 half 之前**的 fp32 之和 ——
                //    Cast 不修改 sc, 所以数值契约不变 (reference.py 的 p = p.half().float()
                //    对应的正是"分母用 pre-cast 的 fp32"这一条)。
                Cast(ph, sc, RoundMode::CAST_NONE, ROWS * NB);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(pgs[slot + j], ph, ROWS * NB);
                //  ---- 行和: 折 NB -> NB/2 再一条 WholeReduceSum, 替掉泛型 ReduceSum ----
                //  ★依据 F229 (读的是 CANN 自己的 reduce_sum_v220_impl.h::ReduceSumArCompute):
                //      last <= 64 (fp32 的一个 repeat) 时, 它只发**一条** WholeReduceSum;
                //      last > 64 走的是 ReduceSumInLargeLast —— 多步 + tmpBuf 搬运。
                //    我们 last = NB = 128, 一直走的是贵的那条。先折成 64 就落进廉价分支。
                //  ★上界由 F228 的消融给出 7.29% (把 ReduceSum 整个换成 Duplicate 时的窗口降幅);
                //    本改动保留归约本身, 故实际只会更少。
                //  ★形参由 scripts/probe_vec_reduce.py 逐位坐实 (与 NumPy 行和相同),
                //    且实测 WholeReduceSum **不会**把 mask 留给后续全宽操作, 无需手动 ResetMask
                //    —— 注意高层 ReduceSum 收尾是有 SetMaskNorm()+ResetMask() 的, 底层这条没有。
                static_assert(NB / 2 == 64, "折叠假定 NB 正好是两个 fp32 repeat (每个 64 元素)");
                Add(sc[0], sc[NB / 2], sc[0], uint64_t(NB / 2), ROWS,
                    BinaryRepeatParams(1, 1, 1, NB / 8, NB / 8, NB / 8));
                PipeBarrier<PIPE_V>();
                WholeReduceSum<float>(mass, sc, NB / 2, ROWS, 1, 1, NB / 8);
                PipeBarrier<PIPE_V>();
                probe.Mark(3);   // t2_vec: Sub + Exp + Cast(ph) + 发布 + 折叠 + WholeReduceSum
                                 //  ★口径变了: 发布已挪进本段, 故不要与改动前的 t2_vec 直接比大小
                    Add(massAcc, massAcc, mass, ROWS);
                    //  ★这里原有一道 PipeBarrier<PIPE_V>, 已删: massAcc 的下一个读者
                    //    要么是下一块的 Add (中间隔着 L251/L298/L300 三道屏障),
                    //    要么是循环之后的 Add(l, l, massAcc) (中间隔着 Mul 之后那道)。
                    //    两条路径上都另有屏障, 故本道不承担任何 RAW。
                    //  ★PipeBarrier<PIPE_V> 在 __NPU_ARCH__==2201 上**只在 AIC 空操作**,
                    //    在 AIV 上是真实指令 (kernel_reg.h:105-112)。F200: AIV 是关键路径,
                    //    其 vec 与 scalar 串行, 屏障强制 V 队列排空正是串行的机械原因。
                    //  ★ph 的 MTE3 必须在下一块重写 ph 之前排空 (先写后读冲突)。
                    //    那由下面的 MTE3_V 旗标负责, **不是**上面那道 V 屏障的职责。
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    probe.Mark(4);   // t2_publish: Cast(ph) + 旗标 + DataCopy + Add 的发射
                }
                MetricsOn();
                CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);   // ★整个 chunk 只置一次
                Mul(l, l, alpha, ROWS);
                PipeBarrier<PIPE_V>();
                Add(l, l, massAcc, ROWS);
                PipeBarrier<PIPE_V>();
                Adds(m, rowmax, 0.0f, ROWS);
                PipeBarrier<PIPE_V>();
                Adds(alphaHold[c & 1], alpha, 0.0f, ROWS);
                PipeBarrier<PIPE_V>();
                probe.Mark(5);   // chunk_tail: P_READY + Mul/Add/Adds/Adds 的发射
                }
                if (c == 0) { probe.Finish(); continue; }   // 第 0 轮只发布, 没有可消费的 PV
                const uint32_t pc = c - 1;
                CrossCoreWaitFlag(O_READY);
                //  ★相位 2 改量这道**等待**本身: 原先它与 pv 的真实工作混在 pv_consume 里,
                //    而"等多久"与"干多久"是两回事, 混在一起无法下结论 (E10)。
                probe.Mark(2);   // wait_o: 等 AIC 把 O 搬出来

                DataCopy(pv, ogs[pc & 1], ROWS * t.dim);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                PipeBarrier<PIPE_V>();
                //  acc = acc * alpha(prev) + pv(prev)。两项独立, 与即时累加**逐位相同** ——
                //  每一项的浮点运算顺序未变, 只是施加时刻推迟。
                // H8: Muls by a scalar and Mul by a vector holding that scalar are the
                // same FP32 multiply, so the result is bit identical.
                Brcb(balpha, alphaHold[pc & 1], ROWS / 8, BrcbRepeatParams(1, 8));
                PipeBarrier<PIPE_V>();
                for (uint32_t off = 0; off < t.dim; off += 64)
                    Mul(acc[off], acc[off], balpha, uint64_t(64), ROWS, accBrc);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, pv, ROWS * t.dim);
                probe.Mark(6);   // pv_consume: 等 O + 搬 pv + Brcb/Mul/Add 的发射
                CrossCoreSetFlag<2, PIPE_V>(O_USED);
                probe.Finish();
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

//  ★二分实验开关 (F191 之后那个数值错的定位): 置 true 则在 QK 的四段之后各加一道
//    保守的 asc_sync(), 标志协议保持不动。若这样能恢复正确, 说明缺陷在**标志协议**;
//    若仍错, 说明缺陷在**四段函数本身或 ping/pong 寻址**, 那是另一条路。
//    读码已经数不出问题 (单轮的 notify/wait 计数逐条核过是平衡的), 故改用实验。
//  ---- 二分旋钮: 四段各自的保守屏障, 用留一法定位哪一道在承担顺序 ----------
//  ★实测前提 (F195, 卡 3, 与对照同卡): 四道全开时流水版**确定**且与 _sync 版
//    逐位相同; 全关时同一形状三次三种结果。故竞态在旗标协议, 不在寻址或结构。
constexpr bool VC_SYNC_GM2L1 = false;   // 序幕与预取的 GM->L1 之后
constexpr bool VC_SYNC_L12L0 = false;   // L1->L0 之后
constexpr bool VC_SYNC_MAD   = false;   // mad 之后
constexpr bool VC_SYNC_FIX   = false;  // fixpipe 之后
//  ★候选: 把屏障由"每块一次"挪到"每 chunk 一次" (j 循环之后)。
//    依据: QK 与 PV **分时共用 L0A/L0B** (vc_l1l0_map.h 的 staged_l0a = 2*max(qk_a, pv_a)),
//    _sync 版每段自带同步故循环结束必已排空; 流水版则可能在最后一块的 mad/fix 还在跑时,
//    PV 相已开始往同一组 L0 槽装数据。若成立, 循环后单屏障就够, 且 CHUNK=4 故屏障少四倍。
constexpr bool VC_SYNC_CHUNK_END = true;
//  ---- PV 相的软件流水 (F209) ----------------------------------------------
//  ★两侧测量: AIC 是关键路径且几乎从不等待 (等 AIV 仅 0.2%), 而它每 chunk 的时间
//    被 pv_issue 吃掉 **80.6%** —— 因为 mm_pv_acc 全用 _sync 变体, 每步自等自己。
//  false 走原来的 mm_pv_acc (保留以便 A/B), true 走四段流水。
constexpr bool VC_PV_PIPE = true;

//  ---- CubeV2: 按 chunk 分阶段的手写 cube ------------------------------------
//  ★与旧 Cube 并存, **本轮尚未接进 RunAttention** —— 先让它被真正代码生成并过 lint,
//    下一轮再与向量侧一起接。这样每一笔提交仓库都是绿的。
//
//  结构 (依据 F186/F188/F189, 三件事必须同时落地, 拆开做任一件都会退):
//    几何 M=64 x N=128  —— 使 L0 各缓冲半满可双缓冲 (断言在 vc_l1l0_map.h)
//    按 chunk 分阶段     —— 同一时刻只有一相占 L0, 故可整块用; 旧结构逐 tile 交替时
//                          L0B 需 2x(32+64)=192 KB > 64 KB, 两相物理上不能共存
//    O 在 L0C 跨 chunk 累加 —— O 只在 chunk 末搬出一次。N 由 256 减半会使 O 字节翻倍
//                          (实测 N=128 单独上慢 2.03 倍), CHUNK=4 把它压回 37.6 GB
//
//  ★错位一个 chunk, 不是提示里那种"相1 相2 严格交替"。
//    理由就是 F189 自己的规则: 严格交替会让 AIC 与 AIV 互等, 重叠比现状更差。
//    代价是 S/P 各要两个 chunk 组 (workspace 448 KB/核, 基线 256 KB/核, 24 核共 10.5 MB)。
template <uint32_t MT, bool Profile>
class CubeV2 {
public:
    __aicore__ inline void RunCube(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR ws,
                                   GM_ADDR trace, const Tiling& t) {
        using namespace vc910::cube2;
        //  ★本类一律用 map::NBLK 与 map::CHUNK, **不用 vc_tiling.h 的 N** ——
        //    旧 Cube 仍是 N=256, 两者并存期间混用会静默取错值。
        constexpr uint32_t NB = map::NBLK;
        constexpr uint32_t CHUNK = map::CHUNK;
        static_assert(MT == map::MBLK, "CubeV2 的查询块必须与地址表的 MBLK 一致");
        const uint32_t core = GetBlockIdx();
        //  ---- AIC 侧相位探针 (role 0) -----------------------------------------
        //  ★此前 cube 侧一个标记都没有, 探针的 role 0 全是空记录 —— 而当前的核心问题
        //    正是"AIC 与 AIV 谁在等谁", 缺了一半的账。相位与 AIV 侧同口径:
        //    全部用**不插屏障**的 Mark (只读 SYS_CNT, 单条 MOV), 且把**等待**单列。
        //    key 取法与 AIV 侧一致, 故同一 chunk 的两侧可以对齐比。
        PhaseProbe<Profile> probe;
        probe.Init(trace, core, 0);
        auto *base = reinterpret_cast<__gm__ uint8_t*>(ws) + uint64_t(core) * t.workspace_bytes;
        //  workspace 布局 (与 vc_tiling_host.h 的 WorkspaceBytesV2 是同一个量的两处写法,
        //  改一处必须改另一处)。S/P 各两个 chunk 组供错位, O 双槽。
        auto *sg = reinterpret_cast<__gm__ int32_t*>(base);
        auto *pg = reinterpret_cast<__gm__ half*>(base + 2 * CHUNK * MT * NB * 4);
        auto *og = reinterpret_cast<__gm__ float*>(base + 2 * CHUNK * MT * NB * 4
                                                        + 2 * CHUNK * MT * NB * 2);
        auto *qg = reinterpret_cast<__gm__ int8_t*>(q);
        auto *kg = reinterpret_cast<__gm__ int8_t*>(k);
        auto *vg = reinterpret_cast<__gm__ half*>(v);

        const uint32_t tiles  = t.kpad / NB;
        const uint32_t chunks = (tiles + CHUNK - 1) / CHUNK;
        //  ---- 内核级 PresetFlag ------------------------------------------------
        //  ★结构照 Laser 的 CubeForward::PresetFlag/ClearFlag: 入口**一次**把全部
        //    "缓冲可覆写"的许可发出来, 循环里每一轮**先认领再释放**, 出口**一次**全收回。
        //  ★此前是自创的"每 chunk 预置 + 每 chunk 排空", 且序幕是**直接 set** 许可 ——
        //    相当于在上一次启动遗留的状态之上**追加**, 于是形状一变, 第一次调用就错
        //    (实测: 预跑 4 个形状后第 1 次 RMSE 1.0, 第 2、3 次恢复 1.84e-6 ——
        //     典型的"上一次的陈旧标志被这一次吸收掉"形态)。
        //  ★每个 chunk 的净收支因此恒为零, 不再需要排空循环。
        using namespace vc910::map;
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1A_PING)));
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1A_PONG)));
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1B_PING)));
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1B_PONG)));
        //  PV 的 P/V 双缓冲四个槽 (同在 MTE2<->MTE1 对上, 号 4..7)
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVP_PING)));
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVP_PONG)));
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVV_PING)));
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVV_PONG)));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0A_PING)));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0A_PONG)));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0B_PING)));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0B_PONG)));
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(uint32_t(EV_L0C_PING)));
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(uint32_t(EV_L0C_PONG)));
        for (uint64_t task = core; task < t.tasks; task += t.cores) {
            const uint64_t head = task / (t.qpad / MT);
            const uint32_t qr   = (task % (t.qpad / MT)) * MT;
            auto *qbase = qg + head * uint64_t(t.qpad) * t.dim + uint64_t(qr) * t.dim;
            auto *kbase = kg + head * uint64_t(t.kpad) * t.dim;
            auto *vbase = vg + head * uint64_t(t.kpad) * t.dim;
            //  ---- Q 常驻 L1: 每**任务**装一次, 不随键块重复读 -------------------
            //  ★Q 在一个任务的全部键块上都相同; 原先它与 K 合在一次 gm2l1 里,
            //    于是被重读了 tiles 次 (37741^2/28 头上约 40 GB = GM_to_L1 的 12%)。
            //  旗标: 认领"L1_Q 可覆写"(MTE1->MTE2), 装完发"数据就绪"(MTE2->MTE1),
            //    在 MTE1 上等一次即可 —— 任务内无人覆写它, 故每块不必再等。
            //    任务末归还可覆写许可, 收支与内核级 Preset/Clear 仍恒为零。
            WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1A_PING)));
            qk_gm2l1_q(qbase, uint16_t(t.dim), uint16_t(t.dim));
            SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(uint32_t(EV_L1A_PING)));
            WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(uint32_t(EV_L1A_PING)));
            uint32_t outstanding = 0;
            for (uint32_t c = 0; c <= chunks; ++c) {
                probe.Begin(task, c ? (c - 1) * CHUNK * NB : 0, t);
                //  相 1: 本 chunk 的全部 QK, 末尾只置一次 S_READY
                if (c < chunks) {
                    const uint32_t slot = (c & 1) * CHUNK;
                    const uint32_t base_i = c * CHUNK;
                    const uint32_t n_in = (tiles - base_i) < CHUNK ? (tiles - base_i) : CHUNK;
                    //  ---- QK 的软件流水 (本步只动重叠, 流量一字不改) --------------
                    //  ★依据 F191: 四流水占比之和只有 0.96 (完全串行), 窗口/binding 2.322,
                    //    而两条流量指标已达预期 —— 慢的是重叠不是流量。
                    //  结构: 第 j+1 块的 GM->L1 发在第 j 块的 L1->L0 / mad / fixpipe **之前**,
                    //    各 pipe 由 notify/wait 串起依赖后自行并行; 重叠即由此而来。
                    //  ★同步用 AscendC 的 SetFlag/WaitFlag<HardEvent::X_Y> —— pipe 对由**模板参数**
                    //    静态检查, 比裸 set_flag 更安全, 且本文件的向量侧早已在用这套。
                    //    事件号的类型是**作用域枚举 event_t** (CANN 自己写 event_t::EVENT_ID0);
                    //    运行时取值用 static_cast<event_t>(...)。
                    //    ★F183 当时试的是 uint8_t 与 **AscendC::event_t** —— 后者是命名空间限定的
                    //      另一个东西, 不是这里要的全局 event_t, 所以三次构建都没对上。
                    //  事件号按 (from,to) pipe 对分作用域, 取值见 vc_l1l0_map.h 的 EV_GROUPS。
                    //  序幕: **先认领** PING 槽再装 —— 不是直接 set 许可 (见入口处的说明)。
                    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1B_PING)));
                    qk_gm2l1_k(1, kbase + uint64_t(base_i) * NB * t.dim,
                               uint16_t(t.dim), uint16_t(t.dim));
                    if (VC_SYNC_GM2L1) asc_sync();
                    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(uint32_t(EV_L1B_PING)));
                    for (uint32_t j = 0; j < n_in; ++j) {
                        //  ★j 为偶数用 PING, 奇数用 PONG —— 必须与上面序幕装的那一槽一致。
                        //    初版写成 `cur = j & 1`, 于是 j=0 得 cur=0 即 PONG, 而序幕装的是 PING:
                        //    序幕 notify 了 PING, 第 0 轮却在等 PONG, **直接死锁**
                        //    (实测: 141 例的第 1 例就卡住不动)。
                        //    ★这类错 detsync_lint 的 H7 看不见 —— 旗标**次数**是相等的, 错的是**配对**。
                        const uint32_t cur = (j & 1) ? 0u : 1u, nxt = cur ^ 1u;
                        //  ★A 侧已常驻, 不再有 ev_l1a_*; 只有 K 随块交替。
                        const uint32_t ev_l1b_c = cur ? EV_L1B_PING : EV_L1B_PONG;
                        const uint32_t ev_l1b_n = nxt ? EV_L1B_PING : EV_L1B_PONG;
                        const uint32_t ev_l0a = cur ? EV_L0A_PING : EV_L0A_PONG;
                        const uint32_t ev_l0b = cur ? EV_L0B_PING : EV_L0B_PONG;
                        const uint32_t ev_l0c = cur ? EV_L0C_PING : EV_L0C_PONG;
                        //  ★先发下一块的搬运 —— 它与本块的 MTE1/M/FIX 并行, 这是收益来源
                        if (j + 1 < n_in) {
                            WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(ev_l1b_n));
                            qk_gm2l1_k(nxt, kbase + uint64_t(base_i + j + 1) * NB * t.dim,
                                       uint16_t(t.dim), uint16_t(t.dim));
                            if (VC_SYNC_GM2L1) asc_sync();
                            SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(ev_l1b_n));
                        }
                        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(ev_l1b_c));
                        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0a));
                        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0b));
                        qk_l12l0(cur, uint16_t(t.dim));
                        if (VC_SYNC_L12L0) asc_sync();
                        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(ev_l1b_c));
                        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0a));
                        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0b));

                        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0a));
                        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0b));
                        WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(ev_l0c));
                        qk_mad(cur, uint16_t(t.dim));
                        if (VC_SYNC_MAD) asc_sync();
                        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0a));
                        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0b));
                        SetFlag<HardEvent::M_FIX>(static_cast<event_t>(ev_l0c));

                        WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(ev_l0c));
                        qk_fix(cur, sg + uint64_t(slot + j) * MT * NB, uint16_t(NB));
                        if (VC_SYNC_FIX) asc_sync();
                        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(ev_l0c));
                    }
                    probe.Mark(0);   // qk_issue: 本 chunk 全部 QK 的发射
                    //  ★QK 相与 PV 相之间的交接: 两相分时共用 L0A/L0B, 故 PV 开始装载前
                    //    本 chunk 的 QK 必须全部落定。
                    if (VC_SYNC_CHUNK_END) asc_sync();
                    CrossCoreSetFlag<2, PIPE_FIX>(S_READY);
                    probe.Mark(1);   // chunk_drain: chunk 末全屏障 (真排空, 故是墙上时间)
                }
                if (c == 0) { probe.Finish(); continue; }   // 第 0 轮只发不消费 (错位一拍)
                //  相 2: 消费第 c-1 个 chunk —— 全部 PV 在 L0C 累加, 只在末尾搬出一次
                CrossCoreWaitFlag(P_READY);
                probe.Mark(2);   // wait_p: 等 AIV 把 P 算出来
                if (outstanding >= 2) { CrossCoreWaitFlag(O_USED); --outstanding; }
                probe.Mark(3);   // wait_o_used: 等 AIV 用掉上上个 O (背压)
                //  ★QK 相与 PV 相之间的 L0C 交接 —— 少了它就是本轮那个数值错的成因。
                //    PV 的累加器是 L0C_PING (= L0C_ACC), 而 QK 的 j 为偶数时也写 L0C_PING。
                //    把每 chunk 的排空挪到内核级之后, 这道交接**没了**: PV 的 mad 可以在
                //    QK 最后一次 fixpipe 还在读 L0C 时就开始写它。
                //    ★订正: 此处原有一段"四段后各加全屏障仍错"的推理, 那是在**污染卡**上的
                //      观测 (F194), 已被 F195 推翻 —— 四段拆分与槽位本身是正确的。
                //    按 claim/release 补: 认领 L0C_PING, 用完释放, 内核级 Preset/Clear 的收支不变。
                WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(uint32_t(map::EV_L0C_PING)));
                probe.Mark(6);   // claim_l0c: 认领 L0C_ACC 的等待, 原先混在 pv_issue 里
                const uint32_t pc = c - 1;
                const uint32_t pslot = (pc & 1) * CHUNK;
                const uint32_t pbase = pc * CHUNK;
                const uint32_t n_pv = (tiles - pbase) < CHUNK ? (tiles - pbase) : CHUNK;
                if (!VC_PV_PIPE) {
                    for (uint32_t j = 0; j < n_pv; ++j) {
                        auto *pp = pg + uint64_t(pslot + j) * MT * NB;
                        auto *vv = vbase + uint64_t(pbase + j) * NB * t.dim;
                        //  ★只有 chunk 内第一块把 L0C 置零; 其余累加 —— 这正是 O 只搬一次的机制。
                        if (j == 0) mm_pv_acc<true>(pp, vv, uint16_t(t.dim), uint16_t(NB), uint16_t(t.dim));
                        else        mm_pv_acc<false>(pp, vv, uint16_t(t.dim), uint16_t(NB), uint16_t(t.dim));
                    }
                } else {
                    //  ---- PV 的四段流水 (与 QK 同形) ------------------------------
                    //  序幕: **先认领** PING 槽再装 —— 与 QK 同一教训: 直接 set 许可
                    //  等于在上一次启动的遗留状态之上追加。
                    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVP_PING)));
                    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVV_PING)));
                    pv_gm2l1(1, pg + uint64_t(pslot) * MT * NB,
                             vbase + uint64_t(pbase) * NB * t.dim,
                             uint16_t(t.dim), uint16_t(NB), uint16_t(t.dim));
                    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(uint32_t(EV_PVP_PING)));
                    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(uint32_t(EV_PVV_PING)));
                    for (uint32_t j = 0; j < n_pv; ++j) {
                        //  ★槽配对与 QK 同: j 偶用 PING, 奇用 PONG, 必须与序幕装的那一槽一致
                        //    (写成 cur = j & 1 会让 j=0 取 PONG 而序幕装的是 PING, 直接死锁;
                        //     H7 看不见 —— 次数相等, 错的是配对)。
                        const uint32_t cur = (j & 1) ? 0u : 1u, nxt = cur ^ 1u;
                        const uint32_t ev_p_c = cur ? EV_PVP_PING : EV_PVP_PONG;
                        const uint32_t ev_v_c = cur ? EV_PVV_PING : EV_PVV_PONG;
                        const uint32_t ev_p_n = nxt ? EV_PVP_PING : EV_PVP_PONG;
                        const uint32_t ev_v_n = nxt ? EV_PVV_PING : EV_PVV_PONG;
                        const uint32_t ev_l0a = cur ? EV_L0A_PING : EV_L0A_PONG;
                        const uint32_t ev_l0b = cur ? EV_L0B_PING : EV_L0B_PONG;
                        if (j + 1 < n_pv) {
                            WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(ev_p_n));
                            WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(ev_v_n));
                            pv_gm2l1(nxt, pg + uint64_t(pslot + j + 1) * MT * NB,
                                     vbase + uint64_t(pbase + j + 1) * NB * t.dim,
                                     uint16_t(t.dim), uint16_t(NB), uint16_t(t.dim));
                            SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(ev_p_n));
                            SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(ev_v_n));
                        }
                        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(ev_p_c));
                        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(ev_v_c));
                        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0a));
                        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0b));
                        pv_l12l0(cur, uint16_t(t.dim));
                        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(ev_p_c));
                        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(ev_v_c));
                        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0a));
                        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0b));

                        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0a));
                        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(ev_l0b));
                        //  ★只有第一块把 L0C 置零; 其余累加 —— O 只搬一次的机制不变。
                        if (j == 0) pv_mad<true>(cur, uint16_t(t.dim));
                        else        pv_mad<false>(cur, uint16_t(t.dim));
                        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0a));
                        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(ev_l0b));
                    }
                    //  L0C 的累加全部落定之后才能 fixpipe 搬出 (原先由 asc_mmad_sync 保证)。
                    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(uint32_t(map::EV_L0C_PING)));
                    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(uint32_t(map::EV_L0C_PING)));
                }
                pv_flush(og + uint64_t(pc & 1) * MT * t.dim, uint16_t(t.dim), uint16_t(t.dim));
                //  释放 L0C_PING, 与上面的认领配对 (pv_flush 用的是 _sync 变体, 已自带排空)。
                SetFlag<HardEvent::FIX_M>(static_cast<event_t>(uint32_t(map::EV_L0C_PING)));
                CrossCoreSetFlag<2, PIPE_FIX>(O_READY);
                ++outstanding;
                probe.Mark(4);   // pv_issue: PV 累加 + pv_flush + O_READY 的发射
                probe.Finish();
            }
            while (outstanding) { CrossCoreWaitFlag(O_USED); --outstanding; }
            //  归还 L1_Q 的"可覆写"许可, 与任务头的认领配对。
            SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1A_PING)));
        }
        //  ---- 内核级 ClearFlag: 把入口发出的许可全部收回 ----------------------
        //  ★少收一个就会留给**下一次启动**, 表现为"形状一变第一次调用就错"。
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1A_PING)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1A_PONG)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1B_PING)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_L1B_PONG)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVP_PING)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVP_PONG)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVV_PING)));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(uint32_t(EV_PVV_PONG)));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0A_PING)));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0A_PONG)));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0B_PING)));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(uint32_t(EV_L0B_PONG)));
        WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(uint32_t(EV_L0C_PING)));
        WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(uint32_t(EV_L0C_PONG)));
    }
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
        CubeV2<TileM, Profile> cube;
        cube.RunCube(q, k, v, ws, trace, t);
        (void)qt; (void)vt;   // 手写路径不读 CANN 的 TCubeTiling
    }
    if ASCEND_IS_AIV {
        Vector<TileM, Profile> vector;
        vector.RunVector(qs, ks, vs, lut, output, ws, trace, t);
    }
}

extern "C" __global__ __aicore__ void vc910_forward(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<64, false>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

extern "C" __global__ __aicore__ void vc910_forward_m64(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<64, false>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

extern "C" __global__ __aicore__ void vc910_profile(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<64, true>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

extern "C" __global__ __aicore__ void vc910_profile_m64(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qs, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR lut, GM_ADDR output, GM_ADDR ws, GM_ADDR qt, GM_ADDR vt, GM_ADDR tiling, GM_ADDR trace) {
    RunAttention<64, true>(q, k, v, qs, ks, vs, lut, output, ws, qt, vt, tiling, trace);
}

//  ---- CubeV2 的**数值**探针 -------------------------------------------------
//  ★不是编译探针。公开 API 有声明, 编译器能查**类型**, 但 repeat / src_stride /
//    dst_gap / dst_frac_stride 这几个参数的**语义**仍要确证 —— CANN 树里没有这几个
//    函数的调用样例 (查过 c_api 全树与 impl 层)。语义只能从内建推, 而"推"正是
//    F183 里烧掉三次构建的失败模式。
//  对策: 在固定小输入上单独跑这三个原语, 结果写 GM, 主机侧与 NumPy 参考逐元素比。
//  猜错在**一次构建 + 一次运行**里暴露, 而不是污染整个内核后去 141 个用例里排查。
//  d 固定 128 (基准形状), 不读 tiling —— 探针要尽量少的活动部件。
extern "C" __global__ __aicore__ void vc910_cube_v2_probe(
        GM_ADDR s_out, GM_ADDR o_out, GM_ADDR r1_out, GM_ADDR r2_out,
        GM_ADDR q, GM_ADDR k, GM_ADDR pmat, GM_ADDR v, GM_ADDR dmat, GM_ADDR obase) {
    if ASCEND_IS_AIC {
        constexpr uint16_t D = 128;
        using namespace vc910::cube2;
        //  QK: S[64][128] int32 = Q[64][128] · K[128][128]^T
        mm_qk(reinterpret_cast<__gm__ int32_t*>(s_out),
              reinterpret_cast<__gm__ int8_t*>(q),
              reinterpret_cast<__gm__ int8_t*>(k),
              D, D, D, vc910::map::NBLK);
        //  PV: 连做两次累加且用同一对 P/V —— 结果必须恰好是单次的两倍。
        //  ★这一条专门验 c_matrix_init_val=false 真的在**累加**而不是覆写:
        //    若它其实是覆写, 结果会等于单次而不是两倍, 主机侧一眼看出。
        //    累加正是 chunk 内跨 key 块只搬出一次 O 的全部机制 (F186/F189)。
        auto *pp = reinterpret_cast<__gm__ half*>(pmat);
        auto *vv = reinterpret_cast<__gm__ half*>(v);
        mm_pv_acc<true>(pp, vv, D, vc910::map::NBLK, D);
        mm_pv_acc<false>(pp, vv, D, vc910::map::NBLK, D);
        pv_flush(reinterpret_cast<__gm__ float*>(o_out), D, D);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

        //  ---- 重标定 (fp32 对角阵乘) 的参数语义探针 ----------------------------
        //  ★这三条 fp32 搬运 (nd2nz b32s / l12l0a / l12l0b_trans) 的 repeat /
        //    src_stride / dst_frac_stride 语义是照着本机 MindIE Laser 的 matmul_fp32
        //    抄的, 但 Laser 的几何是 M=K=N=128 而我们是 M=K=64, N=d ——
        //    "照抄一个不同几何的调用" 与 "推参数" 在风险上同级, 故同样先数值坐实。
        //  ★l12l0b_trans 映射的 load_cbuf_to_cb_transpose 与 PV 用的
        //    load_cbuf_to_cb(transpose=true) 是**两条不同指令** (见 mm_pv_acc 处的注记,
        //    搞混过一次, 报 507015), 这里走的是前者, 与 Laser 同。
        auto *dd = reinterpret_cast<__gm__ float*>(dmat);
        auto *ob = reinterpret_cast<__gm__ float*>(obase);
        //  r1 = D x O_base —— 只验搬运与 mmad 的 fp32 通路本身。
        resc_gm2l1(dd, ob, D); asc_sync();
        resc_l12l0(D);         asc_sync();
        resc_mad(D);           asc_sync();
        pv_flush(reinterpret_cast<__gm__ float*>(r1_out), D, D);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        //  r2 = D x O_base + P x V —— 正是新内核的结构: 重标定打底, PV 以 Init=false 累上去。
        resc_gm2l1(dd, ob, D); asc_sync();
        resc_l12l0(D);         asc_sync();
        resc_mad(D);           asc_sync();
        mm_pv_acc<false>(pp, vv, D, vc910::map::NBLK, D);
        pv_flush(reinterpret_cast<__gm__ float*>(r2_out), D, D);
    }
}

//  ---- fixpipe 逐通道反量化 (VDEQF16) 的**数值**探针 ---------------------------
//  ★要确证的是三件事, 每件都有一个会让它无声走错的替代解释:
//    (1) **那条 uint64 的编码**。两处本机文档互相矛盾 (见 vc_cube_v2.h 的注记):
//        模拟器说 bits[0:31] 是 fp32 位模式, c220 的标量支注释说 bits[31:13]。
//        主机侧按"fp32 位模式零扩展"构造; 若这条是错的, 结果会整体偏掉。
//    (2) **scale 是按哪个方向施加的**。若主机给的每通道 scale 全相等,
//        "逐通道"与"标量"给出同一答案 —— 那样的样例钉不住契约 (E6)。
//        故探针的 scale **逐列各不相同**, 且特意让第 0 列与第 1 列差一个量级。
//    (3) **舍入到 fp16 的方式**。与 NumPy 的 float32->float16 (就近偶数) 逐位比。
//  ★两条出口读同一份 L0C, 所以 int32 那份既是参考也是"FPC 没破坏别的东西"的旁证。
//  ★几何与 vc910_cube_v2_probe 一致 (M=64, N=128, d=128), 活动部件尽量少。
extern "C" __global__ __aicore__ void vc910_fixpipe_vdeq_probe(
        GM_ADDR s_i32, GM_ADDR s_f16, GM_ADDR q, GM_ADDR k, GM_ADDR deq) {
    if ASCEND_IS_AIC {
        constexpr uint16_t D = 128;
        using namespace vc910::cube2;
        mm_qk_two_exits(reinterpret_cast<__gm__ int32_t*>(s_i32),
                        reinterpret_cast<__gm__ half*>(s_f16),
                        reinterpret_cast<__gm__ uint64_t*>(deq),
                        reinterpret_cast<__gm__ int8_t*>(q),
                        reinterpret_cast<__gm__ int8_t*>(k),
                        D, D, D, vc910::map::NBLK);
    }
}

//  ---- 向量侧的**数值**探针: 确证 WholeReduceSum 的形参语义 (E33 的前置) -------
//  ★为什么要它 (F183 定的次序): `WholeReduceSum` 的 dstRepStride / srcBlkStride /
//    srcRepStride 虽有本机权威样例 (CANN 自己的 reduce_sum_v220_impl.h 的 ReduceSumArCompute:
//    `WholeReduceSum<T,false>(dst[i*248], src[i*248*padLast], MASK_PLACEHOLDER, 248, 1, 1,
//     padLast/elePerBlk)`), 但那里的几何是 248 个 repeat, 我们是 ROWS 个; 且 **dst 的排布**
//    (连续? 按 32B 块跳?) 从声明里看不出来。猜错在 141 个用例里排查, 远贵于在这里一次测掉。
//  ★取值刻意用小整数: fp32 内精确, **任何求和顺序都同果**, 故"不逐位相同"只可能是形参错,
//    不会是舍入。同时输出现行 ReduceSum 的结果作对照 —— 两者都要与 NumPy 逐位相同。
extern "C" __global__ __aicore__ void vc910_vec_reduce_probe(
        GM_ADDR x, GM_ADDR out_new, GM_ADDR out_old, GM_ADDR out_after) {
    if ASCEND_IS_AIV {
        constexpr uint32_t R = 32, NN = 128;   // 与生产的 ROWS / N 同
        TPipe pipe;
        TBuf<TPosition::VECCALC> xb, tb, ob;
        pipe.InitBuffer(xb, R * NN * 4);
        pipe.InitBuffer(tb, 8192);             // reduce 的暂存 (本机 shape{32,128} 实测需 8192)
        pipe.InitBuffer(ob, 3 * R * 4);
        auto sc = xb.Get<float>();
        auto tmp = tb.Get<uint8_t>();
        auto o = ob.Get<float>();
        GlobalTensor<float> xg, ng, og, ag;
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), R * NN);
        ng.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out_new), R);
        og.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out_old), R);
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out_after), R);

        DataCopy(sc, xg, R * NN);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        //  对照组: 现行写法。isReuseSource=false 故**不破坏** sc, 可以先跑。
        uint32_t shape[2] = {R, NN};
        ReduceSum<float, Pattern::Reduce::AR, false>(o[R], sc, tmp, shape, true);
        PipeBarrier<PIPE_V>();

        //  候选组: 折 128 -> 64 (原地), 再一条 WholeReduceSum。
        //  ★Add 的 repeat 参数: 每个 repeat 覆盖 64 个 fp32, 行间跨 NN/8 个 32B 块。
        Add(sc[0], sc[64], sc[0], uint64_t(64), R,
            BinaryRepeatParams(1, 1, 1, NN / 8, NN / 8, NN / 8));
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(o[0], sc, 64, R, 1, 1, NN / 8);
        PipeBarrier<PIPE_V>();

        //  ★第三条判据: `WholeReduceSum` 带 isSetMask=true 会**自己设 mask**, 而底层这条
        //    与高层 `ReduceSum` 不同 —— 后者收尾有 `SetMaskNorm(); ResetMask();`
        //    (reduce_sum_v220_impl.h:293-294), 前者**没有**。
        //    若它把 mask 留在 64, 后续的全宽操作就会被悄悄截断。
        //    这里故意**不**手动恢复, 先测它到底需不需要:
        //    全宽写 1.0 再按旧路径求行和, 每行必须恰为 NN = 128.0; 若 mask 粘住则会是别的数。
        Duplicate(sc, 1.0f, R * NN);
        PipeBarrier<PIPE_V>();
        ReduceSum<float, Pattern::Reduce::AR, false>(o[2 * R], sc, tmp, shape, true);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(ng, o[0], R);
        DataCopy(og, o[R], R);
        DataCopy(ag, o[2 * R], R);
    }
}

//  ★shape_probe 已删: CubeV2 现在由 RunAttention 真正派发, 不再需要额外入口驱动代码生成。
