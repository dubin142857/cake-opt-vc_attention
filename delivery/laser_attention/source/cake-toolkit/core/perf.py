"""msprof 性能诊断 — 基础能力, 算子无关. 反馈信号明确 + 归责到 pipe.

看板只给"慢多少"(wall-clock), 本工具给"慢在哪": 用 msprof op 采 pipe 分解, 判读瓶颈,
并对两版 (baseline/candidate) 做 pipe 级 diff, 指出确定性/优化的代价具体落在哪条 pipe.

msprof op 真实产物布局 (实测 CANN 9.1, 非旧版 op_summary):
    <out>/OPPROF_*/<kernel>/<launch_id>/PipeUtilization_*.csv   # 每 block 一行, sub_block_id ∈ {cube0,vector0,vector1}
                                        /OpBasicInfo_*.csv       # Task Duration(us) / Block Dim
    cube 行带 aic_* ratio, vector 行带 aiv_* ratio. launch 0 视为冷启动丢弃.
判读 (ratio 0..1, 越高该 pipe 越占): mte2 高=搬入瓶颈; mte3 高=搬出瓶颈; cube/vec 高=计算;
    scalar 高=地址/控制/同步. dual-loop 收敛: 一次 msprof 实测 > 10 轮 prose 推理.
更全版本见 vendor/perf/code-performance-advisor/scripts/analysis_engine/profiling_extractor.py.
"""
from __future__ import annotations

from ._refs import Ref, print_refs

#  ★本模块此前**没有**判据出处表, 而模块开头就记着它自己出过错 (冲突比折算分母搞错低估 3 倍)。
#    一条不能被证伪的判据不是方法, 是断言。这里把 --signals 里那几条会驱动决策的判据登记进来。
REFS = {
    "PF_region_scoped_pmu_asc_prof": Ref(
        what=[
            "**msprof 给的是整核聚合值**; 要把 PMU 采集收窄到一段代码, 厂商接口是",
            "`__asc_aicore::asc_prof_start()` / `asc_prof_stop()` (头 `utils/debug/asc_time.h`),",
            "函数体即 `bisheng::cce::metrics_prof_start/stop()`。**本机 (arch 2201) 可用**。",
            "用途举例: 把 `CrossCoreWaitFlag` 排除在区间外, 拿到不含跨核等待的 scalar 时间。",
            "★**它没有任何编译期开关** —— `asc_time_stamp` 有 `ASCENDC_TIME_STAMP_ON`、",
            "  `TRACE_*` 有 `ASCENDC_TRACE_ON`, 唯独这一对只要 arch 对就无条件编进去。",
            "  官方文档的'约束说明: 无'把这件事漏了。交付前必须从源码摘掉或自套一层宏。",
            "★**不要用 `TRACE_START/STOP` 代替它**: 在 2201 上 `TRACE_START_1(apid)` 展开成",
            "  `set_lpcnt(...)` + `prof_mark_event()`, 而后者发 `NOP_BAR` 到 V/M/MTE1/MTE2/MTE3",
            "  **五条流水** —— 每个打点一次全流水屏障, 恰好改掉'标量与等待如何重叠'这件要测的事。",
            "  文档也写明 `TRACE_*` 是**仿真器**场景, `asc_prof_*` 才是 msOpProf 上板场景。",
        ],
        status="measured",
        evidence=[
            "★实验设计里有两处是本仓教训直接换来的, 缺一条结论都不成立:",
            " (1) **先比对三个包的内核 .o 指纹, 三者不互不相同就判本轮无效**。",
            "     第一轮正是三个包逐字节相同 (都是 796ce6b84616) 而读数 98.1%/99.7%",
            "     '看起来合理', 几乎被写成结论 —— 成因是 opbuild 的 --incremental 根本不同步",
            "     源码 (已修, 见该模块的 _assert_tree_matches_source)。本轮三者指纹分别为",
            "     796ce6b84616 / 47e2e28d8988 / 7e36c296985f, 守卫通过。",
            " (2) **wide 作负例控制**: 两个变体都含 asc_prof 调用, 所以 narrow/wide 之差",
            "     隔离的是**区间范围**而不是'有没有调用'。",
            "2026-10-30 两份来源交叉核对 (官方文档 asc.gitcode.com 的 tuning_interface 各页",
            " + 本机 CANN 9.1.0-beta.1 的 asc 头文件), 并用 ccec 编译验证:",
            "`asc_time.h` 对 impl 的门控是 `__NPU_ARCH__ == 2201 || 3510`, 2201 即 910B2;",
            "空核加这一对后 `.text` 由 92 涨到 132 字节 (每个约 20 字节, 无屏障宏)。",
            "★文档与实现对不上的三处 (均已编译/读码验证):",
            " (1) `TRACE_START(pipe, apid)` 两参形式在 2201 上 `#define TRACE_START_2(pipe,idx)`",
            "     **展开成空**, 而文档把它与'A2 支持'并列; 它转调的 `asc_mark_stamp` 自己的页面",
            "     却写着 'Atlas A2 系列产品: 不支持' —— **文档自相矛盾**, 头文件判了这一局;",
            " (2) `clock()` 文档写 A2 支持, 而本机只声明在 `namespace __asc_simt_vf` 且门控为",
            "     arch 3510/5102, 编译直接报 `use of undeclared identifier 'clock'`。",
            "     2201 上的等价物是 `AscendC::GetSystemCycle()` (`core/kcount` 用的就是它);",
            " (3) `asc_time_stamp` 的 `desc_id`: **文档**说 [0,0xffff] 是 Ascend C 内部预留、",
            "     自定义要用 >0xffff (头文件没这条); 而**头文件**里 `check_ringbuf_space()`",
            "     不足时直接 return、**记录静默丢弃** (文档没这条)。两边各缺一半。",
            "★`asc_time_stamp` 是官方版的相位打点, 与本仓 `core/kcount` 同类: 前者写 GM 环形",
            "  缓冲 (满了悄悄丢, 丢了的现象与'那段没执行'一样), 后者写 workspace 固定槽位。",
            "★★**2026-10-30 端到端验了: 在这条路径上它不收窄**。",
            "  三点实验 (`mat_mul_v3` 的 `MatmulBaseKernel::Process()`, 各独立全量构建):",
            "    none   不插调用                     aic_time 14.7925us  100%",
            "    narrow 区间只含 2 条 SetHF32 指令     aic_time 14.4492us  **97.7%**",
            "    wide   区间覆盖整个 Process()        aic_time 14.7447us  99.7%",
            "  narrow 的区间约占整核 26000 周期的**万分之一**, 若收窄成立应塌到接近零;",
            "  实测与基线无实质差别, 各列 (cube/mte1/mte2/fixpipe/scalar) 偏差全在 0.2~2.3%。",
            "  ★`wide` 是**负例控制**, 它 99.7% 说明'调用存在'本身不改读数 —— 只比 none/narrow",
            "    的话, 那 2.3% 可能被误读成'有一点收窄'。预期效应是四个数量级, 2% 排除得掉。",
            "  **作用域**: `msprof op --application` + `--aic-metrics=PipeUtilization`, arch 2201。",
            "    这正是官方文档说的 msOpProf 上板场景, 不是用错了路。",
            "    没试过的: 别的 aic-metrics 档位、`--config` 模式、以及 950 系列 (3510)。",
        ],
        verified="2026-10-30"),
    "PF16_bpath_b10_ratio_is_not_a_constant": Ref(
        what="B 路径 `Duration` 与 `aicore_time` 的差**不是一个固定比例**: 短核是加性底, "
             "长核才是比例。引用'混用偏 X%%'必须带核长作用域。",
        status="measured",
        evidence=[
            "2026-10-22 跨 173 份 kernel_details.csv / 262 行实测:",
            " · Duration < 10us   : 差 1.24~2.45us (加性底), 比值中位 0.765;",
            " · Duration 10~100us : 差 1.13~2.36us (同一个底), 比值中位 0.908;",
            " · Duration > 10ms   : 差 0.64~2.15ms (约 7%%, 比例性), 比值中位 0.929。",
            "**同一条失真在不同核长上差一个数量级** —— 5us 的核偏 24%%, 93ms 的核偏 1%%。",
            "★唯一判 miss 的那份 (SparseFlashAttentionGrad, 93ms, 比值 0.9903) **不是反例**:",
            "  绝对差仍有 902us, 只是相对差落到 0.99 阈值之上。所以 B10 的 `miss`",
            "  **不读作'没有失真'**, 只读作'这份产物的核够长'。证据串已改成比值与绝对差同报。",
            "★配套: `--bpath-corpus` 跨语料汇总。单份上的 miss 分不开'失真不存在'与",
            "  '这份考不动'; 全 na 的判据等于从没被考过。B11 (_ratio>1) 考得动 40 份",
            "  却一次没命中, 它是**从代码逆向出的断言**而非实测结论。",
            "★极性: B7/B8 是恒等式检验 (miss = 成立), B9/B10/B11 是失真检测 (hit = 在场)。",
            "  汇总第一版用通用判读, 把 72 份上成立的恒等式报成'未被证实' —— 已分开。",
        ],
        verified="2026-10-22"),
    "PF15_file_ageing_is_silent_and_only_dir_shows_it": Ref(
        what="第三条丢数通路 —— 记录落了盘又被 **FileAgeing 删掉**; 它在默认日志级别下"
             "**完全静默**, 唯一证据是 PROF 目录自身的 `slice_N` 前缀型断号。",
        status="measured",
        evidence=[
            "2026-10-22 反汇编 msprof(8142256) 定三个调用点, DlogRecord 的 level 实参",
            "(常量见 aarch64-linux/include/base/log_types.h: DEBUG=0 INFO=1 WARN=2 ERROR=3):",
            " · `RemoveAgeingFile begin, storagedFileSize/overThdSize` 265 行  w1=0 **DEBUG**",
            " · `remove fileName:…, filePath:…, fileCount_`             257 行  w1=0 **DEBUG**",
            " · `remove file:%s failed`                                 245 行  w1=3 ERROR",
            "**成功的删除是 DEBUG, 只有删除失败才是 ERROR** —— 本机 ASCEND_GLOBAL_LOG_LEVEL",
            "未设, 全机 80683 份日志里这三个串**零命中**, 而淘汰确实发生过 (见下)。",
            "即: 日志里关于淘汰的唯一记录, 恰好是淘汰**没发生**的时候。",
            "样本外检验 (两份带 --storage-limit 的真产物, 2026-10-21 采):",
            " · age_big1 (200MB): 4 个通道断号, **全前缀型 ∧ 全 aging**;",
            "   additional.tensor_info 缺 0..33 (现有最大 58), api_event 缺 0..9;",
            " · age_lim400 (400MB): 1 处, 同样全前缀型全 aging;",
            " · age_c3 (200MB 但负载小): **0 处** —— 有限额不等于一定淘汰, 判据没有被配置带偏。",
            "阴性语料: runs/ 与 cases/ 下 39 个 PROF 目录、429 个通道-目录组合, **断号 0**,",
            "且其中含 **195 个 aging 通道** —— 所以'只落在 aging'不是因为没有 aging 通道。",
            "★这同时解释了 prof_inventory 一直在解析却从没解释过的 aging 标记: **可被淘汰**。",
            "★前缀型与中间断号必须分开: 淘汰按最旧先删, 只留前缀型缺口;",
            "  中间缺口是另一回事 (采集中途出错 / 文件被手动删), 处置方向不同。",
        ],
        verified="2026-10-22"),
    "PF14_drop_evidence_is_in_slog_not_prof_dir": Ref(
        what="采集有没有**悄悄少记录**, 证据一个字节都不在 PROF 目录里, 全在昇腾 slog; "
             "且'看不见'与'没丢'必须分开报。",
        status="measured",
        evidence=[
            "2026-10-22 对 libprofapi.so (CANN 9.1.0-beta.1) 反汇编定调用点:",
            " · `total_size_report [<buf>] read count/write count` @ prof_inner_api.cpp:106/109,",
            "   DlogRecord 首参 w0=0x0100001f = PROFILING(31)|RUN 掩码 → 落 run/plog/, INFO 级;",
            " · `Block buffer overflow, [%s] capacity/read/write` @ prof_inner_api.cpp:96,",
            "   w0=0x1f (无 RUN 掩码) w1=3 → 落 debug/plog/, **ERROR 级**。",
            "★两条判据查不同的洞: 溢出时写入方失败返回, 那条记录**从未计入 write count**,",
            "  所以 R 仍等于 W —— 只做计数对账会把丢弃漏掉。",
            "本机全量实测 (80683 个日志文件): total_size_report 22851 行 **R==W 无一例外**,",
            "六类 overflow 串**零命中**; 缓冲区共 5 个 —— api_event/compact/additional/variable",
            "(即 03 的四条 host 通道) + **AicpuBuffer** (此前未进过任何文档)。",
            "★零命中可信的前提是这些串确实会被记下来: ERROR(prof_inner_api.cpp:96, w1=3) 或",
            "  WARNING(prof_cann_plugin.cpp:141, w1=2) 级, 而同模块 INFO 行在 plog 里",
            "  实际可见 (prof_cann_plugin.cpp:270 等), 故级别不构成遮蔽。",
            "★反过来: 扫到 0 行计数时结论是 BLIND 不是 CLEAN —— 已做成两种不同字样的输出。",
        ],
        verified="2026-10-22"),
    "PF13_extcorr_needs_sync_and_sentinel": Ref(
        what="自定义归责单元 (ExternalCorrelation) 的两条硬前提: **pop 前必须同步** (E5) 与 "
             "**末单元需哨兵** (E1)。两条都**静默失效** —— 不报错, 只是归属悄悄变少或为空。",
        status="measured",
        evidence=[
            "2026-10-20 卡 5 实测, 三个单元各 8 次下发:",
            " · 不同步就 pop: 8 次下发只归到 2 个, 极端情形**一条关联记录都不产生** ——",
            "   关联记录是与被关联的活动一起产出的, 下发还在飞时 pop, 记录就永远不会建;",
            " · 加 sync_on_exit 后 8/8/8 全中, 未归属的 10 个 = 哨兵 8 + Cast + 建张量, 判得对;",
            " · 末单元哨兵: 九单元实验里末位恒为 0, 倒序后换成另一个算子仍为 0, 加哨兵后九个全齐。",
            "   FlushAll 不够, synchronize() 不够, flush x3 + 每次沉降 1s 也不够。",
            "另: msptiActivityEnable 返回 MSPTI_SUCCESS **不代表采集起来了** (少 msptiSubscribe 时",
            "七个 kind 全返回 0 而只有 ExternalCorrelation 记录会来)。判采集看收没收到记录, 不看返回码。",
        ],
        verified="2026-10-20"),
    "PF0_msprof_empty_is_refuted": Ref(
        what="「msprof 对 mix 核返回空记录」—— **已被推翻**, 是本仓取数层的文件名模式没对上",
        status="refuted",
        evidence=[
            "2026-09-22 实测: 对 sparse_flash_attention (三个 AIC_ONLY 声明但编成 mix 的核)",
            "跑 msprof op, 产物完整 —— 28 个 block 的 PipeUtilization / ResourceConflictRatio /",
            "MemoryL0 / L2Cache 全在, 解出 cann_bench_sfa_qk / cube / 14.89us,",
            "忙 MTE2 69.2% 标量 47.1% 矩阵 13.6%, 等 矩阵 84% MTE1 83%。",
            "★原结论的来路: core.perf.extract 抛'未找到 PipeUtilization_*.csv' (本机 CANN",
            "  9.1.0-beta.1 产出的是无后缀的 PipeUtilization.csv), 我把'取数层报错'读成",
            "  '硬件没有这个能力', 并据此在整个会话里放弃 msprof, 改用手搓的一次性开关。",
            "★教训不是'下次别读错', 是**取数层找不到时必须报出实际有什么** ——",
            "  见 _what_is_there(); 以及**能左右工具选用的判断必须进这张表**, 只活在",
            "  工作笔记里的判断没有任何证伪把手, 却在起门禁作用。",
        ],
        verified="2026-09-22"),
    "PF1_mspti_preload_blocks_msprof": Ref(
        what="被采集进程 `LD_PRELOAD` 了 `libmspti.so` 时, `msprof op` 产出空目录 —— "
             "下发前必须摘掉它 (`strip_mspti_preload`)",
        status="measured",
        evidence=[
            "msprof 逆向 2026-10-20, 四档实测 (docs/msprof-re/10 §9):",
            "  msprof 单独 -> 出 CSV; MSPTI 单独 -> 200 条记录;",
            "  两者同开 -> MSPTI 80 条而 msprof `No profiling data dumped` + 0 个 CSV;",
            "  **只 LD_PRELOAD 不 start monitor -> 同样 0 个 CSV**。",
            "★第四档是关键: 它把'加载了库'与'开了采集'分开, 肇因因此钉在 **LD_PRELOAD 本身**。",
            "  少这一档就会得出'开 MSPTI 采集会挤掉 msprof'这个错归因, 进而去找怎么让两个",
            "  采集器错开 —— 而正确处理只是摘掉一个环境变量。(flaky F2 单变量条款用在环境变量上)",
            "★互斥是**单向**的: MSPTI 照常收齐全部记录, 被挤掉的是 msprof。",
            "★端到端验过: 同一条命令同一个环境, 加守卫前 0 个 CSV, 加守卫后 15 个。",
            "  本次逆向 15 条 core/ 改动里只有这一条有'关掉它就复现、打开它就不复现'这一对。",
            "★推论: MSPTI 的时间线与 msprof 的 per-pipe PMU 分解**不能在同一次运行里拿**, 要分两次。",
        ],
        verified="2026-10-20"),
    "P1_latency_bound_mutual_wait": Ref(
        what="AIC 与 AIV **各自窗口内**都空 >15% => 互等形态, 端到端不由任何单条流水的忙时决定",
        status="measured",
        evidence="FIAS MLA decode 100k_64P (cases/fused_infer_attention_score/runs/msprof_rect):\n"
                 "  AIC 窗口 566.43 而 cube 忙 443.11 (空 21.6%); AIV 窗口 577.99 而 Σpipe 441.35 (空 23.5%);\n"
                 "  端到端 581.3 ≈ AIV 窗口。三次独立改动各自证实了'单条流水的忙时不是墙':\n"
                 "  KV 搬运量减半 (mte2 忙时 -211us) 只回收 38.16us; 缓冲深度 2->4 收益 0;\n"
                 "  unitFlag 打通 Mmad->fixpipe 收益 0。",
        verified="2026-09-14"),
    "P9_working_set_before_traffic": Ref(
        what="访存绑定时先看**瞬时工作集相对 L2 的位置**, 再谈减流量。"
             "工作集才是自变量, 不是'随机还是连续'; 达成带宽贴着某个标定值也不等于到顶 —— "
             "那个标定值本身可能是 L2 命中下的观测",
        status="measured",
        evidence="sfag 2026-09-15 (B=1 S1=1024 S2=128K topk=2048 BSND det=1, 逻辑流量 28.44GB 恒定):\n"
                 "  band 扫描 (bench.py --idx-pattern banded_<N>, 索引**保持随机**只缩取值范围):\n"
                 "    kv 工作集 144MB -> 18.43ms | 72MB -> 15.24ms | 18MB -> 11.91ms | 4.5MB -> 11.78ms\n"
                 "  单调, 退化点在 18-72MB 之间 (L2 是 192MB, 但要与 q/dy/workspace 分享)。\n"
                 "★这条推翻了同日上午的判断。当时按 '28.44GB / 17.45ms = 1629 GB/s, 而 dram_GBps\n"
                 "  标定 1690, 已达 96.4%' 得出'只能减流量'; 而 1690 那个标定本身来自\n"
                 "  '索引改连续'的实验 —— 连续同时把工作集压到 2.36MB 全命中 L2, 它测的不是 DRAM。\n"
                 "  按工作集重新定向后, 空间是 36% 而不是 3%。dram_GBps 已降级为 uncalibrated,\n"
                 "  曲线记在 chip_calib 的 l2_working_set_curve。\n"
                 "★实现层的配套结论 (反例): 光让**索引分布**对齐没用。striped 实验让每行第 k 批\n"
                 "  严格落在第 k 段, 测得 18.16ms 与 random 的 18.15ms 无异 —— 因为 24 个核\n"
                 "  各自处在不同的行/批上, 瞬时工作集是它们的**并集**。要压住工作集必须\n"
                 "  **跨核同步推进**, 这是算子实现要做的事, 不是输入数据能给的。",
        verified="2026-09-15"),
    "P19_an_exactly_determined_fit_is_not_a_model": Ref(
        what=[
            "用 **n 个点拟 n 个未知数**得到的分解不是模型, 是插值 —— 残差恒为 0, 它一定'拟合得很好'。",
            "在用它立候选之前**必须先在留出点上验证一次**; 没有留出点就不要据它推导任何动作。",
        ],
        status="measured",
        evidence=[
            "VC-Attention 2026-10-27: 用 m32/N256 与 m64/N256 两个点拟 `fixpipe = 调用次数 x",
            "  (setup + 行数 x r)`, 解出'每次排空 0.637 us 固定 setup, 占 fixpipe 时间 67%'。",
            "  据此连立三个候选 (异步 IterateAll / PV 的 dbL0C / QK 的 dbL0A,dbL0B), **全部落空**。",
            "  取第三个点 m32/N512 作留出: 模型预测 161754 us, 实测 **326773 us**, 差 2.02 倍。",
            "★再逐一检验候选解释变量 (调用次数 / 总字节 / 总行数 / L0C fractal 数), **四个都不成立**:",
            "  m64/N256 与 m32/N256 的字节、行数、fractal 数三者全相同而 fixpipe 差 1.67 倍;",
            "  m32/N512 与 m64/N256 的排空次数相同、前者字节更少而 fixpipe 反而多 1.69 倍。",
            "代价: 三个候选各一次构建加一次测量; 而留出点的数据上一会话就已采在盘上。",
        ],
        verified="2026-10-27"),
    "P18_low_duty_cycle_indicts_the_model_first": Ref(
        what=[
            "按字节模型算出的**占空比远低于 1** 时, 第一嫌疑是**模型的形状参数与采集时不一致**,",
            "不是'这条通路被按住等待'。把它读成后者会得出'通路空着'的结论, 从而放弃真正有效的",
            "减字节方向。可证伪: 模型字节与 msprof 自报的片上搬运量 (L0C_to_GM / GM_to_UB 等)",
            "直接对账, 吻合才允许往硬件行为上解释。",
        ],
        status="measured",
        evidence=[
            "VC-Attention 2026-10-27: 我按 heads=2 写字节模型 (采集实为 heads=28), 算出 fixpipe",
            "  '占用里只有 7.7% 在搬运', 据此在台账里写下'依赖链受限, 通路 92% 时间没数据在动' (F107)。",
            "  同一口径的 AIV MTE2 也给 7.6%, 两条'互不相干的通路'给同一个数, 看上去是交叉验证 ——",
            "  其实只是同一个错误口径被用了两次。13.67 倍恰好是 28/2。",
            "改用采集时的 heads=28 后: 模型 10.015 GB/核 vs msprof L0C_to_GM **10.015 GB/核** (逐字节吻合),",
            "  AIV 侧 5.007 vs 5.021 GB/核。fixpipe 在 193.2 ms 占用里搬 10.015 GB = 51.8 GB/s,",
            "  与自报 active_bw 43.6 GB/s 同量级 —— 它**基本满负荷**, 结论与 F107 相反。",
        ],
        verified="2026-10-27"),
    "P16_wait_columns_must_be_discovered_not_hardcoded": Ref(
        what=[
            "每个跨核标志的等待时间列形如 `aic|aiv_scalar_wait_id<N>_time(us)`, 而 msprof",
            "**只输出该内核实际用到的编号**。把编号写死 = 只对某一个算子有效, 别的算子静默全空。",
            "★而且这一族在 **PipeUtilization.csv** 里, 不在 ResourceConflictRatio (后者是等待**比**)。",
            "两处取错任一处, 表现都是 wait_us_delta 恒为空字典且不报错。",
        ],
        status="measured",
        evidence=[
            "VC-Attention 2026-10-27: WAIT_COLS 写死 id14/id5 (来自 gmfr), 本算子用 8/9/10/15。",
            "候选 h16 的 O_READY 等待实际动了 37243 us (aiv 侧 47.9%->34.3%),",
            "S_READY 同时涨了 42027 us —— abdiff 一个字都没说, 因为两族都没收到。",
            "后果是实打实的: 我只能看 waitattr 的百分比, 据此预测收益, 而那一版端到端变慢 0.35%。",
            "改为发现式后同一对产物立刻给出上述两个绝对增量。",
        ],
        verified="2026-10-27"),
    "P17_no_single_wait_tracks_the_window": Ref(
        what=[
            "**单个标志的等待变化不预示窗口变化**, 跨构建更不能用它的百分比 —— 分母(窗口)自己会动。",
            "可比较的量是**绝对时间**, 而且要把同一侧的全部标志加起来看净值。",
        ],
        status="measured",
        evidence=[
            "VC-Attention 2026-10-27 两个候选, 同一改动只换放置位置:",
            "  h16  O_READY -37243 us 而窗口 **+2609 us** (AIV 侧净等待 +4784: S_READY 吃掉了 O_READY 的减量)",
            "  h16b O_READY  -6641 us 而窗口 **-2493 us** (AIV 侧净等待 -6398)",
            "即 O_READY 一个标志在 h16 上与窗口**反向**。两例中 AIV 侧**净**等待与窗口同向,",
            "但 fixpipe 的绝对增量 (+16647 / -19691) 也同向, 两个数据点分不开这两种解释 —— 不要只凭其一下结论。",
            "★第三例 (同日 h17, 更极端): S_READY 由 71005 us 降到 0.25 ms (**-70755 us, 完全消除**),",
            "  而窗口 **+5020 us**, 且 P_READY 不降反升 (+3468)。等待被消除后原地迁移到 O_READY (+34198)。",
            "  结论: 把某个等待消干净, 既不保证窗口下降, 也不保证与它相邻的等待下降。",
        ],
        verified="2026-10-27"),
    "P12_every_bandwidth_needs_its_denominator": Ref(
        what="每条带宽都要带硬件参照并**整组呈现**; 禁止用筛选规则决定只算哪一条",
        status="measured",
        evidence=[
            "sfag 2026-09-18 复盘 (dubin 指出): 我反复引用 `aiv_mte3_active_bw 18.50 -> 11.88 GB/s`",
            "并据此断言'这就是原子写的代价, 要减它得动分核' —— 这个结论关掉了整条搜索路径。",
            "而 **aiv_mte2 是更大的一条 pipe (18739us vs 13606us), 它的带宽我一次都没算过**,",
            "两个数在同一份 msprof 产物里, 算第二个成本为零。",
            "★为什么只算了一条: 判据 P4 (wait_ratio 最低者 binding) 说 MTE3, 我就只查 MTE3 ——",
            "  **让筛选规则替代了普查**。规则的职责是给注意力排序, 不是截断注意力。",
            "  更糟: 那个 wait_ratio 是在 Set/Wait 紧邻的串行结构上测的 (detsync_lint H10 错误级),",
            "  本身就不可靠 —— 一个不可靠的判读决定了去算哪个数。",
            "★配上分母之后原结论站不住: 11.88 GB/s = 端口能力的 **10.3%** (ub_to_ddr 64B/周期",
            "  x 1.8GHz = 115 GB/s/核)。端口 90% 是空的, 那是搬得太碎或忙时含等待,",
            "  **不是原子写的固有成本**。gather 那条 17.6 GB/s = 30.6%, 同样量级的问题。",
            "★工具侧缺口很具体: `_print_mte2_bw_verdict` 做的正是对的事 (带宽+参照+反推命中率),",
            "  但它**只对 aic_mte2 一条做**; 其余每条带宽都是裸数字。",
            "  没有参考区间的化验值读不了, 于是只有印了参考区间的那一行被读。",
            "★修法 `_print_bw_panel`: 全部带宽配参照、按占比**升序**排 (最差的排最前,",
            "  不让人只读自己想读的那行), 并**显式列出没有可靠参照的那几条**。",
            "★判读纪律: 本表只说'离端口能力还有多远', 不说该不该动它 —— 低占比可能是搬得碎,",
            "  也可能是忙时含等待, 两者下一步完全不同。要定量就去算 hwfloor 的超额表。",
        ],
        verified="2026-09-18"),
    "P11_calibration_held_fixed": Ref(
        what="引用标定值/标定曲线做预测前, 逐条核对它的 held_fixed —— "
             "本次改动只要动了其中任何一条, 该标定不适用, 不许据此预测。"
             "已工具化: `core/socspec.calib_check(name, changed)` 核不过直接取不到值",
        status="measured",
        evidence="本仓**三次**误用同一形态, 都是'看了值、没看条件':\n"
                 "  1. dram_GBps 按负载类型错用;\n"
                 "  2. dram_GBps 按工作集错用 —— 1690 GB/s 实为'索引改连续'实验的观测,\n"
                 "     而连续同时把工作集压到 2.36MB 全命中 L2, 它测的根本不是 DRAM。\n"
                 "     据此算出'28.44GB/17.45ms = 1629, 已达 96.4%', 把方向定成'只能减流量';\n"
                 "     按工作集重新定向后, 空间是 36% 而不是 3%。\n"
                 "  3. l2_working_set_curve 按调度结构错用 (2026-09-15 sfag v8a):\n"
                 "     二维分核把瞬时工作集 144MB 压到 7.1MB, 照曲线 18MB 档预测约 13ms,\n"
                 "     **实测 44.59ms / 修好 reload 折叠后 40.88ms** —— 不是偏差, 是方向相反。\n"
                 "     曲线是靠 band 扫描 (只改索引取值范围) 标的, 那个实验里改工作集**不改调度结构**;\n"
                 "     而二维分核压工作集的手段正是切分核 —— L1 每核私有, 8 个核各装一份整行的\n"
                 "     query/dY, 行切换次数乘 8, 代价远超 L2 局部性的收益。\n"
                 "★三次都是提示型判据拦不住的: 值就在那里, 取了就能算出一个数。\n"
                 "  所以 calib_check 的契约是**拦住时返回 None**, 调用方拒判, 不许退回 calib_value。\n"
                 "★未声明 held_fixed 的标定项同样拒判 —— 没记测量条件的值不许用来做预测。\n"
                 "  自检见 socspec.selftest_calib_check (11 条, 含三次真实误用与正当引用的放行)。",
        verified="2026-09-15"),
    "P10_traffic_beats_latency_when_memory_bound": Ref(
        what="访存绑定的算子里, 改动分两类: **减流量 / 去真实串行点** 有效, "
             "**减延迟或减指令开销** 基本无效。动手前先判自己属于哪一类",
        status="measured",
        evidence="sfag 2026-09-15 一个算子上的 6 次改动, 3 成功 3 证伪, 分界干净:\n"
                 "  ✓ 减流量:   dK/dV 合进同一块 L0C 只 Fixpipe 一次  -25.1%\n"
                 "  ✓ 减流量:   mm4 中间结果降 fp16                    -14.7%\n"
                 "  ✓ 去串行点: 去掉每 token 一次的两个全局汇合        -5.8ms\n"
                 "  ✗ 减延迟:   gather 的逐轮屏障改槽级 (v4a)          0 (AIV mte2 忙时反增)\n"
                 "  ✗ 减延迟:   topk 索引批量进 UB 消除 GM 标量读 (v7a) 0 (18.131 vs 17.852)\n"
                 "  ✗ 换并发度: 减核以压 L2 工作集                      负 (24/12/8 = 17.85/23.02/31.71)\n"
                 "  ✗ 减等待:   det scatter 中转槽 2->4 (v11a)          0 (18.02 vs 17.86, 噪声内)\n"
                 "              且 workspace +108 MB (464->572); 扩槽减的是 cube 等 vec 腾槽的等待\n"
                 "四个失败项的共同点: 攻的是'每次操作多花了多少时间', 而访存绑定下\n"
                 "端到端由**搬了多少字节**和**有没有被迫串行**决定, 单次操作的延迟被流水掩盖掉了。\n"
                 "★注意第三个 (减核) 的教训不同: 它同时改了并发度与工作集, 两者方向相反,\n"
                 "  实测 1.29x/1.78x 好于纯算力的 2.00x/3.00x —— 工作集效应真实但不足以翻盘。\n"
                 "★2026-09-16 补第四项 (扩桶) 时记下的直觉纠正: 有余量就拿去换性能, 在访存\n"
                 "  绑定下要先过这一关 —— 余量本身不产生收益, 除非那块内存真的减少了 DRAM\n"
                 "  流量或解开了一个串行点。扩桶是算子原始目标里写着的方向, 实测否定。\n"
                 "★用途: 拿到一个候选改动先问'它减了字节还是减了延迟'。后者在访存绑定下\n"
                 "  需要先拿真机计数器证明该条流水确实在等 (wait_ratio 高), 否则大概率白做一轮构建。\n"
                 "★**前提补充 (2026-09-15 晚, v7b 实撞)**: '减流量'还要再问一句 ——\n"
                 "  减的那部分**真的走 DRAM 吗**? 流量构成表里的 GB 是**逻辑流量**,\n"
                 "  与压在 DRAM 上的字节不是一回事。v7b 让 mm2 复用 mm1 已搬进 L1 的 kv,\n"
                 "  按字节算省 1.7GB/28.44GB≈6%, 实测只快 1.2% —— 因为那份数据是 cube1\n"
                 "  刚读过的, 大概率还在 L2, 省掉的是 L2 访问而非 DRAM 访问。\n"
                 "  对照: band 扫描改的是**L2 命中率**, 逻辑流量一字节没动, 效应却有 36%。\n"
                 "  所以判一个减流量方案值不值, 要看它动的是 DRAM 流量还是 L2 流量。",
        verified="2026-09-15"),
    "P2_issue_order_beats_buffer_depth": Ref(
        what="互等形态下先查**指令发出次序 (尤其预取的位置)**, 缓冲深度这条轴可能完全是死的",
        status="measured",
        evidence="[干净实测, 三点单变量] FIAS 100k_64P, 基线 557.90us:\n"
                 "  mm2 结果缓冲深度 1 / 2 / 4 = 558.71 / 557.90 / 558.25us —— **三点全平**。\n"
                 "  而把预取从'每轮最前'挪后一档 (mm2 插到它前面) = 712.53us (**+27.7%**);\n"
                 "  2026-09-11 挪到每轮最后 (预取距离归零) = +43%。\n"
                 "★**已更正**: 先前把 +35.2% 记作'深度降一级'的代价, 那一版同时改了发出次序与 KV 槽数;\n"
                 "  单变量复测 (只降深度) 只有 +0.15%。这一轮里同一个根源犯了两次归因错\n"
                 "  (另一次是把'减搬运量'记成零收益, 实际回收 38.16us)。\n"
                 "  教训: **一次构建放多个改动时, 任何单项的归因都不成立**, 哪怕其中一项看起来是主因。\n"
                 "[推导部分, **未跨算子验证**] 机理: 双缓冲吸收的是一个任务的错位, 再多只在抖动\n"
                 "  超过一个任务时才有用。FIAS 每任务工作量恒定 (无数据相关分支、块大小不变),\n"
                 "  抖动近零, 所以饱和点落在 2。**抖动大的算子饱和点可能更高, 本仓没有反例也没有正例。**\n"
                 "★这条的前身写成了'深度曲线到 2 就饱和'(把单点当规律), 2026-09-14 改成带判准的形式。",
        verified="2026-09-14"),
    "P8_footprint_does_not_predict_time": Ref(
        what="**按 footprint 是否装得进 L2 去估时间, 已证伪。** 工作集从装得进 L2 变成装不进, "
             "时间不按命中率模型走; 这一项的代价远小于按 L2/DRAM 带宽比推出来的量。",
        status="refuted",
        evidence="sfag 2026-09-17 消融实测。det=0 优化路径上只把 scatterTokenCapacity 从 128 "
                 "改成 2048 (host 与 kernel 同改), workspace 与写入地址摊开 16 倍, "
                 "footprint 由约 40 MB 变到远超 L2 的 192 MB。\n"
                 "预测 (按此前文档里的局部性放大拆法): 约 +10 ms。实测: **+1.64 ms** "
                 "(13.516 -> 15.160, 两趟 15.160/15.511)。\n"
                 "同族的前一个反例: 该算子上减掉 1.983 GB 访存量后时间纹丝不动 —— "
                 "因为减掉的全是 L2 命中, miss 只降 1.7%。\n"
                 "★作用域: 这一刀把地址**摊开**而每核工作集不变, 所以它给的是\"摊开\"的代价; "
                 "但足以否掉\"footprint 是主要成本\"这一类判断。要估时间只能实测。",
        verified="2026-09-17"),
    "P9_core_time_magnitude_is_not_critical_path": Ref(
        what="**比较 aic_time 与 aiv_time 的大小, 判不出关键路径。** 两者接近不等于两边都被"
             "自己的活撑满 —— 一边可能整段在等另一边。判\"某一侧省下来会不会落到墙钟\", "
             "只能消融, 不能看表推。",
        status="measured",
        evidence="sfag 2026-09-17。det=1 实测 aic_time 36173us / aiv_time 37075us / 墙钟 37.8ms。\n"
                 "据此推断: 把 aic_fixpipe 从 13243 降到 7354 后 aic_time 约 30.3us 量级, "
                 "仍低于 aiv_time, 故若 AIV 不是在等 cube 就省不下来 —— 差点据此放弃该改动。\n"
                 "实测: 只改 cube 侧输出格式 (ND -> NZ), det=1 墙钟 36.130 -> **30.578**, 降 5.55ms。\n"
                 "★可用的信号是 wait_ratio 而不是 time 的大小: det=1 的 aiv_vec/mte2/mte3 "
                 "wait_ratio 全在 0.5 上下, **没有一条接近 0**, 说明 AIV 三条都在等别人; "
                 "而\"等谁\"表里读不出来, 必须消融。\n"
                 "★与 P4 的关系: P4 说先用 wait_ratio 定位因, 本条补的是**跨核类型**的情形 —— "
                 "因可能整个在另一类核上, 此时同侧的 pipe 表怎么读都读不出来。",
        verified="2026-09-17"),
    "P10_stall_counter_does_not_identify_the_binding_constraint": Ref(
        what=[
            "**stall 计数器很大, 不等于那一侧是约束。** `aiv_scalar_vector_stall` 占标量时间 95%,",
            "并不意味着减少向量发射就有收益 —— 它只说明标量在等, 没说等的那件事是不是瓶颈。",
            "判 binding 用 **wait_ratio 最低的那条 pipe**, 不用 stall 的绝对值或占比。",
        ],
        status="measured",
        evidence=[
            "sfag 2026-09-17, 连续两刀打同一处、都约等于零, 合起来才看清:",
            "det=1 实测 aiv_scalar 13248us, 其中 scalar_vector_stall 12572us (95%),",
            "而 aiv_vec 只忙 2775us —— 看起来强烈指向'逐条发射、发射开销压倒执行'。",
            "g10 (索引预取进 UB, 去掉上千次 GM 标量读): aiv_scalar 只降 74us, 端到端 +0.43%。",
            "g11 (每行一次 Add 合成带 repeat 的 2 次调用): 端到端 **−0.48%**, 反而更慢。",
            "两刀合计约 0。真正 binding 的是 wait_ratio 最低的 aiv_mte3 (0.21), 不是标量侧。",
            "★与 P4 的关系: P4 说用 wait_ratio 定位'因'; 本条补的是**反面** ——",
            "  stall 计数器与忙时占比都不能替代 wait_ratio, 它们指向的常常是'果'。",
            "★同族: P8 (按 footprint 估时间已证伪) 与 g10 那次 (按**次数**估成本亦不可靠)。",
        ],
        verified="2026-09-17"),
    "P7_cube_busy_vs_mmad_count": Ref(
        what="cube 流水的**实测**忙时明显高于理论 MAC 周期, 且随 **Mmad 条数**增加而增加 —— "
             "所以'把 k 切细让每片都填满'是反向优化; 但只有两个实测点, **模型尚未定**",
        status="disputed",
        evidence="FIAS 950PR_957b @100k_64P, PipeTimeline 实测 cube 忙时 (6 核均值) vs "
                 "aic_cube_total_instr_number (msprof 实测的 Mmad 条数):\n"
                 "  vendor_cv   (baseK=128, mm1 5 片含一个 K=64 的尾片 + mm2 4 片): Mmad 3024/核, cube 忙 **502us**\n"
                 "  vendor_bk64 (baseK=64,  mm1 9 片全满             + mm2 4 片): Mmad 4368/核, cube 忙 **563us**\n"
                 "  两者的有效 MAC 完全相同 (8,912,896/任务), 理论下限都是 442.1us。\n"
                 "★结论一 (确立): **消除半满尾片反而更慢**, cube 忙时 +61us。'让每个 k 片都填满'在这颗芯片上\n"
                 "  是反向优化; 方向是**更粗的切分**, 而 baseK<=128 与 mm2 的 baseN<=128 都由 L0B 的 32KB 槽定死。\n"
                 "★结论二 (未确立, 故 disputed): 想把它写成 '忙时 = 理论 MAC 周期 + Mmad 条数 x 常数' 对不上 ——\n"
                 "  按此式两点各自解出 32.7 与 45.7 周期/Mmad (差 40%); 两点线性拟合得斜率 74.9 周期/Mmad、\n"
                 "  截距 364.7us, 而截距低于理论下限 442.1us, **物理上不成立**。\n"
                 "  说明差异不只是每条指令的固定开销 —— K=64 的 Mmad 很可能本身 MAC 效率就低\n"
                 "  (脉动阵列在 K 方向填不满), 即每 Mmad 的代价随形状变。\n"
                 "  **两个点定不下三个参数。要用这条估收益, 先补第三、四个点 (改 baseN 或 mBase 各取一个)。**",
        verified="2026-09-14"),
    "P6_cube_time_is_derived_not_measured": Ref(
        what="**PipeUtilization 的 aic_cube_time 是按指令流算出的理论下限, 不是实测占用** —— "
             "拿它算'停顿 = aic_time - aic_cube_time'会得到一个并不存在的空闲量; 要实测占用必须采 PipeTimeline",
        status="measured",
        evidence="FIAS 950PR_957b @100k_64P, vendor_cv:\n"
                 "  aic_cube_time = 443.11us, 在本会话全部 6 个构建上**逐位相同** (base/bk64/msk/rg3g2/pf/rect)。\n"
                 "  换算: 443.11us x 1.65GHz / 336 任务/核 = 2176 周期/任务;\n"
                 "  每任务有效 MAC = mm1 64x128x576 + mm2 64x512x128 = 8,912,896;\n"
                 "  8912896 / (2176 x 4096 MAC/周期) = **恰好 100.0%** —— 它就是理论下限本身。\n"
                 "  ⇒ 它的用途只有一个: 当**可比性校验**(两次采集若它不同, 别的数字也别比)。\n"
                 "★实测占用要采 --aic-metrics=PipeTimeline, 产出是 trace.json (不是 CSV,\n"
                 "  core/perf.run_msprof_op 的后置检查按 CSV 找会报错, 但数据在)。\n"
                 "  按 pid ('core<N>.cubecore' / '.veccore0' / '.veccore1') 分核, 再按 name 分流水。\n"
                 "★**同样是派生的还有 aiv_vec_time** (2026-09-28 fusion_attention_inference 补)。\n"
                 "  原条目只点了 aic_cube_time, 于是本会话据 aiv_vec_ratio 得出'AIV vec 是占用最高的\n"
                 "  一条流水 (占墙钟 61.8%)'并差点据此定优化方向 —— 那个数字同样是指令流下限。\n"
                 "★不用 PipeTimeline 也能分辨派生与实测: **同一构建重复采 3 次, 看列的重复间抖动**。\n"
                 "  910B2 / FIA base 段 (S1=44672 S2=42647 N=40 D=128 fp16) 三次采集:\n"
                 "    aic_mac_time 120262.494 / 120260.375 / 120262.890  极差 2.5us   = 2e-5  -> 派生\n"
                 "    aiv_vec_time 162648.425 / 162645.323 / 162647.600  极差 3.1us   = 2e-5  -> 派生\n"
                 "    aic_mte2_time 90219.367 / 89886.224 / 89936.151    极差 333us   = 4e-3  -> 实测\n"
                 "    aic_fixpipe_time 135925.8 / 135402.2 / 135560.7    极差 524us   = 4e-3  -> 实测\n"
                 "  判据: 重复间相对极差 <1e-4 的列是按指令流算出来的, 只能当可比性校验与**下限**,\n"
                 "  不能当占用率; >1e-3 的列才是真计数器。这个筛子零成本 (采集本来就要重复多趟)。\n"
                 "  FIAS vendor_cv 实测 (6 个 cube 核): 窗口 532.7~548.5us, CUBE 忙 500.2~504.1us (**占窗口 94%**),\n"
                 "  空闲 20.5~37.3us (566 段 x ~40ns), 核间 CUBE 忙时极差仅 0.8% (负载均衡)。\n"
                 "  于是损失的真实构成是: 空闲仅 ~22us, 而 cube '忙'的 60us 里只做了 442.1/502 = **88.3%** 的有效 MAC。\n"
                 "★这条一改, 由旧口径推出的结论要跟着重算: 本会话曾据 aic_time-aic_cube_time=94.8us\n"
                 "  推出'停顿是每任务固定开销 x 任务数', 该推导不成立 —— 真实空闲只有 22us。\n"
                 "★这条口径**有预测力, 不只是事后解释** (2026-09-14 当场验过一次):\n"
                 "  按旧口径 (停顿 94.8us = 每任务固定开销) 做的候选 vendor_sc —— 把 CalcParams 里 28 个\n"
                 "  逐行不变的字段 (含 3 次除数为运行期值的整数除法 + 一次 GetPreNextTokenLeftUp)\n"
                 "  从每块重算改为每行一次 —— 原估 1.5~2.7%。改用新口径后**在测之前**把预期下调为\n"
                 "  '很小', 理由是真实空闲只有 22us 且形态是 566 段 x 39ns 的每指令级开销,\n"
                 "  标量工作本来就藏在 cube 的影子里。实测: 560.05 vs 557.90 (+0.39%) 与 2200.99 vs\n"
                 "  2201.39 (-0.02%), 精度四条全过且 MERE 与基线逐位相同 —— **等价变换成立但不值钱**,\n"
                 "  与下调后的判断一致。"
                 "★作用域与入口 (2026-09-17 补, 本条此前只说'必须采 PipeTimeline'却没说**哪里采得到**): "
                 "本仓所有 trace.json 都长在 `OPPROF_*/simulator/<core>/trace.json`, "
                 "即全部由 **`msprof op simulator`** (core/sim.py) 产出 —— "
                 "**--timeline 从一开始就是读模拟器产物的**, 不是读实机 PipeTimeline。"
                 "把它当实机工具会走死路: 实机 op 模式下本条实证来自 **950PR**, 在 **910B2** 上 msprof 直接拒收: "
                 "`--aic-metrics=pipetimeline: Unexpected argument, maybe in wrong soc platform`。"
                 "★但「实机没有流水时间线」这个结论**是错的, 2026-09-17 当天订正**: "
                 "910B2 的 `msprof op --help` 列出 **TimelineDetail** —— 它就是本平台上的流水时间线, "
                 "只是**不叫 PipeTimeline**。照搬别处的档位名被拒之后, 一度据此误判成"
                 "「本 SoC 没有可用手段」, 差点把唯一的分岔点放弃掉。"
                 "**教训: 拿到 `Unexpected argument ... maybe in wrong soc platform` 时, "
                 "先敲 `--help` 看本机的档位名, 不要直接结论成'不支持'。**"
                 "core/sim (msprof op simulator) 是另一条路, 给的是**模拟器**上的 per-instruction 轨迹, "
                 "与实机 TimelineDetail 并列而非替代。"
                 "★**2026-10-23 补一条采不到的情形 (01 §14 一)**: `TimelineDetail` 与 "
                 "`--replay-mode` **互斥于两档** —— msopprof(9251896) 的 .rodata 里写着 "
                 "`--aic-metrics=TimelineDetail is invalid when --replay-mode=application` (@7864872) 与 "
                 "`--aic-metrics=TimelineDetail/Source/MemoryDetail is invalid when --replay-mode=range` (@7864992)。"
                 "默认是 `kernel` 档所以平时采得到; 一旦换成另外两档, **P6 这条路直接没有产物**, "
                 "而报错落在 msprof 侧、不会指向本条。另: `range` 还要 `--mstx=on`, 且该档下 "
                 "`GM to L2Cache` **不计 L2 evict** —— 换档同时换了测量口径, 跨档的数不可直接比。"
                 "★这条路**在政策上是许可的**: CLAUDE.md 的原文是"
                 "'sim 只看**指令级机理**, 禁用小 sim shape 成本分布定大 shape 优先级'—— "
                 "'流水是忙还是在等、空隙是多而短还是少而长'(见 P8) 正是**机理**, "
                 "不是成本分布优先级。两者的界线要分清, 否则会把唯一可用的机理工具也一并禁掉。", verified="2026-09-14"),
    "P8_gap_shape_tells_the_cause": Ref(
        what="定位停顿要看空隙的**形态** (段数 x 单段长度), 不是空隙总量 —— "
             "'多而短'是每指令级开销, '少而长'是跨核握手, 两者的修法完全不同",
        status="measured",
        evidence="FIAS 950PR_957b @100k_64P, PipeTimeline 按核判读 (core/perf --timeline):\n"
                 "  vendor_cv  CUBE 空闲  22us = **566 段 x 39ns**   -> 每指令级, 没有单点可修\n"
                 "  vendor_rg5 CUBE 空闲  87us = **261 段 x 333ns**  -> 少而长, 约 550 周期, 每块一次\n"
                 "  vendor_rg3 CUBE 空闲 218us (调度已崩, 作为上限参照)\n"
                 "两者空闲总量差 65us, 但**真正指路的是形态**: 261 段 ~= 每个 KV 块一次, 而 rg5 相对 cv\n"
                 "唯一的每块一次的新等待就是合并 mm1 之后 AIC 在同一任务里连发 groupRows 次 fixpipe,\n"
                 "每次都要等对应 AIV 腾出 UB, 而 AIV 每任务只释放一个槽。\n"
                 "★段数对上'每 X 一次'是最有力的定位线索: 拿段数去除任务数/块数, 得到的比值直接指向\n"
                 "  是哪一级循环里的同步。本例 261/168 块 ~= 1.55, 而 566/336 任务 ~= 1.68 (每任务一次)。\n"
                 "★反面教训: 本会话此前三次归责 (P4 指 mte2 / P5 的'忙时全降反更慢' / A16 判死行分组)\n"
                 "  都只看了忙时总量, 直到采 PipeTimeline 看形态才拆开 —— cube 侧真实收益 30.6us 与\n"
                 "  握手代价 65us 此前一直被混在一个'+26.6us 更慢'里。",
        verified="2026-09-14"),
    "P5_all_pipes_down_yet_slower": Ref(
        what="**跨构建比忙时**: 一个构建把**每一条**流水的忙时都压下去了而端到端反而更慢 —— "
             "这是'约束不在任何一条流水的工作量上'的判决性形态, 出现它就停止减工作量, 改查依赖与发出次序",
        status="measured",
        evidence="FIAS 950PR_957b @100k_64P, 交付版 vs 行分组版 (rg5, 精度四条全过, 归因干净):\n"
                 "  aic_cube   443.11 -> 443.11  (确定性量, 用来校验两次采集可比)\n"
                 "  aic_mte2   377.26 -> 216.63  (-43%, KV 一块喂 2 行)\n"
                 "  aic_mte1   363.46 -> 279.65  (-23%, 合并 mm1 后 K 的 L1->L0B 载入减半)\n"
                 "  aic_scalar 313.60 -> 281.04  (-10%, k 片段数减少)\n"
                 "  aic_time   537.68 -> **569.91** (+6.0%); 端到端 557.90 -> 584.52 (+4.77%)\n"
                 "  停顿 = aic_time - aic_cube: 94.6 -> **126.8us** (+34%)\n"
                 "三条流水各自的设计意图都兑现了, 停顿把收益全吃掉还倒欠。\n"
                 "★用法: 这个形态一出现, 再减工作量就是白做。此时要量的是**实测占用**, 而量法只有一个 ——\n"
                 "  采 PipeTimeline 看真实空闲 (见 P6)。**不要**用 aic_time - aic_cube_time 当停顿:\n"
                 "  aic_cube_time 是派生的理论下限, 那样算会凭空造出一个并不存在的空闲量。\n"
                 "  本会话曾据此算出 94.8us 停顿并推出'每任务固定开销'; PipeTimeline 实测真实空闲只有 22us,\n"
                 "  cube 已 94% 在发射 —— 该推导作废 (原文保留, 免得后人重走)。\n"
                 "★它也是 P4 (wait_ratio 定因) 的兜底: wait_ratio 表里没有标量流水, 会指错;\n"
                 "  而本判据只看忙时与总时, 不依赖任何一条比值是否被采到。",
        verified="2026-09-14"),
    "P4_find_cause_by_wait_ratio": Ref(
        what="互等形态下**先采 ResourceConflictRatio 的 wait_ratio 定位'因', 再动手** —— "
             "只看 PipeUtilization 的忙时会把'果'当成'因', 白做好几轮",
        status="disputed",
        evidence="判读本身是旧的 (docs/perf-ceiling-estimation.md §5, gmfr 标定): wait_ratio≈0 = 因, 高 = 果。\n"
                 "FIAS 2026-09-14 才第一次采它 (此前只采忙时), vendor_cv @100k_64P:\n"
                 "  aic_mte2 **0.599** (最低=因) / aic_cube 0.979 / aic_mte1 0.973 (果);\n"
                 "  AIV 三条 vec 0.961 / mte2 0.947 / mte3 0.958 **全是果**。\n"
                 "★这一条解释了此前五次'减了活却不快': 减 AIV 工作量 41.87us -> 端到端 -0.08%;\n"
                 "  去跨核旗标 -> 精确中性 (559.06 -> 559.03, **改之前按此判据预测中性, 实测吻合**);\n"
                 "  缓冲深度 1/2/4 三点全平; unitFlag 中性。它们都不碰 mte2。\n"
                 "  而碰 mte2 的两项都有效: 预取位置 27~43%, KV 流量减半 38.16us。\n"
                 "★同日推翻了据此得出的结论 (判据本身仍有效, 但**前提是表里有那条流水**)。\n"
                "ResourceConflictRatio 只报 aic 的 cube/mte1/mte2/mte3 与 aiv 的 vec/mte2/mte3,\n"
                "**没有标量流水的 wait_ratio**; 而 FIAS 的 aic_scalar_time 稳定占 aic_time 的 55%\n"
                "(305/538 @shape1, 1269/2196 @4BS)。于是'四条比值都高、mte2 略低'被读成 mte2 是因。\n"
                "反证来自忙时表本身: rg3g2 把 aic_mte2 从 460.45 砍到 193.38us (不到一半),\n"
                "aic_time 却最差 701.33us (base 649.16, 最好 537.68)。据此判据做的三轮减 KV 字节\n"
                "(rg2 +32% / rg3 +28% / rg5 +4.66%) 全部为负。\n"
                "真正在变的是**停顿** = aic_time - aic_cube_time (cube 忙时 443.11us 恒定, 可跨采集比):\n"
                "  base 206.1 / bk64 151.7 / msk 95.4 / rg3g2 258.2 / pf 94.6 / rect 122.3。\n"
                "交付版停顿 94.8us, 而每任务固定空转 564 周期 x 336 任务/核 / 1.65GHz = 114.9us, 量级吻合。\n"
                "⇒ 用法修正: wait_ratio 指出方向后, **先用一个把该流水忙时大幅砍掉的构建去反证**;\n"
                "  忙时降了而端到端不降, 这条流水就不是因, 无论它的 wait_ratio 有多低。",
        verified="2026-09-14"),
    "P3_intra_task_cv_chain": Ref(
        what="[**已被 P4 取代**] 曾推测深度加不动之后剩下的是任务内部的跨核依赖链",
        status="refuted",
        evidence="**排除法剩下的, 没有被直接测过**: FIAS 上依次证伪了带宽 (搬运量减半兑现率 <20%)、\n"
                 "  缓冲深度 (2->4 为 0)、Mmad->fixpipe 重叠 (unitFlag 为 0) 之后才轮到它;\n"
                 "  而 L1 装不下第 4 级流水 (Q 72K + 3 x 144K = 504K/512K, 第 4 个槽要再 144K),\n"
                 "  所以**没能验证**。量级上对得上 (两侧各空 550~850 周期/任务, 每任务约 3 次跨核等待),\n"
                 "  但那只是量级吻合, 不是证据。\n"
                 "★2026-09-14 被 wait_ratio 直接推翻: 因是 **aic_mte2** (wait_ratio 0.599, 全场最低),\n"
                 "  cube/mte1/AIV 全是果。跨核依赖链那条的两个可改项 (旗标条数、流水级数) 里,\n"
                 "  旗标条数已实测中性。**保留本条是为了让后人不必重推一遍这个死胡同。**",
        verified="2026-09-14"),
    "P15_vec_ratio_includes_unexported_dtypes": Ref(
        what="`aiv_vec_ratio` 里**含有本机不导出 dtype 列的那部分向量工作** —— "
             "按 `aiv_vec_fp32/fp16/int32/int16/misc_ratio` 去拆它, 最多会漏掉 14%",
        status="measured",
        evidence=[
            "2026-10-23 (08 §3.98, 09 第 153/154 条): 逐行量 "
            "`aiv_vec_ratio - Σ(五个 dtype 分项)`:",
            "  动因样本 gmfr   中位 **0.00%**  —— 配分精确, 五列就是全部",
            "  GemmV3          中位 **14.33%** —— 这部分落在本机**从未导出**的那族列",
            "  ★**Cast 跨目录 0.23% ~ 95.79%** (2026-10-23 复核, 08 §3.98 的表):",
            "    这个比例由**该次运行用了哪些向量单元**决定, **不是算子的属性** ——",
            "    按算子记它下次就会拿错一个数, 要用就现跑 `--vecshare`。",
            "                  (`vec_sfu`/`ldu`/`stu`/`simt_vf`/`vf`, 08 §3.81 的 59 列)",
            "★后果: 若拿 dtype 分项去回答'这个核的向量工作是什么类型', 在 GemmV3 这类核上",
            "  会得到一个**少了七分之一**的构成, 而总量 `aiv_vec_ratio` 本身是对的。",
            "  要量那部分: `python -m core.aprof --vecshare <kernel 目录>`。",
            "★不要把比值读成整数比: 第一版看到 GemmV3 的最小值 1.166667 写了'恰好 1/7',",
            "  而逐行检验 `vr×6/7 == Σ` 48/48 全不成立 (中位 1.16724)。",
        ],
        verified="2026-10-23"),
    "P13_single_block_pipe_is_not_busy": Ref(
        what="一条流水只有**一段**就覆盖了几乎整个窗口时, 那不是'这条流水忙了 99%', "
             "而是**不可判读** —— 多半是 spin 等待被记成一整段",
        status="measured",
        evidence=[
            "SMLA 950PR_957b @prefill 4096x8192x640, `perf --timeline` 实测 (2026-10-21):",
            "  core0.veccore0 SCALAR: **6 段, 其中一段 2444.9us**, 窗口 2470us -> '占 99.7%'",
            "  core1.veccore1 SCALAR: 22 段 (超过任何按条数的门限), 但最长一段仍占 99%",
            "对照: 同一次采集里 cubecore 的 SCALAR 有 6141 段 / 691us —— 同一个 pipe 名,",
            "  两个核类型上的读数含义完全不同。",
            "★代价: op_summary 的 aiv_scalar_time=1653.8us (92.3% of e2e) 被读成"
            "'标量是瓶颈, 去减少标量指令', 据此切了两刀 (C2 常数折叠 -14.5us, C3 掩码外提 -3.7us),"
            "  合计只拿回 18us —— 因为那条读数压根不是'工作量'。",
            "★判据: `最长一段 / 窗口 >= 0.90` 即判不可判读, **且不许进跨核并集** ——",
            "  它一进来就把那一侧顶到 100%, '谁定墙钟'永远答成该侧, 卡点归因直接失效。",
        ],
        verified="2026-10-21"),
    "P14_cross_core_coupling_names_the_side": Ref(
        what="判'墙钟由哪一侧设定'看**跨核并集占用 + 空闲覆盖**, 不看单条流水的忙时占比 —— "
             "AIC 空闲里有多少落在 AIV 忙区, 就是 AIC 等 AIV 的量",
        status="measured",
        evidence=[
            "SMLA 950PR_957b @prefill 4096x8192x640, `perf --timeline` (2026-10-21, 6 个采样核):",
            "  AIC 并集 1965~1984us (79.5~80.1%), AIV 并集 2448~2478us (98.9~100%)",
            "  AIC 空闲 496~505us, **其中 100% 落在 AIV 忙区里**; AIV 空闲仅 0.4~26us",
            "  => 墙钟由 AIV 设定; AIC 侧的 cube 只忙 44%, 削 AIC 工作量不减墙钟。",
            "★这与'每核单条流水忙时'给出的图景相反: CUBE 44% 看着像 cube 有大把空闲,"
            "  AIV 各条 27~68% 看着像都没满 —— 只有把同侧多条流水并起来、再看另一侧的空闲"
            "  落在哪, 才看得出是 AIV 在定节奏。",
            "★探针放大必须先扣: 开了 PipeTimeline 之后内核本身被拖慢, 本例 2482us vs 未插桩",
            "  Task Duration 1785.9us = **1.39x**。绝对忙时可比 (CUBE 1088 vs aic_mac 1065),"
            "  占比不可直接与 op_summary 对照。工具把这一条打在 warnings 里。",
            "★尚未验证的部分: 这是**一个算子的一次采集**, 比的是'同一次采集内两侧谁更满',"
            "  还没有跨算子验证; 按 dual-loop 的 promotion 阈值, 它到 'measured' 为止, 未到 'policy'。",
        ],
        verified="2026-10-21"),
    "P20_a_pipe_floor_is_a_floor_on_the_current_work_assignment": Ref(
        what=[
            "`墙钟 >= max(各 pipe 忙时)` 这个下界**只对当前的「工作量 -> pipe」分派成立**。",
            "它不是设计的下界, 更不是硬件的下界 —— **同一份数学工作可以换一条通路算**。",
            "引用这个数时必须同时写出它依赖的三项: (1) 工作量分派方式 (2) tile 数 (3) 数值契约。",
            "少写第三项会被当成实现问题, 少写第一项会被当成'没路了'。**我两样都犯过。**",
            "★与 P9/P10 的分工: 那两条讲**归因**(哪条 pipe 是因, 要用 wait_ratio 不能用忙时大小);",
            "  本条讲**再分派**(这份活根本不必在那条 pipe 上)。两者不冲突, 回答的是不同问题。",
            "★本条给的是**必要**条件不是充分条件: 降低 binding pipe 才可能越过下界, 但会不会",
            "  落到墙钟上仍受 P9 约束 —— 只能消融, 不能看表推。",
        ],
        status="measured",
        evidence=[
            "VC-Attention 2026-10-29。我按 max(各 pipe) 算出 control 臂 192.6 ms, 并把它写成",
            "  '本设计的硬下界', 据此一度判定'怎么优化都比 Laser 差'。**两步都错**。",
            "★存在性证明 (同机同卡同形状 [1,28,37741,128] BF16, 24 AIC / 48 AIV, msOpProf 实测):",
            "          通路        Laser    CANN FA    我们 control   我们/Laser",
            "          aic_cube     62731     82729        50767        0.81",
            "          aic_fixpipe  34611     95497       192589      **5.56**",
            "          aiv_vec      59172     92023       137343        2.32",
            "          窗口         87299    142000       280251        3.21",
            "  各自的下界与重叠度: Laser cube 62.7ms / 3.74 · FA fixpipe 95.5ms / 2.51 ·",
            "  我们 fixpipe 192.6ms / 1.64。",
            "★三条读数:",
            "  (甲) **Laser 是 cube 受限的**, 且其 cube 是 100% fp16 (cube_int8_ratio=0)。",
            "       量化恰好打在它的瓶颈上 —— 所以'量化路线必然差'不成立, 是**数据流**选错了。",
            "  (乙) 我们相对**自己下界**的效率 280.3/192.6=1.46, Laser 87.3/62.7=1.39, 相当。",
            "       即调度不是问题, **下界本身才是** —— 这反过来印证 H17/H18 两次证伪是对的,",
            "       我错在把'调度到头了'读成了'没路了'。",
            "★**口径陷阱 (2026-10-29 实测补)**: `*_scalar_time` **不能**填进 max(各 pipe)。",
            "  本算子 m64 实测 aic_scalar_time 262245 us/核, **比 fixpipe 的 192511 还大**,",
            "  我当场据此写下'scalar 才是最大通路'。查分解: single+dual (真发射) 只占 30.1%,",
            "  而 wait_id9 (等跨核标志) 占 **42.7%**、wait_ib 占 5.3% —— 近一半是等别人。",
            "  要算工作量下界只能用 single+dual; 用 scalar_time 会得出错误的 binding pipe。",
            "  ★上面那张表没有 scalar 行, 那个遗漏**碰巧没出错, 但理由是错的** ——",
            "   我不是'知道它该排除', 而是'不知道它存在'。与 P10 同族 (占用/stall ≠ 约束)。",
            "  (丙) 搬运是真实可行的: Laser 把 softmax 行重缩放做成**对角阵矩乘**搬到 cube ——",
            "       AIV 只算 128 个 1/l_j 再摊成对角阵 (VectorForward.h:464-474),",
            "       AIC 用 diag_matmul 左乘 (CubeForward.h:201/:353)。教科书 FA 里那是 AIV 上",
            "       16384 元素的全宽 Muls。**工作量没变, 通路换了。**",
            "代价: 这个盲点让我在'怎么优化都追不上'这个错误结论上停留了整整一轮对话,",
            "  而把它证伪只需要一次 msOpProf (十几分钟)。",
        ],
        verified="2026-10-29"),
}


import argparse
import ast
import contextlib
import csv
import ctypes
import glob
import math
import json
import os
import re
import statistics
import subprocess
import sys

import pandas as pd

# 真实 PipeUtilization 列名 (aic_ = cube 行, aiv_ = vector 行)
RATIO_COLS = [
    "aic_cube_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_mte3_ratio",
    "aic_fixpipe_ratio", "aic_scalar_ratio",
    "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_scalar_ratio",
]
# 绝对 pipe 时间 (us). stall 主导的劣化里 ratio 会误导 (总时膨胀→所有 ratio 下降),
# 必须看绝对时间 + wait/stall 才能正确归责. 这是 dual-loop 反向环补的 infra 缺口.
TIME_COLS = [
    "aic_cube_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)", "aic_mte3_time(us)",
    "aic_fixpipe_time(us)", "aic_scalar_time(us)",
    "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)", "aiv_scalar_time(us)",
]
# 屏障/等待/停顿列 — SyncAll 等同步开销的直接证据。
# ★这里只列**与标志编号无关**的那些。每个跨核标志的等待时间列形如
#   `aic_scalar_wait_id<N>_time(us)`, 而 msprof **只输出该内核实际用到的编号** ——
#   把编号写死就等于只对某一个算子有效。
#   2026-10-27 实撞 (VC-Attention): 此表原本写死 id14/id5 (来自 gmfr), 而本算子用 8/9/10/15,
#   于是 `diagnose abdiff` 的 wait_us_delta **恒为空字典且不报错**。
#   后果是实打实的: 候选 h16 的 O_READY 等待动了 535 万周期 (47.9%->34.3%), abdiff 一个字
#   都没说; 我因此只能看 waitattr 的百分比, 据此预测收益 —— 而那一版端到端是变慢的。
#   现改为按正则从产物里**发现**编号, 见 _WAIT_ID_RE 与 _parse_launch。
WAIT_COLS = [
    "aic_scalar_wait_time(us)", "aiv_scalar_wait_time(us)",
    "aic_scalar_cube_stall_time(us)", "aiv_scalar_vector_stall_time(us)",
    "aiv_scalar_stall_by_ub_time(us)",
]
#  每标志等待时间列。★它们在 **PipeUtilization.csv** 里, 不在 ResourceConflictRatio ——
#  后者给的是等待**比**。两族同名相近, 取错文件的表现同样是"全空"。
_WAIT_ID_RE = re.compile(r"^(aic|aiv)_scalar_wait_id\d+_time\(us\)$")

_KIND = {
    "aic_cube_ratio": "compute(cube)", "aiv_vec_ratio": "compute(vector)",
    "aic_mte2_ratio": "memory-in(GM->L1)", "aiv_mte2_ratio": "memory-in(GM->UB)",
    "aic_mte3_ratio": "cube-out(->GM)", "aiv_mte3_ratio": "memory-out(UB->GM)",
    "aic_mte1_ratio": "onchip(L1->L0)", "aic_fixpipe_ratio": "cube-out(fixpipe)",
    "aic_scalar_ratio": "scalar/ctrl", "aiv_scalar_ratio": "scalar/ctrl",
}


class ProfilingData:
    def __init__(self, op_name, task_type, task_duration_us, block_dim, ratios, times=None, waits=None):
        self.op_name = op_name
        self.task_type = task_type
        self.task_duration_us = task_duration_us
        self.block_dim = block_dim
        self.ratios = ratios
        self.times = times or {}
        self.waits = waits or {}

    def bottleneck(self) -> str:
        r = {k: v for k, v in self.ratios.items() if v is not None}
        if not r or max(r.values()) < 0.10:
            return "unknown"
        top = max(r, key=r.get)
        return f"{_KIND.get(top, top)} [{top}={r[top]:.2f}]"

    def as_dict(self) -> dict:
        return {"op_name": self.op_name, "task_type": self.task_type,
                "task_duration_us": self.task_duration_us, "block_dim": self.block_dim,
                "bottleneck": self.bottleneck(), "ratios": self.ratios,
                "times_us": self.times, "waits_us": self.waits}



#  ---- 工作量再分派: 按「哪条 pipe 受限」索引的手段表 (P20 的可操作面) ----------
#  动因: P20 记下了"下界是对当前分派的下界"这个教训, 但记住教训不等于下次想得起来。
#  这张表的作用是: 画像指出 binding pipe 之后, **直接给出候选手段**, 而不是等人
#  想起去读别人的内核源码。每条必须写全四项 —— 机理 / 代价搬到哪条 / 本仓试过没有
#  及结果 / 外部实例。**没有实例或没有机理的条目不收**, 否则这张表会退化成猜想清单。
#  ★两类要分开: "搬" (work moves to another pipe) 与 "减" (work is eliminated)。
#   两者的下一步不同 —— 搬要看目标 pipe 有没有余量, 减要看契约允不允许。
#  ★★"搬"**不天然**意味着总工作量不变, "减"也**不天然**意味着其它通路没有新增成本。
#   代价要以**整个执行图**的变化衡量, 不能只看单条 pipe 的忙时降了多少 (用户 2026-10-29 指出)。
#  ★★**binding pipe 只是检索入口, 不是完整适用条件。** 每条必须带 `ctx`, 至少写明
#   形状 / dtype / 布局 / 当时的资源余量 / CANN·SoC 版本 —— 否则会检索到
#   "机制对而适用条件错"的技巧, 那比没有这条更糟 (同上)。
_CTX_VC = ("形状 [1,37741,28,128] BF16 自注意力 dense noncausal; 布局 NHD; "
           "契约 INT8 QK (Q32/K128) + E4M3 V 解码成 FP16; tile 64x256, 101873 tile/核; "
           "资源余量: cube 占窗口 18.1% (有余), fixpipe 68.3% (binding); "
           "Ascend910B2 / CANN 9.0.0 / torch_npu 2.9.0; 24 AIC / 48 AIV @1800MHz")
_CTX_LASER = ("MindIE Laser original (提交 71098c20), 同机同卡同形状; 契约 FP16 不量化; "
              "基本块 128x128; 资源余量: 它是 **cube 受限** (下界 62.7ms, cube_int8_ratio=0); "
              "Ascend910B2 / CANN 9.0.0")
_WORK_MOVES = {
    "aiv_vec": [
        dict(kind="搬", name="行重缩放 -> 对角阵矩乘搬到 cube",
             how="在线 softmax 的逐行重缩放, 教科书 FA 是对整块 [128x128] 累加器做全宽 Muls "
                 "(16384 元素, 在 AIV)。改成: AIV 只算 128 个 1/l_j, 摊成 128x128 对角阵写 GM, "
                 "AIC 用一次 128x128x128 矩乘左乘它。",
             cost="aic_cube (+一次方阵矩乘/块) 与 L1/L0A 的一块驻留",
             ours="**未试**。F143 指出我们是把 cube 的活挪给了向量 (方向相反)。"
                  "★2026-10-30 补实测依据 (F214 同机对剖 CANN FlashAttentionScore, 同形状): "
                  "我们 cube **25.2%** / 窗口 202.3ms, CANN FA cube **70.1%** / 窗口 119.7ms; "
                  "我们 GM->L1 297.6GB 对它 158.4GB (它的基本块 128x128, 我们 64)。"
                  "F215 读 Laser 源码坐实: 重缩放与最终归一化都走 cube 对角阵矩乘, "
                  "AIV 因此不碰 O 累加器 -> 一趟在线 softmax -> 不需 64KB 的 fp32 chunk 分数缓冲 "
                  "-> UB 腾出 -> 基本块才做得到 128x128。**128 是结果, 搬到 cube 才是前提。**",
             ctx="**观察于**: " + _CTX_LASER + "; **拟用于**: " + _CTX_VC + " —— ★两者契约不同 (FP16 对 INT8/E4M3)、块大小不同 (128x128 对 64x256), 机制可迁移不等于代价可迁移, 搬过去之前先核目标通路 cube 的余量。",
             ext="MindIE Laser: VectorForward.h:464-474 form_diagmat / CubeForward.h:201,:353 "
                 "diag_matmul; L1 专留 l1_local_diag_ping_/pong_ 各 64KB。2026-10-29 读源码确认。"),
        dict(kind="搬", name="行归约 (rowsum) -> 与 ones 向量矩乘搬到 cube",
             how="rowsum = S x 1, 用矩乘代替 AIV 上的全宽归约。",
             cost="aic_cube",
             ours="**未试**。",
             ctx="**观察于**: " + _CTX_LASER + "; **拟用于**: " + _CTX_VC + " —— ★两者契约不同 (FP16 对 INT8/E4M3)、块大小不同 (128x128 对 64x256), 机制可迁移不等于代价可迁移, 搬过去之前先核目标通路 cube 的余量。",
             ext="Laser 的 PhyAddrCube2Rowsum 结构体带 rowsum_out 字段 (matmul_const.h), "
                 "即 cube 调用直接产出行和。★**只读到结构体, 未读到调用点**, 记为待证。"),
        dict(kind="减", name="反量化移出 AIV",
             how="int8 x int8 的 QK 经 fixpipe 的 quant 参数直出 fp16, 省掉 AIV 上的 "
                 "int32->fp32 Cast 与逐量化块 Muls 两条全宽指令。",
             cost="aic_fixpipe (但同时把输出位宽减半, 见下条) ; **改数值契约**",
             ours="**已试** (F62 机理确认 / H12 实测): 141 用例过, 但只 -2.9%~-5.2%, "
                  "且 decoded_reference 1.1089% 过不了 1% 精度闸。",
             ctx=_CTX_VC,
             ext="CANN prompt_flash_attention 在用 SetQuantScalar (F62 旁证)。"),
    ],
    "aic_fixpipe": [
        dict(kind="减", name="中间结果位宽 int32 -> half",
             how="QK 结果以 2 字节而不是 4 字节写回, fixpipe 搬运量直接减半。",
             cost="精度 (S 由精确 int32 变 fp16) —— **改数值契约**",
             ours="**已试** (H12), 见上条, 被精度闸拦住。",
             ctx=_CTX_VC,
             ext="Laser CubeForward<half,false,half> workspace 即 half; CANN FA 同样以 T "
                 "(fp16 路径 2 字节) 写回 —— F63 早已指出本算子是 4 字节。"),
        dict(kind="★判定", name="fixpipe 绑定时的决策规则 (2026-10-29 实测定)",
             how="fixpipe 是 L0C 出口的**唯一**通路, 工作量**搬不走**, 只能减; 减有两种: "
                 "(a) 更少的排空次数 (放大基本块) (b) 更少的每次字节 (收窄输出 dtype)。",
             cost="(a) 受 API tiling 上限约束 (b) **改数值契约**",
             ours="本算子实测: m32->m64 (tile 面积 2 倍) 给出 fixpipe **x0.67**、窗口 **x0.49**; "
                  "即 fixpipe **既不随 tile 数线性 (那该是 x0.5) 也不是常数**。"
                  "★**不得外推**: 同一算子的 N 轴实测 128->256 为 -13.9% 而 256->512 为 **+1.0%** "
                  "(F92), 非单调。每一档都要实测。"
                  "★本算子两个轴**都已到 API 上限** (M<=64 受 L0C, N<=256 受 PV 的 K, F116), "
                  "故 (a) 已耗尽; (b) 已实现 (H12) 且机理成立, 但只 -2.9~-5.2% 且过不了 1% 精度闸。"
                  "**结论: 本契约下 fixpipe 不可再降 —— 这是契约决策, 不是工程决策, 应上报而非继续优化。**",
             ctx=_CTX_VC,
             ext="先用 host 侧 standalone tiling 探针问 API 允不允许 (2 秒, 不编内核不占卡, F116 的做法), "
                 "再决定要不要投入 —— 本算子正是靠它一轮内关掉两个轴。"),
        dict(kind="减", name="基本块放大以降低 fixpipe 次数",
             how="fixpipe 时间不与字节成正比 (F115 已证伪字节模型)。"
                 "★但它**也不与排空次数成正比** (2026-10-29 同产物两臂实测: tile 数 2 倍而 "
                 "fixpipe 只 1.487 倍)。两端模型都不成立, 且因两臂里 tile 数与每 tile 字节 "
                 "同步反向变化, **不可分别拟合系数** (E10)。块放大仍有效, 但**幅度必须逐档实测**。",
             cost="L0C/L0A/L0B 容量; 尾块与波量化",
             ours="**未成** (F116: M=128 即使 S 改 fp16 仍 rc=-1; N=512 被 PV 的 K<=256 挡住)。"
                  "★但那是**实现约束**, 不是硬件约束 —— 见 ext。",
             ctx=_CTX_VC,
             ext="Laser 的基本块就是 128x128, 且 L0AB ping-pong 32KB / L0C ping-pong 64KB "
                 "恰好各装一块, L1 512KB 切 8x64KB 全用满 (matmul_const.h, CubeForward.h:85-110)。"),
    ],
    "aic_cube": [
        dict(kind="减", name="QK 改 int8 量化矩乘",
             how="int8 cube 吞吐约为 fp16 的 2 倍。",
             cost="aiv_vec (反量化) 与 aic_fixpipe (int32 输出加宽); **改数值契约**",
             ours="**已做, 而且这正是本算子的路线**: cube 82729 -> 50767 (省 3.2 万 us)。"
                  "★但我们**不是 cube 受限**的 (cube 占窗口 18.1%), 所以这笔交易净亏 ——"
                  "fixpipe 与 aiv_vec 合计亏 14.2 万 us。",
             ctx=_CTX_VC,
             ext="★**Laser 是 cube 受限的** (下界 62.7ms 就是它的 cube, 且 100% fp16)。"
                 "同样一刀打在它身上才会赚。2026-10-29 msOpProf 实测。"),
    ],
}


def push_work_moves(busy: dict, wait: dict = None) -> int:
    """采样之后**主动推**候选手段 —— 不等人想起来去 `--moves` 查表。

    沉淀动机 (2026-10-30 用户指出, 当场复盘出实例):
      `_WORK_MOVES` 里 2026-10-29 就收了"行重缩放 -> 对角阵矩乘搬到 cube", 连
      Laser 的行号都写全了 (CubeForward.h:201,:353)。而 2026-10-30 我重新读了一遍
      Laser 源码, 把同一条结论**从头推了一次** —— 表在, 我没查。
      ★**拉取式的表等于没有表**: 它只在"我已经想到要查"时才起作用, 而那正是不需要它的时候。
      故本函数挂在 `profile_pipes` 的末尾, 让"采样"与"看候选"成为同一个动作。

    ★同时打出**目标通路的余量** —— "搬"的前提是接收方闲着。
      这条规则本身是用户 2026-10-30 总结的: **把工作从瓶颈单元挪到空闲单元**。
      实例: 我们 cube 占 25.2% 而 CANN FA 占 70.1%, 同一形状它快 1.7 倍 (F214);
      Laser 正是把 softmax 的重缩放与归一化用对角阵矩乘搬给了 cube (F215)。
    """
    if not busy:
        print("  ⇒ 工作量再分派: 本次没有忙时比, 跳过")
        return 0
    top_raw = max(busy.items(), key=lambda kv: kv[1])[0]
    key = top_raw[:-6] if top_raw.endswith("_ratio") else top_raw
    cands = work_moves(key)
    print(f"  ⇒ 工作量再分派 (binding = {key}):", end="")
    if not cands:
        print(f" 本表未收录 {key}。")
        print(f"      ★空表不等于「没有办法」, 只等于**本表没收**。已收录: {sorted(_WORK_MOVES)}")
        return 0
    print(f" {len(cands)} 条候选")
    for m in cands:
        print(f"      [{m['kind']}] {m['name']}")
        print(f"            代价 -> {m['cost']}   | 本仓: {m['ours'][:46]}")
    #  "搬"的前提: 目标通路要有余量。把最闲的几条打出来, 让它成为同一屏里的事实。
    idle = sorted(busy.items(), key=lambda kv: kv[1])[:3]
    txt = " · ".join(f"{k[:-6] if k.endswith('_ratio') else k} {v*100:.1f}%" for k, v in idle)
    print(f"      ★「搬」的前提是**接收方有余量**。本次最闲的通路: {txt}")
    print(f"      ★「减」的前提是**契约允许**。两类的下一步不同, 不要混着选。")
    print(f"      详情 (机理/代价/本仓试过没有/外部实例): python -m core.perf --moves {key}")
    return 0


def untried_moves() -> int:
    """列出**尚未在本仓试过**的手段 —— 可直接抄进下一轮的任务简报。

    ★动因同上: 压缩上下文之后, 能活下来的只有简报。把"未兑现的线索"做成一条命令,
      简报就能机械地带上它, 而不靠我记得。
    """
    n = 0
    print("=== 尚未在本仓试过的工作量再分派手段 ===")
    for pipe, items in sorted(_WORK_MOVES.items()):
        for m in items:
            if "未试" not in m["ours"]:
                continue
            n += 1
            print(f"  [{pipe}] [{m['kind']}] {m['name']}")
            print(f"        机理: {m['how'][:78]}")
            print(f"        代价: {m['cost']}")
            print(f"        实例: {m['ext'][:78]}")
    if not n:
        print("  (没有未试条目)")
    else:
        print(f"\n共 {n} 条。★每条都要先核 ctx 的适用条件再动手 —— "
              f"机制可迁移不等于代价可迁移。")
    return 0


def selftest_push_moves() -> int:
    """推送式候选手段的正负例。

    ★负例二 (binding 不在表里) 是关键: 旧的拉取式接口在这种情况下**什么也不说**,
      而"沉默"会被读成"没有办法"。推送式必须明说"本表没收"。
    """
    import io, contextlib
    bad = 0
    print("=== push_work_moves 自检 ===")
    cases = [
        ("正例: binding 在表里 -> 必须列出候选",
         {"aiv_vec_ratio": 0.68, "aic_cube_ratio": 0.25}, ["行重缩放", "工作量再分派"]),
        ("★负例: binding 不在表里 -> 必须明说「本表没收」而不是沉默",
         {"aic_mte2_ratio": 0.50, "aic_cube_ratio": 0.25}, ["本表未收录", "不等于"]),
        ("正例: 必须打出「最闲的通路」(「搬」的接收方)",
         {"aiv_vec_ratio": 0.68, "aic_cube_ratio": 0.25}, ["最闲的通路", "aic_cube 25.0%"]),
        ("正例: 空输入不崩, 明说跳过", {}, ["跳过"]),
    ]
    for title, busy, musts in cases:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            push_work_moves(busy)
        out = buf.getvalue()
        ok = all(m in out for m in musts)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {title}")
    #  未试清单: 必须只列 ours 含「未试」的条目
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        untried_moves()
    out = buf.getvalue()
    tried = [m["name"] for items in _WORK_MOVES.values() for m in items if "未试" not in m["ours"]]
    ok = all(t not in out for t in tried) and "行重缩放" in out
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} untried 只列未试条目 (已试的 {len(tried)} 条不得出现)")
    print(f"perf push_moves 自检: {5} 条, {bad} 条未过")
    return 1 if bad else 0


def work_moves(pipe: str) -> list:
    """按 binding pipe 查「工作量再分派」的候选手段。查不到返回空表。"""
    return _WORK_MOVES.get(pipe, [])


def print_work_moves(pipe: str = "") -> int:
    pipes = [pipe] if pipe else sorted(_WORK_MOVES)
    unknown = [q for q in pipes if q not in _WORK_MOVES]
    if unknown:
        print(f"没有收录 {unknown} 的手段。已收录: {sorted(_WORK_MOVES)}")
        print("★空表不等于'没有办法', 只等于**本表没收**。收录门槛是四项写全 (机理/代价/本仓/实例)。")
        return 1
    for q in pipes:
        print(f"\n=== binding pipe = {q} 时的候选手段 ({len(_WORK_MOVES[q])} 条) ===")
        for m in _WORK_MOVES[q]:
            print(f"  [{m['kind']}] {m['name']}")
            print(f"      机理: {m['how']}")
            print(f"      代价: {m['cost']}")
            print(f"      本仓: {m['ours']}")
            print(f"      实例: {m['ext']}")
            if m.get("ctx"):
                print(f"      ★适用条件 (binding pipe 只是检索入口, 不是全部): {m['ctx']}")
    print("\n★这张表给的是**候选**, 不是结论。按 P9/P10: 降低 binding pipe 是越过下界的"
          "必要条件, 会不会落到墙钟上只能消融。")
    return 0


def _find_opprof(out_dir: str) -> str | None:
    cands = glob.glob(os.path.join(out_dir, "OPPROF_*")) + glob.glob(os.path.join(out_dir, "*", "OPPROF_*"))
    return sorted(cands)[-1] if cands else None


#  ★msprof 的产物文件名有**两种形态**, 随 CANN 版本而变:
#     多 launch: <OPPROF>/<launch序号>/PipeUtilization_<核名>.csv
#     单 launch: <OPPROF>/PipeUtilization.csv            (9.1.0-beta.1 实见)
#  首版只 glob 前一种。2026-09-22 实撞的代价极大: 对 sparse_flash_attention 跑 msprof,
#  extract 抛"未找到 PipeUtilization_*.csv", 我把它读成**「msprof 对 mix 核返回空记录」**,
#  并据此在整个会话里**放弃了 msprof**, 改用手搓的一次性环境开关做三次消融。
#  真相是产物就在那里 (28 个 block 的 PipeUtilization / ResourceConflictRatio 一应俱全),
#  只是文件名没有下划线后缀。
_PU_PATTERNS = ("PipeUtilization_*.csv", "PipeUtilization.csv")


def _find_csv_any(root: str, stem: str) -> list:
    """递归版: 整棵树里找 <stem>_*.csv 或 <stem>.csv。"""
    out = []
    for pat in (f"{stem}_*.csv", f"{stem}.csv"):
        out += glob.glob(os.path.join(root, "**", pat), recursive=True)
    return sorted(out)


def _find_csv(d: str, stem: str) -> list:
    """按 stem 找 msprof 的 csv, **两种文件名形态都认** (<stem>_*.csv 与 <stem>.csv)。

    收成一个函数是因为同一个坑在三处各写了一遍 (PipeUtilization / OpBasicInfo /
    ResourceConflictRatio)。改 glob 只治一处, 收成函数才治住这一类。
    """
    return sorted(glob.glob(os.path.join(d, f"{stem}_*.csv")) + glob.glob(os.path.join(d, f"{stem}.csv")))


def _metric_csv(d: str) -> list:
    """一次 launch 目录里的**任何一种** metric csv (按 pipe 分解的那一类), 按优先级排。

    ★2026-10-21: 原来只认 PipeUtilization (`_PU_PATTERNS`)。于是
      `--metrics Memory` / `--metrics L2Cache` 采完之后, `extract` 报
      "未找到 PipeUtilization_*.csv", 而同一条报错里自己列出了
      "目录里**确实有** csv: L2Cache.csv, Memory.csv, OpBasicInfo.csv" ——
      信号早就补上了, 取数层却没跟上。这两个档位正是回答"搬了多少字节、命中率多少、
      带宽用了几成"的唯一来源 (见 SMLA: 它们才说清了 gather 每次指令只搬 568 B)。
      判据: 有 `sub_block_id` 列的 csv 就是按 pipe 分解的那一类, 与文件名无关。
    """
    cands = list(_find_csv(d, "PipeUtilization"))
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        stem = os.path.basename(p).split("_")[0].split(".")[0]
        if stem in ("PipeUtilization", "OpBasicInfo") or p in cands:
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                if "sub_block_id" in f.readline():
                    cands.append(p)
        except Exception:
            continue

    def _rank(p: str):
        """按**含多少条 ratio 列**排, 不按文件名 —— 同一次采集里 L2Cache.csv 也带
        sub_block_id 但只有缓存命中列, 挑到它会让忙时比整张表变成空的。"""
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                cols = [c.strip() for c in f.readline().split(",")]
        except Exception:
            return 0
        return sum(1 for c in RATIO_COLS if c in cols)

    return sorted(cands, key=lambda p: (-_rank(p), 0 if "PipeUtilization" in p else 1))


def _launch_dirs(opprof: str) -> list:
    dirs = set()
    for pat in _PU_PATTERNS + ("Memory*.csv", "L2Cache*.csv", "ResourceConflictRatio*.csv"):
        dirs |= {os.path.dirname(p)
                 for p in glob.glob(os.path.join(opprof, "**", pat), recursive=True)}
    return sorted(dirs, key=lambda d: int(os.path.basename(d)) if os.path.basename(d).isdigit() else 1 << 30)


def _what_is_there(opprof: str, limit: int = 12) -> str:
    """找不到时报出**实际有什么** —— 让「查不到」不可能伪装成「不存在」。

    这是本会话第四次栽在同一形态上 (证人核实静默跳过 / golden 取不到按通过计 /
    泛化闸中止但分母不变 / 这次)。前三次都是我读到一个空值就当成"没有",
    而工具对两种情形返回了同一个值。所以这里不只是改 glob, 是把**信号补上**。
    """
    names = sorted({os.path.basename(p)
                    for p in glob.glob(os.path.join(opprof, "**", "*.csv"), recursive=True)})
    if not names:
        return "该目录下**一个 csv 都没有** —— 多半是 msprof 没跑成或输出目录给错了"
    head = ", ".join(names[:limit]) + (f" ... 共 {len(names)} 个" if len(names) > limit else "")
    return f"目录里**确实有** csv: {head}"


def _parse_launch(d: str):
    _csvs = _metric_csv(d)
    if not _csvs:
        raise SystemExit(f"未找到任何按 pipe 分解的 metric csv 于 {d} —— {_what_is_there(d)}")
    pu = pd.read_csv(_csvs[0])
    pu.columns = [c.strip() for c in pu.columns]
    sub = pu["sub_block_id"].astype(str)
    cube = pu[sub.str.contains("cube", case=False, na=False)]
    vec = pu[sub.str.contains("vector", case=False, na=False)]

    def _col_mean(col):
        if col not in pu.columns:
            return None
        src = cube if col.startswith("aic_") else vec
        v = src[col].dropna()
        return float(v.mean()) if len(v) else None

    ratios = {c: _col_mean(c) for c in RATIO_COLS}
    times = {c: _col_mean(c) for c in TIME_COLS}
    #  ★每标志的等待**时间**列在这张表里, 按正则发现而不按编号写死 (见 _WAIT_ID_RE 上方)。
    _id_waits = {c: _col_mean(c) for c in pu.columns if _WAIT_ID_RE.match(c)}
    #  ★等待比住在 **ResourceConflictRatio.csv**, 不在 PipeUtilization 里。
    #    首版从 pu 的表里找这些列, 列名根本不存在 -> waits 全空, 而"全空"又长得像
    #    "这颗芯片不报等待比"。同一个"查不到 == 不存在"的坑, 第三处。
    #  ★两族**不同的量**, 不要混: WAIT_COLS 是等待**时间**(us, 另一组 metrics 才有),
    #    这份产物给的是等待**比**(aic_cube_wait_ratio 等)。有哪族收哪族, 缺的族要报出来。
    waits = {}
    _rc = _find_csv(d, "ResourceConflictRatio")
    if _rc:
        rc = pd.read_csv(_rc[0])
        rc.columns = [c.strip() for c in rc.columns]
        _rsub = rc["sub_block_id"].astype(str) if "sub_block_id" in rc.columns else None
        _want = list(WAIT_COLS) + [c for c in rc.columns if c.endswith(("_wait_ratio", "_cflt_ratio"))]
        _missing = [c for c in WAIT_COLS if c not in rc.columns]
        if _missing:
            #  ★不静默: 说清楚缺的是哪一族、这份产物给的是哪一族。
            waits["__缺列__"] = (f"期望的等待**时间**列 {len(_missing)} 个不在这份产物里 "
                               f"(需 --aic-metrics 采到它们); 本份给的是等待**比**, 已按比收")
        #  ★分路前先查一次取值空间 (见 rc_sub_unknown 上方的实测注释)。报在循环外,
        #    否则每列重复一遍。没见过的取值会被两条分路同时丢掉, 不许它静默消失。
        if _rsub is not None:
            _unk = rc_sub_unknown(_rsub.unique())
            if _unk:
                waits["__未知分路__"] = (
                    f"`sub_block_id` 出现没见过的取值 {_unk[:5]} —— 这些行既不归 cube "
                    f"也不归 vector, **被两条分路同时丢掉**; 本语料此前只有 "
                    f"{'/'.join(_RC_SUB_SEEN)} 三种")
        for c in _want:
            if c not in rc.columns:
                continue
            src = rc
            if _rsub is not None:
                src = rc[_rsub.str.contains("cube" if c.startswith("aic_") else "vector",
                                            case=False, na=False)]
            v = src[c].dropna()
            waits[c] = float(v.mean()) if len(v) else None
    else:
        waits = {c: None for c in WAIT_COLS}
    #  ★发现到的每标志等待时间并进来。它们与上面那族 (等待比) 是两回事, 不互相替代。
    waits.update(_id_waits)
    if not _id_waits:
        waits["__无每标志等待__"] = (
            "PipeUtilization 里一个 `*_scalar_wait_id<N>_time(us)` 列都没有 —— "
            "要么该内核不用跨核标志, 要么采集时没开相应 metrics。"
            "**不要把它读成'等待没有变化'** (2026-10-27 实撞就是这样漏掉了 535 万周期的变化)")
    dur, block, opname, ttype = None, 0.0, "?", "?"
    ob = _find_csv(d, "OpBasicInfo")
    if ob:
        o = pd.read_csv(ob[0])
        o.columns = [c.strip() for c in o.columns]
        dc = next(c for c in o.columns if "Task Duration" in c)
        dur = float(o[dc].iloc[0])
        block = float(o["Block Dim"].iloc[0]) if "Block Dim" in o.columns else 0.0
        opname = str(o["Op Name"].iloc[0]) if "Op Name" in o.columns else "?"
        ttype = str(o["Op Type"].iloc[0]) if "Op Type" in o.columns else "?"
    return dur, block, opname, ttype, ratios, times, waits


#  ★`sub_block_id` 的取值空间 (2026-10-22 全语料实测): 11028 份 ResourceConflictRatio.csv,
#    759898 行, 取值**只有三种** —— cube0(255418) / vector0(262320) / vector1(242160),
#    既不含 cube 也不含 vector 的 **0 行**。所以下面按子串分路在本语料上不误分组。
#    但子串匹配仍是在赌命名约定, 按 expdesign E14 "分组函数也是判据, 要负例检验":
#    **不改分法** (改了反而可能漏掉真的 cube1/vector2), 改成**让没见过的取值显出来**。
_RC_SUB_SEEN = ("cube0", "vector0", "vector1")


def rc_sub_unknown(values) -> list:
    """返回既不归 cube 也不归 vector 的 `sub_block_id` 取值 (去重排序)。

    空列表 = 全部可分路; 非空 = 这些行会被两条分路**同时丢掉**, 必须报出来。
    """
    out = set()
    for v in values:
        if v is None:
            continue
        t = str(v)
        if t.lower() in ("nan", ""):
            continue
        if "cube" not in t.lower() and "vector" not in t.lower():
            out.add(t)
    return sorted(out)


#  ★默认同时采 PipeUtilization (忙时) 与 ResourceConflictRatio (**等待比**)。
#    只采忙时会把"果"当成"因" —— FIAS 2026-09-14 实撞, 因此白做了五轮:
#      按忙时排序 mte2 是"第二长杆", 于是去减它的搬运量 (−49%) -> 端到端 0;
#      减 AIV 工作量 41.87us -> −0.08%; 加缓冲深度 -> 0; unitFlag -> 0; 去跨核旗标 -> 0。
#    最后采了 wait_ratio 才一步定位: aic_mte2 0.599 (全场最低 = 因), cube/mte1/AIV 全 >0.94 (果)。
#    判读见 docs/perf-ceiling-estimation.md §5。采一次的代价只是多一组计数器, **不值得省**。
_TIMELINE_ALIASES = ("TimelineDetail", "PipeTimeline")
_timeline_cache = {}


def timeline_metric() -> str:
    """本机 msprof 认哪个**流水时间线**档位名 —— 按**实际 SoC** 定, 不按 --help 的顺序猜。

    ★2026-09-20 立: 拿到
      `--aic-metrics=pipetimeline: Unexpected argument, maybe in wrong soc platform`
      就判成"本平台不支持流水时间线", 差点放弃唯一能回答"忙还是等"的手段。
      而本模块的 REFS 里 **2026-09-17 就逐字记过同一个误判**。
      于是把它变成代码: 名字由 `--help` 决定。950PR 叫 PipeTimeline, 910B2 叫 TimelineDetail。

    ★★2026-10-21 打补丁 —— 上面那条**只做了一半**: `msprof op --help` 在 950PR 上把
      **两个名字都列出来** (它是一份跨 SoC 的通用清单), 而 `_TIMELINE_ALIASES` 里
      TimelineDetail 排在前, 于是"问 --help"在这台机器上**每次都返回错的** TimelineDetail:
        实撞: `PipeTimeline -> TimelineDetail`, 随后 msprof 报
        `Unexpected argument --dump` (那是 TimelineDetail 专用的落盘开关), 整趟采集 rc!=0。
      help 文本里出现 ≠ 本机支持 —— 能区分平台的只有 SoC 自己。
      同一条病根第二次: 把"能不能"判成了"名字在不在清单里"。
    """
    if "name" in _timeline_cache:
        return _timeline_cache["name"]
    name = ""
    soc = ""
    try:
        from core import socspec
        soc = str(socspec.spec().get("soc") or "")
    except Exception:
        pass
    s = soc.lower()
    if "950" in s:
        name = "PipeTimeline"
    elif "910b" in s:
        name = "TimelineDetail"
    if not name:
        #  认不出 SoC 时才退回问 help, 并且**优先 PipeTimeline** —— 新名字在旧平台上被拒,
        #  旧名字在新平台上也被拒, 但新平台的概率更高 (本仓主战场是 950)。
        try:
            h = subprocess.run(["msprof", "op", "--help"], capture_output=True, text=True,
                               timeout=30).stdout
            for a in ("PipeTimeline", "TimelineDetail"):
                if a in h:
                    name = a
                    break
            if name:
                print(f"★认不出 SoC (socspec 给的是 {soc!r}), 退回按 --help 猜档位名 = {name}。"
                      f"**这是猜的** —— 采集若报 'maybe in wrong soc platform', 换另一个名字再试。",
                      flush=True)
        except Exception:
            pass
    if not name:
        print("★问不到本机的流水时间线档位名 (so 认不出, 且 msprof op --help 里既无 "
              "TimelineDetail 也无 PipeTimeline)。**这不等于不支持** —— 先手工敲一次 --help "
              "再下结论。", flush=True)
    _timeline_cache["name"] = name
    return name


def is_timeline(metrics: str) -> bool:
    """这组 metrics 要不要流水时间线 (任一别名都算)。"""
    return any(a.lower() in (metrics or "").lower() for a in _TIMELINE_ALIASES)


def normalize_metrics(metrics: str) -> str:
    """把任何别名换成**本机认的**那个名字。问不到就原样返回, 让 msprof 自己报错。"""
    if not is_timeline(metrics):
        return metrics
    real = timeline_metric()
    if not real:
        return metrics
    out = []
    for part in (metrics or "").split(","):
        out.append(real if part.strip().lower() in [a.lower() for a in _TIMELINE_ALIASES]
                   else part)
    return ",".join(out)


DEFAULT_AIC_METRICS = "PipeUtilization,ResourceConflictRatio"


def profile_pipes(app_cmd: str, out_dir: str, kernel_name: str = "",
                  launch_count: int = 10, skip: int = 5,
                  metrics: str = DEFAULT_AIC_METRICS) -> dict:
    """★一条命令拿到**忙时比 + 等待比**。有工具就用工具, 不要靠猜 + 手搓开关。

    2026-09-22 的代价: PV 段高出下界 21 倍, 我先后猜了"核发射开销""L1 容量没用满"
    "计算还是搬运", 为此手搓三个一次性环境开关、做了三次构建, 全部不结论;
    最后 msprof 一跑两分钟就给出答案 —— 矩阵 84% 在等、MTE2 忙 69%、标量忙 47%。
    没有先用它, 是因为取数层的文件名模式没对上, 而我把"取数层报错"读成了
    "msprof 对这类核没有数据"(见 REFS PF0)。

    所以这条命令存在的意义是**让工具路径成为最省力的路径** —— 只要还得手拼 msprof
    再调 extract, 下次仍然会先去猜。

    ★**2026-10-22 改默认: `3/1` → `10/5`** (与同模块 `run_msprof_op` 对齐)。
    全仓 1656 个 launch 的实测 (`docs/msprof-re/tools/passdev.py`, 01 §13.18):
    十遍重放里出现离群遍的比例**按 launch 序号单调衰减** ——
      launch 0:27%  1:23%  2:22%  3:18%  4:7%  5:2%  6 之后:0~1%
    而旧默认 `launch_count=3, skip=1` 采的正是 0/1/2 这三个**离群率最高**的 launch。
    一个为"让工具路径最省力"而建的入口, 落在了噪声最大的窗口里。
    ★按**遍次**序号则是平的 (2130~3000 次命中), 所以决定性的是**第几次 launch**,
    不是第几遍 —— 调 `--aic-metrics` 的档位顺序没有用。
    """
    d = run_msprof_op(app_cmd, kernel_name, out_dir, launch_count, skip,
                      reuse=False, metrics=metrics)
    #  ★时间线档位的产物**一张 metric CSV 都没有** (只有 trace.json + OpBasicInfo.csv),
    #    继续走 extract 会报"未找到 PipeUtilization_*.csv" —— 采集成功却报失败。
    #    2026-10-21 之前这只有文档里的一句话("PipeTimeline 出 trace.json, 不是 CSV"),
    #    没有任何调用路径体现它, 于是 `--pipes --metrics PipeTimeline` 是个**必定失败**的组合。
    if is_timeline(metrics):
        print_timeline(d)
        return {"op": "", "busy": {}, "wait": {}, "opprof": d, "timeline": True}
    pd_ = extract(d)
    busy = {k: v for k, v in pd_.ratios.items() if isinstance(v, float) and v > 0.005}
    wait = {k: v for k, v in pd_.waits.items() if isinstance(v, float) and v > 0.005}
    print(f"核 {pd_.op_name}  类型 {pd_.task_type}  blockDim {pd_.block_dim:.0f}  "
          f"单次 {pd_.task_duration_us}us")
    print("  忙时比 (谁在干活):")
    for k, v in sorted(busy.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<26}{v * 100:6.1f}%")
    if wait:
        print("  等待比 (谁在等 —— **关键路径看这个**):")
        for k, v in sorted(wait.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<26}{v * 100:6.1f}%")
    else:
        print("  等待比: 本次 metrics 没采到 —— 加 --metrics PipeUtilization,ResourceConflictRatio")
    if busy:
        tb = max(busy.items(), key=lambda kv: kv[1])
        print(f"  ⇒ 忙的是 {tb[0]} ({tb[1] * 100:.1f}%)")
    if wait:
        tw = max(wait.items(), key=lambda kv: kv[1])
        print(f"  ⇒ 等得最久的是 {tw[0]} ({tw[1] * 100:.1f}%) —— "
              f"★削「忙的那条」未必减墙上时间, 关键路径在**等**的那一侧 (见 REFS 里 gmfr 那次中性实验)")
    #  ★采样之后**主动推**候选手段: 拉取式的表只在"已经想到要查"时起作用, 而那正是不需要它的时候。
    push_work_moves(busy, wait)
    return {"op": pd_.op_name, "busy": busy, "wait": wait, "opprof": d}


def selftest_multi_kernel_guard() -> int:
    """多核产物必须拒跑; 单核与 dump/simulator 同级目录不许误伤。

    正例比负例要紧: 本仓 338 份产物里 41 份含 dump/simulator 同级目录, 只有 21 份
    是真的多核。把前者误判成多核, 等于把一条天天在用的路径堵死。
    """
    import tempfile
    import shutil
    bad = 0
    print("=== perf 多核产物守卫 自检 (4 条) ===")
    with tempfile.TemporaryDirectory() as d:
        def mk(root, subs):
            for s in subs:
                os.makedirs(os.path.join(root, s, "0"), exist_ok=True)
                with open(os.path.join(root, s, "0", "PipeUtilization_1.csv"), "w") as fh:
                    fh.write("block_id,sub_block_id,aic_time(us)\n0,cube0,1.0\n")
            return root
        one = mk(os.path.join(d, "one"), ["MatMulV2_x"])
        two = mk(os.path.join(d, "two"), ["MatMulV2_x", "Add_y"])
        withdump = mk(os.path.join(d, "wd"), ["MatMulV2_x", "dump", "simulator", "device0"])
        checks = [
            ("正例: 单核 -> 正常出数", one, None, True),
            ("负例: 双核不指定 -> 必须拒跑", two, None, False),
            ("正例: 双核指定其一 -> 正常出数", two, "Add_y", True),
            ("正例: dump/simulator/device0 不算核 -> 不许误伤", withdump, None, True),
        ]
        for name, root, kern, want_ok in checks:
            try:
                extract_signals(root, kern)
                got_ok = True
            except SystemExit:
                got_ok = False
            ok = (got_ok == want_ok)
            bad += 0 if ok else 1
            print(f"  {'过' if ok else '**未过**'} {name}  "
                  f"({'出数' if got_ok else '拒跑'})")
    print(f"perf 多核守卫自检: {len(checks)} 条, {bad} 条未过")
    return 1 if bad else 0


def ratio_window_us(op_type: str, task_duration_us: float,
                    total_cycles: float | None = None,
                    freq_mhz: float | None = None) -> float:
    """`*_ratio` 这一列的分母, 换算成微秒。**分母有两种, 由 `Op Type` 决定。**

    ★2026-10-23 全仓反解 (1598 个 kernel 目录, 逐行 `D = ratio 的分子周期数 / ratio`):

    | `Op Type` | 分母 | 目录数 | 反例 |
    |---|---|---|---|
    | `mix` | **整个任务的常数** `floor(TaskDur) x freq` | 1508 | 0 |
    | `cube` / `vector` | **本行自己的** `<aic\|aiv>_total_cycles` | 90 | 0 |

    判法不是投票: 逐行反解出 D 之后, 目录内 D 的极差/中位数在 `mix` 上是 **5e-6**
    (常数), 在纯 cube/vector 上是 **9e-2** (逐行变)。两族各自对上自己的候选到 1e-6
    以内, 对不上对方到 2e-2 以上 —— 不是容差问题。

    ★**为什么必须分开**: 两个分母在 `mix` 上几乎相等 (`floor(TaskDur)xfreq / total_cycles`
    中位 1.0017), 所以在 gmfr 上做的那次解只能看见其中一个; 而在纯 **vector** 算子上
    两者中位差 **1.24 倍, 最大 1.79 倍**。拿 `mix` 上解出的式子去读一个 `Add`,
    折算出的墙钟会高估近一倍。

    ★不给 `op_type` 不替调用方猜 —— 猜错的代价是上面那个 1.79 倍, 而两种写法都"跑得通"。
    """
    t = (op_type or "").strip().lower()
    if t == "mix":
        return math.floor(task_duration_us)
    if t in ("cube", "vector", "aiv", "aic"):
        if not total_cycles or not freq_mhz:
            raise ValueError(
                f"Op Type={op_type!r} 的 *_ratio 分母是本行 total_cycles, "
                f"必须同时给 total_cycles 与 freq_mhz (拿 floor(TaskDur) 顶替在纯向量算子上最大错 1.79 倍)")
        return total_cycles / freq_mhz
    raise ValueError(f"未知 Op Type {op_type!r} —— 分母取哪个没有依据, 不猜 (见 08 §3.82)")


def cflt_wall_us(cflt_ratio: float, task_duration_us: float, op_type: str,
                 total_cycles: float | None = None,
                 freq_mhz: float | None = None) -> float:
    """UB 冲突比 -> 它占掉的墙钟微秒数 = `ratio x ratio_window_us(...)`。

    分母是任务窗口 (08 §3.42 实测定论), **不是** aiv_vec_time。做成函数是因为这一步
    此前写错过一次, 而错的方向是**低估 5.7 倍** —— 一个被低估的瓶颈不会报警,
    只会让人去优化别处。

    ★**2026-10-23 订正两次**:
    1. 窗口要取 `floor(Task Duration)` 而不是 `Task Duration` (`08` §3.78/§3.79;
       长核差 0.007%, TaskDur=10.9us 的短核差 8.3%)。
    2. **那条只对 `Op Type == mix` 成立** —— 纯 `cube`/`vector` 算子的分母是本行
       `total_cycles`, 见 `ratio_window_us` 的表。第 1 条是在 gmfr (mix) 上解的,
       而 mix 上两个候选只差 0.17%, 所以当时分辨不出来。
    """
    return cflt_ratio * ratio_window_us(op_type, task_duration_us, total_cycles, freq_mhz)


def selftest_cflt_wall() -> int:
    """负例是旧处方: 同一组数拿 vec_time 去乘, 必须给出明显不同的答案。

    ★2026-10-23 加两条: 分母里的 `floor` 不能漏 (08 §3.78/§3.79 从原始 PMU 记录解出
    `*_ratio` 的分母是 `floor(TaskDur) x freq`)。长核上漏了只差万分之几, **短核上差
    百分之几** —— 所以正例用整数 TaskDur (看不出区别), 负例用带小数的短 TaskDur。
    """
    bad = 0
    print("=== perf UB 冲突比换算 自检 ===")
    dur, vec_t, cflt = 100.0, 17.6, 0.05     # vec_ratio = 0.176, 本机实测量级
    got = cflt_wall_us(cflt, dur, "mix")
    #  短核: TaskDur 10.9 -> floor 10, 差 9.00%; 这一条**就是漏 floor 的后果**
    short_got = cflt_wall_us(0.030798, 10.9, "mix")
    #  ★纯 vector 算子: 窗口是本行 total_cycles/freq, 不是 floor(TaskDur)。
    #    取自 cases/gemm_v3/runs/prof_probe/.../Add_93b3f8e9.../0: TaskDur 5.5us @1800MHz,
    #    某行 total_cycles = 6462 -> 窗口 3.590us, 而 floor(5.5) = 5us —— 差 1.39 倍。
    vec_win = ratio_window_us("vector", 5.5, total_cycles=6462, freq_mhz=1800.0)
    def _raises(fn):
        try:
            fn(); return False
        except ValueError:
            return True
    checks = [
        ("墙钟 = cflt × Task Duration", abs(got - 5.0) < 1e-9, f"{got:.3f} us"),
        ("★分母取 floor: 短核 TaskDur=10.9 要按 10 算",
         abs(short_got - 0.030798 * 10) < 1e-12, f"{short_got:.4f} us"),
        ("★漏 floor 在短核上偏大 9.00% —— 方向是**高估**, 会把瓶颈指错",
         abs((0.030798 * 10.9) / short_got - 1.09) < 1e-9,
         f"漏 floor {0.030798*10.9:.4f} vs 正确 {short_got:.4f}"),
        ("旧处方 (× aiv_vec_time) 偏小 5.68 倍 —— 方向必须是低估",
         abs(got / (cflt * vec_t) - dur / vec_t) < 1e-9 and got > cflt * vec_t,
         f"旧 {cflt*vec_t:.3f} us vs 新 {got:.3f} us"),
        ("零冲突 -> 零墙钟", cflt_wall_us(0.0, dur, "mix") == 0.0, "0.000 us"),
        #  ---- ★2026-10-23: 分母由 Op Type 决定, mix 与纯核不是同一个量 ----
        ("★纯 vector: 窗口 = 本行 total_cycles/freq, 不是 floor(TaskDur)",
         abs(vec_win - 6462 / 1800.0) < 1e-12, f"{vec_win:.4f} us"),
        ("★拿 mix 的式子读纯 vector 会**高估** 1.39 倍 (全仓中位 1.24, 最大 1.79)",
         abs(5.0 / vec_win - 1.3929) < 1e-3, f"floor 5.000 vs 正确 {vec_win:.4f} us"),
        ("★mix 上两个候选几乎相等 (中位 1.0017), 所以 gmfr 那次分辨不出来",
         abs(ratio_window_us("mix", 5113.340332) - 5113.0) < 1e-9, "5113 us"),
        ("★纯核缺 total_cycles 时**拒算**, 不拿 floor(TaskDur) 顶替",
         _raises(lambda: ratio_window_us("vector", 5.5)), "ValueError"),
        ("★未知 Op Type 拒算, 不猜",
         _raises(lambda: ratio_window_us("aicpu", 5.5)), "ValueError"),
    ]
    for name, ok, shown in checks:
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {name}  ({shown})")
    print(f"perf 冲突比换算自检: {len(checks)} 条, {bad} 条未过")
    return 1 if bad else 0


def product_columns(root: str) -> dict:
    """{列名: 最先见到它的 csv 基名}。只读表头, 不读数据 —— 扫几万份 csv 也秒级。"""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                hdr = next(csv.reader(fh), [])
        except OSError:
            continue
        for c in hdr:
            c = c.strip()
            if c and c not in out:
                out[c] = os.path.basename(p)
    return out


def new_columns(new_root: str, base_roots: list) -> dict:
    """采了一个新档位, 它带来了哪些**此前没见过**的列。

    为什么要工具而不是肉眼比: 一个新档位是不是"同一批列换个文件名", 肉眼比对必错;
    而"这个新档值不值得再采"只取决于**新增列**, 不取决于它一共有多少列。
    2026-10-20 实用: Roofline 与 KernelScale 各 136 列, 新增 0 ——
    它们改变的是采集行为, 不是导出面。
    """
    known = {}
    for b in base_roots:
        known.update(product_columns(b))
    new = product_columns(new_root)
    return {"known": len(known), "new_total": len(new),
            "fresh": {c: f for c, f in new.items() if c not in known}}


def selftest_new_columns() -> int:
    """正例 (自己比自己必须 0 新列) + 负例 (多一列必须被报出来)。

    正例不能省: 一个只会报"有新列"的比对器, 在"没有新列"这个**最常见也最有用**的
    结论上就是坏的 —— 而那恰恰是 2026-10-20 四档里三档的答案。
    """
    import tempfile
    bad = 0
    print("=== perf 列清单对比 自检 (3 条) ===")
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        os.makedirs(a); os.makedirs(b)
        with open(os.path.join(a, "Memory_1.csv"), "w") as fh:
            fh.write("block_id,aic_time(us),GM_to_L1_datas(KB)\n0,1.0,2.0\n")
        r = new_columns(a, [a])
        ok = (len(r["fresh"]) == 0 and r["new_total"] == 3)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 正例: 自己比自己必须 0 新列 "
              f"(实际新增 {len(r['fresh'])}, 共 {r['new_total']} 列)")
        with open(os.path.join(b, "PipeUtilization_1.csv"), "w") as fh:
            fh.write("block_id,aic_time(us),aic_mte1_active_bw(GB/s)\n0,1.0,9.0\n")
        r = new_columns(b, [a])
        ok = (set(r["fresh"]) == {"aic_mte1_active_bw(GB/s)"})
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 多出一列必须被点名 "
              f"(实际 {sorted(r['fresh'])})")
        r = new_columns(b, [os.path.join(d, "nonexistent")])
        ok = (len(r["fresh"]) == 3 and r["known"] == 0)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 边界: 基线根不存在时不许静默当成'全是旧列' "
              f"(known={r['known']}, 新增 {len(r['fresh'])})")
    print(f"perf 列清单对比自检: 3 条, {bad} 条未过")
    return 1 if bad else 0


def strip_mspti_preload(environ) -> tuple:
    """从 LD_PRELOAD 里摘掉 libmspti.so, 返回 (新 env, 是否摘过)。

    为什么要摘而不是报错退出: 调用方要的是 PMU 分解, 这一轮本来就拿不到 MSPTI 记录,
    摘掉是唯一能让它拿到想要的东西的做法。但必须由调用方**打印出来** —— 同进程里若有人
    指望 MSPTI, 不说一声就会变成零记录而没有任何提示。

    只摘这一项, 不动 LD_PRELOAD 里的其它库 (常见的还有 jemalloc / tcmalloc 之类,
    摘掉它们会改变被采集进程的行为, 那就不是同一个被测对象了)。
    """
    env = dict(environ)
    pre = env.get("LD_PRELOAD", "")
    if "libmspti.so" not in pre:
        return env, False
    rest = [x for x in pre.split(":") if x and "libmspti.so" not in x]
    if rest:
        env["LD_PRELOAD"] = ":".join(rest)
    else:
        env.pop("LD_PRELOAD", None)   # 留一个空串会让某些加载器告警
    return env, True


def selftest_mspti_preload() -> int:
    """负例自检: 肇因是 LD_PRELOAD 本身, 所以判据只认路径不认"有没有 start monitor"。"""
    bad = 0
    print("=== perf LD_PRELOAD/MSPTI 互斥守卫 自检 (4 条) ===")
    root = "/usr/local/Ascend/x/lib64/libmspti.so"
    cases = [
        ("只有 libmspti -> 整条 LD_PRELOAD 删掉 (不留空串)",
         {"LD_PRELOAD": root}, True, None),
        ("与别的库共存 -> 只摘 libmspti, 其余原样保留",
         {"LD_PRELOAD": f"/l/libjemalloc.so:{root}"}, True, "/l/libjemalloc.so"),
        ("正例: 不含 libmspti -> 一个字都不许改",
         {"LD_PRELOAD": "/l/libjemalloc.so"}, False, "/l/libjemalloc.so"),
        ("正例: 根本没有 LD_PRELOAD -> 不许凭空造一个",
         {}, False, None),
    ]
    for name, src, want_strip, want_val in cases:
        env, did = strip_mspti_preload(src)
        got = env.get("LD_PRELOAD")
        ok = (did == want_strip and got == want_val)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {name}  "
              f"(摘={did} 期望{want_strip}; 值={got!r} 期望{want_val!r})")
    print(f"perf MSPTI 守卫自检: {len(cases)} 条, {bad} 条未过")
    return 1 if bad else 0


def run_msprof_op(app_cmd: str, kernel_name: str, out_dir: str,
                  launch_count: int = 10, skip: int = 5, reuse: bool = True,
                  metrics: str = DEFAULT_AIC_METRICS) -> str:
    #  ★A 路径结构上不收 `Custom:`, 且**拒了之后退出码仍是 0** —— 不拦就表现为
    #    "采集跑完了, 产物里却没有原始计数列", 而那与"算子没被匹配上"长得一模一样。
    #    拦在下发之前, 并指出该走哪条路 (理由见 _CUSTOM_A_PATH_WHY 引的 01 §3.1c)。
    if metrics and "custom:" in metrics.lower():
        raise SystemExit("[perf] " + _CUSTOM_A_PATH_WHY)
    if reuse:
        ex = _find_opprof(out_dir) if os.path.isdir(out_dir) else None
        #  ★复用要求**两组都在**: 只有忙时的旧产物必须重采, 否则又回到只看忙时的老路。
        if ex and is_timeline(metrics):
            if glob.glob(os.path.join(ex, "**", "trace.json"), recursive=True):
                return ex
        #  ★复用判断也要认两种文件名形态 —— 同一个坑的第四处。写死 _*.csv 会让复用永远
        #    命中不了, 表现是"每次都重跑 msprof"(不致命但白等), 且与 extract 的判据不一致。
        elif ex and _find_csv_any(ex, "PipeUtilization") \
              and ("ResourceConflictRatio" not in metrics
                   or _find_csv_any(ex, "ResourceConflictRatio")):
            return ex
    os.makedirs(out_dir, exist_ok=True)
    #  ★MSPTI 与 msprof op 在同一进程内互斥, 且是**单向**的: MSPTI 照常收齐记录,
    #  msprof 这一侧报 `No profiling data dumped` + 空 OPPROF 目录。
    #  2026-10-20 实测四档定位到肇因是 **LD_PRELOAD 本身**, 与有没有 start monitor 无关
    #  (`opdbg symptom "No profiling data dumped"` 第 4 条)。
    #  这里主动摘掉而不是报错退出: 调用方要的是 PMU 分解, 这一轮本来就拿不到 MSPTI 记录,
    #  摘掉是唯一能让它拿到想要的东西的做法; 但必须**说出来**, 否则同进程里若有人
    #  指望 MSPTI, 会在没有任何提示的情况下变成零记录。
    env, _stripped = strip_mspti_preload(os.environ)
    if _stripped:
        print("  [perf] ★已从 LD_PRELOAD 摘掉 libmspti.so —— 它与 msprof op 互斥, "
              "留着会让本次采集产出空目录。本次运行拿不到 MSPTI 记录。", flush=True)
    #  ★msprof 的 --application 要的是**可执行文件路径 + 参数**, 不是 shell 命令 ——
    #  传 `cd X && python y.py` 会直接报
    #      [ERROR] application to be profiled is not exist or not executable
    #  而本函数的签名是 app_cmd: str ("命令"), 调用方自然会传命令。
    #  签名承诺的与实际接受的不一致, 就该由工具来弥合, 而不是让每个调用方自己踩一遍。
    #  (2026-09-17 实撞: 四趟采集全 rc=1, 全栽在这一条上。)
    if re.search(r"&&|\|\||;|^\s*cd\s", app_cmd):
        import stat as _stat
        _w = os.path.join(out_dir, "_app_wrapper.sh")
        with open(_w, "w", encoding="utf-8") as _f:
            _f.write("#!/bin/bash\n" + app_cmd + "\n")
        os.chmod(_w, os.stat(_w).st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH)
        app_cmd = _w
    cmd = ["msprof", "op", f"--kernel-name={kernel_name}",
           f"--launch-count={launch_count}", f"--launch-skip-before-match={skip}",
           f"--output={out_dir}", f"--application={app_cmd}"]
    if metrics:
        #  ★别名翻译放在**下发那一刻** (2026-09-20): 同一个能力在 950PR 叫 PipeTimeline,
        #  在 910B2 叫 TimelineDetail。写死名字会让调用方在另一台机器上拿到
        #  "Unexpected argument ... maybe in wrong soc platform", 进而误判成"不支持"。
        _m = normalize_metrics(metrics)
        if _m != metrics:
            print(f"  [perf] 档位名按本机翻译: {metrics} -> {_m}", flush=True)
        cmd.insert(2, f"--aic-metrics={_m}")
        #  ★TimelineDetail 专用的落盘开关 (2026-09-20): `--help` 注明它"仅在
        #  --aic-metrics=TimelineDetail 时生效"。不加它时本机采集能产出 visualize_data.bin
        #  (每核 1.79MB), 但**解析阶段**报 `aclrtSetDevice 107001` + `Parse kernel data failed`,
        #  最后只落下 OpBasicInfo_*.csv —— 数据在盘上却没被解码。
        #  ★★按**翻译后的真名**判, 不按 is_timeline(别名集合) 判 (2026-10-21 实撞):
        #    PipeTimeline **不认** --dump, 带上就是 `Unexpected argument --dump`, 整趟采集 rc!=0。
        #    写成 `if is_timeline(metrics)` 时, 任何时间线档位都会带上它 —— 别名匹配了, 平台没匹配。
    #  ★★本行里「整趟采集 rc!=0」这半句**与 2026-10-29 的实测冲突, 未解**: 同形的
    #    `msprof op --no-such-flag` 在 910B2 上给 **rc=0** (七类对照见
    #    `core/opcfg` 的 REFS['J_msprof_op_exit_code_is_not_a_failure_signal'] ——
    #    `msprof op` 的五类错误全部 rc=0, 只有顶层 `msprof` 包装器才设码 255)。
    #    这里的案例**本机复现不了** (PipeTimeline 是 950PR 的档位别名), 所以两边都没改。
    #    ★要紧的是: 若 rc 其实也是 0, 那么**靠 rc 判这趟采集成没成就是失效的**,
    #    得改成解析输出里的 `[ERROR]`。能复现时先验这一条, 再决定动哪边。
        if _m == "TimelineDetail":
            cmd.insert(3, "--dump=on")
    #  ★边打印边留存 msprof 的输出: 它把"过滤名一个都没匹配上"这件事只写在 stdout 的
    #  `Kernel <名> skipped: not selected via --kernel-name` 里, 不进任何产物。
    #  不留存就只能看到最后那句 `No profiling data dumped`, 于是归因必然落到 launch-count/skip
    #  上 —— 2026-09-17 实撞: 传了 `--kernel-name sparse_flash_attention_grad` (小写下划线),
    #  而真实核名是 `SparseFlashAttentionGrad_ecdedf...` (PascalCase), 两趟采集全废,
    #  工具却建议去调 --skip/--launch-count, 方向完全错。
    _log = []
    _pr = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
    #  ★`Kernel <名> skipped: not selected via --kernel-name` 每次 launch 每个核各一行,
    #  一趟采集能刷几十上百行。它**留存**用于归因, 但不往屏幕上打 —— 调用方为了躲这堆噪声
    #  会去接 `| grep -v`, 而那样一来退出码就变成 grep 的 (本项目已因此栽过四次)。
    #  工具自己的噪声不该逼出一个会吞退出码的调用习惯。
    _skipped = 0
    for _ln in _pr.stdout:
        _log.append(_ln)
        if "skipped: not selected via --kernel-name" in _ln:
            _skipped += 1
            continue
        sys.stdout.write(_ln)
    if _skipped:
        print(f"  [msprof] 另有 {_skipped} 行 `skipped: not selected via --kernel-name` 未打印 "
              f"(按 --kernel-name={kernel_name} 过滤掉的核; 采不到数据时会据此归因)。", flush=True)
    _rc = _pr.wait()
    _txt = "".join(_log)
    if _rc != 0:
        raise subprocess.CalledProcessError(_rc, cmd)
    d = _find_opprof(out_dir)
    if d is None:
        raise RuntimeError(f"msprof 未产出 OPPROF, 查 {out_dir}")
    #  ★OPPROF 建出来了但里面是空的 —— 最常见的原因不是环境坏了, 而是
    #    **驱动发的 kernel 次数 <= launch-skip-before-match**, 于是要采的那几次全被跳过,
    #    msprof 仍旧报 `All task success` + `No profiling data dumped`, 极具误导性。
    #    2026-09-10 实撞: 把 repro 的 --repeat 3 --warmup 1 (共 5 次 launch) 配上默认 skip=5,
    #    连着三次归错因 (先怪 /dev/shm 满, 再怪磁盘满), 各烧掉一次排查。
    #  ★不同 metric 的产物形态不同: PipeUtilization/ResourceConflictRatio 出 CSV,
    #    而 **PipeTimeline 出 trace.json + visualize_data.bin, 一张 CSV 都没有**。
    #    按 CSV 判空会把一次成功的 PipeTimeline 采集报成失败 (2026-09-14 实撞), 故按 metric 选判据。
    #  ★第三次同类失败后改成**泛化判据** (2026-09-17): 每出现一个新档位, 产物形态就不同,
    #    而校验一直在**猜文件名** —— PipeUtilization 出 CSV, PipeTimeline 出 trace.json,
    #    而 Occupancy / TimelineDetail 出的是**未解析的 DeviceProf*.bin**, 一张 metric CSV 都没有。
    #    按 CSV 判空会把一次**成功的采集**报成失败 (Occupancy 实测产出 68 个文件却被判失败)。
    #    改判"有没有实质产物", 并在只有原始 bin 时**直接告诉下一步是 msprof --export**,
    #    而不是让人以为采集坏了。
    _csvs = [f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
             if "OpBasicInfo" not in os.path.basename(f)]
    _trace = glob.glob(os.path.join(d, "**", "trace.json"), recursive=True)
    _raw = glob.glob(os.path.join(d, "**", "DeviceProf*.bin"), recursive=True)
    ok = bool(_csvs or _trace)
    if not ok and _raw:
        print(f"  [msprof] {d} 只有**未解析**的原始数据 ({len(_raw)} 个 DeviceProf*.bin), "
              f"没有 metric CSV —— 采集本身是成功的, 缺的是导出这一步。\n"
              f"           下一步: msprof --export --output {d}\n"
              f"           (PipeUtilization 会自动解析, Occupancy / TimelineDetail 不会。)",
              flush=True)
        return d
    want = ("trace.json" if is_timeline(metrics)
            else "任何 metric CSV / trace.json / DeviceProf*.bin")
    if not ok:
        #  ★先判**过滤名有没有匹配上** —— 它与 launch-count/skip 是两种不同的"采不到",
        #    而 msprof 对两者打印的最后一句完全一样 (`No profiling data dumped`)。
        _seen, _matched = _kernel_name_miss(_txt, kernel_name)
        if _seen and not _matched:
            raise RuntimeError(
                f"msprof 采不到数据, 但原因**不是** launch-count/skip —— "
                f"`--kernel-name={kernel_name}` 一个核都没匹配上。\n"
                f"  这一趟驱动实际发过的核 (全被过滤掉了): {_seen}\n"
                f"  ★核名是**算子的 PascalCase 名**加哈希后缀, 不是源码目录的小写下划线名。\n"
                f"    例: 目录 sparse_flash_attention_grad -> 核名 SparseFlashAttentionGrad_<哈希>。\n"
                f"    msprof 的 --kernel-name 按前缀匹配, 给 PascalCase 名即可, 不必带哈希。\n"
                f"  ★两种'采不到'的最后一句都是 `No profiling data dumped`, 所以必须看\n"
                f"    `skipped: not selected via --kernel-name` 这些行才分得清。")
        raise RuntimeError(
            f"msprof 建了 {d} 但**里面没有 {want}**。\n"
            f"  当前参数: --launch-skip-before-match={skip} --launch-count={launch_count}\n"
            f"  ★先数驱动实际发了几次 kernel: 它必须 > skip, 否则要采的全被跳过, 而 msprof\n"
            f"    仍会打印 'All task success' + 'No profiling data dumped' —— 看起来像环境坏了。\n"
            f"    repro 的 launch 次数 = 1(staged) + warmup + repeat。\n"
            f"  改法: 调大 --repeat/--warmup, 或给本工具传 --skip 小一点 (如 --skip 1 --launch-count 3)。\n"
            f"  ★别先怀疑 /dev/shm 或磁盘: msprof op 是原生 ELF, 在 /dev/shm 满时实测照常工作;\n"
            f"    root 有 ext4 保留块, df 显示 0 可用时仍写得进去。")
    return d


def extract(opprof_dir: str, op_name_filter: str | None = None) -> ProfilingData:
    dirs = _launch_dirs(opprof_dir)
    if not dirs:
        raise RuntimeError(
            f"未找到 {' 或 '.join(_PU_PATTERNS)} 于 {opprof_dir}；{_what_is_there(opprof_dir)}。"
            f"★「找不到」不等于「没有数据」: 先看上面这行列出的文件名, 再判是不是模式没对上")
    if len(dirs) > 1:
        dirs = dirs[1:]  # 丢弃冷启动 launch 0
    parsed = [_parse_launch(d) for d in dirs]
    durs = [p[0] for p in parsed if p[0] is not None]

    def _agg(idx, cols, nd):
        #  ★按**实际出现的键**取并集, 不只按写死的列表: 否则 _parse_launch 收到的新列
        #    (比如这份产物只有等待**比**没有等待**时间**) 会在聚合这一步被静默丢掉。
        #    2026-09-22 实撞: 解析层已经收对了, 聚合层又丢了一次, 表现同样是"全空"。
        cols = list(dict.fromkeys(list(cols) + [k for p in parsed for k in p[idx]]))
        out = {}
        for c in cols:
            vals = [p[idx][c] for p in parsed
                    if isinstance(p[idx].get(c), (int, float))]
            if vals:
                out[c] = round(sum(vals) / len(vals), nd)
            else:
                #  非数值 (如 __缺列__ 的说明串) 原样带过去 —— 它是**信号**, 不许丢
                nonnum = [p[idx][c] for p in parsed if isinstance(p[idx].get(c), str)]
                out[c] = nonnum[0] if nonnum else None
        return out

    return ProfilingData(
        op_name=parsed[0][2], task_type=parsed[0][3],
        task_duration_us=round(statistics.median(durs), 2) if durs else float("nan"),
        block_dim=parsed[0][1],
        ratios=_agg(4, RATIO_COLS, 4), times=_agg(5, TIME_COLS, 2), waits=_agg(6, WAIT_COLS, 2),
    )


def pipe_diff(baseline: ProfilingData, candidate: ProfilingData) -> dict:
    """归责: candidate 相对 baseline 的变化.

    关键: stall 主导的劣化里 ratio 会误导 (总时膨胀→所有 ratio 下降), 必须以**绝对时间**归责.
    pipe_time_us_delta / wait_us_delta 才是"代价落在哪条 pipe"的正确信号; ratio_delta 仅作辅助.

    ★`task_duration_us` **不能跨两次采集比** —— 它带着当次采集的 profiling 开销, 而那个开销
      本身会变。2026-09-14 实撞: 同一算子两次采集, msprof 的 dur 分别比同口径干净计时高
      22.3us 与 2.3us, 于是两版的 dur 差出 21.5us, 而干净计时的真实差只有 0.4us ——
      照 dur 判就会把一个中性改动记成 -3.7% 的收益。
      pipe 忙时**比 dur 可信, 但也不是干净的**: 同一次实撞里, 一个只动 vector 的改动让
      `aic_mte2` 差出 45.74us、`aic_scalar` 差出 10.73us, 而那两条在源码上根本没被碰过 ——
      缓存冷热与采集时机会一起漂进来。经验界线: **同量级的 pipe 差 (几十 us) 当噪声,
      成倍的差 (如 1499 -> 763) 才当信号**; 而 `aic_cube` 这类纯计算计数器是确定的
      (两次采集逐位相同 443.11), 可以当基准核对采集是否可比。
      判收益一律回 core/diagnose gate (逐位==基线 ∧ 趟稳 ∧ 生产构建 ∧ **同时段实测基线**)。
    """
    def _delta(a, b, cols, nd):
        #  ★按**并集**取键: 写死的列表之外, 两侧都出现过的键 (如发现来的每标志等待列)
        #    必须一起算差, 否则"解析层收对了、差分层又丢一次"——2026-10-27 实撞过。
        keys = list(dict.fromkeys(list(cols) + [k for k in a if k in b]))
        d = {c: round(b.get(c) - a.get(c), nd) for c in keys
             if isinstance(a.get(c), (int, float)) and isinstance(b.get(c), (int, float))}
        return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))

    #  ★重叠度 = Σ各 AIC pipe 忙时 / aic_time。它必须与 pipe 忙时的增减一起看, 否则会把
    #    "砍掉了工作量却更慢"这一类完全读反 —— 那不是某条 pipe 变慢, 是**重叠塌了**。
    #    2026-09-14 FIAS 实撞: 行分组把 GM->L1 精确减半、aic_mte2 从 423.38 降到 192.93us (-54%),
    #    而端到端 581.3 -> 715.68us (+23%); 同期重叠度 2.45 -> 1.64。
    #    若只看 pipe_time_us_delta 会得出"每条 pipe 都没变慢, 不知道时间去哪了"的死结论。
    def _overlap(pd):
        tot = sum(v for k, v in pd.times.items() if k.startswith("aic_") and k.endswith("_time(us)")
                  and k != "aic_time(us)" and v)
        win = pd.times.get("aic_time(us)") or pd.task_duration_us
        return round(tot / win, 2) if win else None

    ob, oc = _overlap(baseline), _overlap(candidate)
    verdict = None
    if ob and oc:
        pipe_drop = min((candidate.times.get(c, 0) - baseline.times.get(c, 0))
                        for c in TIME_COLS if c.startswith("aic_") and c in baseline.times)
        #  ★下面这个 slower 只用于措辞, **不作收益判定** —— dur 跨采集不可比 (见函数首注释)。
        slower = candidate.task_duration_us > baseline.task_duration_us
        if slower and pipe_drop < -0.05 * baseline.task_duration_us and oc < ob - 0.15:
            verdict = (f"★**砍掉了工作量却更慢**: 有 pipe 忙时降了 {-pipe_drop:.1f}us 而端到端涨了 "
                       f"{candidate.task_duration_us - baseline.task_duration_us:.1f}us, "
                       f"同期重叠度 {ob} -> {oc}。结论是**重叠塌了**, 不是哪条 pipe 变慢 —— "
                       f"该查缓冲深度/流水级数/发出次序 (少一代缓冲就少一轮余量), 而不是继续削工作量。")

    return {
        "baseline_bottleneck": baseline.bottleneck(),
        "candidate_bottleneck": candidate.bottleneck(),
        "aic_pipe_overlap": {"baseline": ob, "candidate": oc},
        **({"★判定": verdict} if verdict else {}),
        "duration_us": {"baseline": baseline.task_duration_us, "candidate": candidate.task_duration_us},
        "candidate_over_baseline_pct": round(
            (candidate.task_duration_us - baseline.task_duration_us) / baseline.task_duration_us * 100, 1),
        "pipe_time_us_delta": _delta(baseline.times, candidate.times, TIME_COLS, 2),
        "wait_us_delta": _delta(baseline.waits, candidate.waits, WAIT_COLS, 2),
        "pipe_ratio_delta": _delta(baseline.ratios, candidate.ratios, RATIO_COLS, 4),
        "baseline": {"times_us": baseline.times, "waits_us": baseline.waits, "ratios": baseline.ratios},
        "candidate": {"times_us": candidate.times, "waits_us": candidate.waits, "ratios": candidate.ratios},
    }



# ---------------------------------------------------------------------------
# 全量信号提取 (msprof op 产物 -> 一张固定的特征信号表)
# 沉淀动机: 每次做性能判读都在临时写 csv 解析, 且每次只取当时想到的几列 —— 反复漏掉关键信号
# (gmfr 一例: 直到第 N 轮才注意到 L2Cache 表里有现成的 aic_write_hit_rate, 而 staging 是否落 DRAM
# 正是靠它判). 固定成一张表后, 每次测量都产出同一组可跨配置对比的信号.
# ---------------------------------------------------------------------------
SIGNAL_GROUPS = {
    "时间/占用": ["aic_time(us)", "aiv_time(us)", "aic_cube_time(us)", "aic_mte1_time(us)",
                  "aic_mte2_time(us)", "aic_mte3_time(us)", "aic_fixpipe_time(us)",
                  "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)", "aiv_scalar_time(us)"],
    "阻塞比": ["aic_cube_wait_ratio", "aic_mte1_wait_ratio", "aic_mte2_wait_ratio",
               "aiv_vec_wait_ratio", "aiv_mte2_wait_ratio", "aiv_mte3_wait_ratio"],
    #  ★★2026-10-19 补一条硬限制 (docs/msprof-re/08 §3.10 实测, 3024 行):
    #    **这些分量不构成对 scalar_time 的划分, 不许相加、不许算"占比"。**
    #    gmfr 实测中位数: aic 侧 `scalar_mte1_stall_time` 单独一项就是 `scalar_time` 的
    #    **1.054 倍** —— 一个分量大于总量, 划分读法当场被证伪;
    #    aiv 侧反过来, single+dual+全部 stall 只占 0.46, 另外 54% 没有任何列解释。
    #    四种分解 (含把 wait_id 加进来) 全试过, 没有一种是恒等式, 误差 -54% ~ +35%。
    #    ★所以"stall 占 scalar 的 N%"这句话在本工具的产物上**没有定义**。
    #      下面那条 gemm_v3 的结论 (削标量上限只有 single+dual) 仍然成立 ——
    #      它靠的是"single+dual 才是真执行时间", 不靠"stall 是它的补集"。
    #  ★这一组必须整组一起看, 单看 stall 那一个数会把方向读反 (gemm_v3 实撞, 白花一轮去削标量):
    #    `aic_scalar_mte1_stall_time` 量的是**标量流水停在那里等 MTE1**, 不是"标量拖累了 MTE1"。
    #    要判标量本身是不是瓶颈, 看的是 scalar_single + scalar_dual —— 那才是标量指令真正执行的时间。
    #    gemm_v3 @1024^3: scalar_time 10.39 里 stall 占 7.28, 而 single+dual 只有 1.81 ——
    #    削减循环内标量的收益上限从一开始就只有 1.81us, 实测确实纹丝不动。
    "标量流水 (整组看, 别单看 stall)": ["aic_scalar_time(us)", "aic_scalar_single_time(us)",
                                        "aic_scalar_dual_time(us)", "aic_scalar_mte1_stall_time(us)",
                                        "aic_scalar_mte2_stall_time(us)", "aic_scalar_cube_stall_time(us)",
                                        "aiv_scalar_vector_stall_time(us)"],
    "片上搬运量(KB/核)": ["GM_to_L1_datas(KB)", "L0C_to_GM_datas(KB)", "GM_to_UB_datas(KB)",
                          "UB_to_GM_datas(KB)"],
    "DRAM 流量(KB/核)": ["read_main_memory_datas(KB)", "write_main_memory_datas(KB)"],
    "DRAM 带宽(GB/s/核)": ["aic_main_mem_read_bw(GB/s)", "aic_main_mem_write_bw(GB/s)",
                           "aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)"],
    # ★L1/UB 侧带宽是判"MTE2 撞自己天花板 vs 被 DRAM 拖住"的关键: 若 l1_write_bw 已接近硬件上限,
    #   则只能减少字节; 若远低于上限而 DRAM 侧接近饱和, 则是被 DRAM 拖住, 减 DRAM 流量即可提速.
    "L1/UB 带宽(GB/s/核)": ["aic_l1_read_bw(GB/s)", "aic_l1_write_bw(GB/s)",
                            "aiv_ub_to_gm_bw(GB/s)", "aiv_gm_to_ub_bw(GB/s)",
                            "aic_l0c_write_bw_cube(GB/s)", "aic_fixpipe_active_bw(GB/s)",
                            "aic_mte3_active_bw(GB/s)"],
    "L2 命中率(%)": ["aic_read_hit_rate(%)", "aic_write_hit_rate(%)", "aic_total_hit_rate(%)",
                     "aiv_read_hit_rate(%)", "aiv_write_hit_rate(%)", "aiv_total_hit_rate(%)"],
    #  ★★2026-10-19 实测 (docs/msprof-re/08 §3.16-§3.18, 73738 行) —— **两侧结论相反, 不要混用**:
    #      aiv: 用 r0/r1 重算 `aiv_read_hit_rate(%)`  48855 行 **零失配**, 误差 1.0e-07  -> 可以重算
    #      aic: 同样的算法 **21.87% 的行对不上** (5443/24883), 偏差 1.036~1.402 倍
    #           (按 run 聚类, 同 run 内稳)                                    -> **不要重算**
    #    成因未定, 两个候选 csv 分不开: (a) AIC 侧还有未导出的读通道 (cube 经 L1/L0 的通路,
    #    AIV 没有), 该列把它们算进去了; (b) 它用的是另一个分母。
    #    ★反过来用是有价值的: AIC 侧两者**不一致本身**是信号 —— 它指出这一行存在 r0/r1 之外的读流量,
    #      差额是"看不见的那部分读"的下界指标。一个在 AIC 上失效的公式, 因此变成了一个检测器。
    "L2 计数": ["aic_write_cache_hit", "aic_write_cache_miss_allocate",
                "aic_r0_read_cache_hit", "aic_r0_read_cache_miss_allocate",
                "aiv_write_cache_hit", "aiv_write_cache_miss_allocate"],
    "指令数": ["aic_mte1_instructions", "aic_mte2_instructions", "aic_mte3_instructions",
               "aiv_mte2_instructions", "aiv_mte3_instructions", "aic_cube_total_instr_number"],
    "算力": ["aic_cube_ratio", "aic_cube_int8_ratio", "aiv_vec_ratio", "Current Freq", "Rated Freq"],
    # ★UB 冲突比 (来自 ResourceConflictRatio_*.csv, msprof 本就采了, 本表此前一直没打出来).
    #   ★★2026-10-20 定论 (docs/msprof-re/08 §3.42): 分母是**任务窗口**, 与 F2″ 的 pipe ratio
    #     同一个 —— mix 核 floor(Task Duration)×1800。**不是该核的向量周期数。**
    #         冲突占用的墙钟 ≈ aiv_vec_total_cflt_ratio × Task Duration
    #     用 `cflt_wall_us()` 换算, 不要自己乘。
    #     ★此前这里写的是"务必乘一次 aiv_vec_time", 那是**错的**: 本机 vec_ratio ≈ 0.176,
    #       按旧处方算会把冲突的墙钟影响**低估约 5.7 倍**。
    #     定论方法 (2334 个 (块,核) 组合, 取 vec_time 跨 launch 不变而 TaskDur 变 >2% 的那些):
    #         cflt × floor(TaskDur) 的跨 launch 跨度  中位 2e-05   ← 不变量
    #         cflt × vec_time       的跨 launch 跨度  中位 3.6e-02 ← 与 cflt 自己一样, 毫无不变性
    #       同一块干同样的活, 哪个假设推出的周期数不变, 分母就是哪个。差三个数量级。
    #     ★2026-10-19 当时挂质疑标记而不改处方是对的 —— 那时只有相关性证据;
    #       这次换的不是"更多数据", 是**一个能把两个候选分母分开的配置**。
    #   四项分开看, 成因不同, 改法也不同:
    #     bankgroup  跨 bank group 的冲突 —— 典型成因是 UB 里的**行跨距是大的 2 的幂**,
    #                各行同偏移落进同一 group。改法: 给行跨距加一个非 2 的幂的 pad。
    #     bank       同 group 内 bank 冲突 —— 同一指令内多路访问撞同 bank, 与 blockStride 有关。
    #     resc       通用资源冲突 (发射/端口)。
    #     mte        向量与 MTE 抢 UB 端口 —— 不为 0 说明搬运与计算在抢同一块 UB, 靠错开缓冲解。
    "UB 冲突比": ["aiv_vec_total_cflt_ratio", "aiv_vec_bankgroup_cflt_ratio",
                  "aiv_vec_bank_cflt_ratio", "aiv_vec_resc_cflt_ratio", "aiv_vec_mte_cflt_ratio",
                  "aiv_total_cycles", "aic_total_cycles"],
    # ★标量流水的**停顿归因**: 本项目一直只看 aiv_scalar_vector_stall, 于是 aiv_scalar 那几千 μs
    #   长期没有分解。实测 (gmfr v20p issue#2): scalar 共 5437μs, 其中 vector_stall 2623 +
    #   mte3_stall 1055 + mte2_stall 454 = 4132μs 是**停顿**, 只有约 1300μs 在真算 ——
    #   "标量很贵"其实是"标量在等别的流水"。mte3_stall 尤其关键: 它是逐行循环在等写流水,
    #   融合两条写会直接压它, 属可验证的机制而不是推测。
    "标量停顿归因": ["aiv_scalar_vector_stall_time(us)", "aiv_scalar_mte2_stall_time(us)",
                     "aiv_scalar_mte3_stall_time(us)", "aiv_scalar_single_time(us)",
                     "aiv_scalar_dual_time(us)", "aic_scalar_mte2_stall_time(us)",
                     "aic_scalar_mte3_stall_time(us)"],
    # ★**逐同步标志位的等待时间** —— msprof 一直在报, 本表到 2026-10-19 才接进来。
    #   上面"标量停顿归因"那组只能说"标量在等某条流水", 这一组直接说**等在哪一个 flag id 上**,
    #   把一个聚合量劈成 16 份可定位的量。改同步只能逐个 flag 改, 所以这才是可动手的粒度。
    #   ★实测 (gmfr, runs/sweep_64_512_128_0_4_4 与 runs/auth_det_realshape_off):
    #     16 个 id 里只有 id5 与 id14 非零, 而 id5 一个就占 aiv_time 的 38.3~39.0%
    #     (跨 6 份独立采集, 极稳); id14 只有 3.24us。一个 flag 吃掉近四成 AIV 时间,
    #     这件事在聚合的 aiv_scalar_* 里完全看不出来。
    #   ★产物里只出现**非零**的那几列 (实测恒为 3 列), 所以这里把 32 个名字全列出来,
    #     由 print_signals 的 `if g(c) is not None` 自动跳过缺席的 —— 不要改成动态前缀匹配,
    #     那会把"这次采集根本没有这组列"与"这次全是 0"混成一回事。
    #   发现路径: 逆 msopprof 二进制的 204 个列名时撞见 (docs/msprof-re/06 V9), 回头一查
    #     本仓 PipeUtilization 产物里**一直都有**。属典型的"数据在、没人读"。
    "逐同步标志等待 (定位到具体 flag id)": (
        [f"aic_scalar_wait_id{i}_time(us)" for i in range(16)] +
        [f"aiv_scalar_wait_id{i}_time(us)" for i in range(16)]),
    # ★AIV 侧的**活跃窗口带宽**: 我们一直只取了 AIC 侧 (aic_mte3_active_bw / aic_fixpipe_active_bw),
    #   AIV 侧的两列一直漏着, 于是每次都手算"字节 ÷ pipe 时间"。msprof 本来就报。
    #   注意与 *_bw_usage_rate(%) 不是一回事: 后者的分母是整个任务窗口, 前者只算活跃期,
    #   前者才是"这条流水动起来有多快"。gmfr 上 aiv_mte3_active_bw i1 32.1 / i2 17.5 GB/s/核,
    #   与手算的 34.5 / 19.0 吻合 —— 早接进来就不用手算, 也不会拿错的天花板去除。
    "AIV 活跃带宽 / UB 端口": ["aiv_mte2_active_bw(GB/s)", "aiv_mte3_active_bw(GB/s)",
                               "aiv_ub_read_bw_vector(GB/s)", "aiv_ub_write_bw_vector(GB/s)",
                               "aiv_ub_read_bw_scalar(GB/s)", "aiv_ub_write_bw_scalar(GB/s)"],
    # ★指令缓存: 便宜, 且能一次排掉一整类假设。本算子上 aiv 0.0001~0.0003 / aic 0 —— 于是
    #   "界写的成本 100% 在指令上"这件事**不能**用 icache 未命中来解释, 那条路已经堵死。
    "指令缓存": ["aiv_icache_miss_rate", "aic_icache_miss_rate"],
    # ★AIV 侧 L2 读命中的**原始计数** (只有比率时算不出缺失的绝对量).
    "L2 计数(AIV 读)": ["aiv_r0_read_cache_hit", "aiv_r0_read_cache_miss_allocate",
                        "aiv_r1_read_cache_hit", "aiv_r1_read_cache_miss_allocate"],
}


#  来源: 03 §4.7 的 block 级字段表 (`ffts_block_pmu.py:26-40`) —— 记录 128 字节,
#  `total_cycle`@32 · `t_start`@112 · `t_end`@120 · 溢出标志在 f5@16 的第 10 位。
#  2026-10-22 拿真字节对过账 (01 §13.21): 十遍的 t_start 单调递增、跨度一致, 且
#  `cycle/span` 恒为 36 —— 正是核频率 1800 MHz ÷ 时间戳频率 50 MHz。
_PD_REC, _PD_CYC, _PD_F5, _PD_OV, _PD_TS, _PD_TE = 128, 32, 16, 10, 112, 120
#  物理上限: 一个时间戳 tick 里最多执行 36 个核周期。全仓 215856 条实测
#  p50 35.99 / p90,p99,p99.9 全是 36.00 / >36.5 的只有 168 条 (0.0778%, 最小 39.68)。
_PD_RATIO_TOL = 36.5


def passdev_violations(prof_dir: str) -> list:
    """返回 [(dump 目录, 遍次号, 记录号, cycle, span, 比值)] —— 周期计数器失真的遍。

    ★与本模块的 `_corrupt_launches` **抓的不是一回事** (01 §13.22 实测):
      · `_corrupt_launches` (F2) 抓**失真漏进 csv** 的那一类 —— 全仓 34 例里的 1 例;
      · 本函数抓**失真留在原始记录**的那一类 —— 另外 33 例, F2 对它们静默。
    两者互补, 不是谁取代谁; 而 **F2 抓的恰好是唯一会影响报出数字的那一类**。

    判据是**自洽**的: 分子分母同在一条记录里, 上限是频率比这个硬常数。
    它还能分开两种失效: 计数器失真 (cycle 高而 span 不变 -> 比值爆) vs
    真的跑慢 (两者同步升高 -> 比值不变, 正确地静默)。
    """
    import struct as _st
    out = []
    for d in sorted(glob.glob(os.path.join(prof_dir, "**", "dump"), recursive=True)):
        for f in sorted(glob.glob(os.path.join(d, "DeviceProf*.bin"))):
            digits = "".join(c for c in os.path.basename(f) if c.isdigit())
            if not digits:
                continue
            try:
                b = open(f, "rb").read()
            except OSError:
                continue
            for i in range(len(b) // _PD_REC):
                o = i * _PD_REC
                if (_st.unpack_from("<H", b, o + _PD_F5)[0] >> _PD_OV) & 1:
                    continue          # 溢出标志置位 -> total_cycle 作废
                tc = _st.unpack_from("<Q", b, o + _PD_CYC)[0]
                ts = _st.unpack_from("<Q", b, o + _PD_TS)[0]
                te = _st.unpack_from("<Q", b, o + _PD_TE)[0]
                if te <= ts or not tc:
                    continue
                r = tc / (te - ts)
                if r > _PD_RATIO_TOL:
                    out.append((d, int(digits), i, tc, te - ts, r))
    return out


def print_passdev(prof_dir: str) -> int:
    v = passdev_violations(prof_dir)
    print(f"=== A 路径遍次失真检查 (core/perf --passdev) === {prof_dir}")
    if not v:
        print("  过: 没有 total_cycle/(t_end-t_start) 超过 36 的遍。")
        print("  ★这只说**原始记录**没失真; 失真漏进 csv 的那一类由 "
              "`_corrupt_launches` 的 F2 不变量管 (01 §13.22)。")
        return 0
    bydir = {}
    for d, k, _i, tc, sp, r in v:
        bydir.setdefault(d, {}).setdefault(k, (tc, sp, r))
    print(f"  ★**{len(v)} 条记录**失真, 分布在 {len(bydir)} 个 launch:\n")
    for d, ks in sorted(bydir.items()):
        print(f"    {'/'.join(d.split('/dump')[0].split('/')[-2:])}")
        for k, (tc, sp, r) in sorted(ks.items()):
            print(f"       第 {k} 遍: cycle={tc} span={sp} 比值={r:.1f} (上限 36)")
    print("\n  后果: **墙上时间没变, 只有周期计数器爆了** —— 由这些遍供数的档位,")
    print("        其比值相对 csv 报的分母整体偏掉, 而 csv 里没有任何提示。")
    return 1


def _corrupt_launches(out_dir: str) -> list:
    """找出 PMU 数据**物理上不可能**的 launch 目录, 返回 [(目录, 原因)].

    判据来自 A 路径公式 F2 (docs/msprof-re/08 §2): `<kind>_time(us)` = 该核 total_cycles/1800,
    即**该核在任务窗口内的活跃时长**, 所以它不可能超过同一 launch 的 `Task Duration(us)`。
    超了就不是"偏高", 是这一份计数器废了。

    ★沉淀动机 (2026-10-19, runs/pipe_v18a_i1_on 实撞): 该次采集 10 个 launch,
      **只有 launch 0 坏**, 且坏得整齐 —— 每一列都是稳态值的约 5.6 倍
      (aiv_time 25196us 对中位数 4494us, 而那个 launch 自己的 Task Duration 只有 4498us)。
      extract() 早就写了 `dirs[1:]  # 丢弃冷启动 launch 0`, 但 extract_signals() 是另一条
      取数路径, 它 `glob("**/*.csv")` 把**全部** launch 一起平均 —— 于是同一份产物
      在两条路径上给出不同的数, 且 signals 这条被抬高 **46%** (mean/median = 1.46,
      六列齐步, 见 08 §3.3)。hwfloor.verify_against_msprof 走的正是这条。
    ★为什么用不变量而不是"丢第一个": 丢第一个是**位置**判据, 它假设坏的总是 launch 0。
      本次恰好是 0, 但没有证据说一定是。不变量是**物理**判据, 坏在哪一个都抓得住,
      而且对没坏的 launch 0 不误伤 (本机多数样本的 launch 0 是好的)。
    ★不静默: 丢了什么、为什么丢, 必须打出来 —— "空目录"与"对不上"长得一样这件事,
      本文件已经栽过四次 (见 _what_is_there)。
    """
    import csv as _csv
    import glob as _glob
    bad = []
    for d in _launch_dirs(out_dir):
        ob = _glob.glob(os.path.join(d, "OpBasicInfo*.csv"))
        if not ob:
            continue
        try:
            obr = list(_csv.DictReader(open(ob[0], errors="ignore")))
            td = float(obr[0]["Task Duration(us)"])
        except Exception:
            continue
        if td <= 0:
            continue
        worst = None
        for p in _glob.glob(os.path.join(d, "*.csv")):
            try:
                rows = list(_csv.DictReader(open(p, errors="ignore")))
            except Exception:
                continue
            for r in rows:
                for c in ("aic_time(us)", "aiv_time(us)"):
                    try:
                        v = float(r.get(c))
                    except (TypeError, ValueError):
                        continue
                    #  5% 余量: F2 的 D 反解有万分之二的量化误差, 真坏的是成倍而非百分之几
                    if v > td * 1.05 and (worst is None or v > worst[1]):
                        worst = (c, v)
        if worst:
            bad.append((d, f"{worst[0]}={worst[1]:.1f}us > Task Duration={td:.1f}us "
                           f"({worst[1] / td:.1f} 倍) —— 核的活跃时长不可能超过任务窗口"))
    return bad


#  OPPROF 目录下**不是核函数**的那些子目录 (原始 dump / 模拟器产物 / 设备目录)
_NON_KERNEL_DIRS = {"simulator", "dump", "tmp_dump"}


def kernel_dirs(out_dir: str) -> list:
    """OPPROF 目录下的核函数子目录名。排除 dump/simulator/deviceN 这些非核目录。"""
    out = []
    for d in sorted(glob.glob(os.path.join(out_dir, "*"))):
        if not os.path.isdir(d):
            continue
        b = os.path.basename(d)
        if b in _NON_KERNEL_DIRS or re.fullmatch(r"device\d+", b):
            continue
        out.append(b)
    return out


def extract_signals(out_dir: str, kernel: str | None = None) -> dict:
    """扫 msprof 产物下**全部** csv, 按核聚合 (均值/最大), 返回 {列名: {"mean":, "max":, "n":}}.

    不区分 mix_aic / mix_aiv 目录: 两侧 csv 的列名本身带 aic_/aiv_ 前缀, 同名列取并集聚合.

    ★一个 OPPROF 目录里可能有**多个核函数** (本仓实测 21 份这样的产物, 其中一份含 5 个:
    Cast / Add / MemSet / MatMulV2 / Cast)。把它们一起平均出来的数**没有意义** ——
    MemSet 的 mte3 与 MatMul 的 cube 混进同一个均值, 再拿去判瓶颈必然判错。
    所以多核时**拒跑**而不是加个警告照算: 警告会被滚上去, 而那个错误的数会被用下去。
    用 `kernel=` (CLI `--kernel-dir`) 指定一个。

    这条是 2026-10-20 一次**差点发表的错误结论**带出来的 (08 §3.41):
    当时按 run 目录分组统计两个 stall 列, 而每个 run 目录下有不止一个核,
    于是得出"这两列从不同时非零"的假互斥 —— 按核一拆当场消失。
    **分组键必须是被测对象 (核函数), 不是目录。** 同一个病在这里是生产路径上的。
    """
    import csv as _csv
    import glob as _glob
    import statistics as _st
    ks = kernel_dirs(out_dir)
    if len(ks) > 1 and not kernel:
        raise SystemExit(
            f"拒跑: {out_dir} 下有 {len(ks)} 个核函数, 混在一起求均值没有意义。\n"
            + "".join(f"    {k}\n" for k in ks)
            + "  用 --kernel-dir <上面之一> 指定。(perf.extract_signals 的 kernel= 参数)")
    if kernel:
        if kernel not in ks:
            raise SystemExit(f"拒跑: {out_dir} 下没有核函数目录 {kernel!r}; 有的是 {ks}")
        out_dir = os.path.join(out_dir, kernel)
    _bad = _corrupt_launches(out_dir)
    if _bad:
        print(f"  [perf] ★丢弃 {len(_bad)} 个**数据损坏**的 launch (不是冷启动偏高, 是物理上不可能):")
        for d, why in _bad:
            print(f"         {os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}  {why}")
    agg = {}
    for p in _glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True):
        if os.path.dirname(p) in {d for d, _ in _bad}:
            continue
        try:
            rows = list(_csv.DictReader(open(p, errors="ignore")))
        except Exception:
            continue
        for r in rows:
            for c, v in r.items():
                if not c:
                    continue
                try:
                    agg.setdefault(c.strip(), []).append(float(v))
                except (TypeError, ValueError):
                    continue
    return {c: {"mean": _st.mean(vs), "max": max(vs), "n": len(vs)} for c, vs in agg.items() if vs}


def op_type_of(out_dir: str, kernel: str | None = None) -> str | None:
    """从 `OpBasicInfo_*.csv` 读 `Op Type` (`mix` / `cube` / `vector`)。

    ★`extract_signals` 只留得下数值列, 而 `Op Type` 是字符串 —— 它偏偏是
    `*_ratio` 分母取哪一个的唯一依据 (见 `ratio_window_us`), 所以单读一次。
    读不到返回 None, 由调用方决定是报错还是跳过; **不默认成 `mix`**。
    """
    import glob as _glob
    d = os.path.join(out_dir, kernel) if kernel else out_dir
    for p in sorted(_glob.glob(os.path.join(d, "**", "OpBasicInfo*.csv"), recursive=True)):
        #  ★只吞 IO/解码错 —— 第一版写的是 `except Exception`, 而里面把 `csv` 误写成
        #    `_csv` (那是别的函数里的局部别名), NameError 被一并吞掉, 结果是每次都返回
        #    None 而上层如实报"读不到 Op Type"。**看起来像没数据, 其实是代码错**。
        try:
            fh = open(p, errors="ignore")
        except OSError:
            continue
        with fh:
            for r in csv.DictReader(fh):
                t = (r.get("Op Type") or "").strip()
                if t:
                    return t
    return None


# ---------------------------------------------------------------------------
# MFU 四段分解 —— 恒等式, 不是模型
# ---------------------------------------------------------------------------
#  为什么加它 (2026-09-10, 从 cases/flash_attention 与 cases/sparse_flash_attention 沉淀):
#  「MFU 只有 7%」这句话本身不指向任何动作。而 MFU 可以**无条件**分解成四项, 每项各自
#  指向不同的下一步工具。分解是纯代数, 对任何算子/形状/dtype 都精确成立:
#
#      MFU = 有效FLOP / (T_e2e · 峰值)
#          = (cube忙 / aic_time) · (aic_time / T_e2e) · (有效FLOP / (cube忙 · 峰值))
#          =    忙占比 (busy)   ·    窗口占比 (win)   ·        忙时效率 (eff)
#
#  实测四条上逐位闭合 (SFA case1 / FlashAttn case6,19 / FAS case7)。
#
#  ★第四项「访存侧上界」不是恒等式而是判据, 且**第一版写错过**:
#    最初写成「忙占比 ≤ 算术强度 / (峰值/DRAM带宽)」, 在 FlashAttn case6 上算出 150% 的
#    达成率 —— 因为该形状的访存主要由 L2 供 (实测命中 66.5%), 用 DRAM 平衡当分母是错的。
#    正确形式要用**实测命中率**把两条通路分开, 并取 max 而不是相加 (两条通路并行服务;
#    相加的写法曾在 core/attnplan 上把 shared 掩码预测成 8us 而实测 3430us):
#        忙占比上界 = t_cube理想 / max( 字节·(1-命中)/BW_dram , 字节·命中/BW_l2 )
#    它**只是访存侧**的上界 —— mte1/fixpipe/vec/scalar 都不在里面, 所以忙占比可以远低于它。
#    那恰恰是它的用处: 达成率低就说明**墙不在访存**。

def mfu_breakdown(sig: dict, flops: float, e2e_us: float, cores: int,
                  peak: float, bw_dram: float, bw_l2: float) -> dict:
    """四段分解. 返回各项; 打印判读与下一步。flops = **有效** FLOP (不含填充/浪费)."""
    #  ★信号键带单位后缀 (aic_cube_time(us) 而不是 aic_cube_time), 按前缀匹配
    def g(pfx):
        for k in sig:
            if k == pfx or k.startswith(pfx + "("):
                return sig[k]["mean"]
        return None
    cube, aic = g("aic_cube_time"), g("aic_time")
    if not cube or not aic or not e2e_us:
        print("  ★缺 aic_cube_time / aic_time, 无法分解")
        return {}
    busy, win = cube / aic, aic / e2e_us
    eff = flops / (cube * 1e-6) / peak
    mfu = flops / (e2e_us * 1e-6) / peak
    print("\n[MFU 四段分解]  MFU = 忙占比 x 窗口占比 x 忙时效率   (**恒等式**, 纯代数, 无假设)")
    print(f"  忙占比   cube忙/aic_time      {busy*100:6.1f}%   cube 有多少时间在算")
    print(f"  窗口占比 aic_time/T_e2e       {win*100:6.1f}%   AIC 活跃窗口占端到端多少 (起停/尾部)")
    print(f"  忙时效率 有效FLOP/(忙x峰值)   {eff*100:6.1f}%   算的时候有多满")
    print(f"  三项乘积                      {busy*win*eff*100:6.2f}%   <- 应等于实测 MFU {mfu*100:.2f}%")
    out = {"busy": busy, "win": win, "eff": eff, "mfu": mfu}

    #  ★第四项: 忙占比的上界。**用实测的最长 pipe 当分母, 不用带宽模型。**
    #    第一版用的是 attnplan 那套 max(t_dram, t_l2) 带宽模型, 在四条实测上偏乐观
    #    1.40~2.80 倍 (SFA case1 1.92x / FlashAttn case6 1.40x / FAS case7 2.08x /
    #    case19 2.80x), 于是"访存侧达成"被系统性高估、把本来受访存限的形状判成"墙不在访存"。
    #    实测的 per-core MTE2 速率 45~96 GB/s, 而 ini 给 DDR 44.5 / L2 165 —— L2 命中带来的
    #    加速远不到模型那么多 (case7 命中 52.9% 却只跑到 45.4 GB/s = 恰好 DDR 速率)。
    #    带宽模型的用处是 build 前规划 (core/attnplan), 有了 msprof 就该用实测量。
    PIPES = [("cube", "aic_cube_time"), ("mte1", "aic_mte1_time"), ("mte2", "aic_mte2_time"),
             ("mte3", "aic_mte3_time"), ("fixpipe", "aic_fixpipe_time"),
             ("aic_scalar", "aic_scalar_time"), ("vec", "aiv_vec_time"),
             ("aiv_mte2", "aiv_mte2_time"), ("aiv_mte3", "aiv_mte3_time"),
             ("aiv_scalar", "aiv_scalar_time")]
    busy_pipes = [(n, g(k)) for n, k in PIPES if g(k)]
    tc = flops / peak * 1e6
    out["t_cube_ideal"] = tc
    if busy_pipes:
        top_n, top_t = max(busy_pipes, key=lambda x: x[1])
        #  ★推导 (两次自我纠正后的形态):
        #    各 pipe 完美重叠时 T_e2e 的下界就是 max(pipe_i), 于是
        #      MFU = 有效FLOP/(T_e2e·峰值) <= 有效FLOP/(最长pipe·峰值)
        #    这个上界**不含任何带宽模型**, 只用 msprof 报的各 pipe 忙时。
        #    化简后达成率就是「最长pipe / 端到端」= **流水重叠质量**, 与 FLOP 无关。
        #    错法一 (已进自检): 上界写成「算术强度/(峰值/DRAM带宽)」—— case6 上给 153%。
        #    错法二 (已进自检): 上界写成带宽模型 max(t_dram,t_l2) —— 四点偏乐观 1.40~2.80 倍,
        #      把 mte2 受限的 case7 误判成"墙不在访存"(41%), 而实测形态给 85%。
        #    错法三: 上界写成 t_cube理想/最长pipe —— 那等于假定忙时效率 100%,
        #      SFA case1 上算出 104% 的达成率 (忙时效率 94% 时分子应该用 cube **实际**忙时)。
        ub_mfu = flops / (top_t * 1e-6) / peak
        att = top_t / e2e_us
        out.update(top_pipe=top_n, top_pipe_us=top_t, mfu_ub=ub_mfu, overlap=att)
        print(f"\n  MFU 上界 = 有效FLOP / (**最长 pipe 的实测忙时** x 峰值)   (判据; 无带宽模型)")
        print(f"    各 pipe 忙时: " + "  ".join(f"{n}:{t:.1f}" for n, t in
                                              sorted(busy_pipes, key=lambda x: -x[1])[:6]))
        print(f"    最长 = {top_n} {top_t:.2f}us  ->  MFU 上界 {ub_mfu*100:.2f}% "
              f"(实测 {mfu*100:.2f}%)")
        print(f"    达成 = 最长pipe/端到端 = {top_t:.2f}/{e2e_us:.2f} = {att*100:.0f}%"
              f"   <- 这就是**流水重叠质量**, 与 FLOP 无关")

    print("\n  [怎么用这四个数] —— 它是**路由器**不是模型:")
    if eff < 0.8:
        print(f"    · 忙时效率 {eff*100:.0f}% < 80%: cube 自己没吃饱**或在算废的**。这一项混了两件事 —")
        print("      满发率 与 有用率 (M 轴分形填充/尾块浪费)。先分开再动: core/mmplan (tiling 可行性)")
        print("      与 core/pipeplan (双缓冲/流水深度)。★Sq 小于 mBase 时这一项天然低, 不是缺陷。")
    else:
        print(f"    · 忙时效率 {eff*100:.0f}%: cube 算的时候是满的 —— **不要去动 tiling/双缓冲**。")
    if out.get("overlap") is not None:
        tn, ta = out["top_pipe"], out["overlap"]
        HOWTO = {
            "mte2": "减 GM->L1 字节: 查 K/V 重读倍数 (循环序)、CV 是否经 GM 中转、行跨距对齐。"
                    "工具 core/attnplan 的 A2 字节账",
            "aiv_mte2": "AIV 侧在从 GM 读 —— 通常是 CV 中转落了 GM。查 L0C_to_GM_datas 是否 >0",
            "aiv_scalar": "标量下发是墙: 地址计算/离散搬运的指令条数。用 core/sim 看指令序, "
                          "判是条数问题还是 Set/Wait 串行",
            "aic_scalar": "AIC 标量: 同上",
            "fixpipe": "fixpipe 搬出成为瓶颈: 查输出布局与 L0C->UB/GM 的每次搬出宽度",
            "mte1": "L1->L0 是墙: 查 L0A/L0B 双缓冲与每次载入的 tile 形状",
            "vec": "vector 计算是墙: softmax/update 的向量指令数, 查是否有可省的中间量",
            "cube": "cube 自己就是最长 pipe —— 已接近算力受限, 只能减 FLOP 或提 dtype 密度",
        }
        if ta > 0.8:
            print(f"    · 重叠质量 {ta*100:.0f}% (高): 流水已基本压平, **{tn} 就是墙**。")
            print(f"      要再快只能减这条的量: {HOWTO.get(tn, '查该 pipe 的量与速率')}")
        else:
            print(f"    · 重叠质量 {ta*100:.0f}% (低): 各 pipe 在**互等**, 端到端比最长的 {tn} "
                  f"还长 {(1/ta-1)*100:.0f}%。")
            print("      **先压平流水再减量** —— 看本表的 `aic pipe 重叠度` (接近 1 = 串行),")
            print("      查 Set/Wait 是否紧邻、双缓冲有没有真用上 (core/detsync_lint 的 H10)。")
            print(f"      压平后上界是 {out['mfu_ub']*100:.2f}%; 之后才轮到减 {tn} 的量。")
    if win < 0.9:
        print(f"    · 窗口占比 {win*100:.0f}% < 90%: 起停/尾部占掉一成以上, 小形状常见的固定地板。")
    print("    ★不适用的场合: 无 cube 的算子 (纯 vector) MFU 无意义; dtype 混合时峰值要按 dtype 取;")
    print("      AIV 也做 FLOP 时 cube 侧不是全部; 且本分解**完全不管这些 FLOP 是否必要** ——")
    print("      算多余对的实现会显示好 MFU 而在做废功, 必须另配一个有用工作量的核对。")
    return out


def selftest_csv_layout() -> int:
    """负例控制: 两种文件名形态都要认, 且找不到时必须报出实际有什么。

    这条是 PF0 的防回归闸 —— 没有它, 下次有人把 _find_csv 改回单一模式, 又会表现成
    "msprof 对这个核没有数据"。
    """
    import tempfile
    cases, bad = [], 0
    with tempfile.TemporaryDirectory() as d:
        #  形态一: 无后缀
        open(os.path.join(d, "PipeUtilization.csv"), "w").write("a\n1\n")
        cases.append(("无后缀形态被认出", len(_find_csv(d, "PipeUtilization")) == 1, True))
        #  形态二: 带后缀
        open(os.path.join(d, "PipeUtilization_foo.csv"), "w").write("a\n1\n")
        cases.append(("两种形态并存时都收", len(_find_csv(d, "PipeUtilization")) == 2, True))
        #  找不到时要报出实际有什么, 不能只说"没有"
        msg = _what_is_there(d)
        cases.append(("找不到时列出实际文件", "PipeUtilization.csv" in msg, True))
    with tempfile.TemporaryDirectory() as d2:
        cases.append(("空目录要明说一个 csv 都没有", "一个 csv 都没有" in _what_is_there(d2), True))
    for name, got, want in cases:
        ok = bool(got) == want
        if not ok:
            bad += 1
        print(f"  {'过' if ok else '**未过**'} {name}")
    print(f"perf csv 布局自检: {len(cases)} 条, {bad} 条未过")
    return bad


def selftest_mfu() -> int:
    """判据负例自检: 恒等式必须闭合; 「无条件用 DRAM 平衡」这个错法必须能被抓出来。"""
    bad = 0
    print("=== perf MFU 分解 判据负例自检 (4 条) ===")
    #  1) 恒等式闭合 (合成数)
    peak, bwd, bwl = 378.4704e12, 1.2474e12, 4.62e12
    sig = {"aic_cube_time": {"mean": 10.0}, "aic_time": {"mean": 20.0},
           "GM_to_L1_datas": {"mean": 1000.0}, "aic_read_hit_rate": {"mean": 50.0}}
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = mfu_breakdown(sig, flops=1e9, e2e_us=25.0, cores=28,
                          peak=peak, bw_dram=bwd, bw_l2=bwl)
    ok = abs(r["busy"] * r["win"] * r["eff"] - r["mfu"]) < 1e-12
    print(f"  {'过' if ok else '★未过'} 恒等式闭合: 三项乘积 vs 实测 MFU 差 "
          f"{abs(r['busy']*r['win']*r['eff']-r['mfu']):.2e}")
    bad += 0 if ok else 1
    #  2) 上界不许超 100% (min 钳住)
    ok = r["mfu"] <= r["mfu_ub"] + 1e-12 and r["overlap"] <= 1.0 + 1e-12
    print(f"  {'过' if ok else '★未过'} 实测 MFU {r['mfu']*100:.2f}% <= 上界 "
          f"{r['mfu_ub']*100:.2f}%, 且重叠质量 {r['overlap']*100:.0f}% <= 100%")
    bad += 0 if ok else 1
    #  3) ★两条已被实测否掉的错法都必须被抓住:
    #     (a) 上界用「算术强度 / (峰值/DRAM带宽)」—— FlashAttn case6 上给 150%, 荒谬;
    #     (b) 上界用带宽模型 max(t_dram,t_l2) —— 四点上偏乐观 1.40~2.80 倍, 会把受访存限的
    #         形状误判成"墙不在访存"。回放 case6/case7 的实测值断言两条错法都露馅。
    peak_, bwd_, bwl_ = peak, bwd, bwl
    bad3 = 0
    for lbl, byts, flops_, cube_, aic_, e2e_, hit_, mte2_ in (
            ("case6", 34.9e6, 3.666e9, 10.31, 19.30, 19.46, 0.665, 13.18),
            ("case7", 604.0e6, 68.719e9, 181.57, 500.12, 561.23, 0.529, 474.70)):
        b = (cube_ / aic_) * (aic_ / e2e_)
        tc_ = flops_ / peak_ * 1e6
        wrong_a = min(1.0, (flops_ / byts) / (peak_ / bwd_))
        wrong_b = min(1.0, tc_ / (max(byts * (1 - hit_) / bwd_, byts * hit_ / bwl_) * 1e6))
        right = flops_ / (mte2_ * 1e-6) / peak_        # 新式: 有效FLOP/(最长pipe x 峰值)
        mfu_ = flops_ / (e2e_ * 1e-6) / peak_
        #  (a)(b) 两条错法都会把「达成率」算成与实测形态不同的数; 新式必须 <=100%
        ok_i = (mfu_ <= right + 1e-9) and (b / wrong_b) < (mfu_ / right) - 1e-9
        print(f"  {'过' if ok_i else '★未过'} {lbl}: 错法(a) AI/DRAM平衡 -> {b/wrong_a*100:.0f}%; "
              f"错法(b) 带宽模型 -> {b/wrong_b*100:.0f}%; 新式 实测MFU/上界 -> "
              f"{mfu_/right*100:.0f}% (须 <=100% 且高于 (b))")
        bad3 += 0 if ok_i else 1
    bad += bad3
    print(f"perf MFU 分解 自检: 4 条, {bad} 条未过")
    return 1 if bad else 0



def _print_mte2_bw_verdict(g, cores: int) -> None:
    """判 mte2 的忙时是**带宽打满**还是**在里面等** —— 两者的下一步完全不同。

    msprof 直接给 `aic_mte2_active_bw` (每核字节 / 每核 mte2 忙时)。把它与本机 ini 的每核端口
    速率比: L2 = l2_rate x 频率, DRAM = ddr_rate x 频率。若实测值接近"按命中率调和混合"的值,
    说明 mte2 忙时就是在搬数据, 只能靠**少搬字节**来降; 若远低于调和值, 说明忙时里含等待。

    ★调和混合而不是 max: FIAS 950PR_957b 实测 (2026-09-14) 命中 88.1% 那点
      aic_mte2_active_bw = 110GB/s, 调和式给 97.6GB/s (用实测达成的 DRAM 每核速率),
      而 max 式换算出来是 165GB/s —— 差 1.5 倍。见 attnplan REFS A17。
    """
    bw = g("aic_mte2_active_bw(GB/s)")
    if not bw:
        return
    try:
        from . import socspec
        sp = socspec.spec(socspec.default_soc())
        rates = sp.get("rate_B_per_cycle", {}) or {}
        freq = float(sp.get("freq_ghz", 0) or 0)
        l2_bc = float(rates.get("l2_rate", 0) or 0)
        dram_bc = float(rates.get("ddr_rate", 0) or 0)
    except Exception:
        return
    if freq <= 0 or l2_bc <= 0:
        return
    l2_pc, dram_pc = l2_bc * freq, dram_bc * freq
    print(f"  aic_mte2 活跃带宽 (每核)               {bw:12.2f} GB/s")
    print(f"    本机每核端口: L2 {l2_pc:.0f} GB/s (ini l2_rate={l2_bc:g} B/周期), "
          f"DRAM 峰值 {dram_pc:.0f} GB/s (ddr_rate={dram_bc:g})")
    if dram_pc > 0 and bw < l2_pc:
        #  由实测带宽反推"等效命中率": 1/bw = h/L2 + (1-h)/DRAM
        inv = 1.0 / bw
        denom = (1.0 / dram_pc) - (1.0 / l2_pc)
        h = ((1.0 / dram_pc) - inv) / denom if denom else None
        if h is not None and 0.0 <= h <= 1.0:
            print(f"    反推等效命中率 {h:.1%} (调和式 1/bw = h/L2 + (1-h)/DRAM)")
            print("    ★与 msprof L2Cache.csv 的 aic_read_hit_rate 对一下: 接近 = mte2 忙时就是在搬,"
                  " 只能靠少搬字节降; 反推值明显更低 = 忙时里含等待, 该查依赖而不是带宽。")
            print("    ★DRAM 常达不到 ini 峰值 (FIAS 实测达成仅 24.4GB/s/核 vs 峰值 44.6),"
                  " 所以反推值会偏低, 只作量级参考。")



#  ---- 带宽体检: 每条带宽都配硬件参照, 按离能力的距离排序 --------------------------
#  沉淀动机 (2026-09-18 dubin 指出, sfag 复盘):
#    我反复引用 `aiv_mte3_active_bw 18.50 -> 11.88 GB/s` 并据此断言"这是原子写的代价",
#    而 **aiv_mte2 是更大的一条 pipe (18739us vs 13606us), 它的带宽我一次都没算过**。
#    两个数在同一份 msprof 产物里, 算第二个成本为零。
#    为什么只算了一个: 判据 P4 (wait_ratio 最低者 binding) 说 MTE3, 我就只查 MTE3 ——
#    **让筛选规则替代了普查**。规则的职责是给注意力排序, 不是截断注意力。
#  ★而工具这一侧的缺口很具体: `_print_mte2_bw_verdict` 做的正是对的事 (带宽 + 硬件参照 +
#    反推命中率), 但**它只对 aic_mte2 一条做**。其余每条带宽都是裸数字, 没有分母。
#    一个没有参考区间的化验值是读不了的, 于是只有那一行被读。
#  ★判读纪律: 本表只说"离端口能力还有多远", **不说该不该动它** ——
#    低带宽可能是搬得少 (粒度小/事务多), 也可能是忙时里含等待, 两者下一步完全不同。
#    要分清得看 wait_ratio 与 L2 命中率, 或者直接算 hwfloor 的超额表。
_BW_REF = {
    "aiv_gm_to_ub_bw(GB/s)":     [("DRAM 端", "ddr_read_rate"), ("L2 端", "l2_read_rate")],
    "aiv_ub_to_gm_bw(GB/s)":     [("DRAM 端", "ub_to_ddr_rate"), ("L2 端", "ub_to_l2_rate")],
    "aic_mte3_active_bw(GB/s)":  [("DRAM 端", "ddr_write_rate"), ("L2 端", "l2_write_rate")],
    "aic_main_mem_read_bw(GB/s)":  [("DRAM", "ddr_read_rate")],
    "aic_main_mem_write_bw(GB/s)": [("DRAM", "ddr_write_rate")],
    "aiv_main_mem_read_bw(GB/s)":  [("DRAM", "ddr_read_rate")],
    "aiv_main_mem_write_bw(GB/s)": [("DRAM", "ddr_write_rate")],
}
#  ★没有可靠参照的, 必须**显式列出来**说查不了, 不许省略 ——
#    "没查成"与"查过没问题"长得一样, 是本仓反复栽的那个形态。
_BW_NOREF = {
    "aic_fixpipe_active_bw(GB/s)": "fixpipe 的出口端口在 ini 里没有对应的 rate 键",
    "aic_l1_read_bw(GB/s)":  "L1 读侧有 l1_to_l0_a/l1_to_l0_b/l1_to_ub 三个口, 按哪个算取决于数据流向",
    "aic_l1_write_bw(GB/s)": "L1 写侧同理 (ub_to_l1 与 GM->L1 不是同一个口)",
    "aic_l0c_write_bw_cube(GB/s)": "L0C 写入由 cube 发射节奏决定, 不是端口带宽问题",
}


def _print_bw_panel(g) -> None:
    try:
        from . import socspec
        sp = socspec.spec(socspec.default_soc())
        rates = sp.get("rate_B_per_cycle", {}) or {}
        freq = float(sp.get("freq_ghz", 0) or 0)
    except Exception:
        return
    if freq <= 0 or not rates:
        return
    rows, missing = [], []
    for key, refs in _BW_REF.items():
        bw = g(key)
        if not bw:
            continue
        pairs = []
        for label, rk in refs:
            r = float(rates.get(rk, 0) or 0)
            if r > 0:
                pairs.append((label, rk, r * freq))
        if not pairs:
            missing.append((key, "ini 里没有对应的 rate 键"))
            continue
        worst = min(p[2] for p in pairs)          # 最宽松的口 = DRAM 端, 用它算占比
        rows.append({"key": key, "bw": bw, "pairs": pairs, "pct": bw / worst * 100})
    for key, why in _BW_NOREF.items():
        if g(key):
            missing.append((key, why))
    if not rows and not missing:
        return
    rows.sort(key=lambda r: r["pct"])             # ★最差的排最前 —— 不让人只读自己想读的那行
    print("\n  --- 带宽体检 (每核; 占比对最**宽松**的那个口算, 故是乐观值) ---")
    for r in rows:
        ref = "  ".join(f"{lab} {cap:.0f}" for lab, _rk, cap in r["pairs"])
        flag = "  ★" if r["pct"] < 30 else ""
        print(f"    {r['key']:<30s}{r['bw']:8.2f} GB/s = 端口的 {r['pct']:5.1f}%   [{ref}]{flag}")
    if any(r["pct"] < 30 for r in rows):
        print("    ★占比 <30% 的那几条: 要么每次搬得太碎 (查 DataCopy 的 blockLen 与事务数),")
        print("      要么忙时里含等待 (查该 pipe 的 wait_ratio)。两者下一步完全不同, 别跳过这一分叉。")
    if missing:
        print("    以下带宽**没有可靠参照, 本表查不了** (不是查过没问题):")
        for k, why in missing:
            print(f"      {k:<30s}{why}")


def _print_cv_balance(g) -> None:
    """cube 侧与 vector 侧**谁的窗口更长**, 以及各自窗口里最长的那条流水占多少。

    混合核算子的端到端由两侧里晚结束的那个定。只看 aic 会漏掉"vector 才是长杆"这一类;
    而两侧窗口都远大于各自最长流水时, 说明两边在**互等** (谁也跑不到对方前面), 这时该动的是
    缓冲深度/流水级数, 不是任何一条流水的带宽。

    ★FIAS 950PR_957b 实测 (2026-09-14): aic_time 564.90 / aiv_time 576.66 —— vector 侧窗口更长,
      但它自己最长的流水 (vec) 只有 221.37us, 占窗口 38%; cube 侧最长的 cube 443.11us 占 78%。
      即 vector 基本在等 cube, 而 cube 在等自己的四条流水互相让路。
    """
    aic, aiv = g("aic_time(us)"), g("aiv_time(us)")
    if not aic or not aiv:
        return
    aic_pipes = {k: g(f"aic_{k}_time(us)") for k in ("cube", "scalar", "mte1", "mte2", "mte3", "fixpipe")}
    aiv_pipes = {k: g(f"aiv_{k}_time(us)") for k in ("vec", "scalar", "mte2", "mte3")}
    ac = max(((v, k) for k, v in aic_pipes.items() if v), default=(0.0, "-"))
    av = max(((v, k) for k, v in aiv_pipes.items() if v), default=(0.0, "-"))
    longer = "vector" if aiv > aic else "cube"
    print(f"  CV 窗口                                aic {aic:.2f}us / aiv {aiv:.2f}us"
          f"   <- 端到端由**{longer}**侧定")
    print(f"    aic 最长流水 {ac[1]} {ac[0]:.2f}us = 窗口的 {ac[0]/aic:.0%}; "
          f"aiv 最长流水 {av[1]} {av[0]:.2f}us = 窗口的 {av[0]/aiv:.0%}")
    if ac[0] / aic < 0.85 and av[0] / aiv < 0.85:
        print("    ★两侧最长流水都远小于各自窗口 = **互等**。该动缓冲深度/流水级数, "
              "不是任何一条流水的带宽。")


def print_signals(out_dir: str, sig: dict | None = None, cores: int = 24,
                  kernel: str | None = None) -> dict:
    """按 SIGNAL_GROUPS 打印信号表; 并派生几个跨表推导量 (DRAM 总量、staging 往返、瓶颈比)."""
    sig = sig or extract_signals(out_dir, kernel)
    g = lambda k: sig[k]["mean"] if k in sig else None
    print(f"=== msprof 特征信号 @ {out_dir} ===")
    for grp, cols in SIGNAL_GROUPS.items():
        have = [(c, g(c)) for c in cols if g(c) is not None]
        if not have:
            continue
        print(f"\n[{grp}]")
        for c, v in have:
            print(f"  {c:38s} {v:12.2f}" + (f"   (max核 {sig[c]['max']:.2f})" if sig[c]["max"] != v else ""))
    # 派生量
    rd, wr = g("read_main_memory_datas(KB)"), g("write_main_memory_datas(KB)")
    l0c_gm, gm_l1 = g("L0C_to_GM_datas(KB)"), g("GM_to_L1_datas(KB)")
    aic, cube, mte2 = g("aic_time(us)"), g("aic_cube_time(us)"), g("aic_mte2_time(us)")
    print("\n[派生]")
    if rd is not None and wr is not None:
        tot = (rd + wr) * cores / 1024 / 1024
        print(f"  DRAM 总流量 (全卡)                     {tot:12.2f} GB")
        if aic:
            print(f"  DRAM 实际带宽                          {tot*1024/ (aic/1000):12.2f} GB/s"
                  f"   <- 与卡的可达带宽比较即知是否带宽受限")
    if l0c_gm is not None:
        print(f"  CV staging 写出 (全卡)                 {l0c_gm*cores/1024/1024:12.2f} GB"
              f"   <- 910B 无 cube->vec 直连, 必经 GM")
    if gm_l1 is not None:
        print(f"  GM->L1 (全卡)                          {gm_l1*cores/1024/1024:12.2f} GB")
    if cube and mte2:
        print(f"  mte2/cube 比                           {mte2/cube:12.2f}"
              f"   <- >1 表示 cube 在等搬运 (空泡)")
    _print_mte2_bw_verdict(g, cores)
    _print_bw_panel(g)
    _print_cv_balance(g)
    #  ★pipe 重叠度 = Σ各 pipe 活跃时间 / aic_time。1.0 = 完全串行 (一条 pipe 忙时其它全停),
    #    越大越说明流水真的在并行。实测对照 (1024^3 fp16, 910B2, 2026-08-27):
    #      MatMulV2 (AscendC matmul 库)  2.25  —— aic_time 14.49 而 pipe 时间和 32.6
    #      GemmV3   (手写 pp_matmul 循环) 1.24  —— aic_time 20.12 而 pipe 时间和 25.0
    #    两者 cube 总工作量相当 (16 核 x 9.50 vs 24 核 x 6.63 核·us), 差的就是这一项。
    #    判读: 低重叠度 + 各 pipe 的 wait_ratio 也低 = **串行**, 不是某条 pipe 卡住;
    #    这时该查的是指令序 (Set/Wait 是否紧邻、双缓冲有没有真用起来), 不是加带宽。
    pipes = [g(c) for c in ("aic_cube_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)",
                            "aic_mte3_time(us)", "aic_fixpipe_time(us)")]
    tot_pipe = sum(x for x in pipes if x)
    if aic and tot_pipe:
        ov = tot_pipe / aic
        hint = "   <- 接近 1 = 各 pipe 串行, 查指令序 (Set/Wait 紧邻? 双缓冲真用上了?)" if ov < 1.5 else ""
        print(f"  aic pipe 重叠度 (Σpipe/aic_time){' ' * 8}{ov:12.2f}{hint}")
    #  ★CV 耦合判据 (2026-09-14 由 FIAS MLA 撞出): 只看 AIC 会把"延迟受限"误判成"带宽受限"。
    #    端到端 = max(aic_time, aiv_time); 若**两侧各自的窗口里都空掉 15% 以上**, 说明谁也没
    #    跑满 —— 一个任务要过 cube->vec->cube->vec 四道跨核握手, 任务之间靠缓冲深度错开,
    #    深度不够时往返延迟直接落到墙上。此时该加的是**缓冲深度**, 不是带宽也不是核数。
    #    实证 (FIAS 100k_64P, vendor_cv): AIC 窗口 566.43 而 cube 443.11 (空 21.8%);
    #      AIV 窗口 577.99 而 Σpipe 441.35 (空 23.6%); 端到端 581.3 ≈ AIV 窗口。
    #      同一算子上: 加一级 KV 预取 (深度 +1) -4%; mm2 结果双缓冲降单缓冲 (深度 -1) **+31%**;
    #      而 KV 搬运量减半 (mte2 忙时 -736us) 换来 **0** 墙时间 —— 三条互相印证。
    aiv = g("aiv_time(us)")
    aiv_pipes = [g(c) for c in ("aiv_vec_time(us)", "aiv_scalar_time(us)",
                                "aiv_mte2_time(us)", "aiv_mte3_time(us)")]
    aiv_tot = sum(x for x in aiv_pipes if x)
    cube_t = g("aic_cube_time(us)")
    if aic and aiv and aiv_tot and cube_t:
        aic_idle = 1.0 - cube_t / aic
        aiv_idle = 1.0 - aiv_tot / aiv
        print(f"  aiv pipe 合计/aiv_time{' ' * 18}{aiv_tot / aiv:12.2f}")
        print(f"  AIC 窗口内空转 (1-cube/aic_time){' ' * 9}{aic_idle:11.1%}")
        print(f"  AIV 窗口内空转 (1-Σpipe/aiv_time){' ' * 8}{aiv_idle:11.1%}")
        if aic_idle > 0.15 and aiv_idle > 0.15:
            print(f"  ★两侧窗口内都空掉 >15% = 谁也没跑满的**互等**形态。可以排除的是:")
            print(f"    端到端不由任何单条流水的忙时决定, 所以加核数、提某条流水的带宽都不对症。")
            print(f"  ★★**第 0 步: 去看 wait_ratio, 别按忙时排序猜。**")
            print(f"    本表的忙时只说明'谁在干活', 说明不了'谁是因'。判读 (gmfr 标定, 见")
            print(f"    docs/perf-ceiling-estimation.md §5): wait_ratio≈0 = 从不等待 = **因**; 高 = 在等别人 = 果。")
            print(f"    采法: msprof op --aic-metrics=PipeUtilization,**ResourceConflictRatio**")
            print(f"    (core/perf 的 run_msprof_op 现在默认两组都采; 只有忙时的旧产物会被强制重采)。")
            print(f"    ★FIAS 2026-09-14 因为只采了忙时, 白做五轮 (减搬运量/减 AIV 工作量/加深度/")
            print(f"      unitFlag/去跨核旗标, 端到端全是 0); 采了 wait_ratio 一步定位到 aic_mte2 0.599。")
            print(f"    ★★**同日更正 —— 本判据有一个未写明的前提: 表里必须有那条流水。**")
            print(f"      ResourceConflictRatio 只报 aic 的 cube/mte1/mte2/mte3 与 aiv 的 vec/mte2/mte3,")
            print(f"      **没有标量流水的 wait_ratio**。于是标量受限的算子会表现成'四条比值都很高、")
            print(f"      某条略低', 按'最低即为因'就会指向一条并非瓶颈的流水。")
            print(f"      FIAS 上按此判据指向 aic_mte2(0.599), 据此做了三轮减 KV 字节 (rg2/rg3/rg5),")
            print(f"      全部为负。真正的反证来自忙时表本身: rg3g2 把 aic_mte2 从 460.45 砍到 193.38us")
            print(f"      (不到一半), aic_time 却是最差的 701.33us (base 649.16, 最好 537.68)。")
            print(f"      ⇒ **先用一个把该流水忙时大幅砍掉的构建去反证**: 忙时降了而端到端不降,")
            print(f"        这条流水就不是因, 无论它的 wait_ratio 有多低。")
            print(f"      判'停顿在哪': 看 aic_time - aic_cube_time (cube 忙时是确定性量, 可跨采集比)。")
            print(f"      FIAS 交付版停顿 = 537.93 - 443.11 = 94.8us, 而每任务固定空转 564 周期")
            print(f"      x 336 任务/核 / 1.65GHz = 114.9us —— 量级吻合, 说明停顿是**每任务固定开销**,")
            print(f"      而 aic_scalar_time 稳定占 aic_time 的 55% 却无 wait_ratio 可查。")
            print(f"    下面按**代价从低到高**排检查次序, 每条注明它的证据强度。")
            print(f"    ★★这些数字来自**一个算子一个形状** (FIAS MLA decode, D=512, 每任务工作量恒定)。")
            print(f"      拿它挑**下一步做什么**, 不要拿它预测你自己的收益。")
            print(f"    1. **先查指令的发出次序, 不要先查缓冲深度。**")
            print(f"       [干净实测, 单变量] FIAS 上这两件事的量级差了两个数量级:")
            print(f"         - 把下一块数据的预取从'每轮最前'挪后一档 (让另一条 matmul 插到它前面):")
            print(f"           **+27.7%** (557.90 -> 712.53us); 挪到每轮最后 (距离归零): 此前量到 +43%。")
            print(f"         - mm2 结果缓冲深度 2 -> 1: **+0.15%**; 2 -> 4: **±0**。三点全平。")
            print(f"       ⇒ 在这个算子上**缓冲深度完全不影响性能, 而预取距离值 27% 以上**。")
            print(f"       ★2026-09-14 更正: 先前记的'深度降一级 +35%'是错的 —— 那一版同时改了发出次序,")
            print(f"         +35% 几乎全是次序的账。**一次构建放多个改动时, 任何单项归因都不成立。**")
            print(f"    2. 缓冲深度: 先按上面的量级预期放低, 再决定要不要花构建去试。")
            print(f"       可迁移的判准: 再加一级只在**单任务耗时的抖动超过一个任务**时才有用 ——")
            print(f"       双缓冲已经吸收掉一个任务的错位, 更多的只是空占片上空间。")
            print(f"       所以先问: 你的任务耗时随数据/形状变吗 (数据相关分支、块大小不定、尾块多)?")
            print(f"       变化大才值得试; 恒定就大概率无效。")
            print(f"       [实测] FIAS 每任务工作量完全相同, 抖动近零: 2 -> 4 收益 **0**")
            print(f"       (557.90 -> 558.25us, 还白付了把结果落 fp16 的精度代价), 2 -> 1 也只有 +0.15%")
            print(f"       (558.71us) —— **三点全平, 这个算子上深度这条轴是死的**。")
            print(f"       ★**降一级贵推不出加一级赚** —— 这两侧不对称, 别外推。")
            print(f"    3. 深度加不动之后, 两个候选, 证据强度不同:")
            print(f"       (a) [**已实测: 无效**] 每任务的跨核旗标条数。AIC 侧对 UB 缓冲的一次")
            print(f"           WaitCrossCore/SetCrossCore 要发**两条** (一个 AIC 对两个 AIV), FIAS 上每任务 10 条;")
            print(f"           去掉其中可证冗余的 2 条 (保持发出次序不变) 端到端 559.06 -> 559.03us, **精确中性**。")
            print(f"           ⇒ 旗标条数不是墙。先去采 ResourceConflictRatio 定位'因'再动手。")
            print(f"       (b) [**待验假设, 未直接测过**] 任务内部的跨核依赖链长度")
            print(f"           (cube 出中间结果 -> vector 加工 -> cube 再算)。它的错开量由**流水级数**")
            print(f"           决定而非缓冲份数; 加一级要多一份跨级存活的片上资源。")
            print(f"           ★FIAS 上这条是**排除法剩下的**: 依次证伪了带宽、缓冲深度、")
            print(f"             Mmad->fixpipe 重叠之后才轮到它, 而 L1 装不下第 4 级, 所以**也没能验证**。")
            print(f"             当成待验假设看, 别当结论。")
            print(f"    ★关于'减少某条流水的搬运量': 这种形态下它**远不成比例, 但不是零**。")
            print(f"      [实测] FIAS 同一份二进制切开关: KV 搬运量减半 (mte2 忙时约 -211us),")
            print(f"      端到端只回收 38.16us (755.14 -> 716.98), 兑现率 <20%。")
            print(f"      ★更正: 先前把它记成'收益为零/反而 +31%'是错的 —— 那次是同一改动里")
            print(f"      顺带把缓冲从双降单 (第 1 条, 值 +35%), 两件事被混在一起了。")
        #  ★msprof 只看得见 device 时间 —— host 侧的开销一个字都不报。
    #    2026-08-28 实撞: 给 host tiling 加了一次 MultiCoreMatmulTiling::GetTiling 搜索后,
    #    aic_time 只从 20.34 涨到 27.87 (+37%), 而端到端 per-launch 从 26.5 涨到 306.4 (**11 倍**),
    #    那 279us 全在 host 且 msprof 看不到。凡是在 host tiling 里加了搜索/枚举/缓存失效的改动,
    #    判收益必须看端到端 (core/measure 或驱动的 per-launch), 不能只看这张表。
    #  ★三条主 pipe 之和与 aic_time 的关系 —— 比总重叠度更能指路:
        #    严格相加 (比值 ~1.00) = **全链串行**, 要逐道缝找, 只修其中一道不会有整体收益
        #    (GemmV3 2026-08-27: 照 H10 修了 MTE1↔M 那道, 重叠度 1.22->1.23 纹丝不动,
        #     因为 MTE2 与 cube 之间那道更宽的缝没动)。
        m3 = [g(c) for c in ("aic_mte2_time(us)", "aic_cube_time(us)", "aic_mte1_time(us)")]
        if all(x for x in m3):
            ser = sum(m3) / aic
            tag = ("   <- 严格相加 = mte2/cube/mte1 全链串行, 逐道缝找" if 0.95 <= ser <= 1.05
                   else ("   <- <1 说明三者已有重叠" if ser < 0.95 else ""))
            print(f"  其中 mte2+cube+mte1 / aic_time{' ' * 11}{ser:12.2f}{tag}")
    #  ★标量流水的判读 —— 工具替人把方向定下来, 不要只摆一个 stall 数字出去。
    #    `aic_scalar_mte1_stall` 的名字有歧义, 极易读成"标量拖累 MTE1"; 它实际是**标量停在
    #    那里等 MTE1**。方向读反的代价是整整一轮白工 (gemm_v3 2026-08-28: 据此去削减循环内的
    #    64 位乘法与除法, checksum 逐位不变而耗时纹丝不动)。判据是 single+dual 这个上界。
    sc_t = g("aic_scalar_time(us)")
    sc_run = (g("aic_scalar_single_time(us)") or 0) + (g("aic_scalar_dual_time(us)") or 0)
    sc_stall = g("aic_scalar_mte1_stall_time(us)")
    if sc_t and sc_stall is not None:
        share = sc_stall / sc_t
        print(f"  标量真正执行 (single+dual){' ' * 14}{sc_run:12.2f}"
              f"   <- 削减循环内标量运算的收益**上界**就是这个数")
        if share >= 0.30:
            print(f"  其中 mte1_stall 占 scalar_time{' ' * 10}{share:12.0%}"
                  f"   <- 标量是在**等** MTE1, 不是拖累它")
            print(f"{' ' * 4}方向: 该降的是 MTE1 的指令条数 (L1->L0 的 16x16 分形数, 由 m0/n0 定),"
                  f" 不是标量代码本身;")
            print(f"{' ' * 4}      也别用'保存上一拍参数'的写法去排软件流水 —— 那会再加标量, 净亏"
                  f" (detsync_lint H10 已写入该前置条件)。")
    #  ★核间不均衡 —— 本表此前只打 mean 和 max 两个数, 比值每次都要人去手算, 而它恰恰是
    #    "tile 数不是核数整数倍"这类问题的唯一直接信号 (GemmV3 2026-08-27: cube 1.55 而
    #    MatMulV2 1.01, 端到端差 29%). 判读: 拿 **cube** 那一行, 它最接近纯计算量之比;
    #    aic_time 那一行会被等待稀释, 偏小。
    for key, label in (("aic_cube_time(us)", "cube"), ("aic_time(us)", "aic 总")):
        if key in sig and sig[key]["mean"]:
            r = sig[key]["max"] / sig[key]["mean"]
            tail = ""
            if key.endswith("cube_time(us)") and r > 1.10:
                #  ★这里只报**现象**并指出查成因的命令, 不再暗示"应该把 tile 数凑成核数整数倍"。
                #    改 m0/n0 会同时改访问模式与 kFrag 整除性, 那两维没有判据在看:
                #    gemm_v3 2026-08-28 实测, wave_eff 从 0.667 提到 1.000 的 128x192 反而慢 73%,
                #    分形系数最小的 176x176 更是慢 155% —— 十个 tile 点扫下来默认的最优。
                tail = ("   <- 最忙核比平均多干这么多。查成因: python -m core.mmplan --wave "
                        "--m <M> --n <N> --k <K> --m0 <m0> --n0 <n0>; "
                        "★但别据此直接改 tile —— 改分块必然同时改访问模式, 净收益一律实测")
            print(f"  核间不均衡 {label:<6s} (max/mean){' ' * 12}{r:12.2f}{tail}")
    #  ★冲突比的分母是 **task_cyc**, 不是向量周期 —— 已从 CANN 源码核实:
    #    tools/profiler/.../common_func/constant.py 里 vec_*_cflt_ratio 全走
    #    pmu_metric_calculate_without_freq(1.0, pmu, task_cyc) = pmu / task_cyc。
    #    而 aiv_total_cycles / 主频 实测**恰好等于** aiv_time (gmfr: 4489.9 vs 4489.86;
    #    12233.4 vs 12233.44), 故 task_cyc 就是整个 AIV 窗口的周期数。
    #    ★于是 ratio x aiv_total_cycles / 频率 **接近**墙钟微秒, 不用再乘任何占比。
    #    2026-08-25 第一版按 aiv_vec_time x ratio 算, 把 i2 的冲突低估到 216us (实际 657us) ——
    #    分母搞错一次就差 3 倍, 故把出处写死在注释里。
    #  ★**2026-10-23 订正: 分母不是 aiv_total_cycles, 是 floor(TaskDur) x freq。**
    #    上面那句"aiv_total_cycles / 主频恰好等于 aiv_time"是对的 —— 但它证的是 F1,
    #    **不是 ratio 的分母**。`08` §3.78/§3.79 从原始 PMU 记录逐行逐位解出所有
    #    `*_ratio` 的分母是 `floor(TaskDur) x freq`。本样本上旧式偏 -0.071%
    #    (157.359 vs 157.470 us); **短核上会放大到百分之几** (floor 的相对影响)。
    #    改用 cflt_wall_us(), 它乘的是 floor(Task Duration)。
    #    ★这是"推理正确而前提没核"的又一例: 从一条成立的等式推出了另一条不成立的。
    tcyc = g("aiv_total_cycles")
    cf = g("aiv_vec_total_cflt_ratio")
    freq = (g("Current Freq") or 1800.0) * 1e6
    if tcyc and cf is not None and freq:
        bg = g("aiv_vec_bankgroup_cflt_ratio") or 0.0
        bk = g("aiv_vec_bank_cflt_ratio") or 0.0
        _td = g("Task Duration(us)") or (tcyc / freq * 1e6)
        #  ★2026-10-23: 分母由 Op Type 决定 —— mix 是常数 floor(TaskDur)xfreq,
        #    纯 cube/vector 是本行 total_cycles。两者在 mix 上只差 0.17%,
        #    在纯 vector 上差到 1.79 倍, 所以不能共用一条式子 (见 ratio_window_us)。
        _ot = op_type_of(out_dir, kernel)
        if not _ot:
            print("  UB 冲突折算时间                        ★读不到 Op Type, 不折算 —— "
                  "分母取哪个没有依据 (mix 用 floor(TaskDur), 纯核用本行 total_cycles, "
                  "两者最大差 1.79 倍, 见 08 §3.82)")
            return sig
        _win = ratio_window_us(_ot, _td, total_cycles=tcyc, freq_mhz=freq / 1e6)
        us = lambda r: r * _win
        aivt = g("aiv_time(us)"); aict = g("aic_time(us)")
        crit = ("AIV 是关键路径, 这笔就是墙钟" if (aivt and aict and aivt >= aict)
                else "AIC 是关键路径, 这笔时间未必进 wall")
        _why = (f"floor(TaskDur) {math.floor(_td):.0f}us (Op Type=mix)" if _ot.lower() == "mix"
                else f"total_cycles/freq {_win:.3f}us (Op Type={_ot}, 分母是本行周期数)")
        print(f"  UB 冲突折算时间                        {us(cf):12.1f} us"
              f"   <- ratio {cf:.4f} x {_why}"
              f"  (见 08 §3.82)")
        print(f"  {'':38s}其中 bankgroup {us(bg):.1f}us / bank {us(bk):.1f}us; {crit}")
        if bg > 0.02:
            print(f"  {'':38s}★bankgroup 占大头 -> 查 UB 行跨距是不是大的 2 的幂 "
                  f"(各行同偏移落进同一 group), 加个非 2 的幂的 pad 通常直接掉下来")
    return sig



#  ---- PipeTimeline 判读: 实测占用 (见 P6) ------------------------------------------------
#  产物是 trace.json (chrome tracing 格式), 事件形如
#     {"pid": "core0.cubecore", "tid": "CUBE", "name": "CUBE", "ts": ..., "dur": ...}
#  ★必须**按 pid 分核**再算占用: 把所有核的事件混在一条时间轴上, 各核的忙区互相填补,
#    空隙会被掩盖 (实测 4~7% 被算成 0.9%), 忙时占比还会算出 500% 这种数字。
def workrate(dir_a: str, dir_b: str, label_a: str = "A", label_b: str = "B",
             vol_tol: float = 0.10, rate_tol: float = 0.10) -> dict:
    """两份采集: 差别是**多搬了字节**还是**同样的字节搬得慢**? 这一问分岔整条优化路线。

    ★为什么需要它 (2026-09-17 sfag_train 实撞): det=1 比 det=0 慢 24.16 ms (2.8 倍),
      per-pipe 归责显示增量全在 aiv 的 mte2/mte3/scalar 上。看到这里很容易接着问
      "怎么把这些搬运减下来" —— 但那是**错的下一问**。把带宽乘回耗时还原成字节:
          aiv_ub 读 0.549 GB vs 0.533 GB (1.03x)   带宽 14.7 vs 40.7 GB/s (0.36x)
          aiv_ub 写 0.594 GB vs 0.583 GB (1.02x)   带宽 15.9 vs 44.4 GB/s (0.36x)
      **字节一样多, 带宽只有三分之一**。所以要减的不是搬运量, 是取得的带宽 ——
      方向从"删搬运"变成"查粒度/对齐/停顿"。两者的手段与上限完全不同。

    ★作用域 (如实记, 免得被当成万能归责):
      本判据只回答**量 vs 率**, **不回答"哪条 pipe 花了多少时间"**。后者必须看 pipe 时间增量
      (见 core/diagnose abdiff 的实证: 按字节推出来的优先级曾经是反的)。
      两条判据分工: abdiff 说"时间花在哪条 pipe", workrate 说"那条 pipe 是被量压的还是被率压的"。

    ★它读的是各 *_bw(GB/s) 计数器与 Duration(us), 字节由 bw x duration 还原 ——
      因此它继承这些计数器的口径: 计数器若只覆盖部分通路, 还原出的字节也只是那部分。
      不要把它当成端到端流量核算。
    """
    a, b = extract_signals(dir_a), extract_signals(dir_b)

    def _dur(sig):
        for k in ("Duration(us)", "Task Duration(us)", "aicore_time(us)"):
            if k in sig:
                return sig[k]["mean"]
        return None

    da, db = _dur(a), _dur(b)
    if not da or not db:
        out = {"verdict": "NO_DURATION",
               "note": ("两份产物里找不到 Duration(us) —— 无法把带宽还原成字节。"
                        "确认喂的是含 kernel_details.csv 的采集目录。")}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return out

    #  两类计数器都收:
    #    *_bw(GB/s)   —— 带宽, 字节要靠 bw x duration **反推**
    #    *_datas(KB)  —— **直接**的字节计数 (MemoryAccess 集给的就是这种), 证据更强, 不用反推
    #  后者才覆盖 GM/HBM 通路 (aic/aiv_read_main_memory_datas), 而 Memory 集的 *_bw 只有片上。
    #  2026-09-17 实撞: 只认 *_bw 时, 喂 MemoryAccess 产物直接报 NO_BW_COUNTERS ——
    #  而那正是唯一能回答"GM 层面是不是真的少搬"的那一组。
    cols = sorted(c for c in (set(a) & set(b))
                  if c.endswith("_bw(GB/s)") or c.endswith("_datas(KB)"))
    rows, vol_diff, rate_diff = [], [], []
    for c in cols:
        va, vb = a[c]["mean"], b[c]["mean"]
        if va <= 0 and vb <= 0:
            continue                      # 该通路没用到, 与本判据无关
        if c.endswith("_datas(KB)"):
            gba, gbb = va * 1e-6, vb * 1e-6          # KB -> GB (十进制, 只用于比值)
            bwa = gba / (da * 1e-6) if da else 0.0   # 由字节与耗时**导出**速率
            bwb = gbb / (db * 1e-6) if db else 0.0
        else:
            bwa, bwb = va, vb
            gba, gbb = bwa * da * 1e-6, bwb * db * 1e-6   # GB/s x us x 1e-6 = GB
        vr = (gbb / gba) if gba > 0 else float("inf")
        rr = (bwb / bwa) if bwa > 0 else float("inf")
        same_vol = abs(vr - 1.0) <= vol_tol
        if same_vol and abs(rr - 1.0) > rate_tol:
            kind = "同量不同率"
        elif not same_vol:
            kind = "量不同"
        else:
            kind = "量率都相当"
        rows.append(dict(counter=c, gb_a=round(gba, 4), gb_b=round(gbb, 4),
                         vol_ratio=round(vr, 3), bw_a=round(bwa, 2), bw_b=round(bwb, 2),
                         rate_ratio=round(rr, 3), kind=kind, _w=max(gba, gbb)))

    #  ★按字节量加权决定**总判定**: 不加权时, 一条只占 1% 字节的通路就能把结论翻掉。
    #  2026-09-17 真产物上实撞 —— aiv_main_mem 只有 0.005 GB (主干 aiv_ub 是 0.547 GB),
    #  它的量比 1.44 把 SAME_WORK_LOWER_RATE 翻成了 MIXED, 而那 0.002 GB 的差毫无行动价值。
    #  自造的两计数器自检样本发现不了这个, 只有真产物能 —— 判据必须在真产物上跑过。
    #  次要通路仍然照常列出 (数据不丢), 只是不参与总判定。
    wmax = max((r["_w"] for r in rows), default=0.0)
    major = [r for r in rows if wmax > 0 and r["_w"] >= 0.05 * wmax]
    minor = [r["counter"] for r in rows if r not in major]
    vol_diff = [r["counter"] for r in major if r["kind"] == "量不同"]
    rate_diff = [r["counter"] for r in major if r["kind"] == "同量不同率"]
    for r in rows:
        r.pop("_w", None)

    #  ★L2 命中率: L2Cache 集给的是**原始计数** (*_cache_hit / *_cache_miss_allocate),
    #  既不是带宽也不是字节, 上面两类都读不到它 —— 而它恰恰能回答"同样的字节为什么慢":
    #  量相同、粒度相同而命中率差一倍, 就是同样的字节多付了一倍 HBM 往返。
    #  2026-09-17 sfag_train 实撞: 三个 metric 集里只有这一组说清了机制
    #  (AIC 写命中 39.5% vs 98.1%, miss 次数差 47.6 倍), 而当时没有任何工具能读它, 只能手写提取。
    hits = []
    for c in sorted(set(a) & set(b)):
        if not c.endswith("_cache_hit"):
            continue
        cm = c[: -len("_cache_hit")] + "_cache_miss_allocate"
        if cm not in a or cm not in b:
            continue
        ha, ma = a[c]["mean"], a[cm]["mean"]
        hb, mb = b[c]["mean"], b[cm]["mean"]
        ta, tb = ha + ma, hb + mb
        if ta <= 0 or tb <= 0:
            continue
        hits.append(dict(channel=c[: -len("_cache_hit")],
                         hit_pct_a=round(ha / ta * 100, 1), hit_pct_b=round(hb / tb * 100, 1),
                         accesses_a=int(ta), accesses_b=int(tb),
                         miss_a=int(ma), miss_b=int(mb),
                         miss_ratio=round(ma / mb, 2) if mb > 0 else None))
    hits.sort(key=lambda x: -(x["miss_a"] + x["miss_b"]))

    if not rows and hits:
        worst = max(hits, key=lambda x: (x["hit_pct_b"] - x["hit_pct_a"]))
        verdict = "CACHE_ONLY"
        note = (f"这两份产物只有缓存计数, 没有字节/带宽列 —— 量的对比要另采 MemoryAccess 集。"
                f"命中率差最大的是 {worst['channel']}: {label_a} {worst['hit_pct_a']}% vs "
                f"{label_b} {worst['hit_pct_b']}%。**量相同而命中率差**就是同样的字节多付了 "
                f"HBM 往返, 手段是提高复用/驻留, 不是减搬运。")
    elif not rows:
        verdict, note = "NO_BW_COUNTERS", "两份产物没有共同的 *_bw(GB/s) 或 *_datas(KB) 计数器可比。"
    elif rate_diff and not vol_diff:
        verdict = "SAME_WORK_LOWER_RATE"
        note = (f"{label_b} 与 {label_a} 搬的字节相当 (各通路量比都在 ±{vol_tol:.0%} 内), "
                f"差别在**取得的带宽**。下一步查粒度/对齐/停顿, 不要去删搬运 —— "
                f"减不下来的东西减它是白费。")
    elif vol_diff and not rate_diff:
        verdict = "MORE_WORK"
        note = (f"{label_b} 确实搬了更多字节 ({', '.join(vol_diff)})。"
                f"这是工作量差异, 先减搬运本身。")
    elif vol_diff and rate_diff:
        verdict = "MIXED"
        note = (f"量与率都有差: 量差在 {', '.join(vol_diff)}, 率差在 {', '.join(rate_diff)}。"
                f"两条要分开处理, 别用一个手段去打两个成因。")
    else:
        verdict = "EQUIVALENT"
        note = "量与率都相当 —— 时间差不在这些访存计数器覆盖的范围内, 去看 pipe 时间增量。"

    out = {"verdict": verdict, "labels": {"a": label_a, "b": label_b},
           "duration_us": {label_a: round(da, 1), label_b: round(db, 1)},
           "counters": rows,
           "次要通路(<5%字节量, 不参与总判定)": minor,
           **({"L2 命中率": hits} if hits else {}),
           "note": note,
           "作用域": "本判据只答'量 vs 率'; '时间花在哪条 pipe'要看 core/diagnose abdiff。"}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def selftest_workrate() -> int:
    """负例取自 2026-09-17 sfag_train 的真实数字; 正例是构造的等价与多量两种。"""
    import tempfile
    bad = 0
    print("=== workrate 自检 (量 vs 率) ===")

    def mk(td, name, dur_us, bws):
        d = os.path.join(td, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "kernel_details.csv"), "w") as f:
            cols = ["Duration(us)"] + list(bws)
            f.write(",".join(cols) + "\n")
            f.write(",".join([str(dur_us)] + [str(v) for v in bws.values()]) + "\n")
        return d

    with tempfile.TemporaryDirectory() as td:
        #  真实数字: det=1 37286us @ 14.727/15.925 GB/s vs det=0 13123us @ 40.659/44.433
        a = mk(td, "det1", 37286.0, {"aiv_ub_read_bw(GB/s)": 14.727, "aiv_ub_write_bw(GB/s)": 15.925})
        b = mk(td, "det0", 13123.0, {"aiv_ub_read_bw(GB/s)": 40.659, "aiv_ub_write_bw(GB/s)": 44.433})
        r = workrate(a, b, "det1", "det0")
        ok = r["verdict"] == "SAME_WORK_LOWER_RATE"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例(真实数字): 同字节低带宽 -> SAME_WORK_LOWER_RATE "
              f"(实得 {r['verdict']})")

        #  真多搬: 同带宽, 时间翻倍 -> 字节翻倍
        c = mk(td, "more", 26246.0, {"aiv_ub_read_bw(GB/s)": 40.659, "aiv_ub_write_bw(GB/s)": 44.433})
        r = workrate(b, c, "base", "more")
        ok = r["verdict"] == "MORE_WORK"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 正例: 同带宽而耗时翻倍 = 字节翻倍 -> MORE_WORK "
              f"(实得 {r['verdict']})")

        #  完全相同 -> EQUIVALENT, 不许报出差异
        r = workrate(b, b, "x", "x")
        ok = r["verdict"] == "EQUIVALENT"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 正例: 同一份产物自比 -> EQUIVALENT (实得 {r['verdict']})")

        #  ★真产物实撞 (2026-09-17): 一条只占 1% 字节量的通路把总判定翻掉。
        #  主干 aiv_ub 0.547 GB 同量低率, 而 aiv_main_mem 只有 0.005 GB 却量比 1.44 ——
        #  不加权时报 MIXED, 把"去查粒度"这条正确方向淹掉。自造的两计数器样本发现不了,
        #  只有真产物能。
        e = mk(td, "wsub_a", 37286.0, {"aiv_ub_read_bw(GB/s)": 14.727,
                                       "aiv_main_mem_read_bw(GB/s)": 0.133})
        f_ = mk(td, "wsub_b", 13123.0, {"aiv_ub_read_bw(GB/s)": 40.659,
                                        "aiv_main_mem_read_bw(GB/s)": 0.542})
        r = workrate(e, f_, "det1", "det0")
        ok = r["verdict"] == "SAME_WORK_LOWER_RATE"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例(真产物): 1% 字节量的通路不许翻掉总判定 "
              f"(实得 {r['verdict']}, 次要通路 {r.get('次要通路(<5%字节量, 不参与总判定)')})")

        #  ★L2 命中率 (2026-09-17 真数字): 只有缓存计数、没有字节列时要报 CACHE_ONLY 并算出命中率,
        #  不能因为"没有 *_bw 列"就报 NO_BW_COUNTERS 把这一组丢掉 —— 三个 metric 集里
        #  只有这一组说清了 sfag det=1 慢在哪 (AIC 写命中 39.5% vs 98.1%, miss 差 47.6 倍)。
        c1 = mk(td, "l2a", 37304.0, {"aic_write_cache_hit": 11816634.0,
                                     "aic_write_cache_miss_allocate": 18134902.0})
        c2 = mk(td, "l2b", 13001.8, {"aic_write_cache_hit": 20133299.0,
                                     "aic_write_cache_miss_allocate": 381053.0})
        r = workrate(c1, c2, "det1", "det0")
        hh = r.get("L2 命中率", [])
        ok = (r["verdict"] == "CACHE_ONLY" and len(hh) == 1
              and abs(hh[0]["hit_pct_a"] - 39.5) < 0.2 and abs(hh[0]["hit_pct_b"] - 98.1) < 0.2)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 只有缓存计数 -> CACHE_ONLY 且算出命中率 "
              f"(实得 {r['verdict']}, {[(h['hit_pct_a'], h['hit_pct_b']) for h in hh]})")
        #  缺 Duration 时必须说清楚, 不能拿 0 当字节静默算下去
        d = os.path.join(td, "nodur")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "k.csv"), "w") as f:
            f.write("aiv_ub_read_bw(GB/s)\n40.0\n")
        r = workrate(d, d, "n", "n")
        ok = r["verdict"] == "NO_DURATION"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 没有 Duration 列 -> NO_DURATION 而不是静默算 "
              f"(实得 {r['verdict']})")
    print(f"workrate 自检: 6 条, {bad} 条未过")
    return bad


#  ★单块事件判据 (2026-10-21 立, 代价见下): 一个 pipe 若只有很少几段、却覆盖了几乎整个窗口,
#    那不是"这条流水忙了 99%", 而是**采样在这一条流水上没有细分** —— 常见于 spin 等待:
#    标量核在 `cube_sync_block_wait` 之类的轮询里不出来, 探针插不进去, 整段被记成一段。
#    实撞: SMLA 的 aiv_scalar_time=1653.8us (92.3% of e2e) 被读成"标量是瓶颈, 去减少标量指令",
#    据此切了两刀只拿回 14.5us; 时间线里 AIV 的 SCALAR **只有 6 段, 一段 2444.9us** ——
#    它根本没有"忙闲"可言, 那一列不可判读。这个比值越接近 1, 越该怀疑是等待而不是工作。
MONOLITH_MIN_OCC = 0.90
MONOLITH_MAX_N = 10


def _union(iv):
    """区间并集。同一 pipe 的事件**会互相重叠**, 直接把 dur 求和会把忙时算多。"""
    out = []
    for s, t in sorted(iv):
        if out and s <= out[-1][1]:
            if t > out[-1][1]:
                out[-1][1] = t
        else:
            out.append([s, t])
    return out


def _span(iv) -> float:
    return sum(t - s for s, t in iv)


def _inter(a, b) -> float:
    """两个**已并集化**区间表的交叠总长。"""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        t = min(a[i][1], b[j][1])
        if t > s:
            tot += t - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def _complement(u, t0: float, t1: float):
    out, prev = [], t0
    for s, t in u:
        if s > prev:
            out.append([prev, s])
        prev = max(prev, t)
    if prev < t1:
        out.append([prev, t1])
    return out


def _read_task_duration(prof_dir: str):
    """从 OpBasicInfo.csv 里取未放大时的 Task Duration (us), 取不到返回 None。

    ★为什么必须有它: 开了 PipeTimeline 之后内核本身被拖慢 —— SMLA 实测 2470us vs 1786us
      (1.38x, 探针开销)。于是 trace 里的**绝对忙时**仍可用 (CUBE 1088us vs op_summary 的
      aic_mac 1065us, 差 2%), 但**占比**被系统性摊薄。不报这个倍数, 读者会拿 44.1% 去和
      op_summary 的 59.4% 对照, 得出"采集之间口径不一致"这种错误的结论。
    """
    #  ★按表头取列, 不按 rsplit(",",1) —— 这份 csv 每行**以逗号结尾**, 于是最后一个字段是空串,
    #    按"最后一列"取会永远拿到 "" (本仓的 CSV 布局坑已经踩过不止一次)。
    import csv as _csv
    vals = []
    for p in glob.glob(os.path.join(prof_dir, "**", "OpBasicInfo.csv"), recursive=True):
        try:
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                rd = _csv.reader(f)
                hdr = next(rd, [])
                idx = next((i for i, h in enumerate(hdr) if h.strip().startswith("Task Duration")), None)
                if idx is None:
                    continue
                for row in rd:
                    if len(row) > idx and row[idx].strip():
                        vals.append(float(row[idx]))
        except Exception:
            continue
    return max(vals) if vals else None


#  pid 名字里的侧别提示。命中即用; 命中不了才看流水集合 —— 名字是**某一代 tracing 的格式**,
#  不能当唯一依据 (本仓在别处已经为同类写死付过代价: PipeTimeline 的档位名)。
_AIC_HINT = ("cubecore", "aic", "cube")
_AIV_HINT = ("veccore", "aiv", "vector", "vec")
#  两侧都能出现的流水 (MTE2/MTE3/SCALAR), 只有 CUBE/VECTOR 这类才分得开。
_CUBE_PIPES = {"CUBE", "MAC", "AIC_MAC", "CUBE_ACC"}
_VEC_PIPES = {"VECTOR", "VEC", "AIV_VEC", "VEC_ALU"}


def _pipe_side(pid: str, pipes) -> str | None:
    """'aic' | 'aiv' | None。先看 pid 名, 再看它携带的流水集合, 都不成立就给 None。"""
    s = str(pid).lower()
    if any(h in s for h in _AIC_HINT):
        return "aic"
    if any(h in s for h in _AIV_HINT):
        return "aiv"
    ps = {str(p).upper() for p in pipes}
    if ps & _CUBE_PIPES:
        return "aic"
    if ps & _VEC_PIPES:
        return "aiv"
    return None


def _trace_block_dim(prof_dir: str):
    """这次 launch 的 Block Dim (来自 OpBasicInfo.csv)。读不到返回 None —— **不猜**。"""
    for p in sorted(glob.glob(os.path.join(prof_dir, "**", "OpBasicInfo.csv"), recursive=True)):
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        if len(lines) < 2:
            continue
        head = [h.strip() for h in lines[0].split(",")]
        val = [v.strip() for v in lines[1].split(",")]
        if "Block Dim" in head:
            i = head.index("Block Dim")
            if i < len(val) and val[i].isdigit():
                return int(val[i])
    return None


def timeline_report(prof_dir: str, task_duration_us=None) -> dict:
    """PipeTimeline trace.json -> 按核分流的**实测占用 + 卡点归因**。

    与 `extract()` 的忙时表分工: 那张表是**总量** (每条流水一共忙了多久),
    这张表是**形态** (忙在哪段、空在哪段、空的时候另一侧在干什么)。
    总量答不了"这 500us 的空闲是 cube 在等 AIV 还是在等自己的下一条指令" —— 形态答得了。
    """
    import json as _json
    tj = sorted(glob.glob(os.path.join(prof_dir, "**", "trace.json"), recursive=True))
    if not tj:
        raise SystemExit(f"{prof_dir} 下没有 trace.json —— PipeTimeline 的产物不是 CSV, "
                         f"采集时要给 --aic-metrics=PipeTimeline")
    d = _json.load(open(tj[0]))
    ev = d.get("traceEvents", d if isinstance(d, list) else [])
    per = {}
    for e in ev:
        if e.get("ph") != "X":
            continue
        per.setdefault(e["pid"], {}).setdefault(e["name"], []).append((e["ts"], e.get("dur", 0.0)))

    out = {"file": tj[0], "cores": {}, "coupling": {}, "warnings": []}
    intervals = {}                      # pid -> name -> 已并集化的绝对区间
    for pid in sorted(per):
        by = per[pid]
        t0 = min(t for v in by.values() for t, _ in v)
        t1 = max(t + dd for v in by.values() for t, dd in v)
        win = t1 - t0
        rec = {"window_us": win, "pipes": {}}
        intervals[pid] = {}
        for k, v in by.items():
            v = sorted(v)
            busy = sum(dd for _, dd in v)
            gaps = [v[i][0] - (v[i - 1][0] + v[i - 1][1]) for i in range(1, len(v))]
            gaps = [g for g in gaps if g > 1e-9]
            u = _union([(s, s + dd) for s, dd in v])
            intervals[pid][k] = u
            #  ★占比按**并集**算, 不按求和 —— 同一 pipe 的事件会互相重叠, 求和能算出 >100%。
            occ = _span(u) / win if win else 0.0
            #  ★判"单块"看**最长那一段**占窗口的比例, 不看事件数 —— 实撞: core1.veccore1 的
            #    SCALAR 有 22 段 (超过任何按条数的门限), 但其中一段就 2445us / 2470us = 99%,
            #    按条数判会漏掉它, 于是那一侧的并集仍被顶到 100%。
            max_seg = max((dd for _, dd in v), default=0.0)
            mono = (max_seg / win >= MONOLITH_MIN_OCC) if win else False
            rec["pipes"][k] = {
                "n": len(v), "busy_us": busy, "union_us": _span(u), "occ": occ,
                "max_seg_us": max_seg,
                "idle_us": sum(gaps), "n_gaps": len(gaps),
                "avg_gap_ns": (sum(gaps) / len(gaps) * 1000.0) if gaps else 0.0,
                "max_gap_us": max(gaps) if gaps else 0.0,
                "attributable": not mono,
            }
            if mono:
                out["warnings"].append(
                    f"{pid} 的 {k}: 最长一段 {max_seg:.1f}us 占窗口 {max_seg / win * 100:.1f}% "
                    f"(共 {len(v)} 段) —— **不可判读**, 多半是等待被记成一整段 (spin), "
                    f"不能当'忙'读")
        #  ---- 同核流水互补性: 一条流水空闲时, 同核别的流水在忙吗 ----------------------
        #  ★2026-10-21 SMLA 实撞: `aiv_mte2 1208.8 + aiv_vec 864.0 = 2072.8`, 墙 1783.2
        #    —— 两条流水只重叠了 ~290us。但"忙时总量"答不了这是 **MTE2 自己饱和** 还是
        #    **VEC 把 MTE2 堵住了 (串行)**; 两者的手段完全相反 (前者要减字节, 后者要重排)。
        #    分开它们的唯一办法是把 A 的**空闲区间**拿去和 B 的**忙区**求交:
        #      idle_covered (别人在忙) -> A 不是饱和, 是被同核的串行段挡住了;
        #      idle_bare   (整核都闲)  -> 该核在等**核外**的东西 (跨核握手/依赖上游),
        #                                 此时盯着 A 自己优化是白干。
        #    此前这条推理是人肉做的, 推错过两次 (见 --refs P7)。
        _attr = [k for k, p in rec["pipes"].items() if p["attributable"]]
        for k in _attr:
            others = _union([iv for o in _attr if o != k
                             for iv in intervals[pid][o]])
            idle_iv = _complement(intervals[pid][k], t0, t1)
            #  ★口径是 **窗口 − 该流水并集**, 不是上面表里的 `idle_us` (只累计**段间**空隙)。
            #    差别在首尾: 一条只有 1 段、却只覆盖窗口 40% 的流水, 段间空隙 = 0,
            #    按 idle_us 算它根本不会进互补性表 —— 而这恰恰是最该判"串行还是等核外"的形态。
            #    自检里 core0 的 MTE2 就是这个形状, 第一版因此判反 (实撞)。
            idle_full = _span(idle_iv)
            cov = _inter(idle_iv, others) if others else 0.0
            #  ★并集只说"有人忙", 不说是谁。要动手就得知道压着它的是 VEC 还是 MTE3 ——
            #    两者手段完全不同 (重排计算 vs 改写出节奏)。分项取交集, 报最大的那个。
            #    2026-10-21 SMLA: MTE2 空闲 794us 中 VEC 占 501 / MTE3 占 346, 并集 719 —
            #    只看并集会把它读成"都是 VEC 的锅", 而 MTE3 那 346 是另一条要单独动的线。
            by_pipe = {}
            for o in _attr:
                if o == k:
                    continue
                #  ★SCALAR 不当"占位者": 它是**发射口**, 任何流水在跑它都忙
                #    (C5 trace 里 91%), 于是它对任何空闲都能"解释", 挑不出信息量。
                #    它真堵住 MTE2 的那一类 (发射饥饿) 要看下面的空隙形态, 不靠这一列。
                if "SCALAR" in str(o).upper():
                    continue
                v = _inter(idle_iv, intervals[pid][o])
                if v > 0.0:
                    by_pipe[o] = v
            top = max(by_pipe.items(), key=lambda z: z[1]) if by_pipe else (None, 0.0)
            rec["pipes"][k]["idle_full_us"] = idle_full
            rec["pipes"][k]["idle_covered_us"] = cov
            rec["pipes"][k]["idle_bare_us"] = max(0.0, idle_full - cov)
            rec["pipes"][k]["idle_by_us"] = by_pipe
            rec["pipes"][k]["idle_top_pipe"] = top[0]
            rec["pipes"][k]["idle_top_us"] = top[1]
        out["cores"][pid] = rec

    #  ---- 按**物理核**合并 cube / vector 两侧, 做跨核卡点归因 ----------------------------
    #  物理核 = pid 的第一个点号前一段 (950PR 上是 "core3.cubecore" / "core3.veccore0")。
    #  ★两侧的判定**不写死 pid 命名**: 这个名字是某个 SoC 的 tracing 格式, 换平台 (910B 的
    #    TimelineDetail 等) 就变了。先按名字提示, 认不出再按该 pid 携带的流水集合兜底,
    #    两边都不成立就如实报"分不出两侧", 而不是把它们当成同一侧静默合并。
    phys = {}
    for pid in intervals:
        phys.setdefault(pid.split(".", 1)[0], []).append(pid)

    #  ---- 这次 trace 覆盖了全片多少核? -------------------------------------------------
    #  ★2026-10-21 SMLA 实撞: OpBasicInfo.csv 写 Block Dim=28 / Mix Block Dim=56, 而 trace.json
    #    里只有 core0..core5 六个核 —— 实机 PipeTimeline **只导出一部分核** (片上 trace 空间有限)。
    #    不报这一条, "AIC 忙 79.5%" 会被读成全片 79.5%; 它其实是 6 核样本。
    #    这里只报"样本 / 全体", **不对没看见的核做外推** —— 它们更忙还是更闲, 这份 trace 答不了。
    _bd = _trace_block_dim(prof_dir)
    out["coverage"] = {"traced_cores": len(phys), "block_dim": _bd}
    if _bd and len(phys) < _bd:
        out["warnings"].append(
            f"trace 只覆盖 {len(phys)}/{_bd} 个核 (Block Dim={_bd}) —— 下面的**每核**数字是"
            f"**样本**, 不是全片。未导出的 {_bd - len(phys)} 个核更忙还是更闲, 这份 trace 答不了; "
            f"要全片只能靠按流水的忙时总量 (extract) 或减核重测。")
    out["side_of"] = {}
    for pid in intervals:
        out["side_of"][pid] = _pipe_side(pid, intervals[pid].keys())
    for core, pids in sorted(phys.items()):
        all_iv = [iv for p in pids for iv in intervals[p].values()]
        t0 = min(i[0][0] for i in all_iv if i)
        t1 = max(i[-1][1] for i in all_iv if i)
        win = t1 - t0
        cube_pids = [p for p in pids if out["side_of"][p] == "aic"]
        vec_pids = [p for p in pids if out["side_of"][p] == "aiv"]
        unknown = [p for p in pids if out["side_of"][p] is None]
        if unknown:
            out["warnings"].append(
                f"{core}: {', '.join(unknown)} 判不出是 cube 还是 vector 侧 —— "
                f"它们**没有**计入任一侧, 两侧占用会被低估。")
        #  ★不可判读的流水 (单块事件) **不进并集** —— 它一进来就把该侧的占用顶到 100%,
        #    于是"这一侧一直在忙"变成永远成立, 卡点归因直接失效。
        def side(ps):
            return _union([iv for p in ps for k, ivs in intervals[p].items()
                           if out["cores"][p]["pipes"][k]["attributable"] for iv in ivs])
        aic, aiv = side(cube_pids), side(vec_pids)
        aic_b, aiv_b = _span(aic), _span(aiv)
        aic_idle = _complement(aic, t0, t1)
        aiv_idle = _complement(aiv, t0, t1)
        out["coupling"][core] = {
            "window_us": win, "t0": t0, "t1": t1,
            "aic_pids": cube_pids, "aiv_pids": vec_pids,
            "aic_busy_us": aic_b, "aiv_busy_us": aiv_b,
            "aic_occ": aic_b / win if win else 0.0,
            "aiv_occ": aiv_b / win if win else 0.0,
            #  AIC 空着的那些时间里, 有多少是 AIV 在忙 (即 AIC 在等 AIV)
            "aic_idle_us": _span(aic_idle),
            "aic_idle_cov_by_aiv": _inter(aic_idle, aiv),
            "aiv_idle_us": _span(aiv_idle),
            "aiv_idle_cov_by_aic": _inter(aiv_idle, aic),
        }
    td = task_duration_us or _read_task_duration(prof_dir)
    out["task_duration_us"] = td
    wins = [c["window_us"] for c in out["cores"].values()]
    out["window_us"] = max(wins) if wins else 0.0
    if td and td > 0:
        out["inflation"] = out["window_us"] / td
        out["warnings"].append(
            f"探针放大: 时间线窗口 {out['window_us']:.1f}us vs 未插桩 Task Duration {td:.1f}us "
            f"= {out['inflation']:.2f}x。**绝对忙时可比, 占比不可直接与 op_summary 对照**。")
    return out


def _timeline_verdict(r: dict) -> str:
    """从跨核耦合里给一句**卡点**判词。只在一侧明显更满时下结论, 否则如实说分不开。"""
    if not r.get("coupling"):
        return "没有 cube/vector 两侧同时出现 —— 单核类型的内核, 看单流水的空隙形态即可。"
    cov = [c for c in r["coupling"].values() if c["window_us"] > 0]
    if not cov:
        return "耦合窗口为空。"
    aic = sum(c["aic_occ"] for c in cov) / len(cov)
    aiv = sum(c["aiv_occ"] for c in cov) / len(cov)
    #  AIC 空闲里被 AIV 忙覆盖的比例: 高 = AIC 在等 AIV, AIC 侧削不动墙钟
    ai = sum(c["aic_idle_cov_by_aiv"] for c in cov)
    aj = sum(c["aic_idle_us"] for c in cov)
    av = sum(c["aiv_idle_cov_by_aic"] for c in cov)
    vb = sum(c["aiv_idle_us"] for c in cov)
    frac_aic_wait_aiv = (ai / aj) if aj else 0.0
    frac_aiv_wait_aic = (av / vb) if vb else 0.0
    if aiv - aic >= 0.10:
        return (f"**AIV 是连续忙的一侧** (AIV {aiv * 100:.0f}% vs AIC {aic * 100:.0f}%); "
                f"AIC 空闲的 {frac_aic_wait_aiv * 100:.0f}% 正落在 AIV 忙区里 -> "
                f"墙钟由 AIV 设定, 削 AIC 的工作量不减墙钟 (但会减 AIC 自身忙时, 看着像有效)。")
    if aic - aiv >= 0.10:
        return (f"**AIC 是连续忙的一侧** (AIC {aic * 100:.0f}% vs AIV {aiv * 100:.0f}%); "
                f"AIV 空闲的 {frac_aiv_wait_aic * 100:.0f}% 正落在 AIC 忙区里 -> "
                f"墙钟由 AIC 设定, AIV 侧加工作**只有落在 AIC 空闲段里才是免费的**。")
    return (f"两侧接近同步饱和 (AIC {aic * 100:.0f}% / AIV {aiv * 100:.0f}%), "
            f"单看占比分不出谁定墙钟 —— 要拿空隙形态 (判读二) 找具体那一个握手。")


def _idle_verdict(idle: float, cov: float, bare: float) -> str:
    """一条流水的空闲归到了谁身上 -> 下一步该动什么。

    抽成函数是为了自检能直接断言判词 —— 判据不能只在真机跑一遍时才被验证
    (同 `_kernel_name_miss` 的理由)。
    """
    frac = (cov / idle) if idle else 0.0
    if frac >= 0.6:
        return "被同核串行段挡住 -> 手段是重排/交错, 不是减它自己的量"
    if bare >= 0.6 * idle:
        return "整核在等核外 (跨核握手/上游依赖) -> 优化它自己没用"
    return "两者都有 -> 先按下面的分流水表定位具体的段"


def print_timeline(prof_dir: str, task_duration_us=None) -> None:
    r = timeline_report(prof_dir, task_duration_us)
    print(f"=== PipeTimeline 实测占用 + 卡点归因: {r['file']} ===")
    print("★这是**实测**, 与 PipeUtilization 的 aic_cube_time (派生的理论下限) 不是一回事, 见 --refs P6。")
    _cov = r.get("coverage") or {}
    if _cov.get("block_dim"):
        _n, _m = _cov.get("traced_cores"), _cov["block_dim"]
        _tag = "★样本" if _n < _m else "全片"
        print(f"覆盖: trace 里有 {_n} 个核, 本次 launch Block Dim={_m} —— {_tag}")
    print("\n[每核两侧]  物理核  窗口us   AIC忙(%)   AIV忙(%)  AIC等AIV   AIV等AIC")
    for core, c in r["coupling"].items():
        print(f"   {core:8s}{c['window_us']:8.1f}  {c['aic_busy_us']:7.1f}({c['aic_occ'] * 100:5.1f}%)"
              f"  {c['aiv_busy_us']:7.1f}({c['aiv_occ'] * 100:5.1f}%)"
              f"  {c['aic_idle_cov_by_aiv']:8.1f}  {c['aiv_idle_cov_by_aic']:8.1f}")
    print("\n[分流水] (窗内)  核.流水  段数  忙us  占比  空隙段  平均空隙ns  最长空隙us")
    for pid, c in r["cores"].items():
        for k, p in sorted(c["pipes"].items()):
            flag = "" if p["attributable"] else "  ★不可判读(单块事件)"
            print(f"   {pid:20s}{k:7s}{p['n']:6d}{p['busy_us']:8.1f}{p['occ'] * 100:6.1f}%"
                  f"{p['n_gaps']:8d}{p['avg_gap_ns']:12.0f}{p['max_gap_us']:11.1f}{flag}")
    #  ---- 同核互补性: 空闲是"别人在忙"还是"整核在等" ----------------------------------
    _rows = [(pid, k, p) for pid, c in r["cores"].items()
             for k, p in c["pipes"].items()
             if p.get("idle_full_us", 0.0) > 0.05 * c["window_us"]]
    if _rows:
        print("\n[同核互补性]  一条流水的空闲落在谁身上 —— **判'饱和'还是'被串行堵住'的唯一读数**")
        print("   核.流水                 空闲us   别人在忙us   整核都闲us   最大占位   判词")
        for pid, k, p in sorted(_rows, key=lambda z: -z[2]["idle_full_us"]):
            idle = p["idle_full_us"]
            cov = p.get("idle_covered_us", 0.0)
            bare = p.get("idle_bare_us", 0.0)
            v = _idle_verdict(idle, cov, bare)
            tp = p.get("idle_top_pipe")
            tv = p.get("idle_top_us", 0.0)
            tps = f"{tp or '-':>7s}{tv:8.1f}" if tp else f"{'-':>7s}{0.0:8.1f}"
            print(f"   {pid:20s}{k:8s}{idle:9.1f}{cov:12.1f}{bare:13.1f}  {tps}   {v}")
        print("  ★口径: 空闲 = 窗口 − 该流水并集 (含首尾未覆盖段), 与上表'空隙'(只累计段间)不同;")
        print("     只统计**可判读**的流水; '整核都闲' 已扣除同核其它可判读流水的忙区。")
        print("     '最大占位' = 单个其它流水吃掉的空闲 (各项可重叠, 故和 > 别人在忙us)。")
    for w in r["warnings"]:
        print(f"\n★ {w}")
    #  ★原来这里写死 `pid.endswith("cubecore")` 与流水名 "CUBE" —— 两个都是 950PR 的格式。
    #    改成按侧别 + "cube 侧最长的那条流水"取, 换平台不必再改。 
    def _cube_busy(c):
        for k, p in c["pipes"].items():
            if str(k).upper() in _CUBE_PIPES:
                return p["busy_us"]
        return None
    cube = [(pid, c) for pid, c in r["cores"].items()
            if r.get("side_of", {}).get(pid) == "aic" and _cube_busy(c) is not None]
    if cube:
        w = [c["window_us"] for _, c in cube]
        b = [_cube_busy(c) for _, c in cube]
        print(f"\n核间: 窗口极差 {max(w) - min(w):.2f}us ({(max(w) - min(w)) / max(w) * 100:.1f}%), "
              f"CUBE 忙时极差 {max(b) - min(b):.2f}us ({(max(b) - min(b)) / max(b) * 100:.1f}%) "
              f"—— 极差大 = 分核不均, 端到端由最慢的核定。")
        _ck = lambda c: next(k for k in c["pipes"] if str(k).upper() in _CUBE_PIPES)
        ng = sum(c["pipes"][_ck(c)]["n_gaps"] for _, c in cube)
        idl = sum(c["pipes"][_ck(c)]["idle_us"] for _, c in cube)
        avg = (idl / ng * 1000.0) if ng else 0.0
        kind = "每指令级开销 (没有单点可修)" if avg < 100 else "跨核握手 (少而长 -> 找是哪一级循环里的同步)"
        print(f"判读二 (空隙**形态**, 见 --refs P8): CUBE 平均 {avg:.0f}ns/段 -> 看起来像**{kind}**。")
        print(f"  段数 {int(ng / len(cube))} 段/核。拿它去除任务数或块数, 比值 ~1 就说明"
              f"'每任务一次'或'每块一次', 直接指向那一级的同步点。")
        print("  ★形态比总量指路: 总量只说'慢了多少', 形态说'慢在哪一级'。")
    print(f"\n★★卡点判词: {_timeline_verdict(r)}")



def selftest_timeline() -> int:
    """PipeTimeline 判读的负例自检。抓的是**按核分开算**这一条 ——
    把多核事件混在一条时间轴上, 各核的忙区互相填补, 空隙会被掩盖成接近 0,
    忙时占比还会算出 >100%。2026-09-14 实撞: 4~7% 的真实空隙被算成 0.9%。"""
    import json as _json, tempfile, shutil
    bad = 0
    print("=== perf PipeTimeline 判读自检 ===")
    #  造两个核, 各自 CUBE 忙 2 段 x 1us, 中间空 1us; 两核的空隙**错开**,
    #  于是混算时一个核的忙区正好盖住另一个核的空隙。
    ev = []
    for core, off in (("core0.cubecore", 0.0), ("core1.cubecore", 1.0)):
        for k in range(2):
            ev.append({"ph": "X", "pid": core, "tid": "CUBE", "name": "CUBE",
                       "ts": off + k * 2.0, "dur": 1.0})
    d = tempfile.mkdtemp()
    #  互补性样本单独一个目录: timeline_report 取 glob 到的第一份 trace.json,
    #  同一目录里再放一份会互相覆盖, 自检就成了掷骰子。
    d2 = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "sub"), exist_ok=True)
        with open(os.path.join(d, "sub", "trace.json"), "w") as f:
            _json.dump({"traceEvents": ev}, f)
        r = timeline_report(d)
        cases = []
        #  每核: 窗口 3us, 忙 2us (66.7%), 空隙 1 段 1us
        for pid in ("core0.cubecore", "core1.cubecore"):
            #  ★不用 r["cores"][pid] 直接下标: 实现若忘了按 pid 分核, 这里会 KeyError 崩掉,
            #    而"崩掉"作为自检信号比"报不过"差 —— 看不出是哪条判据失守。
            c = r["cores"].get(pid, {}).get("pipes", {}).get("CUBE")
            if c is None:
                cases.append((f"{pid} 在结果里缺失 (实现没有按 pid 分核?)", False))
                continue
            cases.append((f"{pid} 忙时=2us", abs(c["busy_us"] - 2.0) < 1e-6))
            cases.append((f"{pid} 空隙=1us (混算会变成 0)", abs(c["idle_us"] - 1.0) < 1e-6))
            cases.append((f"{pid} 占比<=100% (混算会 >100%)", c["occ"] <= 1.0 + 1e-9))
        for name, ok in cases:
            print(f"  {'过' if ok else '★未过'} {name}")
            if not ok:
                bad += 1

        #  ---- 卡点归因的负例 (2026-10-21 立, 全部取自 SMLA 实撞) ----------------------
        #  场景: core0 = cube 有 40% 空闲, vector 侧有一条**单块 SCALAR** 盖满全程。
        #  ① 单块事件必须被标成不可判读; ② 它不许进并集 (进了就把 AIV 顶成 100%,
        #  于是"谁定墙钟"永远答成 AIV); ③ cube 的空闲要能被 AIV 的忙区覆盖上。
        ev2 = [
            #  AIC: 两段各 2us, 中间空 (2..8) 共 6us
            {"ph": "X", "pid": "core0.cubecore", "tid": "CUBE", "name": "CUBE",
             "ts": 0.0, "dur": 2.0},
            {"ph": "X", "pid": "core0.cubecore", "tid": "CUBE", "name": "CUBE",
             "ts": 8.0, "dur": 2.0},
            #  AIV: 单块 SCALAR 盖满全程 (不可判读), 真实工作只有 MTE2 的 2..8
            {"ph": "X", "pid": "core0.veccore0", "tid": "SCALAR", "name": "SCALAR",
             "ts": 0.0, "dur": 10.0},
            {"ph": "X", "pid": "core0.veccore0", "tid": "MTE2", "name": "MTE2",
             "ts": 2.0, "dur": 6.0},
            #  同一 pipe 重叠事件: 求和 4us, 并集 3us (占比必须按并集)。
            #  放在**另一个物理核**上, 免得干扰 core0 的耦合样本。
            {"ph": "X", "pid": "core1.veccore1", "tid": "MTE3", "name": "MTE3",
             "ts": 0.0, "dur": 2.0},
            {"ph": "X", "pid": "core1.veccore1", "tid": "MTE3", "name": "MTE3",
             "ts": 1.0, "dur": 2.0},
            {"ph": "X", "pid": "core1.veccore1", "tid": "VECTOR", "name": "VECTOR",
             "ts": 0.0, "dur": 10.0},          # 只为把该 pid 的窗口撑到 10us
        ]
        with open(os.path.join(d, "sub", "trace.json"), "w") as f:
            _json.dump({"traceEvents": ev2}, f)
        r = timeline_report(d, task_duration_us=5.0)
        sc = r["cores"].get("core0.veccore0", {}).get("pipes", {}).get("SCALAR", {})
        ok = sc.get("attributable") is False
        print(f"  {'过' if ok else '★未过'} 负例: 单块事件 (1 段 100%) 判为不可判读 "
              f"(实得 attributable={sc.get('attributable')})")
        bad += 0 if ok else 1
        ok = any("不可判读" in w for w in r["warnings"])
        print(f"  {'过' if ok else '★未过'} 负例: 单块事件必须出声, 不许静默 (warnings={len(r['warnings'])})")
        bad += 0 if ok else 1
        cp = r["coupling"].get("core0", {})
        ok = cp.get("aiv_busy_us") is not None and abs(cp["aiv_busy_us"] - 6.0) < 1e-6
        print(f"  {'过' if ok else '★未过'} 负例: 单块 SCALAR 不进 AIV 并集 (应只算 MTE2 的 6us, "
              f"实得 {cp.get('aiv_busy_us')})")
        bad += 0 if ok else 1
        ok = abs(cp.get("aic_idle_cov_by_aiv", -1) - 6.0) < 1e-6
        print(f"  {'过' if ok else '★未过'} 负例: AIC 空闲 6us 全被 AIV 忙区覆盖 -> 卡点在 AIV "
              f"(实得 {cp.get('aic_idle_cov_by_aiv')})")
        bad += 0 if ok else 1
        ok = "墙钟由 AIV 设定" in _timeline_verdict(r)
        print(f"  {'过' if ok else '★未过'} 负例: 判词指出 AIV 设定墙钟 (实得: {_timeline_verdict(r)[:28]}...)")
        bad += 0 if ok else 1
        m3 = r["cores"]["core1.veccore1"]["pipes"]["MTE3"]
        ok = abs(m3["busy_us"] - 4.0) < 1e-6 and abs(m3["union_us"] - 3.0) < 1e-6
        print(f"  {'过' if ok else '★未过'} 负例: 同一 pipe 事件重叠时 busy(求和)=4us 而 "
              f"union=3us (实得 {m3['busy_us']}/{m3['union_us']})")
        bad += 0 if ok else 1
        ok = abs(m3["occ"] - 0.3) < 1e-9
        print(f"  {'过' if ok else '★未过'} 负例: 占比按并集算 (3/10=0.3, 按求和会得 0.4) "
              f"(实得 {m3['occ']})")
        bad += 0 if ok else 1
        ok = r.get("inflation") is not None and abs(r["inflation"] - r["window_us"] / 5.0) < 1e-9
        print(f"  {'过' if ok else '★未过'} 正例: 探针放大倍数 = 窗口/未插桩时长 "
              f"(实得 {r.get('inflation')})")
        bad += 0 if ok else 1

        #  ---- 覆盖范围: trace 只有一部分核时必须出声 (2026-10-21 SMLA 实撞) ----------
        #  负例先立: 没有 OpBasicInfo.csv 就**不许猜** —— 报不出 Block Dim 时保持沉默,
        #  而不是把"没读到"写成"全片都覆盖了"。
        ok = not any("只覆盖" in w for w in r["warnings"]) and \
            (r.get("coverage") or {}).get("block_dim") is None
        print(f"  {'过' if ok else '★未过'} 负例: 读不到 Block Dim 时不外推覆盖范围 "
              f"(实得 coverage={r.get('coverage')})")
        bad += 0 if ok else 1
        with open(os.path.join(d, "OpBasicInfo.csv"), "w") as f:
            f.write("Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim\n")
            f.write("kernel_x,mix,100.0,28,56\n")
        r2 = timeline_report(d, task_duration_us=100.0)
        ok = any("只覆盖 2/28" in w for w in r2["warnings"]) and \
            r2["coverage"]["traced_cores"] == 2 and r2["coverage"]["block_dim"] == 28
        print(f"  {'过' if ok else '★未过'} 正例: Block Dim=28 而 trace 只有 2 核 -> 报"
              f"「只覆盖 2/28」且标成样本 (实得 coverage={r2.get('coverage')})")
        bad += 0 if ok else 1

        #  ---- 同核流水互补性 (2026-10-21 SMLA 实撞后立) ------------------------------
        #  要分开的两件事, 用**同一条读数**答:
        #    core0: MTE2 忙 [0,4), VEC 忙 [4,10) —— 首尾相接, 谁闲的时候对方都在忙。
        #           -> 这是**串行**, MTE2 不是饱和, 手段是交错而不是减字节。
        #    core1: 两条都只忙 [0,2), 之后整核全闲 —— 空闲落在**核外**。
        #           -> 此时优化它自己没用, 该去查跨核握手/上游。
        ev3 = [
            {"ph": "X", "pid": "core0.veccore0", "tid": "MTE2", "name": "MTE2",
             "ts": 0.0, "dur": 4.0},
            {"ph": "X", "pid": "core0.veccore0", "tid": "VEC", "name": "VEC",
             "ts": 4.0, "dur": 6.0},
            #  core1 的 SCALAR 单块只为把窗口撑到 10us, 它本身不可判读、不参与归因
            {"ph": "X", "pid": "core1.veccore1", "tid": "SCALAR", "name": "SCALAR",
             "ts": 0.0, "dur": 10.0},
            {"ph": "X", "pid": "core1.veccore1", "tid": "MTE2", "name": "MTE2",
             "ts": 0.0, "dur": 2.0},
            {"ph": "X", "pid": "core1.veccore1", "tid": "VEC", "name": "VEC",
             "ts": 0.0, "dur": 2.0},
        ]
        with open(os.path.join(d2, "trace.json"), "w") as f:
            _json.dump({"traceEvents": ev3}, f)
        r3 = timeline_report(d2)
        m2 = r3["cores"]["core0.veccore0"]["pipes"]["MTE2"]
        ok = abs(m2.get("idle_covered_us", -1) - 6.0) < 1e-6 and \
            abs(m2.get("idle_bare_us", -1) - 0.0) < 1e-6
        print(f"  {'过' if ok else '★未过'} 正例: 首尾相接时 MTE2 的空闲 6us **全部**落在 VEC 忙区 "
              f"(covered={m2.get('idle_covered_us')}, bare={m2.get('idle_bare_us')})")
        bad += 0 if ok else 1
        ok = "串行段挡住" in _idle_verdict(m2["idle_full_us"], m2["idle_covered_us"],
                                          m2["idle_bare_us"])
        print(f"  {'过' if ok else '★未过'} 正例: 该形态判为**串行**, 不说'减字节' "
              f"(判词: {_idle_verdict(m2['idle_full_us'], m2['idle_covered_us'], m2['idle_bare_us'])[:20]}...)")
        bad += 0 if ok else 1
        n2 = r3["cores"]["core1.veccore1"]["pipes"]["MTE2"]
        ok = abs(n2.get("idle_bare_us", -1) - 8.0) < 1e-6 and \
            abs(n2.get("idle_covered_us", -1) - 0.0) < 1e-6
        print(f"  {'过' if ok else '★未过'} 负例: 整核全闲时 bare=8us, **不许**把另一个同闲的流水"
              f"算成'别人在忙' (covered={n2.get('idle_covered_us')}, bare={n2.get('idle_bare_us')})")
        bad += 0 if ok else 1
        ok = "核外" in _idle_verdict(n2["idle_full_us"], n2["idle_covered_us"], n2["idle_bare_us"])
        print(f"  {'过' if ok else '★未过'} 负例: 该形态判为**等核外**, 不说'优化它自己'")
        bad += 0 if ok else 1
        #  不可判读的流水不许进互补性统计 —— 单块 SCALAR 若被算作"别人在忙",
        #  core1 的空闲会被它整段吃掉, 判词就翻转成"串行"。
        ok = "idle_covered_us" not in r3["cores"]["core1.veccore1"]["pipes"]["SCALAR"]
        print(f"  {'过' if ok else '★未过'} 负例: 不可判读的流水不参与互补性归因 "
              f"(SCALAR 实得 {r3['cores']['core1.veccore1']['pipes']['SCALAR'].get('idle_covered_us')})")
        bad += 0 if ok else 1
        #  并集只说"有人忙", 分项才说是谁。core0 里 MTE2 的空闲**只**被 VEC 盖住,
        #  所以最大占位必须是 VEC 且等于 covered —— 若实现错把 MTE2 自己算进去,
        #  这里会得到 covered=6 但 top=0 (自己不算别人)。
        ok = m2.get("idle_top_pipe") == "VEC" and abs(m2.get("idle_top_us", -1) - 6.0) < 1e-6
        print(f"  {'过' if ok else '★未过'} 正例: 最大占位点名到具体流水 "
              f"(实得 {m2.get('idle_top_pipe')}/{m2.get('idle_top_us')})")
        bad += 0 if ok else 1
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)
    print(f"perf PipeTimeline 判读 自检: 21 条, {bad} 条未过")
    return 1 if bad else 0



def _kernel_name_miss(txt: str, kernel_name: str):
    """抽出被 --kernel-name 过滤掉的核名, 并判请求名是否匹配过。

    抽成函数是为了自检能直接喂真实日志文本 —— 判据不能只在真机跑一遍时才被验证。
    """
    import re as _re
    seen = sorted({m for m in _re.findall(
        r"Kernel\s+([A-Za-z_][A-Za-z0-9_]*?)(?:_[0-9a-f]{16,}|_\d{6,})?\s+skipped:"
        r"\s+not selected via --kernel-name", txt)})
    matched = _re.search(r"Kernel\s+%s" % _re.escape(kernel_name), txt) is not None
    return seen, matched


#  --------------------------------------------------------------------------
#  同步等待归责: 把 wait_id<N> 列接回 kernel 里那一行 CrossCoreWaitFlag
#  --------------------------------------------------------------------------
#  为什么长在 perf 里而不是新模块: 它读的就是 --signals 读的那份产物, 消费者也是同一批人。
#  与 hangdiag 的分工 —— hangdiag 查**跨文件约定对不对** (置位/等待偏移集合不一致 = 会挂死),
#  这里答**对的同步花了多少时间**。前者判生死, 后者判成本, 不重叠。
#
#  ★硬约束: 标志位是**全仓共享的 0~14 编号空间**, 同一个 id 在不同算子里含义不同
#    (gmfr 源码注释原话: "仓内其它算子用到 0~14, 故 7 是安全的空位")。
#    所以**不内置任何 id→语义表**, 必须每次喂 kernel 目录现 grep。内置一张表就等于把
#    "在这个算子里 3 是什么"错当成"3 是什么"。
#  ★证据等级: 在 gmfr 上**先预测后测**成立 (docs/msprof-re/08 §3.5) —— 预测
#    SYNC_RETRY_V2C=7 只在回退路径用且 Wait 在 AIC 侧, 故 aic_..._id7 只应出现在一部分采集里
#    而 aiv_..._id7 应为零; 实测 58/640 份且 aiv 侧最大 0.1us。
#    **这是单算子三点吻合, 不是跨算子验证过的定律。** 换算子对不上时先怀疑本函数。
#  ★2026-10-24 修: 原正则是 `constexpr <type> <name> = <digits> ;`, 要求数字后**紧跟分号**,
#    于是**单语句多声明符**一个都解析不到 —— vc_attention 的
#    `constexpr uint16_t S_READY = 8, P_READY = 9, O_READY = 10, O_USED = 15;`
#    四个标志全落空, --sync-attr 把 663ms 的 id9 报成 "源码里查不到", 并把成因猜成
#    "--kernel 目录没喂全"(目录是全的)。改成先切出 constexpr 语句, 再在声明符列表里逐个取值。
#    声明符必须以 `,` 或语句尾收口, 所以 `constexpr int E = 1 + 2;` 这类表达式初值**不收**
#    (宁可查不到, 不可给错号 —— 给错号会把人引到别的标志上)。
_SYNC_CONST_STMT_RE = re.compile(
    r"constexpr\s+(?:u?int(?:8|16|32|64)_t|unsigned|int)\s+([^;{}]+);")
_SYNC_CONST_DECL_RE = re.compile(r"(\w+)\s*=\s*(\d+)\s*(?=,|$)")
_SYNC_WAIT_RE = re.compile(r"CrossCoreWaitFlag\s*\(\s*([A-Za-z_]\w*|\d+)\s*\)")
_SYNC_SET_RE = re.compile(r"CrossCoreSetFlag\s*<[^>]*>\s*\(\s*([A-Za-z_]\w*|\d+)\s*\)")
_SYNC_COL_RE = re.compile(r"^(aic|aiv)_scalar_wait_id(\d+)_time\(us\)$")


def scan_sync_flags(kernel_dir: str) -> dict:
    """扫 kernel 源码, 返回 {flag_id: {"names":[...], "wait":[(file,line)], "set":[...]}}。

    ★只收**真正被 Set/Wait 用到**的常量: 源码里一个 `constexpr uint64_t X = 3` 若从未出现在
      这两个调用的实参里, 它就不是标志号, 收进来会制造假对应 (自检里有这条负例)。
    """
    srcs = []
    for ext in ("*.h", "*.hpp", "*.cpp", "*.cc"):
        srcs += glob.glob(os.path.join(kernel_dir, "**", ext), recursive=True)
    consts, uses = {}, []
    for p in sorted(set(srcs)):
        try:
            lines = open(p, errors="ignore").read().splitlines()
        except Exception:
            continue
        for i, ln in enumerate(lines, 1):
            for stmt in _SYNC_CONST_STMT_RE.finditer(ln):
                for m in _SYNC_CONST_DECL_RE.finditer(stmt.group(1)):
                    consts.setdefault(m.group(1), int(m.group(2)))
            for m in _SYNC_WAIT_RE.finditer(ln):
                uses.append(("wait", m.group(1), p, i))
            for m in _SYNC_SET_RE.finditer(ln):
                uses.append(("set", m.group(1), p, i))
    out = {}
    for kind, tok, p, i in uses:
        fid = int(tok) if tok.isdigit() else consts.get(tok)
        if fid is None:
            continue
        e = out.setdefault(fid, {"names": [], "wait": [], "set": []})
        if tok not in e["names"]:
            e["names"].append(tok)
        e[kind].append((p, i))
    return out


#  框架自己占用的标志号 (AscendC 的 SyncAll 实现内部用, 与用户 kernel 共用同一组硬件旗标).
#  取自 $ASCEND/aarch64-linux/asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:121-125,
#  dav_c220 / dav_m510 同值; dav_m200 / dav_l311 没有这组常量。
#  同文件 GetffstMsg 把 flagId 掩成 `& 0xf` (:133), 所以标志空间是 **0~15**, 不是 0~14。
#  ★这一条推翻了 gmfr 源码注释里的说法 ("仓内其它算子用到 0~14, 故 7 是安全的空位") ——
#    按那个推理挑 12 会直接撞上框架的 SYNC_AIV_FLAG。7 本身安全, 但理由给错了。
_FRAMEWORK_FLAGS = {11: "SYNC_AIC_FLAG", 12: "SYNC_AIV_FLAG",
                    13: "SYNC_AIC_AIV_FLAG", 14: "SYNC_AIV_ONLY_ALL"}


def reserved_flag_warnings(kernel_dir: str) -> list:
    """kernel 自己用的标志号撞上框架保留号时报出。返回 [(id, 框架名, 本地名列表, 同目录有无 SyncAll)]。

    ★**只报可疑, 不下定论**。真正的碰撞条件是"同一个编译单元既调 SyncAll 又用这个号",
      而头文件最终进哪个编译单元取决于 tilingKey 与 include 图, 本函数不解析这两样。
      已实见反例形态: CANN 自带的 flash_attention_score_grad/arch35 里
      `quant.h` 用 flag 11 但不调 SyncAll, `deter.h` 调 SyncAll 但不用 flag 11 ——
      文件级是分开的, 编译单元级未知。所以这里输出的是"去查 tilingKey", 不是"这里有 bug"。
    """
    flags = scan_sync_flags(kernel_dir)
    has_sync_all = bool(glob.glob(os.path.join(kernel_dir, "**", "*.h"), recursive=True)) and any(
        "SyncAll" in open(p, errors="ignore").read()
        for p in glob.glob(os.path.join(kernel_dir, "**", "*.h"), recursive=True))
    out = []
    for fid, fw in _FRAMEWORK_FLAGS.items():
        e = flags.get(fid)
        if e:
            out.append((fid, fw, e["names"], has_sync_all))
    return out


def sync_attr(prof_dir: str, kernel_dir: str, min_us: float = 0.5) -> dict:
    """同步等待归责: 哪个标志等得最久, 它在源码的哪几行。"""
    flags = scan_sync_flags(kernel_dir)
    sig = extract_signals(prof_dir)
    waits = {}
    for c, v in sig.items():
        m = _SYNC_COL_RE.match(c.strip())
        if m and isinstance(v, dict):
            waits[(m.group(1), int(m.group(2)))] = v
    print(f"\n=== 同步等待归责 @ {prof_dir} ===\n    kernel: {kernel_dir}\n")
    if not waits:
        print("  ★产物里**一个 wait_id 列都没有**。这不等于'没有同步等待' —— msprof 只产出\n"
              "    实际用到的那几列, 也可能是这次采的 metric 档位不含它。先确认档位\n"
              "    (本仓 PipeUtilization 档是有的)。")
        return {"rows": [], "unmapped": [], "unused": sorted(flags)}
    rows, unmapped = [], []
    for (side, fid), v in waits.items():
        us = v.get("mean") or 0.0
        if us < min_us:
            continue
        e = flags.get(fid) or {}
        rows.append({"side": side, "id": fid, "mean_us": us, "max_us": v.get("max"),
                     "names": e.get("names", []), "wait": e.get("wait", []), "set": e.get("set", [])})
        if not e:
            unmapped.append((side, fid, us))
    rows.sort(key=lambda r: -r["mean_us"])
    if not rows:
        print(f"  所有 wait_id 列都小于 {min_us} us —— 这个算子的跨核同步不是瓶颈。")
    for r in rows:
        nm = "/".join(r["names"]) if r["names"] else "**源码里查不到**"
        who = "AIC 在等" if r["side"] == "aic" else "AIV 在等"
        print(f"  {r['side']}_scalar_wait_id{r['id']:<2d} {r['mean_us']:10.1f} us"
              f"  (max核 {r['max_us']:.1f})   {who}   {nm}")
        for p, i in r["wait"]:
            print(f"        WaitFlag  {os.path.basename(p)}:{i}")
        for p, i in r["set"]:
            print(f"        SetFlag   {os.path.basename(p)}:{i}   <- 配对的置位在另一侧")
    if unmapped:
        print("\n  ★产物里有、源码里**查不到**的标志 (不要当噪声忽略):")
        for side, fid, us in unmapped:
            fw = _FRAMEWORK_FLAGS.get(fid)
            if fw:
                print(f"      {side}_scalar_wait_id{fid} = {us:.1f} us —— **是框架自己的** {fw}\n"
                      f"        (AscendC SyncAll 实现内部用, 见 asc/impl/basic_api/dav_3510/\n"
                      f"         kernel_operator_sync_impl.h:121-125)。不是你的同步点, 改 kernel 改不动它;\n"
                      f"         它大说明 SyncAll 本身贵 —— 要减就减屏障次数, 不是减某一次的等待。")
            else:
                print(f"      {side}_scalar_wait_id{fid} = {us:.1f} us —— 源码里查不到, 也不是框架保留号。\n"
                      f"        多半是 --kernel 目录没喂全 (子目录/别的模板分支)。")
    warn = reserved_flag_warnings(kernel_dir)
    if warn:
        print("\n  ★本算子自己用了**框架保留**的标志号 —— 去查 tilingKey, 不要当成已确认的缺陷:")
        for fid, fw, names, has_all in warn:
            print(f"      id{fid} 本地名 {'/'.join(names)}  撞框架的 {fw}"
                  f"   (该目录{'有' if has_all else '没有'} SyncAll 调用)")
        print("      碰撞的真正条件是**同一个编译单元既调 SyncAll 又用这个号**, 而头文件进哪个\n"
              "      编译单元取决于 tilingKey 与 include 图, 本函数不解析这两样 —— 用 core.tkcheck 定。")
    unused = sorted(f for f in flags if f not in {fid for _, fid in waits})
    if unused:
        print(f"\n  源码里用了但本次产物里没出现的标志 id: {unused}")
        print("      多半是那条路径本次没走到 (例: 重试/回退分支未触发)。**不是**采集丢数。")
    print("\n  判读纪律 (两条, 都会让人改错地方):")
    print("    1. 等待量常**大于**端到端增长 (被重叠掉一部分), 所以它是**上界**;")
    print("       削它之前先用 core.diagnose 的 critical_path_hint 判它在不在关键路径上。")
    print("    2. ★上面列的行号是该目录下**全部**调用点, 本函数**不解析 tilingKey** ——")
    print("       同一算子的别的模板分支 (如 a8w4_msd 系列) 本次可能根本没被编译进来。")
    print("       改之前先用 core.tkcheck 确认这次走的是哪个模板, 不要照着没走到的分支改。")
    return {"rows": rows, "unmapped": unmapped, "unused": unused}


def selftest_sync_attr() -> int:
    import tempfile
    cases, bad = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        k = os.path.join(tmp, "op_kernel")
        os.makedirs(k)
        with open(os.path.join(k, "u.h"), "w") as f:
            f.write("constexpr uint64_t SYNC_A2C = 3;\n"
                    "constexpr uint64_t SYNC_C2A = 5;\n"
                    "constexpr uint64_t NEVER_USED = 9;\n")
        with open(os.path.join(k, "m.h"), "w") as f:
            f.write("void a(){ CrossCoreWaitFlag(SYNC_A2C); }\n"
                    "void b(){ CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_A2C); }\n"
                    "void c(){ CrossCoreWaitFlag(SYNC_C2A); }\n"
                    "void d(){ CrossCoreWaitFlag(11); }\n")
        fl = scan_sync_flags(k)
        cases.append(("常量名解析成 id, 只收被用到的 (3/5/11)", sorted(fl) == [3, 5, 11], sorted(fl)))
        cases.append(("★负例: 未被 Set/Wait 用到的 constexpr 不收 (NEVER_USED=9)",
                      9 not in fl, sorted(fl)))
        cases.append(("字面量实参也认 (CrossCoreWaitFlag(11))", 11 in fl, sorted(fl)))
        cases.append(("Wait 与 Set 位置分开记", bool(fl.get(3, {}).get("wait")) and bool(fl[3]["set"]),
                      {kk: len(vv) for kk, vv in fl.get(3, {}).items()}))
        cases.append(("★负例: 只被 Wait 的标志不伪造 Set 位置", fl.get(5, {}).get("set") == [],
                      fl.get(5, {}).get("set")))
        #  ★2026-10-24 新增: 单语句多声明符 (vc_attention 实撞)。
        #    修之前这组 fixture 里四个标志一个都解析不到, --sync-attr 全报 "源码里查不到"。
        k3 = os.path.join(tmp, "k3")
        os.makedirs(k3)
        with open(os.path.join(k3, "t.h"), "w") as f:
            f.write("constexpr uint16_t S_READY = 8, P_READY = 9, O_READY = 10, O_USED = 15;\n"
                    "constexpr uint16_t UNUSED_A = 6, UNUSED_B = 7;\n"
                    "constexpr int EXPR_INIT = 1 + 2;\n")
        with open(os.path.join(k3, "t.cpp"), "w") as f:
            f.write("void w(){ CrossCoreWaitFlag(P_READY); CrossCoreWaitFlag(O_USED); }\n"
                    "void v(){ CrossCoreWaitFlag(S_READY); CrossCoreWaitFlag(O_READY); }\n"
                    "void s(){ CrossCoreSetFlag<2, PIPE_FIX>(S_READY); }\n"
                    "void e(){ CrossCoreWaitFlag(EXPR_INIT); }\n")
        f3 = scan_sync_flags(k3)
        cases.append(("★单语句多声明符: 四个标志全解析 (8/9/10/15)",
                      sorted(f3) == [8, 9, 10, 15], sorted(f3)))
        cases.append(("★负例: 同语句里未被 Set/Wait 用到的不收 (UNUSED_A=6, UNUSED_B=7)",
                      6 not in f3 and 7 not in f3, sorted(f3)))
        cases.append(("★负例: 表达式初值不猜值 (EXPR_INIT = 1 + 2 不得解析成 1)",
                      1 not in f3 and 3 not in f3, sorted(f3)))
        cases.append(("多声明符里 Wait/Set 位置分别记 (S_READY 两边都有)",
                      bool(f3.get(8, {}).get("wait")) and bool(f3.get(8, {}).get("set")),
                      {kk: len(vv) for kk, vv in f3.get(8, {}).items()}))
        empty = os.path.join(tmp, "nothing")
        os.makedirs(empty)
        cases.append(("负例: 空 kernel 目录 -> 空表而不是异常", scan_sync_flags(empty) == {}, None))
        #  框架保留号告警
        #  ★这条原本写成负例("不该报"), 自检当场打脸: 上面的 fixture 里有
        #    `CrossCoreWaitFlag(11)` 而 11 就是框架的 SYNC_AIC_FLAG, 警告**本就该报**。
        #    改成正例后反而多覆盖两件事: 字面量实参也走保留号检查; 目录无 SyncAll 时标 False。
        w0 = reserved_flag_warnings(k)
        cases.append(("正例: 字面量 11 也走保留号检查, 且目录无 SyncAll 时标 False",
                      len(w0) == 1 and w0[0][0] == 11 and w0[0][3] is False, w0))
        k2 = os.path.join(tmp, "k2")
        os.makedirs(k2)
        with open(os.path.join(k2, "x.h"), "w") as f:
            f.write("constexpr uint8_t MY_FLAG = 12;\n"
                    "void f(){ CrossCoreWaitFlag(MY_FLAG); }\n"
                    "void g(){ SyncAll<false>(); }\n")
        w = reserved_flag_warnings(k2)
        cases.append(("正例: 用了框架保留的 12 -> 报出, 且标明该目录有 SyncAll",
                      len(w) == 1 and w[0][0] == 12 and w[0][1] == "SYNC_AIV_FLAG" and w[0][3] is True,
                      w))
    print("\n=== perf 同步等待归责 自检 ===")
    for name, ok, got in cases:
        if not ok:
            bad += 1
        print(f"  {'过' if ok else '**未过**'} {name}" + ("" if ok else f"  (实得 {got})"))
    print(f"perf 同步等待归责 自检: {len(cases)} 条, {bad} 条未过")
    return 1 if bad else 0


def selftest_corrupt_launch() -> int:
    """`_corrupt_launches` 的样本外检验 —— 两个正例两个负例, 都是合成目录。

    ★本仓的教训是"判据要样本外检验": 只在撞到的那一份上过, 等于没验。所以这里既要
      抓得住真坏 (含**坏在非 0 位**的那种, 位置判据会漏), 也不能误伤正常的 launch 0。
    """
    import tempfile
    cases, bad = [], 0

    def _mk(root, idx, td, aic, aiv):
        d = os.path.join(root, str(idx))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "OpBasicInfo_1.csv"), "w") as f:
            f.write("Op Name,Op Type,Task Duration(us),Block Dim\n")
            f.write(f"op,mix,{td},24\n")
        with open(os.path.join(d, "PipeUtilization_1.csv"), "w") as f:
            f.write("block_id,sub_block_id,aic_time(us),aiv_time(us)\n")
            f.write(f"0,cube0,{aic},{aiv}\n")
        return d

    with tempfile.TemporaryDirectory() as tmp:
        # 正例 1: 坏在 launch 0 (本次实撞的形态)
        r1 = os.path.join(tmp, "r1")
        _mk(r1, 0, 4498.0, 4300.0, 25203.4)
        _mk(r1, 1, 4500.0, 4280.0, 4503.0)
        got = {os.path.basename(d) for d, _ in _corrupt_launches(r1)}
        cases.append(("正例: 坏在 launch 0 -> 抓住且只抓它", got == {"0"}, got))
        # 正例 2: 坏在 launch 2 —— 位置判据("丢第一个")会**漏掉**这一例
        r2 = os.path.join(tmp, "r2")
        for i, (td, aic, aiv) in enumerate([(100.0, 95.0, 96.0), (100.0, 94.0, 97.0),
                                            (100.0, 93.0, 560.0)]):
            _mk(r2, i, td, aic, aiv)
        got = {os.path.basename(d) for d, _ in _corrupt_launches(r2)}
        cases.append(("正例: 坏在 launch 2 (位置判据会漏) -> 抓住", got == {"2"}, got))
        # 负例 1: 全部正常 (含一个活跃时长贴近窗口的 launch 0)
        r3 = os.path.join(tmp, "r3")
        _mk(r3, 0, 100.0, 99.9, 99.95)
        _mk(r3, 1, 100.0, 98.0, 99.0)
        got = [d for d, _ in _corrupt_launches(r3)]
        cases.append(("负例: 活跃时长贴近窗口但未超 -> 不误伤", got == [], got))
        # 负例 2: 恰在 5% 余量之内 (量化误差区, 不许报)
        r4 = os.path.join(tmp, "r4")
        _mk(r4, 0, 100.0, 100.0, 104.0)
        got = [d for d, _ in _corrupt_launches(r4)]
        cases.append(("负例: 超出 4% 在余量内 -> 不报 (F2 反解有量化误差)", got == [], got))

    print("\n=== perf 损坏 launch 判废 自检 ===")
    for name, ok, got in cases:
        if not ok:
            bad += 1
        print(f"  {'过' if ok else '**未过**'} {name}" + ("" if ok else f"  (实得 {got})"))
    print(f"perf 损坏 launch 判废 自检: {len(cases)} 条, {bad} 条未过")
    return 1 if bad else 0


def selftest_kernel_name() -> int:
    """--kernel-name 没匹配上 vs launch-count/skip 不够 —— 两种'采不到'必须分得清。

    2026-09-17 实撞: 传了源码目录名 `sparse_flash_attention_grad` (小写下划线),
    而真实核名是 `SparseFlashAttentionGrad_ecdedf..._mix_aic` (PascalCase)。两趟采集全废,
    msprof 最后一句是 `No profiling data dumped` —— 与 skip 过大时**一模一样**,
    于是工具把人指向了调 --skip/--launch-count, 方向完全错。
    """
    bad = 0
    print("=== perf --kernel-name 归因 自检 ===")
    #  负例: 真实日志的形态 (目标核也在被过滤之列 —— 名字写对就能采到)
    LOG_MISS = (
        "[INFO]  Kernel Cast_15ccf3aee15572ed7572778d4afbef60_high_performance_210010000"
        " skipped: not selected via --kernel-name.\n"
        "[INFO]  Kernel SparseFlashAttentionGrad_ecdedf297118bbcd87ce0445eedfc881_111101_mix_aic"
        " skipped: not selected via --kernel-name.\n"
        "[WARN]  No profiling data dumped.\n")
    seen, matched = _kernel_name_miss(LOG_MISS, "sparse_flash_attention_grad")
    ok = (not matched) and any(x.startswith("SparseFlashAttentionGrad") for x in seen)
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 负例: 过滤名没匹配上 -> 判为没匹配, 且把真实核名列出来 "
          f"(matched={matched}, 列出 {seen})")
    #  正例一: 名字写对时不得报"没匹配上" —— 否则这条判据会盖住真正的 skip 问题
    seen2, matched2 = _kernel_name_miss(LOG_MISS, "SparseFlashAttentionGrad")
    ok = matched2
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 正例: 名字写对 -> 判为匹配过 (matched={matched2})")
    #  正例二: 日志里根本没有 skipped 行 (skip 过大那一类) -> 不得归因到过滤名
    LOG_SKIP = "[INFO]  All task success.\n[WARN]  No profiling data dumped.\n"
    seen3, matched3 = _kernel_name_miss(LOG_SKIP, "SparseFlashAttentionGrad")
    ok = (not seen3)
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 正例: 没有 skipped 行 (skip 过大那一类) -> 不归因到过滤名 "
          f"(列出 {seen3})")
    print(f"perf --kernel-name 归因 自检: 3 条, {bad} 条未过")
    return 1 if bad else 0


# ------------------------------------------------------------ ExternalCorrelation
# **自定义归责单元** —— 把"这个内核属于我的哪个阶段/候选/迭代"变成下发方声明的事实。
#
# ## 解决什么问题, 不解决什么问题
#
# **解决**: msprof 的归责单元是它自己定的 (内核 / API / 算子)。要问"我这段自定义区间
# 花了多少", 只能事后按时间窗去切它的产物 —— 而 stream 异步下 host 侧时间窗与 device 侧
# 执行窗口并不重合, 切出来的归属是猜的。MSPTI 的
# `msptiActivityPushExternalCorrelationId` 把归责单元的定义权交给调用方。
#
# **不解决**: 不给 PMU / per-pipe / cycle —— MSPTI 头文件里零这类字段。
# 要"这个内核内部慢在哪条流水"仍然走 `perf --signals` (msprof op)。两者互补。
#
# ## 本仓为什么需要它 (2026-10-20 沉淀, 来自 msprof 逆向第 4~6 轮)
#
# `ablate` 做多候选对比时, 候选编号目前靠**目录名**事后拼。而"分组键是目录而被测对象是
# 核函数"是本仓已经吃过亏的一类 (见 report §3.2 聚合掩盖结构)。有了归责轴之后,
# 候选编号可以**在下发时声明**, 不再依赖目录。
#
# ## 三条用真机代价换来的判据, 全部做成默认行为而不是文档提醒
#
# E1 **末单元哨兵**: 最后一批 activity 记录**需要后续活动才会被投递**。
#    `msptiActivityFlushAll` 不够, `torch.npu.synchronize()` 不够,
#    flush x3 + 每次沉降 1 s 也不够 —— 但随便再下发一点东西就来了。
#    不加这一步**最后一个归责单元会静默为空**, 而它看起来和"这个单元确实没有内核"一样。
#    所以 `close()` 默认跑哨兵; 关掉它要显式传 `sentinel=0` 并自担。
#    (实证: 九单元实验里末位恒为 0; 倒序后换成另一个算子仍为 0; 加哨兵后九个全齐。)
#
# E2 **结构必须 `_pack_ = 1`**: 记录带 `__attribute__((packed))` + `aligned(8)`
#    (`mspti_activity.h:21,28`)。不设则 `ds{uint32,uint32}` 之后的 `uint64` 错位 4 字节,
#    读出来的 `correlationId` 是**垃圾, 但长得像一个合法的大整数** —— join 全空而不报错。
#
# E3 **回调对象必须被持有**: `CFUNCTYPE` 包出来的对象被 GC 之后 MSPTI 再回调是野指针。
#
# 另记一条判据 (E4): `msptiActivityEnable` 返回 `MSPTI_SUCCESS` **不代表采集起来了** ——
# 少调 `msptiSubscribe` 时七个 kind 全返回 0 而只有 ExternalCorrelation 记录会来,
# Kernel 一条都没有。判"采集真起来了"要看收到没收到该 kind 的记录, 不看返回码。

_MSPTI_SUCCESS = 0
_MSPTI_KIND_KERNEL = 2
_MSPTI_KIND_EXTCORR = 8
_EXT_KIND_BASE = 2                      # CUSTOM0=2 / CUSTOM1=3 / CUSTOM2=4


class _ExtDS(ctypes.Structure):
    _pack_ = 1                          # E2
    _fields_ = [("deviceId", ctypes.c_uint32), ("streamId", ctypes.c_uint32)]


class _ExtKernelRec(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("kind", ctypes.c_int), ("start", ctypes.c_uint64), ("end", ctypes.c_uint64),
                ("ds", _ExtDS), ("correlationId", ctypes.c_uint64),
                ("type", ctypes.c_char_p), ("name", ctypes.c_char_p)]


class _ExtCorrRec(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("kind", ctypes.c_int), ("externalKind", ctypes.c_int),
                ("externalId", ctypes.c_uint64), ("correlationId", ctypes.c_uint64)]


def _extcorr_lib_path() -> str | None:
    root = os.environ.get("ASCEND_HOME_PATH") or ""
    for cand in (os.path.join(root, "tools/mspti/lib64/libmspti.so"),
                 os.path.join(root, "lib64/libmspti.so")):
        if cand and os.path.exists(cand):
            return cand
    return None


class ExtCorrSession:
    """按**调用方定义的单元**归责内核。

    用法::

        with perf.ExtCorrSession() as s:
            with s.unit("候选A"):
                ...下发...
            with s.unit("候选B"):
                ...下发...
        print(s.report())          # 每个单元: 内核数 + 合计 device 时间

    `unit` 可嵌套 (push/pop 是真栈: 内层区间挂内层 id, 退出后回到外层), 也可以用
    `axis=1/2` 开第二、第三条轴 —— 三条轴能**同时**挂在同一个内核上, 一次跑回答
    "属于哪个阶段 / 第几次迭代 / 哪个候选"。
    """

    def __init__(self, sentinel: int = 8, settle: float = 0.5, buf_bytes: int = 4 << 20,
                 sync_on_exit: bool = True):
        self.sentinel, self.settle, self.buf_bytes = sentinel, settle, buf_bytes
        self.sync_on_exit = sync_on_exit
        self.lib = None
        self._keep, self._bufs = [], []     # E3
        self.records, self._names = [], {}
        self._next_id = 1000

    # -- 拒跑判据 (与 cases/*/repro.py 同口径: 看 ASCEND_RT_VISIBLE_DEVICES 首项) --
    @staticmethod
    def preflight() -> str | None:
        raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if not items:
            return "未设 ASCEND_RT_VISIBLE_DEVICES —— 默认会落到物理 0 卡 (CI runner 专用)"
        if items[0] == "0":
            return f"ASCEND_RT_VISIBLE_DEVICES 首项是物理 0 卡 (CI runner 专用): {raw!r}"
        if "libmspti.so" not in os.environ.get("LD_PRELOAD", ""):
            return ("未 LD_PRELOAD libmspti.so —— 采集不会启动。"
                    "注意它与 msprof 互斥, 见 strip_mspti_preload")
        if _extcorr_lib_path() is None:
            return "找不到 libmspti.so (试过 $ASCEND_HOME_PATH 下 tools/mspti/lib64 与 lib64)"
        return None

    def __enter__(self):
        why = self.preflight()
        if why:
            raise RuntimeError(f"ExtCorrSession 拒跑: {why}")
        self.lib = ctypes.CDLL(_extcorr_lib_path(), mode=ctypes.RTLD_GLOBAL)

        req_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                                 ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        cpl_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8),
                                 ctypes.c_size_t, ctypes.c_size_t)
        cbf_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.c_uint32, ctypes.c_void_p)

        def on_request(buffer, size, max_records):
            b = (ctypes.c_uint8 * self.buf_bytes)()
            self._bufs.append(b)
            buffer[0] = ctypes.cast(b, ctypes.POINTER(ctypes.c_uint8))
            size[0] = self.buf_bytes
            max_records[0] = 0

        def on_complete(buffer, size, valid):
            if valid:
                self._drain(buffer, valid)

        cbs = (req_t(on_request), cpl_t(on_complete), cbf_t(lambda *_: None))
        self._keep.extend(cbs)                                  # E3
        sub = ctypes.c_void_p()
        self.lib.msptiSubscribe(ctypes.byref(sub), cbs[2], None)   # E4: 少了它只有一半产物
        self._keep.append(sub)
        self.lib.msptiActivityRegisterCallbacks(cbs[0], cbs[1])
        for k in (_MSPTI_KIND_EXTCORR, _MSPTI_KIND_KERNEL):
            _re = self.lib.msptiActivityEnable(ctypes.c_int(k))
            if _re != _MSPTI_SUCCESS:
                print(f"[warn] ActivityEnable(kind={k}) rc={_re}")
        return self

    def _drain(self, buf_ptr, valid):
        rec = ctypes.POINTER(ctypes.c_int)()
        while True:
            r = self.lib.msptiActivityGetNextRecord(buf_ptr, ctypes.c_size_t(valid),
                                                    ctypes.byref(rec))
            if r != _MSPTI_SUCCESS:
                break
            addr = ctypes.cast(rec, ctypes.c_void_p).value
            k = ctypes.c_int.from_address(addr).value
            if k == _MSPTI_KIND_EXTCORR:
                e = _ExtCorrRec.from_address(addr)
                self.records.append(("x", e.externalKind, e.externalId, e.correlationId, 0, ""))
            elif k == _MSPTI_KIND_KERNEL:
                kr = _ExtKernelRec.from_address(addr)
                self.records.append(("k", 0, 0, kr.correlationId, kr.end - kr.start,
                                     (kr.name or b"").decode("utf-8", "replace")))

    @contextlib.contextmanager
    def unit(self, name: str, axis: int = 0):
        """一个归责单元。可嵌套 (真栈), `axis` 0/1/2 对应 CUSTOM0/1/2 三条独立轴。"""
        self._next_id += 1
        uid = self._next_id
        self._names[(axis, uid)] = name
        kind = ctypes.c_int(_EXT_KIND_BASE + axis)
        _rc = self.lib.msptiActivityPushExternalCorrelationId(kind, ctypes.c_uint64(uid))
        if _rc != _MSPTI_SUCCESS:
            print(f"[warn] push 外部关联 id 失败: kind={kind.value} id={uid} rc={_rc} "
                  f"—— 该单元不会有任何归属")
        try:
            yield uid
        finally:
            if self.sync_on_exit:
                # E5: pop 之前必须同步。不同步则**还在飞的下发丢归属** —— 实测 8 次下发
                # 只归到 2 个, 而且丢的那些会落进 __unattributed__ 而不是报错。
                try:
                    import torch
                    torch.npu.synchronize()
                except Exception:  # noqa: BLE001
                    pass
            last = ctypes.c_uint64(0)
            self.lib.msptiActivityPopExternalCorrelationId(kind, ctypes.byref(last))

    def close(self, sentinel_fn=None):
        """E1: 默认跑一段哨兵负载再 flush, 否则**最后一个单元会静默为空**。"""
        if self.sentinel:
            fn = sentinel_fn or self._default_sentinel
            try:
                fn(self.sentinel)
            except Exception as ex:                             # noqa: BLE001
                print(f"[warn] 哨兵负载没跑成 ({ex}) —— **最后一个单元可能为空**, "
                      f"见 core/perf 的判据 E1")
        import time
        for _ in range(3):
            self.lib.msptiActivityFlushAll(ctypes.c_uint32(1))
            time.sleep(self.settle)

    @staticmethod
    def _default_sentinel(n: int):
        import torch
        t = torch.randn(256, 256, device="npu", dtype=torch.float16)
        for _ in range(n):
            t2 = torch.mm(t, t)
            del t2
        torch.npu.synchronize()

    def __exit__(self, *exc):
        self.close()
        return False

    def report(self) -> dict:
        """{单元名: {"kernels": n, "ns": 合计, "us": 合计/1000}}。未归属的进 `__unattributed__`。"""
        corr2unit = {}
        for t, ek, eid, corr, _d, _n in self.records:
            if t == "x":
                corr2unit.setdefault(corr, {})[ek - _EXT_KIND_BASE] = eid
        out = {}
        for t, _ek, _eid, corr, dur, _nm in self.records:
            if t != "k":
                continue
            axes = corr2unit.get(corr)
            if not axes:
                key = "__unattributed__"
            else:
                key = " | ".join(self._names.get((ax, uid), f"axis{ax}:{uid}")
                                 for ax, uid in sorted(axes.items()))
            e = out.setdefault(key, {"kernels": 0, "ns": 0})
            e["kernels"] += 1
            e["ns"] += dur
        for e in out.values():
            e["us"] = round(e["ns"] / 1000.0, 2)
        return out


def selftest_extcorr() -> int:
    """判据负例自检 (不占卡: 只验结构布局、拒跑判据与归并逻辑)。"""
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    # E2: packed 布局。不设 _pack_=1 时 correlationId 会后移 4 字节。
    ck("E2 Kernel 记录按 packed 算 offset: correlationId 在 28",
       _ExtKernelRec.correlationId.offset, 28)
    ck("E2 ExternalCorrelation 记录: externalId 在 8, correlationId 在 16",
       (_ExtCorrRec.externalId.offset, _ExtCorrRec.correlationId.offset), (8, 16))

    class _Unpacked(ctypes.Structure):          # 对照: 不加 _pack_ 的同一组字段
        _fields_ = _ExtKernelRec._fields_
    ck("E2 ★不加 _pack_ 时 offset 确实会漂 (所以这条判据必须存在)",
       _Unpacked.correlationId.offset != _ExtKernelRec.correlationId.offset, True)

    # 拒跑判据
    import os as _os
    keep = (_os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), _os.environ.get("LD_PRELOAD"))
    try:
        _os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,5"
        ck("物理 0 卡打头必须拒跑 (判据看首项, 不看是否出现)",
           "物理 0 卡" in (ExtCorrSession.preflight() or ""), True)
        _os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "5,0"
        _os.environ["LD_PRELOAD"] = ""
        ck("非 0 卡打头 + 缺 LD_PRELOAD -> 报的是 LD_PRELOAD 不是 0 卡 (两种毛病不混)",
           "LD_PRELOAD" in (ExtCorrSession.preflight() or ""), True)
        _os.environ.pop("ASCEND_RT_VISIBLE_DEVICES")
        ck("未设可见卡也必须拒跑 (不设 = 落物理 0 卡)",
           "未设" in (ExtCorrSession.preflight() or ""), True)
    finally:
        for k, v in zip(("ASCEND_RT_VISIBLE_DEVICES", "LD_PRELOAD"), keep):
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v

    # E1: 哨兵默认开
    ck("E1 哨兵默认开 (关掉要显式传 0)", ExtCorrSession().sentinel > 0, True)
    ck("E5 pop 前同步默认开 (关掉则在飞的下发静默丢归属)", ExtCorrSession().sync_on_exit, True)

    # 归并逻辑: 多轴 join 与未归属
    s = ExtCorrSession()
    s._names = {(0, 1001): "阶段A", (1, 2001): "候选X"}
    s.records = [("x", 2, 1001, 77, 0, ""), ("x", 3, 2001, 77, 0, ""),
                 ("k", 0, 0, 77, 5000, "MatMul"), ("k", 0, 0, 999, 300, "Cast")]
    rep = s.report()
    ck("多轴 join: 同一内核挂两条轴时键是两段拼起来的",
       sorted(rep), ["__unattributed__", "阶段A | 候选X"])
    ck("归属到的内核时间按 ns 聚合并换算 us", rep["阶段A | 候选X"]["us"], 5.0)
    ck("★区间外的内核必须单独记 __unattributed__, 不许摊进某个单元",
       rep["__unattributed__"]["kernels"], 1)

    print(f"ExternalCorrelation 自检: 11 条, {bad} 条未过")
    return 1 if bad else 0


# ------------------------------------------------------------------ B 路径体检
# B 路径 (torch_npu profiler / `msprof --export` 的 kernel_details.csv) 与本模块平时读的
# A 路径 (msopprof 的 OPPROF csv) 是**两套互不相同的实现** (见 docs/msprof-re/06 V9)。
# 本仓的性能判读走 A 路径, 但 cases/*/runs 下确实存着 B 路径产物, 有人会从里面读数 ——
# 而 04 篇逆向出 28 条失真点, 说明那些数字不能直接当真值用。
#
# 这里只收**能从 csv 直接判**的那一部分。判不了的显式报"查不了", 不装作通过:
# 一个把"没数据"说成"没问题"的体检, 比没有体检更糟。

# ★核类型是本族产物里反复出现的混淆因子, 到 2026-10-22 已第 4 次成为解释:
#     1. `08` §3.47  L0C_to_L1 的 71808 个 NA 是 AIV 行 (L0C 是 cube 侧缓冲);
#     2. B14         aicore_time=0 的 390 行全是 AIV 核 (纯 AIV 核的 AIC 时间本就为 0);
#     3. B15         aicore_time > Task Duration 的 2908 行全是 AIV 核;
#     4. B7          恒等式的**除数随核类型变**: AIV 行要再除以 2。
#   按成熟度阶梯同类 3 次升工具, 这里抽成公共件而不是第 4 次硬编。
def _bp_core_kind(row: dict) -> str:
    """判一行属 AIC 还是 AIV。返回 "aic" / "aiv" / "" (判不出)。

    两条线索, 任一成立即判 AIV: `Task Type`/`Accelerator Core` 写明 vector,
    或 `aiv_time(us)` 有正值而 `aic_time(us)` 没有。
    **判不出时返回空串而不是默认 aic** —— 默认会把判不出的行悄悄算进某一类。
    """
    # ★用 `GeTaskType` 那张**权威枚举**, 不用子串匹配。
    #   枚举见 `03` §原始格式 (0 AI_CORE / 1 AI_CPU / 2 AI_VECTOR_CORE / 3 WRITE_BACK /
    #   4 MIX_AIC / 5 MIX_AIV / 6 FFTS_PLUS / 7 DSA_SQE / 8 DVPP / 9 HCCL / 10 FUSION)。
    #   ★2026-10-22 实撞: 原来写的是 `if "VECTOR" in tag`, 而 **`MIX_AIV` 里没有 `VECTOR`**
    #   —— 它会穿过这一条掉进下面的时间列回退, 在缺 `aic_time` 列的产物上判成 `aic`。
    #   同一天我在一段一次性脚本里犯了同一个错, 整条归因链跟着错了两步 (总账第 57 条)。
    #   **子串匹配看着稳, 实际是在赌命名约定; 枚举是写下来的事实。**
    #   非计算核的类型 (AI_CPU / WRITE_BACK / DSA_SQE / DVPP / HCCL / FUSION) 显式返回空串,
    #   它们既不是 AIC 也不是 AIV, 混进任一侧都会污染该侧的统计。
    tag = ((row.get("Task Type") or row.get("Accelerator Core") or "")).strip().upper()
    if tag in ("AI_VECTOR_CORE", "MIX_AIV", "AIV"):
        return "aiv"
    if tag in ("AI_CORE", "MIX_AIC", "AIC"):
        return "aic"
    if tag in ("AI_CPU", "WRITE_BACK", "FFTS_PLUS", "DSA_SQE", "DVPP", "HCCL", "FUSION"):
        return ""          # 不是计算核, 不归任一侧
    def _n(c):
        try:
            return float((row.get(c) or "").strip())
        except (TypeError, ValueError):
            return None
    av, ac = _n("aiv_time(us)"), _n("aic_time(us)")
    if (av or 0) > 0 and not ac:
        return "aiv"
    if (ac or 0) > 0 and not av:
        return "aic"
    return ""


# AIV 行的核数换算: 910B2 每个 AI Core 配 2 个 AIV。
# ★2026-10-22 实测: 在 36194 个 AIV 行上 `aic_total_cycles/(freq*aicore_time)/Block Num`
#   **恰为 0.5, 零例外** (取值分布只有 0.5 这一个值)。
_BP_AIV_PER_AIC = 2.0

_BP_RATIO_COLS = ("aic_mac_ratio", "aic_mte2_ratio", "aiv_vec_ratio")

# ★D7 的被影响列 -> 它的分母列。2026-10-22 实测发现**两者从不同时出现**:
#   被影响列 (*_main_mem_*_bw) 只在 5 份产物 / 15 行里有, 而那 15 行的分母列
#   (*_mte2_ratio / *_mte3_ratio) **一列都不在**; 反过来有分母列的 41 份产物 / 588 行里
#   没有 bw 列。两组列来自不同的 metric 档位。
#   `icache_miss_rate` 那一支更直接: 它的分母 `icache_req_ratio` **一份产物都不导出**。
#   所以 D7 这条失真在**导出的 csv 上原理上不可观测** —— 不是"还没遇到", 是看不到。
#   判定用单独的 `unobs`, 与 `na` 分开: `na` 是"这份产物没有这些列",
#   `unobs` 是"**任何产物都不会同时有**"。两者处置不同 —— 前者再采一份可能就有了,
#   后者再采多少份都没有, 要换数据源 (原始 PMU 记录, 见 03) 或读公式 lambda。
_BP_D7_PAIRS = (
    ("aic_main_mem_read_bw(GB/s)", "aic_mte2_ratio"),
    ("aic_main_mem_write_bw(GB/s)", "aic_mte2_ratio"),
    ("aiv_main_mem_read_bw(GB/s)", "aiv_mte2_ratio"),
    ("aiv_main_mem_write_bw(GB/s)", "aiv_mte3_ratio"),
    ("aic_icache_miss_rate", "icache_req_ratio"),
    ("aiv_icache_miss_rate", "icache_req_ratio"),
)

# ★极性不一致, 汇总时必须分开: B7/B8 是**恒等式检验** (hit = 恒等式不成立, miss = 成立),
#   B9/B10/B11 是**失真检测** (hit = 失真在场)。2026-10-22 实撞: 语料汇总的第一版用了
#   一句通用判读 "考得动 N 份但一次没命中", 于是把 **72 份上确认成立的恒等式**
#   说成了"未被产物证实" —— 把正面结果报成了未验证。
#   ★2026-10-22 补 B11: 它改标注之后也属这一类 (hit = `*_ratio > 1` 即不变量被破坏,
#   miss = 不变量成立)。此前它被当成失真检测, 于是 41 份上成立的不变量被报成"未命中"。
_BP_IDENTITY = frozenset(("B7", "B8", "B11"))


def _cov(seen: int = 0, **skipped) -> dict:
    """一条判据的**看见范围**: 看了多少行, 按什么理由跳过多少行。

    ★为什么要有它 (2026-10-22, 由 B7 的实撞逼出来): B7 的恒等式在 76 份产物上"成立",
    而那是因为 `if not t: continue` 把 `aicore_time == 0` 的 **AI_VECTOR_CORE 行全跳过了** ——
    反例被判据自己的跳过条件过滤掉, 结论却照常写成"验过"。
    **一个判据的跳过条件和它的判定条件一样决定结论, 而跳过条件通常不写进结论里。**
    所以每条判据都要把它跳过了什么报出来; 跳过远多于看见时, 它的"成立"要降级看待。
    """
    return {"seen": seen, "skipped": {k: v for k, v in skipped.items() if v}}


def bpath_lint(csv_path: str, freq_mhz: float | None = None) -> list:
    """对一份 kernel_details.csv 跑可机检的判据。返回 [(编号, 判定, 说明, 证据, 看见范围)]。

    第 5 项是 `_cov()` 给的 {"seen": 行数, "skipped": {理由: 行数}} —— 见 `_cov` 的注释。

    判定取值: hit (命中失真点) / miss (这份产物没发生) / na (这份产物判不了)。
    **miss 不等于该失真点不存在** —— 只说明这一份产物上没出现。
    """
    import csv as _csv
    out = []
    covs = {}          # 每条判据的"看见范围", 返回时拼成第 5 项 (见 _cov)
    try:
        rows = list(_csv.DictReader(open(csv_path, encoding="utf-8", errors="replace")))
    except OSError as ex:
        return [("B0", "na", "读不出来", str(ex))]
    if not rows:
        return [("B0", "na", "空表", csv_path)]
    cols = set(rows[0].keys())

    def _f(r, c):
        try:
            return float((r.get(c) or "").strip())
        except ValueError:
            return None

    if freq_mhz is None:
        try:
            from core import socspec
            freq_mhz = socspec.spec()["freq_ghz"] * 1000
        except Exception:  # noqa: BLE001
            freq_mhz = None

    # B7/B8: aicore_time 是**按实际占用核数平均**的, 不是各核之和。
    #        恒等式 time(us) = total_cycles / (freq_MHz * Block Num)。
    #        2026-10-20 在本仓 39 份产物 105 行上 **105/105 精确成立**, 含两行 Block Num=20
    #        的非满核情形 —— 正是它排除了"除数是固定核数"这个更简单但错的读法。
    for tag, tcol, ccol, bcol in (("B7", "aicore_time(us)", "aic_total_cycles", "Block Num"),
                                  ("B8", "aiv_time(us)", "aiv_total_cycles", "Mix Block Num")):
        if not {tcol, ccol, bcol} <= cols:
            out.append((tag, "na", f"缺列 {tcol}/{ccol}/{bcol}", "本产物不含这些列"))
            continue
        if freq_mhz is None:
            out.append((tag, "na", "取不到本机频率", "socspec 不可用"))
            continue
        bad = []
        aiv_rows = seen7 = sk_zero = 0
        for i, r in enumerate(rows):
            t, c, b = _f(r, tcol), _f(r, ccol), _f(r, bcol)
            if not t or not c or not b:
                # ★正是这一行在 76 份产物上把 AIV 反例全过滤掉了 (aicore_time 恒为 0),
                #   所以它必须被计数并报出来, 不能只是 `continue`。
                sk_zero += 1
                continue
            seen7 += 1
            # ★除数随核类型变: AIV 行再除以 2 (910B2 每 AIC 配 2 AIV)。
            #   不分核类型时本条在 36194 个 AIV 行上全部报不符, 而它们的比值
            #   **恰为 0.5 零例外** —— 那不是失真, 是除数取错了。
            if tag == "B7" and _bp_core_kind(r) == "aiv":
                b /= _BP_AIV_PER_AIC
                aiv_rows += 1
            if abs(c / t / freq_mhz - b) > 0.01:
                bad.append((i, round(c / t / freq_mhz, 3), b))
        covs[tag] = _cov(seen7, **{f"{tcol}/{ccol}/{bcol} 任一为空或 0": sk_zero})
        if bad:
            out.append((tag, "hit", f"{tcol} 与 {ccol}/(freq*{bcol}) 对不上",
                        f"{len(bad)} 行不符, 例: 行{bad[0][0]} 推出 {bad[0][1]} vs {bcol}={bad[0][2]}"))
        else:
            extra = (f"; 其中 {aiv_rows} 行按 AIV 核数换算 (除 {bcol}/2)" if aiv_rows else "")
            out.append((tag, "miss", f"{tcol} = {ccol}/(freq*{bcol}) 成立",
                        f"{len(rows)} 行全部吻合 —— 该列是**按核平均**不是求和{extra}"))

    # B9: Start Time 列被 format_high_precision_for_csv 加了一个结尾制表符 (04 篇 §8.3)。
    #     它让朴素的字符串比较/排序错位, 而 float() 仍然能解析 —— 所以不报错, 只会悄悄错。
    if "Start Time(us)" in cols:
        n = sum(1 for r in rows if (r.get("Start Time(us)") or "").endswith("\t"))
        covs["B9"] = _cov(len(rows))
        out.append(("B9", "hit" if n else "miss",
                    "Start Time(us) 带**结尾制表符**" if n else "Start Time(us) 无结尾制表符",
                    f"{n}/{len(rows)} 行"))
    else:
        out.append(("B9", "na", "无 Start Time(us) 列", ""))

    # B10 (D21): Duration 与 aicore_time 是两个独立的量, 混用会按下面这个比例错。
    if {"Duration(us)", "aicore_time(us)"} <= cols:
        pr = [(d, a) for r in rows
              if (d := _f(r, "Duration(us)")) and (a := _f(r, "aicore_time(us)")) and d > 0]
        covs["B10"] = _cov(len(pr), **{"两列任一为空或 0": len(rows) - len(pr)})
        rs = [a / d for d, a in pr]
        if rs:
            gaps = [d - a for d, a in pr]
            # ★比值与差值要一起报。2026-10-22 在 173 份产物上实测: 差值在
            #   Duration<100us 时恒为 1.1~2.4us (加性底), 在 Duration>10ms 时是 0.6~2.1ms
            #   (约 7%, 比例性)。于是**同一条失真在不同核长上差一个数量级** ——
            #   5us 的核偏 24%, 93ms 的核偏 1%。只报比值会让人以为那是个常数。
            # ★`miss` 在这一条上**不读作"没有失真"**: 它只说这份产物的核足够长,
            #   相对差落到阈值以下。绝对差照样在 (那一份实测 902us)。
            out.append(("B10", "hit" if min(rs) < 0.99 else "miss",
                        "Duration 与 aicore_time 不是同一个量; 相对差随核长变化, "
                        "**miss 只意味着这份产物的核够长, 不意味着没有差**",
                        f"aicore_time/Duration {min(rs):.4f}~{max(rs):.4f}; "
                        f"绝对差 {min(gaps):.3f}~{max(gaps):.3f}us ({len(rs)} 行)"))
        else:
            out.append(("B10", "na", "两列都在但取不到数", ""))
    else:
        out.append(("B10", "na", "缺 Duration(us) 或 aicore_time(us)", ""))

    # B11: `*_ratio` 这类占比列不得 >1。
    # ★2026-10-22 订正来源: 本条此前标成 "D7", **那是错的** —— D7 影响的是
    #   icache_miss_rate / control_flow_mis_prediction_rate / main_mem_*_bw 三支,
    #   与这里查的 mac/mte2/vec 占比列毫无交集。本条是一条独立成立的不变量,
    #   D7 本身由下面的 B12 处理。
    covs["B11"] = _cov(sum(1 for r in rows
                            for c in _BP_RATIO_COLS if c in cols and _f(r, c) is not None))
    over = [(c, r.get(c)) for r in rows for c in _BP_RATIO_COLS
            if c in cols and (v := _f(r, c)) is not None and v > 1.0]
    if any(c in cols for c in _BP_RATIO_COLS):
        out.append(("B11", "hit" if over else "miss",
                    "有 _ratio 列取值 > 1 (分母为 0 时保留中间值的典型形态)" if over
                    else "全部 _ratio ≤ 1",
                    f"{len(over)} 处" + (f", 例 {over[0][0]}={over[0][1]}" if over else "")))
    else:
        out.append(("B11", "na", "无受检的 _ratio 列", ""))

    # B13 (D23): 设备侧时间戳的硬件分辨率。`Duration(us)` 打到小数点后 3 位 (= 1 ns),
    #   但 hwts 是 50 MHz, 一切 syscnt 差都是 **20 ns 的整数倍** —— 后两位是换算产物不是精度。
    #   对本仓的直接含义: **两次测量相差 0.001 us 没有意义**, 真正的量子是 0.02 us。
    #   2026-10-22 实测 4165/4165 行全部为 0.02us 的整数倍, 零例外。
    if "Duration(us)" in cols:
        ds = [d for r in rows if (d := _f(r, "Duration(us)")) is not None]
        off = [d for d in ds if abs(round(d / 0.02) - d / 0.02) > 1e-6]
        covs["B13"] = _cov(len(ds), **{"Duration(us) 取不到数": len(rows) - len(ds)})
        if ds:
            out.append(("B13", "hit" if not off else "miss",
                        "Duration(us) 全部是 20ns (0.02us) 的整数倍 —— 小数点后两位是换算产物"
                        if not off else
                        "有 Duration(us) 不是 20ns 整数倍 —— 与 hwts 50MHz 的口径不符",
                        f"{len(ds) - len(off)}/{len(ds)} 行是整数倍"))
    else:
        out.append(("B13", "na", "无 Duration(us) 列", ""))

    # B14 (D14): GE 查不到 block_num 时 total_time 直接写 0 而不是 N/A,
    #   于是该算子在求和/排序里被当成"零耗时"。
    #   ★判据必须先分核类型。2026-10-22 实撞: 不分核类型时命中 390 行, 逐行查完
    #   **全部是 AI_VECTOR_CORE 且 aiv_time 有值** —— 纯 AIV 核的 aicore_time 本就该是 0,
    #   那是**正确值不是缺失**。同一个混淆在 `08` §3.47 的 L0C_to_L1 上已经出现过一次
    #   (71808 个 NA 是 AIV 行)。凡按 AIC/AIV 分列的量, 任何"为 0 / 为 NA"的判据
    #   都必须先分核类型, 否则抓到的是核类型不是缺陷。
    if {"aicore_time(us)", "Duration(us)"} <= cols:
        zr = aiv_excl = seen14 = sk14 = 0
        for r in rows:
            a_, d_ = _f(r, "aicore_time(us)"), _f(r, "Duration(us)")
            if a_ is None or d_ is None or d_ <= 0:
                sk14 += 1
                continue
            seen14 += 1
            if a_ != 0:
                continue
            av = _f(r, "aiv_time(us)")
            if (r.get("Accelerator Core") or "").strip() == "AI_VECTOR_CORE" or (av or 0) > 0:
                aiv_excl += 1          # 纯 AIV 核: 正确值, 不是 D14
            else:
                zr += 1
        covs["B14"] = _cov(seen14, **{"两列任一取不到数": sk14,
                                      "非 AIC 核 (AIV 的 aicore_time 本就为 0)": aiv_excl})
        out.append(("B14", "hit" if zr else "miss",
                    "有 AIC 行的 aicore_time 为 0 而 Duration > 0 (GE 缺 block_num 的形态)"
                    if zr else "没有这样的 AIC 行",
                    f"命中 {zr} 行; **另有 {aiv_excl} 行已按核类型排除** (纯 AIV 核的 "
                    f"aicore_time 本就为 0, 是正确值)"))
    else:
        out.append(("B14", "na", "缺 aicore_time(us) 或 Duration(us)", ""))

    # B15: `aicore_time(us)` 不得超过 `Task Duration(us)` —— 核忙时长于任务时长讲不通。
    #   这一条只在 op_summary.csv 上考得动 (kernel_details 没有 `Task Duration(us)` 这个列名)。
    #   ★2026-10-22 实测 118899 行, 按 Task Type 分开之后两个分布**完全不重叠**:
    #     AI_CORE        82700 行  比值 0.5325~0.7832  **超过 1 的占 0.00%**
    #     AI_VECTOR_CORE 36194 行  比值 0.8115~1.0987  **超过 1 的占 8.03%**
    #   即 **AIV 侧的时间换算至少对部分行偏大, 上界约 10%**; AIC 侧一行都没越界。
    #   与 D28 (AIV 的时间换算用 aic_frequency) 相容, 但**不构成证明** ——
    #   本机 info.json 只有一个频率, 分析代码里也不存在 aiv_frequency,
    #   所以"用错频率"这件事在当前实现里无从区分对错: 没有第二个频率可用。
    if {"aicore_time(us)", "Task Duration(us)"} <= cols:
        # ★两个分母都要报。2026-10-22 实撞: 文档里那个 8.03% 用的是**非零行**做分母,
        #   而工具修掉 falsy-zero 之后用的是**全集**(含 aicore_time=0 的行), 同一条判据
        #   两处给 8.03% 与 1.22%。零值行平凡满足这条不变量, 把它们算进分母, 比例就随
        #   "这次采集里有多少纯 AIV 核"浮动, 不再反映失真本身。两个都报, 并写明各自口径。
        n = ov = sk15 = nz = 0
        worst = 0.0
        kinds = {}
        for r in rows:
            a_, d_ = _f(r, "aicore_time(us)"), _f(r, "Task Duration(us)")
            # ★别写成 `if not a_`: `aicore_time == 0.0` 是假值, 那样会把整份全零的产物
            #   判成"两列都在但取不到数"(na) —— **把"有数据且没越界"说成了"考不动"**。
            #   2026-10-22 第一版就是这么写的, 68 份可考产物只认出 9 份。
            if a_ is None or d_ is None or d_ <= 0:
                sk15 += 1
                continue
            n += 1
            if a_ != 0:
                nz += 1
            k = (r.get("Task Type") or "?").strip()
            e = kinds.setdefault(k, [0, 0])
            e[0] += 1
            if a_ > d_:
                ov += 1
                e[1] += 1
                worst = max(worst, a_ / d_)
        if n:
            det = "; ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(kinds.items()))
            covs["B15"] = _cov(n, **{"两列任一取不到数": sk15})
            out.append(("B15", "hit" if ov else "miss",
                        "有 aicore_time 超过 Task Duration (核忙时长于任务时长, 讲不通)"
                        if ov else "aicore_time 全部不超过 Task Duration",
                        f"{ov}/{n} 行越界 (含 aicore_time=0 的行); "
                        f"只数非零行则 {ov}/{nz} = {(100.0 * ov / nz) if nz else 0:.2f}%; "
                        f"最大比值 {worst:.4f}; 按 Task Type: {det}"))
        else:
            out.append(("B15", "na", "两列都在但取不到数", ""))
    else:
        out.append(("B15", "na", "缺 aicore_time(us) 或 Task Duration(us) "
                                 "(kernel_details 没有后者, 本条只在 op_summary 上考得动)", ""))

    # B12 (这才是 D7): 被影响列与它的分母列是否同时在场。
    covs["B12"] = _cov(len(rows))
    pres = [(c, d) for c, d in _BP_D7_PAIRS if c in cols]
    if not pres:
        out.append(("B12", "na", "本产物没有 D7 的任何被影响列", ""))
    else:
        both = [(c, d) for c, d in pres if d in cols]
        if not both:
            out.append(("B12", "unobs",
                        "D7 的被影响列在场, 但**它的分母列一列都不在** —— 本产物看不到这条失真",
                        "在场: " + ", ".join(c for c, _ in pres)
                        + "; 缺分母: " + ", ".join(sorted({d for _, d in pres}))))
        else:
            zr = [(c, d) for c, d in both
                  for r in rows if _f(r, d) == 0 and _f(r, c) is not None]
            out.append(("B12", "hit" if zr else "miss",
                        "分母为 0 且被影响列有值 (D7 的触发形态)" if zr
                        else "两列同时在场, 但没有分母为 0 的行",
                        f"可考列对 {len(both)}, 触发行 {len(zr)}"))
    # ★统一补第 5 项: 没有单独记过的判据按"看了全部行"算。
    # ★blind 守卫 (2026-10-22): **一条判据看了 0 行就不许说"成立"或"没发生"**。
    #   动因就在落地当场撞到: B8 在一份 op_summary 上报 `miss`(恒等式成立),
    #   而它的 `seen` 是 **0** —— 100 行全被 `任一为空或 0` 跳掉了。
    #   这与 B7 在 76 份上"成立"是同一个病, 只是这次被覆盖信息当场照出来。
    #   `blind` 与 `na` 分开: `na` 是"这份产物没有这些列",
    #   `blind` 是"**列都在, 但每一行都被跳过了**" —— 后者要去看跳过理由对不对。
    cols_all = cols
    res = []
    for t, v, w, e in out:
        c = covs.get(t, _cov(len(rows)))
        if v == "miss" and c["seen"] == 0 and c["skipped"]:
            # ★把"为什么跳光"说成领域里的话, 而不是只报跳过理由的字面。
            #   2026-10-22 实测: B7/B8 的 blind **绝大多数是"这份采集没采那一侧的计数器"**
            #   (80 份 op_summary 里只有 2 份的 AIV 行有数; kernel_details 里 AIV 行的
            #   `aicore_time` 恒为 0)。读的人若按"去查跳过理由"去查, 查到的是一个
            #   改不了的结构事实 —— 那是白查。
            # ★按 `/` 与空白切, 不要把 `/` 算进词内 —— 否则
            #   "aiv_time(us)/aiv_total_cycles/Mix Block Num" 会被切成一个 token, 永远匹配不上列名。
            cols = [x for k in c["skipped"] for x in re.split(r"[/\s]+", k) if x in cols_all]
            # 列名里带空格的 (Block Num / Mix Block Num) 单独补
            cols += [x for x in cols_all if x in ("Block Num", "Mix Block Num")
                     and any(x in k for k in c["skipped"])]
            allzero = [x for x in dict.fromkeys(cols)
                       if all((_f(r, x) or 0) == 0 for r in rows)]
            v, w = "blind", "★列都在但**每一行都被跳过**, 这一条什么也没看到"
            e = (f"★{'/'.join(allzero)} 在**每一行上都是 0 或缺失** —— "
                 "多半是这次采集没采这一侧的计数器, 不是判据取错了列"
                 if allzero else
                 f"要去看跳过理由对不对 ({len(c['skipped'])} 条理由, 见下行)")
        res.append((t, v, w, e, c))
    return res


def bpath_corpus(roots: list) -> dict:
    """把 bpath_lint 跑遍多个根下的全部 kernel_details.csv, 汇总判定。

    为什么要有它: 单份产物上的 `miss` 几乎没有信息 —— 它可能是"这条失真不存在",
    也可能是"这份产物考不动这条判据"。**只有跨语料才分得开**:
    一条判据若在 N 份产物上全 `na`, 那它从来没被考过, 不该被当成验过的判据。

    ★2026-10-22 这条扫描当场推翻了一个**已被标为"实测核过"的前提**:
      `11` 篇把 B1 (验这几条判据) 记成"B 组唯一真被资源挡住的", 依据是
      "26 份 kernel_details.csv, 每份恰好 3 行, 单 stream, 零 N/A"。
      那次实测是真的, 但**范围只在仓内**; 全机实有 173 份, 其中 82 份既 >3 行又含 N/A,
      最大的 275 行。结论"被资源挡住"**不从一次仓内扫描得出**。
      教训与 feedback-方法声明必须属实 同源: 声明里缺的不是数字, 是**范围**。
    """
    import glob as _g
    # ★2026-10-22 扩到 op_summary: 两族产物来自不同前端 (kernel_details 是 torch_npu,
    #   op_summary 是 msprof --export), 列名不同, 因此**同一条判据在另一族上多半判 na** ——
    #   那正是要的: 判据按列存在性自己选族, 而语料不该因为"不是我熟的那一族"就被排除在外。
    #   B15 就是扩进来之后才看见的 (它只在 op_summary 上考得动)。
    files = sorted({f for r in roots for pat in ("kernel_details.csv", "op_summary*.csv")
                    for f in _g.glob(os.path.join(r, "**", pat), recursive=True)})
    agg, example, coverage = {}, {}, {}
    for f in files:
        for tag, v, why, ev, cov in bpath_lint(f):
            agg.setdefault(tag, {}).setdefault(v, 0)
            agg[tag][v] += 1
            example.setdefault(tag, {}).setdefault(v, (why, ev, f))
            c = coverage.setdefault(tag, {"seen": 0, "skipped": {}})
            c["seen"] += cov["seen"]
            for k, n2 in cov["skipped"].items():
                c["skipped"][k] = c["skipped"].get(k, 0) + n2
    return {"files": files, "agg": agg, "example": example, "coverage": coverage}


def print_bpath_corpus(roots: list) -> int:
    r = bpath_corpus(roots)
    if not r["files"]:
        print(f"这些根下没有 kernel_details.csv: {roots}")
        return 2
    kd = sum(1 for f in r["files"] if os.path.basename(f) == "kernel_details.csv")
    print(f"=== B 路径判据语料扫描: {len(r['files'])} 份 "
          f"(kernel_details {kd}, op_summary {len(r['files']) - kd}) ===")
    print("  ★两族来自不同前端, 列名不同 —— 同一条判据在另一族上多半判 na, 那是正常的。")
    print(f"{'判据':<6}{'hit':>6}{'miss':>6}{'unobs':>7}{'blind':>7}{'na':>6}"
          f"{'看见行':>9}{'跳过行':>9}  判读")
    never = []
    for tag in sorted(r["agg"]):
        a = r["agg"][tag]
        h, m, n = a.get("hit", 0), a.get("miss", 0), a.get("na", 0)
        u, bl = a.get("unobs", 0), a.get("blind", 0)
        cv = r["coverage"].get(tag, {"seen": 0, "skipped": {}})
        skn = sum(cv["skipped"].values())
        ident = tag in _BP_IDENTITY
        if h == 0 and m == 0 and u:
            note = (f"★{u} 份判**不可观测** —— 被影响列与它的分母**从不同时出现**; "
                    "再采多少份都看不到, 要换数据源")
        elif h == 0 and m == 0:
            note = "★从来没被考过 (全 na) —— 不算验过的判据"
            never.append(tag)
        elif ident:
            # ★极性与失真检测相反: 这一类里 hit = 恒等式**不成立**, miss = 成立。
            note = (f"在 {m} 份上成立" if h == 0
                    else f"★在 {h}/{h + m} 份上**不成立** —— 该读法有反例")
            if h == 0 and m == 0:
                never.append(tag)
        elif h == 0:
            note = f"★考得动 {m} 份但**一次没命中** —— 失真是从代码读出的, 未在产物上见过"
            never.append(tag)
        elif m == 0 and n == 0:
            note = "全语料命中"
        else:
            note = f"命中 {h}/{h + m} 份可考产物"
        print(f"{tag:<6}{h:>6}{m:>6}{u:>7}{bl:>7}{n:>6}{cv['seen']:>9}{skn:>9}"
              f"  {'[不变量] ' if ident else ''}{note}")
        # ★跳过远多于看见时, 它的"成立"要降级看待 —— B7 就是这么混过 76 份的。
        if cv["seen"] and skn > cv["seen"] * 3:
            top = max(cv["skipped"].items(), key=lambda kv: kv[1])
            print(f"        ★跳过 {skn} 行 / 看 {cv['seen']} 行 = {skn / cv['seen']:.1f} 倍 —— "
                  f"主要理由「{top[0]}」{top[1]} 行。结论的作用域只覆盖看见的那部分。")
    for tag in sorted(r["agg"]):
        for v, (why, ev, f) in sorted(r["example"][tag].items()):
            if v != "na":
                print(f"   {tag} [{v}] {why}")
                print(f"        {ev}")
    if never:
        print(f"\n★未被产物证实的判据: {', '.join(never)}。"
              "它们现在的地位是**从代码逆向出的断言**, 不是实测结论 —— 引用时要这么说。")
    return 0


# ------------------------------------------------- A 路径自洽体检
#  A 路径 (`OPPROF_*`, msopprof 产的) 是**本仓性能诊断实际读的那一族**, 却直到
#  2026-10-22 没有被任何一条判据扫过 —— B 路径当时的 9 条判据 (今已更多, 这个 9 是
#  **当时的**, 留着是因为下一句的论点不依赖它) 全部按 kernel_details /
#  op_summary 的列名取数, 在 A 路径上一条都不适用。
#
#  两族的粒度根本不同: B 路径一行是一个 kernel (所以要除 Block Num, 而除数还随核类型变),
#  **A 路径一行是一个 block** (`block_id` / `sub_block_id`), 所以同一条恒等式在这边
#  干净得多 —— 不用除核数。
#
#  已有的 `_corrupt_launches` 查的是 F2 (各 pipe 时长不得超过任务时长), 它是**丢弃规则**;
#  下面三条是**只报不丢**的自洽检查, 因为违反它们未必说明这一份废了
#  (比如换了频率), 自动丢会把判断替人做掉。
_AP_RATIO_SUF = ("_ratio", "_rate(%)")


def apath_lint(csv_path: str, freq_mhz: float | None = None) -> list:
    """对一份 A 路径 csv 跑自洽判据。返回 [(编号, 判定, 说明, 证据, 看见范围)]。"""
    import csv as _csv
    out, covs = [], {}
    try:
        rows = list(_csv.DictReader(open(csv_path, encoding="utf-8", errors="replace")))
    except OSError as ex:
        return [("A0", "na", "读不出来", str(ex), _cov())]
    if not rows:
        return [("A0", "na", "空表", csv_path, _cov())]
    cols = set(rows[0].keys())

    def _f(r, c):
        try:
            return float((r.get(c) or "").strip())
        except (TypeError, ValueError):
            return None

    if freq_mhz is None:
        try:
            from core import socspec
            freq_mhz = socspec.spec()["freq_ghz"] * 1000
        except Exception:  # noqa: BLE001
            freq_mhz = None

    # A1: 一行一个 block, 所以 cycles/(freq*time) 应当恰为 1 —— 不除核数。
    #     ★2026-10-22 实测 2880 行 **100% 等于 1.0000**, 这正是"A 路径逐 block"的直接证据。
    for tag, tcol, ccol in (("A1", "aic_time(us)", "aic_total_cycles"),
                            ("A1v", "aiv_time(us)", "aiv_total_cycles")):
        if not {tcol, ccol} <= cols or freq_mhz is None:
            out.append((tag, "na", f"缺列 {tcol}/{ccol} 或取不到频率", "", _cov()))
            continue
        seen = sk = 0
        bad = []
        for i, r in enumerate(rows):
            t, c = _f(r, tcol), _f(r, ccol)
            if not t or not c:
                sk += 1
                continue
            seen += 1
            q = c / t / freq_mhz
            if abs(q - 1.0) > 0.01:
                bad.append((i, round(q, 4)))
        covs[tag] = _cov(seen, **{f"{tcol}/{ccol} 任一为空或 0": sk})
        out.append((tag, "hit" if bad else "miss",
                    f"{tcol} 与 {ccol}/freq 对不上 —— 该列不是逐 block" if bad
                    else f"{tcol} = {ccol}/freq 恰好成立 —— 一行一个 block, 不除核数",
                    (f"{len(bad)} 行不符, 例: 行{bad[0][0]} 推出 {bad[0][1]}" if bad
                     else f"{seen} 行全部吻合")))

    # A2: 命中率与**同一行里**的计数器自洽。这一条不依赖任何外部常数, 是最硬的一条。
    # ★读命中率合并 **r0 与 r1 两组**计数器。2026-10-22 实撞: 第一版只配了 r0,
    #   于是 6 份产物被报成"对不上"(最大偏差 0.0296 个百分点) —— 那不是失真,
    #   是我配错了对。合并 r0+r1 之后偏差降到 **7e-6 个百分点**。
    #   对读这几列的人的含义: **只看 r0 会得到一个接近但系统性偏低的命中率**。
    pairs = [("aic_write_hit_rate(%)", ["aic_write_cache_hit"],
              ["aic_write_cache_miss_allocate"]),
             ("aic_read_hit_rate(%)", ["aic_r0_read_cache_hit", "aic_r1_read_cache_hit"],
              ["aic_r0_read_cache_miss_allocate", "aic_r1_read_cache_miss_allocate"])]
    usable = [p for p in pairs if {p[0], *p[1], *p[2]} <= cols]
    if not usable:
        out.append(("A2", "na", "本产物没有命中率与它的计数器", "", _cov()))
    else:
        seen = sk = 0
        worst = 0.0
        bad = 0
        for r in rows:
            for rate_c, h_cs, m_cs in usable:
                hs = [_f(r, c) for c in h_cs]
                ms = [_f(r, c) for c in m_cs]
                rate = _f(r, rate_c)
                if rate is None or any(x is None for x in hs + ms):
                    sk += 1
                    continue
                h, m = sum(hs), sum(ms)
                if (h + m) <= 0:
                    sk += 1
                    continue
                seen += 1
                d = abs(100.0 * h / (h + m) - rate)
                worst = max(worst, d)
                if d > 0.01:
                    bad += 1
        covs["A2"] = _cov(seen, **{"计数器或命中率取不到数 / 分母为 0": sk})
        out.append(("A2", "hit" if bad else "miss",
                    "命中率与它同一行的计数器对不上" if bad
                    else "命中率与它同一行的计数器自洽 —— 这一条不依赖任何外部常数",
                    f"{bad}/{seen} 行不符, 最大偏差 {worst:.6f} 个百分点"))

    # A3: 比值与百分率列的上界。`_ratio` 不得 >1, `_rate(%)` 不得 >100。
    rc = [c for c in cols if c.endswith(_AP_RATIO_SUF)]
    if not rc:
        out.append(("A3", "na", "没有比值或百分率列", "", _cov()))
    else:
        seen = 0
        over = []
        for r in rows:
            for c in rc:
                v = _f(r, c)
                if v is None:
                    continue
                seen += 1
                lim = 100.0 if c.endswith("_rate(%)") else 1.0
                if v > lim:
                    over.append((c, v))
        covs["A3"] = _cov(seen)
        out.append(("A3", "hit" if over else "miss",
                    "有比值或百分率列越过上界" if over else "全部比值 ≤1、百分率 ≤100",
                    (f"{len(over)} 处, 例 {over[0][0]}={over[0][1]}" if over
                     else f"{seen} 个取值, {len(rc)} 列")))
    return [(t, v, w, e, covs.get(t, _cov(len(rows)))) for t, v, w, e, *_ in
            [(x[0], x[1], x[2], x[3]) for x in out]]


def print_apath_lint(path: str) -> int:
    """给一份 csv 或一个 `OPPROF_*` 目录。"""
    import glob as _g
    files = ([path] if path.endswith(".csv")
             else sorted(_g.glob(os.path.join(path, "**", "*.csv"), recursive=True)))
    if not files:
        print(f"{path} 下没有 csv —— 要给一份 A 路径 csv 或一个 OPPROF_* 目录")
        return 2
    agg, cov = {}, {}
    for f in files:
        for tag, v, why, ev, c in apath_lint(f):
            agg.setdefault(tag, {}).setdefault(v, 0)
            agg[tag][v] += 1
            e = cov.setdefault(tag, {"seen": 0, "skipped": 0, "why": why, "ev": ev})
            e["seen"] += c["seen"]
            e["skipped"] += sum(c["skipped"].values())
            if v != "na":
                e["why"], e["ev"] = why, ev
    print(f"=== A 路径自洽体检: {len(files)} 份 csv ===")
    print(f"{'判据':<6}{'hit':>6}{'miss':>6}{'na':>6}{'看见':>9}{'跳过':>8}  说明")
    for tag in sorted(agg):
        a = agg[tag]
        c = cov[tag]
        print(f"{tag:<6}{a.get('hit', 0):>6}{a.get('miss', 0):>6}{a.get('na', 0):>6}"
              f"{c['seen']:>9}{c['skipped']:>8}  {c['why']}")
        if c["ev"]:
            print(f"        {c['ev']}")
    print("\n  ★A 路径一行是一个 block, 所以 A1 不除核数 —— 这与 B 路径的 B7 是同一条恒等式"
          "在两种粒度上的样子。")
    return 1 if any(a.get("hit") for a in agg.values()) else 0


# ------------------------------------------------- 一站式: 这批产物可不可信
#  ★动机: 到 2026-10-22 本模块已有四条"采集之后"的检查 (dropcheck / profdir 的切片断号 /
#    profenv / apath-lint 与 bpath-lint), 它们都在 attr 区里注册着, 但**没有任何一步
#    告诉人采集之后要先验数据** —— 本仓的工具时间轴是
#    envcheck → symptom → preflight → ablate → diagnose gate → hwfloor → shipcheck,
#    中间没有"验这批数字可不可信"这一格。于是四条命令全靠人记得敲, 而**最该敲它们的时刻
#    恰恰是人最想直接看数字的时刻**。
#  做成一条, 并且遵守本会话自己立的那条规矩: **说不出话时要说清是哪一种说不出话** ——
#  每一项分 过 / 不过 / 没看见 / 不适用 四档, 而不是合成一个"没发现问题"。
def trust_report(path: str, since_min: float | None = 60.0) -> list:
    """给一个产物目录 (PROF_* / OPPROF_* / 它们的父目录), 返回 [(项, 判定, 说明)]。"""
    import glob as _g
    out = []
    roots_p = [p for p in ([path] if os.path.basename(path).startswith("PROF_")
                           else _g.glob(os.path.join(path, "**", "PROF_*"), recursive=True))
               if os.path.isdir(p)]
    roots_a = [p for p in ([path] if os.path.basename(path).startswith("OPPROF_")
                           else _g.glob(os.path.join(path, "**", "OPPROF_*"), recursive=True))
               if os.path.isdir(p)]
    if not roots_p and not roots_a:
        return [("目录", "不适用", f"{path} 下既没有 PROF_* 也没有 OPPROF_* —— 给错层了")]

    # 1 丢条数对账。★这一项查的是**昇腾 slog**, 不是这个目录 —— 不对齐时间窗的话,
    #   无论给哪个产物它都给同一个答案, 而摆在一张以"这批产物"为题的表里就是误导。
    #   所以按产物目录名里的采集时间戳把窗口对齐到"那次采集之后到现在", 并在说明里
    #   写明实际用的窗口; 取不到时间戳就退回调用方给的窗口并标注"机器级, 未对齐"。
    import datetime as _dt
    stamp = None
    for r in (roots_p + roots_a):
        m = re.search(r"_(\d{14})", os.path.basename(r))
        if m:
            try:
                t = _dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
                stamp = t if stamp is None or t < stamp else stamp
            except ValueError:
                pass
    if stamp:
        win = max(1.0, (_dt.datetime.now() - stamp).total_seconds() / 60.0)
        scope = f"窗口按采集时间对齐 ({stamp:%Y-%m-%d %H:%M} 至今, {win:.0f} 分钟)"
    else:
        win, scope = since_min, "★机器级, **未按采集时间对齐** —— 目录名里没有时间戳"
    d = prof_drops(since_min=win)
    out.append(("丢条数对账",
                {"CLEAN": "过", "DROP": "不过", "UNDRAINED": "不过", "BLIND": "没看见"}[d["verdict"]],
                f"{d['verdict']}: 计数行 {d['tsr_lines']} 条"
                + (f", 越界/溢出 {len(d['hard']) + len(d['mismatch'])} 处"
                   if d["hard"] or d["mismatch"] else "")
                + f"; {scope}"))

    # 2 切片断号 (只对 PROF_*)
    if roots_p:
        g = [x for r in roots_p for x in slice_gaps(prof_inventory(r))]
        pre = [x for x in g if x[4]]
        out.append(("切片断号", "不过" if g else "过",
                    (f"{len(g)} 处断号, 其中前缀型 {len(pre)} (= 最旧先删, 淘汰的签名)"
                     if g else f"{len(roots_p)} 个 PROF 目录, 编号连续")))
    else:
        out.append(("切片断号", "不适用", "没有 PROF_* 目录 (A 路径不分切片)"))

    # 3 环境指纹
    envs = prof_env_scan([path])
    if not envs:
        out.append(("采集环境", "没看见", "没有可读的采集配置"))
    else:
        noconf = sum(v["n"] for k, v in envs.items() if k[0].startswith("<A 路径"))
        total = sum(v["n"] for v in envs.values())
        # ★单份产物上这一项**永远不会不过** —— 一份产物当然只属于一个环境。
        #   2026-10-22 样本外实测: 300 份里 300 份判"过"。
        #   **一项永远不会不过的检查等于没有检查, 而它还报"过", 读起来像保证。**
        #   所以样本只有一份时判"不适用"并说明: 这一项要跨产物才有意义。
        if total <= 1:
            out.append(("采集环境", "不适用",
                        "只有 1 份产物 —— 这一项要**跨产物**才有意义 "
                        "(单份当然只属于一个环境); 要查混环境请把父目录传进来"))
            envs = None
    if envs:
        noconf = sum(v["n"] for k, v in envs.items() if k[0].startswith("<A 路径"))
        out.append(("采集环境", "不过" if len(envs) > 1 else "过",
                    f"{len(envs)} 个环境"
                    + (f" (其中 {noconf} 份 A 路径无采集配置, 归属靠推断)" if noconf else "")
                    + ("  ★混了环境的产物不能放进同一张表" if len(envs) > 1 else "")))

    # 4 列判据
    for lab, pat, fn in (("B 路径判据", ("kernel_details.csv", "op_summary*.csv"), bpath_lint),
                         ("A 路径判据", ("*.csv",), apath_lint)):
        files = sorted({f for r in (roots_a if fn is apath_lint else roots_p + [path])
                        for p2 in pat
                        for f in _g.glob(os.path.join(r, "**", p2), recursive=True)})
        if fn is apath_lint:
            files = [f for f in files if "OPPROF_" in f]
        if not files:
            out.append((lab, "不适用", "没有这一族的 csv"))
            continue
        hit = blind = 0
        for f in files:
            for t, v, _w, _e, _c in fn(f):
                hit += v == "hit"
                blind += v == "blind"
        out.append((lab, "不过" if hit else ("没看见" if blind and not hit else "过"),
                    f"{len(files)} 份 csv, 命中 {hit} 项"
                    + (f", 另有 {blind} 项**列都在但每行都被跳过**" if blind else "")))
    return out


def print_trust(path: str, since_min: float | None = 60.0) -> int:
    rows = trust_report(path, since_min)
    mark = {"过": "  过", "不过": "★不过", "没看见": "★没看见", "不适用": "  不适用"}
    print(f"=== 这批产物可不可信 {path} ===")
    for k, v, why in rows:
        print(f"  {mark[v]:<8}{k:<12}{why}")
    bad = [r for r in rows if r[1] == "不过"]
    blind = [r for r in rows if r[1] == "没看见"]
    print()
    if bad:
        print(f"  ★{len(bad)} 项不过 —— 在用这批数字之前先看它们。")
    if blind:
        print(f"  ★{len(blind)} 项**没看见** —— 这不是'没问题', 是这次检查什么也没看到。")
    if not bad and not blind:
        print("  四项全过。★'过'只覆盖这四条查得了的, 不等于这批数字就对 —— "
              "导出列的失真另见 --bpath-lint / --apath-lint 的逐条输出。")
    return 2 if bad else (1 if blind else 0)


#  ★自配 PMU 事件的回读 (01 §3.1b, 2026-10-27)。
#  顶层 `msprof --aic-metrics=Custom:<hex>,<hex>` 把事件号**原样下发**, 产物里以
#  `aic_r<hex>` / `aiv_r<hex>` 的原始计数列出现。这条通路**只在 B 路径**:
#  `msprof op` 对同一个参数直接拒 (`Invalid metric options!`), 而且**拒了之后退出码仍是 0**。
def event_readback(prof_dir: str) -> dict:
    """从一次 B 路径采集里读回: 要的事件号 / 实际下发的 / 产物里拿得到的列。

    ★**三者要分开报**, 不能只报一个。实测它们会不一致:
      预置档位 `PipeUtilization` 下发的是展开后的 8 个事件, 而产物列是派生名
      (`aic_mac_ratio` …), 根本看不出事件号; `Custom:` 下发几个就是几个,
      产物列直接是 `aic_r<hex>`。只看产物列会以为没采到, 只看下发列会以为能解出。
    """
    import glob as _g
    out = {"prof_dir": prof_dir, "asked": None, "sent": [], "cols": [],
           "sample_json": None, "summary_csv": None}
    sj = sorted(_g.glob(os.path.join(prof_dir, "**", "sample.json"), recursive=True))
    if sj:
        out["sample_json"] = sj[0]
        try:
            d = json.load(open(sj[0], encoding="utf-8", errors="replace"))
        except Exception:                                    # noqa: BLE001
            d = {}
        out["asked"] = d.get("ai_core_metrics")
        raw = (d.get("ai_core_profiling_events") or "").strip()
        out["sent"] = [e.strip() for e in raw.split(",") if e.strip()]
    cs = sorted(_g.glob(os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True))
    if cs:
        out["summary_csv"] = cs[0]
        with open(cs[0], newline="", errors="ignore") as f:
            hdr = next(csv.reader(f), [])
        out["cols"] = [c for c in hdr if re.match(r"^ai[cv]_r[0-9a-fA-F]+$", c)]
    return out


def aic_event_names(analysis_root: str | None = None, chip: str | None = None) -> dict:
    """`{事件号(小写, 去 0x) -> 事件名}`, 取自解析侧 msconfig 的 ai_core 配置。

    ★**这张表本仓早就登记过** (`01` §6.6 的 `events` 节; `07` E-9 还订正过它的条数)。
      本函数不是"发现"它, 是把它**接到 `Custom:` 这条路上** —— `--aic-metrics=Custom:0x9,0xc`
      的产物列叫 `aic_r9` / `aic_rc`, `r` 后面那串十六进制就是事件号, 查这张表即得
      `scalar_busy_cycles` / `mte2_busy_cycles`。
      **登记在册与用得上之间差的就是这一步**, 而 2026-10-27 查这件事时, 我是第四次
      把仓里已有的东西当成新发现 —— 毛病在按**内容**去 grep, 而不先问"这个主题在索引里
      是哪一节" (本仓有 `opdbg whatfor` 正是干这个的)。
    ★用 AST 读, 不 import CANN 的包 (与 `core/aprof` 读记录布局同法)。

    ★★**2026-10-27 修一处真缺陷**: 该文件里有**三张** `events` 表
    (`DATA_DEFAULT` 155 条 / `CHIP_V5_MAP` 0 条 / `CHIP_V6_MAP` 41 条), 而 CANN 自己是
    `DATA = DATA_MAP.get(chip_id, DATA_DEFAULT)` —— **按芯片选一张, 不合并**。
    首版把三张**合并**, 于是 7 个重叠号拿到了后扫到那张的名字, 在本机 (`CHIP_V4_1_0`,
    既不是 V5 也不是 V6, 应落 `DATA_DEFAULT`) 全是错的:

        0x1   本机应为 vec_instr_exec, 合并后给成 scalar_ratio
        0x3   cube_instr_exec -> ub_read_bw_scalar(GB/s)
        0x5   mte2_instr_exec -> ub_write_bw_scalar(GB/s)
        0x34 / 0x35 / 0x206 / 0x502 同病

    **这类错不会报错, 只会给出一个讲得通的错名字** —— 按它去 `--measure-events` 就会
    采到另一个事件, 而表头仍写着你要的那个名字。`chip` 不给时按 CANN 的默认走
    `DATA_DEFAULT`, 与它的 `.get(..., DATA_DEFAULT)` 同义。
    """
    from core.aprof import bm_find_analysis
    an = bm_find_analysis(analysis_root)
    if not an:
        return {}
    f = os.path.join(an, "msconfig", "ai_core_config.py")
    if not os.path.isfile(f):
        return {}
    try:
        tree = ast.parse(open(f, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return {}

    def _events_of(dict_node):
        got = {}
        for k, v in zip(dict_node.keys, dict_node.values):
            if not (isinstance(k, ast.Constant) and k.value == "events"):
                continue
            if not isinstance(v, (ast.List, ast.Tuple)):
                continue
            for el in v.elts:
                if (isinstance(el, (ast.Tuple, ast.List)) and len(el.elts) == 2
                        and all(isinstance(x, ast.Constant) for x in el.elts)):
                    num, name = el.elts[0].value, el.elts[1].value
                    if isinstance(num, str) and num.lower().startswith("0x"):
                        got[num.lower()[2:]] = name
        return got

    tables = {}                      # {变量名: {号: 名}}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for t in node.targets:
            nm = getattr(t, "id", None) or getattr(t, "attr", None)
            if nm:
                ev = _events_of(node.value)
                if ev:
                    tables[nm] = ev
    if not tables:
        return {}
    if chip:
        #  `DATA_MAP` 的键写成 `ChipModel.CHIP_V6_1_0`, 取 attr 当芯片名。
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(getattr(t, "id", "") == "DATA_MAP" for t in node.targets):
                continue
            for k, v in zip(node.value.keys, node.value.values):
                key = getattr(k, "attr", None)
                var = getattr(v, "id", None)
                if key == chip and var in tables:
                    return tables[var]
    return tables.get("DATA_DEFAULT", next(iter(tables.values())))


def print_event_readback(prof_dir: str) -> int:
    r = event_readback(prof_dir)
    print(f"=== 自配 PMU 事件回读 (01 §3.1b) === {prof_dir}")
    if not r["sample_json"]:
        print("  找不到 sample.json —— 这不是一次 B 路径 (顶层 msprof) 采集。")
        print("  ★`msprof op` 不支持 `Custom:`, 它直接拒 `Invalid metric options!`,")
        print("    **而且拒了之后退出码仍是 0** —— 别拿退出码判成功。")
        return 2
    print(f"  要的档位 (`ai_core_metrics`)      : {r['asked']!r}")
    print(f"  实际下发 (`ai_core_profiling_events`): {len(r['sent'])} 个  {','.join(r['sent'])}")
    names = aic_event_names()
    if r["sent"]:
        print("  下发事件的名字 (查解析侧 msconfig 的 events 表, 见 01 §6.6):")
        for e in r["sent"]:
            print(f"      {e:<8s} {names.get(e.lower().replace('0x', ''), '★表里查不到这个号')}")
        if not names:
            print("    ★**表没读到** (找不到 analysis 目录或该配置文件) —— 上面只有号没有名,")
            print("      这与'号不在表里'是两回事, 别混")
        else:
            miss = [e for e in r["sent"] if e.lower().replace("0x", "") not in names]
            if miss:
                print(f"    ★{len(miss)} 个号不在表里: {','.join(miss)} —— "
                      f"表按芯片档给, 换档可能对不上")
    if r["cols"]:
        print(f"  产物里的原始计数列               : {len(r['cols'])} 个  {','.join(r['cols'])}")
        for c in r["cols"]:
            h = c.split("_r", 1)[1].lower()
            print(f"      {c:<12s} = 事件 0x{h}  {names.get(h, '★表里查不到这个号')}")
        want = {e.lower().replace("0x", "") for e in r["sent"]}
        got = {c.split("_r", 1)[1].lower() for c in r["cols"]}
        miss = sorted(want - got)
        print(f"  ★下发了而产物里没有的事件       : {miss if miss else '无'}")
    else:
        print("  产物里**没有** `ai[cv]_r<hex>` 形态的原始计数列。")
        print("  ★这不等于没采到: 预置档位出的是派生列名 (`aic_mac_ratio` …),")
        print("    事件号在列名里看不出来。只有 `Custom:` 档出原始计数列。")
    return 0


def aic_event_ids(analysis_root: str | None = None) -> dict:
    """`{事件名 -> [事件号(小写, 去 0x), ...]}` —— `aic_event_names` 的反向表。

    ★值是**列表**不是单值: 同一个名字可以挂多个号 (表按芯片档拼, 不同档的同名事件号不同)。
      做成单值就会静默丢掉其中一个, 而丢掉哪个取决于字典序 —— 这正是"看起来能用、
      换台机器就对不上"的形态。重名时由调用方显式挑, 不由本函数替它挑。
    """
    out = {}
    for num, name in aic_event_names(analysis_root).items():
        out.setdefault(name, []).append(num)
    return {k: sorted(v) for k, v in out.items()}


#  `Custom:` 只在 B 路径 (顶层 msprof) 成立; A 路径 (`msprof op` -> msopprof) 结构上不收。
#  证据见 `docs/msprof-re/01-cli-surface.md` §3.1c: msopprof(9251896) 里没有 `Custom:`
#  这个词, 档位判定是精确小写键查表, `Custom:0x9,0xc` 与乱写的 `Bogus` 报逐字相同的错。
#  **不存在的分支没有开关** —— 所以这里是拒跑, 不是换个参数重试。
_CUSTOM_A_PATH_WHY = (
    "`msprof op` (A 路径) 结构上不收 `Custom:` —— 不是白名单漏了它, 是 msopprof 里\n"
    "    根本没有这个词 (见 docs/msprof-re/01-cli-surface.md §3.1c)。\n"
    "    ★它拒绝之后**退出码仍是 0**, 所以不拦的话会表现为'跑完了但没有原始计数列'。\n"
    "    自配事件走 B 路径: `python -m core.perf --measure-events <事件> --app <命令>`")


#  一次 `Custom:` 能点几个事件。**实测扫出来的, 不是从文档抄的** (2026-10-27):
#  顶层 msprof 的参数校验对 1/2/4/8 个放行, 9/10/12/16/17 个一律报
#  `invalid value:Custom:<原串>` —— 切在 8 与 9 之间, 与 `--npu-events` 的上限同为 8
#  (`01` §7.1)。8 个的那一跑已端到端验过: 8 事件 × AIC/AIV 两侧 = 16 列全部到位。
#  来源: 本仓自定 (实测值, 复现命令见 `01` §3.1e)。
CUSTOM_EVENT_MAX = 8


def resolve_events(spec: str, analysis_root: str | None = None) -> tuple:
    """把 `--measure-events` 的取值解析成事件号列表。返回 (号列表, 名字表, 问题列表)。

    取值可混写**事件名**与**事件号**: `scalar_busy_cycles,0xc` 与 `0x9,mte2_busy_cycles`
    等价。名字查 `aic_event_ids`, 号直接用。

    ★解不开的 token **不静默跳过**, 进 problems 由调用方拒跑 —— 少采一个事件的表
      与采全了的表长得一模一样, 而列是按实际下发来的, 事后分不出"没要"与"要了没给"。
    """
    names = aic_event_names(analysis_root)
    rev = aic_event_ids(analysis_root)
    ids, probs = [], []
    if not names:
        probs.append("读不到解析侧的 events 表 (找不到 analysis 目录) —— "
                     "只能用事件号, 名字解不了")
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        low = tok.lower()
        if re.fullmatch(r"(0x)?[0-9a-f]+", low) and (low.startswith("0x") or not rev.get(tok)):
            h = low[2:] if low.startswith("0x") else low
            ids.append(h)
            if names and h not in names:
                probs.append(f"事件号 0x{h} 不在本机的 events 表里 —— "
                             f"下发多半仍会成功, 但解析侧给不出名字")
            continue
        cand = rev.get(tok)
        if not cand:
            near = sorted(n for n in rev if low in n.lower())[:6]
            probs.append(f"事件名 {tok!r} 查不到" +
                         (f"; 形近的有: {', '.join(near)}" if near else
                          "; 敲 `--list-events` 看全表"))
            continue
        if len(cand) > 1:
            probs.append(f"事件名 {tok!r} 对应多个号 ({', '.join('0x' + c for c in cand)}) "
                         f"—— 请直接写号, 本工具不替你挑")
            continue
        ids.append(cand[0])
    dup = sorted({h for h in ids if ids.count(h) > 1})
    if dup:
        probs.append(f"重复的事件号: {', '.join('0x' + d for d in dup)} —— "
                     f"同一个号下发两次只会得到一列")
    if len(ids) > CUSTOM_EVENT_MAX:
        probs.append(f"一次最多点 {CUSTOM_EVENT_MAX} 个事件, 给了 {len(ids)} 个 —— "
                     f"提前拒, 免得白跑一趟采集")
    return ids, names, probs


def print_list_events(analysis_root: str | None = None) -> int:
    names = aic_event_names(analysis_root)
    if not names:
        print("读不到解析侧的 events 表 —— 本机找不到 analysis 目录或该配置文件。")
        print("★这与'表是空的'是两回事: 前者是环境问题, 后者才是真缺。")
        return 2
    print(f"=== 本机可点的 AI Core PMU 事件 {len(names)} 个 (01 §6.6 的 events 表) ===")
    print("  用法: --measure-events scalar_busy_cycles,0xc   (名字与号可混写)\n")
    for h, n in sorted(names.items(), key=lambda kv: int(kv[0], 16)):
        print(f"  0x{h:<6s} {n}")
    print("\n★这张表是**解析侧**登记的全集; 下发时芯片实际支持哪些, 以一次采集的回读为准。")
    return 0


def _guard_phys_card0() -> None:
    """硬约束 0: 物理 0 卡是管理员的 CI runner, 上面起进程会被杀。

    判据看 `ASCEND_RT_VISIBLE_DEVICES` 的**首项** —— 置了可见卡之后进程内逻辑卡号被
    重编号为 0, 按 `aclrtSetDevice` 的实参判会把正常用法全拦掉 (与 `core/precloc`
    的 `_guard_device` 同法)。
    """
    v = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    first = v.split(",")[0].strip() if v else ""
    if not first:
        raise SystemExit("[perf] 未设置 ASCEND_RT_VISIBLE_DEVICES —— 拒跑, "
                         "以免落到物理 0 卡 (管理员的 CI runner)")
    if first == "0":
        raise SystemExit("[perf] ASCEND_RT_VISIBLE_DEVICES 首项是物理 0 卡 —— 拒跑")


def run_bpath_events(app_cmd: str, ids: list, out_dir: str, timeout_s: float = 1800.0) -> dict:
    """B 路径 (顶层 `msprof`) 按自配 PMU 事件采一次。返回 {"rc", "out_dir", "stdout"}。

    与 `run_msprof_op` 的分工: 那条是 A 路径 (`msprof op` -> msopprof), 给 per-pipe 的
    **派生**分解, 是本仓性能诊断的主路; 这条是 B 路径, 给**原始计数**, 用于回答
    A 路径的派生列答不了的问题 (想测的量没有现成列时)。两条不能在一次运行里都拿。
    """
    _guard_phys_card0()
    os.makedirs(out_dir, exist_ok=True)
    #  MSPTI 与 msprof 互斥且**单向** (MSPTI 收齐, msprof 被挤空), 肇因是 LD_PRELOAD
    #  本身而非 start monitor —— 与 run_msprof_op 同一条, 复用同一个摘除函数。
    env, stripped = strip_mspti_preload(os.environ)
    if stripped:
        print("  [perf] ★已从 LD_PRELOAD 摘掉 libmspti.so —— 它与 msprof 互斥, "
              "留着本次采集会产出空目录。", flush=True)
    #  顶层 msprof 的 --application 与 `msprof op` 一样只收**可执行文件 + 参数**,
    #  不收 shell 命令 —— 同一个坑, 同一个弥合法 (2026-09-17 曾四趟采集全栽在这里)。
    if re.search(r"&&|\|\||;|^\s*cd\s", app_cmd):
        import stat as _stat
        w = os.path.join(out_dir, "_app_wrapper.sh")
        with open(w, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n" + app_cmd + "\n")
        os.chmod(w, os.stat(w).st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH)
        app_cmd = w
    metric = "Custom:" + ",".join("0x" + h for h in ids)
    cmd = ["msprof", f"--aic-metrics={metric}", "--ai-core=on",
           f"--output={out_dir}", f"--application={app_cmd}"]
    print(f"  [perf] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    return {"rc": p.returncode, "out_dir": out_dir, "stdout": p.stdout + p.stderr}


def bpath_event_counts(prof_dir: str) -> list:
    """从一次 B 路径采集的 `op_summary_*.csv` 里把原始计数列按算子汇总。

    返回 [{"op": 算子名, "n": 行数, "cols": {列名: 均值}}]。

    ★分母**一起取**: `aic_total_cycles` / `aiv_total_cycles` 在自配档的产物里照常存在
      (2026-10-27 实测该档 29 列里就有这两列), 所以**它不占事件槽** —— 一次只能点 8 个
      事件, 要是还得拿一个去换分母就只剩 7 个可用。首版的输出文案写的是"分母也要作为
      一个事件一起采", 那是错的, 已改。
    """
    import glob as _g
    cs = sorted(_g.glob(os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True))
    if not cs:
        return []
    rows = list(csv.DictReader(open(cs[0], newline="", errors="ignore")))
    cols = [c for c in (rows[0].keys() if rows else [])
            if re.match(r"^ai[cv]_r[0-9a-fA-F]+$", c or "")]
    #  每个 r<hex> 列按它自己的前缀配分母: aic_* 配 aic_total_cycles, aiv_* 同理。
    #  ★两侧的 total_cycles **不是同一个量**, 且 block 口径不同 (`03` §2: AIC 来自 context
    #    记录的 total_cycle, AIV 是 48 条 block 记录之和) —— 混用会得到差一个核数倍的比率。
    dens = [c for c in (rows[0].keys() if rows else [])
            if c in ("aic_total_cycles", "aiv_total_cycles")]
    opkey = next((k for k in ("Op Name", "op_name", "Op_Name") if rows and k in rows[0]), None)
    allc = cols + dens
    agg = {}
    for r in rows:
        op = (r.get(opkey) or "?") if opkey else "?"
        d = agg.setdefault(op, {"op": op, "n": 0, "sum": {c: 0.0 for c in allc},
                                "seen": {c: 0 for c in allc}})
        d["n"] += 1
        for c in allc:
            try:
                d["sum"][c] += float(r.get(c) or 0.0)
                d["seen"][c] += 1
            except (TypeError, ValueError):
                pass
    out = []
    for d in agg.values():
        mean = {c: (d["sum"][c] / d["seen"][c]) if d["seen"][c] else None for c in allc}
        #  ★比率按**该算子这些行的总和之比**算, 不是逐行比率再取均值 ——
        #    后者给的是"每次 launch 的占比的平均", 两者在各次 launch 时长不等时不相等,
        #    而我们要的是这段负载整体上这条 pipe 占了多少。
        ratio = {}
        for c in cols:
            den = "aic_total_cycles" if c.startswith("aic_") else "aiv_total_cycles"
            s_den = d["sum"].get(den) or 0.0
            ratio[c] = (d["sum"][c] / s_den) if s_den else None
        out.append({"op": d["op"], "n": d["n"],
                    "cols": {c: mean[c] for c in cols},
                    "den": {c: mean[c] for c in dens},
                    "ratio": ratio})
    return sorted(out, key=lambda x: -x["n"])


def print_measure_events(spec: str, app_cmd: str, out_dir: str,
                         analysis_root: str | None = None) -> int:
    """「我想测 X」-> 一次采集 + 一张可读的表, 一条命令。

    ★这条命令存在的理由: 2026-10-27 跑通自配事件之后, **能力有了而入口没有** ——
      要用就得手拼 `--aic-metrics=Custom:0x9,0xc`、自己记住事件号、自己去产物里找
      `aic_r9` 是哪个事件。能力与可用之间差的就是这一段, 而手拼的那条路每次都要
      重走一遍 (`01` §3.1c 的"登记在册的是清单, 接上才是菜单"说的是同一件事)。
    """
    ids, names, probs = resolve_events(spec, analysis_root)
    print(f"=== 自配 PMU 事件采集 (B 路径, 01 §3.1b/§3.1c) ===")
    print("  要测的事件:")
    for h in ids:
        print(f"      0x{h:<6s} {names.get(h, '★表里查不到这个号')}")
    if probs:
        print("  ★拒跑, 先解决这些:")
        for p in probs:
            print(f"      - {p}")
        #  少采一个事件的表与采全了的表长得一样, 事后分不出"没要"与"要了没给" ——
        #  所以这里拒跑而不是跳过解不开的 token。
        return 2
    if not ids:
        print("  没有解析出任何事件号。"); return 2
    r = run_bpath_events(app_cmd, ids, out_dir)
    if r["rc"] != 0:
        print(f"\n★msprof 退出码 {r['rc']} —— 采集失败, 下面的回读不作数。")
    #  ★不拿退出码判成功: `msprof op` 拒参数之后 rc 仍是 0 (`01` §3.1b)。顶层 msprof
    #    未必同病, 但判据不该建立在"这一条不会"上 —— 以**产物里有没有原始计数列**为准。
    print()
    rc = print_event_readback(out_dir)
    counts = bpath_event_counts(out_dir)
    if not counts:
        print("\n★产物里没有可汇总的原始计数 —— 采集没成, 或算子一个都没被采到。")
        return 2 if rc == 0 else rc
    cols = sorted({c for d in counts for c in d["cols"]})
    w = max([12] + [len(c) + 2 for c in cols])
    print(f"\n=== 按算子汇总 — 原始计数 (每列取均值) ===")
    print(f"{'算子':<34s}{'次数':>6s}" + "".join(f"{c:>{w}s}" for c in cols))
    for d in counts:
        cells = "".join((f"{d['cols'][c]:>{w},.0f}" if d["cols"].get(c) is not None
                         else f"{'-':>{w}s}") for c in cols)
        print(f"{d['op'][:33]:<34s}{d['n']:>6d}{cells}")
    #  ★分母不占事件槽 (一次只能点 8 个, 拿一个换分母就只剩 7 个) —— 自配档的产物里
    #    aic/aiv_total_cycles 照常存在, 所以比率是白送的, 直接出第二张表。
    if any(d.get("ratio") for d in counts):
        print(f"\n=== 同一份数据 — 占 total_cycles 的比例 ===")
        print(f"{'算子':<34s}" + "".join(f"{c:>{w}s}" for c in cols))
        for d in counts:
            cells = "".join((f"{d['ratio'][c]:>{w}.3f}" if d.get("ratio", {}).get(c) is not None
                             else f"{'-':>{w}s}") for c in cols)
            print(f"{d['op'][:33]:<34s}{cells}")
        dn = sorted({k for d in counts for k in d.get("den", {})})
        print(f"  分母: {', '.join(dn)} (同一份 csv 里就有, **不占事件槽**)")
        print("  ★按该算子所有行的**总和之比**算, 不是逐行比率再取均值 —— "
              "各次 launch 时长不等时两者不相等。")
        print("  ★aic 与 aiv 配各自的 total_cycles: 两侧不是同一个量且 block 口径不同 "
              "(`03` §2), 混用会差一个核数倍。")
    print("\n  列名对照:")
    for c in cols:
        h = c.split("_r", 1)[1].lower()
        print(f"      {c:<12s} = 事件 0x{h}  {names.get(h, '★表里查不到这个号')}")
    print("\n★两条读表纪律:")
    print("  (1) 上面第一张表是**原始计数**, 解析侧对 `Custom:` 不做任何派生。"
          "事件名里带 `ratio` 的 (如 0x416 `mac_fp_ratio`) **给的也是计数**, "
          "那是派生指标的名字, 照字面读会把 27 万读成比率。")
    print("  (2) 跨两次采集的计数不可直接相除 —— 负载与时钟窗口都不同 "
          "(时钟 16.98 ppm 漂移, 见 `03` §9.26b)。要比就在同一次采集里比。")
    return 0


def selftest_events() -> int:
    """自配事件入口的判据负例自检。"""
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'OK ' if ok else '★未过'} {name}" + ("" if ok else f"  得到 {got!r}"))

    names = aic_event_names()
    rev = aic_event_ids()
    #  表读不到时只验不依赖表的那几条 —— "表没读到"与"判据不成立"必须分开报。
    if not names:
        print("  ★读不到 events 表 (本机没有 analysis 目录), 仅验不依赖表的判据")
    else:
        ck("E1 名字解得出号 (0x9 的名字反查回 9)",
           rev.get(names["9"]), ["9"])
        ck("E2 反向表的值是**列表**不是字符串 (重名不丢号)",
           isinstance(next(iter(rev.values())), list), True)
        ids, _, probs = resolve_events("0x9,0xc")
        ck("E3 纯事件号解析", ids, ["9", "c"])
        ck("E3b 纯事件号不报问题", probs, [])
        ids, _, _ = resolve_events(names["9"] + ",0xc")
        ck("E4 名字与号可混写", ids, ["9", "c"])
    #  —— 负例: 这四条是工具"看起来在工作但给的是错东西"的四种形态 ——
    _, _, p = resolve_events("no_such_event_name_xyz")
    ck("E5 负例: 查不到的事件名必须报问题 (不能静默跳过)", bool(p), True)
    ids, _, _ = resolve_events("no_such_event_name_xyz")
    ck("E5b 负例: 查不到的 token 不进号列表", ids, [])
    _, _, p = resolve_events("0x9,0x9")
    ck("E6 负例: 重复事件号必须报 (下发两次只得一列)", any("重复" in x for x in p), True)
    _, _, p = resolve_events("0xffff")
    ck("E7 负例: 表外事件号必须报 (下发会成功但解不出名字)",
       any("不在本机" in x for x in p), True)
    ck("E8 A 路径拒收 Custom 的理由串指向 §3.1c", "§3.1c" in _CUSTOM_A_PATH_WHY, True)
    #  ★负例: 超过上限要**提前**拒。不拦就是白跑一趟采集 (msprof 会在参数校验那一步拒,
    #    但那时负载已经准备好了); 而上限是实测扫出来的 8, 不是抄来的。
    _many = ",".join("0x%x" % (i + 1) for i in range(CUSTOM_EVENT_MAX + 1))
    _, _, p = resolve_events(_many)
    ck(f"E9 负例: 超过 {CUSTOM_EVENT_MAX} 个事件必须提前拒",
       any("最多点" in x for x in p), True)
    _, _, p = resolve_events(",".join("0x%x" % (i + 1) for i in range(CUSTOM_EVENT_MAX)))
    ck(f"E9b 正例: 恰好 {CUSTOM_EVENT_MAX} 个不许被上限判据拒 (判据过宽会砍掉合法用法)",
       any("最多点" in x for x in p), False)
    #  —— 比率算法: 用合成 csv 验, 不依赖真机产物 ——
    import tempfile as _tf
    _hdr = "Op Name,aic_total_cycles,aic_r9,aiv_total_cycles,aiv_r9\n"
    #  两行**时长悬殊**: 逐行比率再取均值 = (1/1 + 1/100)/2 = 0.505,
    #  总和之比 = (1+1)/(1+100) ≈ 0.0198。两者差 25 倍, 足以把算法分开。
    _body = "op,1,1,0,0\nop,100,1,0,0\n"
    with _tf.NamedTemporaryFile("w", suffix=".csv", delete=False, dir=_tf.mkdtemp()) as _f:
        _f.write(_hdr + _body)
        _p = _f.name
    _dir = os.path.dirname(_p)
    os.rename(_p, os.path.join(_dir, "op_summary_selftest.csv"))
    _r = bpath_event_counts(_dir)
    ck("E10 比率按总和之比, 不是逐行比率取均值 (各次 launch 时长不等时两者差 25 倍)",
       round(_r[0]["ratio"]["aic_r9"], 4) if _r else None, round(2 / 101, 4))
    ck("E10b aiv 分母为 0 时给 None 而不是 0.0 (0.0 会被读成'这条 pipe 完全空闲')",
       _r[0]["ratio"]["aiv_r9"] if _r else "没读到", None)
    print(f"  —— {('全部通过' if not bad else str(bad) + ' 条未过')}")
    return 1 if bad else 0


def print_bpath_lint(csv_path: str) -> int:
    res = bpath_lint(csv_path)
    mark = {"hit": "★命中", "miss": "  未命中", "na": "  查不了",
            "unobs": "★不可观测", "blind": "★什么也没看到"}
    print(f"=== B 路径体检 {csv_path} ===")
    for tag, verdict, what, ev, cov in res:
        print(f"  {mark.get(verdict, verdict)}  [{tag}] {what}")
        if ev:
            print(f"            {ev}")
        if cov["skipped"]:
            sk = ", ".join(f"{k} {v}" for k, v in cov["skipped"].items())
            print(f"            看见范围: 看 {cov['seen']} 行, 跳过 {sk}")
    tal = {}
    for r in res:
        tal[r[1]] = tal.get(r[1], 0) + 1
    name = {"hit": "命中", "miss": "未命中", "na": "查不了",
            "unobs": "★不可观测", "blind": "★什么也没看到"}
    # ★四档以上都要单独计, 不能把 blind/unobs 并进"未命中" ——
    #   那正是本轮要堵的病: 把"没看到"报成"没问题"。
    print(f"\n  {len(res)} 条判据: "
          + ", ".join(f"{name.get(k, k)} {v}" for k, v in sorted(tal.items())))
    print("  ★'未命中'只说明**这一份产物**没发生, 不说明该失真点不存在;")
    print("    04 篇的 28 条里多数要看原始 .data 或跨报表对账, 这里只收能从 csv 直接判的。")
    #  ★2026-10-29 修: dd2f01d 把计数改成上面的 `tal` 字典时删掉了
    #    `nh = sum(1 for r in res if r[1] == "hit")`, 却把这行 `return 1 if nh else 0`
    #    留在原地 —— 于是本函数**每次调用都在最后一行抛 NameError**, 打印全做完才崩。
    #    取值不是猜的, 是从 dd2f01d 的 diff 里恢复的 (那一行被删前就是"命中条数")。
    #    `core/pycheck` 当场抓住了它; 它存在的理由正是这一类"要跑到那一行才报"的错。
    return 1 if tal.get("hit", 0) else 0


def selftest_rc_sub() -> int:
    """`sub_block_id` 分路函数的负例自检 (expdesign E14: 分组函数也是判据)。

    正例是实测取值空间, 负例是它赌的命名约定被打破的四种形态。
    """
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    ck("实测取值空间全部可分路 (11028 份 csv 只有这三种)",
       rc_sub_unknown(["cube0", "vector0", "vector1"]), [])
    ck("★新核别名不含 cube/vector 时必须报出来 (否则该行被两条分路同时丢掉)",
       rc_sub_unknown(["cube0", "vector0", "aicpu0"]), ["aicpu0"])
    ck("★多个未知取值全列出且去重排序", rc_sub_unknown(["dsa1", "aicpu0", "dsa1"]),
       ["aicpu0", "dsa1"])
    ck("大小写不影响归类 (contains 用的就是 case=False)",
       rc_sub_unknown(["CUBE0", "Vector1"]), [])
    ck("★空值与 nan 不算未知取值 (缺值是另一回事, 不该混进分路告警)",
       rc_sub_unknown(["cube0", None, "nan", ""]), [])
    ck("★编号变了但词根还在就不该报 (守卫不许宽到把正常扩编也拦下)",
       rc_sub_unknown(["cube1", "vector2", "vector3"]), [])
    print(f"sub_block_id 分路函数自检: 6 条, {bad} 条未过")
    return 1 if bad else 0


def traffic_duty(model_bytes: float, occupancy_us: float,
                 active_bw_gbs: float = None, measured_bytes: float = None):
    """占空比判读。返回 (verdict, 说明)。verdict in {"saturated","model_suspect","impossible"}。

    ★这条判据的要点是**默认归因的方向**: 占空比远低于 1 时先怀疑模型, 不先怀疑硬件 (P18)。
    给了 measured_bytes (msprof 自报的片上搬运量) 就直接对账, 比走 active_bw 更硬。
    """
    if measured_bytes:
        ratio = model_bytes / measured_bytes if measured_bytes else float("inf")
        if not 0.9 <= ratio <= 1.1:
            return "model_suspect", (
                f"**模型与实测搬运量对不上**: 模型 {model_bytes/1e9:.3f} GB vs "
                f"msprof {measured_bytes/1e9:.3f} GB (比值 {ratio:.2f})。"
                f"先查模型的形状参数是否就是采集时用的那一组 —— 比值常常恰好等于某一维的倍数 "
                f"(本仓实撞: 13.67 = 头数 28/2)。对不上之前, 不要据此谈硬件行为。")
        model_bytes = measured_bytes
    if not (occupancy_us and active_bw_gbs):
        return "model_suspect", "缺 occupancy 或 active_bw, 判不了占空比"
    duty = model_bytes / (occupancy_us * 1e-6 * active_bw_gbs * 1e9)
    if duty > 1.15:
        return "impossible", (
            f"占空比 {duty:.2f} **大于 1**: 模型字节超过该通路在占用时间内可能搬运的上限。"
            f"模型偏大, 或 active_bw 取错了列。")
    if duty >= 0.85:
        return "saturated", (
            f"占空比 {duty:.2f} —— 该通路在占用期间**基本满负荷**, 是吞吐受限。"
            f"减字节才有用; 改调度只能提高它的占用比例, 动不了这段时间本身。")
    return "model_suspect", (
        f"占空比 {duty:.2f} 远低于 1。**第一嫌疑是模型的形状参数与采集时不一致** (P18), "
        f"不是'通路被按住等待'。先拿 msprof 自报的片上搬运量对账 (L0C_to_GM / GM_to_UB 等); "
        f"对上了才允许往'等待'上解释。")


def selftest_traffic_duty() -> int:
    """占空比判读的负例自检 (不占卡)。"""
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    #  实测口径 (h16b, 同一份产物): fixpipe 10.015 GB / 193674 us / active_bw 48.16 GB/s
    #  ★写这条自检时我自己又错配了一次口径: 拿 h15 的 active_bw (43.62) 配 h16b 的占用,
    #    算出 duty 1.19 被判 impossible。同口径取数后是 1.07。
    #    在**检验口径错配的判据里**再错配一次 —— 正好说明这条判据该有自检。
    v, _ = traffic_duty(10.015e9, 193674.0, 48.16)
    ck("本算子的 fixpipe 判为满负荷 (h16b: 10.015GB / 193.7ms / 48.16GB/s)", v, "saturated")
    #  ★F107 的那次: heads 写成 2, 模型小 14 倍
    v2, t2 = traffic_duty(10.015e9 / 14, 193674.0, 48.16)
    ck("★负例: 模型小 14 倍时必须判'先怀疑模型'而不是'通路在等'", v2, "model_suspect")
    ck("★说明里要点名先查形状参数", "形状参数" in t2, True)
    v3, _ = traffic_duty(10.015e9 * 3, 193674.0, 48.16)
    ck("★负例: 占空比大于 1 要判 impossible (模型或列取错)", v3, "impossible")
    #  直接对账优先于 active_bw
    v4, t4 = traffic_duty(10.015e9 / 14, 193674.0, 48.16, measured_bytes=10.015e9)
    ck("★给了实测搬运量时, 对不上就直接判模型可疑", v4, "model_suspect")
    ck("★对账说明里要给出比值", "13." in t4 or "14." in t4, True)
    v5, _ = traffic_duty(10.015e9, 193674.0, 48.16, measured_bytes=10.015e9)
    ck("模型与实测吻合时回到满负荷判定", v5, "saturated")
    ck("★负例: 缺 active_bw 时不许瞎判", traffic_duty(1e9, 0, None)[0], "model_suspect")
    print(f"占空比判读自检: 8 条, {bad} 条未过")
    return 1 if bad else 0


def selftest_wait_id_cols() -> int:
    """每标志等待时间列的**发现式**采集自检 (造表, 不占卡)。

    2026-10-27 实撞: 此前这些列按编号写死 (id14/id5, 来自 gmfr), 而 msprof **只输出
    该内核实际用到的编号**。VC-Attention 用 8/9/10/15, 于是 abdiff 的 wait_us_delta
    恒为空字典且不报错 —— 候选 h16 的 O_READY 等待动了 535 万周期, 工具一个字没说。
    """
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    m = lambda c: bool(_WAIT_ID_RE.match(c))
    ck("本算子的编号 (8/9/10/15) 全部认得",
       [m(f"{s_}_scalar_wait_id{i}_time(us)") for s_, i in
        (("aiv", 8), ("aic", 9), ("aiv", 10), ("aic", 15))], [True] * 4)
    ck("别的算子的编号 (gmfr 的 5/14) 同样认得 —— 发现式不该偏向某一个算子",
       [m("aiv_scalar_wait_id14_time(us)"), m("aiv_scalar_wait_id5_time(us)")], [True, True])
    ck("多位编号也认 (编号不止一位数)", m("aic_scalar_wait_id123_time(us)"), True)
    ck("★负例: 不带编号的总等待列不归这一族 (它在 WAIT_COLS 里, 重复收会双计)",
       m("aic_scalar_wait_time(us)"), False)
    ck("★负例: 等待**比**不是等待**时间**, 两族不许混",
       m("aic_cube_wait_ratio"), False)
    ck("★负例: stall 列不是跨核标志等待", m("aic_scalar_cube_stall_time(us)"), False)
    ck("★负例: 编号位置缺数字不匹配 (防正则放宽到把别的列吃进来)",
       m("aic_scalar_wait_id_time(us)"), False)
    ck("★负例: 前缀必须是 aic/aiv", m("cpu_scalar_wait_id8_time(us)"), False)

    #  差分层: 解析层收对了, 差分层也不能再丢一次 (2026-10-27 两层都丢过)。
    a = {"aiv_scalar_wait_id10_time(us)": 100.0, "aic_scalar_wait_time(us)": 1.0}
    b = {"aiv_scalar_wait_id10_time(us)": 60.0, "aic_scalar_wait_time(us)": 1.0}
    keys = list(dict.fromkeys(list(WAIT_COLS) + [k for k in a if k in b]))
    d = {c: round(b[c] - a[c], 2) for c in keys
         if isinstance(a.get(c), (int, float)) and isinstance(b.get(c), (int, float))}
    ck("★写死列表之外的键必须一起算差 (否则'解析层收对了, 差分层又丢一次')",
       d.get("aiv_scalar_wait_id10_time(us)"), -40.0)
    print(f"每标志等待列自检: 9 条, {bad} 条未过")
    return 1 if bad else 0


def selftest_apath() -> int:
    """A 路径自洽判据的负例自检 (造表, 不占卡)。"""
    import tempfile
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    def run(header, rows_, freq=1800.0):
        fd, p = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows_:
                f.write(",".join(str(x) for x in r) + "\n")
        try:
            return {t: v for t, v, _w, _e, _c in apath_lint(p, freq_mhz=freq)}
        finally:
            os.unlink(p)

    H1 = ["block_id", "aic_time(us)", "aic_total_cycles"]
    ck("A1 一行一个 block: cycles = freq*time 时判 miss (不除核数)",
       run(H1, [[0, 10.0, 18000.0]])["A1"], "miss")
    ck("★A1 若按'除核数'读则必须命中 (这条判据分得开 A/B 两种粒度)",
       run(H1, [[0, 10.0, 18000.0 * 24]])["A1"], "hit")
    H2 = ["aic_read_hit_rate(%)", "aic_r0_read_cache_hit", "aic_r0_read_cache_miss_allocate",
          "aic_r1_read_cache_hit", "aic_r1_read_cache_miss_allocate"]
    ck("★A2 读命中率要按 r0+r1 合并算 —— 只用 r0 会把自洽的产物报成对不上",
       run(H2, [[75.0, 30, 10, 45, 15]])["A2"], "miss")
    ck("A2 真对不上时命中", run(H2, [[90.0, 30, 10, 45, 15]])["A2"], "hit")
    ck("★缺列判 na 而不是 miss (把'考不动'说成'没问题'比没有体检更糟)",
       run(["block_id"], [[0]])["A1"], "na")
    H3 = ["aic_cube_ratio", "aic_write_hit_rate(%)"]
    ck("A3 比值 ≤1 且百分率 ≤100 时判 miss", run(H3, [[0.5, 99.9]])["A3"], "miss")
    ck("★A3 要按列名分上界: _ratio 比 1, _rate(%) 比 100",
       (run(H3, [[1.5, 99.9]])["A3"], run(H3, [[0.5, 101.0]])["A3"]), ("hit", "hit"))
    print(f"A 路径自洽体检自检: 7 条, {bad} 条未过")
    return 1 if bad else 0


def selftest_bpath() -> int:
    """判据负例自检 (不占卡, 造表)。"""
    import tempfile
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    def run(header, rows, freq=1800.0):
        fd, p = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        try:
            return {t: v for t, v, _w, _e, _c in bpath_lint(p, freq_mhz=freq)}
        finally:
            os.unlink(p)

    # ★`Accelerator Core` 必须在表头里: B7 的除数随核类型变 (AIV 行再除 2),
    #   夹具不带核类型列时 `_bp_core_kind` 会按 aiv_time 有值回退判成 AIV, 于是正例变成命中。
    #   真实产物两族都带显式标签 (kernel_details 有 Accelerator Core, op_summary 有 Task Type),
    #   所以这是夹具欠规格而不是判据过宽 —— 回退本身对 B14/B15 仍然必要。
    H = ["Start Time(us)", "Duration(us)", "aicore_time(us)", "aic_total_cycles", "Block Num",
         "aiv_time(us)", "aiv_total_cycles", "Mix Block Num", "aic_mac_ratio",
         "Accelerator Core"]
    # 正例: 恒等式成立 (10us * 1800MHz * 24 核 = 432000 cycles)
    good = ["1.0\t", 11.0, 10.0, 432000, 24, 10.0, 864000, 48, 0.5, "AI_CORE"]
    r = run(H, [good])
    ck("B7 恒等式成立时判 miss", r["B7"], "miss")
    ck("B8 AIV 侧同理", r["B8"], "miss")
    ck("B9 结尾制表符必须被抓住 (它不报错, 只会悄悄错)", r["B9"], "hit")
    ck("B10 Duration != aicore_time 时命中", r["B10"], "hit")
    ck("B11 ratio<=1 时判 miss", r["B11"], "miss")
    # 负例: 把 cycles 改成"各核之和"的读法 -> 必须命中
    r2 = run(H, [["1.0\t", 11.0, 10.0, 432000 * 24, 24, 10.0, 864000, 48, 0.5, "AI_CORE"]])
    ck("★B7 若按'各核之和'读则必须命中 (这条判据分得开两种读法)", r2["B7"], "hit")
    # 非满核: Block Num=20 也必须成立 (本仓真有两行)
    r3 = run(H, [["1.0\t", 11.0, 10.0, 10.0 * 1800 * 20, 20, 10.0, 10.0 * 1800 * 40, 40, 0.5, "AI_CORE"]])
    ck("B7/B8 非满核 (Block Num=20) 同样成立 —— 除数不是固定核数",
       (r3["B7"], r3["B8"]), ("miss", "miss"))
    r4 = run(H, [["1.0", 10.0, 10.0, 432000, 24, 10.0, 864000, 48, 1.5, "AI_CORE"]])
    # ★AIV 行的除数是 Block Num/2 (910B2 每 AIC 配 2 AIV)。
    #   2026-10-22 实测: 36194 个 AIV 行上比值**恰为 0.5 零例外**。
    #   不分核类型时这一条会把 36194 行全报成失真 —— 那不是失真, 是除数取错了。
    rv7 = run(H, [["1.0\t", 11.0, 10.0, 10.0 * 1800 * 16, 32, 10.0, 864000, 48, 0.5,
                   "AI_VECTOR_CORE"]])
    ck("★B7 在 AIV 行上按 Block Num/2 换算 (32 核的 AIV 行, cycles 只够 16 核) -> miss",
       rv7["B7"], "miss")
    rv8 = run(H, [["1.0\t", 11.0, 10.0, 10.0 * 1800 * 32, 32, 10.0, 864000, 48, 0.5,
                   "AI_VECTOR_CORE"]])
    ck("★B7 的 AIV 换算不得把真失真也放过 (同一行按 32 核算就该命中)", rv8["B7"], "hit")
    ck("B9 无制表符时判 miss", r4["B9"], "miss")
    ck("B10 两列相等时判 miss", r4["B10"], "miss")
    ck("B11 ratio>1 时命中", r4["B11"], "hit")
    # 缺列必须判 na, 不许当成通过
    r5 = run(["Duration(us)"], [[1.0]])
    ck("★缺列判 na 而不是 miss (把'没数据'说成'没问题'比没有体检更糟)",
       (r5["B7"], r5["B9"], r5["B11"]), ("na", "na", "na"))
    # ---- 看见范围 / blind 守卫 (2026-10-22 加) ----
    #   动因: B7 的恒等式在 76 份上"成立", 而反例被它自己的 `if not t: continue`
    #   全过滤掉了。落地当场又撞到第二例: B8 在一份 op_summary 上报 miss 而 seen=0。
    def cov(header, rows_, tag):
        for t, _v, _w, _e, c in bpath_lint(_mk(header, rows_)):
            if t == tag:
                return c
        return None

    def _mk(header, rows_):
        fd, q = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(",".join(header) + "\n")
            for rr in rows_:
                f.write(",".join(str(x) for x in rr) + "\n")
        return q

    # 列都在但每一行都被跳过 -> blind, 不得判 miss
    blind_rows = [["1.0\t", 11.0, 10.0, 432000, 24, 0, 0, 0, 0.5, "AI_CORE"]]
    rb = {t: v for t, v, _w, _e, _c in bpath_lint(_mk(H, blind_rows))}
    ck("★B8 列都在但每行都被跳过 -> blind 而不是 miss "
       "(说'成立'而 seen=0, 正是 B7 混过 76 份的那个病)", rb["B8"], "blind")
    ck("同一份里 B7 有数可看, 仍判 miss (blind 守卫不得误伤)", rb["B7"], "miss")
    c8 = cov(H, blind_rows, "B8")
    ck("看见范围如实记: B8 看 0 行, 跳过 1 行", (c8["seen"], sum(c8["skipped"].values())), (0, 1))
    c7 = cov(H, blind_rows, "B7")
    ck("B7 在同一份上看 1 行跳过 0 行", (c7["seen"], sum(c7["skipped"].values())), (1, 0))
    cb = [e for t, _v, _w, e, _c in bpath_lint(_mk(H, blind_rows)) if t == "B8"][0]
    ck("★blind 的证据要说成领域里的话: 哪几列**每行都是 0 或缺失** —— "
       "只报跳过理由的字面会让人去查一个改不了的结构事实",
       ("每一行上都是 0 或缺失" in cb and "aiv_time(us)" in cb), True)
    ck("★缺列判 na 时不得被 blind 守卫改写 (na 与 blind 是两回事: "
       "前者列都没有, 后者列都在而行被跳光)",
       {t: v for t, v, _w, _e, _c in bpath_lint(_mk(["Duration(us)"], [[1.0]]))}["B7"], "na")

    # ---- B13 / B14 (2026-10-22 加) ----
    HD = ["Duration(us)", "aicore_time(us)", "aiv_time(us)", "Accelerator Core"]
    rA = run(HD, [[0.04, 0.03, "NA", "AI_CORE"], [0.06, 0.05, "NA", "AI_CORE"]])
    ck("B13 全是 0.02us 整数倍 -> hit", rA["B13"], "hit")
    rB = run(HD, [[0.045, 0.03, "NA", "AI_CORE"]])
    ck("B13 出现非 20ns 整数倍 -> miss (与 hwts 50MHz 口径不符)", rB["B13"], "miss")
    rC = run(HD, [[0.04, 0.0, 0.03, "AI_VECTOR_CORE"]])
    ck("★B14 纯 AIV 核的 aicore_time=0 **不得命中** —— 那是正确值不是缺失 "
       "(不分核类型时本机 390 行全是这一类)", rC["B14"], "miss")
    rD = run(HD, [[0.04, 0.0, "NA", "AI_CORE"]])
    ck("B14 AIC 行的 aicore_time=0 而 Duration>0 -> hit", rD["B14"], "hit")
    rE = run(HD, [[0.04, 0.0, 0.03, "AI_CORE"]])
    ck("★B14 按 aiv_time 有值也要排除 (Accelerator Core 列可能缺或写法不同)",
       rE["B14"], "miss")
    # ★分类器本身的负例: 看着属于 A 其实属于 B 的那种样本
    ck("★MIX_AIV 必须判 aiv —— 它里面没有 `VECTOR`, 子串匹配会漏掉 (总账 57 的实撞)",
       _bp_core_kind({"Task Type": "MIX_AIV"}), "aiv")
    ck("★MIX_AIC 判 aic", _bp_core_kind({"Task Type": "MIX_AIC"}), "aic")
    ck("★非计算核的类型 (DSA_SQE/AI_CPU/HCCL...) 判空串, 不许混进任一侧",
       [_bp_core_kind({"Task Type": t}) for t in ("DSA_SQE", "AI_CPU", "HCCL", "DVPP")],
       ["", "", "", ""])

    # ---- B15: 核忙时不得长于任务时长 (2026-10-22 加) ----
    HO = ["aicore_time(us)", "Task Duration(us)", "Task Type"]
    ck("B15 全部不越界 -> miss", run(HO, [[5.0, 6.0, "AI_CORE"]])["B15"], "miss")
    rv = run(HO, [[6.6, 6.0, "AI_VECTOR_CORE"], [5.0, 6.0, "AI_CORE"]])
    ck("B15 有一行越界 -> hit", rv["B15"], "hit")
    rz = run(HO, [[0.0, 5.0, "AI_CORE"], [6.6, 6.0, "AI_VECTOR_CORE"]])
    ck("★B15 证据串必须同时给两个分母 (含零值行 / 只数非零行) —— "
       "文档曾写 8.03% 而工具算 1.22%, 两处用的不是同一个分母",
       all(x in [e for t, _v, _w, e, _c in bpath_lint(_mk(HO,
           [[0.0, 5.0, "AI_CORE"], [6.6, 6.0, "AI_VECTOR_CORE"]])) if t == "B15"][0]
           for x in ("1/2 行越界", "1/1 =")), True)
    ck("★B15: aicore_time 全为 0 的产物判 miss 不判 na —— 0 是假值, "
       "写成 `if not a_` 会把'有数据且没越界'说成'考不动'(第一版 68 份只认出 9 份)",
       run(HO, [[0.0, 11.72, "AI_CORE"], [0.0, 5.0, "AI_CORE"]])["B15"], "miss")
    ck("★B15 在 kernel_details 上必须判 na 而不是 miss "
       "(它没有 `Task Duration(us)` 这个列名, 判 miss 等于把'考不动'说成'没问题')",
       run(["Duration(us)", "aicore_time(us)"], [[6.0, 5.0]])["B15"], "na")

    # ---- B12: D7 的可观测性 (2026-10-22 加) ----
    #   动因: B11 此前标注来源 "D7", 而 D7 影响的三支列与 B11 查的占比列毫无交集。
    #   把 D7 单独做成 B12 之后才看见真正的结论: 它在导出的 csv 上**不可观测**。
    r6 = run(["aic_main_mem_read_bw(GB/s)", "Duration(us)"], [[1.5, 10.0]])
    ck("★被影响列在场而分母列不在 -> unobs (不是 na) —— "
       "na 再采一份可能就有了, unobs 再采多少份都没有",
       r6["B12"], "unobs")
    r7 = run(["aic_main_mem_read_bw(GB/s)", "aic_mte2_ratio", "Duration(us)"],
             [[1.5, 0.0, 10.0]])
    ck("两列同时在场且分母为 0 -> hit (D7 的触发形态)", r7["B12"], "hit")
    r8 = run(["aic_main_mem_read_bw(GB/s)", "aic_mte2_ratio", "Duration(us)"],
             [[1.5, 0.3, 10.0]])
    ck("两列同时在场但分母非 0 -> miss", r8["B12"], "miss")
    r9 = run(["Duration(us)"], [[10.0]])
    ck("一个被影响列都没有 -> na", r9["B12"], "na")
    ck("★B11 改归不变量类之后不得再被当成失真检测 (否则 41 份上成立会被报成'未命中')",
       "B11" in _BP_IDENTITY, True)

    # ---- 语料级汇总 (2026-10-22 加) ----
    import csv as _csv2
    d = tempfile.mkdtemp()
    for i, body in enumerate(([["1.0\t", 11.0, 10.0, 432000, 24, 10.0, 864000, 48, 0.5, "AI_CORE"]],
                              [["1.0", 10.0, 10.0, 432000, 24, 10.0, 864000, 48, 0.5, "AI_CORE"]])):
        sub = os.path.join(d, f"r{i}", "ASCEND_PROFILER_OUTPUT")
        os.makedirs(sub)
        with open(os.path.join(sub, "kernel_details.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = _csv2.writer(f); w.writerow(H)
            for row in body:
                w.writerow(row)
    cp = bpath_corpus([d])
    ck("语料扫描扫到两份", len(cp["files"]), 2)
    ck("B9 一份带制表符一份不带 -> hit/miss 各一",
       (cp["agg"]["B9"].get("hit"), cp["agg"]["B9"].get("miss")), (1, 1))
    ck("★B7/B8 的极性与失真检测相反 (miss = 恒等式成立), 不得混进'未被证实'那一类",
       ("B7" in _BP_IDENTITY, "B9" in _BP_IDENTITY), (True, False))
    cp2 = bpath_corpus([tempfile.mkdtemp()])
    ck("根下没有 csv 时不报'全部通过' (空语料不是阴性证据)", cp2["files"], [])
    print(f"B 路径体检自检: 41 条, {bad} 条未过")
    return 1 if bad else 0


# ------------------------------------------------------------- PROF 目录清点
#  **这个 `.data` 是哪条通道的?** —— 本仓此前没有一处能回答, 看 PROF 目录只能逐个猜。
#
#  通道清单来自 `libprofimpl.so` 的 `.rodata` (docs/msprof-re/02 §9.3): 一段连续的
#  27 个产物文件名。清单是**反汇编取的, 不是手抄的**, 但它仍可能不全 ——
#  真目录里就有不在清单上的 `lpmInfoConv.data`。所以本函数的默认动作是
#  **把不认识的也列出来**, 而不是只报认识的那些: 一份只显示"已知项"的清单,
#  会让新出现的通道静默消失 (同 §3.55 "手维护的分类会漏掉自己没想到的那一类")。

_PROF_CHANNELS = {
    "ddr": "DDR 控制器 PMU", "hbm": "HBM PMU", "llc": "LLC (L2/L3) PMU",
    "dvpp": "DVPP 编解码", "nic": "网卡", "pcie": "PCIe", "hccs": "HCCS 片间互联",
    "ub": "UB", "roce": "RoCE", "netdev_stats": "网络设备统计",
    "npu_mem": "NPU 内存", "npu_module_mem": "NPU 分模块内存", "qos": "QoS",
    "aicpu": "AI CPU", "tscpu": "TS CPU", "ts_track": "TS 任务轨迹 (旧格式)",
    "aicore": "AI Core PMU (旧格式)", "hwts": "HWTS 任务时间 (旧格式)",
    "training_trace": "训练迭代轨迹", "socpmu": "SoC 级 PMU", "l2_cache": "L2 Cache",
    "stars_soc": "STARS SoC 级", "stars_block": "STARS block 级",
    "stars_soc_profile": "STARS SoC profile", "ffts_profile": "FFTS+ PMU (本机主路径)",
    "adprof": "adprof", "ai_ctrl_cpu": "AI 控制 CPU",
    "host_syscall": "host 系统调用", "host_pthreadcall": "host 线程调用",
    "host_disk": "host 磁盘",
}

#  ★2026-10-21 修: 第一版只认 `*.data*`, 于是**整整三族文件被静默跳过** ——
#    `unaging.additional.hash_dic.slice_0` / `all_file.complete` / `sample.json`。
#    后果不止是少报: 上一轮据此得出"`.complete` 只出现在 lpmInfoConv 上 (39/39)",
#    那个 39/39 的整齐**完全是正则造出来的** —— 带 `.complete` 的另外两族根本没进统计。
#    判据把一部分数据排除在外时, 剩下那部分会显得异常干净。
#
#  真目录里实际存在的四族:
#    <chan>.data[.<dev>][.slice_N][.done|.complete]          设备/host 通道产物
#    [aging.|unaging.]<kind>.<name>.slice_N[.done|.complete]  带老化标记的结构化流
#    all_file.complete                                        **目录级**完成标记
#    sample.json[.done]                                       本次运行的**实际采集配置**
_PROF_SUFFIX = re.compile(
    r"^(?P<name>.+?)(?:\.data)?(?:\.(?P<dev>\d+))?"
    r"(?:\.slice_(?P<slice>\d+))?(?:\.(?P<mark>done|complete))?$")
_PROF_AGING = re.compile(r"^(?P<age>aging|unaging)\.(?P<rest>.+)$")


def prof_inventory(prof_dir: str) -> dict:
    """清点一个 PROF 目录: 每条通道几个切片、多大、有没有完成标记、认不认识。"""
    out = {}
    for dp, _dn, fns in os.walk(prof_dir):
        for fn in fns:
            m = _PROF_SUFFIX.match(fn)
            if not m:
                continue
            name = m.group("name")
            am = _PROF_AGING.match(name)
            aging = bool(am) and am.group("age") == "aging"
            base = am.group("rest") if am else name
            e = out.setdefault(base, {"slices": 0, "bytes": 0, "done": 0, "complete": 0,
                                      "aging": False, "devs": set(), "nums": {}})
            e["aging"] |= aging
            if m.group("dev"):
                e["devs"].add(m.group("dev"))
            mark = m.group("mark")
            if mark:
                e[mark] += 1
            else:
                e["slices"] += 1
                try:
                    e["bytes"] += os.path.getsize(os.path.join(dp, fn))
                except OSError:
                    pass
            if m.group("slice") is not None:
                # ★切片编号按 (子目录, 设备号) 分桶, **不跨桶合并** —— 合并会把
                #   device_0 的 slice_0..2 与 device_1 的 slice_0..2 并成一串,
                #   任何一侧的真实断号都会被另一侧填上而消失。
                k = (os.path.relpath(dp, prof_dir), m.group("dev") or "")
                e["nums"].setdefault(k, set()).add(int(m.group("slice")))
    return out


# -------------------------------------------------- 第三条丢数通路: 文件被淘汰删除
#  §9.16 的两条判据查的是"记录有没有进/出环形缓冲"。但还有第三条路:
#  记录写了、读了、**落了盘, 然后文件被删掉** —— read/write 计数一个字都看不出来。
#
#  删除方是 `analysis::dvvp::transport::FileAgeing` (msprof 二进制内)。
#  ★它在**默认日志级别下完全静默**: 2026-10-22 反汇编 msprof(8142256) 定三个调用点,
#    `DlogRecord` 的 level 实参 (`mov w1,#imm`, 常量见 include/base/log_types.h) 分别是
#      `RemoveAgeingFile begin, storagedFileSize/overThdSize`  第265行  w1=0 DLOG_DEBUG
#      `remove fileName:…, filePath:…, fileCount_`             第257行  w1=0 DLOG_DEBUG
#      `remove file:%s failed`                                 第245行  w1=3 DLOG_ERROR
#    **成功的删除是 DEBUG, 只有删除失败才是 ERROR。** 本机 ASCEND_GLOBAL_LOG_LEVEL 未设,
#    slog 里这三个串零命中 —— 于是日志里关于淘汰的唯一记录, 恰好是淘汰**没发生**的时候。
#
#  所以这一条只能从**目录本身**查: `slice_N` 编号断号 = 该切片曾存在而现已不在。
#  实测签名 (两份带 --storage-limit 的真产物, 2026-10-21 采):
#    · 断号**全部落在 `aging.` 前缀的通道上**, `unaging.*` 一个没缺 ——
#      这就是 prof_inventory 一直在解析却从没解释过的 aging 标记的含义: **可被淘汰**;
#    · 缺的是**前缀 0..N**, 存活切片构成后缀, 与 §9.10 实测的"每个 key 内最旧的先删"一致。
#  负样本: 本仓 runs/ 与 cases/ 下 39 个 PROF 目录、429 个通道-目录组合, 断号 0。


def slice_gaps(inv: dict) -> list:
    """从 prof_inventory 的结果里挑出断号的通道。

    返回 [(通道名, 桶, 缺的编号, 现有最大编号, 是不是前缀型)]。
    **前缀型** (缺的恰好是 0..k) 才符合"最旧先删"的签名; 中间断号是另一回事
    (采集中途出错 / 文件被人手动删), 两者处置不同, 所以分开标。
    """
    out = []
    for name, e in sorted(inv.items()):
        for k, s in sorted(e.get("nums", {}).items()):
            mx = max(s)
            miss = sorted(set(range(mx + 1)) - s)
            if miss:
                out.append((name, k, miss, mx, miss == list(range(len(miss)))))
    return out


def print_prof_inventory(prof_dir: str) -> int:
    inv = prof_inventory(prof_dir)
    if not inv:
        print(f"{prof_dir} 下没有 *.data* —— 多半给错了目录 (要给 PROF_xxx 那一层)")
        return 2
    known = [k for k in inv if k in _PROF_CHANNELS]
    unknown = [k for k in inv if k not in _PROF_CHANNELS]
    print(f"=== PROF 目录清点 {prof_dir} ===")
    print(f"{'通道':<22}{'切片':>5}{'字节':>12}{'done':>6}{'cmpl':>6}  说明")
    for k in sorted(inv):
        e = inv[k]
        tag = _PROF_CHANNELS.get(k, "★不在已知清单里")
        if e["aging"]:
            tag = "[aging] " + tag
        print(f"{k:<22}{e['slices']:>5}{e['bytes']:>12}{e['done']:>6}{e['complete']:>6}  {tag}")
    print(f"\n  共 {len(inv)} 条通道有产物: 认识 {len(known)}, **不认识 {len(unknown)}**")
    if unknown:
        print(f"  ★不认识的: {', '.join(sorted(unknown))}")
        print("    清单取自 libprofimpl.so 的 .rodata (02 §9.3), 它不保证全 —— "
              "不认识**不等于**不该出现。")
    nd = [k for k, e in inv.items() if e["slices"] and not e["done"]]
    if nd:
        print(f"  ★有切片但没有 .done 标记的: {', '.join(sorted(nd))} —— 采集可能没正常收尾")
    gaps = slice_gaps(inv)
    if gaps:
        pre = [g for g in gaps if g[4]]
        print(f"\n★切片编号断号 {len(gaps)} 处 —— **这些文件曾经存在, 现在不在了**:")
        for name, k, miss, mx, is_pre in gaps:
            rng = f"{miss[0]}..{miss[-1]}" if len(miss) > 3 else ",".join(map(str, miss))
            tag = "前缀型 (= 最旧先删, --storage-limit 淘汰的签名)" if is_pre else "★中间断号 (不是淘汰)"
            print(f"    {name:<30} {k[0]:<14} 缺 {rng} (共 {len(miss)}), 现有最大 {mx}  {tag}")
        if pre:
            print("  淘汰只删 `aging.` 前缀的通道; 它**在默认日志级别下完全静默** (DEBUG 级),"
                  " 所以目录自身是唯一证据。处置: 调大 --storage-limit (★写的数会被除以 2)。")
        if len(pre) < len(gaps):
            print("  ★中间断号不是淘汰 —— 淘汰按最旧先删, 只会留下前缀型缺口。"
                  "中间缺口要往采集中途出错 / 文件被手动删的方向查。")
    return 0


def selftest_prof_inventory() -> int:
    """判据负例自检 (造目录, 不占卡)。"""
    import tempfile
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    d = tempfile.mkdtemp()
    sub = os.path.join(d, "device_2", "data")
    os.makedirs(sub)
    for fn, sz in (("ffts_profile.data.2.slice_0", 100), ("ffts_profile.data.2.slice_0.done", 0),
                   ("ffts_profile.data.2.slice_1", 50), ("stars_soc.data.2.slice_0", 7),
                   ("aging.api_event.data.slice_0", 9), ("lpmInfoConv.data.2.slice_0", 3),
                   # ★下面三族是第一版正则漏掉的, 它们是这条自检存在的理由
                   ("unaging.additional.hash_dic.slice_0", 11),
                   ("unaging.compact.expand_stream_spec.slice_0.complete", 0),
                   ("all_file.complete", 0), ("sample.json", 13)):
        with open(os.path.join(sub, fn), "wb") as f:
            f.write(b"x" * sz)
    inv = prof_inventory(d)
    ck("切片与字节按通道累加", (inv["ffts_profile"]["slices"], inv["ffts_profile"]["bytes"]), (2, 150))
    ck("完成标记不计入切片数", inv["ffts_profile"]["done"], 1)
    ck("★aging. 前缀被剥掉并单独标记 (否则同一通道会被拆成两行)",
       (inv["api_event"]["aging"], "aging.api_event" in inv), (True, False))
    ck("不在清单里的通道照样列出 (只显示已知项会让新通道静默消失)",
       "lpmInfoConv" in inv, True)
    ck("设备号被收集", inv["stars_soc"]["devs"], {"2"})
    ck("★不含 .data 的那三族必须也被收进来 (第一版正则把它们整族跳过了)",
       all(k in inv for k in ("additional.hash_dic", "compact.expand_stream_spec",
                              "all_file", "sample.json")), True)
    ck("★unaging. 也要剥前缀, 且**不**标成 aging",
       (inv["additional.hash_dic"]["aging"], "unaging.additional.hash_dic" in inv), (False, False))
    ck("带 .complete 的不止一族 (上一轮'只属 lpmInfoConv'正是漏判造出来的)",
       sorted(k for k, e in inv.items() if e["complete"]),
       ["all_file", "compact.expand_stream_spec"])
    # ---- 切片断号 (第三条丢数通路: 文件被淘汰删除) ----
    d2 = tempfile.mkdtemp()
    s2 = os.path.join(d2, "host", "data")
    os.makedirs(s2)
    for fn in ("aging.api_event.data.slice_3", "aging.api_event.data.slice_4",
               "unaging.compact.x.slice_0", "unaging.compact.x.slice_1"):
        open(os.path.join(s2, fn), "wb").close()
    g2 = {r[0]: r for r in slice_gaps(prof_inventory(d2))}
    ck("★前缀型断号被抓住 (slice_0..2 没了 = 最旧先删的签名)",
       (g2["api_event"][2], g2["api_event"][4]), ([0, 1, 2], True))
    ck("连号的通道不报 (否则每份产物都红, 判据等于没有)", "compact.x" in g2, False)
    # 中间断号: 与淘汰的签名不同, 必须分开标
    s3 = os.path.join(tempfile.mkdtemp(), "host", "data")
    os.makedirs(s3)
    for n in (0, 1, 3):
        open(os.path.join(s3, f"aging.api_event.data.slice_{n}"), "wb").close()
    g3 = slice_gaps(prof_inventory(os.path.dirname(os.path.dirname(s3))))
    ck("★中间断号不得被标成前缀型 (它不是淘汰, 处置方向不同)",
       (g3[0][2], g3[0][4]), ([2], False))
    # 分桶: 两个设备各自 slice_0..1, 合并会把真断号填平
    d4 = tempfile.mkdtemp()
    for dev in ("0", "1"):
        sd = os.path.join(d4, f"device_{dev}", "data")
        os.makedirs(sd)
        for n in (0, 1):
            open(os.path.join(sd, f"ffts_profile.data.{dev}.slice_{n}"), "wb").close()
    os.remove(os.path.join(d4, "device_1", "data", "ffts_profile.data.1.slice_0"))
    g4 = slice_gaps(prof_inventory(d4))
    ck("★按 (子目录,设备号) 分桶: device_1 缺 slice_0 不能被 device_0 填平",
       [(r[1][1], r[2]) for r in g4], [("1", [0])])
    print(f"PROF 目录清点自检: 13 条, {bad} 条未过")
    return 1 if bad else 0


# -------------------------------------------------- 采集丢条数对账 (slog, 不在 PROF 目录里)
#  清点 (prof_inventory) 回答"目录里有什么", 这一节回答**"路上丢没丢"** —— 两件事,
#  而丢没丢的证据**一个字节都不在 PROF 目录里**, 全在昇腾 slog (`/root/ascend/log`)。
#
#  两条判据, 来路都是 libprofapi.so 的 .rodata + 调用点反汇编 (2026-10-22 核):
#
#  P1 计数对账 —— `prof_inner_api.cpp:106/109` 每次采集收尾时逐缓冲区打一行
#       `total_size_report [<buf>] read count: R, write count: W`
#     调用点 `w0 = 0x0100001f` = PROFILING(31) | RUN 掩码, 所以落 `run/plog/` 而非
#     `debug/plog/`。R != W = **写进环形缓冲但没落盘**。
#     缓冲区名共 5 个: api_event / compact / additional / variable / AicpuBuffer ——
#     前四个正是 03 里那四条 host 通道, 第五个 (AicpuBuffer) 此前没在任何文档里出现过。
#
#  P2 溢出错误 —— 同库另外几条串, 调用点 `w0=0x1f`(无 RUN 掩码) `w1=3`, 即
#     **module=PROFILING level=ERROR, 落 `debug/plog/`**。命中 = 记录被丢弃。
#     ★同一件事在两层各打一条且级别不同: `Block cycle overflow` 在
#     prof_cann_plugin.cpp:141 是 WARNING(w1=2)。两条都要找, 故 _DROP_HARD 收了六个串。
#     丢弃语义见 02 §9.15: 有界重试后返回失败并丢弃, 不阻塞业务。
#
#  ★为什么 P1 抓不到 P2 的丢弃: 溢出时写入方是**失败返回**的, 那条记录从未计入
#    write count, 于是 R 仍然等于 W。两条判据查的是不同的洞, 缺一不可。
#
#  ★盲检与干净必须分开报 (本模块最该防的误读):
#    P1 一行都没扫到时, 正确结论是"**这次检查什么也看不见**"(日志级别关了 / 给错根 /
#    日志已轮转), **不是**"没丢"。本机 2026-10-22 全量扫 22851 行 R==W 无一例外、
#    三类 overflow 串零命中 —— 那是 CLEAN; 扫到 0 行则是 BLIND, 两者在输出里不同字样。

_DROP_TSR = re.compile(
    r"total_size_report \[([^\]]+)\] read count: (\d+), write count: (\d+)")
#  P2 的串按 libprofapi.so/libprofimpl.so 的 .rodata 原文截取 (去掉 printf 占位符部分)
_DROP_HARD = ("Block buffer overflow, [", "Report buffer overflow, [",
              "Variable block buffer overflow, [", "Block cycle overflow, buffer name:",
              "Cycle overflow, QueueName:", "Variable Block cycle overflow, buffer name:")
_DROP_WARN = ("Block buffer about to overflow", "Variable block buffer about to overflow")


def prof_drops(log_root: str = "/root/ascend/log", since_min: float | None = None) -> dict:
    """扫昇腾 slog 对账采集丢条数。since_min 给了就只看 mtime 在这些分钟内的文件。"""
    import time
    cutoff = (time.time() - since_min * 60) if since_min else None
    files, tsr_lines, buffers = 0, 0, {}
    mismatch, hard, warn = [], [], []
    for sub in ("run/plog", "debug/plog", "run", "debug"):
        d = os.path.join(log_root, sub)
        if not os.path.isdir(d):
            continue
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    if cutoff and os.path.getmtime(p) < cutoff:
                        continue
                except OSError:
                    continue
                files += 1
                try:
                    txt = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for m in _DROP_TSR.finditer(txt):
                    tsr_lines += 1
                    buf, r, w = m.group(1), int(m.group(2)), int(m.group(3))
                    e = buffers.setdefault(buf, {"lines": 0, "read": 0, "write": 0})
                    e["lines"] += 1; e["read"] += r; e["write"] += w
                    if r != w:
                        mismatch.append((os.path.basename(p), buf, r, w))
                for s in _DROP_HARD:
                    if s in txt:
                        hard.append((os.path.basename(p), s))
                for s in _DROP_WARN:
                    if s in txt:
                        warn.append((os.path.basename(p), s))
    if hard:
        verdict = "DROP"
    elif mismatch:
        verdict = "UNDRAINED"
    elif tsr_lines == 0:
        verdict = "BLIND"
    else:
        verdict = "CLEAN"
    return {"log_root": log_root, "files": files, "tsr_lines": tsr_lines,
            "buffers": buffers, "mismatch": mismatch, "hard": hard, "warn": warn,
            "verdict": verdict}


def print_prof_drops(log_root: str, since_min: float | None = None) -> int:
    r = prof_drops(log_root, since_min)
    scope = f"最近 {since_min:g} 分钟" if since_min else "全部历史"
    print(f"=== 采集丢条数对账 {r['log_root']} ({scope}) ===")
    print(f"扫了 {r['files']} 个日志文件, 其中 total_size_report 计数行 {r['tsr_lines']} 条")
    if r["buffers"]:
        print(f"\n{'缓冲区':<16}{'行数':>7}{'read 合计':>12}{'write 合计':>12}  对账")
        for b in sorted(r["buffers"]):
            e = r["buffers"][b]
            ok = "齐" if e["read"] == e["write"] else f"★差 {e['write'] - e['read']}"
            print(f"{b:<16}{e['lines']:>7}{e['read']:>12}{e['write']:>12}  {ok}")
    if r["verdict"] == "BLIND":
        print("\n★BLIND —— 一行计数都没扫到。这**不是**'没丢', 是这次检查什么也看不见。")
        print("  三种可能: (a) 给错 --log-root; (b) PROFILING 模块 INFO 级被关 "
              "(计数行是 INFO, 见 ASCEND_GLOBAL_LOG_LEVEL); (c) 日志已轮转出去。")
        print("  ★排除 (b) 之前不要把本次结果当成阴性证据。")
        return 1
    if r["verdict"] == "DROP":
        print(f"\n★DROP —— 命中 {len(r['hard'])} 处溢出错误 (ERROR 级), **记录已被丢弃**:")
        for f, s in r["hard"][:10]:
            print(f"    {f}: {s}")
        print("  丢弃语义见 02 §9.15: 有界重试后返回失败并丢弃, 业务不阻塞 —— "
              "所以**业务侧一切正常**, 只有 profiling 产物少了东西。")
        print("  处置: 调大 --storage-limit (★写的数会被除以 2, 见 01)。")
        #  ★"减少采集档位"这条建议 2026-10-22 实测后撤掉 (01 §13.16): A 路径的
        #    `--aic-metrics` **不控制采集, 只过滤导出** —— 1 档与 7 档都产 10 个
        #    `DeviceProf*.bin`, 总输出只差 0.23% (1825263 -> 1829404 B), 墙上时间不增。
        #    拿它减压等于没减。
        print("  ★**不要靠减少 --aic-metrics 档位减压** —— 它只过滤导出的 csv, "
              "不减少采集 (1 档与 7 档总输出只差 0.23%)。见 01 §13.16。")
        return 2
    if r["verdict"] == "UNDRAINED":
        print(f"\n★UNDRAINED —— {len(r['mismatch'])} 行 read != write, 写进了环形缓冲但没落盘:")
        for f, b, rr, ww in r["mismatch"][:10]:
            print(f"    {f}: [{b}] read {rr} write {ww}  差 {ww - rr}")
        return 2
    print(f"\nCLEAN —— {r['tsr_lines']} 行计数全部 read == write, 且六类 overflow 串零命中。")
    if r["warn"]:
        print(f"  ☆但有 {len(r['warn'])} 处 'about to overflow' 预警: 这次没丢, 余量已经不多。")
    print("  ★这条阴性结论的可信度来自上面那个行数 —— 行数越少越接近 BLIND。")
    return 0


def selftest_prof_drops() -> int:
    """判据负例自检 (造日志目录, 不占卡, 不读真 slog)。"""
    import tempfile
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    def mk(lines):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "run", "plog")
        os.makedirs(p)
        with open(os.path.join(p, "plog-1_x.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return d

    ok_lines = ["[INFO] PROFILING(1,python3):t [prof_inner_api.cpp:106] (tid:1) "
                f"total_size_report [{b}] read count: {n}, write count: {n}"
                for b, n in (("api_event", 159), ("compact", 61), ("additional", 71),
                             ("variable", 0))]
    ck("全齐 = CLEAN", prof_drops(mk(ok_lines))["verdict"], "CLEAN")
    ck("★空目录 = BLIND 而不是 CLEAN (把'看不见'报成'没丢'是本模块最该防的误读)",
       prof_drops(mk([]))["verdict"], "BLIND")
    bad_lines = list(ok_lines)
    bad_lines[1] = bad_lines[1].replace("read count: 61", "read count: 55")
    r = prof_drops(mk(bad_lines))
    ck("read < write = UNDRAINED", r["verdict"], "UNDRAINED")
    ck("差值报在对的缓冲区上", (r["mismatch"][0][1], r["mismatch"][0][3] - r["mismatch"][0][2]),
       ("compact", 6))
    ck("★溢出错误优先于对账 (溢出的那条从未计入 write, R==W 掩盖不了它)",
       prof_drops(mk(ok_lines + ["[ERROR] PROFILING(1,x):t [prof_inner_api.cpp:96] (tid:1) "
                                 "Block buffer overflow, [compact] capacity: 4194304, "
                                 "read count: 61, write count: 61"]))["verdict"], "DROP")
    ck("about to overflow 只是预警, 不翻 CLEAN",
       prof_drops(mk(ok_lines + ["[WARNING] PROFILING(1,x):t [x:1] (tid:1) "
                                 "Block buffer about to overflow, buffer name: compact, "
                                 "buffer capacity: 4194304, currWriteCursor: 1, "
                                 "currReadCursor: 0, inSize: 64"]))["verdict"], "CLEAN")
    ck("5 个缓冲区名都认 (AicpuBuffer 是第五个, 此前没进过文档)",
       sorted(prof_drops(mk(ok_lines + [
           "[INFO] PROFILING(1,x):t [prof_inner_api.cpp:109] (tid:1) "
           "total_size_report [AicpuBuffer] read count: 3, write count: 3"]))["buffers"]),
       ["AicpuBuffer", "additional", "api_event", "compact", "variable"])
    #  ---- 事件号 -> 事件名 (01 §3.1b / §6.6, 2026-10-27) ----
    _ev = aic_event_names()
    if _ev:
        #  ★手读锚点: 上机实测 `Custom:0x9,0xc` 的产物列是 aic_r9/aic_rc, 这两个号的名字
        #    与实测数吻合 (matmul 上 scalar 忙 44%、mte2 忙 74%)。表读歪了这两条先炸。
        ck("★手读锚点: 0x9 / 0xc 的事件名", (_ev.get("9"), _ev.get("c")),
           ("scalar_busy_cycles", "mte2_busy_cycles"))
        ck("★预置档位 PipeUtilization 展开的 8 个号在表里都查得到",
           [e for e in ("416", "417", "9", "302", "c", "303", "54", "55")
            if e not in _ev], [])
        ck("表规模像一张完整字典而不是几条 (>100 条)", len(_ev) > 100, True)
        #  ★负例: 名字不许带 `0x`, 号不许带 `0x` —— 两边都规范化过才查得动。
        ck("★负例: 键已去掉 0x 前缀 (没去的话 r<hex> 一个都对不上)",
           [k for k in _ev if k.startswith("0x")], [])
        #  ★负例 (2026-10-27 实撞): 该文件里有三张 events 表, CANN 自己是
        #    `DATA = DATA_MAP.get(chip_id, DATA_DEFAULT)` —— **按芯片选一张, 不合并**。
        #    首版合并了, 于是 7 个重叠号在本机 (CHIP_V4_1_0 -> 应落 DATA_DEFAULT)
        #    拿到了 CHIP_V6 的名字。这类错不报错, 只给一个讲得通的错名字,
        #    照着它去采就采到另一个事件而表头写着你要的那个名字。
        #    判据: 默认表里**不许**出现只存在于 V6 表的号。
        _v6 = aic_event_names(chip="CHIP_V6_1_0")
        if _v6:
            #  非循环的判法: **发生合并时 V6 的号会一个不剩地出现在返回表里**,
            #  于是 `V6 - 默认` 为空; 按芯片选表则必然非空。
            #  (写成"默认表不含 V6 独有的号"是循环的 —— "独有"本身就按返回表定义,
            #   那条判据恒真、永远抓不住回归。)
            ck("★负例: 按芯片选表而非合并 (V6 必有默认表没有的号)",
               bool(set(_v6) - set(_ev)), True)
            ck("★重叠号取的是本机该用的那张 (0x1 不是 V6 给的名字)",
               _ev.get("1") != _v6.get("1"), True)
    else:
        print("  (跳过事件名自检: 找不到 analysis 目录或该配置文件)")
    print(f"采集丢条数对账自检: {'11' if _ev else '7'} 条, {bad} 条未过")
    return 1 if bad else 0


# ------------------------------------------------------- 采集配置 (sample.json)
#  `PROF_xxx/host/sample.json` 是这次采集**实际生效的 131 个开关**, 由采集端自己写下
#  (缺口 6 的 `CreateSampleJsonFile`)。它比"我当时敲了什么命令行"可靠 —— 记的是生效值。
#
#  本仓为什么需要它: 测量纪律要求**同法对测**, 而"同法"此前只能靠人记得。
#  两次采集如果 metrics 档不同, per-pipe 列根本不是同一组 PMU 事件 —— 比它们的数字无意义。
#
#  ★`--vs` 比较时**显式忽略四个每次必不同的键** (job_id / result_dir / devices / profMode):
#    不忽略的话每次比较都会报差异, 而一个永远报差异的比较器等于没有比较器。

_PROFCONF_IGNORE = {"job_id", "result_dir", "devices", "profMode", "hostProfiling",
                    "app_parameters", "result_dir_host"}


# ------------------------------------------------- 采集环境指纹
#  ★2026-10-22 由一次连错三轮的事逼出来: 语料里混进了**另一个环境**的 46 份产物,
#    而它们在目录名、档位名、列名上都看不出来 —— 唯一的标志是 `sample.json` 里的
#    `app_dir` 前缀, 以及**多出来的两个键** (`ntsMetrics` / `host_numa_profiling`,
#    当前 msprof 根本不写它们)。
#  后果: 那 46 份是全机**唯一**的 9 事件 `PipeUtilization` 采集, 于是
#    (a) D4 的"前提确实发生"被当成本机结论 —— 实际本机从未发生;
#    (b) 拿本机的芯片配置表去解释那批产物的列名, 怎么对都对不上, 白查一轮。
#  **跨产物下结论之前先分环境** —— 两个环境的产物混在一张表里, 任何"全语料"结论都失真。
_PROF_ENV_MARKER_KEYS = ("ntsMetrics", "ntsPmuEvents", "host_numa_profiling")


def prof_env(prof_dir: str) -> dict:
    """一份产物的采集环境指纹: app_dir 前缀 + 当前工具不写的那些键。

    ★A 路径 (`OPPROF_*`, msopprof 产的) **根本没有采集配置文件** —— 目录里只有
    `<算子>_<哈希>_<id>_<核型>/<n>/` 加 csv 与 `dump/`。所以它**无法从自身判断来源环境**。
    这一条要报出来而不是跳过: 上一版遇到没有配置就 `return {}`, 于是全机 380 个 A 路径
    目录一个都没进分组, 而那正是本仓性能诊断实际读的那一族。
    **一个悄悄跳过半数语料的环境检查, 比没有更误导。**
    """
    cfg = prof_config(prof_dir)
    if not cfg:
        if os.path.basename(prof_dir.rstrip(os.sep)).startswith("OPPROF_"):
            return {"app_root": "<A 路径, 无采集配置>", "marker_keys": (),
                    "metrics": "无 sample.json", "events": 0, "no_config": True}
        return {}
    ad = (cfg.get("app_dir") or "").strip()
    extra = tuple(k for k in _PROF_ENV_MARKER_KEYS if k in cfg)
    return {"app_root": "/".join(ad.split("/")[:3]) if ad.startswith("/") else (ad or "<无>"),
            "marker_keys": extra, "metrics": cfg.get("ai_core_metrics") or "?",
            "events": len([x for x in (cfg.get("ai_core_profiling_events") or "").split(",")
                           if x.strip()])}


def prof_env_scan(roots: list) -> dict:
    """扫多个根下的全部 PROF 目录, 按环境指纹分组。"""
    import glob as _g
    seen, groups = set(), {}
    for r in roots:
        # ★三个产物族都要扫, 不止 PROF_。2026-10-22 实撞: 只认 PROF_ 时,
        #   A 路径 (OPPROF_, msopprof 产的, **正是本仓性能诊断实际读的那一族**)
        #   1609 份 csv 全判"未知" —— 一个认不出半数语料的环境检查, 比没有更误导。
        # ★根目录本身也要算进去。2026-10-22 实撞: 只 glob 根之**下**, 于是直接把一个
        #   `OPPROF_*` 目录传进来时一份都匹配不到, 报"没有可读的采集配置" ——
        #   又一次"检查自己什么也没看到却报得像结论"。
        cands = ([r] if os.path.basename(r.rstrip(os.sep)).startswith(("PROF_", "OPPROF_"))
                 else [])
        for p in (cands + _g.glob(os.path.join(r, "**", "PROF_*"), recursive=True)
                  + _g.glob(os.path.join(r, "**", "OPPROF_*"), recursive=True)):
            parts = p.split(os.sep)
            root = None
            for i, x in enumerate(parts):
                if x.startswith(("PROF_", "OPPROF_")):
                    root = os.sep.join(parts[: i + 1])
                    break
            if not root or root in seen:
                continue
            seen.add(root)
            e = prof_env(root)
            if not e:
                continue
            k = (e["app_root"], e["marker_keys"])
            g = groups.setdefault(k, {"n": 0, "metrics": {}, "example": root})
            g["n"] += 1
            g["metrics"][(e["metrics"], e["events"])] = \
                g["metrics"].get((e["metrics"], e["events"]), 0) + 1
    return groups


def print_prof_env(roots: list) -> int:
    groups = prof_env_scan(roots)
    if not groups:
        print(f"这些根下没有带 sample.json 的 PROF 目录: {roots}")
        return 2
    print(f"=== 采集环境分组: {sum(g['n'] for g in groups.values())} 份产物, "
          f"**{len(groups)} 个环境** ===")
    for (ar, mk), g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        tag = f"  ★额外键 {','.join(mk)}" if mk else ""
        print(f"\n  app_dir 根 {ar:<20} {g['n']:>4} 份{tag}")
        for (m, n), c in sorted(g["metrics"].items(), key=lambda kv: -kv[1]):
            print(f"      {m:<24} {n} 事件   {c} 份")
    noconf = sum(g["n"] for k, g in groups.items() if k[0].startswith("<A 路径"))
    if noconf:
        print(f"\n  ★其中 {noconf} 份是 **A 路径 (`OPPROF_*`)**, 它没有采集配置文件, "
              "**无法从产物自身判断来源环境** —— 只能靠它落在哪个目录来推断。\n"
              "    拿 A 路径做跨产物结论时, 这一段的环境归属是推断不是读出来的。")
    if len(groups) > 1:
        print("\n  ★**不止一个环境**。跨产物下结论之前先分环境 —— 两个环境的产物混在一张表里, "
              "任何'全语料'结论都失真。")
        print("    额外键 = 当前工具不写、而那批产物里有的 sample.json 键, "
              "它是比目录名可靠得多的环境标志。")
    return 0


def prof_config(prof_dir: str) -> dict:
    """读 PROF 目录的 sample.json (优先 host 侧)。读不到返回 {}。"""
    for sub in ("host", "device_*", "."):
        for f in sorted(glob.glob(os.path.join(prof_dir, sub, "sample.json"))):
            try:
                return json.load(open(f, encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return {}


def print_prof_config(prof_dir: str, other: str | None = None) -> int:
    c = prof_config(prof_dir)
    if not c:
        print(f"{prof_dir} 下没有 sample.json —— 要给 PROF_xxx 那一层")
        return 2
    if other is None:
        on = {k: v for k, v in c.items() if v not in ("", "off", False, None)}
        print(f"=== 采集配置 {prof_dir} ===")
        print(f"  共 {len(c)} 个开关, 真正生效的 {len(on)} 个:")
        for k in sorted(on):
            print(f"    {k:34s} {on[k]!r}")
        ev = c.get("ai_core_profiling_events") or ""
        if ev:
            print(f"\n  ★本次下发的 AI Core PMU 事件: {ev}")
            print(f"    (档位 ai_core_metrics = {c.get('ai_core_metrics')!r};"
                  f" 不同档位下发的是**不同的硬件计数器**, 跨档比较 per-pipe 列没有意义)")
        return 0
    d = prof_config(other)
    if not d:
        print(f"{other} 下没有 sample.json")
        return 2
    diff = {k: (c.get(k), d.get(k)) for k in set(c) | set(d)
            if c.get(k) != d.get(k) and k not in _PROFCONF_IGNORE}
    print(f"=== 采集口径比较 ===\n  A = {prof_dir}\n  B = {other}")
    if not diff:
        print("  ★两次采集口径**一致** (已忽略 job_id/result_dir 等每次必不同的键)")
        return 0
    print(f"  ★**{len(diff)} 个开关不同** —— 这两次不是同法对测:")
    for k in sorted(diff):
        a, b = diff[k]
        print(f"    {k:34s} A={a!r}  B={b!r}")
    if "ai_core_metrics" in diff or "ai_core_profiling_events" in diff:
        print("\n  ★★metrics 档或 PMU 事件不同 —— **per-pipe / PMU 派生列不是同一组计数器**, "
              "拿它们的数字相比是错的。")
        #  ★但不是整张表都不可比。2026-10-21 实测两对同负载同形状的跨档采集
        #  (bytes_pair_20260918 与 gmtraffic_g5, 各自同时有 L2Cache 与 MemoryAccess):
        #  Duration / aicore_time / aic|aiv_total_cycles / Block Num 四项跨档差 **≤0.13%**,
        #  属趟间噪声量级。只说"不可比"会让人把本来能用的基座也丢掉。
        print("     但**基座列跨档实测可比** (两对同负载跨档采集, 差 ≤0.13%, 属趟间噪声):")
        print("       Duration(us) · aicore_time(us) · aic/aiv_total_cycles · Block Num")
    return 1


def selftest_prof_config() -> int:
    import tempfile
    bad = 0

    def ck(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    base = {"ai_core_metrics": "L2Cache", "ai_core_profiling_events": "0x500,0x502",
            "aicore_sampling_interval": 10, "runtimeApi": "off", "acl": ""}

    def mk(extra):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "host"))
        c = dict(base, **extra)
        json.dump(c, open(os.path.join(d, "host", "sample.json"), "w"))
        return d

    a = mk({"job_id": "J1", "result_dir": "/tmp/a"})
    b = mk({"job_id": "J2", "result_dir": "/tmp/b"})
    ck("★只差 job_id/result_dir 时必须判'口径一致' (否则比较器永远报差异)",
       print_prof_config(a, b), 0)
    c2 = mk({"job_id": "J3", "ai_core_metrics": "MemoryAccess"})
    ck("metrics 档不同必须判不一致", print_prof_config(a, c2), 1)
    ck("读得到配置且键数对", len(prof_config(a)), 7)
    ck("给错目录时报 2 而不是装作没配置", print_prof_config(tempfile.mkdtemp()), 2)
    # ---- 采集环境指纹 (2026-10-22 加) ----
    #   动因: 语料里混进另一个环境的 46 份产物, 目录名/档位名/列名都看不出来,
    #   只有 app_dir 前缀与两个多出来的 sample.json 键能认出来。
    here = mk({"job_id": "J9", "app_dir": "/tmp/x/y/z",
               "ai_core_metrics": "PipeUtilizationExct",
               "ai_core_profiling_events": "0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8"})
    there = mk({"job_id": "JA", "app_dir": "/workspace/vent/jobs/abc/work",
                "ai_core_metrics": "PipeUtilization",
                "ai_core_profiling_events": "0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9",
                "ntsMetrics": "", "host_numa_profiling": "off"})
    e1, e2 = prof_env(here), prof_env(there)
    ck("app_dir 只取前三段做根 (完整路径每次都不同, 当指纹会把同环境拆开)",
       (e1["app_root"], e2["app_root"]), ("/tmp/x", "/workspace/vent"))
    ck("★当前工具不写的键被当成环境标志 (它比目录名可靠)",
       (e1["marker_keys"], e2["marker_keys"]),
       ((), ("ntsMetrics", "host_numa_profiling")))
    ck("事件数如实数出来 (9 与 8 要分得开)", (e1["events"], e2["events"]), (8, 9))
    # 分组: 两份不同环境必须分成两组, 不得并成一组
    d = tempfile.mkdtemp()
    for i, cfg in enumerate((
            {"app_dir": "/tmp/x/y/z", "ai_core_metrics": "PipeUtilizationExct",
             "ai_core_profiling_events": "0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8"},
            {"app_dir": "/workspace/vent/jobs/abc/work", "ai_core_metrics": "PipeUtilization",
             "ai_core_profiling_events": "0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9",
             "ntsMetrics": "", "host_numa_profiling": "off"})):
        sub = os.path.join(d, f"PROF_00000{i}_2026", "host")
        os.makedirs(sub)
        with open(os.path.join(sub, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    g = prof_env_scan([d])
    ck("★两个环境必须分成两组 (并成一组则'全语料'结论会悄悄跨环境)", len(g), 2)
    # ★A 路径没有采集配置, 不许静默跳过 —— 上一版 return {} 让 380 个目录全不进分组
    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "OPPROF_20260101_ABC", "op_x", "1"))
    g2 = prof_env_scan([d2])
    ck("★A 路径 (OPPROF_, 无 sample.json) 必须进分组并标明「无采集配置」, 不许跳过",
       [k[0] for k in g2], ["<A 路径, 无采集配置>"])
    ck("普通目录没有配置时仍然跳过 (那只是不是产物, 不值得单列)",
       len(prof_env_scan([tempfile.mkdtemp()])), 0)
    # ★根目录本身就是产物目录时也要算进去 (只 glob 根之下会一份都匹配不到)
    d3 = tempfile.mkdtemp()
    r3 = os.path.join(d3, "OPPROF_20260101_XYZ")
    os.makedirs(os.path.join(r3, "op", "1"))
    ck("★直接把 OPPROF_ 目录本身传进来也要能认出 (不能只 glob 根之下)",
       len(prof_env_scan([r3])), 1)
    # 一站式: 给错层要说"给错层了", 不能装作全过
    tr = trust_report(tempfile.mkdtemp())
    ck("★--trust 给错层时报'不适用'并说明, 不许给出一张全过的表",
       (len(tr), tr[0][1]), (1, "不适用"))
    # ★单份产物上"采集环境"永远不会不过 —— 永远不会不过的检查等于没有检查
    d4 = tempfile.mkdtemp()
    r4 = os.path.join(d4, "OPPROF_20260101_AAA")
    os.makedirs(os.path.join(r4, "op", "1"))
    env_row = [x for x in trust_report(r4) if x[0] == "采集环境"]
    ck("★单份产物时'采集环境'判不适用而不是过 (300/300 全过 = 这一项在单份上没有判别力)",
       env_row[0][1] if env_row else "缺", "不适用")
    ck("每组的份数对", sorted(x["n"] for x in g.values()), [1, 1])

    print(f"采集配置自检: 14 条, {bad} 条未过")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="msprof op 采集 + pipe 瓶颈判读")
    ap.add_argument("--signals", help="只读已有 msprof 产物目录并打印全量特征信号表 (不重跑采集)")
    ap.add_argument("--kernel-dir", help="产物下有多个核函数时指定看哪一个; 不指定则拒跑 (混算的均值没有意义)")
    ap.add_argument("--timeline", metavar="DIR",
                    help="判读 PipeTimeline 产物 (trace.json): 按核分别给出各流水的**实测**占用与空隙。"
                         "与 --signals 的忙时不是一回事 —— 见 --refs 的 P6")
    ap.add_argument("--mfu-flops", type=float, default=None,
                    help="给**有效** FLOP 数, 额外打印 MFU 四段分解 (忙占比 x 窗口占比 x 忙时效率) "
                         "与访存侧忙占比上界。配 --e2e-us 用 (不给则拿 Task Duration)")
    ap.add_argument("--e2e-us", type=float, default=None,
                    help="端到端 device 时间 us (不给则用 msprof 的 Task Duration)")
    ap.add_argument("--profdir", metavar="PROF_DIR",
                   help="清点一个 PROF_xxx 目录: 每条通道几个切片/多大/有无完成标记/认不认识 "
                        "(通道清单取自 libprofimpl.so 的 .rodata, 见 docs/msprof-re/02 §9.3)")
    ap.add_argument("--profconf", metavar="PROF_DIR",
                   help="读一次采集**实际生效的**配置 (PROF_xxx/host/sample.json)")
    ap.add_argument("--vs", metavar="PROF_DIR2",
                   help="配 --profconf 用: 比较两次采集的口径是否相同 (同法对测的前置检查)")
    ap.add_argument("--events", metavar="PROF_DIR",
                    help="回读一次 B 路径采集真正下发了哪几个 PMU 事件, "
                         "以及产物里拿不拿得到 (见 01 §3.1b)")
    ap.add_argument("--measure-events", metavar="事件[,事件...]",
                    help="「我想测 X」一条命令: 按自配 PMU 事件采一次 (B 路径) 并出表。"
                         "名字与事件号可混写 (scalar_busy_cycles,0xc)。需配 --app, "
                         "可配 --out。★A 路径 (msprof op) 结构上不收, 见 01 §3.1c")
    ap.add_argument("--list-events", action="store_true",
                    help="列本机可点的 AI Core PMU 事件 (--measure-events 的菜单)")
    ap.add_argument("--bpath-lint", metavar="KERNEL_DETAILS_CSV",
                   help="B 路径产物体检: 对 kernel_details.csv 跑可机检的失真点判据 "
                        "(04 篇 28 条里能从 csv 直接判的那部分)")
    ap.add_argument("--extcorr-demo", action="store_true",
                   help="自定义归责单元端到端演示 (占卡): 三个单元 + 一条第二轴, 报每单元内核数与时间。"
                        "需 ASCEND_RT_VISIBLE_DEVICES=<非0> 与 LD_PRELOAD=libmspti.so")
    ap.add_argument("--untried", action="store_true",
                    help="列出尚未在本仓试过的工作量再分派手段 (供任务简报机械带上)")
    ap.add_argument("--moves", nargs="?", const="", metavar="PIPE",
                   help="按 binding pipe 查工作量再分派的候选手段与决策规则 (P20 的可操作面); "
                        "不给参数则列全部")
    ap.add_argument("--refs", action="store_true",
                    help="打印本模块判据的出处/实证/最后验证日期 (含 disputed 与 refuted)")
    ap.add_argument("--new-cols", metavar="DIR",
                    help="新采的产物目录: 报它带来了哪些**此前没见过**的列 "
                         "(基线默认取 runs/ 与 cases/*/runs/)")
    ap.add_argument("--cols-base", action="append", default=[],
                    help="--new-cols 的基线根, 可给多次")
    ap.add_argument("--selftest", action="store_true",
                    help="MFU 分解的判据负例自检 (恒等式闭合 + 上界不许超 100%% + 旧错法必须被抓住)")
    ap.add_argument("--compare", metavar="DIR1,DIR2,...",
                    help="跨构建对比: 多个 msprof 产物目录, 按 --cols 打成一张表。"
                         "沉淀动机: 本会话手写了十几遍 'for T in ...; do perf --signals | grep ...' "
                         "的循环, 每次挑的列还不一样, 极易漏掉关键信号")
    ap.add_argument("--cols", default="aic_time(us),aiv_time(us),aic_mte2_time(us),aiv_mte3_time(us),"
                                      "aiv_mte3_active_bw(GB/s),aic_read_hit_rate(%),aiv_write_hit_rate(%),"
                                      "GM_to_L1_datas(KB),UB_to_GM_datas(KB),aiv_vec_bankgroup_cflt_ratio",
                    help="--compare 要打的列, 逗号分隔")
    ap.add_argument("--workrate", metavar="DIR_A,DIR_B",
                    help="两份采集: 差别是**多搬了字节**还是**同样的字节搬得慢**? "
                         "把 *_bw(GB/s) 乘回 Duration 还原成字节再比。这一问分岔整条优化路线 —— "
                         "量差就去减搬运, 率差就去查粒度/对齐/停顿, 手段与上限完全不同。"
                         "★只答'量 vs 率', 不答'时间花在哪条 pipe' (后者看 diagnose abdiff)")
    ap.add_argument("--workrate-labels", default="A,B", help="--workrate 两份产物的标签")
    ap.add_argument("--sync-attr", metavar="PROF_DIR",
                    help="同步等待归责: 哪个跨核标志等得最久, 并把它接回 kernel 里那一行 "
                         "CrossCoreWaitFlag。须配 --kernel。"
                         "沉淀动机: msprof 一直在报 wait_id0..15, 本表 2026-10-19 才接进来, "
                         "接进来就在 gmfr v8.5 的 det off->on 对比里排第一 (15.1 -> 3081.1us)")
    ap.add_argument("--kernel", metavar="OP_KERNEL_DIR",
                    help="--sync-attr 用: 算子的 op_kernel 目录。**不可省** —— 标志位是全仓共享的 "
                         "0~14 编号空间, 同一个 id 在不同算子里含义不同, 必须现 grep 本算子源码")
    ap.add_argument("--metrics", default=DEFAULT_AIC_METRICS,
                    help="msprof 采集的 aic-metrics。★**档位名随平台变**, 先敲 "
                         "`msprof op --help` 看本机支持哪些, 别照搬别处的名字 —— "
                         "910B2 上是 ArithmeticUtilization / MemoryUB / Memory / MemoryL0 / "
                         "L2Cache / PipeUtilization / ResourceConflictRatio / BasicInfo / "
                         "Roofline / Occupancy / **TimelineDetail** / KernelScale / Source / "
                         "MemoryDetail / Default; **没有 PipeTimeline** —— 它在本平台叫 "
                         "TimelineDetail。2026-09-17 因照搬 950PR 的名字被拒, 一度误判成"
                         "'本 SoC 不支持流水时间线', 实为**能力在而名字不同**。"
                         "torch_npu 的 AiCMetrics 里档位更少 (无 TimelineDetail), 所以要走 msprof。")
    ap.add_argument("--bpath-corpus", metavar="根[,根...]",
                    help="把 --bpath-lint 跑遍这些根下**全部** kernel_details.csv 并汇总。"
                         "单份上的 miss 几乎没有信息 (分不开'失真不存在'与'这份考不动'), "
                         "**只有跨语料才分得开** —— 全 na 的判据等于从没被考过")
    ap.add_argument("--apath-lint", metavar="CSV 或 OPPROF_ 目录",
                    help="A 路径自洽体检: 逐 block 恒等式 / 命中率与它同一行的计数器 / "
                         "比值上界。★A 路径是本仓性能诊断实际读的那一族, 而 --bpath-lint "
                         "的判据按 kernel_details 的列名取数, 在这边一条都不适用")
    ap.add_argument("--trust", metavar="产物目录",
                    help="★**采集之后的第一条命令**: 一次跑完丢条数对账 / 切片断号 / "
                         "采集环境 / 列判据, 给一张四档表 (过 / 不过 / 没看见 / 不适用)。"
                         "配 --since-min 限定日志窗口")
    ap.add_argument("--profenv", metavar="根[,根...]",
                    help="把这些根下全部 PROF 目录按**采集环境**分组 (app_dir 前缀 + "
                         "当前工具不写的那些 sample.json 键)。★跨产物下结论之前先敲它 —— "
                         "语料里混进另一个环境的产物时, 目录名/档位名/列名都看不出来")
    ap.add_argument("--passdev", metavar="OPPROF 目录",
                    help="A 路径**遍次失真**检查: 十遍重放里有没有某遍的 "
                         "total_cycle/(t_end-t_start) 超过物理上限 36 —— "
                         "csv 里看不出来 (01 §13.21/§13.22)")
    ap.add_argument("--dropcheck", action="store_true",
                    help="采集丢条数对账: 这次 profiling 有没有**悄悄少了记录**。"
                         "证据全在昇腾 slog 不在 PROF 目录 —— 两条判据 (逐缓冲区 "
                         "read/write 计数对账 + ERROR 级 overflow 串), 并把'看不见' "
                         "(BLIND) 与'没丢'(CLEAN) 分开报。配 --log-root / --since-min")
    ap.add_argument("--log-root", default="/root/ascend/log",
                    help="--dropcheck 扫的 slog 根 (默认 /root/ascend/log)")
    ap.add_argument("--since-min", type=float,
                    help="--dropcheck 只看 mtime 在最近这些分钟内的日志。"
                         "★刚跑完一次采集就该带上它 —— 不带会把全机历史一起算进来, "
                         "历史干净会把本次的问题稀释掉")
    ap.add_argument("--app")
    ap.add_argument("--kernel-name")
    ap.add_argument("--out")
    ap.add_argument("--launch-count", type=int, default=10)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--cores", type=int, default=24, help="派生量按此核数换算全卡流量")
    ap.add_argument("--pipes", action="store_true",
                    help="★跑一次 msprof 并直接给 忙时比/等待比 + 判读 (需 --app 与 --out)")
    a = ap.parse_args()
    if a.moves is not None:
        return print_work_moves(a.moves)
    if a.untried:
        return untried_moves()
    if a.refs:
        return print_refs("perf", REFS)

    if a.compare:
        dirs = [d.strip() for d in a.compare.split(",") if d.strip()]
        cols = [c.strip() for c in a.cols.split(",") if c.strip()]
        sigs, miss = {}, []
        for d in dirs:
            if not os.path.isdir(d):
                miss.append(d); continue
            sigs[d] = extract_signals(d)
        if miss:
            print(f"★以下目录不存在, 已跳过 (**不是没有差异**): {miss}", file=sys.stderr)
        if not sigs:
            print("没有可读的产物目录", file=sys.stderr); return 2
        w = max(len(os.path.basename(d)) for d in sigs) + 2
        print(f"{'信号':<38s}" + "".join(f"{os.path.basename(d):>{w}s}" for d in sigs))
        for c in cols:
            vals = [sigs[d].get(c, {}).get("mean") for d in sigs]
            if all(v is None for v in vals):
                continue
            cells = "".join((f"{v:>{w}.2f}" if v is not None else f"{'-':>{w}s}") for v in vals)
            print(f"{c:<38s}{cells}")
        print("\n★只列均值。判读前先问三件事: (1) 哪条 pipe 的 wait_ratio≈0 (那条是因);\n"
              "  (2) 搬运量列是否**没变** —— 没变而时间变了, 说明是带宽/延迟不是字节;\n"
              "  (3) 两条 pipe 的增量是否互相抵消 —— 抵消 = 共享资源饱和, 重新分配是零和。")
        return 0
    if a.apath_lint:
        return print_apath_lint(a.apath_lint)
    if a.trust:
        return print_trust(a.trust, a.since_min if a.since_min is not None else 60.0)
    if a.profenv:
        return print_prof_env([x.strip() for x in a.profenv.split(",") if x.strip()])
    if a.profconf:
        return print_prof_config(a.profconf, a.vs)
    if a.profdir:
        return print_prof_inventory(a.profdir)
    if a.passdev:
        return print_passdev(a.passdev)
    if a.dropcheck:
        return print_prof_drops(a.log_root, a.since_min)
    if a.list_events:
        return print_list_events()
    if a.measure_events:
        if not a.app:
            print("--measure-events 需要 --app <被采集的命令>", file=sys.stderr); return 2
        _out = a.out or os.path.join("runs", "events_" +
                                     time.strftime("%Y%m%d_%H%M%S"))
        return print_measure_events(a.measure_events, a.app, _out)
    if a.events:
        return print_event_readback(a.events)
    if a.bpath_lint:
        return print_bpath_lint(a.bpath_lint)
    if a.bpath_corpus:
        return print_bpath_corpus([x.strip() for x in a.bpath_corpus.split(",") if x.strip()])
    if a.extcorr_demo:
        why = ExtCorrSession.preflight()
        if why:
            print(f"拒跑: {why}")
            return 2
        # ★顺序: 先进会话再初始化 device。反过来则内核记录照常来、**关联记录一条不来**,
        # 结果是全部 __unattributed__ —— 静默失效。
        with ExtCorrSession() as s:
            import torch
            import torch_npu  # noqa: F401
            torch.npu.set_device(0)        # 可见卡重编号后逻辑 0 = 指定的物理卡
            x = torch.randn(512, 512, device="npu", dtype=torch.float16)
            torch.npu.synchronize()
            with s.unit("候选A", axis=1):              # 第二条轴: 跨越多个阶段
                with s.unit("预热"):
                    for _ in range(8):
                        y = torch.mm(x, x); del y
                with s.unit("主体"):
                    for _ in range(8):
                        y = torch.relu(x); del y
            with s.unit("候选B", axis=1):
                with s.unit("主体"):
                    for _ in range(8):
                        y = torch.mm(x, x); del y
            torch.npu.synchronize()
        print("\n每个归责单元 (轴0 阶段 | 轴1 候选):")
        for k, v in sorted(s.report().items()):
            print(f"   {k:<24} {v['kernels']:>3} 个内核  {v['us']:>9.2f} us")
        print("\n★三个'主体'里有两个属于不同候选, 内核名相同 —— 按名字分不开, 按轴分得开。")
        return 0
    if a.selftest:
        return (selftest_csv_layout() | selftest_mfu() | selftest_timeline() | selftest_workrate()
                | selftest_kernel_name() | selftest_corrupt_launch() | selftest_sync_attr()
                | selftest_mspti_preload() | selftest_new_columns() | selftest_cflt_wall()
                | selftest_multi_kernel_guard() | selftest_extcorr() | selftest_bpath() | selftest_apath()
                | selftest_rc_sub() | selftest_prof_inventory() | selftest_prof_config() | selftest_wait_id_cols()
                | selftest_prof_drops() | selftest_events() | selftest_traffic_duty()
                | selftest_push_moves())
    if a.new_cols:
        bases = a.cols_base
        if not bases:
            _r = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            bases = [os.path.join(_r, "runs")] + glob.glob(
                os.path.join(_r, "cases", "*", "runs"))
            bases = [x for x in bases if os.path.isdir(x)]
        r = new_columns(a.new_cols, bases)
        print(f"已有列 {r['known']} (扫了 {len(bases)} 个根)  "
              f"新采列 {r['new_total']}  **此前没见过 {len(r['fresh'])}**")
        # ★范围必须打出来。2026-10-22 实撞 (11 篇 B1): 一次范围未声明的仓内扫描
        #   被写成"实测核过", 结论"被资源挡住"挂了两天 —— 缺的不是数字, 是范围。
        print("  基线范围:")
        for b in bases:
            n = len(glob.glob(os.path.join(b, "**", "*.csv"), recursive=True))
            print(f"    {b}  ({n} 份 csv)")
        if not a.cols_base:
            print("  ★这是**默认基线 = 仓内** (runs/ 与 cases/*/runs)。会话 scratchpad 里的"
                  "产物不在其中, 用 --cols-base 可以加。")
        print("  ★缩窄基线会让 `fresh` **变多**, 所以'零新列'在窄基线上是**保守结论** —— "
              "加宽只会更稳。反过来, '没有合格样本'这类结论在窄范围上会**假阳**。")
        if not r["fresh"]:
            print("  —— 没有带来任何新列。这本身是结论: 不必为它再采第二次。")
            return 0
        for c, f in sorted(r["fresh"].items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  {f:<40} {c}")
        return 0
    if a.workrate:
        parts = [x.strip() for x in a.workrate.split(",") if x.strip()]
        if len(parts) != 2:
            print("--workrate 要两个目录, 逗号分隔")
            return 2
        labs = ([x.strip() for x in a.workrate_labels.split(",")] + ["A", "B"])[:2]
        r = workrate(parts[0], parts[1], labs[0], labs[1])
        return 0 if r["verdict"] not in ("NO_DURATION", "NO_BW_COUNTERS") else 1
    if a.pipes:
        if not (a.app and a.out):
            print("[用法] --pipes 需要 --app <可执行文件或脚本> 与 --out <输出目录>")
            print("  ★--application 要的是**可执行文件路径 + 参数**, 不是 shell 命令;")
            print("    要 cd / 设环境变量就先写成一个 chmod +x 的 .sh 再传进来。")
            return 2
        profile_pipes(a.app, a.out, a.kernel_name or "",
                      launch_count=a.launch_count, skip=a.skip, metrics=a.metrics)
        return 0
    if a.timeline:
        print_timeline(a.timeline, a.e2e_us)
        return 0
    if a.sync_attr:
        if not a.kernel:
            ap.error("--sync-attr 需同时给 --kernel <op_kernel 目录>。**不可省** —— 标志位是全仓"
                     "共享的 0~14 编号空间, 同一个 id 在不同算子里含义不同, 没有源码就无从解释。")
        sync_attr(a.sync_attr, a.kernel)
        return 0
    if a.signals:
        sig = extract_signals(a.signals, a.kernel_dir)
        print_signals(a.signals, sig=sig, cores=a.cores, kernel=a.kernel_dir)
        if a.mfu_flops:
            from core import socspec           # 硬件常数唯一取数口
            sp = socspec.spec()
            e2e = a.e2e_us
            if e2e is None:
                try:
                    e2e = extract(_find_opprof(a.signals) or a.signals).task_duration_us
                except Exception:  # noqa: BLE001
                    e2e = None
            if not e2e:
                print("\n  ★没有端到端时间: 给 --e2e-us, 或让产物里带 Task Duration")
            else:
                mfu_breakdown(sig, a.mfu_flops, e2e, a.cores,
                              sp["derived"]["peak_fp16_TFLOPS"] * 1e12,
                              sp["derived"]["dram_model_TBps"] * 1e12,
                              sp["derived"]["l2_model_TBps"] * 1e12)
        return 0
    if not (a.app and a.kernel_name and a.out):
        ap.error("采集模式需 --app/--kernel-name/--out; 或用 --signals <已有产物目录>")
    d = run_msprof_op(a.app, a.kernel_name, a.out, a.launch_count, a.skip, metrics=a.metrics)
    print(json.dumps(extract(d).as_dict(), indent=2, ensure_ascii=False))
    print()
    print_signals(a.out, cores=a.cores)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
