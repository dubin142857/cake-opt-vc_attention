"""kernel UB 用量静态估算 — build 前料敌于先, 溢出方案编译前筛掉 (省 ~5min/次 build, 加速 loop).

原理: AscendC 的 UB 分配全在 `pipe->InitBuffer(que, num, bytesPerBuf)` (TQue, 总 = num*bytes)
或 `pipe->InitBuffer(buf, bytes)` (TBuf, 单块). 解析这些调用, 代入 tiling 值, 求和 vs UB 容量 (910B=192KB).
新增/改动缓冲前先跑本工具; 超 192KB 直接毙, 不浪费 build.

用法:
    python -m core.ub_budget <kernel.h> --values '{"ubCalSize":8192,"baseN":256,"ubRestBytes":40000,
        "BUFFER_NUM":2,"DETER_UB_SIZE":12288, "sizeof_cT":4,"sizeof_DTYPE_OUT":4,"sizeof_float":4,"sizeof_bfloat16_t":2}'
未提供的符号按 name 原样保留并标 UNRESOLVED (提示需补值).
"""
from __future__ import annotations
import argparse, json, re, os, shutil, sys

from core._refs import Ref, print_refs
from core import _dialect

REFS = {
    "U8_scaling": Ref(
        what="哪几个 InitBuffer 的尺寸随**核函数运行期入参**增长 —— 即 UB 占用会被输入撑爆的那一维。"
             "只定位不定量: 答'哪一维', 不答'现在占了多少' (那是 U1-U7)",
        status="measured",
        evidence=[
            "roi_align 2026-09-20: 内核把整张 H*W 特征图装进 UB",
            "  (InitBuffer(inQ_, 2, planeB), planeB <- HWp_ <- H_*Wp_ <- 入参 H, W)。",
            "  公开 20 个 case 的 H*W 最大 128*128 = 16384 元素, 一直没暴露;",
            "  逐步放大实测 176x176 (30976) 通过、200x200 (40000) 起报 507035",
            "  (device error type 3, copy_between_host_and_device)。榜上 ROIAlign hidden",
            "  19 个判失败里 4 条正是这个报错, 另有 13 条是它带崩设备后的",
            "  「设备不可恢复, 用例跳过」级联。",
            "★负例检验 (判据必须分得开修前修后):",
            "  修前源码 -> inQ_ 报「随入参增长: H, W」(整平面)",
            "  改成按行分带后 -> inQ_ 报「随入参增长: W」(一带若干行), H 不再进尺寸表达式。",
            "★为什么不能用 U1 代替: U1 要先给一组 tiling 值, 它答的是'这一组下占多少'。",
            "  没人会拿包络外的形状去代 —— 公开 case 里根本没有那么大的图。",
        ],
        verified="2026-09-20",
    ),
    "U7_afford": Ref(
        what="一个行数/粒度常量能不能调大 —— 点名几个同时存活的缓冲求和与池容量比。只证伪, 不证安全",
        status="measured",
        evidence=[
            "sfag 2026-09-17: 想把 NON_OPT_UB_ROW_SIZE 从 16 调到 32 (det 路径的 scatter 读粒度),",
            "因为同一个杠杆 4->16 那次 (g3) 拿了 +3.10%。手算发现装不下, 于是把手算做成判据。",
            "真数据: dimDAlign=dimD2Align=512, 池 vecQue=191KB=195584B。",
            "  rows=32 -> 2 x 32 x 512 x 2 x 4 = 262144 B, 超 66560 B -> 判死",
            "  rows=24 -> 196608 B, 超 1024 B -> 判死 (边界)",
            "  rows=20/18 -> not_refuted (不给结论)",
            "  rows=16 (**已知良品**, g10 已交付并端到端验收) -> not_refuted, **不报警**",
            "★为什么不复用 U5 的绝对峰值: 那条判据在已知良品 v9a 上就报越界 (见 pool_offset_report",
            "  的 docstring), 对良品报警的判据不能拿来下结论, 所以至今没挂进 analyze()。",
            "★U7 换了个只会出真阳性的方向: 不推整条路径的峰值, 只加**调用者点名**的那几个缓冲。",
            "  调用者断言它们同时存活, 于是这个和是真实峰值的**下界**; 下界超了就一定超。",
            "  反过来不成立 -> sum<=cap 时必须拒绝给结论, 输出写成 U7_not_refuted 而不是 OK。",
            "★配套修掉两个真缺陷 (都是这次样本外检验暴露的):",
            "  (1) 池容量常是普通局部量 `uint32_t totalUbSpace = 191*1024;`, 不是 constexpr,",
            "      source_constants 抓不到 -> 第一次跑报 U7_no_pool。补 _literal_locals(), 只收",
            "      纯字面量算式 (掺进标识符就可能是运行期量, 那会把越界报成 headroom 充足)。",
            "  (2) `main()` 的返回值被丢掉 (`main()` 而非 `sys.exit(main())`) —— 到 U7 之前每条",
            "      分支都 return 0, 所以这个缺陷一直没有表现。U7 是第一条要报非零退出码的。",
            "★副产物 (同次排查, 与判据无关但值得记): det=1 时 host 的 EnableOptimizedScatterPath",
            "  第一条就 `if (deterministic) return disabled`, 所以确定性路径走的确实是",
            "  NON_OPT_* 那支分配 —— g3/g10 的 detRows=16 与分配口径一致, 不是 U6 那类分叉。",
        ],
        verified="2026-09-17"),
    "U6_dual_capacity": Ref(
        what="同一个缓冲在互斥分支里用不同的容量常量分配 —— 读写两端可能各选一支, 按小的给、按大的写就是越界",
        status="measured",
        evidence="sfag 2026-09-16 实撞, 一个 507015 排了五轮 (v13a/v13c/v14a/v15a 四轮改的都是无关代码):\n"
                 "gatherTensorPing 在非优化分支按 NON_OPT_MAX_GATHER_SIZE(64) 行分配, 在优化分支按\n"
                 "MAX_GATHER_SIZE(32) 行分配。**两支都分配了**, 所以任何'有没有赋值'的检查都看不见。\n"
                 "真正分叉的是条件: UB 分配按 enableOptimizedScatter 走分支, 而 GatherKV 的分派按\n"
                 "IS_DETERMINISTIC —— det=1 且门开时缓冲按 32 行给、gather 按 64 行写, 一半落在界外。\n"
                 "门关着时那两个条件恒等价, 所以错配藏了四轮。\n"
                 "★与 U5/U5D 的分工: 那两条算**总量**, 这条看同一缓冲的**口径分叉** —— 本例 UB 总量\n"
                 "  一直判过 (99%, headroom 1024B), 总量正常不代表每个缓冲的读写两端对得上。\n"
                 "★判据只报'有两种口径', 不判对错: 哪一支配哪一支是语义, 静态判不了。它省掉的是\n"
                 "  '根本没想到这里有两种口径'。真仓命中 7 个, 无噪声。\n"
                 "★2026-09-16 二次扩展: 每支分配的**守卫条件**由 guard_map() 自动提出来并列在\n"
                 "  输出里 (兜底那支标'无守卫')。sfag 这例的全部信息量就在这一行 —— 门开时生效的是\n"
                 "  兜底那支(32 行), 不是门内那支(64 行), 此前要人工回源码追四层大括号才看得出。\n"
                 "★已挂进**默认**路径: 跑 ub_budget 不加参数也会出 U6 提示, 不依赖'记得敲 --dual-cap'。\n"
                 "★反向环复测 (判据是事后写的, 必须证明它在**事前**的代码上也报): 取 v15a 那次提交\n"
                 "  (问题仍在、已连改四轮无关代码的状态) 的源码跑 U6, 直接报出 gatherTensorPing/Pong\n"
                 "  的两支口径 —— 行 458 在 !enableOptimizedScatter 下按 NON_OPT_MAX_GATHER_SIZE,\n"
                 "  行 548 是无守卫的兜底按 MAX_GATHER_SIZE。当时人工用了四轮 (v13a/v13c/v14a/v15a)\n"
                 "  才走到这一步, 期间四次改动全在改无关代码。",
        verified="2026-09-16"),
    "U10_storage_class": Ref(
        what="InitBuffer 的目标**不一定在 UB**。TSCM / TQueBind 到 A1,B1 的缓冲绑的是 L1, "
             "CO1/CO2 是 L0C。按缓冲变量的**声明类型**归属, 不按「出现在 InitBuffer 里」归属。",
        status="measured",
        evidence=[
            "vc_attention 2026-10-27 实撞: vc_kernel.cpp:49-50",
            "  `TSCM<TPosition::GM, VSCM_SLOTS> vScm; pipe.InitBuffer(vScm, VSCM_SLOTS, VSCM_SLOT_BYTES);`",
            "  本模块把这 131072 B 计进 UB, 于是打出 191104/196608 (97%)。",
            "  真实 UB 只有 RunVector 的那份, tile_m=64 下 135424/196608, 空闲 61184 B。",
            "  **后果**: 「UB 已满, AIV 侧放不下双缓冲」这一判断据此成立并关闭了一个优化方向,",
            "  跨多轮未被复查。错数比没数危险, 因为它看起来像已经查过了。",
            "★负例检验: 同一文件里 RunVector 的 rawBuf/scoreBuf 声明为 TBuf<TPosition::VECCALC>,",
            "  必须仍计进 UB —— 判据不能把真 UB 缓冲也排除掉。",
            "★找不到声明时**仍按 UB 计**并标记: 少算比多算危险 (静默假阴性是本模块最危险的失效形态)。",
        ],
        verified="2026-10-27",
    ),
    "U11_headline_matches_verdict": Ref(
        what="**打印出来的那一行必须与结构化判定用同一个数。**多函数时判定取单函数最大值, "
             "而顶部那行若打全文件求和, 则工具自己的 note 说不能这么判, 打出来的却正是那么判的。",
        status="measured",
        evidence=[
            "vc_attention 2026-10-27 实撞: per_function 已给 RunCube 131072 / RunVector 60032,",
            "  over_cap 也已按单函数最大值修正, 但末行打的是 `UB: 191104/196608 B (97%)` —— 全加。",
            "  人只读那一行。note 里写着「全加会误报」救不回来, 因为 note 在 JSON 里而结论在终端。",
            "★这条与 U8 是**同一个缺陷的两半**: U8 在 2026-09-20 cummin 上修的是判定, ",
            "  没修打印; 一年后在另一个算子上由打印那半重新咬了一次。",
            "  判据: 多函数时顶行必须报 worst_fn 与它的字节数, 并显式标注这是单函数最大值。",
        ],
        verified="2026-10-27",
    ),
    "U9_assert_margin": Ref(
        what=["源码自带的 `static_assert(<LHS> <= <RHS>)` 容量约束要被**解出来**报余量, ",
              "并能回答「点名的常量最大调到多少」—— 判据不重新推导约束, 只解已有的"],
        status="measured",
        evidence=[
            "缺口来源 2026-09-30 (CAKE-PI 裸基线 T2): 要判「把 GATHER_UB_ROWS 从 8 调到 16, ",
            "UB 放不放得下」, U7_afford 报 U7_no_pool —— 它只认两参的 `InitBuffer(buf, cap)` ",
            "(TBuf 池), 而那份源码用三参的 `InitBuffer(que, 1, bytes)` (TQue 队列), 且两个行数",
            "常量**不产生自己的 InitBuffer**, 它们在已分配好的队列里切槽位。agent 只能手算。",
            "而约束本来就写在源码里: sfag block_vec.h:128/897/898 那三条 static_assert。",
            "本判据在同一文件上复现出手算的全部数字: GATHER_UB_ROWS 9216<=16640 余量 7424 B,",
            "  最大可调到 **14**; SCATTER_DK_BYTES 18432<=18688 与 SCATTER_DV_BYTES 16384<=16640",
            "  余量各 **256 B**, SCATTER_UB_ROWS 当前 4 **已到顶**。",
            "★两处实现上的坑, 各配了负例: (1) 顶层逗号切分不能把 `<=` 里的 `<` 当嵌套开括号,",
            "  否则条件里混进断言消息, 三条全解不出来; (2) 常量求解必须**不动点**而非单遍 ——",
            "  `VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO` 这类链式常量单遍会被跳过。",
            "★「相关」不是「字面提到」: 约束常写派生常量 (SCATTER_DK_BYTES), 而要调的是源头",
            "  (SCATTER_UB_ROWS)。判法是把该常量 +1 重解不动点, 看约束两边的值变不变。",
            "★上探用指数+二分但**显式验边界** (best 成立且 best+1 不成立), 核不过退回线性扫 ——",
            "  单调性是容量约束的常态而非定理。",
        ],
        verified="2026-09-30"),
    "U7_guard_divergence": Ref(
        what="(已撤销) 曾想独立成条: 同一标识符在不同函数里被互不相交的条件守卫 —— 两个真相来源",
        status="refuted",
        evidence="2026-09-16 实现后在真仓上**自我证伪**, 留档以免后人重做:\n"
                 "第一版对任意标识符报, sfag 真仓 82 处命中, 混进 else(关键字被当标识符)、\n"
                 "tilingData(容器)、BLOCK_FP32(常量) —— 判据过宽等于没有。\n"
                 "收窄到'一处赋值、另一处读取'降到 8 处; 再补'存在无条件赋值点则跳过'后**降到 0**。\n"
                 "★原因是形态判断错了: 声称要抓的七次事故, 真仓里的实际形态全部是**两支都分配、\n"
                 "  容量口径不同**(U6 已覆盖), 而不是'一支分配另一支没有'。收窄到真实形态后,\n"
                 "  它只能靠自造的自检负例活着 —— 一条在真仓零命中的判据是断言不是方法。\n"
                 "★处置: 撤掉独立判据, 唯一的增量(守卫条件提取)并入 U6 的输出, 见上条。\n"
                 "  工具数量本身也是熵, 同一形态不开第二条判据。",
        verified="2026-09-16"),
    "U5_absolute_peak": Ref(
        what="用 `ubOffset` 的累加峰值直接判 UB 池内越界",
        status="refuted",
        evidence="sfag 2026-09-16 样本外检验: v9a (已交付、端到端验收过、确定能跑) 与 v16c\n"
                 "(确定崩 507015) 的峰值**完全相同**, 都是 221184 vs 池 195584 —— 对已知良品\n"
                 "同样报'越界 25600 B'。判据对良品报警, 它报的就不是越界。\n"
                 "★U5D (池内偏移**差分**) 不受此影响: 它比的是同一份代码改动前后的同一个量,\n"
                 "  两边的高估相消。要判绝对越界得先做对分段数据流, 在那之前只用差分。",
        verified="2026-09-16"),
    "cap.UB": Ref(
        what="UB 总容量 196608B",
        status="ini_exact",
        evidence="平台配置 Ascend910B2.ini 的 ub_size",
        verified="2026-08-12"),
    "margin.1KB": Ref(
        what="判过的线是 cap-1024B, 不是 cap",
        status="measured",
        evidence="910B2 真机: 总占用 196608B 失败、195584B 通过。差的这 1KB 是编译器/运行时另占, "
                 "ini 不声明, 只能实测",
        verified="2026-08-13"),
    "unresolved.must_report": Ref(
        what="表达式解析不出时必须**显式报 UNRESOLVED 并拒绝判过**, 不能当 0",
        status="measured",
        evidence="解析失败当 0 会给出一个偏小的 total 并'通过' —— 那是工具主动制造假安全。"
                 "gmfr 实际撞到: perTokenScalebufferNum 是运行期三元表达式, 不补 --values 就解不出",
        verified="2026-08-24"),
    "literal_expr_only": Ref(
        what="按 InitBuffer 的**字面表达式**求值, 走局部变量就解不出",
        status="measured",
        evidence="gmfr 的 nBatch 攒批曾把宽度写成局部变量, 本工具只能给下界并拒绝判过; "
                 "改成直接写 tiling 字段的表达式后可解 —— 这条约束反向影响了内核写法",
        verified="2026-08-13"),
}

#  ★键名必须是**具体芯片型号**, 不是 CANN 的目录系列名。
#    `ascend910b` 在 CANN 里是二进制目录名 (binary/ascend910b/ 放 910B 全系列),
#    但作为芯片型号它指的是另一颗真实存在的芯片: Ascend910B 是 arch 1001 / AIC-C-100 /
#    30 个一体核 / L2 32MB / UB 256KB, 与本机的 Ascend910B2 (arch 2201 / 24 cube + 48 vector /
#    L2 192MB / UB 192KB) 是两代。本表原先用 `ascend910b` 当键却存着 910B2 的数,
#    名实不符 —— socspec --audit 报的"本机型号不在表里"正是这件事。2026-08-29 改名。
#  ★多缓冲策略类开几份, 是编译期由**类名**决定的 —— 这一层不数清楚, 容量账就整体错一个倍数。
#    2026-09-14 FIAS 实撞: 把 KV 槽的策略从 3buff 换成 DB 是"省 144K"的唯一手段, 而源码上只是
#    改了一个类名, 字面上的 Init(mgr, size) 一模一样。
POLICY_MULT = {
    "BuffersPolicySingleBuffer": 1,
    "BuffersPolicyDB": 2,
    "BuffersPolicy3buff": 3,
    "BuffersPolicy4buff": 4,
}

#  L1 的容量不进 *_CAP 表: 它由内核自己 `l1BufferManager_.Init(pipe, <字面量>)` 声明,
#  而那个字面量就是本次分配的上限。读源码比查表可靠 —— 查表会在换芯片时静默失配。
UB_CAP = {"ascend910b2": 192 * 1024}


#  ---- UB 容量: 走 socspec 唯一取数口 ------------------------------------------------
#  ★2026-08-29 补。原来是 `UB_CAP.get(soc, 192 * 1024)` —— **静默回落**:
#    型号不在表里就悄悄按 192KB 判, 判出来的"通过"没有任何依据。
#    容量判据静默回落比直接报错危险得多: 它主动制造假安全, 而使用者看到的是绿的。
#    (本模块自己的 unresolved.must_report 判据说的就是这件事 —— "解析不出必须显式报, 不能当 0",
#     同一条原则此前没有应用到容量本身。)
def ub_cap_for(soc: str | None = None) -> tuple:
    """返回 (UB 字节数, 出处)。取不到就**抛**, 不回落。"""
    try:
        from core import socspec
        sp = socspec.spec(soc)
        return sp["cap"]["UB"], f"socspec/{sp['soc']}"
    except Exception as e:
        key = (soc or "").lower()
        if key in UB_CAP:
            return UB_CAP[key], f"内置表/{key}"
        raise RuntimeError(
            f"取不到本机 UB 容量, 拒绝按默认值判 (socspec: {e})。\n"
            f"  容量判据静默回落会制造假安全 —— 用别的芯片的数算出一个'通过'比报错危险。\n"
            f"  补法: 确认平台 ini 可读 (python -m core.socspec), 或显式传 --soc 指定已标定的型号 "
            f"{sorted(UB_CAP)}。")


def _expand_ceil(e: str) -> str:
    """把 Ceil(a, b) 重写为 Python 的向上整除 (-((-a)//b)), 按括号深度切参数.

    不用正则: Ceil 的参数里普遍含括号 (uint32_t(2) / sizeof(...) / 嵌套 Ceil), 用 [^()]+ 之类的
    字符类会被这些括号截断, 整条表达式失解 -> 该缓冲不计入 UB 总量 (静默漏报, 见 _split_args 注释).
    """
    while True:
        i = e.find("Ceil(")
        if i < 0:
            return e
        j, depth, comma = i + 4, 0, -1
        while j < len(e):
            c = e[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            elif c == "," and depth == 1:
                comma = j
            j += 1
        if depth != 0 or comma < 0:
            return e            # 括号不配平: 保持原样, 由 _resolve 报 UNRESOLVED
        a = _expand_ceil(e[i + 5:comma])
        b = _expand_ceil(e[comma + 1:j])
        e = e[:i] + f"(-((-({a}))//({b})))" + e[j + 1:]


def _resolve(expr: str, vals: dict):
    """把 tiling 表达式转数值. 支持 * 与 sizeof(T)->sizeof_T, tiling->a.b->a_b/尾名."""
    e = expr.strip()
    #  ★换行必须先折平: 内核里的字节表达式常写成两行相加, 而 Python 的 eval 把裸换行
    #    当成语句结束 —— 于是 "a * 4\n + b * 4" 抛 SyntaxError, 该缓冲被记成 UNRESOLVED
    #    而不计入总量 (softmax 的 COL 路径 wBuf 即由此漏掉, 2026-09-22)。
    e = " ".join(e.split())
    e = re.sub(r"sizeof\(\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\)", lambda m: f'sizeof_{m.group(1).split("::")[-1]}', e)
    # CANN 通用向上取整辅助 Ceil(a,b) -> -(-a//b); AscendC 各算子普遍用它做对齐, 不认它就整条表达式失解.
    # 同类的 uint32_t(x)/int32_t(x) 是构造式转型, 求值时等价于 x.
    e = re.sub(r"\b(?:u?int(?:8|16|32|64)_t|float|half)\s*\(([^()]*)\)", r"(\1)", e)
    #  ★C 风格强制转型 `(uint32_t)x` 与构造式转型 `uint32_t(x)` 是两种写法, 上一行只认后者。
    #    前者在内核里同样常见 (FIAS 的 `(uint32_t)mBaseSize * 576 * sizeof(Q_T)`), 不剥掉就整条
    #    表达式失解 —— 症状是该缓冲被记成 UNRESOLVED 而**不计入总量**, 即本模块最危险的静默假阴性。
    e = re.sub(r"\(\s*(?:u?int(?:8|16|32|64)_t|size_t|float|half|bfloat16_t)\s*\)", "", e)
    #  ★`static_cast<uint32_t>(x)` 是第三种转型写法, 上两行都不认。AscendC 内核里
    #    InitBuffer 的字节表达式几乎全是这个写法 (softmax 四个 kernel 函数全中),
    #    不剥掉则整条表达式失解 -> 缓冲不计入总量 -> 报"0/253952 B, OK"。
    #    2026-09-22 实撞: softmax ROW 路径整行装 UB, K=16384 真机 507035, 而本模块
    #    对该文件给的是 0% 占用。剥转型是**单向增大**已解析总量, 只会让判定更严。
    e = re.sub(r"\b(?:static|reinterpret|const)_cast\s*<[^<>]*>\s*", "", e)
    #  ★整数字面量的 C 后缀 `0U` / `1UL` / `2ULL` 在 Python 里不是合法字面量, eval 直接抛异常。
    #    内核里的循环头几乎都写成 `for (uint32_t i = 0U; i < N; ++i)`, 不剥后缀就解不出循环下界,
    #    于是整个缓冲数组的份数被记成 UNRESOLVED。
    e = re.sub(r"\b(\d+)[uU][lL]{0,2}\b|\b(\d+)[lL]{1,2}\b", lambda m: m.group(1) or m.group(2), e)
    e = _expand_ceil(e)
    e = e.replace("tiling->", "").replace("->", "_").replace(".", "_")
    # 取每个标识符末段 (matmulTiling_baseN -> baseN 若未定义则尝试尾名)
    def repl(m):
        tok = m.group(0)
        if tok in vals:
            return str(vals[tok])
        tail = tok.split("_")[-1]
        if tail in vals:
            return str(vals[tail])
        return tok
    e2 = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, e)
    #  ★C 的整数除法向零截断, Python 的 `/` 是浮点 —— 不换成 `//` 则每一个
    #    `((v + a - 1) / a) * a` 形态的**对齐取整**都算成浮点再在最后 int() 一次,
    #    结果系统性偏大 (d=256 时 AlignUp(d,64) 算出 319 而不是 256, +25%)。
    #    字节表达式在 C 里全是整型量, 整体换 `//` 是等价的 (被除数非负)。
    e2 = re.sub(r"(?<![/])/(?![/])", "//", e2)
    env = {"__builtins__": {}, "max": max, "min": min}
    try:
        return int(eval(e2, env, {})), None
    except Exception:
        pass
    #  ★三元 `cond ? A : B`: 先按 Python 条件式试; 条件里含解不出的量 (常是
    #    `IsF32()` 这种成员谓词) 就退成 max(A, B) —— 容量判据取**两支的大者**是
    #    安全方向 (同一缓冲两支容量不同这件事另有 U6 专管)。
    m = re.match(r"^(.*?)\?(.*?):(.*)$", e2)
    if m:
        c, x, y = (t.strip() for t in m.groups())
        for cand in (f"(({x}) if ({c}) else ({y}))", f"max(({x}), ({y}))"):
            try:
                return int(eval(cand, env, {})), None
            except Exception:
                continue
    return None, f"UNRESOLVED: {expr} -> {e2}"


#  共享/系统目录不是算子源码树。原推导是「入参文件的上级目录」, 对
#  `tempfile.NamedTemporaryFile(suffix=".h")` 造出的 /tmp/xxx.h 就解析成 **/tmp 整棵树** ——
#  别人的构建 scratch 全在那里, 于是一次单文件检查变成扫几万个文件、并 eval 其中每一条
#  constexpr 初始化表达式。alignlint --selftest 100% CPU 跑满超时就是这一条
#  (2026-09-29 用 faulthandler 定位到 ub_budget.py 的 eval 行)。
#  这些目录下**永远**不存在「该文件所属的算子源码树」, 所以遇到就只看入参那一个文件。
_UNBOUNDED_ROOTS = frozenset({"/", "/tmp", "/var/tmp", "/dev/shm", "/home", "/root", "/usr", "/var"})


def _const_scan_files(src: str) -> list:
    """抓常量时要读哪些源文件。

    入参在 op_kernel/op_host 布局里 -> 扫整个算子根目录 (常量在 host, InitBuffer 在 kernel);
    否则扫它所在目录; 而所在目录若是共享/系统目录, **只看入参这一个文件** ——
    在那里「上级目录」跟算子毫无关系, 扫它既慢又会把陌生项目的常量混进来。
    """
    path = os.path.abspath(src)
    srcdir = os.path.dirname(path)
    if os.path.basename(srcdir) in ("op_kernel", "op_host"):
        root = os.path.dirname(srcdir)
    else:
        root = srcdir
    if os.path.normpath(root) in _UNBOUNDED_ROOTS:
        return [path] if os.path.isfile(path) else []
    out = []
    for r, _d, fs in os.walk(root):
        for f in fs:
            if f.endswith((".h", ".cpp", ".cc")):
                out.append(os.path.join(r, f))
    return out


def constant_exprs(src: str) -> dict:
    """同 source_constants, 但返回**未求值**的 name -> 表达式, 供代入 --values 后做不动点求解。"""
    pat = re.compile(r"constexpr\s+\w[\w:]*\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
    #  ★缓冲大小常常先落在一个**普通局部变量**上 (`uint32_t mm12RightSize = s2BaseSize*576*sizeof(T);`),
    #    再喂给 Init。只收 constexpr 就解不出它, 该缓冲被记成 UNRESOLVED 而不计入总量。
    #    收局部变量有歧义风险 (同名在不同分支赋不同值), 所以**只收全仓里恰好赋值一次的那些**;
    #    赋值多次的一律不收, 宁可报未解析也不能取错一个值。
    localpat = re.compile(r"\b(?:u?int(?:8|16|32|64)_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*([^;{}]+);")
    local_hits = {}
    out = {}
    for f in _const_scan_files(src):
        try:
            _t = open(f, errors="ignore").read()
            for m in pat.finditer(_t):
                out.setdefault(m.group(1), m.group(2).strip())
            for m in localpat.finditer(_t):
                local_hits.setdefault(m.group(1), []).append(m.group(2).strip())
        except OSError:
            pass
    for name, exprs in local_hits.items():
        #  同一个名字在多份姊妹文件里**逐字相同**地算出来 (量化/非量化各一份 kernel), 仍可安全取用;
        #  只有真的出现不同表达式时才放弃 —— 取错一个值比报未解析危险得多。
        if name in out or len(set(exprs)) != 1:
            continue
        out[name] = exprs[0]
    return out


#  ★`sizeof(T)` 在 UB 分配里无处不在 (`n * dimDAlign * sizeof(float)`), 而解析器把它
#  规范成 `sizeof_float` 这个符号后就没人认了 —— 于是整条表达式 UNRESOLVED, 缓冲不计入总量。
#  2026-09-16 写 U5 时发现: 32 个偏移累加项**一个都解不出**, 全卡在 sizeof 上。
#  模板参数 (sizeof_T1 之类) 不在这里猜, 交给 --values —— 猜错比解不出更危险。
_SIZEOF_CONST = {
    "sizeof_float": 4, "sizeof_int32_t": 4, "sizeof_uint32_t": 4, "sizeof_int": 4,
    "sizeof_int64_t": 8, "sizeof_uint64_t": 8, "sizeof_double": 8,
    "sizeof_half": 2, "sizeof_bfloat16_t": 2, "sizeof___bf16": 2,
    "sizeof_int16_t": 2, "sizeof_uint16_t": 2,
    "sizeof_int8_t": 1, "sizeof_uint8_t": 1, "sizeof_bool": 1, "sizeof_char": 1,
}


def source_constants(src: str) -> dict:
    """自动从源码目录抓 constexpr 常量 (UBCALSIZE/BEST_BASE_N/A8W8_UBRESTBYTES/DETER_UB_SIZE/BUFFER_NUM 等),
    避免手动传值造成 GIGO 假阳性 (实战验证暴露的失效模式). 值支持简单 * 表达式."""
    d = dict(_SIZEOF_CONST)
    # 常量常在 op_host, InitBuffer 在 op_kernel — 扫算子根目录整棵树 (op_kernel 上两级);
    # 上级目录是共享目录时只看这一个文件, 见 _const_scan_files。
    pat = re.compile(r"constexpr\s+\w[\w:]*\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
    for f in _const_scan_files(src):
        try:
            for m in pat.finditer(open(f, errors="ignore").read()):
                name, expr = m.group(1), m.group(2).strip()
                try:
                    d.setdefault(name, int(eval(re.sub(r"[uU]?[lL]*$", "", expr), {"__builtins__": {}}, dict(d))))
                except Exception:
                    pass
        except Exception:
            pass
    # 常见 sizeof 默认 (可被 --values 覆盖)
    d.setdefault("sizeof_float", 4); d.setdefault("sizeof_DTYPE_OUT", 4)
    d.setdefault("sizeof_cT", 4); d.setdefault("sizeof_T", 4); d.setdefault("sizeof_bfloat16_t", 2)
    # 常见 tiling->常量 别名 (ubCalSize=UBCALSIZE, baseN=BEST_BASE_N, ubRestBytes 取 a8w8)
    if "UBCALSIZE" in d: d.setdefault("ubCalSize", d["UBCALSIZE"])
    if "BEST_BASE_N" in d: d.setdefault("baseN", d["BEST_BASE_N"])
    if "A8W8_UBRESTBYTES" in d: d.setdefault("ubRestBytes", d["A8W8_UBRESTBYTES"])
    return d



# ---- U5: **池内二次分配的偏移累加超过池容量** —— InitBuffer 那一层看不见的越界。
#      AscendC 的 UB 常常分两层要: 先 `pipe->InitBuffer(que, cap)` 向硬件要一大块,
#      再 `que.GetWithOffset<T>(n, off)` 从这块里切给各个张量, off 由 `off += ...` 逐个推进。
#      本模块原来只解析第一层 —— 只要 `cap <= UB 容量` 就报 OK。但第二层切出去的总和
#      **可以超过 cap 而第一层一字不变**, 那时报的仍是"OK, headroom 充足"。
#      2026-09-16 sparse_flash_attention_grad 实撞: 把一个临时缓冲从
#      `(UB_ROW_SIZE*2+1)*dimDAlign` 扩到 `UB_ROW_SIZE*3*dimDAlign` (+6.9 KB),
#      `InitBuffer(vecQue, 191*1024)` 没动, 本模块两版给出**逐字相同**的
#      "195584/196608 B (99%), OK", 而真机 12 组里 10 组 507015 (aicore trap)。
#      ★连不相干的路径一起崩: off 一推进, 它后面每个张量的位置都移, 而 UB 布局是所有
#        模板实例共用的 —— 所以改确定性路径的缓冲, 非确定性路径同样越界。
#      判法: 按顶层 if/else 分支分别累加 off, 取最大的那一支与 cap 比。互斥分支取大者
#      (同一时刻只有一支生效), 这与本模块对 InitBuffer 同名缓冲的处理一致。
def pool_offset_delta(text_old: str, text_new: str, vals: dict) -> list:
    """改动**让池内偏移多占了多少字节** —— U5 真正可靠的用法。

    为什么是差分而不是绝对值: UB 分配几乎都依赖 tiling 决定的运行时量
    (sftBaseM = get_G()、dimDAlign、sftDataSize …), 静态解不出, 靠 --values 给就是
    GIGO —— 给错一个数, 绝对值就差好几倍, 判据反而误导人。
    2026-09-16 实测: 同一组 values 下 v9a/v12a 的绝对值都高估 5.7 倍 (我把 sftBaseM
    给成了 128 而实际是 64), **但两者的差值 13056 B 分毫不差** —— 因为错的那部分在
    两版里相同, 相减就消掉了。所以差分不依赖 values 给得准, 只依赖两版用同一组。

    用途: 改 UB 分配之前先问"这一改多占多少"。UB 是硬约束且通常接近满载
    (sfag 是 191/192 KB), 多占一个字节都可能是 507015。
    """
    out = []
    pools_old = _pool_offvars(text_old)
    for que, ovs in _pool_offvars(text_new).items():
        for ov in sorted(ovs):
            if ov not in pools_old.get(que, set()):
                out.append({"check": "U5D_new_offvar", "que": que, "offvar": ov,
                            "err": f"新版多出一个偏移变量 `{ov}` —— 无法与旧版对比, 请人工核。"})
                continue
            a, why_a = _peak_offset(text_old, ov, vals)
            b, why_b = _peak_offset(text_new, ov, vals)
            if a is None or b is None:
                out.append({"check": "U5D_unresolved", "que": que, "offvar": ov,
                            "err": (f"`{ov}` 的峰值算不出{why_a or why_b} —— 这条没做成, "
                                    f"不是做了没问题。")})
                continue
            d = b - a
            if d > 0:
                out.append({"check": "U5D_grew", "que": que, "offvar": ov, "delta": d,
                            "old": a, "new": b,
                            "err": (f"池 `{que}` 的偏移 `{ov}` 比旧版**多占 {d} B**。"
                                    f"UB 是硬约束且常接近满载 (改之前先确认余量) —— "
                                    f"超出的症状是 aicore trap (507015), 而且会**连累不相干的路径**: "
                                    f"偏移一推进, 后面每个张量都移位, UB 布局是所有模板实例共用的。"
                                    f"★注意这里只有**差值**可信: 绝对值依赖 tiling 运行时量, "
                                    f"--values 给错就差好几倍。")})
            else:
                out.append({"check": "U5D_ok", "que": que, "offvar": ov, "delta": d,
                            "old": a, "new": b, "err": None})
    return out


def _pool_offvars(text: str) -> dict:
    """{池名: {被 GetWithOffset 当 offset 用的变量名}}"""
    pools = {}
    for m in re.finditer(r"InitBuffer\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^;()]+?)\s*\)\s*;", text):
        pools.setdefault(m.group(1), set())
    for que in list(pools):
        for v in re.findall(rf"{re.escape(que)}\s*\.\s*GetWithOffset\s*<[^>]*>\s*\([^,]*,\s*([A-Za-z_]\w*)\s*\)", text):
            if "offset" in v.lower():
                pools[que].add(v)
    return pools


#  ★变异自检的豁免声明 (2026-10-29)。`opdbg --mutation` 会把每条判据换成恒返回空, 看
#    自检退出码翻不翻; 翻不了的报成"摆设"。`pool_offset_report` 翻不了是**对的** ——
#    它按设计就没挂进 `analyze()`、不会自动跑 (理由见它自己的 docstring: 未完成且会严重
#    高估, 2026-09-16 的样本外检验里对一个已知良品也报"越界 25600 B")。
#    给它补自检等于把一条**已知无区分能力**的判据接上线, 方向是反的。
#    ★等它真修好、重新挂进 analyze() 时, 这条豁免必须同时删掉。
_MUTATION_EXEMPT = {
    "pool_offset_report": "按设计未挂进 analyze(), 不会自动跑; 当前实现会严重高估 "
                          "(2026-09-16 样本外检验: 对已知良品 v9a 也报越界), "
                          "修好并重新接线时必须删掉本豁免",
}


def pool_offset_report(text: str, vals: dict) -> list:
    """★★ 未完成, 当前**会严重高估**, 所以没有挂进 analyze(), 不会自动跑。 ★★

    2026-09-16 又做了一次样本外检验, 结论是**这条判据当前没有区分能力**, 不许拿它下结论:
      拿 sfag 的两个带标签样本跑 —— v9a (已交付、端到端验收过、确定能跑) 与 v16c (确定崩
      507015) —— 两者 ubOffset 峰值**完全相同**, 都是 221184 vs 池 195584, 即对一个已知
      良品也报"越界 25600 B"。判据对良品报警, 它报的就不是越界。
      ★当时差一点就拿这个数字去改 UB 布局了 —— 是"已知良品对照"把它挡住的。
      所以给它补一条接线前提 (见下), 而不是修修补补继续用。

    2026-09-16 实测: 对能正常跑的 v9a 报 peak=2076928 vs cap=195584 (超 10 倍) ——
    因为它把整个文件里所有 `ubOffset +=` 都加起来了, 而真实情况是:
      1. 每个 Init 分支从 `uint32_t ubOffset = 0;` 重新开始累加 (第 394 行);
      2. 中途还有 `rowsumUbOffset = ubOffset;` 这类快照/重置 (第 421 / 511 行);
      3. 顶层 if/else 是互斥的, 同一时刻只有一支生效。
    当前实现只处理了第 3 条的一个粗糙版本, 1 与 2 完全没处理。

    要做对需要一个小型数据流分析: 按 offset 变量的**声明/重置点**切段, 段内线性累加,
    分支取大者。判据过宽就是狼来了, 所以在做对之前不接线。

    ★接线前提 (硬性, 自检里已固化成 U5_control): 挂进 analyze() 之前, 必须在**真仓的一个
    已知良品**上跑出"不报警"。判据自检里那些手写的最小用例证明不了这一点 —— 它们是照着
    判据的逻辑写的, 判据错在哪, 用例就跟着错在哪。只有真仓里一个确定能跑的版本是外部证据。
    """
    out = []
    #  1) 池: InitBuffer(<que>, <cap>) 只有两个实参且第二个能求值的那种
    pools = {}
    for m in re.finditer(r"InitBuffer\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^;()]+?)\s*\)\s*;", text):
        que, cap_e = m.group(1), m.group(2)
        cap, _err = _resolve(cap_e, vals)
        if cap:
            pools[que] = int(cap)
    if not pools:
        return out
    #  2) 每个池: 找它的 GetWithOffset 用哪个变量当 offset
    for que, cap in pools.items():
        offvars = set(re.findall(rf"{re.escape(que)}\s*\.\s*GetWithOffset\s*<[^>]*>\s*\([^,]*,\s*([A-Za-z_]\w*)\s*\)", text))
        offvars = {v for v in offvars if "offset" in v.lower()}
        for ov in sorted(offvars):
            peak, why = _peak_offset(text, ov, vals)
            if peak is None:
                #  ★解不出时必须**说出来**, 不能静默跳过 —— 否则"没查成"与"查过没问题"
                #  长得一模一样。这正是 shguard S7 管的那个形态, 判据自己也会犯。
                out.append({"check": "U5_unresolved", "que": que, "offvar": ov,
                            "err": (f"池 `{que}` 的偏移 `{ov}` **没能算出峰值**{why} —— "
                                    f"这条检查没做成, 不是做了没问题。"
                                    f"给 --values 补上解不出的那些量 (dimDAlign / sftBaseM 之类) 再跑。")})
                continue
            if peak > cap:
                out.append({"check": "U5_pool_offset_overflow", "que": que, "offvar": ov,
                            "peak": peak, "cap": cap,
                            "err": (f"池 `{que}` 用 InitBuffer 开了 {cap} B, 而 `{ov}` 累加到 "
                                    f"**{peak} B**, 超出 {peak - cap} B。{why} "
                                    f"InitBuffer 那一层看不出来 —— 它的参数根本没变。"
                                    f"真机症状是 aicore trap (507015), 且会**连累不相干的路径**: "
                                    f"offset 一推进, 后面每个张量都移位, 而 UB 布局是所有模板实例共用的。")})
            else:
                out.append({"check": "U5_ok", "que": que, "offvar": ov, "peak": peak, "cap": cap,
                            "err": None})
    return out


def _peak_offset(text: str, ov: str, vals: dict):
    """算 `ov` 在一条**可执行路径**上能累加到的最大值。

    2026-09-16 第一版是把整个文件的 `ov +=` 全加起来, 对能正常跑的代码报出超 10 倍的
    假警 —— 因为真实结构是: 声明处归零, 然后 if/else 互斥, 其中一支还以 early return
    结束。三种结构都得按"路径"算, 不是按"出现次数"算:
      顺序语句  -> 累加
      if/else   -> 取两支的大者 (同一时刻只走一支)
      if 内 return -> 分叉: 一条是"进 if 并结束", 一条是"不进 if 继续往下"
    """
    lines = text.split("\n")
    #  从 ov 的声明处起算 (声明即归零点); 找不到声明就不下结论
    decl = None
    for i, ln in enumerate(lines):
        if re.search(rf"\b\w+\s+{re.escape(ov)}\s*=\s*0\s*;", ln):
            decl = i
            break
    if decl is None:
        return None, f" (找不到 `{ov}` 的归零声明, 无法确定从哪起算)"
    #  函数体的结束: 从声明处往回找到函数起始的 `{`, 再配对
    depth, fn_end = 0, len(lines)
    for j in range(decl, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth < 0:
            fn_end = j
            break
    miss = set()

    def _val(expr):
        v, err = _resolve(expr, vals)
        if v is None:
            for tok in re.findall(r"[A-Za-z_]\w*", (err or "").split("->")[-1]):
                if tok not in vals and tok not in _SIZEOF_CONST:
                    miss.add(tok)
            return 0
        return int(v)

    def _blk_end(i):
        """lines[i] 含块起始 `{`, 返回配对 `}` 的行号。"""
        d = 0
        for j in range(i, fn_end + 1):
            d += lines[j].count("{") - lines[j].count("}")
            if d <= 0 and j > i:
                return j
        return fn_end

    def _walk(i, end):
        """返回 (该段能达到的最大累加, 是否在段内 return)。"""
        total = 0
        while i < end:
            ln = lines[i]
            m = re.search(rf"\b{re.escape(ov)}\s*\+=\s*([^;]+);", ln)
            if m:
                total += _val(m.group(1))
            if re.match(r"\s*return\b", ln):
                return total, True
            if re.match(r"\s*(if|for|while)\b.*\{\s*$", ln) or re.match(r"\s*\}?\s*else\b.*\{\s*$", ln):
                be = _blk_end(i)
                sub, sub_ret = _walk(i + 1, be)
                nxt = lines[be] if be < len(lines) else ""
                if re.match(r"\s*\}\s*else\b", nxt):
                    ee = _blk_end(be)
                    sub2, _ = _walk(be + 1, ee)
                    total += max(sub, sub2)
                    i = ee + 1
                    continue
                if sub_ret:
                    #  分叉: 进 if 并 return vs 不进 if 继续
                    rest, _ = _walk(be + 1, end)
                    return total + max(sub, rest), False
                total += sub
                i = be + 1
                continue
            i += 1
        return total, False

    peak, _ = _walk(decl + 1, fn_end)
    if miss:
        names = ",".join(sorted(miss)[:12])
        return None, (f" ({len(miss)} 个量解不出: {names}"
                      f"{' 等' if len(miss) > 12 else ''} —— 用 --values 补上就能算)")
    return peak, "按路径取大者 (if/else 与 early return 都算作互斥分支)。"

def analyze(src, vals: dict, soc: str | None = None, mem: str = "UB") -> dict:
    #  src 可以是一个文件, 也可以是**一组**文件 —— UB 是每个核共用的一块, 同一个核上所有参与者
    #  的分配都要加起来。arch35 的 FIA 就分在两处: kernel 文件用 BufferManager 开 mm1/mm2 结果池,
    #  vec 文件用 InitBuffer 开 softmax/累加器; 只喂一个文件会报"headroom 充足"而实际只剩 4KB。
    srcs = [src] if isinstance(src, str) else list(src)
    merged = source_constants(srcs[0]); merged.update(vals)   # 源码常量自动填, --values 覆盖
    #  ★不动点求解: 源码里的 constexpr 常常**依赖模板参数** (FIA 的 ACT_BYTES <- SOFTMAX_BUF_BYTES
    #    <- mBaseSize/CV_RATIO), 而模板参数只能由 --values 给。原实现在扫源码时就地求值, 那时
    #    --values 还没并进来, 于是这一串永远解不出, 缓冲被记成 UNRESOLVED 而不计入总量。
    #    改成先并 --values, 再对未解出的表达式反复代入直到不再有进展。
    raw = constant_exprs(srcs[0])
    for _ in range(8):
        progress = False
        for name, expr in raw.items():
            if name in merged:
                continue
            v, _err = _resolve(expr, merged)
            if v is not None:
                merged[name] = v
                progress = True
        if not progress:
            break
    vals = merged
    text = "\n".join(open(f).read() for f in srcs)
    # InitBuffer(que, num, bytes) 或 InitBuffer(buf, bytes)
    bufs = []
    #  ★BufferManager<BufferType::UB> 的整池分配: `xxx.Init(pipe, bytes)`。
    #    它不是 InitBuffer, 原解析器完全看不见 —— 这正是本模块自己警告过的那类**静默假阴性**:
    #    FIA arch35 的 ubBufferManager_ 一次开 160KB (mm1Res x2 + mm2Res x2), 漏掉它之后
    #    工具报 34% 占用, 真值是 98%。2026-09-11 由 FIAS 反向补上。
    ub_mgrs = set(re.findall(r"BufferManager\s*<\s*BufferType::" + mem + r"\s*>\s*\*?\s*([A-Za-z_]\w*)", text))
    #  ★策略类的池: `BuffersPolicyXxx<BufferType::M, ...> pol;` + `pol.Init(mgr, size)`。
    #    份数由类名定 (见 POLICY_MULT), 字面上看不出来。
    for pm in re.finditer(r"(BuffersPolicy\w+)\s*<\s*BufferType::" + mem + r"\b[^;]*?>\s*\*?\s*([A-Za-z_]\w*)\s*;", text):
        cls, pol = pm.group(1), pm.group(2)
        mult = POLICY_MULT.get(cls)
        for m in re.finditer(re.escape(pol) + r"\s*(?:\.|->)\s*Init\s*\(([^;]+?)\)\s*;", text, re.S):
            args = [x.strip() for x in _split_args(m.group(1))]
            if len(args) < 2:
                bufs.append({"buf": f"{pol}<{cls}>", "num": f"UNPARSED(argc={len(args)})",
                             "bytes_expr": " , ".join(args)[:80], "total_bytes": None,
                             "err": f"{cls}::Init 参数个数 {len(args)} < 2, 未计入总量"})
                continue
            nsz, err = _resolve(args[1], vals)
            if mult is None:
                bufs.append({"buf": f"{pol}<{cls}>", "num": "UNKNOWN_POLICY",
                             "bytes_expr": args[1], "total_bytes": None,
                             "err": f"策略类 {cls} 不在 POLICY_MULT 表里, 开几份未知 —— 未计入总量"})
                continue
            bufs.append({"buf": f"{pol}<{cls}>", "num": str(mult), "bytes_expr": args[1],
                         "total_bytes": (nsz * mult) if nsz is not None else None, "err": err})
    #  ★手工向 BufferManager 要的槽: `mgr.AllocBuffer<...>(size)`, 常写在 for 循环里,
    #    份数是**运行期**的循环上界。解不出上界时必须记成 UNRESOLVED, 不能按 1 份算。
    for mgr in sorted(ub_mgrs):
        pat = (r"([A-Za-z_]\w*\s*(?:\[[^\]]*\])?)\s*=\s*" + re.escape(mgr)
               + r"\s*(?:\.|->)\s*(?:template\s+)?AllocBuffer\s*<[^;]*?>\s*\(([^;()]*?)\)\s*;")
        for m in re.finditer(pat, text, re.S):
            rep, rep_note = _loop_repeat(text, m.start(), m.group(1).strip(), vals)
            nsz, err = _resolve(m.group(2).strip(), vals)
            total = (nsz * rep) if (nsz is not None and rep is not None) else None
            bufs.append({"buf": f"{m.group(1).strip()}<-{mgr}.AllocBuffer",
                         "num": (str(rep) if rep is not None else rep_note),
                         "bytes_expr": m.group(2).strip(), "total_bytes": total,
                         "err": err or (None if rep is not None else rep_note)})
    #  UB 侧 BufferManager 的 Init(pipe, bytes) 是**一次整池分配** (与 TBuf/TQue 并列, 都要计入);
    #  L1/L0 侧它是**容量声明** (整块片上存储的上限), 计入就会把容量当成占用再加一遍。
    for mgr in (sorted(ub_mgrs) if mem == "UB" else []):
        for m in re.finditer(re.escape(mgr) + r"\s*(?:\.|->)\s*Init\s*\(([^;]+?)\)\s*;", text, re.S):
            args = [x.strip() for x in _split_args(m.group(1))]
            if len(args) != 2:
                bufs.append({"buf": f"{mgr}(pool)", "num": f"UNPARSED(argc={len(args)})",
                             "bytes_expr": " , ".join(args)[:80], "total_bytes": None,
                             "err": f"BufferManager<UB>::Init 参数个数 {len(args)} 非 2, 未计入总量"})
                continue
            nsz, err = _resolve(args[1], vals)
            bufs.append({"buf": f"{mgr}(pool)", "num": "1", "bytes_expr": args[1],
                         "total_bytes": nsz, "err": err})
    for m in (re.finditer(r"InitBuffer\s*\(([^;]+?)\)\s*;", text, re.S) if mem == "UB" else []):
        args = [a.strip() for a in _split_args(m.group(1))]
        #  ★缓冲数组在 for 循环里逐个 InitBuffer 时, 字面上只出现一次调用, 但实际开了 N 块。
        #    按字面记账会把 N 块只算 1 块 —— 又一例静默假阴性 (本模块最危险的那一类)。
        #    判据: 目标写成 NAME[idx] 且 idx 是某个**前置 for 头**的循环变量, 则乘以该循环的次数。
        rep, rep_note = _loop_repeat(text, m.start(), args[0] if args else "", vals)
        if len(args) == 3:
            name, num, sz = args
            nnum, _ = _resolve(num, vals)
            nsz, err = _resolve(sz, vals)
            total = (nnum * nsz * rep) if (nnum is not None and nsz is not None and rep is not None) else None
            bufs.append({"buf": name, "num": (num if rep in (1, None) else f"{num} x {rep}({rep_note})"),
                         "bytes_expr": sz, "total_bytes": total, "pos": m.start(),
                         "err": err or (None if rep is not None else rep_note)})
        elif len(args) == 2:
            name, sz = args
            nsz, err = _resolve(sz, vals)
            total = (nsz * rep) if (nsz is not None and rep is not None) else None
            bufs.append({"buf": name, "num": ("1" if rep in (1, None) else f"{rep}({rep_note})"),
                         "bytes_expr": sz, "total_bytes": total, "pos": m.start(),
                         "err": err or (None if rep is not None else rep_note)})
        else:
            # ★参数个数异常 = 解析失败, 必须显式记账. 早期实现直接忽略, 使漏掉的缓冲不计入总量,
            #   工具报"headroom 充足"而真机 UB 越界 (静默假阴性比不报更危险).
            bufs.append({"buf": args[0] if args else "?", "num": f"UNPARSED(argc={len(args)})",
                         "bytes_expr": " , ".join(args[1:])[:80], "total_bytes": None, "pos": m.start(),
                         "err": f"InitBuffer 参数个数 {len(args)} 非 2/3, 无法定量 —— 该缓冲未计入总量"})
    #  U10: 按声明类型给每块缓冲归属片上存储。TSCM / A1,B1 绑的是 L1, 不占 UB。
    #  找不到声明时按 UB 计并标记 —— 少算比多算危险。
    if mem == "UB":
        for b in bufs:
            if str(b.get("num", "")).startswith("UNPARSED"):
                continue
            cls, why = storage_class(text, b["buf"])
            b["storage"] = cls
            if cls != "UB":
                b["storage_why"] = why
    # ★同一缓冲名出现多次 = 互斥分支 (if constexpr / if-else) 里各 InitBuffer 一次, 运行时只生效一个,
    #   故按**最大者**计账而非求和 (求和会虚报溢出, 把能过的方案判死).
    per_name = {}
    for b in bufs:
        if b["total_bytes"] is None:
            continue
        if b.get("storage") in ("L1", "L0C"):
            continue
        per_name[b["buf"]] = max(per_name.get(b["buf"], 0), b["total_bytes"])
    total = sum(per_name.values())
    if mem == "UB":
        cap, _cap_src = ub_cap_for(soc)
    else:
        #  L1 的上限读源码里 `BufferManager<BufferType::L1> mgr; mgr.Init(pipe, <字面量>)` 的字面量。
        #  ★容量只认**该空间自己那个管理器**的 Init 字面量。早先按 "BufferType::L1 之后 4000 字符内
        #    的任意 Init" 去找, 会跨过去匹配到 L0A 管理器的 Init(tPipe, 65536), 于是 512K 的 L1
        #    被判成 64K, 报出 800% 占用 —— 这类"容量取错"比漏算缓冲更误导。
        caps = []
        for mgr in sorted(ub_mgrs):
            for cm in re.finditer(re.escape(mgr) + r"\s*(?:\.|->)\s*Init\s*\(([^;]+?)\)\s*;", text, re.S):
                args = [x.strip() for x in _split_args(cm.group(1))]
                if len(args) != 2:
                    continue
                v, _e = _resolve(args[1], vals)
                if v is not None:
                    caps.append(v)
        if not caps:
            raise SystemExit(f"{mem} 容量未知: 源码里找不到 BufferManager<BufferType::{mem}>::Init(pipe, <字面量>)。"
                             f"不猜默认值 —— 猜错会把越界报成 headroom 充足。")
        cap, _cap_src = max(caps), "源码 Init 字面量"
    # ★经验安全余量: 910B2 上实测 total==196608 (正好顶满 192KB) 真机报
    #   "VEC instruction error: the ub address out of bounds", 而 195584 正常 —— 说明可用 UB 略小于
    #   标称值 (TPipe 自身有簿记开销). 故留 1KB 余量作为判定线, 依据是 gmfr perf2 的真机对照.
    safe_margin = 1024
    unresolved = [b for b in bufs if b["total_bytes"] is None]
    scopes = _kernel_scopes(text)
    for b in bufs:
        if "pos" in b:
            b["fn"] = _scope_of(scopes, b["pos"])
    perfn = per_function_totals(bufs, cap, safe_margin) if len(scopes) > 1 else None
    out = {"buffers": bufs, "total_bytes_resolved": total, "cap_bytes": cap,
           "headroom_bytes": cap - total, "safe_margin_bytes": safe_margin,
           "over_cap": total > cap - safe_margin,
           "unresolved_count": len(unresolved),
           #  U10 之后"求不出尺寸"要分口径: 一块求不出的 **L1** 缓冲不会让 **UB** 的和
           #  变成下界。调用方判"这个和能不能用"时该看这一个。
           "unresolved_ub_count": len([b for b in unresolved
                                       if b.get("storage") not in ("L1", "L0C")]),
           "note": (_ub_note(bool(unresolved), total > cap - safe_margin, safe_margin))}
    offub = [b for b in bufs if b.get("storage") in ("L1", "L0C")]
    if offub:
        out["off_ub_buffers"] = [{"buf": b["buf"], "storage": b["storage"],
                                  "bytes": b["total_bytes"], "why": b.get("storage_why")}
                                 for b in offub]
        out["note"] = ("★U10: %d 块缓冲按声明类型不在 UB (已从总量剔除): %s。"
                       % (len(offub), ", ".join(f"{b['buf']}->{b['storage']}" for b in offub))
                       ) + out["note"]
    unknown = [b for b in bufs if b.get("storage") == "未知"]
    if unknown:
        out["storage_unknown"] = [b["buf"] for b in unknown]
        out["note"] = ("★U10: %d 块缓冲找不到声明, **按 UB 计**(保守): %s。"
                       % (len(unknown), ", ".join(b["buf"] for b in unknown))) + out["note"]
    if perfn is not None:
        out["per_function"] = perfn
        pn = practical_note(perfn["worst_bytes"], cap)
        if pn:
            out["practical_warning"] = pn
            out["note"] = pn + " " + out["note"]
        # 多函数时, 全加的 over_cap 没有意义 —— 以单函数最大值为准
        out["over_cap"] = perfn["worst_over_cap"]
        out["note"] = perfn["note"] + " " + out["note"]
    return out



#  ---- U10: InitBuffer 的目标不一定在 UB ------------------------------------
#  TSCM / TQueBind 到 A1,B1 绑的是 L1; CO1,CO2 是 L0C。按**声明类型**归属。
#  找不到声明时按 UB 计并标记 —— 少算比多算危险。
_NON_UB_POS = ("A1", "B1", "A2", "B2", "CO1", "CO2", "TSCM")
_UB_POS = ("VECIN", "VECOUT", "VECCALC")


def storage_class(text: str, name: str):
    """返回 (类别, 依据). 类别 ∈ {'UB','L1','L0C','未知'}。"""
    base = re.sub(r"\[.*?\]$", "", name.strip())
    #  ★一条声明常带多个 declarator: `TBuf<TPosition::VECCALC> rawBuf, scoreBuf, pBuf;`
    #    只匹配紧跟模板实参的那一个名字, 会把同一条声明里其余的全报成"找不到声明",
    #    于是它们落进保守分支 —— 不致命, 但把 U10 的输出淹成噪声, 人就不看了。
    m = None
    for d in re.finditer(r"\b(TSCM|TBuf|TQue|TQueBind|TBufPool)\s*(<[^;{}]*?>)?\s+"
                         r"([^;{}()=]+?)\s*;", text):
        names = [re.sub(r"\[[^\]]*\]", "", x).strip() for x in d.group(3).split(",")]
        if base in names:
            m = d
            break
    if not m:
        return "未知", f"找不到 `{base}` 的声明"
    kind, targs = m.group(1), m.group(2) or ""
    if kind == "TSCM":
        return "L1", "TSCM<> 绑 L1"
    poss = re.findall(r"TPosition::(\w+)", targs)
    for pos in poss:
        if pos in ("CO1", "CO2"):
            return "L0C", f"{kind}<... TPosition::{pos} ...>"
    for pos in poss:
        if pos in _NON_UB_POS:
            return "L1", f"{kind}<... TPosition::{pos} ...>"
    for pos in poss:
        if pos in _UB_POS:
            return "UB", f"{kind}<... TPosition::{pos} ...>"
    return "未知", f"`{base}` 声明为 {kind}{targs[:40]}, 位置未识别"

# ---------------------------------------------------------------------------
# U8: 一个文件里有多个 kernel 函数时, 总和是**假的**
# ---------------------------------------------------------------------------
def _kernel_scopes(text: str) -> list:
    """(函数名, 起始偏移, 结束偏移)。只认带 __aicore__ / __global__ 的定义。

    ★为什么必须分组 (cummin 2026-09-20 实证)
      同一个 .h 里放了 row / rowseg / col 三个 kernel 函数, 每个各自 `AscendC::TPipe pipe;`
      —— **各自独立分配整块 UB**, 谁也不叠加谁。把三者的 InitBuffer 加在一起既不是任何
      一个函数的真实用量, 也不是任何有意义的上界。
      实撞代价: 当时工具报 223232/253952 (88%, "OK"), 其中 3 个 UNRESOLVED 项根本属于另一个
      函数; 而真正在跑的那个函数**确实超了 UB**, 上板连报八轮 507035。拆成两个函数、各自
      只留自己需要的缓冲之后立刻不崩 —— 这是 UB 超限的判定性证据。
      总和大于 cap 时更危险: 它会把「三个函数各自都没超」误报成超。
    """
    out = []
    pat = re.compile(r"(?:__aicore__|__global__)[^;{}]*?\b([A-Za-z_]\w*)\s*\([^;{}]*?\)\s*\{")
    for m in pat.finditer(text):
        i = text.index("{", m.end() - 1)
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1), m.start(), j))
    return out


def _scope_of(scopes: list, pos: int) -> str:
    """最内层包含 pos 的函数名; 都不包含则算文件级 (模板外的分配)。"""
    best = None
    for (name, a, b) in scopes:
        if a <= pos <= b and (best is None or a > best[1]):
            best = (name, a)
    return best[0] if best else "<file-scope>"


def per_function_totals(bufs: list, cap: int, safe_margin: int) -> dict:
    """按函数分组求和, 并给出**取最大**的判定 (而不是全加)。"""
    #  同名互斥分支取最大 —— 但只在**同一个函数内**去重: 不同函数里的同名缓冲
    #  (三个 kernel 各有自己的 bVb) 是各自独立的分配, 跨函数合并会少算。
    per = {}
    unres = {}
    for b in bufs:
        fn = b.get("fn") or "<file-scope>"
        if b.get("storage") in ("L1", "L0C"):      # U10: 不占 UB
            continue
        if b.get("total_bytes") is None:
            unres[fn] = unres.get(fn, 0) + 1
            continue
        k = (fn, b["buf"])
        per[k] = max(per.get(k, 0), b["total_bytes"])
    g = {}
    for (fn, _name), v in per.items():
        e = g.setdefault(fn, {"fn": fn, "resolved": 0, "unresolved": 0})
        e["resolved"] += v
    for fn, n in unres.items():
        g.setdefault(fn, {"fn": fn, "resolved": 0, "unresolved": 0})["unresolved"] = n
    rows = sorted(g.values(), key=lambda r: -r["resolved"])
    worst = rows[0] if rows else None
    return {
        "functions": rows,
        "worst_fn": worst["fn"] if worst else None,
        "worst_bytes": worst["resolved"] if worst else 0,
        "worst_over_cap": bool(worst and worst["resolved"] > cap - safe_margin),
        "note": ("★判定看**单个函数的最大值**, 不是全文件求和 —— 每个 kernel 函数各有独立 "
                 "TPipe, 互不叠加。全加会把「各自都没超」误报成超, 也会让某个函数的真实"
                 "超限被别的函数的 UNRESOLVED 项淹掉 (cummin 2026-09-20 实撞, 烧掉八轮上板)。"),
    }


#  ---- U9: cap 是物理容量, 不是**可用**容量 -----------------------------------
#  2026-09-20 cummin 夹出两个实测点 (Ascend950PR, AIV_ONLY, 单函数单 TPipe):
#      218752 B  能跑
#      227968 B  **立即 507035** (aivec exception)
#  而 cap = 253952 (ini 的 ub_size), safe_margin 只留 1024 —— 两次都判「没超」。
#  真实可用上限落在 (218752, 227968), 即 cap 的 **86~90%**。差值去了 TPipe 自身的
#  结构、每个 TBuf/TQue 的对齐填充等工具看不见的地方。
#  ★后果是双向的, 两次都实撞过:
#    - 漏判: 报「88% OK」而真机崩, 排查会被引到别处 (那次误判成同步问题, 烧了多轮);
#    - 一旦按 cap 去卡, 还会让人以为还有 30 KB 余量可用。
#  ★用法: 占用率超过 cap 的 85% 就该当**可疑**处理, 而不是等它越过 safe_margin。
#  这两个点是单一芯片单一 kernel 形态下夹出来的, 换核类型/换 arch 要重标。
UB_PRACTICAL_RATIO = 0.85


def practical_note(total: int, cap: int) -> str | None:
    """占用率进入实测可疑区时给一句提示。判据本身不改判定, 只提示。"""
    if cap <= 0 or total <= 0:
        return None
    r = total / cap
    if r < UB_PRACTICAL_RATIO:
        return None
    return (f"★占用 {r:.0%} 已进入**实测可疑区** (cummin 2026-09-20 夹出的可用上限约 "
            f"cap 的 86~90%: 218752 B 能跑, 227968 B 立即 507035, 而 cap={cap} 判定两者"
            f"都没超)。cap 是 ini 的物理容量, 不是可用容量 —— 差值在 TPipe 结构与对齐"
            f"填充里。这一档请当可疑处理, 先砍缓冲再上板, 别把上板失败往别处查。")


def _ub_note(unresolved: bool, over: bool, safe_margin: int) -> str:
    """把"有未解析缓冲"这件事的**方向**说清楚。

    ★原措辞是"有 UNRESOLVED 缓冲, total 是下界, 不可据此判过" —— 方向反了, 读的人会把
      一条**成立**的判定当成不作数。2026-09-16 用 60 个算子普查时撞到: ops-nn 的 common
      目录 16 个未解析、已解析部分就已 393472B (cap 的两倍), 下界都超了, 判定当然成立。
      下界是单向的: 下界**超了**就是真超 (未解析的只会让它更多); 下界**没超**才什么都证明不了。
    """
    if not unresolved:
        return f"全解析 (同名互斥分支取大者); 判定线 = cap - {safe_margin}B 经验余量"
    if over:
        return ("有 UNRESOLVED 缓冲, total 是**下界** —— 但下界已经越线, 判定成立 "
                "(未解析的那些只会让它更多)。")
    return ("有 UNRESOLVED 缓冲, total 是**下界** —— 现在没越线**什么都不能证明**, "
            "补 --values 或修表达式求值再看。")



#  for 头: `for (uint32_t i = 0; i < N; ...)`。只取 "i = start; i < bound" 这一种形态 ——
#  AscendC 内核里的缓冲数组初始化循环都是这个写法, 放宽到 <= / 递减形态只会引入误判。
_FOR_HEAD = re.compile(
    r"for\s*\(\s*(?:[A-Za-z_][\w:<>\s\*&]*?\s+)?([A-Za-z_]\w*)\s*=\s*([^;]+?)\s*;"
    r"\s*\1\s*<\s*([^;]+?)\s*;")


def _loop_repeat(text: str, pos: int, target: str, vals: dict):
    """返回 (重复次数, 说明). 次数为 None 表示识别到数组下标却解不出循环次数 —— 必须记成未解析,
    不能默认按 1 计, 否则就是把"少算了 N-1 块"藏起来。"""
    mm = re.match(r"^([A-Za-z_]\w*)\s*\[\s*(.+?)\s*\]$", target)
    if not mm:
        return 1, ""
    idx = mm.group(2)
    if re.fullmatch(r"\d+", idx):
        return 1, ""              # 字面下标: 每个元素各写一行, 按字面记账本来就对
    names = set(re.findall(r"[A-Za-z_]\w*", idx))
    best = None
    for fm in _FOR_HEAD.finditer(text, 0, pos):
        if fm.group(1) in names:
            best = fm
    if best is None:
        return None, f"缓冲数组 {target} 的下标 {idx} 找不到对应的 for 头, 无法定量重复次数"
    lo, _ = _resolve(best.group(2), vals)
    hi, _ = _resolve(best.group(3), vals)
    if lo is None or hi is None or hi <= lo:
        return None, f"缓冲数组 {target} 的循环次数解不出: [{best.group(2)}, {best.group(3)})"
    return int(hi - lo), f"for {best.group(1)}<{best.group(3)}"


def _split_args(s: str):
    # ★先把 '->' 归一化成 '.': 下面按 "([<" / ")]>" 记括号深度, 而 C++ 的成员箭头 '->' 里的 '>' 会被
    #   当成右括号使深度错位, 于是嵌套调用 (如 Ceil(expr, 32)) 内部的逗号被误判为顶层参数分隔 ——
    #   后果是该 InitBuffer 被解析成 4 个参数, 既不匹配 TQue(3参) 也不匹配 TBuf(2参), 从而被静默丢弃,
    #   UB 总量算少而报"通过". (gmfr perf2 真机 UB 越界即由此漏报: 漏了 perTokenScaleInQueue 2048B.)
    s = s.replace("->", ".")
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "([<": depth += 1
        elif c in ")]>": depth -= 1
        if c == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return out


#  ★判据的负例自检。ub_budget 吃的是源文件, 故合成临时片段。
#    两条最要命的判据各一个负例:
#      over_cap  —— 超 UB 必须报, 不能放过 (放过 = 白等 5 分钟编译再真机崩)
#      unresolved —— 表达式解不出必须**显式报并拒绝判过**, 不能当 0
#        (当 0 会给出偏小的 total 并"通过", 那是工具主动制造假安全)

def selftest_u5d() -> int:
    """U5D 负例自检: 改动让偏移变大必须抓住; 没变/变小不报; 算不出要明说。"""
    BASE = """__aicore__ inline void VecOp::InitUB(TPipe *pipe) {
    uint32_t ubOffset = 0;
    pipe->InitBuffer(vecQue, 1024);
    if (!enableOptimizedScatter) {
        a = vecQue.GetWithOffset<float>(8, ubOffset);
        ubOffset += 100;
        return;
    }
    b = vecQue.GetWithOffset<float>(8, ubOffset);
    ubOffset += 200;
}
"""
    GREW = BASE.replace("ubOffset += 200;", "ubOffset += 260;")
    SHRANK = BASE.replace("ubOffset += 200;", "ubOffset += 150;")
    #  ★互斥分支必须取大者而不是相加: 两支分别 100 / 200, 峰值是 200 不是 300。
    #    第一版把整个文件的累加全加起来, 对能跑的代码报出超 10 倍的假警。
    UNRES = BASE.replace("ubOffset += 200;", "ubOffset += unknownSym;")
    bad = 0
    cases = [
        ("U5D 负例: 优化支多占 60 B 必须抓住", BASE, GREW, "U5D_grew", 60),
        ("U5D 正例: 两版一样, 不得报", BASE, BASE, "U5D_ok", 0),
        ("U5D 正例: 改小了, 不得报 grew", BASE, SHRANK, "U5D_ok", -50),
        ("U5D 负例: 有量解不出必须明说 (不能静默当没问题)", BASE, UNRES, "U5D_unresolved", None),
    ]
    print("=== ub_budget U5D 自检 (4 条) ===")
    for name, a, b, want_check, want_delta in cases:
        got = [x for x in pool_offset_delta(a, b, {}) if x["offvar"] == "ubOffset"]
        ck = got[0]["check"] if got else "(无输出)"
        dl = got[0].get("delta") if got else None
        ok = (ck == want_check) and (want_delta is None or dl == want_delta)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {name}   (期望 {want_check}/delta={want_delta}, "
              f"实际 {ck}/delta={dl})")
    #  ★互斥取大者的直接断言: BASE 的峰值必须是 200 (两支的大者), 不是 300 (相加)
    pk, _why = _peak_offset(BASE, "ubOffset", {})
    ok = (pk == 200)
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} U5D 覆盖守卫: early-return 两支取大者 "
          f"(期望 peak=200, 实际 {pk})")
    print(f"ub_budget U5D 自检: {len(cases) + 1} 条, {bad} 条未过")
    return 1 if bad else 0



# ---------------------------------------------------------------- U6
_U6_ALLOC = re.compile(r"\b(\w+)\s*=\s*\w+\.GetWithOffset<[^>]*>\s*\(([^,]+),")
_U6_IDENT = re.compile(r"[A-Za-z_]\w*")
_U6_SKIP = {"sizeof", "static_cast", "int", "uint32_t", "int64_t", "uint64_t",
            "float", "half", "T1", "T2", "Max", "Min", "AlignUp", "CeilDiv"}


# ---------------------------------------------------------------- 逐行守卫 (U6 用)
_G_IF = re.compile(r"\bif\s*(?:constexpr\s*)?\(([^)]{1,160})\)")


# ---- U7: **一个常量能不能调大** —— 只证伪, 不证安全。
#      场景: 循环粒度/行数之类的常量 (NON_OPT_UB_ROW_SIZE / UB_ROW_SIZE / MAX_GATHER_SIZE)
#      直接决定几个缓冲的大小, 想调大之前得知道装不装得下。此前只能手算, 而手算正是
#      U5/U6 那两个 507015 的来路。
#      ★为什么不用 U5 的绝对峰值: 那条判据 2026-09-16 在**已知良品** v9a 上就报越界
#        (见 pool_offset_report 的 docstring), 对良品报警的判据不能拿来下结论。
#      ★U7 绕开它的办法: 不去推"整条路径的峰值", 只把**调用者点名的那几个缓冲**加起来。
#        这些缓冲由调用者断言是同时存活的, 所以这个和是真实峰值的一个**下界** ——
#        下界超过池容量就一定超, 这个方向只会出真阳性。
#      ★反过来不成立, 所以 sum <= cap 时**必须拒绝给结论**: 没加进来的缓冲还在那儿。
#        判据只回答"这个方案已经死了吗", 不回答"这个方案能过吗"。
def _literal_locals(text: str) -> dict:
    """普通局部标量里的**纯字面量**定义 —— `uint32_t totalUbSpace = 191 * 1024;` 这种。

    只认右侧完全由数字/运算符/括号组成的, 因为一旦掺进标识符就可能是运行期量,
    把运行期量当常量正是本模块最危险的那类假阴性 (会把越界报成 headroom 充足)。
    只给 U7 用, 不动 source_constants —— 修假阴性不能顺手换来假阳性。
    """
    d = {}
    pat = re.compile(r"\b(?:u?int(?:8|16|32|64)_t|size_t|int|unsigned)\s+([A-Za-z_]\w*)"
                     r"\s*=\s*([0-9\s*+\-/()]+?)\s*;")
    for m in pat.finditer(text):
        try:
            d.setdefault(m.group(1), int(eval(m.group(2), {"__builtins__": {}}, {})))
        except Exception:
            pass
    return d


def _split_top(s, sep=","):
    """按顶层分隔符切, 尊重 () 与**模板**尖括号的嵌套。

    ★`<` 只有在不是比较/移位运算符时才算嵌套开始。首版把 `<=` 里的 `<` 当成开括号,
      于是 `A <= B, "msg"` 整条切不开, 条件里混进了断言消息 —— 判据当场解不出任何一条。
    """
    out, d, cur = [], 0, []
    n = len(s)
    for i, ch in enumerate(s):
        nxt = s[i + 1] if i + 1 < n else ""
        prv = s[i - 1] if i else ""
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        elif ch == "<" and nxt not in "=<" and prv != "<":
            d += 1
        elif ch == ">" and nxt not in "=>" and prv not in ">-":
            d -= 1
        if ch == sep and d == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def source_asserts(text: str) -> list:
    """源码自带的 `static_assert(<LHS> <= <RHS>, "...")` 容量约束。

    ★为什么值得单列一条判据 (2026-09-30 由 CAKE-PI 裸基线 T2 沉淀): 那次要判
      「把 GATHER_UB_ROWS 从 8 调到 16, UB 放不放得下」, U7_afford 报 U7_no_pool ——
      因为它只认两参的 `InitBuffer(buf, cap)` (TBuf 池), 而这份源码用的是三参的
      `InitBuffer(que, 1, bytes)` (TQue 队列), 且那两个常量**不产生自己的 InitBuffer**,
      它们是在已经分配好的队列里切槽位。于是 agent 只能手算出上限 (14 / 已到顶 4)。
      而约束**本来就写在源码里**: block_vec.h:128 那三条 static_assert 就是它。
      判据要做的不是重新推导, 是把已有的约束解出来。
    """
    out = []
    for m in re.finditer(r"\bstatic_assert\s*\(", text):
        i = m.end() - 1
        d, j = 0, i
        while j < len(text):
            if text[j] == "(": d += 1
            elif text[j] == ")":
                d -= 1
                if d == 0: break
            j += 1
        if j >= len(text):
            continue
        body = text[i + 1:j]
        cond = _split_top(body)[0].strip()
        mm = re.match(r"^(.*?)(<=|<|>=|>)(.*)$", cond, re.S)
        if not mm:
            continue
        ln = text.count("\n", 0, m.start()) + 1
        out.append({"line": ln, "lhs": mm.group(1).strip(),
                    "op": mm.group(2), "rhs": mm.group(3).strip(), "cond": cond})
    return out


def _const_fixpoint(base: dict, exprs: dict, depth: int = 8) -> dict:
    """把源码常量表达式反复代入求值, 直到不再增长。

    ★source_constants 是**单遍** setdefault, 于是 `VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO`
      这类链式常量在依赖还没进表时被跳过。constant_exprs 返回未求值的表达式正是为此准备。
      base 里已有的值**压过**源码 (--values 优先), 且 base 里的量不参与重解 —— 这样
      "把某个常量钉成 N 再看派生量" 才成立。
    """
    vals = dict(base)
    for _ in range(depth):
        grew = False
        for nm, ex in exprs.items():
            if nm in vals:
                continue
            v, _e = _resolve(ex, vals)
            if v is not None:
                vals[nm] = int(v)
                grew = True
        if not grew:
            break
    return vals


def _max_const(const: str, rows: list, base: dict, exprs: dict, hi: int = 4096):
    """点名常量在给定约束下最大能取到多少。指数上探 + 二分, 并**显式验边界**。

    单调性 (值越大越占地方) 是容量约束的常态但不是定理, 所以返回前核一次 best 成立、
    best+1 不成立; 核不过就退回线性扫描, 不硬套二分。
    """
    def holds(v):
        t = _const_fixpoint({**base, const: int(v)}, exprs)
        for r in rows:
            lv, _ = _resolve(r["lhs"], t)
            rv, _ = _resolve(r["rhs"], t)
            if lv is None or rv is None:
                return False
            if not ((lv <= rv) if r["op"] == "<=" else (lv < rv) if r["op"] == "<"
                    else (lv >= rv) if r["op"] == ">=" else (lv > rv)):
                return False
        return True

    if not holds(1):
        return None
    lo = 1
    step = 1
    while step <= hi and holds(min(step * 2, hi)):
        step = min(step * 2, hi)
        if step >= hi:
            break
    lo, hi2 = step, min(step * 2, hi)
    while lo < hi2:
        mid = (lo + hi2 + 1) // 2
        if holds(mid):
            lo = mid
        else:
            hi2 = mid - 1
    if holds(lo) and (lo >= hi or not holds(lo + 1)):
        return lo
    #  非单调: 二分不作数, 线性扫一遍取最大成立值
    best = None
    for v in range(1, hi + 1):
        if holds(v):
            best = v
    return best


def assert_margin(text: str, vals: dict, const: str | None = None,
                  src_path: str | None = None) -> list:
    """U9: 解源码自带的容量约束, 报余量; 给了 const 就再答「它最大能调到多少」。

    只解**求得出两边值**的那些; 解不出来显式列出, 不当成通过 —— 与 U7 同纪律。
    """
    vals = dict(_literal_locals(text), **vals)
    #  ★**不动点求解**, 不是单遍。source_constants 按文件顺序单遍 setdefault, 于是
    #    `VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO` 这种链式常量在依赖还没进表时就被跳过,
    #    整条 static_assert 随之解不出来 (2026-09-30 实撞: 三条全 UNRESOLVED)。
    #    constant_exprs 返回的是**未求值**的 name -> 表达式, 正是为这件事准备的。
    exprs = constant_exprs(src_path) if src_path else {}
    base = dict(vals)
    vals = _const_fixpoint(base, exprs)
    rows, unres = [], []
    for a in source_asserts(text):
        lv, le = _resolve(a["lhs"], vals)
        rv, re_ = _resolve(a["rhs"], vals)
        if lv is None or rv is None:
            unres.append({**a, "err": le or re_})
            continue
        ok = (lv <= rv) if a["op"] in ("<=", "<") else (lv >= rv)
        if a["op"] == "<": ok = lv < rv
        if a["op"] == ">": ok = lv > rv
        rows.append({**a, "lval": lv, "rval": rv, "ok": ok,
                     "margin": (rv - lv) if a["op"] in ("<=", "<") else (lv - rv)})
    out = []
    if not rows and not unres:
        out.append({"check": "U9_no_assert", "err":
                    "源码里没有 static_assert 形式的容量约束 —— 本条不适用 (不是通过)。"})
        return out
    for r in rows:
        lvl = "严重" if not r["ok"] else "提示"
        out.append({"check": "U9_assert_margin", "level": lvl, "line": r["line"],
                    "msg": (f"{r['cond'][:96]}  =>  {r['lval']} {r['op']} {r['rval']}"
                            f"  余量 {r['margin']} B" + ("" if r["ok"] else "  **已越界**"))})
    for u in unres:
        out.append({"check": "U9_unresolved", "level": "提示", "line": u["line"],
                    "msg": f"{u['cond'][:96]} —— 解不出来 ({u['err']}); "
                           f"补 --values 后重跑。**没解出来不等于没问题**。"})
    if const:
        #  ★"相关"不是"字面提到"。约束常常写的是**派生常量** (SCATTER_DK_BYTES <= DS_QUE_BYTES),
        #    而要调的是它的源头 (SCATTER_UB_ROWS)。判法改成: 把该常量 +1 重解一遍不动点,
        #    这条约束两边的值有没有变 —— 变了就是相关。2026-09-30 实撞: 按字面匹配时
        #    SCATTER_UB_ROWS 被报成"不受这些约束管", 而它正是那两条 256 B 余量的源头。
        probe = _const_fixpoint({**base, const: int(base.get(const, vals.get(const, 1)) or 1) + 1}, exprs)
        hit = []
        for r in rows:
            lv2, _ = _resolve(r["lhs"], probe); rv2, _ = _resolve(r["rhs"], probe)
            if (lv2, rv2) != (r["lval"], r["rval"]) or re.search(
                    r"\b" + re.escape(const) + r"\b", r["cond"]):
                hit.append(r)
        if not hit:
            out.append({"check": "U9_const_absent", "level": "提示",
                        "msg": f"没有一条解得出的 static_assert 提到 {const} —— "
                               f"它不受这些约束管 (或该约束解不出来, 见上)。"})
        else:
            best = _max_const(const, hit, base, exprs)
            v = int(vals.get(const, 0) or 0)
            if best is None:
                out.append({"check": "U9_afford_const", "level": "提示",
                            "msg": f"{const} 当前 {v}: 没有任何取值能让全部约束成立 —— 先查约束本身"})
            else:
                room = "**已到顶**" if best <= v else f"**最大可调到 {best}** (还能涨 {best - v})"
                out.append({"check": "U9_afford_const", "level": "提示",
                            "msg": f"{const} 当前 {v}, 受 {len(hit)} 条 static_assert 约束, {room}"})
    return out


def afford(text: str, names: list, vals: dict) -> list:
    """把 `names` 点名的那些 GetWithOffset 子分配加起来, 与所属池的 InitBuffer 容量比。"""
    out = []
    vals = dict(_literal_locals(text), **vals)   # --values 仍然压过源码
    pools = {}
    for m in re.finditer(r"InitBuffer\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^;()]+?)\s*\)\s*;", text):
        cap, _e = _resolve(m.group(2), vals)
        if cap:
            pools[m.group(1)] = int(cap)
    if not pools:
        out.append({"check": "U7_no_pool", "err":
                    "源码里没有可求值的 `InitBuffer(que, cap)` —— 没有池容量就无从比较。"})
        return out
    #  子分配: `<lhs> = <que>.GetWithOffset<T>(<count>, <off>);`
    sub = {}
    pat = re.compile(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\.\s*GetWithOffset\s*<\s*([^>]+?)\s*>"
                     r"\s*\(\s*(.+?)\s*,\s*([A-Za-z_]\w*)\s*\)\s*;", re.S)
    for m in pat.finditer(text):
        lhs, que, ty, cnt, _off = m.groups()
        if que not in pools:
            continue
        n, err = _resolve(cnt, vals)
        key = f"sizeof_{ty.strip()}"
        esz = vals.get(key, _SIZEOF_CONST.get(key))
        rec = {"buf": lhs, "que": que, "type": ty.strip(), "count_expr": cnt.strip(),
               "bytes": (n * esz) if (n is not None and esz is not None) else None,
               "err": err or (None if esz is not None else f"元素宽度未知: 给 --values 补 {key}")}
        prev = sub.get(lhs)
        #  同名多支 = 互斥分支, 取大者 (与 analyze() 对同名 InitBuffer 的处理一致)
        if prev is None or (rec["bytes"] or 0) > (prev["bytes"] or 0):
            sub[lhs] = rec
    missing = [n for n in names if n not in sub]
    if missing:
        out.append({"check": "U7_unknown_buffer", "err":
                    f"源码里找不到这些缓冲的 GetWithOffset 分配: {', '.join(missing)}。"
                    f"可用的有: {', '.join(sorted(sub)) or '(无)'}"})
        return out
    picked = [sub[n] for n in names]
    unres = [p for p in picked if p["bytes"] is None]
    if unres:
        out.append({"check": "U7_unresolved", "err":
                    "这几个缓冲**没算出字节数**, 本次检查没做成 (不是做了没问题): "
                    + "; ".join(f"{p['buf']}: {p['err']}" for p in unres)})
        return out
    que = picked[0]["que"]
    cap = pools[que]
    tot = sum(p["bytes"] for p in picked)
    detail = " + ".join(f"{p['buf']}={p['bytes']}B" for p in picked)
    if tot > cap:
        out.append({"check": "U7_over", "total": tot, "cap": cap, "que": que,
                    "err": (f"**装不下**: 点名的 {len(picked)} 个缓冲合计 {tot} B, 而池 `{que}` "
                            f"只有 {cap} B, 超出 {tot - cap} B。\n  {detail}\n"
                            f"  这几个是调用者断言**同时存活**的, 所以这个和是真实峰值的下界 —— "
                            f"下界已经超了, 方案不用建也知道不成立。")})
    else:
        out.append({"check": "U7_not_refuted", "total": tot, "cap": cap, "que": que,
                    "err": (f"合计 {tot} B <= 池 `{que}` 的 {cap} B, **本判据不给'能过'的结论**: "
                            f"没被点名的缓冲还占着地方, 真实峰值只会更高。\n  {detail}\n"
                            f"  U7 只回答'这个方案已经死了吗', 不回答'这个方案能过吗'。")})
    return out


def guard_map(src):
    """行号(1-based) -> 该行所处的 if 条件链 (原文, else 记成 !(...))。

    ★为什么逐字符扫括号: `} else {` 一行里先闭后开, 按 count("{")/count("}") 拿不到
      被闭合的那一组条件, else 分支就会被记成"无守卫", 于是"if/else 两侧都分配"
      (等价于无条件) 看起来像"只在 if 侧分配"。sfag 的 maxTmp/sumTmp 就是这么误报的。
    """
    stack, depth, pend, closed = [], 0, None, []
    gm = {}
    for ln, raw in enumerate(src.split("\n"), 1):
        line = re.sub(r"//.*", "", raw)
        m = _G_IF.search(line)
        if m:
            pend = [m.group(1).strip()]
        gm[ln] = [c for _d, cs in stack for c in cs]
        epos = line.find("else")
        for pos, ch in enumerate(line):
            if ch == "{":
                c = list(pend or [])
                if 0 <= epos < pos:
                    c += ["!(" + x + ")" for x in closed]
                depth += 1
                stack.append((depth, c)); pend = None
            elif ch == "}":
                if stack and stack[-1][0] == depth:
                    closed = stack.pop()[1]
                depth -= 1
    return gm


def check_dual_capacity(paths):
    """U6: 同一个缓冲在互斥分支里用**不同的容量常量**分配 —— 读写两端可能各选一支。

    ★2026-09-16 sfag 实撞, 一个 507015 排了五轮才定位:
      gatherTensorPing 在非优化分支按 NON_OPT_MAX_GATHER_SIZE(64) 行分配, 在优化分支按
      MAX_GATHER_SIZE(32) 行分配 —— 两支都分配了, 所以"有没有赋值"这类检查一律看不见。
      而 GatherKV 的分派按 IS_DETERMINISTIC, UB 分配的分支按 enableOptimizedScatter:
      det=1 且门开时两个条件分叉, 缓冲按 32 行给、gather 按 64 行写, 一半落在界外。
      门关着时那两个条件恒等价, 所以错配藏了整整四轮, 期间四次改动全在改无关的代码。

    判据只报"这个缓冲有两种口径", 不判对错 —— 哪一支该配哪一支是**语义**, 静态判不了。
    它要替人省掉的是"根本没想到这里有两种口径"。与 U5/U5D 的分工: 那两条算**总量**,
    这条看同一个缓冲的**口径分叉**, 总量正常时它照样成立 (本例 UB 用量 99% 一直判过)。
    """
    out = []
    for path in paths:
        try:
            src = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        gm = guard_map(src)
        rows = {}
        for ln, raw in enumerate(src.split("\n"), 1):
            line = re.sub(r"//.*", "", raw)
            m = _U6_ALLOC.search(line)
            if not m:
                continue
            name, expr = m.group(1), m.group(2)
            ids = tuple(sorted({i for i in _U6_IDENT.findall(expr) if i not in _U6_SKIP}))
            rows.setdefault(name, []).append((ln, ids, expr.strip()))
        for name, rs in rows.items():
            if len({r[1] for r in rs}) < 2:
                continue
            #  该缓冲的**访问点**各自引用了哪些容量常量 —— 两边都分叉才是高风险
            cap_ids = set()
            for _, ids, _ in rs:
                cap_ids |= set(ids)
            used, use_guards = set(), []
            for ln, raw in enumerate(src.split("\n"), 1):
                if name in raw and not _U6_ALLOC.search(re.sub(r"//.*", "", raw)):
                    hit = {i for i in _U6_IDENT.findall(raw) if i in cap_ids}
                    used |= hit
                    if hit:
                        for g in gm.get(ln, []):
                            if g not in use_guards:
                                use_guards.append(g)
            out.append({"file": path, "buffer": name,
                        "allocs": [{"line": ln, "expr": e, "guard": gm.get(ln, [])}
                                   for ln, _, e in rs],
                        "cap_symbols_at_use": sorted(used),
                        "guards_at_use": use_guards})
    return out




def selftest_u6() -> int:
    import tempfile, os
    bad = 0
    print("=== U6 自检 (缓冲的两种容量口径) ===")
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "a.h")
        #  负例: 本次 sfag 的最小形态 —— 两支都分配, 但容量常量不同
        open(f, "w").write(
            "if (!opt) {\n  g = q.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimDqk, o);\n"
            "} else {\n  g = q.GetWithOffset<T1>(MAX_GATHER_SIZE * dimDqk, o);\n}\n"
            "DataCopy(g[i], src, NON_OPT_MAX_GATHER_SIZE);\n")
        r = check_dual_capacity([f])
        ok = len(r) == 1 and r[0]["buffer"] == "g" and len(r[0]["allocs"]) == 2
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 同一缓冲两支容量常量不同, 必须命中 (实得 {len(r)} 条)")
        ok = "NON_OPT_MAX_GATHER_SIZE" in (r[0]["cap_symbols_at_use"] if r else [])
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 要指出访问点按哪个常量算 (实得 {r[0]['cap_symbols_at_use'] if r else []})")
        #  正例: 两支容量相同 -> 不报 (否则判据过宽, 真仓会被淹掉)
        open(f, "w").write(
            "if (!opt) {\n  g = q.GetWithOffset<T1>(SZ * dimDqk, o);\n"
            "} else {\n  g = q.GetWithOffset<T1>(SZ * dimDqk, o);\n}\n")
        r = check_dual_capacity([f])
        bad += 0 if not r else 1
        print(f"  {'过' if not r else '**未过**'} 正例: 两支容量相同不报 (实得 {len(r)} 条)")
        #  正例: 只有一支分配 -> 不归 U6 管 (那是"分配缺失", 另一类)
        open(f, "w").write("g = q.GetWithOffset<T1>(SZ * dimDqk, o);\n")
        r = check_dual_capacity([f])
        bad += 0 if not r else 1
        print(f"  {'过' if not r else '**未过**'} 正例: 单处分配不报 (实得 {len(r)} 条)")
        #  负例: 守卫必须被**如实**报出来 —— 一支在 if 里、另一支是兜底。
        #  sfag 507015 的全部信息量就在这一行: 门开时生效的是兜底那支, 不是门内那支。
        open(f, "w").write(
            "void Init()\n{\n"
            "    if (!enableOpt) {\n        g = q.GetWithOffset<T1>(BIG * dimDqk, o);\n    }\n"
            "    g = q.GetWithOffset<T1>(SMALL * dimDqk, o);\n}\n")
        r = check_dual_capacity([f])
        gs = [a["guard"] for a in r[0]["allocs"]] if r else []
        ok = gs == [["!enableOpt"], []]
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 两支各自的守卫要如实报, 兜底那支报成空 (实得 {gs})")
        #  正例: if/else 两侧都分配时, else 那侧必须带**取反**的条件, 不能记成兜底 ——
        #  记成兜底就会把"实际无条件"说成"门内独有", 方向正好反了。
        open(f, "w").write(
            "void Init()\n{\n"
            "    if (opt) {\n        g = q.GetWithOffset<T1>(BIG * dimDqk, o);\n"
            "    } else {\n        g = q.GetWithOffset<T1>(SMALL * dimDqk, o);\n    }\n}\n")
        r = check_dual_capacity([f])
        gs = [a["guard"] for a in r[0]["allocs"]] if r else []
        ok = gs == [["opt"], ["!(opt)"]]
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: else 那侧带取反条件, 不记成兜底 (实得 {gs})")
    print(f"U6 自检: 6 条, {bad} 条未过")
    return bad


def selftest_u7() -> int:
    """U7 正例 + 负例。负例是**已知良品**: 判据对能跑的代码报警就不能用 (U5 栽的正是这一条)。"""
    bad = 0
    src = """
        uint32_t totalUbSpace = 100 * 1024;
        pipe->InitBuffer(vecQue, totalUbSpace);
        aTensor = vecQue.GetWithOffset<float>(ROWS * DIM, off);
        bTensor = vecQue.GetWithOffset<float>(ROWS * DIM, off);
        cTensor = vecQue.GetWithOffset<half>(8, off);
    """
    # 正例: 两个缓冲合计 2*32*512*4 = 131072 > 102400
    r = afford(src, ["aTensor", "bTensor"], {"ROWS": 32, "DIM": 512})
    if r[0]["check"] != "U7_over":
        print(f"U7 自检失败(正例): 期望 U7_over, 实得 {r[0]['check']}"); bad += 1
    # 负例: ROWS=8 -> 32768 <= 102400, 必须**不**报 over
    r = afford(src, ["aTensor", "bTensor"], {"ROWS": 8, "DIM": 512})
    if r[0]["check"] != "U7_not_refuted":
        print(f"U7 自检失败(负例): 期望 U7_not_refuted, 实得 {r[0]['check']}"); bad += 1
    # 解不出必须说出来, 不能静默当 0
    r = afford(src, ["aTensor"], {"DIM": 512})
    if r[0]["check"] != "U7_unresolved":
        print(f"U7 自检失败(解不出): 期望 U7_unresolved, 实得 {r[0]['check']}"); bad += 1
    # 元素宽度未知也算解不出 (不能默认按 4 字节)
    r = afford(src.replace("<float>", "<MY_T>"), ["aTensor"], {"ROWS": 8, "DIM": 512})
    if r[0]["check"] != "U7_unresolved":
        print(f"U7 自检失败(未知元素宽度): 期望 U7_unresolved, 实得 {r[0]['check']}"); bad += 1
    # 点名了不存在的缓冲要报出来, 不能当成 0 字节混过去
    r = afford(src, ["noSuchTensor"], {"ROWS": 8, "DIM": 512})
    if r[0]["check"] != "U7_unknown_buffer":
        print(f"U7 自检失败(名字打错): 期望 U7_unknown_buffer, 实得 {r[0]['check']}"); bad += 1
    if not bad:
        print("U7 自检: 5/5 通过 (正例/已知良品负例/解不出/未知宽度/错名字)")
    return bad


def selftest():
    #  ---- U9 负例/正例: 解源码自带的 static_assert 容量约束 ----
    #  2026-09-30 由 CAKE-PI 裸基线 T2 沉淀: 那次要判「把行数常量调大放不放得下」,
    #  U7 报 U7_no_pool (它只认 TBuf 池 `InitBuffer(buf, cap)`, 而那份源码是从**已分配好的
    #  队列**里切槽位), agent 只能手算出上限。而约束本来就写在 static_assert 里。
    import tempfile as _tf, os as _os, io as _io
    _u9_src = """
    constexpr uint32_t ROWS = 8;
    constexpr uint32_t WIDTH = 576;
    constexpr uint32_t CAP = 16640;
    constexpr uint32_t USED = ROWS * WIDTH * 2;
    static_assert(USED <= CAP, "slot exceeds queue");
    static_assert(ROWS * WIDTH * sizeof(half) <= CAP, "direct form");
    """
    _u9_cases = []
    with _tf.TemporaryDirectory() as _td:
        _f = _os.path.join(_td, "u9.h")
        _io.open(_f, "w", encoding="utf-8").write(_u9_src)
        _r = assert_margin(_u9_src, {"sizeof_half": 2}, None, src_path=_f)
        _m = [x for x in _r if x["check"] == "U9_assert_margin"]
        _u9_cases.append(("U9 两条约束都要解出来且余量正确",
                          len(_m) == 2 and all("余量 7424 B" in x["msg"] for x in _m)))
        #  ★关键: 约束写的是**派生常量** USED, 而要调的是它的源头 ROWS —— 按字面匹配会漏
        _r = assert_margin(_u9_src, {"sizeof_half": 2}, "ROWS", src_path=_f)
        _a = [x for x in _r if x["check"] == "U9_afford_const"]
        _u9_cases.append(("U9 派生常量的源头也算相关 (ROWS 经 USED 受约束)",
                          bool(_a) and "最大可调到 14" in _a[0]["msg"]))
        #  正例: 与任何约束无关的常量, 不许编一个上限出来
        _r = assert_margin(_u9_src, {"sizeof_half": 2}, "UNRELATED", src_path=_f)
        _u9_cases.append(("U9 无关常量必须报不适用, 不许编上限",
                          any(x["check"] == "U9_const_absent" for x in _r)))
        #  负例: 已经越界的约束必须报严重
        _r = assert_margin(_u9_src, {"sizeof_half": 2, "ROWS": 64}, None, src_path=_f)
        _u9_cases.append(("U9 已越界必须报严重",
                          any(x["check"] == "U9_assert_margin" and x["level"] == "严重" for x in _r)))
        #  正例: 没有 static_assert 的源码要报"不适用", 不许当成通过
        _r = assert_margin("int x = 1;", {}, None, src_path=None)
        _u9_cases.append(("U9 源码没有 static_assert 时报不适用",
                          any(x["check"] == "U9_no_assert" for x in _r)))
    _u9_bad = 0
    for _n, _ok in _u9_cases:
        print(f"  {'过' if _ok else '★未过'} {_n}")
        _u9_bad += 0 if _ok else 1

    #  ★别名 vs 游标 的正负例 (2026-09-20: 本检查第一版就把游标说成别名, 13 块全报成共用)
    _cursor = """
        a = q.GetWithOffset<float>(n1, ubOffset);
        ubOffset += n1 * sizeof(float);
        b = q.GetWithOffset<float>(n2, ubOffset);
    """
    _alias = """
        c = q.GetWithOffset<float>(n3, baseOff);
        d = q.GetWithOffset<half>(n4, baseOff);
    """
    _g1 = alias_groups(_cursor)
    _ok1 = all(len(v) == 1 for v in _g1.values()) and len(_g1) == 2
    _g2 = alias_groups(_alias)
    _ok2 = len(_g2) == 1 and len(next(iter(_g2.values()))) == 2
    print(f"  {'过' if _ok1 else '**未过**'} 负例: 游标 `off += ...` 推进后的两块是**顺序**不是别名 "
          f"(把顺序说成别名会把占用从求和变成取 max, 低估到灾难性)")
    print(f"  {'过' if _ok2 else '**未过**'} 正例: 中间没有推进游标的两块是**真别名** (组内取 max)")

    import tempfile
    cases = [
        ("over_cap: 两个 128KB 缓冲共 256KB > 192KB",
         "pipe->InitBuffer(a, 131072);\n    pipe->InitBuffer(b, 131072);\n",
         {}, lambda r: r.get("over_cap") is True),
        ("unresolved: 缓冲大小用了未知符号, 必须拒绝判过",
         "pipe->InitBuffer(a, someUnknownSymbol * 4);\n",
         {}, lambda r: r.get("unresolved_count", 0) > 0),
        ("正例: 一个 64KB 缓冲应判过且无 unresolved",
         "pipe->InitBuffer(a, 65536);\n",
         {}, lambda r: r.get("over_cap") is False and r.get("unresolved_count", 0) == 0),
        #  ★BufferManager<UB> 整池分配曾完全看不见 —— 静默假阴性 (FIA arch35 漏 160KB, 报 34% 而真值 98%)
        ("BufferManager<UB> 的整池 Init 必须计入 (漏了就是静默假阴性)",
         "BufferManager<BufferType::UB> ubMgr;\n    ubMgr.Init(pipe, 200000);\n"
         "    pipe->InitBuffer(a, 65536);\n",
         {}, lambda r: r.get("over_cap") is True),
        #  ★constexpr 链依赖模板参数时, 必须先并 --values 再做不动点, 否则整条链解不出而被当成"没有"
        ("constexpr 链依赖 --values 给的模板参数, 要能解出来",
         "constexpr uint32_t HALF = mBase / 2;\n    constexpr uint32_t SZ = HALF * 1024;\n"
         "    pipe->InitBuffer(a, SZ);\n",
         {"mBase": 64}, lambda r: r.get("unresolved_count", 0) == 0
                                  and r.get("total_bytes_resolved") == 32768),
        #  ★缓冲数组在 for 循环里逐个 InitBuffer: 字面只有一次调用, 实际开 N 块。
        #    按字面记账 = 把 N-1 块藏起来, 又一例静默假阴性 (FIAS 行分组的 stage2OutBuf[2] 实撞)。
        ("for 循环里初始化的缓冲数组必须按循环次数计, 不能只算一块",
         "for (uint32_t i = 0; i < 2; ++i) {\n      pipe->InitBuffer(a[i], 131072);\n    }\n",
         {}, lambda r: r.get("over_cap") is True and r.get("total_bytes_resolved") == 262144),
        ("循环次数是 --values 给的符号时也要能定量",
         "for (uint32_t i = 0; i < GROUP; ++i) {\n      pipe->InitBuffer(a[i], 1024);\n    }\n",
         {"GROUP": 3}, lambda r: r.get("unresolved_count", 0) == 0
                                 and r.get("total_bytes_resolved") == 3072),
        ("下标不是循环变量又找不到 for 头时, 必须记成未解析而不是默认按一块",
         "pipe->InitBuffer(a[k], 1024);\n",
         {}, lambda r: r.get("unresolved_count", 0) > 0),
        ("字面下标 (每个元素各写一行) 仍按字面记账, 不能误乘",
         "pipe->InitBuffer(a[0], 1024);\n    pipe->InitBuffer(a[1], 1024);\n",
         {}, lambda r: r.get("total_bytes_resolved") == 2048),
        #  ---- L1 侧 (mem="L1"): 容量读源码字面量, 份数由策略类名决定 ----
        #  ★策略类名就是份数: 3buff -> 2buff 是"省一整块"的唯一手段, 而源码上只改了一个类名,
        #    Init(mgr, size) 的字面完全一样。数不清这一层, 容量账整体错一个倍数。
        ("L1: 策略类的份数由类名定 (3buff = 3 份)",
         "BufferManager<BufferType::L1> l1Mgr;\n"
         "    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> kv;\n"
         "    l1Mgr.Init(pipe, 524288);\n    kv.Init(l1Mgr, 147456);\n",
         {}, lambda r: r.get("total_bytes_resolved") == 442368 and r.get("cap_bytes") == 524288,
         "L1"),
        ("L1: 管理器自己的 Init 是容量声明, 不是一次分配 (计入就把容量当占用又加一遍)",
         "BufferManager<BufferType::L1> l1Mgr;\n"
         "    BuffersPolicySingleBuffer<BufferType::L1> q;\n"
         "    l1Mgr.Init(pipe, 524288);\n    q.Init(l1Mgr, 73728);\n",
         {}, lambda r: r.get("total_bytes_resolved") == 73728 and r.get("over_cap") is False,
         "L1"),
        ("L1: 手工 AllocBuffer 的份数是循环上界, 解不出必须记未解析而不是按 1 算",
         "BufferManager<BufferType::L1> l1Mgr;\n    l1Mgr.Init(pipe, 524288);\n"
         "    for (uint32_t i = 0U; i < slotCnt; ++i) {\n"
         "      slot[i] = l1Mgr.template AllocBuffer<SyncType::CROSS_CORE_SYNC_FORWARD>(147456);\n    }\n",
         {}, lambda r: r.get("unresolved_count", 0) > 0, "L1"),
        ("L1: 循环上界由 --values 给出时要能定量 (3 x 144K)",
         "BufferManager<BufferType::L1> l1Mgr;\n    l1Mgr.Init(pipe, 524288);\n"
         "    for (uint32_t i = 0U; i < slotCnt; ++i) {\n"
         "      slot[i] = l1Mgr.template AllocBuffer<SyncType::CROSS_CORE_SYNC_FORWARD>(147456);\n    }\n",
         {"slotCnt": 3}, lambda r: r.get("total_bytes_resolved") == 442368
                                   and r.get("unresolved_count", 0) == 0, "L1"),
        #  ★`(uint32_t)x` 与 `0U` 这两种 C 写法一度让整条表达式失解 -> 该缓冲不计入总量
        ("C 风格转型与整数后缀不能让表达式失解 (否则缓冲被静默漏算)",
         "pipe->InitBuffer(a, (uint32_t)mBase * 576 * 2U);\n",
         {"mBase": 64}, lambda r: r.get("total_bytes_resolved") == 73728
                                  and r.get("unresolved_count", 0) == 0),
    ]
    bad = 0
    print(f"=== ub_budget 判据负例自检 ({len(cases)} 条) ===")
    for case in cases:
        name, body, vals, want = case[0], case[1], case[2], case[3]
        mem = case[4] if len(case) > 4 else "UB"
        #  ★用例必须放进**隔离的空目录**。constant_exprs() 会从源码所在目录树 os.walk 收集
        #  constexpr 与局部变量, 放在 /tmp 下就等于把整个 /tmp 拉进常量表 ——
        #  结果既慢又**取决于机器上 /tmp 里有什么**, 自检本身不可复现。
        #  2026-09-15 实撞: "constexpr 链依赖 --values" 那条长期未过, 正是被 /tmp 污染,
        #  表现为整行 InitBuffer 被丢弃 (total=0 且 unresolved=0, 既没计入也没记未解析)。
        tmpdir = tempfile.mkdtemp(prefix="ubselftest_")
        path = os.path.join(tmpdir, "case.h")
        with open(path, "w", encoding="utf-8") as f:
            f.write("class K {\n  void Init() {\n    " + body + "  }\n};\n")
        try:
            r = analyze(path, vals, None, mem) if "analyze" in globals() else None
            if r is None:
                print("  ★本模块没有可直接调用的 analyze(), 自检跳过 (需要先把主流程抽成函数)")
                return 0
            ok = want(r)
        except Exception as e:
            print(f"  ★挂了 {name}: {type(e).__name__}: {e}"); bad += 1; continue
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"  {'过' if ok else '★未命中'} {name}")
        if not ok:
            print(f"      实际 over_cap={r.get('over_cap')} unresolved={r.get('unresolved_count')} "
                  f"total={r.get('total_bytes_resolved')}")
            bad += 1
    #  ---- U5_control: 守住"U5 绝对值判据未接线"这个状态 ----
    #  它 2026-09-16 被样本外检验推翻 (对已知良品 v9a 同样报越界 25600 B)。在做对分段数据流
    #  并在真仓良品上跑出"不报警"之前, 不许挂进 analyze() —— 判据对良品报警就是狼来了。
    #  ★这条自检拦的是**未来的自己**: 半成品判据看起来"差不多能用", 最容易被顺手接上去。
    import inspect as _inspect
    _src = _inspect.getsource(analyze)
    ok = "pool_offset_report" not in _src
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} U5_control: 绝对值判据仍未挂进 analyze() "
          f"(它已被证伪; 接线前提是先在真仓已知良品上跑出不报警)")

    #  note 的**方向**: 下界超了判定成立, 下界没超才不作数 (2026-09-16 普查时发现措辞反了)
    n_over = _ub_note(True, True, 1024); n_under = _ub_note(True, False, 1024)
    ok = "判定成立" in n_over and "什么都不能证明" in n_under
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} note 方向: 有未解析且**下界已越线** -> 说判定成立; "
          f"下界没越线 -> 说什么都不能证明")
    #  ---- U10/U11: 存储归属 + 打印与判定同源 --------------------------------
    _u10_src = """
    TSCM<TPosition::GM, 2> vScm;
    TBuf<TPosition::VECCALC> rawBuf, scoreBuf, pBuf;
    TQueBind<TPosition::GM, TPosition::A1, 2> aQue;
    TQue<TPosition::VECIN, 2> inQue;
    TBuf<TPosition::CO1> accBuf;
    """
    _u10_cases = [
        ("TSCM<> 绑 L1, 不计进 UB", "vScm", "L1"),
        ("TBuf<VECCALC> 是真 UB, 必须仍然计入 (负例: 判据不得把真 UB 排除)", "rawBuf", "UB"),
        ("同一条声明里的第二个 declarator 也要认出来", "scoreBuf", "UB"),
        ("同一条声明里的第三个也要", "pBuf", "UB"),
        ("TQueBind 到 A1 绑 L1", "aQue", "L1"),
        ("TQue<VECIN> 是 UB", "inQue", "UB"),
        ("TBuf<CO1> 是 L0C", "accBuf", "L0C"),
        ("找不到声明时**按 UB 计**并标记 (少算比多算危险)", "ghostBuf", "未知"),
    ]
    for _why, _nm, _want in _u10_cases:
        _got, _ = storage_class(_u10_src, _nm)
        _ok = _got == _want
        bad += 0 if _ok else 1
        print(f"  {'过' if _ok else '**未过**'} U10 {_why}: 得 {_got}, 期望 {_want}")

    #  U10 端到端: TSCM 的字节必须从 UB 总量里剔除, 真 UB 缓冲必须留下
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as _fh:
        _fh.write("""
    TSCM<TPosition::GM, 2> vScm;
    TBuf<TPosition::VECCALC> rawBuf;
    __aicore__ inline void RunCube() { pipe.InitBuffer(vScm, 2, 65536); }
    __aicore__ inline void RunVector() { pipe.InitBuffer(rawBuf, 16384); }
    """)
        _fp = _fh.name
    _r = analyze(_fp, {}, mem="UB")
    _ok = (_r["total_bytes_resolved"] == 16384
           and any(b["buf"] == "vScm" for b in _r.get("off_ub_buffers", [])))
    bad += 0 if _ok else 1
    print(f"  {'过' if _ok else '**未过**'} U10 端到端: TSCM 131072 剔除后 UB 只剩 16384 "
          f"(得 {_r['total_bytes_resolved']}); vScm 列进 off_ub_buffers")

    #  U11: 多函数时, 打印用的数必须等于 per_function 的 worst, 不是全文件和
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as _fh:
        _fh.write("""
    TBuf<TPosition::VECCALC> aBuf, bBuf;
    __aicore__ inline void FnA() { pipe.InitBuffer(aBuf, 100000); }
    __aicore__ inline void FnB() { pipe.InitBuffer(bBuf, 120000); }
    """)
        _fp2 = _fh.name
    _r2 = analyze(_fp2, {}, mem="UB")
    _pf = _r2.get("per_function") or {}
    _ok = (_r2["total_bytes_resolved"] == 220000 and _pf.get("worst_bytes") == 120000
           and _r2["over_cap"] is False)
    bad += 0 if _ok else 1
    print(f"  {'过' if _ok else '**未过**'} U11: 全加 220000 会误报越线, 单函数最大 120000 不越线 "
          f"(得 和 {_r2['total_bytes_resolved']} / 最大 {_pf.get('worst_bytes')} / over_cap {_r2['over_cap']})")
    #  ★判据本身: 打印分支必须读 per_function, 不能读 total_bytes_resolved
    _msrc = _inspect.getsource(main)
    _ok = "_pf[\"worst_bytes\"]" in _msrc or "_pf['worst_bytes']" in _msrc
    bad += 0 if _ok else 1
    print(f"  {'过' if _ok else '**未过**'} U11 接线: 打印分支确实读 per_function 的 worst "
          f"(2026-10-27 的缺陷正是判定已修而打印没修)")

    bad += _u9_bad
    print(f"ub_budget 自检: {len(cases) + len(_u9_cases) + len(_u10_cases) + 4} 条 "
          f"(含 U9 {len(_u9_cases)} 条, U10/U11 {len(_u10_cases) + 4} 条), "
          f"{bad} 条未过")
    bad += selftest_u5d()
    bad += selftest_u6()
    bad += selftest_u7()
    bad += selftest_u8()
    return 1 if bad else 0


#  负例取自 2026-09-22 真实打挂设备的那份 softmax (已提交包 Softmax-now1-86405f8a):
#  ROW 路径把**整行**装进 UB, 公开集最大 K=8193 从没暴露, 隐藏集 80 个 case 判失败 35 个。
#  正例是同一份文件里**已经分块**的 RowWide (缓冲恒定约 4096 lane, 与 K 无关)
#  与 host 侧限了 IT<=1024 / NK<=128 的 COL 路径 —— 判据必须把三者分开。
_U8_NEG = """
__aicore__ inline void Dispatch(GM_ADDR x, GM_ADDR y, int64_t n, int32_t K) {
    if (K <= 8) { Small(x, y, n, K); return; }
    Row<float>(x, y, n, K);
}
template <typename T>
__aicore__ inline void Row(GM_ADDR xg, GM_ADDR yg, int64_t nRows, int32_t K) {
    const int32_t kp = ((K + 64 - 1) / 64) * 64;
    AscendC::TPipe pipe;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(kp) * sizeof(T));
    pipe.InitBuffer(outQ, 2, static_cast<uint32_t>(kp) * sizeof(T));
    pipe.InitBuffer(wBuf, static_cast<uint32_t>(kp) * 2u * sizeof(float) + 4096u);
}
"""
_U8_POS = """
__aicore__ inline void Dispatch(GM_ADDR x, GM_ADDR y, int64_t n, int32_t K) {
    Wide<float>(x, y, n, K);
}
template <typename T>
__aicore__ inline void Wide(GM_ADDR xg, GM_ADDR yg, int64_t nRows, int32_t K) {
    const int32_t slot = ((K + 7) / 8) * 8;
    int32_t kRows = 4096 / slot;
    if (kRows > 255) { kRows = 255; }
    AscendC::TPipe pipe;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(kRows * slot) * sizeof(T));
    pipe.InitBuffer(outQ, 2, static_cast<uint32_t>(kRows * slot) * sizeof(T));
    pipe.InitBuffer(wBuf, static_cast<uint32_t>(kRows * slot * 4) * sizeof(float));
}
"""
_U8_HOSTGUARD = """
template <typename T>
__aicore__ inline void Col(GM_ADDR xg, GM_ADDR yg, int64_t outer, int32_t IT, int32_t NK) {
    const uint32_t tileElems = static_cast<uint32_t>(NK) * static_cast<uint32_t>(IT);
    AscendC::TPipe pipe;
    pipe.InitBuffer(inQ, 2, tileElems * sizeof(T));
    pipe.InitBuffer(wBuf, tileElems * 2u * sizeof(float)
                          + static_cast<uint32_t>(IT) * 2u * sizeof(float));
}
"""
_U8_HOST = """
static Plan plan(int64_t inner, int64_t K) {
    int64_t IT = ((inner + 63) / 64) * 64;
    if (IT > 1024) { IT = 1024; }
    int64_t nkMax = 200000 / (IT * 16);
    if (nkMax > 128) { nkMax = 128; }
    int64_t NK = 1;
    while (NK * 2 <= nkMax) { NK *= 2; }
    return Plan{IT, NK};
}
"""


def selftest_u8() -> int:
    """U8 的负例控制。★负例就是真机打挂设备的那份源码 —— 判据抓不住它就等于没有。"""
    import tempfile
    bad = 0

    def run(src, host=None):
        d = tempfile.mkdtemp()
        f = os.path.join(d, "k_kernel.h")
        open(f, "w", encoding="utf-8").write(src)
        hp = None
        if host:
            hp = os.path.join(d, "k_plugin.cpp")
            open(hp, "w", encoding="utf-8").write(host)
        try:
            return scaling_threshold(f, host_path=hp)
        except Exception as e:                       # 取不到容量: 自检不该静默跳过
            print(f"  **未过** U8 自检跑不起来: {e}")
            return None

    def ck(name, got, want):
        nonlocal bad
        ok = (got == want)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {name}: 得 {got!r}, 期望 {want!r}")

    rows = run(_U8_NEG) or []
    ung = [r for r in rows if r["verdict"] == "UNGUARDED"]
    ck("负例 (真实缺陷: 整行装 UB, 调用点只拦了 K<=8 那一支)", [r["param"] for r in ung], ["K"])
    ck("负例必须给出越线阈值, 不能只说'有风险'",
       bool(ung) and isinstance(ung[0].get("cross"), int) and ung[0]["cross"] > 8, True)

    rows = run(_U8_POS) or []
    ck("正例 (已分块: 缓冲恒定约 4096 lane)",
       [r["verdict"] for r in rows if r["param"] == "K"] or ["(没报这一维)"], ["bounded"])

    rows = run(_U8_HOSTGUARD) or []
    ck("host 守卫缺席时必须报 (只看内核看不出 IT 有没有界)",
       sorted({r["param"] for r in rows if r["verdict"] == "UNGUARDED"}), ["IT", "NK"])
    rows = run(_U8_HOSTGUARD, _U8_HOST) or []
    ck("给了 host 就不该再报 (IT<=1024 / NK<=nkMax<=128)",
       sorted({r["param"] for r in rows if r["verdict"] == "UNGUARDED"}), [])

    #  ★守卫的第三种形态: 函数**自己开头**判装不下就改走分块路径并 return。
    #    这是真实修法的写法 (softmax 2026-09-22 的修复即此形), 判据必须认, 否则会在
    #    缺陷修好之后继续报警。负例与正例只差这一个 if。
    _U8_FIXED = _U8_NEG.replace(
        "    const int32_t kp = ((K + 64 - 1) / 64) * 64;",
        "    const int32_t kp = ((K + 64 - 1) / 64) * 64;\n"
        "    if ((int64_t)kp * (4 * (int64_t)sizeof(T) + 8) + 4096 > 240000) { Big(x, y, n, K); return; }")
    rows = run(_U8_FIXED) or []
    ck("修好后必须不报 (函数内提前返回也是守卫)",
       [r["verdict"] for r in rows if r["param"] == "K"] or ["(没报这一维)"], ["bounded"])

    #  ★整数除法: C 是向零截断。算成浮点会让每个对齐取整都偏大 (d=256 时 AlignUp 出 319)
    ck("整数除法按 C 语义 (对齐取整不得偏大)", _resolve("((256 + 64 - 1) / 64) * 64", {})[0], 256)
    #  ★换行折平: 字节表达式常写成两行相加, 不折平则整条失解而**不计入总量**
    ck("跨行表达式必须解得出", _resolve("4 * 8\n  + 16", {})[0], 48)
    #  ★static_cast 剥离: InitBuffer 的字节表达式几乎全是这个写法
    ck("static_cast 转型不得导致失解", _resolve("static_cast<uint32_t>(3) * 4", {})[0], 12)
    print(f"U8 自检: 9 条, {bad} 条未过")
    return 1 if bad else 0


# ---- U8: **UB 用量随输入尺寸增长而 host 侧没有上限护栏** ----------------------------
#      U1-U7 都是「代入一组 tiling 值, 算这一组下的占用」。它们答不出的是:
#      这个占用随输入的哪一维增长? 增长到多大会越界? host 有没有拦?
#      ROIAlign 2026-09-20 实撞: 内核把整张 H*W 特征图装进 UB (InitBuffer(inQ_, 2, H*Wp*es)),
#      公开 20 个 case 的 H*W 最大 128*128 所以从没暴露; hidden 集里更大的图直接
#      507035 (device error type 3) 打挂设备, 并带崩后面 13 个本来能过的 case
#      (评测判「设备不可恢复, 用例跳过」)。判据只要回答一句话: **哪几个 InitBuffer 的
#      尺寸表达式里出现了核函数的运行期入参**。出现即为需要 host 侧护栏或分块的地方。

_U8_PARAMS = re.compile(
    r"(?:__global__|__aicore__)[^;{}]*?\b\w+\s*\(([^;{}]*?)\)\s*\{", re.S)


def _runtime_params(text: str) -> set:
    """核函数 / 设备端 Process 的入参名 —— 这些量由调用方在运行期给, 不是编译期常量。"""
    out = set()
    for m in _U8_PARAMS.finditer(text):
        for arg in m.group(1).split(","):
            arg = arg.strip()
            if not arg or arg == "void":
                continue
            name = re.sub(r"[\[\]0-9]+$", "", arg.split()[-1]).lstrip("*&")
            if re.fullmatch(r"[A-Za-z_]\w*", name or ""):
                out.add(name)
    # GM_ADDR 之类的指针不参与尺寸表达式, 留着也无害
    return out


def scaling_report(text: str) -> list:
    """返回 [(缓冲名, 尺寸表达式, 展开后用到的运行期入参)]。"""
    params = _runtime_params(text)
    # 成员量常是 `H_ = H;` 这种从入参搬过来的, 要跟着传递
    alias = {}
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;=][^;]*);", text):
        lhs, rhs = m.group(1), m.group(2)
        toks = set(re.findall(r"[A-Za-z_]\w*", rhs))
        if toks & (params | set(alias)):
            alias[lhs] = rhs
    tainted = set(params)
    for _ in range(8):
        grown = False
        for k, rhs in alias.items():
            if k in tainted:
                continue
            if set(re.findall(r"[A-Za-z_]\w*", rhs)) & tainted:
                tainted.add(k)
                grown = True
        if not grown:
            break
    out = []
    for m in re.finditer(r"InitBuffer\s*\(([^;]+?)\)\s*;", text, re.S):
        args = _split_args(m.group(1))
        if len(args) < 2:
            continue
        name = args[0].strip()
        size = args[-1].strip()
        seen = sorted(set(re.findall(r"[A-Za-z_]\w*", size)) & tainted)
        if not seen:
            continue
        # 顺着赋值链一路展开到**核函数入参**本身 —— 报"随 H/W 增长"才有行动意义,
        # 报"随 planeB 增长"只是把同一件事换个名字说。
        roots, seenSym, stack = set(), set(), list(seen)
        chain = []
        while stack:
            sym = stack.pop()
            if sym in seenSym:
                continue
            seenSym.add(sym)
            if sym in params:
                roots.add(sym)
                continue
            rhs = alias.get(sym)
            if rhs is None:
                continue
            if len(chain) < 4:
                chain.append(f"{sym} = {' '.join(rhs.split())[:56]}")
            stack.extend(t for t in re.findall(r"[A-Za-z_]\w*", rhs) if t in tainted)
        out.append((name, size, sorted(roots) or seen, chain))
    return out


# ---- U8b: 把 U8 从「列举」升成「定阈值 + 裁决」 ------------------------------------
#      U8 第一版只回答「哪几个 InitBuffer 的尺寸随运行期入参增长」。它在 softmax 上
#      **报中了** (CannBenchSoftmaxRow 的 inQ/outQ/wBuf 随 K 增长), 但同一份文件里
#      四个 kernel 函数的 12 个缓冲全都随 K 增长 —— 分块内核本来就该这样 —— 于是
#      真正危险的那一处淹在清单里, 没人处置。**判据报了而清单太长 = 没报**
#      (同型失效 2026-09-22 一天内第三次: alignlint A7? 5 条、opbuild B9 每次都报)。
#
#      这一版回答的是可裁决的那句话: **该参数取到多少时 UB 越线, 调用点拦没拦住**。
#      - 分函数各算各的 (每个 kernel 函数有自己的 TPipe, 不叠加);
#      - 别名链**限定在本函数体内**解析 —— 第一版是全文件收集赋值, 于是 RowSmall 的
#        `constexpr kRows = 248` 被 RowWide 的 `kRows = kRowWideWorkLanes / slot` 顶掉,
#        报出来的推导链张冠李戴;
#      - 对每个根参数二分出越线值, 再跟**调用点的包络守卫**比 (`if (K <= 8) RowSmall(...)`
#        这种)。越线值落在包络内 = 无守卫 = 判成立。
#
#      实证 (softmax 2026-09-22): CannBenchSoftmaxRow 报 `K >= 10816 时越线, 调用点无上限
#      守卫`; 真机 K=10432 过、K=16384 报 507035, 与阈值方向一致。榜上该算子 hidden
#      80 个 case 判失败 35 个, 从第 52 个起级联。

_CALL_GUARD = re.compile(r"if\s*\(([^;{}]{0,200}?)\)\s*\{?[^;{}]{0,120}?\b%s\s*[<(]")


_RET_HELPER = re.compile(
    r"(?:__aicore__|__inline__|inline|static|constexpr|\s)+[\w:<>]+\s+([A-Za-z_]\w*)\s*"
    r"\(([^;{}]*)\)\s*\{\s*return\s+([^;]+);\s*\}")


def return_helpers(text: str) -> dict:
    """设备端**单行 return** 的小函数: 名字 -> (形参名列表, 返回表达式)。

    ★为什么必须展开: 内核里的对齐辅助 (`AlignUp32(x, 32)` / `DqAlignUp(x, 32)`) 几乎
      总是这个形态, 而字节表达式几乎总要过一次它。不展开则整条失解 -> 缓冲不计入总量
      -> 报"没有随入参增长的 InitBuffer"。roi_align 全部 15 个缓冲、dynamic_quant
      全部 4 个都卡在这一条上 (2026-09-22)。
    """
    out = {}
    for m in _RET_HELPER.finditer(text):
        名 = m.group(1)
        ps = []
        for arg in m.group(2).split(","):
            arg = arg.strip()
            if not arg or arg == "void":
                continue
            nm = re.sub(r"[\[\]0-9]+$", "", arg.split()[-1]).lstrip("*&")
            if re.fullmatch(r"[A-Za-z_]\w*", nm or ""):
                ps.append(nm)
        if len(ps) == len(m.group(2).split(",")) or not m.group(2).strip():
            out[名] = (ps, m.group(3).strip())
    return out


def inline_helpers(expr: str, helpers: dict, depth: int = 3) -> str:
    """把 expr 里对单行 return 辅助函数的调用逐层替换成它的函数体。"""
    for _ in range(depth):
        hit = False
        for 名, (ps, body) in helpers.items():
            i = 0
            while True:
                m = re.search(r"\b" + re.escape(名) + r"\s*\(", expr[i:])
                if not m:
                    break
                st = i + m.end() - 1
                d, j = 0, st
                while j < len(expr):
                    if expr[j] == "(":
                        d += 1
                    elif expr[j] == ")":
                        d -= 1
                        if d == 0:
                            break
                    j += 1
                if j >= len(expr):
                    break
                args = [t.strip() for t in _split_args(expr[st + 1:j])] if expr[st + 1:j].strip() else []
                if len(args) != len(ps):
                    i = j
                    continue
                sub = body
                for pn, av in zip(ps, args):
                    sub = re.sub(r"\b" + re.escape(pn) + r"\b", f"({av})", sub)
                expr = expr[:i + m.start()] + f"({sub})" + expr[j + 1:]
                hit = True
                i = i + m.start()
            if hit:
                break
        if not hit:
            break
    return expr


def _scope_assigns(text: str, a: int, b: int) -> dict:
    """函数体 [a,b) 内的局部赋值 name -> 右值表达式。**只收本函数体内的** —— 跨函数
    同名局部量是 U8 第一版推导链张冠李戴的成因。后写覆盖先写 (同一函数内少见)。"""
    body = text[a:b]
    out = {}
    #  ★两遍, 且都用 setdefault (**先出现者胜**):
    #    第一遍只认带类型的声明 —— 那是这个量的定义;
    #    第二遍放宽到任意赋值 —— 类写法的成员量 (roi_align 的 `bandWp_ = ...`, 无类型前缀)
    #    否则整条推导链断在成员量上, 该算子一条都不报。
    #    先出现者胜是必需的: 放宽之后 `if (kRows > 255) { kRows = 255; }` 里的
    #    `kRows = 255` 也会被收进来, 后写覆盖会把 kRows 钉成 255, 于是一个有界的
    #    函数被按最大 tile 算成越线 (softmax RowWide 实撞)。
    for m in re.finditer(r"\b(?:constexpr\s+|const\s+|static\s+)*"
                         r"(?:u?int(?:8|16|32|64)_t|int|float|uint8_t|size_t|auto)\s+"
                         r"([A-Za-z_]\w*)\s*=\s*([^;]+);", body):
        out.setdefault(m.group(1), m.group(2).strip())
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;=][^;]*);", body):
        rhs = m.group(2).strip()
        if rhs and not rhs.startswith("="):
            out.setdefault(m.group(1), rhs)
    return out


def _scope_clamps(text: str, a: int, b: int) -> dict:
    """函数体内对某个量的上限截断: `if (x > LIT) { x = LIT; }` / `x = min(x, LIT)`。
    返回 name -> 上限字面量表达式。"""
    body = text[a:b]
    out = {}
    for m in re.finditer(r"if\s*\(\s*([A-Za-z_]\w*)\s*>\s*([^)]+?)\s*\)\s*\{?\s*"
                         r"\1\s*=\s*([^;]+);", body):
        out[m.group(1)] = m.group(3).strip()
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*(?:std::)?min\s*(?:<[^>]*>)?\s*\(", body):
        out.setdefault(m.group(1), None)      # 有截断但取不出界: 记成"有守卫、界未知"
    return out


def _fn_params(text: str, a: int) -> list:
    """函数签名里的形参名 (按序)。"""
    head = text[a:text.index("{", a) + 1]
    m = re.search(r"\(([^;{}]*)\)\s*\{$", head, re.S)
    if not m:
        return []
    out = []
    for arg in m.group(1).split(","):
        arg = arg.strip()
        if not arg or arg == "void":
            continue
        nm = re.sub(r"[\[\]0-9]+$", "", arg.split()[-1]).lstrip("*&")
        if re.fullmatch(r"[A-Za-z_]\w*", nm or ""):
            out.append(nm)
    return out


def early_return_guards(text: str, a: int, b: int) -> list:
    """函数体里**排在第一个 InitBuffer 之前**、且分支体内有 return 的 if 条件。

    ★这是守卫的第三种形态, 前两种是调用点守卫与 host 侧截断。软件里最自然的写法恰恰是
      它: `if (装不下) { 改走分块路径; return; }` 就写在同一个函数的开头。不认它就会
      在缺陷**已经修好之后**继续报警 —— 判据对良品报警等同于没有判据。
    ★求值而不是模式匹配: 条件原样代入当前探测值, 为真则这次探测根本到不了 InitBuffer,
      本函数用量记 0。这样 `kp * (4*sizeof(T)+8) + 4096 > 240000` 这种**算出来的**
      预算守卫也能精确认出, 不必去猜它拦住的是哪个变量。
    """
    body = text[a:b]
    ib = body.find("InitBuffer")
    head = body if ib < 0 else body[:ib]
    out = []
    for m in re.finditer(r"\bif\s*\(", head):
        i = m.end() - 1
        d, j = 0, i
        while j < len(head):
            if head[j] == "(":
                d += 1
            elif head[j] == ")":
                d -= 1
                if d == 0:
                    break
            j += 1
        if j >= len(head):
            continue
        cond = head[i + 1:j]
        tail = head[j + 1:j + 400]
        stmt = tail
        k = tail.find("{")
        if k >= 0 and not tail[:k].strip():
            d2, e = 0, k
            while e < len(tail):
                if tail[e] == "{":
                    d2 += 1
                elif tail[e] == "}":
                    d2 -= 1
                    if d2 == 0:
                        break
                e += 1
            stmt = tail[:e + 1]
        else:
            stmt = tail.split(";")[0] + ";"
        if re.search(r"\breturn\b", stmt):
            out.append(cond)
    return out


def _cond_true(cond: str, vals: dict) -> bool:
    """条件在这组取值下是否为真。解不出按**假**处理 —— 即"守卫没拦住", 宁可多报。"""
    c = cond.replace("&&", " and ").replace("||", " or ")
    c = re.sub(r"(?<![=!<>])!(?!=)", " not ", c)
    v, _ = _resolve(c, vals)
    return bool(v)


def call_envelope(text: str, fn: str, params: list, consts: dict) -> dict:
    """调用点包络: 在调用 fn 的那条语句往前找同一个 `if` 里的 `<形参名> <= LIT` 守卫。

    ★只采信**与被调函数形参同名**的守卫量。按位置做实参↔形参映射在模板+转发下不可靠,
      而 dispatch 写法里两边同名是压倒性惯例; 采信不了就如实返回空 = 无包络 (不是"有界")。
    """
    env = {}
    pat = re.compile(r"if\s*\(([^{};]{0,200})\)\s*\{?[^{};]{0,160}?\b" + re.escape(fn) + r"\s*[<(]")
    for m in pat.finditer(text):
        cond = m.group(1)
        for cm in re.finditer(r"\b([A-Za-z_]\w*)\s*(<=|<)\s*([A-Za-z_]\w*|\d+)", cond):
            nm, op, rhs = cm.group(1), cm.group(2), cm.group(3)
            if nm not in params:
                continue
            v, _ = _resolve(rhs, consts)
            if v is None:
                continue
            hi = v if op == "<=" else v - 1
            env[nm] = min(env.get(nm, hi), hi)
    return env


def _scope_bytes(text: str, a: int, b: int, vals: dict, helpers: dict | None = None):
    """把函数体 [a,b) 内全部 InitBuffer 的字节数求和 (同名取最大)。
    返回 (字节数, 未解析条数)。"""
    body = text[a:b]
    per, unres = {}, 0
    for m in re.finditer(r"InitBuffer\s*\(([^;]+?)\)\s*;", body, re.S):
        args = _split_args(m.group(1))
        if len(args) < 2:
            continue
        name = args[0].strip()
        sz, err = _resolve(inline_helpers(args[-1].strip(), helpers or {}), vals)
        if sz is None:
            unres += 1
            continue
        num = 1
        if len(args) >= 3:
            n, _e = _resolve(args[1].strip(), vals)
            if n is None:
                unres += 1
                continue
            num = n
        per[name] = max(per.get(name, 0), sz * num)
    return sum(per.values()), unres


def _fixpoint(assigns: dict, vals: dict, rounds: int = 8) -> dict:
    v = dict(vals)
    for _ in range(rounds):
        grew = False
        for k, rhs in assigns.items():
            if k in v:
                continue
            got, _e = _resolve(rhs, v)
            if got is not None:
                v[k] = got
                grew = True
        if not grew:
            break
    return v


def host_clamp_bounds(host_path: str, consts: dict) -> dict:
    """host/plugin 源码里对某个量的**上限截断**, 顺着赋值链传递。返回 名字 -> 上限值。

    ★接法与 call_envelope 同一条假设: 按**同名**把 host 侧的量与核函数形参对上。
      host 里 `if (IT > 1024) { IT = 1024; }` 拦住的就是下发给内核的 IT。名字对不上
      就不采信 (返回里没有 = 无守卫), 宁可误报也不假装有界。
    ★这一档只给 `guarded_host?` —— 带问号, 因为"同名"和"链上有截断即整条有界"两条
      都是**充分性不足**的近似: 它足以把真守卫排除出告警, 但不足以证明安全。
    """
    text = open(host_path, encoding="utf-8", errors="replace").read()
    assigns = {}
    for m in re.finditer(r"\b(?:constexpr\s+|const\s+|static\s+)*"
                         r"(?:u?int(?:8|16|32|64)_t|int|auto|float)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);",
                         text):
        assigns.setdefault(m.group(1), m.group(2).strip())
    clamps = {}
    for m in re.finditer(r"if\s*\(\s*([A-Za-z_]\w*)\s*>\s*([^)]+?)\s*\)\s*\{?\s*\1\s*=\s*([^;]+);",
                         text):
        v, _ = _resolve(m.group(3), consts)
        if v is not None:
            clamps[m.group(1)] = min(clamps.get(m.group(1), v), v)
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*(?:std::)?min\s*(?:<[^>]*>)?\s*\(([^;]+)\);", text):
        for part in _split_args(m.group(2)):
            v, _ = _resolve(part, consts)
            if v is not None:
                clamps[m.group(1)] = min(clamps.get(m.group(1), v), v)
    #  顺赋值链传播: NK <- nkMax(<=128) 这种
    out = dict(clamps)
    for name in list(assigns) + list(clamps):
        seen, stack, best = set(), [name], out.get(name)
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            if t in clamps:
                best = clamps[t] if best is None else min(best, clamps[t])
            rhs = assigns.get(t)
            if rhs:
                stack.extend(re.findall(r"[A-Za-z_]\w*", rhs))
        #  `while (NK * 2 <= nkMax) { NK *= 2; }` 形态: NK 由 nkMax 封顶
        for m in re.finditer(r"while\s*\(\s*" + re.escape(name) + r"\s*\*[^)]*<=\s*([A-Za-z_]\w*)",
                             text):
            if m.group(1) in clamps:
                b = clamps[m.group(1)]
                best = b if best is None else min(best, b)
        if best is not None:
            out[name] = best
    return out


def scaling_threshold(src_path: str, soc: str | None = None, base_vals: dict | None = None,
                      probe_max: int = 1 << 21, safe_margin: int = 1024,
                      host_path: str | None = None) -> list:
    """每个 kernel 函数 x 每个根入参: 二分出「本函数 UB 用量越线」的最小取值, 与调用点
    包络比对后裁决。返回 [dict]。

    ★单调性前提: 缓冲字节数对该参数单调不减。不满足时二分给的不是最小越线值, 所以
      越线后再向上抽查一格确认仍越线 —— 抽查不过就记成 `nonmonotone` 而不给裁决。
    """
    text = open(src_path, encoding="utf-8", errors="replace").read()
    cap, capsrc = ub_cap_for(soc)
    limit = cap - safe_margin
    consts = dict(source_constants(src_path))
    if base_vals:
        consts.update(base_vals)
    hostb = host_clamp_bounds(host_path, consts) if host_path else {}
    file_assigns = _scope_assigns(text, 0, len(text))
    helpers = return_helpers(text)
    file_assigns = {k: inline_helpers(v, helpers) for k, v in file_assigns.items()}
    #  模板 kernel 的 sizeof(T) 解不出 —— 不猜一个默认值, 而是**按每种元素宽度各探一次**,
    #  裁决取"任一宽度下无守卫即成立", 并在结论里点名是哪一种。只影响越线阈值的数值,
    #  不影响"该维无界"这个结论 (无界对任何元素宽度都成立)。
    elem_bytes = tuple((base_vals or {}).pop("_elem_bytes", None) or (4, 2))
    rows = []
    for (fn, a, b) in _kernel_scopes(text):
        if "InitBuffer" not in text[a:b]:
            continue
        params = _fn_params(text, a)
        #  ★两级: 本函数体内的赋值**优先**, 文件级的兜底。
        #    只用局部 -> 类写法的成员量 (roi_align 的 planeB 在 LoadGeo 里赋值、在
        #    构造里 InitBuffer) 全解不出, 整个算子一条都不报 (静默漏报);
        #    只用文件级 -> 同名局部量串台 (softmax 四个 kernel 各有一个 kRows),
        #    把有界的函数误判成越线。两级合并才两边都对。
        local = {k: inline_helpers(v, helpers) for k, v in _scope_assigns(text, a, b).items()}
        guards = [inline_helpers(g, helpers) for g in early_return_guards(text, a, b)]
        assigns = dict(file_assigns)
        assigns.update(local)
        clamps = _scope_clamps(text, a, b)
        env = call_envelope(text, fn, params, consts)
        #  只探那些真的进了尺寸表达式的入参
        used = set()
        for m in re.finditer(r"InitBuffer\s*\(([^;]+?)\)\s*;", text[a:b], re.S):
            expr = m.group(1)
            seen, stack = set(), list(re.findall(r"[A-Za-z_]\w*", expr))
            while stack:
                t = stack.pop()
                if t in seen:
                    continue
                seen.add(t)
                if t in params:
                    used.add(t)
                elif t in assigns:
                    stack.extend(re.findall(r"[A-Za-z_]\w*", assigns[t]))
        sizeofs = sorted(set(re.findall(r"sizeof\s*\(\s*([A-Za-z_]\w*)\s*\)", text[a:b])))
        known_t = ("float", "half", "bfloat16_t", "int32_t", "uint32_t", "int64_t",
                   "uint16_t", "uint8_t", "int8_t", "int16_t")
        unknown_so = [f"sizeof_{t}" for t in sizeofs
                      if f"sizeof_{t}" not in consts and t not in known_t]
        probes = [({}, "")] if not unknown_so else [
            ({k: eb for k in unknown_so}, f" [元素 {eb} 字节]") for eb in elem_bytes]
        for p in sorted(used):
            for extra, tag in probes:
                def total(v, _extra=extra, _p=p):
                    #  ★文件级常量里**同名**的项必须先剔掉再代入本函数的局部赋值:
                    #    softmax 四个 kernel 各有一个 kRows, RowSmall 的是 constexpr 248,
                    #    RowWide 的是 4096/slot。不剔就把 248 代进 RowWide, 把一个有界的
                    #    函数误判成 K>=41 越线 (实际它恒定约 4096 lane)。
                    vv = {k: x for k, x in consts.items() if k not in local}
                    vv.update(_extra)
                    vv[_p] = v
                    for q in params:             # 其余入参取 1, 只看这一维的增长
                        vv.setdefault(q, 1)
                    vv = _fixpoint({k: r for k, r in assigns.items() if k not in vv}, vv)
                    #  提前返回的守卫先判: 拦住了就根本到不了 InitBuffer
                    for g in guards:
                        if _cond_true(g, vv):
                            return 0, 0
                    return _scope_bytes(text, a, b, vv, helpers)

                row = {"fn": fn, "param": p, "elem": tag.strip(), "cap": cap, "capsrc": capsrc}
                t0, u0 = total(1)
                thi, _uh = total(probe_max)
                if u0 or thi is None:
                    row.update(verdict="unresolved",
                               note=tag + f"本函数有 {u0} 个缓冲解不出, 不给裁决")
                elif thi <= limit:
                    row.update(verdict="bounded",
                               note=tag + f"{p} 取到 {probe_max} 仍只用 {thi} B —— 不随它无限增长")
                elif t0 > limit:
                    row.update(verdict="over_at_one", note=tag + f"{p}=1 就已 {t0} B > {limit} B")
                else:
                    lo, hi = 1, probe_max        # total(lo) <= limit < total(hi)
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        tm, um = total(mid)
                        if um or tm <= limit:
                            lo = mid
                        else:
                            hi = mid
                    t_up, _u2 = total(min(probe_max, hi * 2))
                    bound, clamped = env.get(p), (p in clamps)
                    if t_up <= limit:
                        row.update(verdict="nonmonotone",
                                   note=tag + f"{p}={hi} 越线但 {min(probe_max, hi*2)} 不越 —— "
                                              f"非单调, 不给裁决")
                    elif bound is not None and bound < hi:
                        row.update(verdict="guarded", cross=hi, bound=bound,
                                   note=tag + f"{p} >= {hi} 才越线, 而调用点已限 {p} <= {bound}")
                    elif hostb.get(p) is not None and hostb[p] < hi:
                        row.update(verdict="guarded_host?", cross=hi, bound=hostb[p],
                                   note=tag + f"{p} >= {hi} 才越线, 而 host 侧按同名量把它"
                                              f"限在 <= {hostb[p]} —— 同名对接是近似, 需人工确认。"
                                              f"★**这一档不证明安全**: 它只说'host 里有一条上限',"
                                              f"没核对**这条上限够不够**。两处已实撞:"
                                              f" (1) 多个量联动时, 单个量各自有界而**组合**仍越线;"
                                              f" (2) host 用来定 tiling 的**每元素成本模型**"
                                              f"比内核真实用量小 —— softmax 的 perElem 只算了一份"
                                              f"fp32 tile 而内核用两份, 靠 budget 取 200000(而非"
                                              f"253952)加 NK 向下取 2 的幂两道余量盖着;"
                                              f"把 IT 往大里搜就爆 507035 (2026-09-23)。"
                                              f"核对办法: 按 host 的公式反推它允许的 (tile 各维) 组合,"
                                              f"代进本模块求和, 看是否 <= cap。")
                    elif clamped:
                        row.update(verdict="guarded", cross=hi, bound=None,
                                   note=tag + f"{p} >= {hi} 越线, 但函数内对它有截断 —— "
                                              f"界值取不出, 需人工确认")
                    else:
                        row.update(verdict="UNGUARDED", cross=hi, bound=bound,
                                   note=tag + f"{p} >= {hi} 时本函数 UB 用量 > {limit} B, "
                                              f"且调用点无上限守卫 —— 该维一旦超出公开集范围"
                                              f"即 507035 并级联")
                rows.append(row)
    return rows


# ---- U9: **host 侧挑 tile 的搜索循环没有「一个都不满足」的兜底护栏** --------------
#      形态: 在候选 tile 尺寸上循环, 用 `bytes(cand) <= budget` 挑最大的一个, 命中就
#      break。若一个候选都不满足, 循环自然结束而变量停在初值, 然后**照常下发** ——
#      内核按这个放不下的尺寸开 UB, 越界打挂设备 (507035), 评测把后面一串判
#      「设备不可恢复, 用例跳过」。
#      一个崩溃 case 的代价是它自己那一份的数倍: moe_gating_top_k_softmax 的 8 个
#      hidden 失败里 5 个是级联, roi_align 的 19 个里 13 个是, sparse_flash_attention
#      的 21 个里 14 个是。
#      正面例子就在同一份代码库里: sparse_flash_attention 的 plugin 用 `bool fits`
#      记住有没有命中, 循环后 `TORCH_CHECK(fits, ...)`, 注释写明「一个被拒的 case 值
#      1.5 分, 一次设备故障赔上整轮和这张卡上别人的任务」。

_U9_BUDGET = re.compile(r"<=\s*(\w*[Bb]udget\w*|\w*UB\w*|\w*UbSize\w*[^;\n]*)")


def host_guard_report(text: str) -> list:
    """返回 [(行号, 循环首行, 说明)]。只看 plugin/host 源码。"""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if not re.search(r"\bfor\s*\(|\bwhile\s*\(", line):
            continue
        #  循环体: 到配对的右括号为止 (粗略地用缩进/花括号计数)
        depth, j, started = 0, i, False
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                started = True
            if started and depth <= 0:
                break
            j += 1
        body = "\n".join(lines[i:j + 1])
        if not _U9_BUDGET.search(body) or "break" not in body:
            continue
        #  循环之后 20 行内要么 TORCH_CHECK, 要么有一个布尔命中标记参与判断
        after = "\n".join(lines[j + 1:j + 21])
        if "TORCH_CHECK" in after:
            continue
        if re.search(r"\b(fits|ok|found|feasible)\b", body) and \
           re.search(r"\b(fits|ok|found|feasible)\b", after):
            continue
        out.append((i + 1, lines[i].strip()[:70],
                    "按预算挑 tile 的搜索循环, 循环后 20 行内既没有 TORCH_CHECK 也没有"
                    "命中标记 —— 一个候选都不满足时变量停在初值并照常下发, "
                    "内核会按放不下的尺寸开 UB (507035), 并带崩后续 case"))
    return out

# ============ 别名分组与生命周期结构 (2026-09-20 立) ============
#  ★动机 (dubin 提的角度): "workspace 复用也是重要优化角度, 有工具查看占用**生命周期**吗?
#    避免死亡变量持续占用周期"。查了: 仓里 184 条子判据**关于生命周期的零条**。
#    最接近的两条都不是 —— pipeplan B1 有时间轴但只管 L1/L0 的流水缓冲;
#    本模块的 U7_afford 要**人手点名**哪几个同时存活, 工具自己算不出来。
#
#  ★本节要回答的三件事 (都靠静态解析, 不上机):
#    1. 哪些缓冲**共用同一个基址** (= 手写的别名分组, 一次人肉生命周期优化)。
#       漏一个重叠 = 静默数据损坏; 多一个保守分组 = 白占空间。**此前没有任何工具验证它**。
#    2. 有没有**吃光式分配** —— `GetWithOffset(总量 - 某偏移, 某偏移)` 这种"剩下的全归我"。
#       它会让"声明占用"永远等于池容量, 于是容量判据恒报 100%, **分不出真满与假满**。
#    3. 由 1 推出的 **max/sum 结构**: 同一基址组内取 max, 组间求和。
#       ★这正是 host 侧 UB 预算模型必须复现的结构 —— 两边对不上就是两本账:
#         host 多算 -> 逼出更小的分块 -> 更多 task -> 更多固定开销 (白亏, 且很难发现);
#         host 少算 -> 内核越界 -> 运行期 507015 或静默算错。
#       sfag 2026-09-20 实测: 该算子的 host 模型写的是
#         `max(gatherSize, softmaxParamSize + max(softmaxWorkSize, scatterSize))`,
#       与内核的两组别名**结构一致**, 且注释明写"Keep this calculation in the same order as
#       VecOp::InitUB" —— 是正例。**本工具存在的意义是把这种一致性变成可自动核对的, 而不是靠注释。**

_GETOFF = re.compile(r"(\w+)\s*=\s*\w+\.GetWithOffset<([^>]+)>\s*\(([^,]+),\s*([^)]+)\)")


def alias_groups(src: str) -> dict:
    """{基址键: [(缓冲名, 长度表达式, 元素类型)]} —— 真正共用基址的才算一个别名组。

    ★2026-09-20 本函数第一版就是个假阳性: 它按**表达式文本**分组, 于是把
      `x = q.GetWithOffset(..., ubOffset); ubOffset += ...; y = q.GetWithOffset(..., ubOffset);`
      里的 13 块全报成"共用 ubOffset" —— 而那是个**每次分配后自增的游标**, 它们是**顺序排布**。
      按文本分组会把顺序说成别名, 于是占用从"求和"变成"取 max", **低估到灾难性**。
      判法: 两次分配之间, 那个偏移标识符有没有被**重新赋值或自增**。有 -> 顺序, 无 -> 别名。
    """
    hits = [(m.start(), *(x.strip() for x in m.groups())) for m in _GETOFF.finditer(src)]
    g, seq = {}, 0
    for i, (pos, name, ty, ln, off) in enumerate(hits):
        key = off
        if re.fullmatch(r"\w+", off):
            #  往前找同 off 的上一次分配; 两者之间若有 `off =` 或 `off +=` 就是游标推进
            prev = None
            for j in range(i - 1, -1, -1):
                if hits[j][4] == off:
                    prev = hits[j][0]
                    break
            if prev is not None:
                between = src[prev:pos]
                if re.search(re.escape(off) + r"\s*(\+=|=[^=])", between):
                    seq += 1
                    key = f"{off}#顺序{seq}"      # 游标推进过 -> 不是别名, 各占各的
        g.setdefault(key, []).append((name, ln, ty))
    return g


def eat_all_allocs(src: str) -> list:
    """吃光式分配: 长度写成 `<池容量> - <某偏移>` 这种形态。"""
    out = []
    for m in _GETOFF.finditer(src):
        name, ty, ln, off = (x.strip() for x in m.groups())
        if re.search(r"\w+\s*-\s*\w+", ln):
            out.append((name, ln, off))
    return out


def cmd_alias(path: str) -> int:
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        print(f"读不了 {path}: {e}", file=sys.stderr)
        return 2
    g = alias_groups(src)
    if not g:
        print(f"{path}: 没有找到 `x = que.GetWithOffset<T>(len, off)` 形态的分配 —— "
              f"**本检查不适用**, 不是'没有别名'。")
        return 2
    eats = eat_all_allocs(src)
    print(f"=== 别名分组 (共用基址 = 一个组; 组内取 max, 组间求和) — {path} ===")
    multi = 0
    for off, items in sorted(g.items(), key=lambda kv: -len(kv[1])):
        tag = "  ★**别名组**" if len(items) > 1 else ""
        print(f"  基址 {off[:42]:44s} {len(items)} 块{tag}")
        for n, ln, ty in items:
            print(f"      {n:26s} <{ty[:12]:12s}> 长 {ln[:46]}")
        multi += 1 if len(items) > 1 else 0
    print(f"\n  共 {len(g)} 个基址位置, 其中 {multi} 个是**真别名组** (期间游标未推进)。")
    print(f"  ★标 `#顺序N` 的是同一个游标变量推进后的不同位置 —— **顺序排布, 各占各的**,")
    print(f"    不是别名。把它们当别名会把占用从求和变成取 max, 低估到灾难性。")
    print(f"  ⇒ 占用应为: **组间求和, 组内取 max** ——")
    print(f"     host 侧的 UB 预算模型必须复现这个结构, 否则两本账:")
    print(f"       host 多算 -> 逼出更小的分块 -> 更多 task -> 白亏且难发现;")
    print(f"       host 少算 -> 内核越界 -> 507015 或静默算错。")
    if eats:
        print(f"\n★**吃光式分配** ({len(eats)} 处) —— 长度写成'总量减某偏移', 即'剩下的全归我':")
        for n, ln, off in eats:
            print(f"     {n:26s} 长 {ln[:56]}")
        print(f"   ⇒ 有它在, **声明占用恒等于池容量**, 容量判据会恒报 100% ——")
        print(f"     那时'真满'与'假满'长得一样。判'还有没有空间'不能只看声明值,")
        print(f"     要去看 host 的预算模型 (它才决定分块开多大)。")
    try:
        from core.trust import verdict_banner
        for ln_ in verdict_banner("ub_budget", refusing=False):
            print(ln_)
    except Exception:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="*", help="一个或多个源文件 —— 同一个核上所有 UB 分配都要一起算 "
                                          "(FIA arch35: kernel 文件开 BufferManager 池 + vec 文件 InitBuffer)")
    ap.add_argument("--values", default="{}")
    ap.add_argument("--soc", default=None, help="芯片**型号** (默认按本机探测); 注意 ascend910b 是 CANN 目录系列名, 作型号它指另一颗芯片")
    ap.add_argument("--mem", default="UB", choices=["UB", "L1"],
                    help="算哪块片上存储。UB: 容量查 socspec; L1: 容量读源码里 "
                         "BufferManager<BufferType::L1>::Init(pipe, <字面量>) 的字面量")
    ap.add_argument("--alias", action="store_true",
                    help="**别名分组与生命周期结构**: 哪些缓冲共用基址(组内取max), "
                         "有没有吃光式分配, 以及 host 预算模型必须复现的 max/sum 结构")
    ap.add_argument("--dual-cap", action="store_true",
                    help="U6: 只列同一缓冲的两种容量口径 (读写两端可能各选一支)")
    ap.add_argument("--afford", default=None, metavar="BUF1,BUF2,...",
                    help="U7: 点名几个**同时存活**的缓冲, 把它们加起来跟池容量比 —— 用来在动手前"
                         "判掉'把某个行数常量调大'这类方案 (配 --values 改那个常量)。"
                         "只证伪不证安全: 和超了就一定不成立, 没超**不给结论**")
    ap.add_argument("--asserts", action="store_true",
                    help="U9: 解源码自带的 static_assert 容量约束, 逐条报**余量**。"
                         "U7 只认 TBuf 池 `InitBuffer(buf, cap)`; 从**已分配好的队列**里切槽位"
                         "那一类它报 U7_no_pool, 而约束往往已写成 static_assert —— 这条解它")
    ap.add_argument("--afford-const", default=None, metavar="常量名",
                    help="U9 配套: 那个常量**最大能调到多少** (会重解由它派生的常量; "
                         "指数上探+二分并显式验边界, 非单调时退回线性扫)")
    ap.add_argument("--scaling-sweep", metavar="ROOT",
                    help="U8 普查: 遍历 ROOT 下每个 <op>/op_kernel/*_kernel.h (自动配同算子的 "
                         "op_plugin/*.cpp), 只列无守卫的越线维。回答'我们还有哪些算子会在"
                         "隐藏集的大形状上打挂设备'")
    ap.add_argument("--host", help="U8 配套: host/plugin 源码 —— 内核形参的上限守卫常写在 host 侧, "
                                   "不给它就会把 host 已经拦住的维报成无守卫")
    ap.add_argument("--host-guard", action="store_true",
                    help="U9: 扫 host/plugin 源码里「按预算挑 tile 的搜索循环」有没有"
                         "「一个都不满足」的兜底护栏。缺护栏 = 越界打挂设备并带崩后续 case")
    ap.add_argument("--scaling", action="store_true",
                    help="U8: 列出尺寸表达式里含**核函数运行期入参**的 InitBuffer —— "
                         "UB 占用随输入尺寸增长的地方, 每一处都要么 host 侧有上限护栏, 要么分块。"
                         "只定位不定量: 它答的是'哪一维会把 UB 撑爆', 不是'现在占了多少'")
    ap.add_argument("--refs", action="store_true", help="打印本模块全部判据的出处/实证/最后验证日期")
    ap.add_argument("--selftest", action="store_true", help="判据负例自检: 超 cap / 解不出必须被抓住")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.refs:
        print_refs("ub_budget", REFS)
        return 0
    if a.host_guard:
        if not a.src:
            ap.error("要给至少一个源文件")
        rc = 0
        for f in a.src:
            text = open(f, encoding="utf-8", errors="replace").read()
            rows = host_guard_report(text)
            print(f"--- {f}")
            if not rows:
                print("    按预算挑 tile 的搜索循环都有兜底 (或没有这类循环)")
                continue
            rc = 1
            for ln, head, why in rows:
                print(f"    {ln}: {head}")
                print(f"        {why}")
        return rc
    if a.scaling_sweep:
        import glob as _glob
        ks = sorted(_glob.glob(os.path.join(a.scaling_sweep, "**", "op_kernel", "*_kernel.h"),
                               recursive=True))
        n_ung, n_und, rc = 0, 0, 0
        print(f"U8 普查: {len(ks)} 个 kernel 文件")
        for k in ks:
            opdir = os.path.dirname(os.path.dirname(k))
            hs = sorted(_glob.glob(os.path.join(opdir, "op_plugin", "*.cpp")))
            try:
                rows = scaling_threshold(k, a.soc, None, host_path=hs[0] if hs else None)
            except Exception as e:
                print(f"  ? {k}: 算不出 ({e})")
                continue
            tag = os.path.basename(opdir)
            for r in rows:
                if r["verdict"] == "UNGUARDED":
                    n_ung += 1
                    print(f"  U8! {tag}/{r['fn']}({r['param']}): {r['note']}")
                elif r["verdict"] in ("unresolved", "nonmonotone"):
                    n_und += 1
            if not hs:
                print(f"  ? {tag}: 没找到 op_plugin/*.cpp —— host 侧守卫看不到, 结论偏严")
        print(f"\n无守卫的越线维 {n_ung} 处; 另有 {n_und} 处不给裁决 "
              f"(解不出/非单调 —— 在它们上面'没报'什么都不能证明)")
        return 1 if n_ung else 0
    if a.scaling:
        if not a.src:
            ap.error("要给至少一个源文件")
        rc = 0
        vals = json.loads(a.values) if getattr(a, "values", None) else None
        for f in a.src:
            print(f"--- {f}")
            try:
                rows = scaling_threshold(f, a.soc, vals, host_path=a.host)
            except Exception as e:                  # 容量取不到等: 如实报, 不回落
                print(f"    算不出, 不给裁决: {e}")
                rc = max(rc, 2)
                continue
            if not rows:
                #  ★同样先问方言: 0 行既可能是"真的有守卫", 也可能是"根本没解析到"
                if not _dialect.not_effective("ub_budget(U8)", [f], "运行期可变的 InitBuffer"):
                    print("    没有尺寸随运行期入参增长的 InitBuffer")
                continue
            bad = [r for r in rows if r["verdict"] == "UNGUARDED"]
            for r in bad:
                print(f"    U8! {r['fn']}({r['param']}): {r['note']}")
            #  其余按裁决归档成一行 —— 有界/已守卫的**不逐条列**, 列表长了等于没报
            rest = {}
            for r in rows:
                if r["verdict"] != "UNGUARDED":
                    rest.setdefault(r["verdict"], []).append(f"{r['fn']}({r['param']})")
            for k, v in sorted(rest.items()):
                print(f"    [{k}] {len(v)} 项: {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
            if bad:
                rc = 1
            if any(r["verdict"] in ("unresolved", "nonmonotone") for r in rows):
                print("    ★有不给裁决的项 —— 「没报 U8!」在它们上面什么都不能证明")
            if not bad:
                print(f"    没有无守卫的越线维 (cap={rows[0].get('cap')} {rows[0].get('capsrc', '')})")
            #  ★这句"没有无守卫的越线维"本身会让人放心过头 (2026-09-23 实撞)。guarded_host?
            #    只说"host 里有一条上限", 不说"这条上限够不够"。把限制写进**摘要**而不是
            #    只写在逐行 note 里 —— 摘要是实际被读的那一行。
            _gh = [r for r in rows if r["verdict"] == "guarded_host?"]
            if _gh:
                print(f"    ★其中 {len(_gh)} 项是 guarded_host? —— **这一档不证明安全**, 只说"
                      f"host 里有一条同名上限, 没核对**这条上限够不够**。")
                print(f"      已实撞两种不够: (1) 多个量各自有界而**组合**仍越线;"
                      f" (2) host 定 tiling 用的**每元素成本模型比内核真实用量小**"
                      f" —— softmax 的 perElem 只算了一份 fp32 tile 而内核用两份,"
                      f" 靠 budget 取 200000 (而非 253952) 加 NK 向下取 2 的幂两道余量盖着,"
                      f" 把 IT 往大里搜就爆 507035。")
                print(f"      核对办法: 按 host 的公式反推它允许的 tile 维组合, 代进"
                      f"`ub_budget --values` 求和, 看是否 <= cap。")
            if not a.host and any(r["verdict"] == "UNGUARDED" for r in rows):
                print("    ★没给 --host: 内核形参的上限守卫常写在 host tiling 里, "
                      "先补 --host <plugin.cpp> 再判")
        return rc
    if not a.src:
        ap.error("要给至少一个源文件")
    if a.alias:
        if not a.src:
            print("要给一个 kernel .h —— 例: opdbg ub_budget --alias <vec_op.h>", file=sys.stderr)
            return 2
        return cmd_alias(a.src[0])
    if a.dual_cap:
        rows = check_dual_capacity(a.src)
        if not rows:
            print("U6: 没有缓冲出现两种容量口径")
            return 0
        print(f"U6: {len(rows)} 个缓冲有**两种容量口径** —— 读写两端必须选同一支, 静态判不了对错, 逐个核对:")
        for it in rows:
            print(f"  {it['buffer']}  ({it['file']})")
            for al in it["allocs"]:
                g = " <- " + " && ".join(al["guard"]) if al["guard"] else " <- (无守卫, 兜底)"
                print(f"      行 {al['line']}: {al['expr']}{g}")
            if it["cap_symbols_at_use"]:
                print(f"      访问点引用的容量常量: {', '.join(it['cap_symbols_at_use'])}")
            if it["guards_at_use"]:
                print(f"      访问点守卫: {' | '.join(it['guards_at_use'][:6])}")
        print("  ★核对法: 上面每一支分配都标了决定它的条件, 访问点的条件也列了出来 —— "
              "两侧必须逐字一致, 不一致就是越界。sfag 2026-09-16: 分配按 enableOptimizedScatter, "
              "gather 按 IS_DETERMINISTIC, det=1 开门时分叉 -> 按 32 行给、按 64 行写 -> 507015。"
              "★注意**无守卫的那一支是兜底**: 它总会执行, 于是门开时生效的是它, 不是门内那支。")
        return 0
    if a.asserts or a.afford_const:
        #  ★U9: 约束**本来就写在源码里** (static_assert), 判据要做的不是重新推导, 是把它解出来。
        vals = source_constants(a.src[0]); vals.update(json.loads(a.values))
        text = "\n".join(open(f).read() for f in a.src)
        rows = assert_margin(text, vals, a.afford_const, src_path=a.src[0])
        bad = 0
        for it in rows:
            tag = it.get("level", "")
            ln = f":{it['line']}" if "line" in it else ""
            print(f"[{it['check']}{ln}] {tag} {it.get('msg') or it.get('err')}")
            if it.get("level") == "严重":
                bad = 1
        return bad
    if a.afford:
        vals = source_constants(a.src[0]); vals.update(json.loads(a.values))
        text = "\n".join(open(f).read() for f in a.src)
        names = [x.strip() for x in a.afford.split(",") if x.strip()]
        rows = afford(text, names, vals)
        bad = 0
        for it in rows:
            print(f"{it['check']}: {it['err']}")
            if it["check"] in ("U7_over", "U7_unresolved", "U7_unknown_buffer", "U7_no_pool"):
                bad = 1
        return bad
    r = analyze(a.src, json.loads(a.values), a.soc, a.mem)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    #  ★一个缓冲都没解析到时**先问方言**, 再决定要不要说 OK。
    #    2026-10-21 实撞 (SMLA, CANNBot-DSL): 本模块打印 "UB: 0/253952 B (0%), OK" 并 rc=0,
    #    而源码里根本没有 InitBuffer —— 那不是"用量为零", 是**没看见**。
    _ne = (not r["buffers"]) and _dialect.not_effective("ub_budget", a.src, "InitBuffer/TBuf 分配")
    if _ne:
        print(f"\n{a.mem}: 未生效 (0 个缓冲被解析到, 不代表用量为 0)")
        return 0
    #  ★第三种 0: 方言对了, 但解析器没认出这种写法 (2026-10-21: ffn_kernel.h 里有
    #    `pipe->InitBuffer(buf, AlignUp<...>(...))` 而本模块解析出 0 个缓冲, 仍报 "OK")。
    #    「解析器漏读」与「真没有」必须分开说, 否则 OK 会替一个没跑过的判据背书。
    if not r["buffers"]:
        _n = _dialect.count(a.src, "InitBuffer")
        if _n:
            print(f"\n★解析器漏读: 源码里有 {_n} 处 `InitBuffer`, 但本模块解析到 0 个缓冲 —— "
                  f"下面是**没读到**的默认值, 不是测量结果。把这种写法报给本模块补解析, "
                  f"或先用 --values 手工代入。")
            return 0
    #  ★U6 挂进**默认**路径, 不要求记得敲 --dual-cap ——
    #    sfag 2026-09-16 那次 UB 总量一直判到 99% 通过, 越界的是同一缓冲的两种口径,
    #    总量这条判据结构上就看不见它。只提示不改退出码 (哪支配哪支是语义, 静态判不了)。
    dc = check_dual_capacity(a.src)
    if dc:
        print(f"\n★U6 提示: {len(dc)} 个缓冲有**两种容量口径** —— 总量通过不代表没越界。"
              f"敲 --dual-cap 看每支的守卫条件与访问点。")
        for it in dc[:5]:
            gs = ["/".join(al["guard"]) or "兜底" for al in it["allocs"]]
            print(f"    {it['buffer']}: " + " vs ".join(
                f"{al['expr']} [{g}]" for al, g in zip(it["allocs"], gs)))
    #  ★有 UNRESOLVED 时 total 是**下界**, 摘要行不能只说 OK ——
    #    2026-10-21 实撞 (ffn_kernel.h): 唯一那个缓冲的尺寸表达式求不出值,
    #    摘要仍打印 "0/253952 B (0%), OK", 而 json 的 note 里才写着"下界"。
    #    摘要行是实际被读的那一行。
    _tail = ('OVER CAP!!' if r['over_cap'] else
             (f"下界 (还有 {r['unresolved_count']} 个缓冲的尺寸求不出 —— **不能据此判 OK**)"
              if r.get('unresolved_count') else 'OK'))
    #  U11: 顶行必须与结构化判定用同一个数。多函数时判定取**单函数最大值**,
    #  这一行若打全文件求和, 就是工具自己 note 里说"不能这么判"的那种判法。
    _pf = r.get("per_function")
    if _pf and _pf.get("worst_fn"):
        _b, _n = _pf["worst_bytes"], _pf["worst_fn"]
        _vs = (f", 不是全文件和 {r['total_bytes_resolved']} B"
               if r['total_bytes_resolved'] != _b else "")
        print(f"\n{a.mem}: {_b}/{r['cap_bytes']} B ({_b/r['cap_bytes']*100:.0f}%), "
              f"headroom {r['cap_bytes'] - _b} B —— **单函数最大值 ({_n})**{_vs}; {_tail}")
        for _f in _pf["functions"]:
            print(f"    {_f['fn']}: {_f['resolved']} B"
                  + (f" (+{_f['unresolved']} 项未解析)" if _f['unresolved'] else ""))
    else:
        print(f"\n{a.mem}: {r['total_bytes_resolved']}/{r['cap_bytes']} B "
              f"({r['total_bytes_resolved']/r['cap_bytes']*100:.0f}%), "
              f"headroom {r['headroom_bytes']} B, {_tail}")
    for _ob in r.get("off_ub_buffers", []):
        print(f"    [U10] {_ob['buf']} {_ob['bytes']} B 不在 UB ({_ob['storage']}: {_ob['why']}) —— 已剔除")
    if r.get("storage_unknown"):
        print(f"    [U10] 找不到声明, 保守按 UB 计: {', '.join(r['storage_unknown'])}")
    #  ★结论和它的资格一起打印 (2026-09-18 立, 代价是一个被错杀的候选)。
    #    我据本模块报的 "UB 100%, 余量 0, OVER CAP" 毙掉了一个候选并写下"这条路封死",
    #    而本模块的 U5_absolute_peak 在判据表里是 **refuted**。
    #    "下次记得敲 trust gate" 与它要防的遗漏是同一形态, 所以改成结论处自报。
    try:
        from core.trust import verdict_banner
        for _ln in verdict_banner("ub_budget", refusing=bool(r["over_cap"])):
            print(_ln)
    except Exception as _e:
        print(f"★信任等级取不到 ({type(_e).__name__}) —— 按最低信任处理, 不要当默许。")


if __name__ == "__main__":
    sys.exit(main())
