#!/usr/bin/env bash
#  两个 git ref 的成对性能对比 —— 交付数字就是用这个脚本跑出来的。
#
#  用法:
#    NPU_DEVICE=1 bash scripts/bench_pairwise.sh 9e3a550 h26-bitdecode
#    NPU_DEVICE=1 SESSIONS=2 bash scripts/bench_pairwise.sh <基线ref> <候选ref>
#
#  协议 (每一条都是为了挡住一类已经实撞过的错):
#   1. **整包切换**, 不是只换 .so。两个 ref 的 Python 侧 (reference.py 的 K_TILE 等)
#      可能不同; 只换 .so 会让内核与 payload 的填充宽度对不上 ——
#      2026-10-28 实撞, 基线对 h26 的全部候选趟直接 TORCH_CHECK 失败。
#      失败形态若是"能跑但慢"而不是崩溃, 就会把一个错的数写进交付文档。
#   2. **每趟一个独立进程**, 每趟重新安装产物并记录 .so 的 sha256。
#      同一进程内连测两个构建做不到 —— .so 一旦 dlopen 就换不掉。
#   3. **臂序 ABBA** 而非 ABAB, 否则臂序偏置与真实效应混叠。
#   4. **至少两个会话**, 两会话之间隔一次别的构建。单次会话内再紧也不足以放行:
#      2026-10-27 有两个候选在单次会话里拿到 -1.9%, 重测全部消失。
#
#  输出: results/bench_<A>_vs_<B>/ 下每个会话一个 .jsonl, 外加一份 summary.txt
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REF_A="${1:?用法: bash scripts/bench_pairwise.sh <refA> <refB>}"
REF_B="${2:?用法: bash scripts/bench_pairwise.sh <refA> <refB>}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
SESSIONS="${SESSIONS:-2}"
TRIALS="${TRIALS:-4}"          # 每个会话每臂的趟数

SITE="$("$PYTHON_BIN" -c 'import vc_attention_ascend,os;print(os.path.dirname(vc_attention_ascend.__file__))' 2>/dev/null || true)"
if [[ -z "$SITE" ]]; then
  echo "先装一次 vc_attention_ascend (bash scripts/run_ascend910b.sh), 本脚本要靠它定位安装目录" >&2
  exit 2
fi

SAFE_A="${REF_A//\//_}"; SAFE_B="${REF_B//\//_}"
OUT="results/bench_${SAFE_A}_vs_${SAFE_B}"
WORK="$OUT/pkgs"
mkdir -p "$WORK"

if [[ -n "$(git status --porcelain -- ascend scripts 2>/dev/null)" ]]; then
  echo "★ ascend/ 或 scripts/ 有未提交改动 —— 切 ref 会把它们带过去, 先提交或 stash" >&2
  exit 2
fi
START_REF="$(git rev-parse --abbrev-ref HEAD)"
[[ "$START_REF" == "HEAD" ]] && START_REF="$(git rev-parse HEAD)"
restore() { git checkout -q "$START_REF" 2>/dev/null || true; }
trap restore EXIT

#  1) 两个 ref 各构建一次, 把**整个安装目录**快照下来
for ref in "$REF_A" "$REF_B"; do
  safe="${ref//\//_}"
  echo "== 构建 $ref =="
  git checkout -q "$ref"
  "$PYTHON_BIN" scripts/check_ascend910b.py --device "$NPU_DEVICE" \
      --output results/bench_env.json >/dev/null
  SOC_VERSION="$("$PYTHON_BIN" -c 'import json;print(json.load(open("results/bench_env.json"))["compiler_soc"])')"
  export SOC_VERSION
  "$PYTHON_BIN" -m pip install ./ascend --no-build-isolation --no-deps -q
  rm -rf "$WORK/$safe"
  cp -a "$SITE" "$WORK/$safe"
  echo "   产物 $(sha256sum "$WORK/$safe/libvc_ascend_ops.so" | cut -c1-16)"
done
restore

#  2) 按会话跑 ABBA。会话之间重建一次 REF_A, 作为"隔一次别的构建"。
for s in $(seq 1 "$SESSIONS"); do
  jsonl="$OUT/session${s}.jsonl"; : > "$jsonl"
  order=(); for _ in $(seq 1 "$TRIALS"); do order+=("$SAFE_A" "$SAFE_B" "$SAFE_B" "$SAFE_A"); done
  #  第二个及以后的会话把臂序翻转, 进一步打散臂序偏置
  if (( s % 2 == 0 )); then order=("${order[@]: -1}" "${order[@]:0:${#order[@]}-1}"); fi
  n=0
  for arm in "${order[@]:0:$((TRIALS*2))}"; do
    n=$((n+1))
    rm -rf "$SITE"; cp -a "$WORK/$arm" "$SITE"
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$NPU_DEVICE}" \
      "$PYTHON_BIN" scripts/bench_one.py --device 0 --label "$arm" >> "$jsonl"
    echo "  会话$s $n/$((TRIALS*2)) 臂 $arm"
  done
done

#  3) 汇总
"$PYTHON_BIN" - "$OUT" "$SAFE_A" "$SAFE_B" <<'PY' | tee "$OUT/summary.txt"
import glob, json, statistics, sys
out, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"=== {a}  ->  {b} ===")
eff = {}
for f in sorted(glob.glob(f"{out}/session*.jsonl")):
    g = {}
    for line in open(f):
        d = json.loads(line); g.setdefault(d["label"], []).append(d)
    if a not in g or b not in g:
        print(f"{f}: 缺臂, 跳过"); continue
    fa = {d["so_sha256_16"] for d in g[a]}; fb = {d["so_sha256_16"] for d in g[b]}
    assert len(fa) == 1 and len(fb) == 1 and fa != fb, f"产物指纹不唯一或两臂相同: {fa} {fb}"
    for key in ("control_m64", "expcast_m64"):
        ma = statistics.median([d[key] for d in g[a]])
        mb = statistics.median([d[key] for d in g[b]])
        e = 100 * (mb / ma - 1)
        eff.setdefault(key, []).append(e)
        print(f"{f.split('/')[-1]:<14} {key:<12} {ma:9.2f} -> {mb:9.2f} ms  ({e:+.2f}%)  "
              f"指纹 {fa.pop() if False else sorted(fa)[0]} / {sorted(fb)[0]}")
print()
for key, es in eff.items():
    med = statistics.median(es); spread = max(es) - min(es)
    ok = len(es) >= 2 and (all(x > 0 for x in es) or all(x < 0 for x in es)) and spread < abs(med) / 2
    verdict = "跨会话复现" if ok else ("只有一个会话, 不可据此放行" if len(es) < 2 else "★跨会话不复现")
    print(f"{key}: 各会话 {['%+.2f%%' % x for x in es]}, 中位 {med:+.2f}%, 极差 {spread:.2f}% -> {verdict}")
PY
echo "明细见 $OUT/"
