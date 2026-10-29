#!/usr/bin/env bash
set -uo pipefail
DEV="$1"; TILE="$2"; REPS="$3"; OUT="$4"; KNAME="$5"
mkdir -p "$OUT"
MS=/usr/local/Ascend/cann-9.0.0/tools/profiler/bin/msprof
cat > "$OUT/app.sh" <<EOF
#!/usr/bin/env bash
exec python3 $(pwd)/results/fixpipe_probe/driver.py $DEV $TILE $REPS
EOF
chmod +x "$OUT/app.sh"
"$MS" op --application="$OUT/app.sh" --output="$OUT" \
     --kernel-name="$KNAME" --aic-metrics=PipeUtilization > "$OUT/msprof.log" 2>&1
echo "MSPROF_RC=$?"
#  ★F18: msprof 产物 0 个时退出码仍为 0 —— 必须核文件数, 不能看 rc。
echo "ARTIFACT_FILES=$(find "$OUT" -type f | wc -l)"
echo "PIPE_CSV=$(find "$OUT" -name '*PipeUtilization*.csv' | wc -l)"
