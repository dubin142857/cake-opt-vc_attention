#!/usr/bin/env python3
"""把测量记录裁剪成可随包交付的形态: 相对化本机路径 + 把逐文件清单压成汇总指纹。

为什么需要它 (2026-10-29, F175/F179):
  delivery/PERF_COMPARISON.md 引用 results/mindie_laser_quant_compare.json 作为
  对照表的原始记录, 但该文件**未被 git 跟踪** —— 甲方 clone 之后拿不到, 文档里是
  一条死引用。而直接 git add 又越过本仓既有惯例两项: 538 KB (已跟踪的 results/*.json
  最大 92 KB) 且含 40 处本机绝对路径 (已跟踪的是 0 处)。

做两件事, 都**不丢证据**:
  (1) 相对化绝对路径 —— 它们是溯源字段, 不影响复核, 且不该外泄内部目录结构。
  (2) 把 source_manifest.source_files (逐文件 path->sha, 约 415 KB) 压成
      {文件数, 合并指纹} —— 它的证据价值是"编的是哪份源码", 一个覆盖全部
      (路径, sha) 的合并哈希同样能证明, 而且**可被独立重算**。

用法: sanitize_measurement_json.py <in.json> <out.json>
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#  ★替换顺序要从长到短, 否则 /data0/x/repo 会被 /data0 先吃掉。
_SUBS = sorted([(ROOT, "<repo>"), ("/mnt/opbuild", "<scratch>"),
                ("/root", "<home>"), ("/data0", "<data>")],
               key=lambda kv: -len(kv[0]))


def relativize(obj):
    if isinstance(obj, str):
        for a, b in _SUBS:
            obj = obj.replace(a, b)
        return obj
    if isinstance(obj, dict):
        #  ★**键也要变换**。本记录里 loaded_library_sha256 / device_binary_sha256 /
        #   host_library_sha256 三个字典都是**以绝对路径为键**的 —— 第一版只变换值,
        #   于是 7 处路径原样留在输出里。自检当场报出来 (残留 7 处, rc=1)。
        return {relativize(k): relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize(v) for v in obj]
    return obj


def digest_file_manifest(files) -> dict:
    """把 {path: sha} 或 [{path, sha}] 压成 {n, combined}。**可独立重算**:
    对 (路径, sha) 按路径排序后逐行 "path\\0sha\\n" 喂进 sha256。"""
    if isinstance(files, dict):
        pairs = sorted((k, v if isinstance(v, str) else json.dumps(v, sort_keys=True))
                       for k, v in files.items())
    else:
        pairs = sorted((f.get("path", ""), f.get("sha256", f.get("sha", "")))
                       for f in files)
    h = hashlib.sha256()
    for p, s in pairs:
        h.update(p.encode()); h.update(b"\0"); h.update(str(s).encode()); h.update(b"\n")
    return {"n_files": len(pairs), "combined_sha256": h.hexdigest(),
            "how": "对 (路径, sha) 按路径排序后逐行 path\\0sha\\n 喂 sha256; 可独立重算",
            "dropped": "原逐文件清单未随包 (约 415 KB); 需要时按本记录里的 upstream "
                       "提交重新生成 build/mindie_laser_source_manifest.json 即可比对"}


def sanitize(d: dict) -> dict:
    d = json.loads(json.dumps(d))          # 深拷贝, 不改入参
    stack = [d]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k == "source_files" and isinstance(v, (dict, list)) and len(v) > 50:
                    o[k] = digest_file_manifest(v)
                else:
                    stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return relativize(d)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__); return 2
    src, dst = sys.argv[1], sys.argv[2]
    d = json.load(open(src, encoding="utf-8"))
    out = sanitize(d)
    out["_sanitized"] = {
        "by": "scripts/sanitize_measurement_json.py",
        "date": "2026-10-29",
        "what": "本机绝对路径已相对化; 超过 50 项的 source_files 逐文件清单已压成汇总指纹",
        "why": "原件 538 KB / 40 处绝对路径, 越过本仓 results/*.json 的惯例 (<=92 KB, 0 处)",
    }
    txt = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    open(dst, "w", encoding="utf-8").write(txt)
    import re
    left = len(re.findall(r"/data0/|/mnt/|/root/", txt))
    print(f"  {os.path.getsize(src):>8d} -> {len(txt.encode()):>7d} 字节")
    print(f"  残留本机绝对路径: {left} 处")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
