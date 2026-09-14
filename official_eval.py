"""LRAC 官方 open-test-set 评测：五模型 × (track × 条件) 四项客观指标。

用法：
  python official_eval.py --limit 5        # 冒烟测试（每条件前 5 条）
  python official_eval.py                  # 全量（约 1100 条 × 5 模型）
  python official_eval.py --models gtcrn   # 只跑指定模型

指标在 16kHz 上计算（PESQ 只支持 16k/8k）；模型接口对外输出 24k，内部自动重采样。
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from enhance_module import REGISTRY, create_enhancer
from evaluate import lsd, pesq_score, si_sdr, stoi

ROOT = os.path.join("..", "official-testset", "LRAC-2025-test-data", "open-test-set")
CONDITIONS = [("track_1", "noise"), ("track_1", "reverb"),
              ("track_2", "noise"), ("track_2", "reverb")]
IN_DIR = {"noise": "noisy", "reverb": "reverb"}
REF_DIR = {"noise": "reference_noisy", "reverb": "reference_reverb"}
SR_METRIC = 16000


def to16k(x, sr):
    if sr == SR_METRIC:
        return x
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, sr, SR_METRIC)[0].numpy()


def metrics(ref, est):
    n = min(len(ref), len(est))
    r, e = ref[:n], est[:n]
    out = {"SI-SDR": si_sdr(r, e), "LSD": lsd(r, e)}
    s = stoi(r, e)
    if s is not None:
        out["STOI"] = s
    p = pesq_score(r, e)
    if p is not None:
        out["PESQ"] = p
    return out


def mean_of(rows):
    keys = [k for k in ("PESQ", "STOI", "SI-SDR", "LSD")
            if any(k in r for r in rows)]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每条件最多处理多少条")
    ap.add_argument("--models", default=",".join(REGISTRY))
    args = ap.parse_args()
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    store = {}          # store[(track,cond,model)] = [metrics...]
    counts = {}

    for track, cond in CONDITIONS:
        indir = os.path.join(ROOT, track, IN_DIR[cond])
        refdir = os.path.join(ROOT, track, REF_DIR[cond])
        files = sorted(f for f in os.listdir(indir) if f.endswith(".wav"))
        if args.limit:
            files = files[:args.limit]
        counts[f"{track}/{cond}"] = len(files)
        print(f"\n=== {track}/{cond}：{len(files)} 条 ===", flush=True)

        # 不增强基线
        for f in files:
            inp, sr1 = sf.read(os.path.join(indir, f), dtype="float32")
            ref, sr2 = sf.read(os.path.join(refdir, f), dtype="float32")
            store.setdefault((track, cond, "不增强"), []).append(
                metrics(to16k(ref, sr2), to16k(inp, sr1)))

        for mk in model_keys:
            enh = create_enhancer(mk)
            t0 = time.time()
            for i, f in enumerate(files, 1):
                inp, sr1 = sf.read(os.path.join(indir, f), dtype="float32")
                ref, sr2 = sf.read(os.path.join(refdir, f), dtype="float32")
                est = enh.enhance(inp, sr1)          # 内部 16k 处理，对外 24k
                store.setdefault((track, cond, mk), []).append(
                    metrics(to16k(ref, sr2), to16k(est, enh.out_sr)))
                if i % 50 == 0:
                    print(f"  [{enh.name}] {i}/{len(files)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            print(f"  [{enh.name}] 完成，用时 {time.time()-t0:.0f}s", flush=True)

    # ---- 输出表格 ----
    lines = ["# LRAC 官方 open-test-set 评测结果", "",
             f"数据：官方 2025 open test set（24kHz，英语），"
             f"条件：{', '.join(f'{k}({n}条)' for k, n in counts.items())}", "",
             "指标在 16kHz 上计算；每格为该条件下全部样本的均值。", ""]
    metric_keys = ["PESQ", "STOI", "SI-SDR", "LSD"]
    for track, cond in CONDITIONS:
        lines += [f"## {track} / {cond}", "",
                  "| 系统 | " + " | ".join(metric_keys) + " |",
                  "|---" * (len(metric_keys) + 1) + "|"]
        for mk in ["不增强"] + model_keys:
            rows = store.get((track, cond, mk))
            if not rows:
                continue
            m = mean_of(rows)
            name = "不增强（基线）" if mk == "不增强" else REGISTRY[mk].name
            cells = [f"{m[k]:.3f}" if k in m else "—" for k in metric_keys]
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
        lines.append("")

    with open("official_result.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    with open("official_result.json", "w", encoding="utf-8") as fp:
        json.dump({f"{t}|{c}|{m}": mean_of(r)
                   for (t, c, m), r in store.items()},
                  fp, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print("\n已生成 official_result.md 和 official_result.json")


if __name__ == "__main__":
    main()
