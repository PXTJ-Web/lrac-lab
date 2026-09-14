"""逐样本 hold 率分析：回答"能否 hold 住绝大多数测试集数据"。

对每个候选系统，逐条计算增强前后的指标差（增强后 − 不增强），统计：
  - 不劣化率（ΔSI-SDR ≥ 0）      —— 系统未使数据变差
  - 有效增强率（ΔSI-SDR ≥ +1 dB） —— 系统带来实质收益
  - 失效/劣化率（ΔSI-SDR < −1 dB）—— 系统使数据变差
  - 显著劣化率（ΔSI-SDR < −3 dB）
同时统计 PESQ 口径的同类比例。

用法：python hold_analysis.py [--models gtcrn,ulunas]
产出：hold_result.md / hold_result.json / hold_perfile.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from enhance_module import create_enhancer
from evaluate import pesq_score, si_sdr

ROOT = os.path.join("..", "official-testset", "LRAC-2025-test-data", "open-test-set")
CONDITIONS = [("track_1", "noise"), ("track_1", "reverb"),
              ("track_2", "noise"), ("track_2", "reverb")]
IN_DIR = {"noise": "noisy", "reverb": "reverb"}
REF_DIR = {"noise": "reference_noisy", "reverb": "reference_reverb"}


def to16k(x, sr):
    if sr == 16000:
        return x
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, sr, 16000)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gtcrn,ulunas")
    args = ap.parse_args()
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []       # (track, cond, model, si_sdr_base, si_sdr_enh, pesq_base, pesq_enh, dur)
    for track, cond in CONDITIONS:
        indir = os.path.join(ROOT, track, IN_DIR[cond])
        refdir = os.path.join(ROOT, track, REF_DIR[cond])
        files = sorted(f for f in os.listdir(indir) if f.endswith(".wav"))
        print(f"\n=== {track}/{cond}：{len(files)} 条 ===", flush=True)

        # 先算基线与参考（两个模型共用）
        cache = []
        for f in files:
            inp, s1 = sf.read(os.path.join(indir, f), dtype="float32")
            ref, s2 = sf.read(os.path.join(refdir, f), dtype="float32")
            r16, i16 = to16k(ref, s2), to16k(inp, s1)
            n = min(len(r16), len(i16))
            cache.append((f, inp, s1, r16, si_sdr(r16[:n], i16[:n]), pesq_score(r16[:n], i16[:n])))
        print("  基线完成", flush=True)

        for mk in model_keys:
            enh = create_enhancer(mk)
            t0 = time.time()
            for i, (f, inp, s1, r16, base_sdr, base_pesq) in enumerate(cache, 1):
                est = to16k(enh.enhance(inp, s1), enh.out_sr)
                n = min(len(r16), len(est))
                rows.append((track, cond, mk, base_sdr, si_sdr(r16[:n], est[:n]),
                             base_pesq, pesq_score(r16[:n], est[:n]), n / 16000))
                if i % 200 == 0:
                    print(f"  [{mk}] {i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
            print(f"  [{mk}] 完成，用时 {time.time()-t0:.0f}s", flush=True)

    # ---- 写逐文件 CSV ----
    with open("hold_perfile.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["track", "cond", "model", "si_sdr_base", "si_sdr_enh",
                    "pesq_base", "pesq_enh", "dur_s"])
        w.writerows(rows)

    # ---- 统计 hold 率 ----
    def stat(rs):
        d = [r[4] - r[3] for r in rs]
        d = np.array(d, dtype=float)
        dp = np.array([(r[6] - r[5]) for r in rs if r[5] is not None and r[6] is not None],
                      dtype=float)
        out = {
            "n": len(d),
            "sdr_mean_base": float(np.mean([r[3] for r in rs])),
            "sdr_mean_enh": float(np.mean([r[4] for r in rs])),
            "sdr_delta": float(d.mean()),
            "not_worse_0": float((d >= 0).mean()),
            "gain_1": float((d >= 1).mean()),
            "gain_3": float((d >= 3).mean()),
            "degrade_1": float((d < -1).mean()),
            "degrade_3": float((d < -3).mean()),
        }
        if len(dp):
            out.update({"pesq_delta": float(dp.mean()),
                        "pesq_not_worse": float((dp >= 0).mean()),
                        "pesq_gain_025": float((dp >= 0.25).mean()),
                        "pesq_degrade_025": float((dp < -0.25).mean())})
        return out

    groups = {}
    for track, cond in CONDITIONS:
        for mk in model_keys:
            rs = [r for r in rows if r[0] == track and r[1] == cond and r[2] == mk]
            if rs:
                groups[(track, cond, mk)] = stat(rs)
    for mk in model_keys:
        rs = [r for r in rows if r[2] == mk]
        groups[("全部", "全部", mk)] = stat(rs)
        for cond in ("noise", "reverb"):
            rs = [r for r in rows if r[2] == mk and r[1] == cond]
            if rs:
                groups[("两个 track", cond, mk)] = stat(rs)

    with open("hold_result.json", "w", encoding="utf-8") as fp:
        json.dump({f"{a}|{b}|{c}": v for (a, b, c), v in groups.items()},
                  fp, ensure_ascii=False, indent=2)

    lines = ["# hold 率分析（逐样本）", "",
             "ΔSI-SDR = 增强后 − 不增强（同一参考），逐条计算后统计比例。", "",
             "| 范围 | 系统 | 条数 | 平均ΔSI-SDR | 不劣化率(Δ≥0) | 有效增强率(Δ≥+1dB) | 显著劣化率(Δ<−3dB) | 平均ΔPESQ | PESQ不劣化率 |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = [(t, c) for t, c in CONDITIONS] + [("两个 track", "noise"), ("两个 track", "reverb"), ("全部", "全部")]
    for key in order:
        for mk in model_keys:
            g = groups.get((key[0], key[1], mk))
            if not g:
                continue
            lines.append(
                f"| {key[0]}/{key[1]} | {mk} | {g['n']} | {g['sdr_delta']:+.2f} dB | "
                f"**{g['not_worse_0']*100:.1f}%** | {g['gain_1']*100:.1f}% | "
                f"{g['degrade_3']*100:.1f}% | {g.get('pesq_delta', float('nan')):+.3f} | "
                f"{g.get('pesq_not_worse', float('nan'))*100:.1f}% |")
    with open("hold_result.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print("\n".join(lines))
    print("\n已生成 hold_result.md / hold_result.json / hold_perfile.csv")


if __name__ == "__main__":
    main()
