"""感知尺度结果分析：ΔScoreQ/ΔUTMOS 按 SNR 分层，与 SI-SDR 结论对照

ScoreQ_ref 是"与参考的距离"，越低越好；UTMOS 越高越好。
"""
import csv
from collections import defaultdict

import numpy as np

CSV = "eval_perceptual_result.csv"

rows = []
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    if r["model"] in ("", "ref"):
        continue
    try:
        r["dsq"] = float(r["sq_enh"]) - float(r["sq_base"])     # <0 = 改善
        r["dut"] = float(r["utmos_enh"]) - float(r["utmos_base"])  # >0 = 改善
        r["sqb"] = float(r["sq_base"]); r["sqe"] = float(r["sq_enh"])
        r["utb"] = float(r["utmos_base"]); r["ute"] = float(r["utmos_enh"])
        r["snr"] = float(r["snr_db"])
        rows.append(r)
    except (ValueError, KeyError):
        continue

# 干净参考天花板
refs = [float(r["utmos_enh"]) for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["model"] == "ref" and r["utmos_enh"]]
if refs:
    print(f"干净参考 UTMOS 天花板: {np.mean(refs):.2f}（{len(refs)} 条唯一参考）\n")

print("=== 总体（2,100 条子集）===")
print("| 模型 | 平均ΔScoreQ↓ | 平均ΔUTMOS↑ | ScoreQ不劣化率(Δ≤0) | UTMOS不劣化率(Δ≥0) | ScoreQ显著劣化率(Δ≥+0.3) |")
print("|---|---|---|---|---|---|")
for mk in ("sepformer", "ulunas", "gtcrn", "lisennet", "metricgan"):
    g = [r for r in rows if r["model"] == mk]
    if not g:
        continue
    dsq = np.array([r["dsq"] for r in g]); dut = np.array([r["dut"] for r in g])
    print(f"| {mk} | {dsq.mean():+.3f} | {dut.mean():+.3f} | "
          f"{float((dsq <= 0).mean())*100:.0f}% | {float((dut >= 0).mean())*100:.0f}% | "
          f"{float((dsq >= 0.3).mean())*100:.1f}% |")

print("\n=== 按输入 SNR 分层（各模型平均ΔUTMOS / ΔScoreQ）===")
snrs = sorted({r["snr"] for r in rows})
models = [m for m in ("sepformer", "ulunas", "gtcrn", "lisennet", "metricgan")
          if any(r["model"] == m for r in rows)]
print("| SNR | " + " | ".join(f"{m} ΔUT/ΔSQ" for m in models) + " |")
print("|---|" + "---|" * len(models))
for s in snrs:
    cells = []
    for m in models:
        g = [r for r in rows if r["model"] == m and r["snr"] == s]
        if g:
            cells.append(f"{np.mean([r['dut'] for r in g]):+.2f} / "
                         f"{np.mean([r['dsq'] for r in g]):+.2f}")
        else:
            cells.append("—")
    print(f"| {int(s):+d} dB | " + " | ".join(cells) + " |")

print("\n=== +25 dB 档不劣化率（感知 vs 信号对照）===")
print("| 模型 | ScoreQ不劣化率 | UTMOS不劣化率 | 绝对分: base→enh (UTMOS) |")
print("|---|---|---|---|")
for m in models:
    g = [r for r in rows if r["model"] == m and r["snr"] == 25.0]
    if g:
        dsq = np.array([r["dsq"] for r in g]); dut = np.array([r["dut"] for r in g])
        print(f"| {m} | {float((dsq <= 0).mean())*100:.0f}% | "
              f"{float((dut >= 0).mean())*100:.0f}% | "
              f"{np.mean([r['utb'] for r in g]):.2f} → {np.mean([r['ute'] for r in g]):.2f} |")

print("\n=== 各 SNR 档 UTMOS 不劣化率（%）===")
print("| SNR | " + " | ".join(models) + " |")
print("|---|" + "---|" * len(models))
for s in snrs:
    cells = []
    for m in models:
        g = [r for r in rows if r["model"] == m and r["snr"] == s]
        cells.append(f"{float((np.array([r['dut'] for r in g]) >= 0).mean())*100:.0f}%" if g else "—")
    print(f"| {int(s):+d} dB | " + " | ".join(cells) + " |")
