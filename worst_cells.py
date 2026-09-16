"""最差场景交叉分析：SNR × 噪声类型 的具体组合，找出最差的格子。"""
import csv, os
from collections import defaultdict
import numpy as np

GT_DIR = "C:/lrac_stage/fsd50k_gt"
labels = {}
for name in ("dev.csv", "eval.csv"):
    p = os.path.join(GT_DIR, name)
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            ls = [x for x in (r.get("labels") or "").split(",") if x]
            if ls: labels[r["fname"].strip()] = ls[-1]

rows = []
for r in csv.DictReader(open("eval_mixtures_result.csv", encoding="utf-8")):
    if not r.get("si_sdr_enh"): continue
    cid = r["noise_id"].split("_",1)[-1].split("-")[0]
    r["cat"] = labels.get(cid, "未知"); r["snr"] = float(r["snr_db"])
    r["d"] = float(r["si_sdr_enh"]) - float(r["si_sdr_base"])
    rows.append(r)

for mk in ("ulunas", "gtcrn"):
    sub = [r for r in rows if r["model"] == mk]
    g = defaultdict(list)
    for r in sub: g[(r["snr"], r["cat"])].append(r["d"])
    cells = []
    for (snr, cat), ds in g.items():
        if len(ds) < 25: continue
        d = np.array(ds)
        cells.append((float((d>=0).mean()), d.mean(), snr, cat, len(d), float((d<-3).mean())))
    cells.sort()
    print(f"\n=== {mk}: 最差 12 个 (SNR × 噪声类型) 组合 ===")
    print("| SNR | 噪声类型 | 条数 | 平均ΔSI-SDR | 不劣化率 | 显著劣化率 |")
    print("|---|---|---|---|---|---|")
    for nw, mean, snr, cat, n, deg in cells[:12]:
        print(f"| {int(snr):+d} dB | {cat} | {n} | {mean:+.2f} | {nw*100:.0f}% | {deg*100:.0f}% |")
