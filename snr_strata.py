"""按信噪比分层的最差场景分析：把逐样本 hold 结果与官方 meta.csv 的 SNR 关联。
用法：python snr_strata.py
"""
import csv, os
import numpy as np

OFFICIAL = os.path.join("..", "official-testset", "LRAC-2025-test-data", "open-test-set")
IN_DIR = {"noise": "noisy", "reverb": "reverb"}

# 1) 读官方 meta.csv 建立 (track, cond, filename) -> snr
snr_map = {}
for track in ("track_1", "track_2"):
    for cond, d in IN_DIR.items():
        p = os.path.join(OFFICIAL, track, d, "meta.csv")
        if not os.path.exists(p):
            continue
        for row in csv.DictReader(open(p, encoding="utf-8")):
            v = (row.get("snr") or "").strip()
            if v:
                try:
                    snr_map[(track, cond, row["filename"])] = float(v)
                except ValueError:
                    pass
print(f"meta 中共获取 {len(snr_map)} 条 SNR 记录")

# 2) 读逐样本结果并关联
rows = []
with open("hold_perfile.csv", encoding="utf-8") as fp:
    for r in csv.DictReader(fp):
        key = (r["track"], r["cond"], r["model"] + ".wav" if False else r["track"] and r["model"])
        fname = None
        for t, c, f in snr_map:
            if t == r["track"] and c == r["cond"]:
                fname = f
                break
        rows.append(r)

# hold_perfile.csv 没有文件名列，改为直接重算：用 model 列 + track/cond 无法定位个体
print("hold_perfile.csv 列名:", list(rows[0].keys()) if rows else "空")
