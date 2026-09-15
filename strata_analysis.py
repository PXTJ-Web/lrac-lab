"""最差场景分层分析：按"输入保真度"(不增强时的 SI-SDR) 分档，统计各档的 hold 率。
用法：python strata_analysis.py
"""
import csv
import numpy as np

BANDS = [(-100, 0, "<0 dB"), (0, 5, "0-5"), (5, 10, "5-10"),
         (10, 15, "10-15"), (15, 20, "15-20"), (20, 25, "20-25"), (25, 100, ">25")]

rows = list(csv.DictReader(open("hold_perfile.csv", encoding="utf-8")))
for r in rows:
    r["base"] = float(r["si_sdr_base"])
    r["delta"] = float(r["si_sdr_enh"]) - float(r["si_sdr_base"])
    r["dpesq"] = (float(r["pesq_enh"]) - float(r["pesq_base"])
                  if r["pesq_base"] and r["pesq_enh"] else None)

print("=" * 78)
print("最差场景分层：按输入保真度（不增强时 SI-SDR）分档")
print("=" * 78)
out = []
for model in ("ulunas", "gtcrn"):
    for cond in ("noise", "reverb"):
        sub = [r for r in rows if r["model"] == model and r["cond"] == cond]
        if not sub:
            continue
        out.append(f"\n### {model} / {cond}（共 {len(sub)} 条）")
        out.append("| 输入 SI-SDR 档 | 条数 | 占比 | 平均ΔSI-SDR | 不劣化率 | 有效增强率(≥+1) | 显著劣化率(<−3) | 平均ΔPESQ |")
        out.append("|---|---|---|---|---|---|---|---|")
        for lo, hi, name in BANDS:
            g = [r for r in sub if lo <= r["base"] < hi]
            if not g:
                continue
            d = np.array([r["delta"] for r in g])
            dp = np.array([r["dpesq"] for r in g if r["dpesq"] is not None])
            out.append(
                f"| {name} | {len(g)} | {len(g)/len(sub)*100:.0f}% | {d.mean():+.2f} | "
                f"{float((d>=0).mean())*100:.0f}% | {float((d>=1).mean())*100:.0f}% | "
                f"{float((d<-3).mean())*100:.0f}% | "
                f"{dp.mean():+.2f} |" if len(dp) else
                f"| {name} | {len(g)} | {len(g)/len(sub)*100:.0f}% | {d.mean():+.2f} | "
                f"{float((d>=0).mean())*100:.0f}% | {float((d>=1).mean())*100:.0f}% | "
                f"{float((d<-3).mean())*100:.0f}% | — |")
text = "\n".join(out)
print(text)
open("strata_result.md", "w", encoding="utf-8").write(
    "# 最差场景分层分析（按输入保真度分档）\n\n"
    "输入保真度 = 不增强时增强前后对比的 SI-SDR（相对于干净参考），可视为输入质量的代理指标。\n"
    + text + "\n")
print("\n已生成 strata_result.md")
