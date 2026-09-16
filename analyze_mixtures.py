"""大规模混合语料的分层归因分析（老师任务：最差场景及其原因 + 覆盖能力）

输入：eval_mixtures.py 产出的逐文件结果 + FSD50K 官方标签
输出：按 SNR 档 / 按噪声类型 的分层 hold 率表（markdown）

用法：python analyze_mixtures.py
"""
import csv
import os
from collections import defaultdict

import numpy as np

RES = "eval_mixtures_result.csv"
GT_DIR = "C:/lrac_stage/fsd50k_gt"
OUT = "mixture_analysis.md"


def load_labels():
    """FSD50K clip 编号 -> 顶层类别（labels 的最后一个，即最宽泛的类别）"""
    m = {}
    for name in ("dev.csv", "eval.csv"):
        p = os.path.join(GT_DIR, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                labels = [x for x in (r.get("labels") or "").split(",") if x]
                if labels:
                    m[r["fname"].strip()] = labels[-1]
    return m


def clip_id(noise_id):
    """fsd50k_317308-074fcb... -> 317308"""
    s = noise_id.split("_", 1)[-1]
    return s.split("-")[0]


def band(snr):
    return f"{int(snr):+d} dB"


def summarize(rows, key_fn, title, lines):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| 分组 | 条数 | 平均ΔSI-SDR | 不劣化率 | 有效增强率(≥+1) | 显著劣化率(<−3) | 平均ΔLSD |")
    lines.append("|---|---|---|---|---|---|---|")
    keys = sorted(groups, key=lambda k: (isinstance(k, str), k))
    for k in keys:
        g = groups[k]
        d = np.array([float(x["si_sdr_enh"]) - float(x["si_sdr_base"]) for x in g])
        dl = np.array([float(x["lsd_enh"]) - float(x["lsd_base"]) for x in g])
        if len(d) < 20:
            continue
        lines.append(
            f"| {k} | {len(d)} | {d.mean():+.2f} | {float((d>=0).mean())*100:.0f}% | "
            f"{float((d>=1).mean())*100:.0f}% | {float((d<-3).mean())*100:.0f}% | {dl.mean():+.2f} |")
    lines.append("")


def main():
    if not os.path.exists(RES):
        raise SystemExit(f"缺少 {RES}（先跑 eval_mixtures.py）")
    labels = load_labels()
    print(f"标签表: {len(labels)} 个 clip")
    rows = []
    missing = 0
    with open(RES, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if not r.get("si_sdr_enh"):
                continue
            cid = clip_id(r["noise_id"])
            cat = labels.get(cid)
            if cat is None:
                missing += 1
                cat = "未知"
            r["cat"] = cat
            r["snr"] = float(r["snr_db"])
            rows.append(r)
    print(f"结果行: {len(rows)}（标签缺失 {missing}）")

    lines = ["# 大规模混合语料分层分析", "",
             f"数据：官方 release 合成的带噪材料（vctk 干净语音 + fsd50k 噪声，7 档 SNR）",
             f"结果行数：{len(rows)}", ""]
    models = sorted({r["model"] for r in rows})
    for mk in models:
        sub = [r for r in rows if r["model"] == mk]
        lines.append(f"## {mk}（{len(sub)} 条）")
        lines.append("")
        summarize(sub, lambda r: band(r["snr"]), "按输入 SNR 分层", lines)
        summarize(sub, lambda r: r["cat"], "按噪声类型分层（Top 类别）", lines)
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\n已生成 {OUT}")


if __name__ == "__main__":
    main()
