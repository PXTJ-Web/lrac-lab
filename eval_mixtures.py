"""大规模混合语料评测（老师任务：基于 ~10G 数据集验证增强效果 + 最差场景归因）

输入：make_mixtures.py 产出的 mixtures.csv（每条含 mix/ref/speech/noise/snr_db）
输出：逐文件结果 CSV（可增量、可续跑）→ 供分层分析（SNR 档 / 噪声类型）

用法示例：
  python eval_mixtures.py --models gtcrn --metrics fast
  python eval_mixtures.py --models ulunas,lisennet --metrics fast
  python eval_mixtures.py --models gtcrn --metrics full --limit 2000
"""
import argparse
import csv
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from enhance_module import create_enhancer, REGISTRY
from evaluate import lsd, pesq_score, si_sdr, stoi

MIX_ROOT = "C:/lrac_eval_mix"
MIX_CSV = os.path.join(MIX_ROOT, "mixtures.csv")
OUT_CSV = "eval_mixtures_result.csv"
SR16 = 16000


def fix_path(p):
    """mixtures.csv 里记的是 WSL 路径，映射到 Windows"""
    if p.startswith("/root/lrac_eval_mix/"):
        return MIX_ROOT + "/" + p[len("/root/lrac_eval_mix/"):]
    return p


def to16(x):
    if len(x) == 0:
        return x
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, 24000, SR16)[0].numpy()


def metrics(ref16, est16, full):
    n = min(len(ref16), len(est16))
    r, e = ref16[:n], est16[:n]
    out = {"si_sdr": si_sdr(r, e), "lsd": lsd(r, e)}
    if full:
        out["stoi"] = stoi(r, e)
        out["pesq"] = pesq_score(r, e)
    return out


def load_done(out_csv=OUT_CSV):
    """已完成的行（支持续跑）"""
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                done.add((r["model"], r["mix"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gtcrn")
    ap.add_argument("--metrics", choices=["fast", "full"], default="fast")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--stride", type=int, default=1, help="抽样步长")
    ap.add_argument("--out", default=OUT_CSV, help="结果 CSV 路径")
    args = ap.parse_args()
    full = args.metrics == "full"
    out_csv = args.out
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = list(csv.DictReader(open(MIX_CSV, encoding="utf-8")))
    if args.stride > 1:
        rows = rows[::args.stride]
    if args.limit:
        rows = rows[:args.limit]
    print(f"混合语料 {len(rows)} 条 | 模型 {models} | 指标 {args.metrics}")

    done = load_done(out_csv)
    fields = ["model", "mix", "ref", "snr_db", "noise_id", "speech_id", "dur_s",
              "si_sdr_base", "si_sdr_enh", "lsd_base", "lsd_enh",
              "stoi_base", "stoi_enh", "pesq_base", "pesq_enh"]
    new_file = not os.path.exists(out_csv)
    fp = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fp, fieldnames=fields)
    if new_file:
        w.writeheader()

    for mk in models:
        enh = create_enhancer(mk)
        todo = [r for r in rows if (mk, r["mix"]) not in done]
        print(f"\n=== {mk}: 待处理 {len(todo)} 条 ===", flush=True)
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            mix, _ = sf.read(fix_path(r["mix"]), dtype="float32")
            ref, _ = sf.read(fix_path(r["ref"]), dtype="float32")
            m16, r16 = to16(mix), to16(ref)
            m_m = metrics(r16, m16, full)          # 不增强基线
            est = enh.enhance(mix, 24000)          # 内部 16k 处理 → 输出 24k
            e_m = metrics(r16, to16(est), full)
            row = {"model": mk, "mix": r["mix"], "ref": r["ref"],
                   "snr_db": r["snr_db"],
                   "noise_id": os.path.basename(r["noise"]).replace(".wav", ""),
                   "speech_id": os.path.basename(r["speech"]).replace(".wav", ""),
                   "dur_s": r["dur_s"],
                   "si_sdr_base": round(m_m["si_sdr"], 4),
                   "si_sdr_enh": round(e_m["si_sdr"], 4),
                   "lsd_base": round(m_m["lsd"], 4),
                   "lsd_enh": round(e_m["lsd"], 4),
                   "stoi_base": round(m_m["stoi"], 4) if m_m.get("stoi") is not None else "",
                   "stoi_enh": round(e_m["stoi"], 4) if e_m.get("stoi") is not None else "",
                   "pesq_base": round(m_m["pesq"], 4) if m_m.get("pesq") is not None else "",
                   "pesq_enh": round(e_m["pesq"], 4) if e_m.get("pesq") is not None else ""}
            w.writerow(row)
            if i % 100 == 0:
                fp.flush()
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  用时 {el/60:.1f} 分  预计还需 {el/i*(len(todo)-i)/60:.1f} 分",
                      flush=True)
        fp.flush()
        print(f"=== {mk} 完成，用时 {(time.time()-t0)/60:.1f} 分 ===", flush=True)
    fp.close()
    print("\n已写出:", OUT_CSV)


if __name__ == "__main__":
    main()
