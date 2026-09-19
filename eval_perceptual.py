"""感知指标评测（ScoreQ_ref + UTMOS）——回答"最差场景在人耳尺度上是否复现"

范围：与五模型对比相同的 2,100 条分层子集（直接取 eval_mixtures_result.csv 中
metricgan 的 mix 列表，保证与既有口径完全一致）。

记分对象：
  - base（带噪输入，不增强）——所有模型共用，只算一次；
  - 5 个模型的增强输出；
  - 干净参考的 UTMOS 作"天花板"参照（ScoreQ 对自身是退化距离，不测）。

用法：
  python eval_perceptual.py                 # 全部模型
  python eval_perceptual.py --models gtcrn  # 指定模型（base 恒算）
"""
import argparse
import csv
import os
import time

# 网络环境（镜像 + 代理），已缓存后无需联网
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

import numpy as np
import soundfile as sf
import torch
import torchaudio

from enhance_module import create_enhancer

MIX_ROOT = "C:/lrac_eval_mix"
REF_CSV = "eval_mixtures_result.csv"        # 从这里取 2,100 条子集（metricgan 行）
OUT_CSV = "eval_perceptual_result.csv"
SR16 = 16000


def fix_path(p):
    if p.startswith("/root/lrac_eval_mix/"):
        return MIX_ROOT + "/" + p[len("/root/lrac_eval_mix/"):]
    return p


def to16(x, sr):
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    if sr != SR16:
        t = torchaudio.functional.resample(t, sr, SR16)
    return t


def load_subset():
    """2,100 条子集：metricgan 跑过的 mix 列表（唯一化）。"""
    seen = {}
    for r in csv.DictReader(open(REF_CSV, encoding="utf-8")):
        if r["model"] == "metricgan" and r["mix"] not in seen:
            seen[r["mix"]] = (r["mix"], r["ref"], float(r["snr_db"]))
    return list(seen.values())


def load_done():
    done = set()
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV, encoding="utf-8")):
            done.add((r["model"], r["mix"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gtcrn,ulunas,lisennet,metricgan,sepformer")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {dev}（{torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'}）", flush=True)

    print("加载 UTMOS ...", flush=True)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(dev).eval()
    print("加载 ScoreQ_ref ...", flush=True)
    from scoreq_pytorch import SCOREQScoreTorch
    scoreq = SCOREQScoreTorch(data_domain="natural", mode="ref", device=dev)

    def score_pair(est16, ref16):
        n = min(est16.shape[-1], ref16.shape[-1])
        e, r = est16[:, :n].to(dev), ref16[:, :n].to(dev)
        with torch.no_grad():
            sq = scoreq.score(e, r).item()
            ut = utmos(e, SR16).item()
        return sq, ut

    subset = load_subset()
    print(f"子集 {len(subset)} 条 | 模型 {models}\n", flush=True)

    done = load_done()
    fields = ["model", "mix", "snr_db", "sq_base", "sq_enh", "utmos_base", "utmos_enh"]
    new_file = not os.path.exists(OUT_CSV)
    fp = open(OUT_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fp, fieldnames=fields)
    if new_file:
        w.writeheader()

    base_cache = {}   # mix -> (sq, ut)

    def get_base(mix_path, ref16):
        if mix_path not in base_cache:
            mix, sr = sf.read(fix_path(mix_path), dtype="float32")
            base_cache[mix_path] = score_pair(to16(mix, sr), ref16)
        return base_cache[mix_path]

    t_all = time.time()
    for mk in models:
        enh = create_enhancer(mk).to_device(dev)
        todo = [s for s in subset if (mk, s[0]) not in done]
        print(f"=== {mk}: 待处理 {len(todo)} 条 ===", flush=True)
        t0 = time.time()
        for i, (mix_path, ref_path, snr) in enumerate(todo, 1):
            ref, _ = sf.read(fix_path(ref_path), dtype="float32")
            ref16 = to16(ref, SR16)
            sq_b, ut_b = get_base(mix_path, ref16)
            mix, sr = sf.read(fix_path(mix_path), dtype="float32")
            est = enh.enhance(mix, sr)
            sq_e, ut_e = score_pair(to16(est, enh.out_sr), ref16)
            w.writerow({"model": mk, "mix": mix_path, "snr_db": snr,
                        "sq_base": round(sq_b, 4), "sq_enh": round(sq_e, 4),
                        "utmos_base": round(ut_b, 4), "utmos_enh": round(ut_e, 4)})
            if i % 100 == 0:
                fp.flush()
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  用时 {el/60:.1f} 分  预计还需 {el/i*(len(todo)-i)/60:.1f} 分",
                      flush=True)
        fp.flush()
        print(f"=== {mk} 完成，用时 {(time.time()-t0)/60:.1f} 分 ===\n", flush=True)

    # 干净参考天花板（UTMOS），按唯一 ref 去重
    refs = sorted({(s[1],) for s in subset})
    todo_refs = [r[0] for r in refs if ("ref", r[0]) not in done]
    if todo_refs:
        print(f"=== 干净参考天花板: {len(todo_refs)} 条 ===", flush=True)
        for j, ref_path in enumerate(todo_refs, 1):
            ref, _ = sf.read(fix_path(ref_path), dtype="float32")
            with torch.no_grad():
                ut = utmos(to16(ref, SR16).to(dev), SR16).item()
            w.writerow({"model": "ref", "mix": ref_path, "snr_db": "",
                        "sq_base": "", "sq_enh": "",
                        "utmos_base": "", "utmos_enh": round(ut, 4)})
            if j % 100 == 0:
                fp.flush()
        fp.flush()
    fp.close()
    print(f"\n全部完成，总用时 {(time.time()-t_all)/60:.1f} 分。已写出: {OUT_CSV}")


if __name__ == "__main__":
    main()
