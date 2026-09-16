"""从官方 release 合成带噪验证材料（vctk 干净语音 + fsd50k 噪声，按 SNR 网格混合）。
运行环境：WSL 内用 lrac_data 的 venv（含 numpy/scipy/soundfile）
用法：python make_mixtures.py --n-speech 200 --snrs -5,0,5,10,15,20,25
"""
import argparse, csv, glob, os, random
import numpy as np
import soundfile as sf

REL = "/root/lrac_2026/releases/audio"
OUT = "/root/lrac_eval_mix"

def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr          # 单声道

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-speech", type=int, default=200)
    ap.add_argument("--snrs", default="-5,0,5,10,15,20,25")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    snrs = [float(s) for s in args.snrs.split(",")]

    speech = sorted(glob.glob(f"{REL}/vctk/**/*.wav", recursive=True))
    noise  = sorted(glob.glob(f"{REL}/fsd50k/**/*.wav", recursive=True))
    print(f"可用: vctk {len(speech)} 条语音, fsd50k {len(noise)} 条噪声")
    rnd = random.Random(2026)
    rnd.shuffle(speech); rnd.shuffle(noise)
    speech = speech[:args.n_speech]

    os.makedirs(f"{args.out}/mix", exist_ok=True)
    os.makedirs(f"{args.out}/ref", exist_ok=True)
    rows = []
    for i, sp in enumerate(speech):
        s, sr = load(sp)
        if sr != 24000:
            continue
        # 参考（干净）
        ref_name = os.path.basename(sp)
        sf.write(f"{args.out}/ref/{ref_name}", s, 24000)
        for snr in snrs:
            nz_path = noise[(i * len(snrs) + int(snr)) % len(noise)]
            n, nsr = load(nz_path)
            if nsr != 24000:
                continue
            # 噪声循环铺满语音长度
            if len(n) < len(s):
                n = np.tile(n, int(np.ceil(len(s) / len(n))))
            n = n[:len(s)]
            p_s = float(np.mean(s ** 2)) + 1e-12
            p_n = float(np.mean(n ** 2)) + 1e-12
            n = n * np.sqrt(p_s / p_n * 10 ** (-snr / 10))
            mix = s + n
            peak = float(np.max(np.abs(mix))) + 1e-12
            if peak > 0.99:
                mix = mix / peak * 0.99
            tag = f"{os.path.splitext(ref_name)[0]}__{os.path.splitext(os.path.basename(nz_path))[0]}_snr{int(snr)}"
            out = f"{args.out}/mix/{tag}.wav"
            sf.write(out, mix, 24000)
            rows.append({"mix": out, "ref": f"{args.out}/ref/{ref_name}",
                         "speech": sp, "noise": nz_path, "snr_db": snr,
                         "dur_s": round(len(s) / 24000, 3)})
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(speech)} 条语音，生成 {len(rows)} 条混合")

    with open(f"{args.out}/mixtures.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["mix", "ref", "speech", "noise", "snr_db", "dur_s"])
        w.writeheader(); w.writerows(rows)
    total_s = sum(r["dur_s"] for r in rows)
    est_gb = total_s * 96000 * 2 / 1e9   # mix + ref 各一份
    print(f"完成：{len(rows)} 条混合，总时长 {total_s/3600:.1f} 小时，占用约 {est_gb:.2f} GB")

if __name__ == "__main__":
    main()
