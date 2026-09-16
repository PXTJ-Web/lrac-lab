"""干净输入下的增强自失真测试：直接对应比赛的 clean 条件（MUSHRA 干净语音质量）
输入：语料里的 1,500 条干净参考；测量 SI-SDR(clean, enhanced) —— 越负说明破坏越严重
"""
import csv, glob, os
import numpy as np, soundfile as sf, torch, torchaudio
from enhance_module import create_enhancer
from evaluate import si_sdr, lsd

SR16 = 16000
def to16(x):
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, 24000, SR16)[0].numpy()

refs = sorted(glob.glob(r"C:\lrac_eval_mix\ref\*.wav"))
print(f"干净语音 {len(refs)} 条")
out = open("clean_input_result.csv", "w", newline="", encoding="utf-8")
w = csv.writer(out); w.writerow(["model", "file", "si_sdr_clean_vs_enh", "lsd_clean_vs_enh", "dur_s"])
for mk in ("ulunas", "gtcrn", "lisennet", "metricgan"):
    enh = create_enhancer(mk)
    vals, vals_lsd = [], []
    for p in refs:
        x, _ = sf.read(p, dtype="float32")
        e = enh.enhance(x, 24000)
        r16 = to16(x); e16 = to16(e)
        n = min(len(r16), len(e16))
        s = si_sdr(r16[:n], e16[:n]); l = lsd(r16[:n], e16[:n])
        vals.append(s); vals_lsd.append(l)
        w.writerow([mk, os.path.basename(p), round(s, 4), round(l, 4), round(len(x)/24000, 3)])
    a, al = np.array(vals), np.array(vals_lsd)
    print(f"{mk:10s} SI-SDR(干净,增强后) 均值 {a.mean():+.2f} dB | 中位 {np.median(a):+.2f} | "
          f"<-1dB 比例 {float((a<-1).mean())*100:.0f}% | <-3dB {float((a<-3).mean())*100:.0f}% | LSD {al.mean():.2f}")
out.close()
