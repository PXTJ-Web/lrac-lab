"""感知指标冒烟测试：ScoreQ_ref + UTMOS 方向验证 + 增强模块 GPU 兼容性检查"""
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR16 = 16000
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to16(x, sr):
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, sr, SR16)


print("== 加载 UTMOS（torch.hub，首次下载 ~1.2GB）==", flush=True)
t0 = time.time()
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                       trust_repo=True).to(DEV).eval()
print(f"UTMOS 就绪 {time.time()-t0:.0f}s", flush=True)

print("== 加载 ScoreQ_ref（HF Blinorot/SCOREQ-PyTorch）==", flush=True)
t0 = time.time()
from scoreq_pytorch import SCOREQScoreTorch
scoreq = SCOREQScoreTorch(data_domain="natural", mode="ref", device=DEV)
print(f"ScoreQ 就绪 {time.time()-t0:.0f}s", flush=True)


def scores(est16, ref16):
    with torch.no_grad():
        ut = utmos(est16.to(DEV), SR16).item()
        sq = scoreq.score(est16.to(DEV), ref16.to(DEV)).item()
    return sq, ut


if __name__ == "__main__":
    noisy, sr = sf.read("test_noisy.wav", dtype="float32")
    clean, _ = sf.read("test_clean.wav", dtype="float32")
    n16, c16 = to16(noisy, sr), to16(clean, sr)

    sq_b, ut_b = scores(n16, c16)
    sq_c, ut_c = scores(c16, c16)
    print(f"\n干净参考自评:      ScoreQ={sq_c:.3f}  UTMOS={ut_c:.3f}")
    print(f"带噪输入（基线）:  ScoreQ={sq_b:.3f}  UTMOS={ut_b:.3f}")

    from enhance_module import create_enhancer
    for mk in ("gtcrn", "ulunas", "sepformer"):
        enh = create_enhancer(mk).to_device(DEV)
        t0 = time.time()
        est = enh.enhance(noisy, sr)
        dt = time.time() - t0
        sq_e, ut_e = scores(to16(est, enh.out_sr), c16)
        print(f"{mk:9s} 增强后:  ScoreQ={sq_e:.3f}  UTMOS={ut_e:.3f}   ({dt:.2f}s/条, GPU)")
