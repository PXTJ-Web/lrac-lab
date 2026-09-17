"""混响去除效果验证：增强输出与'干参考'和'输入本身'分别对比。
   如果增强输出比输入更接近干参考 -> 模型在部分去混响（良性）；
   如果既不像干参考也不像输入 -> 引入了伪影（恶性）。"""
import os
import numpy as np, soundfile as sf, torch, torchaudio
from enhance_module import create_enhancer
from evaluate import lsd, si_sdr

SR16 = 16000
ROOT = os.path.join("..", "official-testset", "LRAC-2025-test-data", "open-test-set")
REV = os.path.join(ROOT, "track_1", "reverb")
CLN = os.path.join(ROOT, "track_1", "clean")
N = 60

def to16(x):
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, 24000, SR16)[0].numpy()

def lsd(ref, est, n_fft=512, hop=256):
    w = np.hanning(n_fft)
    def fr(x):
        idx = np.arange(n_fft)[None,:] + hop*np.arange((len(x)-n_fft)//hop+1)[:,None]
        return np.abs(np.fft.rfft(x[idx]*w, axis=1)) + 1e-10
    R, E = fr(ref), fr(est)
    return float(np.mean(np.sqrt(((np.log10(R)-np.log10(E))**2).mean(axis=1)))*20)

def si_sdr(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t = a * ref; n = est - t
    return 10 * np.log10((np.dot(t, t) + 1e-12) / (np.dot(n, n) + 1e-12))

results = {}
for mk in ("ulunas", "gtcrn"):
    enh = create_enhancer(mk)
    d_in2rev, d_in2cln, d_ref2cln, d_rev2cln = [], [], [], []
    files = sorted(f for f in os.listdir(REV) if f.endswith(".wav") and not f.startswith("_"))[:N]
    print(f"\n=== {mk}: {len(files)} clips ===", flush=True)
    for f in files:
        rev_p = os.path.join(REV, f)
        cln_p = os.path.join(CLN, f.replace("reverb_speech", "clean"))
        if not os.path.exists(cln_p):
            continue
        rev, _ = sf.read(rev_p, dtype="float32")
        cln, _ = sf.read(cln_p, dtype="float32")
        n0 = min(len(rev), len(cln))
        rev, cln = rev[:n0], cln[:n0]
        # 输入 vs 干净（混响造成的劣化量）
        d_in2cln.append(si_sdr(cln, rev))
        # 增强 vs 干净（增强后保真度）
        est = enh.enhance(rev, 24000)
        e16 = to16(est)
        m = min(len(cln), len(e16))
        d_ref2cln.append(si_sdr(cln[:m], e16[:m]))
        d_in2rev.append(si_sdr(rev[:m], e16[:m]))
    a_in2rev = np.array(d_in2rev); a_ref2cln = np.array(d_ref2cln); a_in2cln = np.array(d_in2cln)
    results[mk] = {
        "n": len(a_in2rev),
        "rev_vs_cln(混响造成的损伤)": float(a_in2cln.mean()),
        "enh_vs_cln(增强后保真度)": float(a_ref2cln.mean()),
        "rev_vs_enh(模型对输入的改动量)": float(a_in2rev.mean()),
        "去混响净收益(enh vs rev)": float(a_ref2cln.mean() - a_in2cln.mean()),
    }
    for k, v in results[mk].items():
        print(f"  {k}: {v:+.2f} dB")

print("\n" + "=" * 60)
print("=== 关键判定 ===")
for mk, r in results.items():
    derev_gain = r["去混响净收益(enh vs rev)"]
    print(f"\n{mk}:")
    print(f"  混响造成的损伤(不增强): {r['rev_vs_cln(混响造成的损伤)']:+.2f} dB")
    print(f"  增强后的保真度:          {r['enh_vs_cln(增强后保真度)']:+.2f} dB")
    print(f"  去混响净收益:            {derev_gain:+.2f} dB")
    if derev_gain > 0.5:
        print("  -> 模型在部分去混响（良性）")
    elif derev_gain > -0.5:
        print("  -> 基本中性（没帮也没害）")
    else:
        print("  -> 引入了伪影（恶性）")
