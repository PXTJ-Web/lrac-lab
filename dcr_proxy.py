"""DCR proxy: measure model modification of input vs improvement over clean ref."""
import os, time
import numpy as np, soundfile as sf, torch, torchaudio
from enhance_module import create_enhancer
from evaluate import lsd, si_sdr

def to16(x):
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, 24000, 16000)[0].numpy()

def lsd16(ref, est, n_fft=512, hop=256):
    w = np.hanning(n_fft)
    def fr(x):
        idx = np.arange(n_fft)[None,:] + hop*np.arange((len(x)-n_fft)//hop+1)[:,None]
        return np.abs(np.fft.rfft(x[idx]*w, axis=1)) + 1e-10
    R, E = fr(ref), fr(est)
    return float(np.mean(np.sqrt(((np.log10(R)-np.log10(E))**2).mean(axis=1)))*20)

def si_sdr_np(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t = a * ref; n = est - t
    return 10 * np.log10((np.dot(t, t) + 1e-12) / (np.dot(n, n) + 1e-12))

ROOT = os.path.join("..", "official-testset", "LRAC-2025-test-data", "open-test-set")
N = 60

for mk in ("ulunas", "gtcrn"):
    enh = create_enhancer(mk)
    print(f"\n=== {mk} ===", flush=True)
    for track in ("track_1", "track_2"):
        rev_dir = os.path.join(ROOT, track, "reverb")
        ref_dir = os.path.join(ROOT, track, "reference_reverb")
        files = sorted(f for f in os.listdir(rev_dir) if f.endswith(".wav") and not f.startswith("_"))[:N]
        print(f"\n=== {track}/reverb: {len(files)} clips ===", flush=True)
        mods, gains_sdr, gains_lsd = [], [], []
        t0 = time.time()
        for f in files:
            inp, sr1 = sf.read(os.path.join(rev_dir, f), dtype="float32")
            ref, sr2 = sf.read(os.path.join(ref_dir, f), dtype="float32")
            i16, r16 = to16(inp), to16(ref)
            n = min(len(i16), len(r16))
            mods.append(si_sdr_np(i16[:n], r16[:n]))
            est = enh.enhance(inp, 24000)
            e16 = to16(est)
            n2 = min(len(i16), len(e16))
            gains_sdr.append(si_sdr_np(i16[:n2], e16[:n2]))
            gains_lsd.append(lsd16(r16[:n2], e16[:n2]))
        mods = np.array(mods); gains_sdr = np.array(gains_sdr); gains_lsd = np.array(gains_lsd)
        print(f"  modification SI-SDR(input,output) mean {mods.mean():+.2f} dB")
        print(f"  vs clean SI-SDR(ref,output) mean {gains_sdr.mean():+.2f} dB")
        print(f"  vs clean LSD(ref,output) mean {gains_lsd.mean():.2f} dB")
