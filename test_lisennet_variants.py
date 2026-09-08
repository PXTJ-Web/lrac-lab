"""LiSenNet ONNX 接入变体矩阵：穷举合理的特征组合，用 SI-SDR 选出正确配置。
用法：python test_lisennet_variants.py
"""
import itertools

import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
import torchaudio

sess = ort.InferenceSession("lisennet-onnx/g_best_fp32.onnx",
                            providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name


def load16k(p):
    d, sr = sf.read(p, dtype="float32", always_2d=True)
    w = torch.from_numpy(d.T).mean(0, keepdim=True)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    return w


def si_sdr(ref, est):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-10)
    t = a * ref
    n = est - t
    return 10 * np.log10((np.dot(t, t) + 1e-10) / (np.dot(n, n) + 1e-10))


ref = load16k("testset/_clean_ref.wav").squeeze(0).numpy()
noisy_files = ["testset/fan_10dB.wav", "testset/babble_10dB.wav"]

results = []
orders = [("mag", "real", "imag"), ("mag", "imag", "real"), ("real", "imag", "mag")]
for order, compress, win_name in itertools.product(
        orders, (0.3, 0.0), ("hann", "hann_sqrt")):
    window = torch.hann_window(512)
    if win_name == "hann_sqrt":
        window = window.pow(0.5)

    scores = []
    for f in noisy_files:
        wav = load16k(f)[0]                                # (N,) 一维
        spec = torch.stft(wav, 512, 256, 512, window, return_complex=True)
        chans = {"mag": spec.abs() ** compress if compress > 0 else spec.abs(),
                 "real": spec.real, "imag": spec.imag}
        # 每个通道转成 (T, 257) 再堆通道 -> [1, 3, T, 257]
        feat = torch.stack([chans[c].T for c in order], dim=0)   # (3, T, 257)
        est = torch.from_numpy(sess.run(None, {IN: feat[None].numpy()})[0])[0].T  # 去batch维 -> (257, T)
        if compress > 0:
            est = est ** (1 / compress)                        # 解压缩回线性幅度
        est_spec = est * torch.exp(1j * spec.angle())          # 复用带噪相位
        out = torch.istft(est_spec, 512, 256, 512, window)[None, :]
        n = min(len(ref), len(out.squeeze(0)))
        scores.append(si_sdr(ref[:n], out.squeeze(0)[:n].numpy()))

    mean_sdr = float(np.mean(scores))
    results.append((mean_sdr, order, compress, win_name, scores))
    print(f"order={order} compress={compress} window={win_name} "
          f"-> SI-SDR {scores[0]:.2f} / {scores[1]:.2f}，均值 {mean_sdr:.2f} dB")

print("\n=== 排名（对照：不增强 fan_10dB 基线约 5.0）===")
for mean_sdr, order, compress, win_name, scores in sorted(results, reverse=True)[:3]:
    print(f"{mean_sdr:.2f} dB <- order={order}, compress={compress}, window={win_name}")
