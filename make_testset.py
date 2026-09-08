"""多噪声测试集生成器：1 条干净语音 × 6 类噪声 × 2 档信噪比 → testset/ 文件夹
用法：python make_testset.py
"""
import glob
import os

import soundfile as sf
import torch
import torchaudio

os.makedirs("testset", exist_ok=True)
TARGET_SR = 16000

def load_wav_16k(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)

def loop_to(x, n):                       # 循环铺满到指定长度
    if x.numel() < n:
        x = x.repeat(n // x.numel() + 1)
    return x[:n]

def pink(n):                             # 粉红噪声：频谱按 1/sqrt(f) 衰减
    w = torch.randn(n)
    X = torch.fft.rfft(w)
    f = torch.fft.rfftfreq(n, 1 / TARGET_SR)
    return torch.fft.irfft(X / torch.sqrt(f + 20.0), n=n)

def fan(n):                              # 风扇/路噪：重低通的白噪轰鸣
    w = torch.randn(1, 1, n)
    kernel = torch.ones(1, 1, 200) / 200
    return torch.nn.functional.conv1d(w, kernel).view(-1)

def hum(n):                              # 50Hz 市电嗡声 + 谐波
    t = torch.arange(n) / TARGET_SR
    return sum((0.5 ** k) * torch.sin(2 * torch.pi * 50 * (k + 1) * t) for k in range(4))

def music(n):                            # 简易"音乐"：C大调和弦 + 音量起伏
    t = torch.arange(n) / TARGET_SR
    notes = [261.6, 329.6, 392.0, 523.3]
    x = sum(torch.sin(2 * torch.pi * f * t) for f in notes)
    return x * (0.5 + 0.5 * torch.sin(2 * torch.pi * 2.0 * t))

files = sorted(glob.glob(os.path.join("data", "waves_yesno", "*.wav")))
CLEAN_IDX = 7                            # 挑一条没用过的人声
clean = load_wav_16k(files[CLEAN_IDX])
N = clean.numel()

others = [load_wav_16k(p) for i, p in enumerate(files) if i != CLEAN_IDX]
babble = sum(loop_to(x, N) for x in others[:6])   # 6 条人声叠加 = 嘈杂人声

noises = {
    "white":  torch.randn(N),
    "pink":   pink(N),
    "fan":    loop_to(fan(2 * N), N),
    "hum":    hum(N),
    "babble": babble,
    "music":  music(N),
}

def mix_at_snr(sig, noise, snr_db):      # 按目标信噪比缩放噪声后混合
    p_s, p_n = sig.pow(2).mean(), noise.pow(2).mean()
    noise = noise * torch.sqrt(p_s / p_n * 10 ** (-snr_db / 10))
    mixed = sig + noise
    return mixed / mixed.abs().max() * 0.9

for name, noise in noises.items():
    for snr in (10, 0):
        mixed = mix_at_snr(clean, noise, snr)
        out = os.path.join("testset", f"{name}_{snr}dB.wav")
        sf.write(out, mixed.numpy(), TARGET_SR)
        print("已生成", out)

# 干净参考（evaluate.py 用它算全参考指标），与 testset 同一句话
sf.write(os.path.join("testset", "_clean_ref.wav"), clean.numpy(), TARGET_SR)
print("已生成 testset/_clean_ref.wav")