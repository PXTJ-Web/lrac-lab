"""零准备生成带噪测试音频：自动下载 YESNO 开源语音，混入白噪声。
用法：python make_test_audio.py
"""
import glob
import os

import soundfile as sf
import torch
import torchaudio

os.makedirs("data", exist_ok=True)
# 只借用 YESNO 类完成下载和解压；读文件用 soundfile（新版 torchaudio 读文件需要 torchcodec）
torchaudio.datasets.YESNO(root="data", download=True)

wav_path = sorted(glob.glob(os.path.join("data", "waves_yesno", "*.wav")))[0]
data, sr = sf.read(wav_path, dtype="float32", always_2d=True)   # [采样点, 声道]
wav = torch.from_numpy(data.T)                                  # -> [声道, 采样点]
wav = wav.mean(dim=0, keepdim=True)                             # 混成单声道
wav = torchaudio.functional.resample(wav, sr, 16000)

torch.manual_seed(0)
noise_rms = wav.pow(2).mean().sqrt()
noise = torch.randn_like(wav) * noise_rms * 0.5     # 白噪声，约 +6dB 信噪比

noisy = wav + noise
noisy = noisy / noisy.abs().max() * 0.9
clean = wav / wav.abs().max() * 0.9

sf.write("test_noisy.wav", noisy.squeeze(0).numpy(), 16000)
sf.write("test_clean.wav", clean.squeeze(0).numpy(), 16000)
print("已生成 test_noisy.wav 和 test_clean.wav")