"""语音增强封装：输入任意音频路径，输出 16kHz 的增强 wav。
支持三个模型：GTCRN（本地权重）、MetricGAN+ / SepFormer（SpeechBrain）。
I/O 统一用 soundfile（新版 torchaudio 读文件需要 torchcodec，这里绕开），
SpeechBrain 模型用 separate_batch / enhance_batch 接口直接吃张量，
GTCRN 用 STFT -> 模型 -> iSTFT 手动流水线。
"""
import os

import soundfile as sf
import torch
import torchaudio

MODEL_IDS = {
    "GTCRN（超轻量，老师指定）": "gtcrn",
    "MetricGAN+（快，效果一般）": "speechbrain/metricgan-plus-voicebank",
    "SepFormer（慢，效果好）": "speechbrain/sepformer-dns4-16k-enhancement",
}

MAX_SECONDS = 30          # 截断，防止长音频爆内存
TARGET_SR = 16000         # 三个模型都是 16kHz 训练的
_model_cache = {}         # 模型常驻内存，只加载一次

def load_mono_16k(path, max_seconds=MAX_SECONDS):
    data, sr = sf.read(path, dtype="float32", always_2d=True)   # [采样点, 声道]
    wav = torch.from_numpy(data.T)                              # [声道, 采样点]
    wav = wav.mean(dim=0, keepdim=True)                         # 混成单声道
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav[:, : max_seconds * TARGET_SR]

def _get_model(name):
    if name not in _model_cache:
        if name.startswith("GTCRN"):
            from gtcrn import GTCRN
            model = GTCRN()
            ckpt = torch.load(os.path.join("checkpoints", "model_trained_on_dns3.tar"),
                              map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
            _model_cache[name] = model
            return _model_cache[name]
        model_id = MODEL_IDS[name]
        savedir = os.path.join("pretrained_models", model_id.split("/")[-1])
        if name.startswith("SepFormer"):
            from speechbrain.inference.separation import SepformerSeparation
            _model_cache[name] = SepformerSeparation.from_hparams(
                source=model_id, savedir=savedir)
        else:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
            _model_cache[name] = SpectralMaskEnhancement.from_hparams(
                source=model_id, savedir=savedir)
    return _model_cache[name]

def enhance_file(in_path, model_name):
    """返回增强后 wav 的路径。"""
    os.makedirs("outputs", exist_ok=True)
    wav = load_mono_16k(in_path)                     # [1, N] @ 16kHz

    if model_name.startswith("GTCRN"):
        model = _get_model(model_name)
        with torch.no_grad():
            window = torch.hann_window(512).pow(0.5)
            # 新版 torch 只认复数 STFT；GTCRN 要 (257, T, 2) 的实虚部格式，两头手动转换
            spec = torch.stft(wav.squeeze(0), 512, 256, 512, window,
                              return_complex=True)                   # (257, T) 复数
            model_in = torch.stack((spec.real, spec.imag), dim=-1)   # (257, T, 2)
            est = model(model_in[None])[0]                           # (257, T, 2)
            est_complex = torch.complex(est[..., 0], est[..., 1])    # 转回复数
            audio = torch.istft(est_complex, 512, 256, 512, window)[None, :]
    else:
        model = _get_model(model_name)
        if model_name.startswith("SepFormer"):
            est = model.separate_batch(wav)              # [1, 采样点, 说话人数]
            audio = est[:, :, 0].detach().cpu()
        else:
            audio = model.enhance_batch(wav, torch.ones(1)).detach().cpu()   # [1, 采样点]

    audio = audio / audio.abs().max().clamp(min=1e-8) * 0.9    # 峰值归一化
    tag = MODEL_IDS[model_name].split("/")[-1]
    out_path = os.path.join(
        "outputs",
        os.path.splitext(os.path.basename(in_path))[0] + f"__{tag}.wav")
    sf.write(out_path, audio.squeeze(0).numpy(), TARGET_SR)
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python enhance.py 音频.wav [模型序号1|2|3]")
        sys.exit(1)
    names = list(MODEL_IDS)
    i = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    out = enhance_file(sys.argv[1], names[i])
    print("完成 ->", out)
