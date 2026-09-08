"""统一语音增强模块接口（组内规范 v1.0，2026-09-05）

设计目标（学姐要求）：
  1. 换模型只改"适配器"那块，上层调用方永远不变；
  2. 接口统一：输入任意采样率带噪波形，输出 16kHz 单声道增强波形；
  3. 每个模型自带元信息：处理域、参数量，供对比表自动生成。

使用方法（未来 codec 管线也是这么调）：
    from enhance_module import create_enhancer
    enhancer = create_enhancer("gtcrn")            # 以后换模型只改这一个字符串
    enhanced = enhancer.enhance(wav_noisy, sr)     # numpy float32 一维进，一维出

新增模型三步：
    1. 写一个继承 BaseEnhancer 的类，实现 _load() 和 _forward()；
    2. 在 REGISTRY 里加一行；
    3. 完成——benchmark / demo / 未来管线自动认识它。
"""
import abc
import os

import numpy as np
import torch
import torchaudio

TARGET_SR = 16000
MAX_SECONDS = 30


class BaseEnhancer(abc.ABC):
    """所有增强模型的统一接口。子类只需实现 _load 和 _forward 两个方法。"""

    name = "base"          # 显示名
    key = "base"           # 注册短名
    domain = "未知"         # 处理域：时域 / 频域-复数谱 / 频域-幅度掩码 / 混合(DSP+NN)
    n_params = "待查"       # 加载后自动统计
    out_sr = 24000         # 对外输出采样率（官方管线口径 24k；内部统一 16k 处理）

    def __init__(self):
        self.model = None

    # ---- 对外唯一入口，子类不要覆盖 ----
    def enhance(self, wav, sr):
        """输入带噪波形 numpy float32（任意采样率，可多声道），
        输出 out_sr（默认 24k）增强波形——对外 24k、内部 16k 处理。"""
        if self.model is None:
            self._load()
        wav_t = self._to_16k_tensor(wav, sr)
        with torch.no_grad():
            out = self._forward(wav_t)
        out = out / out.abs().max().clamp(min=1e-8) * 0.9
        if self.out_sr != TARGET_SR:
            out = torchaudio.functional.resample(out, TARGET_SR, self.out_sr)
        return out.squeeze(0).numpy()

    # ---- 统一前后处理 ----
    def _to_16k_tensor(self, wav, sr):
        wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        if wav_t.dim() > 1:                    # 多声道 -> 单声道
            wav_t = wav_t.mean(dim=0, keepdim=True) if wav_t.shape[0] < wav_t.shape[-1] \
                else wav_t.mean(dim=-1, keepdim=True).T
        wav_t = wav_t.reshape(1, -1)
        if sr != TARGET_SR:
            wav_t = torchaudio.functional.resample(wav_t, sr, TARGET_SR)
        return wav_t[:, : MAX_SECONDS * TARGET_SR]

    def count_params(self):
        """统计参数量，返回形如 '48.2K' / '2.31M' 的字符串。"""
        if self.model is None:
            self._load()
        try:
            n = sum(p.numel() for p in self.model.parameters())
            self.n_params = f"{n/1e3:.1f}K" if n < 1e6 else f"{n/1e6:.2f}M"
        except Exception:
            self.n_params = "待查"
        return self.n_params

    # ---- 子类实现这两块 ----
    @abc.abstractmethod
    def _load(self):
        """把模型加载到 self.model（需是 nn.Module，以便统计参数）。"""

    @abc.abstractmethod
    def _forward(self, wav_t):
        """输入 [1, N] 16kHz float32 张量，返回 [1, N] 增强张量。"""


class GtcrnEnhancer(BaseEnhancer):
    key = "gtcrn"
    name = "GTCRN（超轻量，老师指定）"
    domain = "频域-复数谱（STFT 512/256 + ERB 频带映射）"

    def _load(self):
        from gtcrn import GTCRN
        model = GTCRN()
        ckpt = torch.load(os.path.join("checkpoints", "model_trained_on_dns3.tar"),
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model

    def _forward(self, wav_t):
        window = torch.hann_window(512).pow(0.5)
        spec = torch.stft(wav_t.squeeze(0), 512, 256, 512, window,
                          return_complex=True)                       # (257, T) 复数
        model_in = torch.stack((spec.real, spec.imag), dim=-1)       # (257, T, 2)
        est = self.model(model_in[None])[0]                          # (257, T, 2)
        est_complex = torch.complex(est[..., 0], est[..., 1])
        return torch.istft(est_complex, 512, 256, 512, window)[None, :]


class _SpeechBrainEnhancer(BaseEnhancer):
    """SpeechBrain 系模型的公共适配器（只差 HF id 和 forward 细节）。"""
    hf_id = ""

    def _load(self):
        savedir = os.path.join("pretrained_models", self.hf_id.split("/")[-1])
        if self.key == "sepformer":
            from speechbrain.inference.separation import SepformerSeparation
            self.model = SepformerSeparation.from_hparams(source=self.hf_id,
                                                          savedir=savedir)
        else:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
            self.model = SpectralMaskEnhancement.from_hparams(source=self.hf_id,
                                                              savedir=savedir)

    def _forward(self, wav_t):
        if self.key == "sepformer":
            return self.model.separate_batch(wav_t)[:, :, 0].detach().cpu()
        return self.model.enhance_batch(wav_t, torch.ones(1)).detach().cpu()


class MetricGANEnhancer(_SpeechBrainEnhancer):
    key = "metricgan"
    name = "MetricGAN+（基线）"
    domain = "频域-幅度谱掩码"
    hf_id = "speechbrain/metricgan-plus-voicebank"


class SepformerEnhancer(_SpeechBrainEnhancer):
    key = "sepformer"
    name = "SepFormer（重型参照）"
    domain = "时域（波形到波形）"
    hf_id = "speechbrain/sepformer-dns4-16k-enhancement"


class LiSenNetEnhancer(BaseEnhancer):
    """第三方 ONNX 重训版（claroche1/LiSenNet，VoiceBank-DEMAND 16k）。
    特征配置由变体矩阵实验确定：通道序 [幅度, 实部, 虚部]，不压缩，sqrt-hann 窗，
    模型只估幅度谱，相位复用带噪相位（处理域：频域-幅度谱估计）。"""
    key = "lisennet"
    name = "LiSenNet（37K，第三方 ONNX 重训版）"
    domain = "频域-幅度谱估计（STFT 512/256，复用带噪相位）"

    def _load(self):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            os.path.join("lisennet-onnx", "g_best_fp32.onnx"),
            providers=["CPUExecutionProvider"])
        self.model = self.session          # 作为"已加载"标志

    def count_params(self):
        self.n_params = "37K（第三方重训）"
        return self.n_params

    def _forward(self, wav_t):
        window = torch.hann_window(512).pow(0.5)
        spec = torch.stft(wav_t.squeeze(0), 512, 256, 512, window,
                          return_complex=True)                       # (257, T)
        feat = torch.stack((spec.abs(), spec.real, spec.imag),
                           dim=0).permute(0, 2, 1)[None]             # (1, 3, T, 257)
        est = torch.from_numpy(
            self.session.run(None, {"feat": feat.numpy()})[0])[0].T  # (257, T)
        est_spec = est * torch.exp(1j * spec.angle())                # 复用带噪相位
        return torch.istft(est_spec, 512, 256, 512, window)[None, :]


class FspenEnhancer(BaseEnhancer):
    """FSPEN（Samsung ICASSP 2024，79K），权重来自社区 Lightning 复现
    （iliasslasri/fspen，HF，VoiceBank-DEMAND 16k）。STFT 512/128。"""
    key = "fspen"
    name = "FSPEN（79K，社区复现权重）"
    domain = "频域-复数比率掩码+幅度增益（STFT 512/128，全带+子带融合）"

    def _load(self):
        import sys
        sys.path.insert(0, os.path.join("fspen_model"))
        from model import FSPEN
        model = FSPEN()
        ckpt = torch.load("fspen_last_ckpt.ckpt", map_location="cpu",
                          weights_only=False)
        sd = {k[len("net."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("net.")}
        model.load_state_dict(sd)
        model.eval()
        self.window = ckpt["state_dict"]["window"]       # 训练时的窗，直接复用
        self.model = model

    def _forward(self, wav_t):
        spec = torch.stft(wav_t.squeeze(0), 512, 128, 512, self.window,
                          return_complex=True)                    # (257, T) 复数
        x = torch.stack((spec.real, spec.imag), dim=-1)[None]     # (1, 257, T, 2)
        out = self.model(x)[0]                                    # (257, T, 2)
        est_complex = torch.view_as_complex(out.contiguous())
        return torch.istft(est_complex, 512, 128, 512, self.window)[None, :]

    def count_params(self):
        if self.model is None:
            self._load()
        n = sum(p.numel() for p in self.model.parameters())
        self.n_params = f"{n/1e3:.1f}K"
        return self.n_params


class UlunasEnhancer(BaseEnhancer):
    """UL-UNAS（arXiv 2503.00340，GTCRN 同作者的 NAS 产物，169K，零前瞻）。
    官方权重内置（DNS3 训练，checkpoint 键名与 GTCRN 同款）。
    模型内部自带 STFT，直接吃 [1, N] 波形、吐增强波形。"""
    key = "ulunas"
    name = "UL-UNAS（169K，NAS 搜索）"
    domain = "频域（模型内置 STFT 512/256，零前瞻因果）"

    def _load(self):
        from ulunas import ULUNAS
        model = ULUNAS()
        ckpt = torch.load(os.path.join("ulunas_checkpoints", "model_trained_on_dns3.tar"),
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model

    def _forward(self, wav_t):
        return self.model(wav_t)        # 模型内置 STFT，直接吃波形


# ---- 模型注册表：以后新增模型在这里加一行 ----
# 注：FSPEN 暂缓注册——社区权重与官方 model.py 键名不匹配（另一未公开框架训练），
#    待找到配套框架或完成键名映射后再接入（FspenEnhancer 类已写好，代码保留备用）。
REGISTRY = {
    "gtcrn": GtcrnEnhancer,
    "lisennet": LiSenNetEnhancer,
    "ulunas": UlunasEnhancer,
    "metricgan": MetricGANEnhancer,
    "sepformer": SepformerEnhancer,
}


def create_enhancer(key):
    """工厂函数：按短名创建增强器实例。"""
    if key not in REGISTRY:
        raise KeyError(f"未知模型 '{key}'，可选: {list(REGISTRY)}")
    return REGISTRY[key]()
