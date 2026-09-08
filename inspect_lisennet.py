"""LiSenNet(第三方 ONNX 版)冒烟测试：
1) 打印模型的输入/输出签名（告诉咱们它吃什么、吐什么）
2) 打印 config.json（采样率、STFT 参数等）
3) 拿 1 秒真实带噪音频直接试喂，看能不能出结果
用法：python inspect_lisennet.py
"""
import json
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf

MODEL_DIR = "lisennet-onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "g_best_fp32.onnx")

print("=== 1. 输入/输出签名 ===")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
for i in sess.get_inputs():
    print("输入:", i.name, "| 形状:", i.shape, "| 类型:", i.type)
for o in sess.get_outputs():
    print("输出:", o.name, "| 形状:", o.shape, "| 类型:", o.type)

print("\n=== 2. config.json ===")
cfg_path = os.path.join(MODEL_DIR, "config.json")
if os.path.exists(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
else:
    print("(config.json 不存在)")

print("\n=== 3. 冒烟测试：喂 1 秒 fan_0dB.wav ===")
wav, sr = sf.read("testset/fan_0dB.wav", dtype="float32")
wav = wav[:16000]
print(f"输入音频: {len(wav)} 采样点 @ {sr}Hz")
try:
    out = sess.run(None, {sess.get_inputs()[0].name: wav[None, :]})
    for i, o in enumerate(out):
        print(f"成功! 输出[{i}] 形状: {o.shape}, 数值范围: [{o.min():.3f}, {o.max():.3f}]")
except Exception as e:
    print(f"直接喂波形失败 -> {type(e).__name__}: {e}")
    print("(没关系，签名信息已经拿到，下一步按实际要求的形状改造输入)")
