# lrac-lab：轻量语音增强模型统一接口与评测

LRAC 2.0 挑战赛"codec 前端增强模块"的调研与评测工程。

## 内容

- **统一增强接口**（`enhance_module.py`）：基类 + 适配器 + 注册表，换模型只改一行
- **固定测试集**（`make_testset.py`）：6 类噪声 × 2 档 SNR，自动生成同源干净参考
- **一键评测**：`benchmark.py`（参数量/耗时/增强音频）+ `evaluate.py`（PESQ/STOI/SI-SDR/LSD）
- **已接入模型**：GTCRN（48.2K）、UL-UNAS（171K）、LiSenNet（37K，第三方 ONNX 重训）、MetricGAN+（1.9M）、SepFormer（25.6M）
- **调研报告**：`docs/` 下完整调研报告与简报

## 快速开始

```bash
pip install -r requirements.txt        # torch 建议单独按 CUDA 版本安装
python make_test_audio.py              # 生成单条冒烟测试音频
python make_testset.py                 # 生成 12 条多噪声测试集
python enhance.py testset/fan_10dB.wav 1   # 单条增强（1=GTCRN）
python benchmark.py                    # 全模型 × 全测试集
python evaluate.py                     # 客观指标
```

## 接口用法（codec 管线同款调用）

```python
from enhance_module import create_enhancer
enhancer = create_enhancer("gtcrn")     # 换模型只改这一个字符串
enhanced = enhancer.enhance(wav_noisy, sr)   # numpy float32 进/出
```

## 新增模型

继承 `BaseEnhancer` 实现 `_load()` 与 `_forward()`，在 `REGISTRY` 加一行即可。
