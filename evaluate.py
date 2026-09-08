"""客观指标评测：SI-SDR / STOI / LSD（/PESQ，可选）
对 enhanced/<模型>/ 里的增强结果与干净参考 test_clean.wav 计算指标，
输出 evaluate_result.md 对比表（含"不增强"的带噪基线行，用于看提升量）。

用法：
  1) 先跑过 benchmark.py（生成 enhanced/ 下的结果）
  2) pip install pystoi pesq   （pesq 装不上会自动跳过，不影响其余指标）
  3) python evaluate.py
"""
import os

import numpy as np
import soundfile as sf

TEST_DIR = "testset"
ENH_DIR = "enhanced"
CLEAN_REF = os.path.join("testset", "_clean_ref.wav")   # 与 testset 同一句话的干净参考
SR = 16000

# ---------- 指标实现 ----------

def si_sdr(ref, est):
    """尺度不变信噪比 (dB)，越高越好。"""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-10)
    target = est * 0 + alpha * ref
    noise = est - target
    return 10 * np.log10((np.dot(target, target) + 1e-10) /
                         (np.dot(noise, noise) + 1e-10))


def _frame(x, n, hop):
    idx = np.arange(n)[None, :] + hop * np.arange((len(x) - n) // hop + 1)[:, None]
    return x[idx]


def lsd(ref, est, n_fft=512, hop=256):
    """对数谱距离 (dB)，越低越好，编解码领域常用。"""
    w = np.hanning(n_fft)
    R = np.abs(np.fft.rfft(_frame(ref, n_fft, hop) * w, axis=1)) + 1e-10
    E = np.abs(np.fft.rfft(_frame(est, n_fft, hop) * w, axis=1)) + 1e-10
    d = (np.log10(R) - np.log10(E)) ** 2
    return np.mean(np.sqrt(d.mean(axis=1))) * 20


def stoi(ref, est):
    try:
        try:
            from pystoi.stoi import Stoi                 # 旧版：类接口
            return Stoi(ref, est, SR, extended=False).stoi
        except ImportError:
            from pystoi import stoi as pystoi_stoi       # 0.4.x：函数接口
            return pystoi_stoi(ref, est, SR, extended=False)
    except ImportError:
        return None


def pesq_score(ref, est):
    try:
        from pesq import pesq
        return pesq(SR, ref, est, "wb")
    except ImportError:
        return None
    except Exception:                      # 个别文件可能无声段报错，跳过该条
        return None

# ---------- 主流程 ----------

def main():
    have_pystoi = have_pesq = True
    try:
        import pystoi  # noqa: F401
    except ImportError:
        have_pystoi = False
    try:
        import pesq  # noqa: F401
    except ImportError:
        have_pesq = False
    print(f"依赖检查：pystoi={'可用' if have_pystoi else '未装(跳过STOI)'}，"
          f"pesq={'可用' if have_pesq else '未装(跳过PESQ)'}\n")

    ref, sr = sf.read(CLEAN_REF, dtype="float32")
    assert sr == SR

    def metrics_of(est):
        n = min(len(ref), len(est))
        out = {"SI-SDR": si_sdr(ref[:n], est[:n]),
               "LSD": lsd(ref[:n], est[:n])}
        if have_pystoi:
            out["STOI"] = stoi(ref[:n], est[:n])
        if have_pesq:
            p = pesq_score(ref[:n], est[:n])
            if p is not None:
                out["PESQ"] = p
        return out

    # 待评对象：带噪基线 + enhanced/ 下每个模型
    tasks = {"noisy(不增强)": TEST_DIR}
    for d in sorted(os.listdir(ENH_DIR)):
        p = os.path.join(ENH_DIR, d)
        if os.path.isdir(p):
            tasks[d] = p

    # results[模型][噪声类型] = [各条指标 dict]
    results, model_names = {}, list(tasks)
    for mname, mdir in tasks.items():
        per_type = {}
        for f in sorted(os.listdir(mdir)):
            if not f.endswith(".wav") or f.startswith("_"):
                continue
            est, esr = sf.read(os.path.join(mdir, f), dtype="float32")
            if esr != SR:      # 对外 24k 的结果先转回 16k 再算指标
                import torch, torchaudio
                est = torchaudio.functional.resample(
                    torch.from_numpy(est)[None], esr, SR)[0].numpy()
            ntype = f.split("_")[0]
            per_type.setdefault(ntype, []).append(metrics_of(est))
        results[mname] = per_type
        print(f"已评完 {mname}")

    # 汇总成表
    metric_keys = [k for k in ["PESQ", "STOI", "SI-SDR", "LSD"]
                   if any(k in d for per in results.values()
                          for lst in per.values() for d in lst)]
    lines = ["# 客观指标对比（越高越好：PESQ/STOI/SI-SDR；越低越好：LSD）", ""]
    header = "| 模型 | " + " | ".join(metric_keys) + " |"
    lines += [header, "|---" * (len(metric_keys) + 1) + "|"]

    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"

    for mname in model_names:
        per_type = results[mname]
        row = [mname]
        for k in metric_keys:
            vals = [d[k] for t in per_type.values() for d in t if k in d]
            row.append(fmt(np.mean(vals)) if vals else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 分噪声类型明细（SI-SDR dB）", ""]
    ntypes = sorted({t for per in results.values() for t in per})
    lines += ["| 模型 | " + " | ".join(ntypes) + " |", "|---" * (len(ntypes) + 1) + "|"]
    for mname in model_names:
        per_type = results[mname]
        row = [mname]
        for t in ntypes:
            vals = [d["SI-SDR"] for d in per_type.get(t, []) if "SI-SDR" in d]
            row.append(fmt(np.mean(vals)) if vals else "—")
        lines.append("| " + " | ".join(row) + " |")

    with open("evaluate_result.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print("\n".join(lines))
    print("\n已生成 evaluate_result.md")


if __name__ == "__main__":
    main()
