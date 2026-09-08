"""固定测试集一键评测（学姐要求：同一测试集对比效果和参数量大小）

跑法：python benchmark.py
产出：
  1. enhanced/<模型短名>/ 下每个测试音频的增强结果（耳朵验收用）
  2. benchmark_result.md 对比表——参数量/处理域/平均耗时脚本自动出，
     "效果"一列留空，由你听测后手工填

计时说明：每个模型先预热一次（含模型加载），计时只含纯推理。
"""
import os
import time

import soundfile as sf

from enhance_module import REGISTRY, create_enhancer

TEST_DIR = "testset"


def main():
    wavs = sorted(f for f in os.listdir(TEST_DIR)
                  if f.endswith(".wav") and not f.startswith("_"))
    if not wavs:
        raise SystemExit(f"{TEST_DIR} 里没有 wav，先跑 make_testset.py")
    print(f"测试集共 {len(wavs)} 条\n")

    rows = []
    for key in REGISTRY:
        print(f"=== 正在评测 {key} ===")
        enh = create_enhancer(key)
        outdir = os.path.join("enhanced", key)
        os.makedirs(outdir, exist_ok=True)

        # 预热：加载模型 + 首次推理，让计时只反映纯推理速度
        wav0, sr0 = sf.read(os.path.join(TEST_DIR, wavs[0]), dtype="float32")
        enh.enhance(wav0, sr0)

        total, n = 0.0, 0
        for f in wavs:
            wav, sr = sf.read(os.path.join(TEST_DIR, f), dtype="float32")
            t0 = time.time()
            enhanced = enh.enhance(wav, sr)
            dt = time.time() - t0
            total += dt
            n += 1
            sf.write(os.path.join(outdir, f), enhanced, enh.out_sr)
            print(f"  {f:<22} {dt:.2f}s")

        params = enh.count_params()
        rows.append((enh.name, enh.domain, params, f"{total/n:.2f}"))
        print(f"  -> 参数量 {params}，平均 {total/n:.2f}s/条\n")

    with open("benchmark_result.md", "w", encoding="utf-8") as fp:
        fp.write("# 轻量增强模型固定测试集对比\n\n")
        fp.write(f"测试集：testset/（{len(wavs)} 条 = 6 类噪声 × 2 档 SNR）\n\n")
        fp.write("| 模型 | 处理域 | 参数量 | 平均耗时/条 | 效果（听测后填） |\n")
        fp.write("|---|---|---|---|---|\n")
        for r in rows:
            fp.write("| " + " | ".join(r) + " | 待填 |\n")
    print("已生成 benchmark_result.md，听测后把'效果'列补上")


if __name__ == "__main__":
    main()
