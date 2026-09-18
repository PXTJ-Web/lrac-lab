"""会议录音转文字：faster-whisper（本地推理，无需 API key）"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'

from faster_whisper import WhisperModel
import av
import numpy as np
import os

MP4 = r"C:\Users\梁鸣轩\Downloads\meeting_extract\2026-09-16 21.03.49.216\meeting_01.mp4"
OUT_TXT = "meeting_transcript.txt"
OUT_SRT = "meeting_transcript.srt"

print("加载模型（首次运行会下载）...")
model = WhisperModel("medium", device="cpu", compute_type="int8")

print("打开 mp4...")
c = av.open(MP4)
st = c.streams.audio[0]
print(f"音频: {st.codec_context.name} {st.codec_context.sample_rate}Hz {st.codec_context.channels}ch")
duration = float(c.duration / 1e6) if c.duration else 0
print(f"时长: {duration/60:.1f} 分钟")

# 用 PyAV 解码整条音频轨到 16kHz mono float32
c.close()
c = av.open(MP4)
resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
chunks = []
for frame in c.decode(audio=0):
    arr = frame.to_ndarray()
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    # frame.to_ndarray 形状可能是 (1, N) 或 (N, ch)
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    chunks.append(arr.reshape(-1))
c.close()
audio = np.concatenate(chunks).astype(np.float32)
print(f"解码完成: {len(audio)} 采样点 = {len(audio)/16000/60:.1f} 分钟")

print("开始转写（medium 模型，CPU，可能需要较长时间）...")
model = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, language="zh", beam_size=5,
                                   vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
print(f"检测语言: {info.language} (p={info.language_probability:.2f})")

with open(OUT_TXT, "w", encoding="utf-8") as ft, open(OUT_SRT, "w", encoding="utf-8") as fs_:
    fs_.write("")
    idx = 0
    for seg in segments:
        idx += 1
        line = f"[{idx}] [{seg.start/60:.0f}:{seg.start%60:04.1f} -> {seg.end/60:.0f}:{seg.end%60:04.1f}] {seg.text}"
        print(line, flush=True)
        ft.write(seg.text.strip() + "\n")
        srt_start = f"{int(seg.start//3600):02d}:{int(seg.start%3600//60):02d}:{int(seg.start%60):02d},{int(seg.start%1000):03d}"
        srt_end = f"{int(seg.end//3600):02d}:{int(seg.end%3600//60):02d}:{int(seg.end%60):02d},{int(seg.end%1000):03d}"
        fs_.write(f"{idx}\n{srt_start} --> {srt_end}\n{seg.text.strip()}\n\n")

print(f"\n完成: {OUT_TXT} + {OUT_SRT}")
