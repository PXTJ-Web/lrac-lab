"""审计：LSD 是否被峰值归一化污染？对比 raw LSD vs 增益对齐后的 LSD"""
import csv, os
import numpy as np, soundfile as sf, torch, torchaudio
from enhance_module import create_enhancer

SR16=16000
def to16(x):
    t=torch.from_numpy(np.asarray(x,dtype=np.float32))[None]
    return torchaudio.functional.resample(t,24000,SR16)[0].numpy()
def _frame(x,n,hop):
    idx=np.arange(n)[None,:]+hop*np.arange((len(x)-n)//hop+1)[:,None]
    return x[idx]
def lsd_raw(ref,est,n_fft=512,hop=256):
    w=np.hanning(n_fft)
    R=np.abs(np.fft.rfft(_frame(ref,n_fft,hop)*w,axis=1))+1e-10
    E=np.abs(np.fft.rfft(_frame(est,n_fft,hop)*w,axis=1))+1e-10
    return float(np.mean(np.sqrt(((np.log10(R)-np.log10(E))**2).mean(axis=1)))*20)
def lsd_gainmatched(ref,est):
    # 把 est 缩放到与 ref 相同的 RMS，再算 LSD（消除增益差异）
    r_rms=np.sqrt(np.mean(ref**2))+1e-12; e_rms=np.sqrt(np.mean(est**2))+1e-12
    est2=est*(r_rms/e_rms)
    return lsd_raw(ref,est2)

rows=list(csv.DictReader(open("eval_mixtures_result.csv",encoding="utf-8")))
sel=[r for r in rows if r["model"]=="ulunas" and float(r["snr_db"]) in (-5.0,10.0,25.0)][:1200]
by={-5.0:[],10.0:[],25.0:[]}
enh=create_enhancer("ulunas")
cnt={-5.0:0,10.0:0,25.0:0}
for r in sel:
    snr=float(r["snr_db"])
    if cnt[snr]>=40: continue
    cnt[snr]+=1
    mix,_=sf.read("C:/lrac_eval_mix/"+r["mix"].split("/root/lrac_eval_mix/")[-1],dtype="float32")
    ref,_=sf.read("C:/lrac_eval_mix/"+r["ref"].split("/root/lrac_eval_mix/")[-1],dtype="float32")
    est=enh.enhance(mix,24000)
    r16,e16=to16(ref),to16(est)
    n=min(len(r16),len(e16))
    by[snr].append((lsd_raw(r16[:n],e16[:n]), lsd_gainmatched(r16[:n],e16[:n])))
print("| 输入SNR | raw LSD(均值) | 增益对齐后 LSD(均值) | 差值=缩放贡献 |")
print("|---|---|---|---|")
for snr in (-5.0,10.0,25.0):
    a=np.array(by[snr])
    print(f"| {int(snr):+d} dB | {a[:,0].mean():.2f} | **{a[:,1].mean():.2f}** | {a[:,0].mean()-a[:,1].mean():.2f} |")
