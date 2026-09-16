"""直接核对：ΔLSD 的翻转是否真实（对比 base LSD 与 enh LSD）"""
import numpy as np, soundfile as sf, torch, torchaudio
from enhance_module import create_enhancer
import csv, os
SR16=16000
def to16(x):
    t=torch.from_numpy(np.asarray(x,dtype=np.float32))[None]
    return torchaudio.functional.resample(t,24000,SR16)[0].numpy()
def _frame(x,n,hop):
    idx=np.arange(n)[None,:]+hop*np.arange((len(x)-n)//hop+1)[:,None]
    return x[idx]
def lsd(ref,est,n_fft=512,hop=256):
    w=np.hanning(n_fft)
    R=np.abs(np.fft.rfft(_frame(ref,n_fft,hop)*w,axis=1))+1e-10
    E=np.abs(np.fft.rfft(_frame(est,n_fft,hop)*w,axis=1))+1e-10
    return float(np.mean(np.sqrt(((np.log10(R)-np.log10(E))**2).mean(axis=1)))*20)
def si_sdr(ref,est):
    ref=ref-ref.mean(); est=est-est.mean()
    a=np.dot(est,ref)/(np.dot(ref,ref)+1e-10); t=a*ref; n=est-t
    return 10*np.log10((np.dot(t,t)+1e-10)/(np.dot(n,n)+1e-10))

rows=list(csv.DictReader(open("eval_mixtures_result.csv",encoding="utf-8")))
sel={s:[r for r in rows if r["model"]=="ulunas" and float(r["snr_db"])==s][:25] for s in (-5.0,10.0,25.0)}
enh=create_enhancer("ulunas")
print("| 输入SNR | base LSD(带噪vs干净) | enh LSD(增强vs干净) | ΔLSD | base SI-SDR | enh SI-SDR | ΔSI-SDR |")
print("|---|---|---|---|---|---|---|")
for snr in (-5.0,10.0,25.0):
    acc=[]
    for r in sel[snr]:
        mix,_=sf.read("C:/lrac_eval_mix/"+r["mix"].split("/root/lrac_eval_mix/")[-1],dtype="float32")
        ref,_=sf.read("C:/lrac_eval_mix/"+r["ref"].split("/root/lrac_eval_mix/")[-1],dtype="float32")
        est=enh.enhance(mix,24000)
        r16,m16,e16=to16(ref),to16(mix),to16(est)
        n=min(len(r16),len(m16),len(e16))
        acc.append((lsd(r16[:n],m16[:n]), lsd(r16[:n],e16[:n]),
                    si_sdr(r16[:n],m16[:n]), si_sdr(r16[:n],e16[:n])))
    a=np.array(acc).mean(axis=0)
    print(f"| {int(snr):+d} dB | {a[0]:.2f} | {a[1]:.2f} | {a[1]-a[0]:+.2f} | {a[2]:+.2f} | {a[3]:+.2f} | {a[3]-a[2]:+.2f} |")
