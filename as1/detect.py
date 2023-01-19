import os
import sys
import shutil
import math
import time

from IPython.display import display
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import scipy as sp
import wfdb


class HPF():
    """First stage: Linear High Pass"""
    
    def __init__(self, M=5):
        self.b = -(np.ones(M) / M)
        self.b[int((M + 1) / 2)] += 1
        self.a = [1]
        self.nz = M - 1 # required "memory"
        
    def __call__(self, x, z=None):
        return sp.signal.lfilter(self.b, self.a, x, zi=z)
    
class LPF():
    """Second stage: Nonlinear Low Pass"""
    
    def __init__(self, M=30):
        self.M = M
        self.b = np.ones(M)
        self.a = [1]
        self.nz = M - 1
    
    def __call__(self, x, z=None):
        squared = x ** 2
        return sp.signal.lfilter(self.b, self.a, squared, zi=z)
    

class QRSDetector():
    """Filters signal and finds peak(s) for a segment"""
    
    def __init__(self, Mhp=5, Mlp=30, alpha=0.2, gamma=0.15, mode="conv", prom=0.5):
        self.hpf = HPF(Mhp)
        self.lpf = LPF(Mlp)
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode
        self.prom = prom
        
        self.thr = 0.01 # maybe determine dinamically?
        self.shift = int((Mhp + 1) / 2)
        self.hpz = np.zeros(self.hpf.nz)
        self.lpz = np.zeros(self.lpf.nz)
    
    def __call__(self, x):
        # STEP 1 and 2: hpf and lpf
        y, self.hpz = self.hpf(x, self.hpz)
        y, self.lpz = self.lpf(y, self.lpz)
        
        # STEP 3: decision-making
        y[y < self.thr] = 0
        if self.mode == "conv":
            peaks = np.arange(0, len(y))[np.convolve(y, [-1, 2 -1], "same") > 0]
        else:
            peaks, _ = sp.signal.find_peaks(y, prominence=self.prom)
        
        # remove multiple detections, useful when thr. has not yet been adapted
        peaks = remove_spurious(y, peaks)
        peakh = np.mean(y[peaks]) if len(peaks) > 0 else 2
        
        self.thr = self.alpha*self.gamma*peakh + (1 - self.alpha)*self.thr

        #peaks=None
        
        return y, peaks
    
def remove_spurious(sig, peaks):
    """Remove peak repeats, assuming fs=250"""
    thr = 50 #smaller than the distance between 2 heartbeats
    
    keptpeaks = []
    for i in range(0, len(sig), thr):
        cur_peaks = peaks[peaks >= i]
        cur_peaks = cur_peaks[cur_peaks < i+thr]
        if len(cur_peaks) == 0:
            continue
        pk = cur_peaks[np.argmax(sig[cur_peaks])]
        keptpeaks.append(pk)
    peaks = np.array(keptpeaks)
    
    to_rem = []
    for i in range(0, len(peaks)-1):
        if peaks[i+1] - peaks[i] < thr:
            if sig[peaks[i]] < sig[peaks[i+1]]:
                to_rem.append(i)
            else: to_rem.append(i+1)
    return np.delete(peaks, to_rem)
    
def detect_all(sig, detector, seglen=None, segsec=0.25, sr=250, verbose=True):
    """Run detector for all segments in given signal"""
    if seglen is None:
        seglen = int(segsec * sr)
    fsig, peaks = np.zeros((0,)), np.zeros((0,)).astype(int)
    for segst in range(0, len(sig), seglen):
        if verbose and segst%100000 == 0: print(f"sample = {segst}")
        seg = sig[segst: segst+seglen]
        y, pks = detector(seg)
        fsig = np.concatenate([fsig, y])
        peaks = np.concatenate([peaks, pks + segst])
    
    return fsig, peaks


def prepare_data(record_path, secto=None, sampto=None, pn_dir=None):
    """Read record and return signal, annotated peaks"""
    
    header = wfdb.rdheader(record_path, pn_dir=pn_dir)
    fs = header.fs
    if secto is not None: sampto = secto * fs
    if sampto is not None: sampto = min(sampto, header.sig_len)
    record = wfdb.rdrecord(record_path, sampto=sampto, pn_dir=pn_dir)
    ann = wfdb.rdann(record_path, 'atr' , sampto=sampto, pn_dir=pn_dir)
    sig = record.p_signal[:, 0]
    beat_ann =["N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"]
    mask = np.isin(ann.symbol, beat_ann)
    truepks = np.array(ann.sample)[mask]
    
    # RESAMPLE TO 250 - a makeshift solution to avoid adaptive parameter tuning
    ratio = 250 / fs
    sig = sp.signal.resample(sig, int(len(sig) * ratio))
    truepks = (truepks * ratio).astype(int)
    truepks = remove_spurious(sig, truepks)
    return sig, truepks, 250, fs

def plot_all(sig, truepks, fsig=None, predpks=None, labels=False):
    "Plot original signal, filtered signal, peaks."
    
    figsize(12, 4)
    plt.plot(sig, color="gray", linewidth=1)
    plt.plot(truepks, sig[truepks], "x", color="lime")
    if labels:
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
    if fsig is not None:
        plt.plot(fsig, "b", linewidth=1)
        plt.plot(predpks, fsig[predpks], "rx")


def compare_true_pred(truepks, predpks, thr):
    """Slow and dumb"""
    TP, FP, FN = 0, 0, 0
    _predpks = predpks.copy()
    for tpk in truepks:
        FN += 1
        for i,ppk in enumerate(_predpks):
            if abs(tpk - ppk) < thr:
                TP += 1; FN -= 1
                _predpks = np.delete(_predpks, i)
                break
        
    FP = len(_predpks)
    return TP, FP, FN

def eval_all(truepks, predpks, thr):
    TP, FP, FN = compare_true_pred(truepks, predpks, thr)
    ppv = TP / (TP + FP + 1e-10) # positive predictivity
    sensitivity = TP / (TP + FN + 1e-10)
    return ppv, sensitivity


def save_results(record_path, predpks, orig_fs):
    rdir = os.path.dirname(record_path)
    rname = os.path.basename(record_path)
    # UPSAMPLE AGAIN
    ratio = orig_fs / 250
    predpks = (predpks * ratio).astype(int)
    symbols = np.full(len(predpks), "N")
    #wfdb.wrann(rname, "asc", predpks, symbol=symbols, write_dir=rdir)
    
    with open(os.path.join(rdir, rname + ".asc"), "w") as f:
        for peak in predpks:
            f.write(f"0:00:00.00 {peak} N 0 0 0\n")


def qrs_detector_stack(record_path, secto=None, sampto=None, pn_dir=None, verbose=False):
    """Detect and evaluate detection for record at [record_path]"""
    
    print("Loading record...")
    sig, truepeaks, fs, orig_fs = prepare_data(record_path, secto=secto, sampto=sampto, pn_dir=pn_dir)
    
    detector = QRSDetector(Mhp=7, Mlp=10, alpha=0.2, gamma=0.09, mode="conv")
    # OR USE SCIPY's FINDPEAKS WITH A PROMINENCE SETTING FOR DECISION STEP
    # detector = QRSDetector(Mhp=7, Mlp=10, alpha=0.2, gamma=0.05, mode="findpeaks", prom=0.08)
    
    fsig, predpeaks = detect_all(sig, detector, seglen=1000, sr=fs, verbose=verbose)
    ppv, sensitivity = eval_all(truepeaks, predpeaks, 50)
    
    save_results(record_path, predpeaks, orig_fs)
    if verbose:
        print(f"ppv = {ppv:.4f}, sensitivity = {sensitivity:.4f}")
    return predpeaks, ppv, sensitivity


if __name__ == "__main__":

    peaks, ppc, sens = qrs_detector_stack(sys.argv[1], verbose=True)