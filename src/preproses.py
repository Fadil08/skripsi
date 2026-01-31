# src/preprocess.py
import os, glob
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

def bandpass_filter(y, sr, low_hz=200.0, high_hz=10000.0, order=4):
    nyq = 0.5 * sr
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y)

def normalize(y, eps=1e-9):
    m = np.max(np.abs(y)) + eps
    return y / m

def segment_audio(y, sr, seg_sec=5.0):
    seg_len = int(sr * seg_sec)
    n = len(y) // seg_len
    if n == 0:
        return []
    y = y[: n * seg_len]
    return [y[i*seg_len:(i+1)*seg_len] for i in range(n)]

def is_silent(seg, thr_db=-40.0):
    # simple energy gate
    rms = librosa.feature.rms(y=seg)[0].mean()
    db = librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0]
    return db < thr_db

def process_folder(in_root="Dataset/raw", out_root="Dataset/processed", sr=22050, seg_sec=5.0):
    os.makedirs(out_root, exist_ok=True)
    classes = sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))])

    for cls in classes:
        in_dir = os.path.join(in_root, cls)
        out_dir = os.path.join(out_root, cls)
        os.makedirs(out_dir, exist_ok=True)

        files = glob.glob(os.path.join(in_dir, "*.wav")) + glob.glob(os.path.join(in_dir, "*.mp3"))
        idx = 0
        for fp in files:
            y, _ = librosa.load(fp, sr=sr, mono=True)
            y = normalize(y)
            y = bandpass_filter(y, sr)

            segs = segment_audio(y, sr, seg_sec=seg_sec)
            for seg in segs:
                if is_silent(seg):
                    continue
                out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_{idx:06d}.wav")
                sf.write(out_path, seg, sr)
                idx += 1

        print(f"[OK] {cls}: saved {idx} segments")

if __name__ == "__main__":
    process_folder()
