import os, glob
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# ================= FILTER =================
def bandpass_filter(y, sr, low_hz=200.0, high_hz=10000.0, order=4):
    nyq = 0.5 * sr
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y)

def normalize(y, eps=1e-9):
    m = np.max(np.abs(y)) + eps
    return y / m

# ================= SEGMENT =================
def pad_or_trim(y, sr, target_sec=5.0):
    target_len = int(sr * target_sec)
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y

def segment_audio(y, sr, seg_sec=5.0):
    seg_len = int(sr * seg_sec)
    if len(y) < seg_len:
        return [pad_or_trim(y, sr, seg_sec)]
    n = len(y) // seg_len
    y = y[: n * seg_len]
    return [y[i*seg_len:(i+1)*seg_len] for i in range(n)]

# ================= SILENCE & NOISE CHECK =================
def is_silent(seg, thr_db=-35.0, min_peak=0.02):
    rms = librosa.feature.rms(y=seg)[0].mean()
    db = librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0]
    peak = float(np.max(np.abs(seg)))
    return (db < thr_db) or (peak < min_peak)

def has_bird_band_energy(seg, sr, ratio_thr=0.35):
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr,
        n_mels=64, n_fft=1024, hop_length=512,
        fmin=200, fmax=10000
    )
    mel_freqs = librosa.mel_frequencies(64, fmin=200, fmax=10000)
    band = (mel_freqs >= 1000) & (mel_freqs <= 8000)
    band_energy = float(S[band].mean())
    total_energy = float(S.mean()) + 1e-9
    return (band_energy / total_energy) >= ratio_thr

# ================= MAIN =================
def process_folder(
    in_root="Dataset/raw",
    out_root="Dataset/processed",
    sr=22050,
    seg_sec=5.0
):
    os.makedirs(out_root, exist_ok=True)
    classes = sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))])

    for cls in classes:
        in_dir = os.path.join(in_root, cls)
        out_dir = os.path.join(out_root, cls)
        os.makedirs(out_dir, exist_ok=True)

        files = (
            glob.glob(os.path.join(in_dir, "*.wav")) +
            glob.glob(os.path.join(in_dir, "*.mp3")) +
            glob.glob(os.path.join(in_dir, "*.flac"))
        )

        idx = 0
        for fp in files:
            try:
                y, _ = librosa.load(fp, sr=sr, mono=True)
                y = normalize(y)
                y = bandpass_filter(y, sr)

                segs = segment_audio(y, sr, seg_sec=seg_sec)
                for seg in segs:
                    if is_silent(seg):
                        continue
                    if not has_bird_band_energy(seg, sr):
                        continue

                    out_path = os.path.join(
                        out_dir,
                        f"{os.path.splitext(os.path.basename(fp))[0]}_{idx:06d}.wav"
                    )
                    sf.write(out_path, seg, sr)
                    idx += 1
            except Exception as e:
                print(f"[SKIP] {fp} error: {e}")

        print(f"[OK] {cls}: saved {idx} segments")

if __name__ == "__main__":
    process_folder()
