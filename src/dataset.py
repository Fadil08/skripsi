# src/dataset.py
import os, glob, random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

def augment_audio(y, sr, aug=None):
    if aug is None or aug == "none":
        return y

    if aug == "gain":
        g = random.uniform(0.7, 1.3)
        return np.clip(y * g, -1.0, 1.0)

    if aug == "time_stretch":
        rate = random.uniform(0.85, 1.15)
        y2 = librosa.effects.time_stretch(y, rate=rate)
        # pad/crop back to original length
        if len(y2) < len(y):
            y2 = np.pad(y2, (0, len(y)-len(y2)))
        else:
            y2 = y2[:len(y)]
        return y2

    if aug == "pitch_shift":
        steps = random.uniform(-2.0, 2.0)
        y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        return y2

    if aug == "bg_noise":
        # add gaussian noise as proxy; you can swap with real background noise later
        snr_db = random.uniform(5, 20)
        sig_power = np.mean(y**2) + 1e-9
        noise_power = sig_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
        y2 = y + noise
        return np.clip(y2, -1.0, 1.0)

    return y

def mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512, fmin=200, fmax=10000):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # normalize to 0..1
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    return S_db.astype(np.float32)

class BirdSegDataset(Dataset):
    def __init__(self, root, sr=22050, aug="none", files=None, label_map=None):
        self.root = root
        self.sr = sr
        self.aug = aug

        if label_map is None:
            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            self.label_map = {c:i for i,c in enumerate(classes)}
        else:
            self.label_map = label_map

        if files is None:
            self.items = []
            for cls, idx in self.label_map.items():
                for fp in glob.glob(os.path.join(root, cls, "*.wav")):
                    self.items.append((fp, idx))
        else:
            self.items = files  # list of (fp, label_idx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fp, y_label = self.items[i]
        y, _ = librosa.load(fp, sr=self.sr, mono=True)
        y = augment_audio(y, self.sr, self.aug)
        m = mel_spectrogram(y, self.sr)  # (n_mels, T)
        # to tensor with channel dim for EfficientNet: (1, n_mels, T)
        x = torch.from_numpy(m).unsqueeze(0)
        return x, y_label
