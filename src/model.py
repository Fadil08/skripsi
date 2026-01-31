# src/model.py
import torch
import torch.nn as nn
import timm

class EffNetLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=256, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)  # feature extractor
        feat_dim = self.backbone.num_features

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        # EfficientNet expects 3 channels, so repeat:
        x = x.repeat(1, 3, 1, 1)  # (B,3,n_mels,T)

        # We want sequence along time axis T.
        # Trick: split spectrogram into chunks along time, run backbone per chunk.
        # We'll chunk into 5 slices to create a short sequence.
        B, C, H, W = x.shape
        n_steps = 3
        step = W // n_steps
        feats = []
        for s in range(n_steps):
            xs = x[..., s*step:(s+1)*step]  # (B,3,H,step)
            f = self.backbone(xs)           # (B, feat_dim)
            feats.append(f)
        seq = torch.stack(feats, dim=1)     # (B, n_steps, feat_dim)

        out, _ = self.lstm(seq)             # (B, n_steps, hidden)
        last = out[:, -1, :]                # last timestep
        logits = self.head(last)            # (B, num_classes)
        return logits
