# # src/model.py
# import torch
# import torch.nn as nn
# import timm

# class EffNetLSTM(nn.Module):
#     def __init__(self, num_classes, lstm_hidden=256, lstm_layers=1, dropout=0.2):
#         super().__init__()
#         self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)  # feature extractor
#         feat_dim = self.backbone.num_features

#         self.lstm = nn.LSTM(
#             input_size=feat_dim,
#             hidden_size=lstm_hidden,
#             num_layers=lstm_layers,
#             batch_first=True,
#             bidirectional=False
#         )
#         self.head = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(lstm_hidden, num_classes)
#         )

#     def forward(self, x):
#         # x: (B, 1, n_mels, T)
#         # EfficientNet expects 3 channels, so repeat:
#         x = x.repeat(1, 3, 1, 1)  # (B,3,n_mels,T)

#         # We want sequence along time axis T.
#         # Trick: split spectrogram into chunks along time, run backbone per chunk.
#         # We'll chunk into 5 slices to create a short sequence.
#         B, C, H, W = x.shape
#         n_steps = 3
#         step = W // n_steps
#         feats = []
#         for s in range(n_steps):
#             xs = x[..., s*step:(s+1)*step]  # (B,3,H,step)
#             f = self.backbone(xs)           # (B, feat_dim)
#             feats.append(f)
#         seq = torch.stack(feats, dim=1)     # (B, n_steps, feat_dim)

#         out, _ = self.lstm(seq)             # (B, n_steps, hidden)
#         last = out[:, -1, :]                # last timestep
#         logits = self.head(last)            # (B, num_classes)
#         return logits
import torch
import torch.nn as nn
import timm

class EffNetLSTM(nn.Module):
    def __init__(self, num_classes: int, n_steps: int = 3, lstm_hidden: int = 256):
        super().__init__()
        self.n_steps = n_steps

        # backbone feature extractor
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)  # output feat_dim
        feat_dim = self.backbone.num_features  # usually 1280

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        """
        x: (B, 1, n_mels, T)
        """
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)  # EfficientNet expects 3-channel

        # chunk along time axis W
        n_steps = self.n_steps
        step = max(1, W // n_steps)

        feats = []
        for s in range(n_steps):
            w0 = s * step
            w1 = (s + 1) * step if s < n_steps - 1 else W
            xs = x[:, :, :, w0:w1]  # (B, 3, H, w_slice)
            f = self.backbone(xs)   # (B, feat_dim)
            feats.append(f)

        seq = torch.stack(feats, dim=1)  # (B, n_steps, feat_dim)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]             # (B, lstm_hidden)
        logits = self.head(last)         # (B, num_classes)
        return logits
