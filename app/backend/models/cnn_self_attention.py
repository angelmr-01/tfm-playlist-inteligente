import torch
import torch.nn as nn

# ==========================================
# 1. Positional Encoding
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ==========================================
# 2. CNN + Self-Attention Architecture
# ==========================================
class MusicCNNAttention(nn.Module):
    def __init__(self, num_classes=12, nhead=4, num_layers=2):
        super(MusicCNNAttention, self).__init__()

        # --- PHASE 1: Feature Extraction ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        d_model_cnn_out = 256 * 16

        self.bottleneck = nn.Linear(d_model_cnn_out, 512)
        self.d_model = 512

        # --- PHASE 2: Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        # --- PHASE 3: Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- PHASE 4: Multi-Label Classification ---
        self.fc = nn.Linear(self.d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2) # [Batch, Time, Channels, Freq]
        x = x.reshape(batch_size, time, channels * freq) # [Batch, Time, 4096]

        x = self.bottleneck(x)
        x = torch.relu(x)

        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = torch.mean(x, dim=1)
        return self.fc(x)