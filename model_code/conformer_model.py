import torch
import torchaudio
import torch.nn as nn


class Conformer8Mic(nn.Module):
    def __init__(
        self,
        in_channels=8,
        input_freq_bins=128,
        input_time_frames=28,
        cnn_channels=128,
        conformer_dim=192,
        num_heads=6,
        ff_mult=4,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()

        # CNN Frontend: (B, 8, 64, 28) → (B, cnn_channels, T')
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
        )

        # Collapse freq + channel dims → sequence input for Conformer
        self.proj = nn.Linear(cnn_channels * input_freq_bins, conformer_dim)

        # Conformer encoder
        self.encoder = torchaudio.models.Conformer(
            input_dim=conformer_dim,
            num_heads=num_heads,
            ffn_dim=conformer_dim * ff_mult,
            num_layers=num_layers,
            dropout=dropout,
            depthwise_conv_kernel_size=31,
        )

        # Global average pooling across time
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embed_dim = conformer_dim

    def forward(self, x):
        """
        x: (B, 8, 64, 28) → 8-channel log-mel spectrogram
        """
        B = x.size(0)
        x = self.cnn(x)  # (B, C, F, T)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        x = x.view(B, T, C * F)  # (B, T, C*F)

        x = self.proj(x)  # (B, T, D)
        lengths = torch.full(
            (x.size(0),), x.size(1), dtype=torch.long, device=x.device
        )  # (B,) = [T, T, ...]
        x, _ = self.encoder(x, lengths)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return x  # → same structure as ASTLinear
