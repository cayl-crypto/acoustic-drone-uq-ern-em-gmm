import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


class AudioFeatureExtractor:
    def __init__(
        self, sample_rate=96000, n_fft=2048, hop_length=320, win_length=1024, n_mels=128
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        self.mel_transform = T.MelScale(
            n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1
        )

    def extract_stft(self, audio):
        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs()
        phase = stft.angle()
        return magnitude, phase

    def compute_logmel(self, magnitude):
        return torch.log1p(self.mel_transform(magnitude))

    def extract_all(self, audio):

        mag, phase = self.extract_stft(audio)
        logmel = self.compute_logmel(mag)
        return {"logmel": logmel}
