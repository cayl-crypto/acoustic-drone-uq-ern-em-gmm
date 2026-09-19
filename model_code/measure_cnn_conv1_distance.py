from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from conformer_model import Conformer8Mic
from evidential_model import EvidentialLocalization
from feature_extractor import AudioFeatureExtractor


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path("best_evidential_model.pt")
WAV_PATH = Path(
    "/path/to/audio_files/00043904.wav"
)

SAMPLE_RATE = 96000
AUDIO_LENGTH = 9600
ONLY_FEATURE = "logmel"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_LEVELS_DB = [15, 5, -5]


# =========================================================
# HELPERS
# =========================================================
def load_wave(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected SR {SAMPLE_RATE}, got {sr} for {path.name}")

    if waveform.ndim != 2:
        raise ValueError(
            f"Expected 2D waveform, got shape {waveform.shape} for {path.name}"
        )

    num_channels, num_samples = waveform.shape

    if num_samples < AUDIO_LENGTH:
        pad = AUDIO_LENGTH - num_samples
        waveform = F.pad(waveform, (0, pad))
    elif num_samples > AUDIO_LENGTH:
        waveform = waveform[:, :AUDIO_LENGTH]

    return waveform


def add_awgn(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    waveform: [C, T]
    """
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise


def extract_logmel_from_waveform(
    waveform: torch.Tensor, extractor: AudioFeatureExtractor
) -> torch.Tensor:
    """
    waveform: [C, T]
    returns: [1, C, F, T] expected model input
    """
    waveform = waveform.to(DEVICE)
    feats = extractor.extract_all(waveform)
    logmel = feats[ONLY_FEATURE]

    if logmel.ndim == 3:
        logmel = logmel.unsqueeze(0)
    elif logmel.ndim != 4:
        raise ValueError(f"Unexpected logmel shape: {tuple(logmel.shape)}")

    return logmel


# =========================================================
# HOOK
# =========================================================
class SingleLayerHook:
    def __init__(self, layer):
        self.activation = None
        self.handle = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self.activation = out.detach().cpu()

    def close(self):
        self.handle.remove()


# =========================================================
# DISTANCE COMPUTATION
# =========================================================
def compute_channelwise_euclidean(clean_feat: torch.Tensor, noisy_feat: torch.Tensor):
    """
    clean_feat, noisy_feat: [1, C, H, W]
    returns: list of distances, length C
    """
    if clean_feat.shape != noisy_feat.shape:
        raise ValueError(
            f"Shape mismatch: clean {tuple(clean_feat.shape)} vs noisy {tuple(noisy_feat.shape)}"
        )

    if clean_feat.ndim != 4:
        raise ValueError(f"Expected 4D CNN output, got {tuple(clean_feat.shape)}")

    B, C, H, W = clean_feat.shape

    if B != 1:
        raise ValueError(f"Expected batch size 1, got {B}")

    distances = []
    for c in range(C):
        clean_map = clean_feat[0, c]
        noisy_map = noisy_feat[0, c]
        dist = torch.norm(clean_map - noisy_map, p=2).item()
        distances.append(dist)

    return distances


def print_distance_report(condition_name: str, distances):
    arr = np.array(distances, dtype=np.float64)

    print(f"\n{'=' * 70}")
    print(f"cnn_conv1 channel-wise Euclidean distances: {condition_name}")
    print(f"{'=' * 70}")

    for ch, d in enumerate(distances):
        print(f"Channel {ch:03d}: {d:.6f}")

    print(f"\nSummary for {condition_name}:")
    print(f"Mean distance : {arr.mean():.6f}")
    print(f"Std distance  : {arr.std():.6f}")
    print(f"Min distance  : {arr.min():.6f}")
    print(f"Max distance  : {arr.max():.6f}")

    top5_idx = np.argsort(arr)[-5:][::-1]
    print("\nTop-5 most changed channels:")
    for idx in top5_idx:
        print(f"Channel {idx:03d}: {arr[idx]:.6f}")


# =========================================================
# MAIN
# =========================================================
@torch.no_grad()
def main():
    print(f"Using device: {DEVICE}")

    model = torch.load(MODEL_PATH, map_location="cpu")
    model = model.to(DEVICE)
    model.eval()

    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)
    clean_waveform = load_wave(WAV_PATH)

    # Hook only cnn_conv1
    hook = SingleLayerHook(model.encoder.cnn[0])

    # -------- Clean pass --------
    clean_logmel = extract_logmel_from_waveform(clean_waveform, extractor)
    _ = model(clean_logmel.to(DEVICE))
    clean_cnn_conv1 = hook.activation.clone()

    if clean_cnn_conv1 is None:
        raise RuntimeError("Failed to capture clean cnn_conv1 activation.")

    print(f"Clean cnn_conv1 shape: {tuple(clean_cnn_conv1.shape)}")

    # -------- Noisy passes --------
    for snr_db in NOISE_LEVELS_DB:
        noisy_waveform = add_awgn(clean_waveform, snr_db)
        noisy_logmel = extract_logmel_from_waveform(noisy_waveform, extractor)

        _ = model(noisy_logmel.to(DEVICE))
        noisy_cnn_conv1 = hook.activation.clone()

        if noisy_cnn_conv1 is None:
            raise RuntimeError(
                f"Failed to capture cnn_conv1 activation for {snr_db} dB."
            )

        condition_name = f"{snr_db} dB" if snr_db >= 0 else f"{snr_db} dB"
        distances = compute_channelwise_euclidean(clean_cnn_conv1, noisy_cnn_conv1)
        print_distance_report(condition_name, distances)

    hook.close()


if __name__ == "__main__":
    main()
