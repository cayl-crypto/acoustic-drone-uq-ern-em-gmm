#!/usr/bin/env python3
# extract_random_noise_test_logmel.py
#
# Synthesizes a pure-noise sanity-check condition: random waveforms shaped
# like real input (8, 9600) with amplitude statistics matched to real drone
# recordings (not literally structured audio, just matching amplitude
# range), then extracts logmel the same way as every other condition.
# Useful for checking whether the GMM-based OOD detector correctly flags
# completely unstructured input as highly anomalous.
#
# Checkpoint-agnostic (raw audio -> logmel), like extract_awgn_test_logmel.py
# and extract_clutter_test_logmel.py — no CSV, no real IDs. Output files are
# named rand_00000000.pt .. rand_{N-1:08d}.pt.
#
# Amplitude matching: loads a sample of real WavFiles/*.wav via
# torchaudio.load() (which normalizes int PCM to float32), computes the
# per-channel mean/std and min/max across that sample, then draws Gaussian
# noise with those statistics (clipped to the observed min/max) for each
# synthetic sample.
#
# Output: DDLDataset/random_noise/features/logmel/rand_{i:08d}.pt

from pathlib import Path
import random
import torch
import torchaudio
from tqdm import tqdm

from feature_extractor import AudioFeatureExtractor

# =========================================================
# CONFIG
# =========================================================
CSV_ROOT = Path("/path/to/DDLDataset")
WAV_ROOT = CSV_ROOT / "WavFiles"
RANDOM_LOGMEL_DIR = CSV_ROOT / "random_noise" / "features" / "logmel"

SAMPLE_RATE = 96000
AUDIO_LENGTH = 9600  # samples
NUM_CHANNELS = 8
N_SYNTHETIC = 1500          # number of random samples to generate
N_STATS_WAVS = 200          # how many real wavs to sample for amplitude stats
DEVICE = torch.device("cpu")
ONLY_FEATURE = "logmel"

torch.manual_seed(0)
random.seed(0)


# =========================================================
# HELPERS
# =========================================================
def load_wave(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected SR {SAMPLE_RATE}, got {sr} for {path}")
    if waveform.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Expected {NUM_CHANNELS} channels, got {waveform.shape[0]} for {path}")
    num_samples = waveform.shape[1]
    if num_samples < AUDIO_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, AUDIO_LENGTH - num_samples))
    elif num_samples > AUDIO_LENGTH:
        waveform = waveform[:, :AUDIO_LENGTH]
    return waveform


def compute_amplitude_stats(n_wavs: int):
    """Sample n_wavs real WAVs and return (mean, std, vmin, vmax) over all
    loaded samples (scalar, pooled across channels/time/files)."""
    all_wavs = sorted(WAV_ROOT.glob("*.wav"))
    sample_paths = random.sample(all_wavs, min(n_wavs, len(all_wavs)))

    chunks = []
    n_failed = 0
    for p in tqdm(sample_paths, desc="Sampling real WAVs for amplitude stats"):
        try:
            chunks.append(load_wave(p))
        except Exception:
            n_failed += 1
            continue

    if not chunks:
        raise RuntimeError("Could not load any real WAVs to compute amplitude stats.")

    stacked = torch.cat([c.reshape(-1) for c in chunks])
    mean = stacked.mean().item()
    std = stacked.std().item()
    vmin = stacked.min().item()
    vmax = stacked.max().item()

    print(f"Amplitude stats from {len(chunks)} real WAVs ({n_failed} failed to load):")
    print(f"  mean={mean:.6f}  std={std:.6f}  min={vmin:.6f}  max={vmax:.6f}")
    return mean, std, vmin, vmax


# =========================================================
# MAIN
# =========================================================
def main():
    RANDOM_LOGMEL_DIR.mkdir(parents=True, exist_ok=True)

    mean, std, vmin, vmax = compute_amplitude_stats(N_STATS_WAVS)

    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)

    n_done = 0
    n_skipped = 0
    for i in tqdm(range(N_SYNTHETIC), desc="Extracting random-noise logmel"):
        out_path = RANDOM_LOGMEL_DIR / f"rand_{i:08d}.pt"
        if out_path.exists():
            n_skipped += 1
            continue

        waveform = torch.randn(NUM_CHANNELS, AUDIO_LENGTH, device=DEVICE) * std + mean
        waveform = waveform.clamp(min=vmin, max=vmax)

        feats = extractor.extract_all(waveform)
        torch.save(feats[ONLY_FEATURE].cpu(), out_path)
        n_done += 1

    print(f"\nDone. Newly generated: {n_done}, already-done skipped: {n_skipped}")
    print(f"Random-noise logmel dir: {RANDOM_LOGMEL_DIR}")
    print("\nNext: run extract_features_cnn0_test_noise_types_c0p3_kl0.py")


if __name__ == "__main__":
    main()
