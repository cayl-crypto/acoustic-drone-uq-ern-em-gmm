#!/usr/bin/env python3
# extract_clutter_test_logmel.py
#
# For the TEST split only: synthesizes "clutter" noise and mixes it into
# every test file at target SNRs, then extracts logmel from the mixed
# audio. Structured exactly like extract_awgn_test_logmel.py's AWGN path:
# noise is generated on the fly, independently per sample (no CSV file
# reads, no dependency on the "Non-Drone File" column or external clips).
#
# Clutter is modelled as colored (pink, ~1/f) noise rather than white
# Gaussian noise — this gives it a distinct low-frequency-heavy spectral
# character (closer to real-world background/engine/wind-type clutter)
# so it's a meaningfully different condition from the AWGN one, while
# using the same per-sample-random, whole-test-set-per-SNR approach.
#
# SNR_LEVELS_DB = [15, 5, -5]
#
# Outputs:
#   DDLDataset/clutter_15dB/features/logmel/{id}.pt
#   DDLDataset/clutter_5dB/features/logmel/{id}.pt
#   DDLDataset/clutter_-5dB/features/logmel/{id}.pt

from pathlib import Path
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from feature_extractor import AudioFeatureExtractor

# =========================================================
# CONFIG
# =========================================================
CSV_ROOT = Path("/path/to/DDLDataset")
WAV_ROOT = CSV_ROOT / "WavFiles"
TEST_CSV = CSV_ROOT / "annotations" / "test_dataset_with_nondrone_with_9600_flag.csv"

SNR_LEVELS_DB = [15, 5, -5]
CLUTTER_LOGMEL_DIRS = {
    snr: CSV_ROOT / f"clutter_{snr}dB" / "features" / "logmel" for snr in SNR_LEVELS_DB
}

SAMPLE_RATE = 96000
AUDIO_LENGTH = 9600  # samples
NUM_CHANNELS = 8
EPS = 1e-12
DEVICE = torch.device("cpu")
ONLY_FEATURE = "logmel"

torch.manual_seed(0)


# =========================================================
# HELPERS
# =========================================================
def load_filtered_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df["is_corrupt"] == False) & (df["sample_9600"] == True)].copy()
    return df.reset_index(drop=True)


def id_to_basename(val) -> str:
    return f"{int(val):08d}"


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


def generate_pink_noise(num_channels: int, length: int, device) -> torch.Tensor:
    """(C, T) unit-power pink (~1/f) noise, independent draw per channel."""
    white = torch.randn(num_channels, length, device=device)
    spectrum = torch.fft.rfft(white, dim=-1)
    freqs = torch.fft.rfftfreq(length, device=device).clamp(min=1.0 / length)  # avoid /0 at DC
    spectrum = spectrum / torch.sqrt(freqs)
    pink = torch.fft.irfft(spectrum, n=length, dim=-1)
    pink = pink / pink.std(dim=-1, keepdim=True).clamp(min=EPS)
    return pink


def add_clutter_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """waveform: (C, T). Adds per-call random pink noise scaled to target SNR (dB)."""
    C, T = waveform.shape
    noise = generate_pink_noise(C, T, waveform.device)

    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp(min=EPS)
    snr_linear = 10.0 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    noise = noise * torch.sqrt(target_noise_power / noise_power)

    return waveform + noise


# =========================================================
# MAIN
# =========================================================
def main():
    for d in CLUTTER_LOGMEL_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    test_df = load_filtered_dataframe(TEST_CSV)
    print(f"Test rows (is_corrupt==False & sample_9600==True): {len(test_df)}")

    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)

    n_missing_wav = 0
    for i in tqdm(range(len(test_df)), desc="Extracting clutter logmel (test)"):
        row = test_df.iloc[i]
        basename = id_to_basename(row["ID"])
        wav_path = WAV_ROOT / f"{basename}.wav"

        out_paths = {snr: CLUTTER_LOGMEL_DIRS[snr] / f"{basename}.pt" for snr in SNR_LEVELS_DB}
        if all(p.exists() for p in out_paths.values()):
            continue

        if not wav_path.exists():
            print(f"[MISSING WAV] {wav_path}")
            n_missing_wav += 1
            continue

        try:
            waveform = load_wave(wav_path).to(DEVICE)

            for snr in SNR_LEVELS_DB:
                out_path = out_paths[snr]
                if out_path.exists():
                    continue
                mixed = add_clutter_noise(waveform, snr)
                feats = extractor.extract_all(mixed)
                torch.save(feats[ONLY_FEATURE].cpu(), out_path)

        except Exception as e:
            print(f"[ERROR] ID={basename} -> {e}")

    print(f"\nDone. Missing wavs: {n_missing_wav}")
    for snr, d in CLUTTER_LOGMEL_DIRS.items():
        print(f"Clutter {snr:>4}dB dir : {d}")
    print("\nNext: run extract_features_cnn0_test_noise_types.py")


if __name__ == "__main__":
    main()
