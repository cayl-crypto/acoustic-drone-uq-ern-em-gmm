#!/usr/bin/env python3
# extract_awgn_test_logmel.py
#
# For the TEST split only: extracts clean ("plain") logmel and additive
# white Gaussian noise (AWGN) logmel at target SNRs, from WavFiles/.
#
# AWGN is synthesized on the fly (not read from disk) at:
#   SNR_LEVELS_DB = [15, 5, -5]
#
# NOTE: these AWGN folders are named "awgn_{snr}dB" to avoid colliding
# with the dataset's own pre-baked "15dB"/"5dB"/"-5dB" noise_level
# folders (which are a different, already-recorded noise condition).
#
# Outputs:
#   DDLDataset/plain/features/logmel/{id}.pt            (clean, shared across the whole pipeline)
#   DDLDataset/awgn_15dB/features/logmel/{id}.pt
#   DDLDataset/awgn_5dB/features/logmel/{id}.pt
#   DDLDataset/awgn_-5dB/features/logmel/{id}.pt

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

PLAIN_LOGMEL_DIR = CSV_ROOT / "plain" / "features" / "logmel"
SNR_LEVELS_DB = [15, 5, -5]
AWGN_LOGMEL_DIRS = {
    snr: CSV_ROOT / f"awgn_{snr}dB" / "features" / "logmel" for snr in SNR_LEVELS_DB
}

SAMPLE_RATE = 96000
AUDIO_LENGTH = 9600  # samples
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
    if waveform.shape[0] != 8:
        raise ValueError(f"Expected 8 channels, got {waveform.shape[0]} for {path}")
    num_samples = waveform.shape[1]
    if num_samples < AUDIO_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, AUDIO_LENGTH - num_samples))
    elif num_samples > AUDIO_LENGTH:
        waveform = waveform[:, :AUDIO_LENGTH]
    return waveform


def add_awgn(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """waveform: (C, T). Adds per-call random AWGN scaled to target SNR (dB)."""
    signal_power = waveform.pow(2).mean()
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


# =========================================================
# MAIN
# =========================================================
def main():
    PLAIN_LOGMEL_DIR.mkdir(parents=True, exist_ok=True)
    for d in AWGN_LOGMEL_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    test_df = load_filtered_dataframe(TEST_CSV)
    print(f"Test rows (is_corrupt==False & sample_9600==True): {len(test_df)}")

    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)

    n_missing_wav = 0
    for i in tqdm(range(len(test_df)), desc="Extracting plain + AWGN logmel (test)"):
        row = test_df.iloc[i]
        basename = id_to_basename(row["ID"])
        wav_path = WAV_ROOT / f"{basename}.wav"

        plain_out = PLAIN_LOGMEL_DIR / f"{basename}.pt"
        awgn_outs = {snr: AWGN_LOGMEL_DIRS[snr] / f"{basename}.pt" for snr in SNR_LEVELS_DB}

        if plain_out.exists() and all(p.exists() for p in awgn_outs.values()):
            continue

        if not wav_path.exists():
            print(f"[MISSING] {wav_path}")
            n_missing_wav += 1
            continue

        try:
            waveform = load_wave(wav_path).to(DEVICE)

            if not plain_out.exists():
                feats = extractor.extract_all(waveform)
                torch.save(feats[ONLY_FEATURE].cpu(), plain_out)

            for snr in SNR_LEVELS_DB:
                out_path = awgn_outs[snr]
                if out_path.exists():
                    continue
                noisy_waveform = add_awgn(waveform, snr)
                feats_noisy = extractor.extract_all(noisy_waveform)
                torch.save(feats_noisy[ONLY_FEATURE].cpu(), out_path)

        except Exception as e:
            print(f"[ERROR] ID={basename} -> {e}")

    print(f"\nDone. Missing wavs: {n_missing_wav}")
    print(f"Plain logmel dir : {PLAIN_LOGMEL_DIR}")
    for snr, d in AWGN_LOGMEL_DIRS.items():
        print(f"AWGN {snr:>4}dB dir : {d}")
    print("\nNext: run extract_features_cnn0_test_noise_types.py")


if __name__ == "__main__":
    main()
