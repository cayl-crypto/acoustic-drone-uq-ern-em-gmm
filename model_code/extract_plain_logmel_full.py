#!/usr/bin/env python3
# extract_plain_logmel_full.py
#
# Full-coverage plain logmel extraction: every WAV file in WavFiles/ gets a
# logmel feature .pt, with no annotation-CSV / noise_level filtering at all
# (per user instruction: use every ID, read all from the plain WAV files).
#
# This is checkpoint-agnostic — it's raw audio -> logmel, shared by every
# downstream model-specific pipeline (c0p1, c0p3_kl0, ...), so it's not
# tagged with a checkpoint suffix. It just backfills the existing
# plain/features/logmel/ cache (previously only ~14,540 of the 58,136 IDs,
# because it had only ever been built for noise_level=="plain"-tagged CSV
# rows) up to full coverage.
#
# Recipe matches extract_awgn_test_logmel.py's plain-logmel branch exactly
# (same AudioFeatureExtractor, same 96kHz / 9600-sample assumptions,
# verified against a sample WAV: 8ch, 96kHz, 9600 frames).
#
# Input : WavFiles/{id}.wav                    8ch, 96kHz, ~9600 samples
# Output: plain/features/logmel/{id}.pt         shape (8, 128, 31)

from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

from feature_extractor import AudioFeatureExtractor

# =========================================================
# CONFIG
# =========================================================
CSV_ROOT = Path("/path/to/DDLDataset")
WAV_ROOT = CSV_ROOT / "WavFiles"
PLAIN_LOGMEL_DIR = CSV_ROOT / "plain" / "features" / "logmel"

SAMPLE_RATE  = 96000
AUDIO_LENGTH = 9600  # samples
DEVICE = torch.device("cpu")
ONLY_FEATURE = "logmel"


# =========================================================
# HELPERS
# =========================================================
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


# =========================================================
# MAIN
# =========================================================
def main():
    PLAIN_LOGMEL_DIR.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(WAV_ROOT.glob("*.wav"))
    print(f"Found {len(wav_paths)} wav files in {WAV_ROOT}")
    print(f"Existing logmel files already in {PLAIN_LOGMEL_DIR}: "
          f"{len(list(PLAIN_LOGMEL_DIR.glob('*.pt')))}")

    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for wav_path in tqdm(wav_paths, desc="Extracting plain logmel (full)"):
        out_path = PLAIN_LOGMEL_DIR / f"{wav_path.stem}.pt"

        if out_path.exists():
            n_skipped += 1
            continue

        try:
            waveform = load_wave(wav_path).to(DEVICE)
            feats = extractor.extract_all(waveform)
            torch.save(feats[ONLY_FEATURE].cpu(), out_path)
            n_done += 1

        except Exception as e:
            print(f"[ERROR] {wav_path.name} -> {e}")
            n_failed += 1

    total_now = len(list(PLAIN_LOGMEL_DIR.glob("*.pt")))

    print(f"\nDone.")
    print(f"  Wav files found      : {len(wav_paths)}")
    print(f"  Newly extracted      : {n_done}")
    print(f"  Already-done skipped : {n_skipped}")
    print(f"  Failed               : {n_failed}")
    print(f"  Total logmel files now in {PLAIN_LOGMEL_DIR}: {total_now}")
    print("\nNext: rerun extract_features_cnn0_c0p3_kl0.py (and/or extract_features_cnn0.py) "
          "to backfill conv1 features for the newly-covered IDs.")


if __name__ == "__main__":
    main()
