#!/usr/bin/env python3
# extract_features_cnn0_test_noise_types_c0p3_kl0.py
#
# CNN conv1 ("cnn0") feature extraction for the c0p3_kl0 checkpoint, for the
# 7 non-plain test-robustness conditions: AWGN at 3 SNRs, clutter (pink
# noise) at 3 SNRs, and the random-noise sanity-check condition.
#
# Mirrors extract_features_cnn0_c0p3_kl0.py's extraction logic exactly
# (same model.encoder.cnn[0] call). Sharded as a 7-way array job, one
# condition per job, since each condition's ~11,628 files is independent
# I/O work (same "many small files" cost we saw with the plain-condition
# extraction, so splitting it across concurrent jobs instead of one long
# serial pass). Resumable (skips files whose output already exists).
#
# Usage:
#   python extract_features_cnn0_test_noise_types_c0p3_kl0.py --job_id 0   # awgn_15dB
#   ...
#   python extract_features_cnn0_test_noise_types_c0p3_kl0.py --job_id 6   # random_noise
#
# Input : {condition}/features/logmel/{id}.pt        shape (8, 128, 31)
# Output: {condition}/features/model_c0p3_kl0/{id}.pt shape (128, 128, 31)
#
# Requires extract_awgn_test_logmel.py, extract_clutter_test_logmel.py, and
# extract_random_noise_test_logmel.py to have been run first.

import argparse
from pathlib import Path
import torch
from tqdm import tqdm

# =========================================================
# ARGS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, required=True,
                    help="Job index 0-6. Selects which condition this job processes.")
args = parser.parse_args()

CONDITIONS = [
    "awgn_15dB", "awgn_5dB", "awgn_-5dB",
    "clutter_15dB", "clutter_5dB", "clutter_-5dB",
    "random_noise",
]

assert 0 <= args.job_id < len(CONDITIONS), f"job_id must be 0-{len(CONDITIONS)-1}, got {args.job_id}"
CONDITION = CONDITIONS[args.job_id]

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path("best_evidential_loc_logmel_plain_plain_c0p3_kl0.pt")

DATASET_ROOT = Path("/path/to/DDLDataset")

INPUT_DIR = DATASET_ROOT / CONDITION / "features" / "logmel"
SAVE_DIR  = DATASET_ROOT / CONDITION / "features" / "model_c0p3_kl0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Job ID    : {args.job_id}")
print(f"Condition : {CONDITION}")


# =========================================================
# HELPERS
# =========================================================
def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def extract_conv1_feature(model, x: torch.Tensor) -> torch.Tensor:
    conv1 = model.encoder.cnn[0]
    return conv1(x)


def process_folder(model, input_dir: Path, save_dir: Path, desc: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} files in {input_dir}")
    if not pt_files:
        print(f"[WARNING] No .pt files found in: {input_dir}")
        return

    n_done = 0
    n_skipped = 0
    for pt_path in tqdm(pt_files, desc=desc):
        save_path = save_dir / pt_path.name
        if save_path.exists():
            n_skipped += 1
            continue
        try:
            x = torch.load(pt_path, map_location=DEVICE)
            x = ensure_4d(x).to(DEVICE)

            with torch.no_grad():
                feat = extract_conv1_feature(model, x)

            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]

            torch.save(feat.cpu(), save_path)
            n_done += 1

        except Exception as e:
            print(f"[ERROR] Failed on {pt_path}: {e}")

    print(f"  {desc}: {n_done} extracted, {n_skipped} already-done skipped, "
          f"{len(pt_files) - n_done - n_skipped} failed.")


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Checkpoint  : {MODEL_PATH.resolve()}")

    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()

    if not INPUT_DIR.exists():
        print(f"[WARNING] Input folder does not exist: {INPUT_DIR}")
        return

    process_folder(model, INPUT_DIR, SAVE_DIR, desc=CONDITION)

    print(f"\n[job {args.job_id}] Done. Condition: {CONDITION}")


if __name__ == "__main__":
    main()
