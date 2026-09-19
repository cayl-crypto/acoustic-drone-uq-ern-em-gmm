#!/usr/bin/env python3
# extract_training_scalars_1x1x1_c0p3_kl0.py
#
# Saves one .pt file per scalar position (h, w, t) containing all N training
# values at that position, for the c0p3_kl0 (coeff=0.3, kl_weight=0.0)
# checkpoint's conv1 feature cache.
#
#   Output shape per file: (N,)
#   Total files: 128 x 128 x 31 = 524,288
#   Saved to: SAVE_DIR/t{t:02d}/{h:03d}_{w:03d}.pt
#
# Plain-only, no noise_level filtering: reads every ID in the train CSV
# (all noise_level values) and keeps whichever ones have a matching file in
# the plain conv1 feature dir. Since that feature dir was built from the
# plain-condition logmel cache only, this naturally lands on the plain
# subset without any explicit "noise_level" filter in this script — same
# approach as the original extract_training_scalars_1x1x1.py.
#
# Sharded as an 8-way array job over T-slices (unlike gmm_scalar_fit's
# H-row sharding — here each T-slice's read+write work is already
# independent of every other T-slice, since the algorithm does one full
# read pass over all N samples per T-slice). Same read/write logic as the
# original single-shot version, just restricted to each job's T range —
# not a rewrite of the I/O pattern, just parallelized across jobs so the
# per-file filesystem latency on scratch4weeks gets overlapped instead of
# serialized.
#
# Usage:
#   python extract_training_scalars_1x1x1_c0p3_kl0.py --job_id 0   # t = 0..3
#   python extract_training_scalars_1x1x1_c0p3_kl0.py --job_id 1   # t = 4..7
#   ...
#   python extract_training_scalars_1x1x1_c0p3_kl0.py --job_id 7   # t = 28..30
#
# Strategy per T-slice: load (N, 128, 128) into RAM, then split into 16,384
# individual (N,) files.

import argparse
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# ARGS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, required=True,
                    help="Job index 0-7. Determines T-slice range (~4 slices per job).")
args = parser.parse_args()

FULL_T = 31
N_JOBS = 8
T_PER_JOB = -(-FULL_T // N_JOBS)  # ceil(31 / 8) = 4
T_START = args.job_id * T_PER_JOB
T_END   = min(T_START + T_PER_JOB, FULL_T)

assert 0 <= args.job_id < N_JOBS, f"job_id must be 0-{N_JOBS-1}, got {args.job_id}"
assert T_START < FULL_T, f"job_id {args.job_id} has no T-slices to process (T_START={T_START} >= FULL_T={FULL_T})"

# =========================================================
# PATHS
# =========================================================
TRAIN_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/"
    "train_dataset_with_nondrone_with_9600_flag.csv"
)
TRAIN_FEATURE_DIR = Path(
    "/path/to/DDLDataset/plain/features/model_c0p3_kl0"
)
SAVE_DIR = Path(
    "/path/to/DDLDataset/plain/"
    "training_scalars_1x1x1_c0p3_kl0"
)

FULL_H = 128
FULL_W = 128

print(f"Job ID   : {args.job_id}")
print(f"T range  : {T_START} .. {T_END - 1}")


# =========================================================
# HELPERS
# =========================================================
def build_file_index(folder: Path) -> dict:
    lookup = {}
    for p in sorted(folder.glob("*.pt")):
        file_id = p.stem.split("_")[0]
        digits  = "".join(ch for ch in file_id if ch.isdigit())
        lookup[digits.zfill(8)] = p
    return lookup


def normalise_id(index: str) -> str:
    digits = "".join(ch for ch in str(index).strip() if ch.isdigit())
    return digits.zfill(8)


# =========================================================
# RESOLVE VALID TRAINING FILES
# =========================================================
print(f"Scanning: {TRAIN_FEATURE_DIR}")
file_lookup = build_file_index(TRAIN_FEATURE_DIR)
print(f"Files found in feature dir : {len(file_lookup)}")

train_df  = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["ID"].astype(str).tolist()
print(f"Rows in train CSV (all noise_level values) : {len(train_ids)}")

valid_ids = [
    idx for idx in train_ids
    if file_lookup.get(normalise_id(idx)) is not None
]
N = len(valid_ids)
print(f"Training IDs in CSV : {len(train_ids)}")
print(f"Valid files matched : {N}")

if N < 2:
    raise RuntimeError(f"Need >= 2 valid training samples; found {N}.")

# =========================================================
# PASSES — ONE PER T-SLICE IN THIS JOB'S RANGE
# =========================================================
for t in tqdm(range(T_START, T_END), desc=f"[job {args.job_id}] T-slices"):
    t_dir = SAVE_DIR / f"t{t:02d}"
    t_dir.mkdir(parents=True, exist_ok=True)

    # Check if this T-slice is already done
    if all((t_dir / f"{h:03d}_{w:03d}.pt").exists() for h in range(FULL_H) for w in range(FULL_W)):
        print(f"  t={t:02d} already complete, skipping.")
        continue

    # Load all N values for this T-slice -> (N, H, W)
    buf = torch.zeros((N, FULL_H, FULL_W), dtype=torch.float32)
    for i, idx in enumerate(tqdm(valid_ids, desc=f"  loading t={t:02d}", leave=False)):
        pt = file_lookup[normalise_id(idx)]
        x  = torch.load(pt, map_location="cpu").float()   # (H, W, T)
        buf[i] = x[:, :, t]

    # Save one (N,) file per position
    for h in range(FULL_H):
        for w in range(FULL_W):
            save_path = t_dir / f"{h:03d}_{w:03d}.pt"
            torch.save(buf[:, h, w].clone(), save_path)   # shape (N,)

    del buf
    print(f"  t={t:02d} done — saved {FULL_H * FULL_W} files to {t_dir}")

# Only job 0 writes the summary, once its own slice is done — informational
# only, nothing downstream reads it (gmm_scalar_fit_1x1x1_c0p3_kl0.py
# derives N itself from a sample file).
if args.job_id == 0:
    pd.DataFrame([{
        "n_training_samples": N,
        "full_h": FULL_H,
        "full_w": FULL_W,
        "full_t": FULL_T,
        "total_files": FULL_H * FULL_W * FULL_T,
        "shape_per_file": f"({N},)",
        "save_dir": str(SAVE_DIR.resolve()),
    }]).to_csv(SAVE_DIR / "summary.csv", index=False)

print(f"\n[job {args.job_id}] Done. t={T_START}..{T_END-1} complete.")
print("Run gmm_scalar_fit_1x1x1_c0p3_kl0.py once all 8 jobs finish.")
