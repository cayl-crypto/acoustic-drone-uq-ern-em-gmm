#!/usr/bin/env python3
# extract_training_scalars_1x1x1.py
#
# Saves one .pt file per scalar position (h, w, t) containing all N training
# values at that position.
#
#   Output shape per file: (N,)  e.g. (40692,)
#   Total files: 128 x 128 x 31 = 524,288
#   Saved to: SAVE_DIR/t{t:02d}/{h:03d}_{w:03d}.pt
#
# Strategy: 31 passes through N files (one per T-slice).
# Per pass: load (N, 128, 128) into RAM (~2.7 GB), then split into 16,384
# individual (N,) files.

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
TRAIN_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/"
    "train_dataset_with_nondrone_with_9600_flag.csv"
)
TRAIN_FEATURE_DIR = Path(
    "/path/to/DDLDataset/plain/features/model"
)
SAVE_DIR = Path(
    "/path/to/DDLDataset/plain/"
    "training_scalars_1x1x1"
)

FULL_H = 128
FULL_W = 128
FULL_T = 31


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
print(f"Files found: {len(file_lookup)}")

train_df  = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["ID"].astype(str).tolist()

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
# 31 PASSES — ONE PER T-SLICE
# =========================================================
for t in tqdm(range(FULL_T), desc="T-slices"):
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

pd.DataFrame([{
    "n_training_samples": N,
    "full_h": FULL_H,
    "full_w": FULL_W,
    "full_t": FULL_T,
    "total_files": FULL_H * FULL_W * FULL_T,
    "shape_per_file": f"({N},)",
    "save_dir": str(SAVE_DIR.resolve()),
}]).to_csv(SAVE_DIR / "summary.csv", index=False)

print(f"\nDone. {FULL_H * FULL_W * FULL_T} files saved -> {SAVE_DIR.resolve()}")
print("Next: run gmm_scalar_fit_1x1x1.py")
