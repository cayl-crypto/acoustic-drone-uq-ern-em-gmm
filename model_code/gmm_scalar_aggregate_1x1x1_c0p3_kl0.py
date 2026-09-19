#!/usr/bin/env python3
# gmm_scalar_aggregate_1x1x1_c0p3_kl0.py
#
# Run after all 8 gmm_scalar_fit_1x1x1_c0p3_kl0.py jobs finish.
# Reads per-position .pt files from GMM_ROOT/models/positions/
# and builds dense (H, W, T) tensors for weights, means, variances,
# log-likelihoods, BICs, and optimal k.
#
# Output layout inside GMM_ROOT/aggregate/:
#   optimal_k.pt          (H, W, T)  int32
#   bics_k{k:02d}.pt      (H, W, T)  float32   for k=1..MAX_K
#   lls_k{k:02d}.pt       (H, W, T)  float32
#   weights_k{k:02d}.pt   (H, W, T, k) float32
#   means_k{k:02d}.pt     (H, W, T, k) float32
#   variances_k{k:02d}.pt (H, W, T, k) float32
#
# Kept entirely local under ./gmm_scalar_1x1x1_c0p3_kl0/ (not scratch4weeks)
# so it survives the scratch4weeks retention purge — see
# gmm_scalar_aggregate_1x1x1.py's note on why the original c0p1 pipeline
# moved this local after losing an aggregate this way once.
#
# Usage:
#   python gmm_scalar_aggregate_1x1x1_c0p3_kl0.py

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# =========================================================
# PATHS / CONFIG
# =========================================================
GMM_ROOT       = Path("./gmm_scalar_1x1x1_c0p3_kl0")
GMM_MODEL_ROOT = GMM_ROOT / "models"
POSITION_DIR   = GMM_MODEL_ROOT / "positions"

AGG_OUT_DIR = GMM_ROOT / "aggregate"
AGG_OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_H  = 128
FULL_W  = 128
FULL_T  = 31
MAX_K   = 10

print("GMM scalar aggregate — 1x1x1 — c0p3_kl0")
print(f"  Source : {POSITION_DIR}")
print(f"  Output : {AGG_OUT_DIR}")
print()

# =========================================================
# COUNT / VERIFY POSITION FILES
# =========================================================
total_expected = FULL_H * FULL_W * FULL_T
print(f"Expected position files : {total_expected}")

missing = []
for t in range(FULL_T):
    for h in range(FULL_H):
        for w in range(FULL_W):
            p = POSITION_DIR / f"t{t:02d}" / f"{h:03d}_{w:03d}.pt"
            if not p.exists():
                missing.append(str(p))

if missing:
    print(f"WARNING: {len(missing)} position files missing. First 5:")
    for m in missing[:5]:
        print(f"  {m}")
    print("Aggregation will leave zeros for missing positions.")
else:
    print("All position files present. Proceeding.")
print()

# =========================================================
# ALLOCATE DENSE OUTPUT TENSORS
# =========================================================
optimal_k_map = torch.zeros(FULL_H, FULL_W, FULL_T, dtype=torch.int32)

bics      = {k: torch.zeros(FULL_H, FULL_W, FULL_T, dtype=torch.float32) for k in range(1, MAX_K + 1)}
lls       = {k: torch.zeros(FULL_H, FULL_W, FULL_T, dtype=torch.float32) for k in range(1, MAX_K + 1)}
weights   = {k: torch.zeros(FULL_H, FULL_W, FULL_T, k,  dtype=torch.float32) for k in range(1, MAX_K + 1)}
means     = {k: torch.zeros(FULL_H, FULL_W, FULL_T, k,  dtype=torch.float32) for k in range(1, MAX_K + 1)}
variances = {k: torch.zeros(FULL_H, FULL_W, FULL_T, k,  dtype=torch.float32) for k in range(1, MAX_K + 1)}

# =========================================================
# READ POSITION FILES
# =========================================================
n_loaded = 0
n_skipped = 0

for t in tqdm(range(FULL_T), desc="T-slices"):
    for h in tqdm(range(FULL_H), desc=f"t={t:02d} H rows", leave=False):
        for w in range(FULL_W):
            p = POSITION_DIR / f"t{t:02d}" / f"{h:03d}_{w:03d}.pt"
            if not p.exists():
                n_skipped += 1
                continue

            res = torch.load(p, map_location="cpu", weights_only=False)

            optimal_k_map[h, w, t] = int(res["optimal_k"])

            for k in range(1, MAX_K + 1):
                if k not in res:
                    continue
                bics[k][h, w, t]        = float(res[k]["bic"])
                lls[k][h, w, t]         = float(res[k]["ll"])
                weights[k][h, w, t]     = torch.tensor(res[k]["weights"], dtype=torch.float32)
                means[k][h, w, t]       = torch.tensor(res[k]["means"],   dtype=torch.float32)
                variances[k][h, w, t]   = torch.tensor(res[k]["variances"], dtype=torch.float32)

            n_loaded += 1

print(f"\nLoaded  : {n_loaded} positions")
print(f"Skipped : {n_skipped} positions (missing files)")

# =========================================================
# SAVE DENSE TENSORS
# =========================================================
print("\nSaving dense tensors …")

torch.save(optimal_k_map, AGG_OUT_DIR / "optimal_k.pt")
print(f"  optimal_k.pt  shape={tuple(optimal_k_map.shape)}")

for k in range(1, MAX_K + 1):
    torch.save(bics[k],      AGG_OUT_DIR / f"bics_k{k:02d}.pt")
    torch.save(lls[k],       AGG_OUT_DIR / f"lls_k{k:02d}.pt")
    torch.save(weights[k],   AGG_OUT_DIR / f"weights_k{k:02d}.pt")
    torch.save(means[k],     AGG_OUT_DIR / f"means_k{k:02d}.pt")
    torch.save(variances[k], AGG_OUT_DIR / f"variances_k{k:02d}.pt")
    print(f"  k={k:02d}: bics{bics[k].shape}  lls{lls[k].shape}  "
          f"weights{weights[k].shape}  means{means[k].shape}  variances{variances[k].shape}")

print(f"\nDone. All tensors saved to {AGG_OUT_DIR.resolve()}")
print("Next: run extract_gmm_calibration_distances_1x1x1_c0p3_kl0.py")
