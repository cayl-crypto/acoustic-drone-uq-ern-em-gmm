#!/usr/bin/env python3
# calibrate_gmm_scalar_distance_quantiles_1x1x1_c0p3_kl0.py
#
# Conformal calibration of GMM NLL distance maps (1x1x1 scalar positions)
# for the c0p3_kl0 checkpoint.
#
# Input  : per-sample NLL distance .pt files (shape [128, 128, 31]) from
#            .../plain/distances/model_c0p3_kl0/gmm_k{k}/   k = 1 .. MAX_K
#            .../plain/distances/model_c0p3_kl0/gmm_optimal_k/
#
# Output : per-distance-type conformal quantile maps (shape [128, 128, 31])
#          + min/max calibration maps + stacked tensor + summary CSVs
#          Saved to: SAVE_ROOT / <distance_name>_*.pt
#
# Coverage targets:
#   q6827 = 68.27 %  (1 sigma)
#   q9545 = 95.45 %  (2 sigma)
#   q9973 = 99.73 %  (3 sigma)

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import math

# =========================================================
# PATHS
# =========================================================
DISTANCE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model_c0p3_kl0"
)

SAVE_ROOT = (
    DISTANCE_ROOT
    / "gmm_conformal_quantiles_1x1x1_c0p3_kl0_coverage_targets_with_min_max_distances"
)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
DTYPE  = torch.float32

FULL_H = 128
FULL_W = 128
FULL_T = 31
MAX_K  = 10

TARGET_COVERAGES = {
    "q6827": 0.6827,
    "q9545": 0.9545,
    "q9973": 0.9973,
}

# All distance types to calibrate: k=1..MAX_K and optimal_k
DISTANCE_DIRS = {}
for k in range(1, MAX_K + 1):
    DISTANCE_DIRS[f"gmm_k{k}"] = DISTANCE_ROOT / f"gmm_k{k}"
DISTANCE_DIRS["gmm_optimal_k"] = DISTANCE_ROOT / "gmm_optimal_k"


# =========================================================
# HELPERS
# =========================================================
def conformal_quantile_index(n: int, coverage: float) -> int:
    """
    Split-conformal finite-sample quantile index.
        k   = ceil((n + 1) * coverage)
        idx = k - 1  (zero-based)
    Clamped to [0, n-1].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    k = math.ceil((n + 1) * coverage)
    k = min(max(k, 1), n)
    return k - 1


def load_distance_files(distance_dir: Path):
    files = sorted(distance_dir.glob("*.pt"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt distance files found in: {distance_dir}")
    return files


def compute_quantile_maps(distance_dir: Path, distance_name: str):
    files = load_distance_files(distance_dir)
    n     = len(files)

    print(f"\nCalibrating : {distance_name}")
    print(f"Directory   : {distance_dir}")
    print(f"N calibration samples: {n}")

    # Stack all samples -> [N, 128, 128, 31]
    all_distances = torch.empty(
        (n, FULL_H, FULL_W, FULL_T),
        dtype=DTYPE,
        device=DEVICE,
    )

    valid_files = []
    for i, path in enumerate(tqdm(files, desc=f"Loading {distance_name}")):
        x = torch.load(path, map_location=DEVICE).to(DTYPE)
        if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
            raise ValueError(f"Unexpected shape in {path}: {tuple(x.shape)}")
        all_distances[i] = x
        valid_files.append(path.name)

    # =========================================================
    # Conformal quantile maps — one per coverage level
    # =========================================================
    quantile_maps = {}

    for q_name, coverage in TARGET_COVERAGES.items():
        idx   = conformal_quantile_index(n, coverage)
        q_map = torch.kthvalue(all_distances, k=idx + 1, dim=0).values.cpu()

        quantile_maps[q_name] = q_map

        save_path = SAVE_ROOT / f"{distance_name}_{q_name}.pt"
        torch.save(q_map.float(), save_path)

        print(
            f"  {q_name}: coverage={coverage}, n={n}, "
            f"zero_idx={idx}, one_idx={idx + 1} -> {save_path}"
        )

    # =========================================================
    # Stacked quantile tensor — shape [3, 128, 128, 31]
    # Order: q6827, q9545, q9973
    # =========================================================
    stacked = torch.stack(
        [quantile_maps["q6827"], quantile_maps["q9545"], quantile_maps["q9973"]],
        dim=0,
    )
    stacked_path = SAVE_ROOT / f"{distance_name}_quantiles_q6827_q9545_q9973.pt"
    torch.save(stacked.float(), stacked_path)

    # =========================================================
    # Empirical min/max calibration maps — shape [128, 128, 31]
    # =========================================================
    calib_min_map = all_distances.min(dim=0).values.cpu()
    calib_max_map = all_distances.max(dim=0).values.cpu()

    calib_min_path = SAVE_ROOT / f"{distance_name}_calib_min_map.pt"
    calib_max_path = SAVE_ROOT / f"{distance_name}_calib_max_map.pt"
    torch.save(calib_min_map.float(), calib_min_path)
    torch.save(calib_max_map.float(), calib_max_path)

    print(f"  min map -> {calib_min_path}")
    print(f"  max map -> {calib_max_path}")

    # =========================================================
    # Summary CSVs
    # =========================================================
    summary_rows = []
    for q_name, coverage in TARGET_COVERAGES.items():
        idx = conformal_quantile_index(n, coverage)
        summary_rows.append({
            "distance":                  distance_name,
            "num_calibration_samples":   n,
            "target":                    q_name,
            "coverage":                  coverage,
            "conformal_index_zero_based": idx,
            "conformal_index_one_based":  idx + 1,
            "quantile_path":             str(SAVE_ROOT / f"{distance_name}_{q_name}.pt"),
        })
    pd.DataFrame(summary_rows).to_csv(
        SAVE_ROOT / f"{distance_name}_quantile_summary.csv", index=False
    )

    pd.DataFrame({"distance_file": valid_files}).to_csv(
        SAVE_ROOT / f"{distance_name}_calibration_files_used.csv", index=False
    )

    pd.DataFrame([{
        "distance":                distance_name,
        "num_calibration_samples": n,
        "calib_min_path":          str(calib_min_path),
        "calib_max_path":          str(calib_max_path),
        "stacked_quantile_path":   str(stacked_path),
    }]).to_csv(
        SAVE_ROOT / f"{distance_name}_calibration_range_summary.csv", index=False
    )

    print(f"  stacked  -> {stacked_path}")
    print(f"  CSVs saved.")

    del all_distances


# =========================================================
# MAIN
# =========================================================
print(f"Save root: {SAVE_ROOT}")
print(f"Distance types to calibrate: {list(DISTANCE_DIRS.keys())}")

for distance_name, distance_dir in DISTANCE_DIRS.items():
    if not distance_dir.exists():
        print(f"\nSkipping {distance_name}: directory not found ({distance_dir})")
        continue
    compute_quantile_maps(distance_dir, distance_name)

print(f"\nAll GMM conformal quantile maps saved to: {SAVE_ROOT.resolve()}")
