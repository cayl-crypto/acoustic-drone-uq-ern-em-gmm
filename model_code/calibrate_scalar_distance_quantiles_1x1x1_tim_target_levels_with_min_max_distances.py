#!/usr/bin/env python3
# calibrate_scalar_distance_quantiles_1x1x1.py

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import math

# =========================================================
# PATHS
# =========================================================
DISTANCE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model"
)

DISTANCE_DIRS = {
    "mahalanobis": DISTANCE_ROOT / "mahalanobis",
}

SAVE_ROOT = (
    DISTANCE_ROOT
    / "conformal_quantiles_1x1x1_tim_coverage_targets_with_min_max_distances"
)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
DTYPE = torch.float32

FULL_H = 128
FULL_W = 128
FULL_T = 31

TARGET_COVERAGES = {
    "q6827": 0.6827,
    "q9545": 0.9545,
    "q9973": 0.9973,
}


# =========================================================
# HELPERS
# =========================================================
def conformal_quantile_index(n: int, coverage: float) -> int:
    """
    Split-conformal finite-sample quantile index.

    For target coverage 1-alpha:
        k = ceil((n + 1) * coverage)

    Converted to zero-based index:
        idx = k - 1

    If k > n, clamp to n - 1.
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
    n = len(files)

    print(f"\nCalibrating: {distance_name}")
    print(f"Distance directory : {distance_dir}")
    print(f"Calibration samples: {n}")

    # Shape: [N, 128, 128, 31]
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

    # =====================================================
    # Conformal quantile maps
    # =====================================================
    quantile_maps = {}

    for q_name, coverage in TARGET_COVERAGES.items():
        idx = conformal_quantile_index(n, coverage)

        # kthvalue uses 1-based k.
        q_map = torch.kthvalue(
            all_distances,
            k=idx + 1,
            dim=0,
        ).values

        quantile_maps[q_name] = q_map.cpu()

        save_path = SAVE_ROOT / f"{distance_name}_{q_name}.pt"
        torch.save(q_map.float().cpu(), save_path)

        print(
            f"{distance_name} {q_name}: "
            f"coverage={coverage}, n={n}, "
            f"zero_based_idx={idx}, one_based_idx={idx + 1}, "
            f"saved={save_path}"
        )

    # =====================================================
    # Optional stacked quantile map
    # Shape: [3, 128, 128, 31]
    # Order: q6827, q9545, q9973
    # =====================================================
    stacked = torch.stack(
        [
            quantile_maps["q6827"],
            quantile_maps["q9545"],
            quantile_maps["q9973"],
        ],
        dim=0,
    )

    stacked_path = SAVE_ROOT / f"{distance_name}_quantiles_q6827_q9545_q9973.pt"
    torch.save(stacked.float().cpu(), stacked_path)

    # =====================================================
    # Empirical scalar min/max calibration maps
    # Shape: [128, 128, 31]
    # =====================================================
    calib_min_map = all_distances.min(dim=0).values.cpu()
    calib_max_map = all_distances.max(dim=0).values.cpu()

    calib_min_path = SAVE_ROOT / f"{distance_name}_calib_min_map.pt"
    calib_max_path = SAVE_ROOT / f"{distance_name}_calib_max_map.pt"

    torch.save(calib_min_map.float(), calib_min_path)
    torch.save(calib_max_map.float(), calib_max_path)

    print(f"Saved scalar calibration minimum map: {calib_min_path}")
    print(f"Saved scalar calibration maximum map: {calib_max_path}")

    # =====================================================
    # Save metadata
    # =====================================================
    summary_rows = []

    for q_name, coverage in TARGET_COVERAGES.items():
        idx = conformal_quantile_index(n, coverage)

        summary_rows.append({
            "distance": distance_name,
            "num_calibration_samples": n,
            "target": q_name,
            "coverage": coverage,
            "conformal_index_zero_based": idx,
            "conformal_index_one_based": idx + 1,
            "quantile_path": str(SAVE_ROOT / f"{distance_name}_{q_name}.pt"),
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = SAVE_ROOT / f"{distance_name}_quantile_summary.csv"
    summary.to_csv(summary_path, index=False)

    index_df = pd.DataFrame({
        "distance_file": valid_files,
    })

    index_path = SAVE_ROOT / f"{distance_name}_calibration_files_used.csv"
    index_df.to_csv(index_path, index=False)

    extra_summary = pd.DataFrame([
        {
            "distance": distance_name,
            "num_calibration_samples": n,
            "calib_min_path": str(calib_min_path),
            "calib_max_path": str(calib_max_path),
            "stacked_quantile_path": str(stacked_path),
        }
    ])

    extra_summary_path = SAVE_ROOT / f"{distance_name}_calibration_range_summary.csv"
    extra_summary.to_csv(extra_summary_path, index=False)

    print(f"Saved stacked quantiles: {stacked_path}")
    print(f"Saved summary         : {summary_path}")
    print(f"Saved file index      : {index_path}")
    print(f"Saved range summary   : {extra_summary_path}")

    del all_distances


# =========================================================
# MAIN
# =========================================================
for distance_name, distance_dir in DISTANCE_DIRS.items():
    compute_quantile_maps(distance_dir, distance_name)

print(f"\nAll conformal quantile maps saved to: {SAVE_ROOT.resolve()}")