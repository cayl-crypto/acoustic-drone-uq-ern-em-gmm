#!/usr/bin/env python3
# extract_plain_calibration_distances_1x1x1.py

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
VAL_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/val_dataset_with_nondrone_with_9600_flag.csv"
)

PLAIN_FEATURE_DIR = Path(
    "/path/to/DDLDataset/plain/features/model"
)

SCALAR_MODEL_ROOT = Path("./scalar_models_1x1x1")

DISTANCE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model"
)

MANHATTAN_DIR = DISTANCE_ROOT / "manhattan"
EUCLIDEAN_DIR = DISTANCE_ROOT / "euclidean"
MAHALANOBIS_DIR = DISTANCE_ROOT / "mahalanobis"

for d in [MANHATTAN_DIR, EUCLIDEAN_DIR, MAHALANOBIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"

FULL_H = 128
FULL_W = 128
FULL_T = 31
EPS = 1e-6


# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index: str):
    matches = sorted(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None


# =========================================================
# LOAD SCALAR MEAN / STD
# =========================================================
mu_path = SCALAR_MODEL_ROOT / "mu.pt"
std_path = SCALAR_MODEL_ROOT / "std.pt"

if not mu_path.exists():
    raise FileNotFoundError(f"Missing mean tensor: {mu_path}")

if not std_path.exists():
    raise FileNotFoundError(f"Missing std tensor: {std_path}")

mu = torch.load(mu_path, map_location=DEVICE).float()
std = torch.load(std_path, map_location=DEVICE).float()
std = torch.clamp(std, min=EPS)

if tuple(mu.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected mu shape: {tuple(mu.shape)}")

if tuple(std.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected std shape: {tuple(std.shape)}")

print(f"Loaded mu : {mu_path}")
print(f"Loaded std: {std_path}")


# =========================================================
# LOAD CALIBRATION IDS
# =========================================================
val_df = pd.read_csv(VAL_CSV_PATH)
val_df["ID"] = val_df["ID"].astype(str)

if "conformal_role" not in val_df.columns:
    raise KeyError("CSV does not contain column: conformal_role")

calib_df = val_df[val_df["conformal_role"] == "calibration"].copy()
calib_ids = calib_df["ID"].astype(str).tolist()

print(f"Total validation rows     : {len(val_df)}")
print(f"Calibration rows selected : {len(calib_df)}")


# =========================================================
# EXTRACT AND SAVE DISTANCE TENSORS
# =========================================================
rows = []

for idx in tqdm(calib_ids, desc="Extracting plain calibration distances"):
    pt_path = find_file_by_index(PLAIN_FEATURE_DIR, idx)

    row = {
        "ID": idx,
        "feature_file": None,
        "missing_feature": False,
        "error": None,
        "manhattan_path": None,
        "euclidean_path": None,
        "mahalanobis_path": None,
    }

    if pt_path is None:
        row["missing_feature"] = True
        rows.append(row)
        continue

    row["feature_file"] = pt_path.name

    try:
        x = torch.load(pt_path, map_location=DEVICE).float()

        if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
            raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

        diff = x - mu

        manhattan = torch.abs(diff)
        euclidean = torch.abs(diff)
        mahalanobis = torch.abs(diff) / std

        out_name = pt_path.name

        manhattan_path = MANHATTAN_DIR / out_name
        euclidean_path = EUCLIDEAN_DIR / out_name
        mahalanobis_path = MAHALANOBIS_DIR / out_name

        torch.save(manhattan.float().cpu(), manhattan_path)
        torch.save(euclidean.float().cpu(), euclidean_path)
        torch.save(mahalanobis.float().cpu(), mahalanobis_path)

        row["manhattan_path"] = str(manhattan_path)
        row["euclidean_path"] = str(euclidean_path)
        row["mahalanobis_path"] = str(mahalanobis_path)

    except Exception as e:
        row["missing_feature"] = True
        row["error"] = str(e)

    rows.append(row)


# =========================================================
# SAVE INDEX FILE
# =========================================================
index_df = pd.DataFrame(rows)

index_path = DISTANCE_ROOT / "calibration_distance_index.csv"
index_df.to_csv(index_path, index=False)

summary = pd.DataFrame([{
    "split": "calibration",
    "condition": "plain",
    "num_requested": len(calib_ids),
    "num_missing_or_error": int(index_df["missing_feature"].sum()),
    "num_saved": int((index_df["missing_feature"] == False).sum()),
    "distance_shape": f"({FULL_H}, {FULL_W}, {FULL_T})",
    "manhattan_dir": str(MANHATTAN_DIR),
    "euclidean_dir": str(EUCLIDEAN_DIR),
    "mahalanobis_dir": str(MAHALANOBIS_DIR),
}])

summary_path = DISTANCE_ROOT / "calibration_distance_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"\nSaved distance index  : {index_path}")
print(f"Saved distance summary: {summary_path}")
print(f"Manhattan dir         : {MANHATTAN_DIR}")
print(f"Euclidean dir         : {EUCLIDEAN_DIR}")
print(f"Mahalanobis dir       : {MAHALANOBIS_DIR}")