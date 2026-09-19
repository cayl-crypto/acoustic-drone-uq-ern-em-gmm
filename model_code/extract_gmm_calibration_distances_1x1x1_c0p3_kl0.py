#!/usr/bin/env python3
# extract_gmm_calibration_distances_1x1x1_c0p3_kl0.py
#
# Uses the fitted GMM models (from gmm_scalar_fit_1x1x1_c0p3_kl0.py +
# gmm_scalar_aggregate_1x1x1_c0p3_kl0.py) to compute per-sample NLL
# distance maps for the calibration split of the val set (plain condition).
#
# For each calibration sample and each k in {1, ..., MAX_K}:
#   - Computes NLL(x | GMM_k) at every scalar position -> shape [128, 128, 31]
#   - Saves the distance tensor to:
#       .../plain/distances/model_c0p3_kl0/gmm_k{k}/{sample_id}.pt
#
# Also computes distances using the per-position optimal k (from optimal_k_map.pt):
#   - Saves to: .../plain/distances/model_c0p3_kl0/gmm_optimal_k/{sample_id}.pt
#
# Calibration rows come from the val CSV's "conformal_role" == "calibration"
# column across all noise_level values, but since the feature dir only has
# plain-condition files, only plain-tagged rows will actually match — same
# no-explicit-filter approach as extract_training_scalars_1x1x1_c0p3_kl0.py.
#
# These distance tensors will be consumed by
# calibrate_gmm_scalar_distance_quantiles_1x1x1_c0p3_kl0.py next.

from pathlib import Path
import math
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
VAL_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/"
    "val_dataset_with_nondrone_with_9600_flag.csv"
)
PLAIN_FEATURE_DIR = Path(
    "/path/to/DDLDataset/plain/features/model_c0p3_kl0"
)

GMM_ROOT    = Path("./gmm_scalar_1x1x1_c0p3_kl0")
GMM_AGG_DIR = GMM_ROOT / "aggregate"

DISTANCE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model_c0p3_kl0"
)

DEVICE = "cpu"
FULL_H = 128
FULL_W = 128
FULL_T = 31
MAX_K  = 10
EPS    = 1e-8
LOG2PI = math.log(2.0 * math.pi)


# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index: str):
    matches = sorted(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None


def compute_nll(x, weights, means, variances):
    """
    Args:
        x         : (H, W, T)      float32
        weights   : (H, W, T, k)   float32
        means     : (H, W, T, k)   float32
        variances : (H, W, T, k)   float32

    Returns:
        nll : (H, W, T) float32  — negative log-likelihood under the GMM
    """
    x_e      = x.unsqueeze(-1)                                          # (H, W, T, 1)
    diff     = x_e - means                                              # (H, W, T, k)
    log_gauss = -0.5 * (
        LOG2PI
        + torch.log(variances.clamp(min=EPS))
        + diff ** 2 / variances.clamp(min=EPS)
    )                                                                    # (H, W, T, k)
    log_joint = torch.log(weights.clamp(min=EPS)) + log_gauss           # (H, W, T, k)
    log_p     = torch.logsumexp(log_joint, dim=-1)                      # (H, W, T)
    return -log_p


# =========================================================
# STEP 1: LOAD GMM MODELS FOR ALL k INTO MEMORY
# =========================================================
print("Loading GMM models ...")
gmm_models = {}
for k in range(1, MAX_K + 1):
    gmm_models[k] = {
        "weights":   torch.load(GMM_AGG_DIR / f"weights_k{k:02d}.pt",   map_location=DEVICE).float(),
        "means":     torch.load(GMM_AGG_DIR / f"means_k{k:02d}.pt",     map_location=DEVICE).float(),
        "variances": torch.load(GMM_AGG_DIR / f"variances_k{k:02d}.pt", map_location=DEVICE).float(),
    }
    print(f"  Loaded k={k}  weights shape: {tuple(gmm_models[k]['weights'].shape)}")

optimal_k_map = torch.load(
    GMM_AGG_DIR / "optimal_k.pt", map_location=DEVICE
).int()  # (H, W, T)
print(f"Loaded optimal_k_map shape: {tuple(optimal_k_map.shape)}")

# =========================================================
# STEP 2: LOAD CALIBRATION IDS
# =========================================================
val_df = pd.read_csv(VAL_CSV_PATH)
val_df["ID"] = val_df["ID"].astype(str)

if "conformal_role" not in val_df.columns:
    raise KeyError("val CSV does not contain column: conformal_role")

calib_df  = val_df[val_df["conformal_role"] == "calibration"].copy()
calib_ids = calib_df["ID"].tolist()

print(f"\nTotal val rows (all noise_level values) : {len(val_df)}")
print(f"Calibration rows (all noise_level values): {len(calib_df)}")

# =========================================================
# STEP 3: SET UP OUTPUT DIRECTORIES
# =========================================================
gmm_distance_dirs = {}
for k in range(1, MAX_K + 1):
    d = DISTANCE_ROOT / f"gmm_k{k}"
    d.mkdir(parents=True, exist_ok=True)
    gmm_distance_dirs[k] = d

optimal_k_dir = DISTANCE_ROOT / "gmm_optimal_k"
optimal_k_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# STEP 4: EXTRACT AND SAVE DISTANCE TENSORS
# =========================================================
rows = []

for idx in tqdm(calib_ids, desc="Extracting calibration distances"):
    pt_path = find_file_by_index(PLAIN_FEATURE_DIR, idx)

    row = {
        "ID":              idx,
        "feature_file":    None,
        "missing_feature": False,
        "error":           None,
    }
    for k in range(1, MAX_K + 1):
        row[f"gmm_k{k}_path"] = None
    row["gmm_optimal_k_path"] = None

    if pt_path is None:
        row["missing_feature"] = True
        rows.append(row)
        continue

    row["feature_file"] = pt_path.name

    try:
        x = torch.load(pt_path, map_location=DEVICE).float()

        if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
            raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

        out_name = pt_path.name   # preserves original filename e.g. 00009689.pt

        # --- Distance for each k ---
        for k in range(1, MAX_K + 1):
            m = gmm_models[k]
            nll = compute_nll(x, m["weights"], m["means"], m["variances"])
            save_path = gmm_distance_dirs[k] / out_name
            torch.save(nll.cpu(), save_path)
            row[f"gmm_k{k}_path"] = str(save_path)

        # --- Distance using per-position optimal k ---
        nll_opt = torch.zeros(FULL_H, FULL_W, FULL_T, dtype=torch.float32)
        for k in range(1, MAX_K + 1):
            mask = (optimal_k_map == k)
            if not mask.any():
                continue
            m   = gmm_models[k]
            nll = compute_nll(x, m["weights"], m["means"], m["variances"])
            nll_opt[mask] = nll[mask]

        opt_save_path = optimal_k_dir / out_name
        torch.save(nll_opt.cpu(), opt_save_path)
        row["gmm_optimal_k_path"] = str(opt_save_path)

    except Exception as e:
        row["missing_feature"] = True
        row["error"] = str(e)

    rows.append(row)

# =========================================================
# STEP 5: SAVE INDEX + SUMMARY
# =========================================================
index_df = pd.DataFrame(rows)
index_path = DISTANCE_ROOT / "gmm_calibration_distance_index.csv"
index_df.to_csv(index_path, index=False)

n_saved   = int((index_df["missing_feature"] == False).sum())
n_missing = int(index_df["missing_feature"].sum())

summary_df = pd.DataFrame([{
    "split":                "calibration",
    "condition":            "plain",
    "num_requested":        len(calib_ids),
    "num_saved":            n_saved,
    "num_missing_or_error": n_missing,
    "distance_shape":       f"({FULL_H}, {FULL_W}, {FULL_T})",
    "max_k":                MAX_K,
    "gmm_agg_dir":          str(GMM_AGG_DIR.resolve()),
    "distance_root":        str(DISTANCE_ROOT.resolve()),
}])
summary_path = DISTANCE_ROOT / "gmm_calibration_distance_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nCalibration samples requested : {len(calib_ids)}")
print(f"Saved successfully            : {n_saved}")
print(f"Missing / errors              : {n_missing}")
print(f"Saved index  -> {index_path}")
print(f"Saved summary-> {summary_path}")
for k in range(1, MAX_K + 1):
    print(f"  gmm_k{k:<2d} dir -> {gmm_distance_dirs[k]}")
print(f"  gmm_optimal_k dir -> {optimal_k_dir}")
