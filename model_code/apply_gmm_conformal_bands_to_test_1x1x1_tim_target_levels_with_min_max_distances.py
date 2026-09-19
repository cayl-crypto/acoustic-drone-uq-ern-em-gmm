#!/usr/bin/env python3
# apply_gmm_conformal_bands_to_test_1x1x1_tim_target_levels_with_min_max_distances.py
#
# Applies GMM NLL conformal bands to all 4 test conditions.
# For each test sample and each distance type (gmm_k1..k10, gmm_optimal_k):
#   - Computes NLL distance map [128, 128, 31] from raw features
#   - Assigns conformal bands (low/med/high/very_high) using calibrated quantiles
#   - Computes range exceedance vs calibration min/max
#   - Saves band map as .pt + summary CSVs
#
# Band encoding (6 levels):
#   0  below_min   NLL < calib_min
#   1  low         calib_min <= NLL <= q6827
#   2  med         q6827 < NLL <= q9545
#   3  high        q9545 < NLL <= q9973
#   4  very_high   q9973 < NLL <= calib_max
#   5  above_max   NLL > calib_max

from pathlib import Path
import math
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
TEST_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/"
    "test_dataset_with_nondrone_with_9600_flag.csv"
)

FEATURE_DIRS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB":  Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB":   Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB":  Path("/path/to/DDLDataset/-5dB/features/model"),
}

GMM_AGG_DIR = Path(
    "/path/to/DDLDataset/plain/distances/model"
    "/gmm_scalar_aggregate_1x1x1"
)

QUANTILE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model"
    "/gmm_conformal_quantiles_1x1x1_tim_coverage_targets_with_min_max_distances"
)

SAVE_ROOT = Path(
    "/path/to/DDLDataset/conformal_bands/test/model"
    "/gmm_1x1x1"
)

DEVICE = "cpu"
FULL_H = 128
FULL_W  = 128
FULL_T  = 31
MAX_K   = 10
EPS     = 1e-8
LOG2PI  = math.log(2.0 * math.pi)

DISTANCE_NAMES = [f"gmm_k{k}" for k in range(1, MAX_K + 1)] + ["gmm_optimal_k"]


# =========================================================
# HELPERS
# =========================================================
def normalize_id(x):
    digits = "".join(ch for ch in str(x).strip() if ch.isdigit())
    return digits.zfill(8)


def build_file_index(folder: Path):
    lookup = {}
    for p in sorted(folder.glob("*.pt")):
        file_id = p.stem.split("_")[0]
        lookup[normalize_id(file_id)] = p
    return lookup


def compute_nll(x, weights, means, variances):
    """
    x         : (H, W, T)      float32
    weights   : (H, W, T, k)   float32
    means     : (H, W, T, k)   float32
    variances : (H, W, T, k)   float32
    Returns nll : (H, W, T)    float32
    """
    x_e       = x.unsqueeze(-1)
    diff      = x_e - means
    log_gauss = -0.5 * (
        LOG2PI
        + torch.log(variances.clamp(min=EPS))
        + diff ** 2 / variances.clamp(min=EPS)
    )
    log_joint = torch.log(weights.clamp(min=EPS)) + log_gauss
    log_p     = torch.logsumexp(log_joint, dim=-1)
    return -log_p


def assign_bands(distance_map, q6827, q9545, q9973, calib_min_map, calib_max_map):
    """
    6-level band encoding per scalar position:
        0  below_min   NLL < calib_min
        1  low         calib_min <= NLL <= q6827
        2  med         q6827 < NLL <= q9545
        3  high        q9545 < NLL <= q9973
        4  very_high   q9973 < NLL <= calib_max
        5  above_max   NLL > calib_max
    """
    band_map = torch.ones_like(distance_map, dtype=torch.uint8)  # default: low (1)
    band_map[distance_map < calib_min_map] = 0  # below_min
    band_map[distance_map > q6827]         = 2  # med
    band_map[distance_map > q9545]         = 3  # high
    band_map[distance_map > q9973]         = 4  # very_high
    band_map[distance_map > calib_max_map] = 5  # above_max
    return band_map


def summarize_band_map(band_map):
    vals  = band_map.reshape(-1)
    total = vals.numel()
    below_min = int((vals == 0).sum())
    low       = int((vals == 1).sum())
    med       = int((vals == 2).sum())
    high      = int((vals == 3).sum())
    very_high = int((vals == 4).sum())
    above_max = int((vals == 5).sum())
    return {
        "num_positions":   total,
        "below_min_count": below_min,
        "low_count":       low,
        "med_count":       med,
        "high_count":      high,
        "very_high_count": very_high,
        "above_max_count": above_max,
        "below_min_pct":   100.0 * below_min / total,
        "low_pct":         100.0 * low       / total,
        "med_pct":         100.0 * med       / total,
        "high_pct":        100.0 * high      / total,
        "very_high_pct":   100.0 * very_high / total,
        "above_max_pct":   100.0 * above_max / total,
        "ood_pct":         100.0 * (below_min + above_max) / total,
    }




# =========================================================
# LOAD GMM MODELS
# =========================================================
print("Loading GMM aggregate models ...")
gmm_models = {}
for k in range(1, MAX_K + 1):
    gmm_models[k] = {
        "weights":   torch.load(GMM_AGG_DIR / f"weights_k{k:02d}.pt",   map_location=DEVICE).float(),
        "means":     torch.load(GMM_AGG_DIR / f"means_k{k:02d}.pt",     map_location=DEVICE).float(),
        "variances": torch.load(GMM_AGG_DIR / f"variances_k{k:02d}.pt", map_location=DEVICE).float(),
    }
    print(f"  k={k:02d}  weights={tuple(gmm_models[k]['weights'].shape)}")

optimal_k_map = torch.load(GMM_AGG_DIR / "optimal_k.pt", map_location=DEVICE).int()
print(f"Loaded optimal_k_map: {tuple(optimal_k_map.shape)}")


# =========================================================
# LOAD CONFORMAL QUANTILES + MIN/MAX MAPS
# =========================================================
print("\nLoading conformal quantiles and calibration ranges ...")
quantiles    = {}
calib_ranges = {}

for dist_name in DISTANCE_NAMES:
    q6827 = torch.load(QUANTILE_ROOT / f"{dist_name}_q6827.pt", map_location=DEVICE).float()
    q9545 = torch.load(QUANTILE_ROOT / f"{dist_name}_q9545.pt", map_location=DEVICE).float()
    q9973 = torch.load(QUANTILE_ROOT / f"{dist_name}_q9973.pt", map_location=DEVICE).float()

    for q_name, q in [("q6827", q6827), ("q9545", q9545), ("q9973", q9973)]:
        if tuple(q.shape) != (FULL_H, FULL_W, FULL_T):
            raise ValueError(f"Unexpected {q_name} shape for {dist_name}: {tuple(q.shape)}")

    quantiles[dist_name] = {"q6827": q6827, "q9545": q9545, "q9973": q9973}

    calib_min = torch.load(QUANTILE_ROOT / f"{dist_name}_calib_min_map.pt", map_location=DEVICE).float()
    calib_max = torch.load(QUANTILE_ROOT / f"{dist_name}_calib_max_map.pt", map_location=DEVICE).float()
    calib_ranges[dist_name] = {"min": calib_min, "max": calib_max}

    print(f"  {dist_name}: quantiles + min/max loaded")


# =========================================================
# LOAD TEST IDS
# =========================================================
test_df = pd.read_csv(TEST_CSV_PATH)
test_df["ID"] = test_df["ID"].apply(normalize_id)
test_ids = test_df["ID"].tolist()

meta_cols = [c for c in ["ID", "label", "Label", "split", "conformal_role", "noise_level"]
             if c in test_df.columns]

print(f"\nTotal test samples: {len(test_ids)}")


# =========================================================
# APPLY BANDS TO TEST SET — ALL CONDITIONS
# =========================================================
all_summary_rows = []

for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")

    file_lookup = build_file_index(feature_dir)
    print(f"  Feature files found: {len(file_lookup)}")

    # Create output directories
    for dist_name in DISTANCE_NAMES:
        (SAVE_ROOT / condition / dist_name / "pt").mkdir(parents=True, exist_ok=True)

    for _, meta_row in tqdm(test_df.iterrows(), total=len(test_df), desc=condition):
        idx     = meta_row["ID"]
        pt_path = file_lookup.get(idx)

        base_row = {c: meta_row[c] for c in meta_cols}
        base_row.update({
            "condition":       condition,
            "ID":              idx,
            "feature_file":    None,
            "missing_feature": False,
            "error":           None,
        })

        if pt_path is None:
            for dist_name in DISTANCE_NAMES:
                row = dict(base_row)
                row["distance"]        = dist_name
                row["missing_feature"] = True
                all_summary_rows.append(row)
            continue

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()
            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

            # Compute NLL for all k once — reuse across distance names
            nll_maps = {}
            for k in range(1, MAX_K + 1):
                m = gmm_models[k]
                nll_maps[k] = compute_nll(x, m["weights"], m["means"], m["variances"])

            # Build optimal_k NLL map
            nll_optimal = torch.zeros(FULL_H, FULL_W, FULL_T, dtype=torch.float32)
            for k in range(1, MAX_K + 1):
                mask = (optimal_k_map == k)
                if mask.any():
                    nll_optimal[mask] = nll_maps[k][mask]
            nll_maps["gmm_optimal_k"] = nll_optimal

            out_name = pt_path.stem

            for dist_name in DISTANCE_NAMES:
                # Retrieve precomputed NLL
                if dist_name == "gmm_optimal_k":
                    distance_map = nll_maps["gmm_optimal_k"]
                else:
                    k = int(dist_name.split("gmm_k")[1])
                    distance_map = nll_maps[k]

                q         = quantiles[dist_name]
                calib_min = calib_ranges[dist_name]["min"]
                calib_max = calib_ranges[dist_name]["max"]

                band_map = assign_bands(
                    distance_map,
                    q["q6827"], q["q9545"], q["q9973"],
                    calib_min, calib_max,
                )

                pt_save_path = SAVE_ROOT / condition / dist_name / "pt" / f"{out_name}_bands.pt"
                torch.save(band_map.cpu(), pt_save_path)

                row = dict(base_row)
                row.update(summarize_band_map(band_map))
                row.update({
                    "distance":     dist_name,
                    "feature_file": pt_path.name,
                    "band_pt_path": str(pt_save_path),
                    "q6827_path":   str(QUANTILE_ROOT / f"{dist_name}_q6827.pt"),
                    "q9545_path":   str(QUANTILE_ROOT / f"{dist_name}_q9545.pt"),
                    "q9973_path":   str(QUANTILE_ROOT / f"{dist_name}_q9973.pt"),
                    "calib_min_path": str(QUANTILE_ROOT / f"{dist_name}_calib_min_map.pt"),
                    "calib_max_path": str(QUANTILE_ROOT / f"{dist_name}_calib_max_map.pt"),
                })
                all_summary_rows.append(row)

        except Exception as e:
            for dist_name in DISTANCE_NAMES:
                row = dict(base_row)
                row["distance"]        = dist_name
                row["missing_feature"] = True
                row["error"]           = str(e)
                all_summary_rows.append(row)


# =========================================================
# SAVE SUMMARY CSVs
# =========================================================
summary_df   = pd.DataFrame(all_summary_rows)
summary_path = SAVE_ROOT / "test_conformal_band_summary_per_sample_with_range_exceedance.csv"
summary_df.to_csv(summary_path, index=False)

dataset_summary_cols = [
    "below_min_count", "low_count", "med_count", "high_count", "very_high_count", "above_max_count",
    "below_min_pct",   "low_pct",   "med_pct",   "high_pct",   "very_high_pct",   "above_max_pct",
    "ood_pct",
]

existing_cols = [c for c in dataset_summary_cols if c in summary_df.columns]

dataset_summary = (
    summary_df[summary_df["missing_feature"] == False]
    .groupby(["condition", "distance"])[existing_cols]
    .mean()
    .reset_index()
)

dataset_summary_path = SAVE_ROOT / "test_conformal_band_summary_by_condition_with_range_exceedance.csv"
dataset_summary.to_csv(dataset_summary_path, index=False)

print(f"\nSaved per-sample summary : {summary_path}")
print(f"Saved dataset summary    : {dataset_summary_path}")
print(f"All outputs saved under  : {SAVE_ROOT.resolve()}")
