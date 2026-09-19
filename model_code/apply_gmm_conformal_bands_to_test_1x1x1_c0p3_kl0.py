#!/usr/bin/env python3
# apply_gmm_conformal_bands_to_test_1x1x1_c0p3_kl0.py
#
# Applies GMM NLL conformal bands to all 8 test conditions, for the
# c0p3_kl0 checkpoint:
#   plain, awgn_15dB, awgn_5dB, awgn_-5dB,
#   clutter_15dB, clutter_5dB, clutter_-5dB, random_noise
#
# optimal_k ONLY (not gmm_k1..gmm_k10) — same reasoning and same
# gather-optimization as gmm_scalar_distance_eval_c0p3_kl0.py /
# gmm_test_noise_distance_eval_c0p3_kl0.py: the per-position optimal-k GMM
# parameters are gathered into one padded (H, W, T, MAX_K) tensor once at
# startup (via torch.cat + torch.where -- NOT the `tensor[mask, :k] = ...`
# combined-indexing pattern, which computes the wrong assignment shape in
# PyTorch, as we hit and fixed in the eval scripts), so each sample needs
# only one NLL pass. The calibration quantile/min-max maps for gmm_optimal_k
# already exist on disk from calibrate_gmm_scalar_distance_quantiles_1x1x1_
# c0p3_kl0.py (which calibrated all 11 distance types) -- reused as-is here,
# nothing needs recomputing there.
#
# Sharded as an 8-way array job, one condition per task (same split as
# gmm_test_noise_distance_eval_c0p3_kl0.py), each writing its own
# per-condition CSVs rather than one shared file, since 8 concurrent array
# tasks writing to the same CSV would race.
#
# Band encoding (6 levels):
#   0  below_min   NLL < calib_min
#   1  low         calib_min <= NLL <= q6827
#   2  med         q6827 < NLL <= q9545
#   3  high        q9545 < NLL <= q9973
#   4  very_high   q9973 < NLL <= calib_max
#   5  above_max   NLL > calib_max
#
# random_noise has no annotation CSV / real IDs, evaluated directly from
# whatever files exist in its feature dir, same as in the eval scripts.
#
# Requires gmm_scalar_aggregate_1x1x1_c0p3_kl0.py,
# calibrate_gmm_scalar_distance_quantiles_1x1x1_c0p3_kl0.py,
# extract_features_cnn0_c0p3_kl0.py (plain), and
# extract_features_cnn0_test_noise_types_c0p3_kl0.py (the other 7) to have
# completed first.

import argparse
from pathlib import Path
import math
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# ARGS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, required=True,
                    help="Job index 0-7. Selects which condition this job processes.")
args = parser.parse_args()

CONDITIONS = [
    "plain",
    "awgn_15dB", "awgn_5dB", "awgn_-5dB",
    "clutter_15dB", "clutter_5dB", "clutter_-5dB",
    "random_noise",
]
assert 0 <= args.job_id < len(CONDITIONS), f"job_id must be 0-{len(CONDITIONS)-1}, got {args.job_id}"
CONDITION = CONDITIONS[args.job_id]

# =========================================================
# PATHS
# =========================================================
CSV_ROOT = Path("/path/to/DDLDataset")
TEST_CSV_PATH = CSV_ROOT / "annotations" / "test_dataset_with_nondrone_with_9600_flag.csv"

FEATURE_DIR = CSV_ROOT / CONDITION / "features" / "model_c0p3_kl0"

GMM_ROOT    = Path("./gmm_scalar_1x1x1_c0p3_kl0")
GMM_AGG_DIR = GMM_ROOT / "aggregate"

QUANTILE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model_c0p3_kl0"
    "/gmm_conformal_quantiles_1x1x1_c0p3_kl0_coverage_targets_with_min_max_distances"
)

SAVE_ROOT = Path(
    "/path/to/DDLDataset/conformal_bands/test/model_c0p3_kl0"
    "/gmm_1x1x1"
)

DEVICE = "cpu"
FULL_H = 128
FULL_W  = 128
FULL_T  = 31
MAX_K   = 10
EPS     = 1e-8
LOG2PI  = math.log(2.0 * math.pi)
DIST_NAME = "gmm_optimal_k"

print(f"Job ID    : {args.job_id}")
print(f"Condition : {CONDITION}")


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
    x: (H, W, T); weights/means/variances: (H, W, T, MAX_K) -- already
    gathered to each position's optimal-k parameters, zero-padded.
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
# LOAD GMM MODELS FOR ALL k (needed to build the gathered model)
# =========================================================
print("Loading GMM aggregate models ...")
gmm_models = {}
for k in range(1, MAX_K + 1):
    gmm_models[k] = {
        "weights":   torch.load(GMM_AGG_DIR / f"weights_k{k:02d}.pt",   map_location=DEVICE).float(),
        "means":     torch.load(GMM_AGG_DIR / f"means_k{k:02d}.pt",     map_location=DEVICE).float(),
        "variances": torch.load(GMM_AGG_DIR / f"variances_k{k:02d}.pt", map_location=DEVICE).float(),
    }

optimal_k_map = torch.load(GMM_AGG_DIR / "optimal_k.pt", map_location=DEVICE).int()
print(f"Loaded optimal_k_map: {tuple(optimal_k_map.shape)}")

# =========================================================
# BUILD THE GATHERED OPTIMAL-K MODEL ONCE
# =========================================================
print("Building gathered optimal-k GMM (enables single-pass NLL per sample)...")
gathered_w = torch.zeros(FULL_H, FULL_W, FULL_T, MAX_K, dtype=torch.float32)
gathered_m = torch.zeros(FULL_H, FULL_W, FULL_T, MAX_K, dtype=torch.float32)
gathered_v = torch.ones(FULL_H, FULL_W, FULL_T, MAX_K, dtype=torch.float32)

for k in range(1, MAX_K + 1):
    mask = (optimal_k_map == k)
    if not mask.any():
        continue
    mask4 = mask.unsqueeze(-1)  # (H,W,T,1), broadcasts against the last dim

    pad_shape = (FULL_H, FULL_W, FULL_T, MAX_K - k)
    w_padded = torch.cat([gmm_models[k]["weights"],   torch.zeros(pad_shape)], dim=-1)
    m_padded = torch.cat([gmm_models[k]["means"],     torch.zeros(pad_shape)], dim=-1)
    v_padded = torch.cat([gmm_models[k]["variances"], torch.ones(pad_shape)],  dim=-1)

    gathered_w = torch.where(mask4, w_padded, gathered_w)
    gathered_m = torch.where(mask4, m_padded, gathered_m)
    gathered_v = torch.where(mask4, v_padded, gathered_v)

del gmm_models
print("Gathered optimal-k GMM built.")

# =========================================================
# LOAD CONFORMAL QUANTILES + MIN/MAX MAPS (gmm_optimal_k only)
# =========================================================
print("\nLoading conformal quantiles and calibration ranges (gmm_optimal_k)...")
q6827 = torch.load(QUANTILE_ROOT / f"{DIST_NAME}_q6827.pt", map_location=DEVICE).float()
q9545 = torch.load(QUANTILE_ROOT / f"{DIST_NAME}_q9545.pt", map_location=DEVICE).float()
q9973 = torch.load(QUANTILE_ROOT / f"{DIST_NAME}_q9973.pt", map_location=DEVICE).float()
calib_min = torch.load(QUANTILE_ROOT / f"{DIST_NAME}_calib_min_map.pt", map_location=DEVICE).float()
calib_max = torch.load(QUANTILE_ROOT / f"{DIST_NAME}_calib_max_map.pt", map_location=DEVICE).float()

for q_name, q in [("q6827", q6827), ("q9545", q9545), ("q9973", q9973)]:
    if tuple(q.shape) != (FULL_H, FULL_W, FULL_T):
        raise ValueError(f"Unexpected {q_name} shape: {tuple(q.shape)}")
print("  quantiles + min/max loaded")

# =========================================================
# SET UP OUTPUT DIR
# =========================================================
(SAVE_ROOT / CONDITION / DIST_NAME / "pt").mkdir(parents=True, exist_ok=True)

# =========================================================
# APPLY BANDS TO THIS JOB'S CONDITION
# =========================================================
if not FEATURE_DIR.exists():
    print(f"\n[WARNING] Feature dir does not exist, nothing to do: {FEATURE_DIR}")
    raise SystemExit(0)

summary_rows = []

if CONDITION == "random_noise":
    # Synthetic samples: no annotation CSV, no real IDs.
    pt_files = sorted(FEATURE_DIR.glob("*.pt"))
    print(f"\n{CONDITION}: {len(pt_files)} synthetic samples found in {FEATURE_DIR}")

    for pt_path in tqdm(pt_files, desc=CONDITION):
        base_row = {
            "ID": pt_path.stem, "condition": CONDITION, "distance": DIST_NAME,
            "feature_file": None, "missing_feature": False, "error": None,
        }
        try:
            x = torch.load(pt_path, map_location=DEVICE).float()
            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

            distance_map = compute_nll(x, gathered_w, gathered_m, gathered_v)
            band_map = assign_bands(distance_map, q6827, q9545, q9973, calib_min, calib_max)

            out_name = pt_path.stem
            pt_save_path = SAVE_ROOT / CONDITION / DIST_NAME / "pt" / f"{out_name}_bands.pt"
            torch.save(band_map.cpu(), pt_save_path)

            row = dict(base_row)
            row["feature_file"] = pt_path.name
            row["band_pt_path"] = str(pt_save_path)
            row.update(summarize_band_map(band_map))
            summary_rows.append(row)

        except Exception as e:
            row = dict(base_row)
            row["missing_feature"] = True
            row["error"] = str(e)
            summary_rows.append(row)

else:
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df["ID"] = test_df["ID"].apply(normalize_id)

    meta_cols = [c for c in ["ID", "label", "Label", "split", "conformal_role", "noise_level"]
                 if c in test_df.columns]

    print(f"\n{CONDITION}: {len(test_df)} test samples (all noise_level values)")

    file_lookup = build_file_index(FEATURE_DIR)
    print(f"  Feature files found: {len(file_lookup)}")

    for _, meta_row in tqdm(test_df.iterrows(), total=len(test_df), desc=CONDITION):
        idx     = meta_row["ID"]
        pt_path = file_lookup.get(idx)

        base_row = {c: meta_row[c] for c in meta_cols}
        base_row.update({
            "condition": CONDITION, "distance": DIST_NAME,
            "feature_file": None, "missing_feature": False, "error": None,
        })

        if pt_path is None:
            row = dict(base_row)
            row["missing_feature"] = True
            summary_rows.append(row)
            continue

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()
            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

            distance_map = compute_nll(x, gathered_w, gathered_m, gathered_v)
            band_map = assign_bands(distance_map, q6827, q9545, q9973, calib_min, calib_max)

            out_name = pt_path.stem
            pt_save_path = SAVE_ROOT / CONDITION / DIST_NAME / "pt" / f"{out_name}_bands.pt"
            torch.save(band_map.cpu(), pt_save_path)

            row = dict(base_row)
            row["feature_file"] = pt_path.name
            row["band_pt_path"] = str(pt_save_path)
            row.update(summarize_band_map(band_map))
            summary_rows.append(row)

        except Exception as e:
            row = dict(base_row)
            row["missing_feature"] = True
            row["error"] = str(e)
            summary_rows.append(row)


# =========================================================
# SAVE SUMMARY CSVs (per-condition, to avoid cross-task races)
# =========================================================
summary_df   = pd.DataFrame(summary_rows)
summary_path = SAVE_ROOT / f"{CONDITION}_conformal_band_summary_per_sample.csv"
summary_df.to_csv(summary_path, index=False)

dataset_summary_cols = [
    "below_min_count", "low_count", "med_count", "high_count", "very_high_count", "above_max_count",
    "below_min_pct",   "low_pct",   "med_pct",   "high_pct",   "very_high_pct",   "above_max_pct",
    "ood_pct",
]
existing_cols = [c for c in dataset_summary_cols if c in summary_df.columns]

if existing_cols and (summary_df["missing_feature"] == False).any():
    dataset_summary = (
        summary_df[summary_df["missing_feature"] == False]
        .groupby(["condition", "distance"])[existing_cols]
        .mean()
        .reset_index()
    )
    dataset_summary_path = SAVE_ROOT / f"{CONDITION}_conformal_band_summary_dataset.csv"
    dataset_summary.to_csv(dataset_summary_path, index=False)
    print(f"Saved dataset summary    : {dataset_summary_path}")

print(f"Saved per-sample summary : {summary_path}")
print(f"\n[job {args.job_id}] Done. Condition: {CONDITION}")
