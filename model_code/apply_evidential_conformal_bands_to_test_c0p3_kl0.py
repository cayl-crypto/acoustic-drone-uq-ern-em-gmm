#!/usr/bin/env python3
# apply_evidential_conformal_bands_to_test_c0p3_kl0.py
#
# Applies conformal bands to the model's NATIVE evidential (NIG) uncertainty
# -- not the GMM feature-density approach -- across all 8 test conditions,
# for the c0p3_kl0 checkpoint:
#   plain, awgn_15dB, awgn_5dB, awgn_-5dB,
#   clutter_15dB, clutter_5dB, clutter_-5dB, random_noise
#
# All 13 uncertainty scalars from run_evidential_cal_uncertainty_quantiles_
# c0p3_kl0.py get their own bands (each scalar's own calibrated
# min/q6827/q9545/q9973/max thresholds -- plain scalar values here, not
# spatial (H,W,T) maps like the GMM side, so no gather/broadcast machinery
# needed, just a per-scalar comparison per sample).
#
# Band encoding (6 levels), same convention as the GMM bands:
#   0  below_min   value < calib_min
#   1  low         calib_min <= value <= q6827
#   2  med         q6827 < value <= q9545
#   3  high        q9545 < value <= q9973
#   4  very_high   q9973 < value <= calib_max
#   5  above_max   value > calib_max
#
# The 7 real-audio conditions reuse the sibling ddl_dataloader.py's
# TestCustomAudioDataset (feature_source=<condition> resolves to
# {condition}/features/logmel/{ID}.pt, same layout used throughout this
# project), so real ground truth is available and per-sample error metrics
# get computed alongside the uncertainty bands.
#
# random_noise has no CSV row / ground truth -- evaluated directly from
# whatever files exist in its feature dir (uncertainty + bands only, no
# error metrics).
#
# Sharded as an 8-way array job, one condition per task, each writing its
# own per-condition CSVs (not a shared file, to avoid the 8 parallel tasks
# racing on one CSV).
#
# Requires run_evidential_cal_uncertainty_quantiles_c0p3_kl0.py to have
# completed first.
#
# Outputs go to ./evidential_calibration_results_c0p3_kl0/test_conformal_bands/

import argparse
import sys
import math
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/path/to/variance_aware_evidential_learning_ddl")
from ddl_dataloader import TestCustomAudioDataset, load_filtered_dataframe

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
# PATHS / CONFIG
# =========================================================
CSV_ROOT     = Path("/path/to/DDLDataset")
TEST_CSV     = CSV_ROOT / "annotations" / "test_dataset_with_nondrone_with_9600_flag.csv"
FEATURE_ROOT = CSV_ROOT
MODEL_PATH   = Path("best_evidential_loc_logmel_plain_plain_c0p3_kl0.pt")

CAL_DIR      = Path("./evidential_calibration_results_c0p3_kl0")
QUANTILES_PT = CAL_DIR / "plain_calibration_set_uncertainty_quantiles.pt"

SAVE_DIR = CAL_DIR / "test_conformal_bands"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cpu"
MAX_RANGE  = 250.0
EPS        = 1e-8
BATCH_SIZE = 32

UNCERTAINTY_COLS = [
    "aleatoric_var_sin",   "epistemic_var_sin",   "total_var_sin",   "sigma_sin",
    "aleatoric_var_cos",   "epistemic_var_cos",   "total_var_cos",   "sigma_cos",
    "aleatoric_var_range", "epistemic_var_range", "total_var_range", "sigma_range",
    "bearing_std_deg",
]

print(f"Job ID    : {args.job_id}")
print(f"Condition : {CONDITION}")


# =========================================================
# HELPERS (same NIG math as run_evidential_cal_uncertainty_quantiles_c0p3_kl0.py)
# =========================================================
def nig_uncertainties(pack: torch.Tensor):
    v, alpha, beta = pack[:, 1], pack[:, 2], pack[:, 3]
    denom         = torch.clamp(alpha - 1.0, min=EPS)
    aleatoric_var = beta / denom
    epistemic_var = beta / (torch.clamp(v, min=EPS) * denom)
    total_var     = aleatoric_var + epistemic_var
    sigma         = torch.sqrt(torch.clamp(total_var, min=EPS))
    return aleatoric_var, epistemic_var, total_var, sigma


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def assign_bands_vec(values: torch.Tensor, th: dict) -> torch.Tensor:
    """
    values: (B,) tensor of one uncertainty scalar. th: {"min","q6827","q9545",
    "q9973","max"} calibrated thresholds for that scalar (plain floats).
    Pure single-index boolean assignment throughout -- no combined mask+slice
    indexing, so none of the PyTorch indexing gotcha from the GMM gather fix
    applies here.
    """
    bands = torch.ones_like(values, dtype=torch.uint8)  # default: low (1)
    bands[values < th["min"]]   = 0  # below_min
    bands[values > th["q6827"]] = 2  # med
    bands[values > th["q9545"]] = 3  # high
    bands[values > th["q9973"]] = 4  # very_high
    bands[values > th["max"]]   = 5  # above_max
    return bands


def compute_scalars(sin_pack, cos_pack, rng_pack, sin_mu, cos_mu):
    sin_alea, sin_epi, sin_tot, sin_sig = nig_uncertainties(sin_pack)
    cos_alea, cos_epi, cos_tot, cos_sig = nig_uncertainties(cos_pack)
    rng_alea, rng_epi, rng_tot, rng_sig = nig_uncertainties(rng_pack)

    denom     = sin_mu ** 2 + cos_mu ** 2 + EPS
    var_theta = (cos_mu / denom) ** 2 * sin_sig ** 2 + (-sin_mu / denom) ** 2 * cos_sig ** 2
    theta_std = torch.sqrt(torch.clamp(var_theta, min=EPS))
    bearing_std_deg = torch.rad2deg(theta_std)

    return {
        "aleatoric_var_sin": sin_alea, "epistemic_var_sin": sin_epi,
        "total_var_sin": sin_tot,      "sigma_sin": sin_sig,
        "aleatoric_var_cos": cos_alea, "epistemic_var_cos": cos_epi,
        "total_var_cos": cos_tot,      "sigma_cos": cos_sig,
        "aleatoric_var_range": rng_alea, "epistemic_var_range": rng_epi,
        "total_var_range": rng_tot,      "sigma_range": rng_sig,
        "bearing_std_deg": bearing_std_deg,
    }, theta_std


# =========================================================
# LOAD MODEL
# =========================================================
print("\nLoading model ...")
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# =========================================================
# LOAD CALIBRATION QUANTILES
# =========================================================
if not QUANTILES_PT.exists():
    raise FileNotFoundError(
        f"Missing: {QUANTILES_PT}\nRun run_evidential_cal_uncertainty_quantiles_c0p3_kl0.py first."
    )
print("Loading calibration quantiles ...")
quantile_dict = torch.load(QUANTILES_PT, map_location=DEVICE)
thresholds = {col: quantile_dict[col] for col in UNCERTAINTY_COLS}
print(f"  n_calibration = {quantile_dict['n_calibration']}")

rows = []

# =========================================================
# EVALUATE THIS JOB'S CONDITION
# =========================================================
if CONDITION == "random_noise":
    RANDOM_FEATURE_DIR = FEATURE_ROOT / "random_noise" / "features" / "logmel"
    pt_files = sorted(RANDOM_FEATURE_DIR.glob("*.pt"))
    print(f"\n{CONDITION}: {len(pt_files)} synthetic samples found in {RANDOM_FEATURE_DIR}")

    with torch.no_grad():
        for i in tqdm(range(0, len(pt_files), BATCH_SIZE), desc=CONDITION):
            batch_paths = pt_files[i:i + BATCH_SIZE]
            audio_data = torch.stack(
                [torch.load(p, map_location=DEVICE).float() for p in batch_paths]
            ).to(DEVICE)
            ids = [p.stem for p in batch_paths]

            means, packs = model.forward_all(audio_data)
            sin_mu, cos_mu, rng_mu = means
            sin_pack, cos_pack, rng_pack = packs

            scalar_vals, _ = compute_scalars(sin_pack, cos_pack, rng_pack, sin_mu, cos_mu)
            band_vals = {col: assign_bands_vec(scalar_vals[col], thresholds[col]) for col in UNCERTAINTY_COLS}

            B = audio_data.size(0)
            for b in range(B):
                row = {"ID": ids[b], "condition": CONDITION}
                for col in UNCERTAINTY_COLS:
                    row[col] = float(scalar_vals[col][b])
                    row[f"{col}_band"] = int(band_vals[col][b])
                rows.append(row)

else:
    if not (FEATURE_ROOT / CONDITION / "features" / "logmel").exists():
        print(f"\n[WARNING] Feature dir does not exist, nothing to do: "
              f"{FEATURE_ROOT / CONDITION / 'features' / 'logmel'}")
        raise SystemExit(0)

    test_df = load_filtered_dataframe(TEST_CSV)
    print(f"\n{CONDITION}: {len(test_df)} test rows (is_corrupt==False & sample_9600==True)")

    dataset = TestCustomAudioDataset(
        data_frame=test_df, wav_folder=FEATURE_ROOT, feature_combo="logmel",
        target_type="regression", feature_root=str(FEATURE_ROOT), max_range=MAX_RANGE,
        db_level=CONDITION, split="test", feature_source=CONDITION,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc=CONDITION):
            audio_data, bearing_vec, range_to_target, sample_id = batch
            audio_data      = audio_data.to(DEVICE)
            bearing_vec     = bearing_vec.to(DEVICE)
            range_to_target = range_to_target.to(DEVICE).view(-1)

            y_cos, y_sin, y_range = bearing_vec[:, 0], bearing_vec[:, 1], range_to_target

            means, packs = model.forward_all(audio_data)
            sin_mu, cos_mu, rng_mu = means
            sin_pack, cos_pack, rng_pack = packs

            scalar_vals, theta_std = compute_scalars(sin_pack, cos_pack, rng_pack, sin_mu, cos_mu)
            band_vals = {col: assign_bands_vec(scalar_vals[col], thresholds[col]) for col in UNCERTAINTY_COLS}

            theta_mu   = torch.atan2(sin_mu, cos_mu)
            theta_true = torch.atan2(y_sin, y_cos)
            angle_diff_rad = wrap_angle(theta_true - theta_mu).abs()
            angle_diff_deg = torch.rad2deg(angle_diff_rad)
            range_error_m  = torch.abs(rng_mu - y_range) * MAX_RANGE
            px = rng_mu * MAX_RANGE * cos_mu;  py = rng_mu * MAX_RANGE * sin_mu
            tx = y_range * MAX_RANGE * y_cos;  ty = y_range * MAX_RANGE * y_sin
            euc_error_m = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)

            ids        = sample_id.cpu().numpy()
            gt_bearing = torch.rad2deg(torch.atan2(y_sin, y_cos)).cpu().numpy()
            gt_range_m = (y_range * MAX_RANGE).cpu().numpy()

            B = audio_data.size(0)
            for b in range(B):
                row = {
                    "ID": int(ids[b]), "condition": CONDITION,
                    "bearing_deg_gt": float(gt_bearing[b]), "range_m_gt": float(gt_range_m[b]),
                    "bearing_error_deg": float(angle_diff_deg[b]),
                    "range_error_m":     float(range_error_m[b]),
                    "euclidean_error_m": float(euc_error_m[b]),
                }
                for col in UNCERTAINTY_COLS:
                    row[col] = float(scalar_vals[col][b])
                    row[f"{col}_band"] = int(band_vals[col][b])
                rows.append(row)


# =========================================================
# SAVE PER-SAMPLE CSV
# =========================================================
result_df = pd.DataFrame(rows)
per_sample_path = SAVE_DIR / f"{CONDITION}_evidential_conformal_bands_per_sample.csv"
result_df.to_csv(per_sample_path, index=False)
print(f"\nSaved per-sample CSV -> {per_sample_path}")

# =========================================================
# SAVE DATASET SUMMARY CSV (per scalar: band pct breakdown + ood_pct + mean)
# =========================================================
summary_rows = []
for col in UNCERTAINTY_COLS:
    band_col = f"{col}_band"
    vals  = result_df[band_col].values
    total = len(vals)
    counts = {lvl: int((vals == lvl).sum()) for lvl in range(6)}
    pcts   = {lvl: 100.0 * counts[lvl] / total for lvl in range(6)}
    ood_pct = pcts[0] + pcts[5]
    summary_rows.append({
        "condition":        CONDITION,
        "uncertainty_type": col,
        "num_samples":      total,
        "mean_value":       float(result_df[col].mean()),
        "below_min_pct":    pcts[0],
        "low_pct":          pcts[1],
        "med_pct":          pcts[2],
        "high_pct":         pcts[3],
        "very_high_pct":    pcts[4],
        "above_max_pct":    pcts[5],
        "ood_pct":          ood_pct,
    })

summary_df   = pd.DataFrame(summary_rows)
summary_path = SAVE_DIR / f"{CONDITION}_evidential_conformal_bands_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved dataset summary -> {summary_path}")

print(f"\n[job {args.job_id}] Done. Condition: {CONDITION}")
