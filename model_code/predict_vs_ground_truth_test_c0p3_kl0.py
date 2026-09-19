#!/usr/bin/env python3
# predict_vs_ground_truth_test_c0p3_kl0.py
#
# Point predictions (bearing, range) vs ground truth, across all 8 test
# conditions, for the c0p3_kl0 checkpoint:
#   plain, awgn_15dB, awgn_5dB, awgn_-5dB,
#   clutter_15dB, clutter_5dB, clutter_-5dB, random_noise
#
# This is deliberately separate from apply_evidential_conformal_bands_to_
# test_c0p3_kl0.py -- that script covers uncertainty + bands but never
# wrote out the actual predicted bearing_deg/range_m values, only the
# ground truth and error deltas. This script fills that gap directly:
# per-sample predicted vs actual values, plus a per-condition summary
# (MAE/RMSE). Uses model.forward() (means only) rather than forward_all(),
# since no NIG/uncertainty output is needed here.
#
# The 7 real-audio conditions reuse the sibling ddl_dataloader.py's
# TestCustomAudioDataset (same feature_source=<condition> pattern as the
# other c0p3_kl0 test-time scripts). random_noise has no CSV row / ground
# truth -- predictions only, no error columns.
#
# Sharded as an 8-way array job, one condition per task, each writing its
# own per-condition CSVs (avoids the 8 parallel tasks racing on one file).
#
# Outputs go to ./evidential_calibration_results_c0p3_kl0/test_predictions/

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

SAVE_DIR = Path("./evidential_calibration_results_c0p3_kl0/test_predictions")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cpu"
MAX_RANGE  = 250.0
EPS        = 1e-8
BATCH_SIZE = 32

print(f"Job ID    : {args.job_id}")
print(f"Condition : {CONDITION}")


# =========================================================
# HELPERS
# =========================================================
def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


# =========================================================
# LOAD MODEL
# =========================================================
print("\nLoading model ...")
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

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

            sin_mu, cos_mu, rng_mu = model(audio_data)  # means only
            theta_mu = torch.atan2(sin_mu, cos_mu)

            bearing_deg_pred = torch.rad2deg(theta_mu).cpu().numpy()
            range_m_pred     = (rng_mu * MAX_RANGE).cpu().numpy()

            for b in range(audio_data.size(0)):
                rows.append({
                    "ID": ids[b], "condition": CONDITION,
                    "bearing_deg_gt": None, "range_m_gt": None,
                    "bearing_deg_pred": float(bearing_deg_pred[b]),
                    "range_m_pred":     float(range_m_pred[b]),
                    "bearing_error_deg": None, "range_error_m": None, "euclidean_error_m": None,
                })

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

            sin_mu, cos_mu, rng_mu = model(audio_data)  # means only
            theta_mu   = torch.atan2(sin_mu, cos_mu)
            theta_true = torch.atan2(y_sin, y_cos)

            angle_diff_rad = wrap_angle(theta_true - theta_mu).abs()
            angle_diff_deg = torch.rad2deg(angle_diff_rad)
            range_error_m  = torch.abs(rng_mu - y_range) * MAX_RANGE
            px = rng_mu * MAX_RANGE * cos_mu;  py = rng_mu * MAX_RANGE * sin_mu
            tx = y_range * MAX_RANGE * y_cos;  ty = y_range * MAX_RANGE * y_sin
            euc_error_m = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)

            ids               = sample_id.cpu().numpy()
            gt_bearing         = torch.rad2deg(torch.atan2(y_sin, y_cos)).cpu().numpy()
            gt_range_m          = (y_range * MAX_RANGE).cpu().numpy()
            bearing_deg_pred    = torch.rad2deg(theta_mu).cpu().numpy()
            range_m_pred        = (rng_mu * MAX_RANGE).cpu().numpy()
            bearing_error_deg   = angle_diff_deg.cpu().numpy()
            range_error_m_np    = range_error_m.cpu().numpy()
            euclidean_error_m   = euc_error_m.cpu().numpy()

            for b in range(audio_data.size(0)):
                rows.append({
                    "ID": int(ids[b]), "condition": CONDITION,
                    "bearing_deg_gt":   float(gt_bearing[b]),
                    "range_m_gt":       float(gt_range_m[b]),
                    "bearing_deg_pred": float(bearing_deg_pred[b]),
                    "range_m_pred":     float(range_m_pred[b]),
                    "bearing_error_deg": float(bearing_error_deg[b]),
                    "range_error_m":     float(range_error_m_np[b]),
                    "euclidean_error_m": float(euclidean_error_m[b]),
                })


# =========================================================
# SAVE PER-SAMPLE CSV
# =========================================================
result_df = pd.DataFrame(rows)
per_sample_path = SAVE_DIR / f"{CONDITION}_predictions_vs_ground_truth_per_sample.csv"
result_df.to_csv(per_sample_path, index=False)
print(f"\nSaved per-sample CSV -> {per_sample_path}")

# =========================================================
# SAVE SUMMARY CSV
# =========================================================
has_gt = result_df["bearing_deg_gt"].notna().any()

if has_gt:
    summary = {
        "condition":          CONDITION,
        "num_samples":        len(result_df),
        "mae_bearing_deg":    float(result_df["bearing_error_deg"].mean()),
        "rmse_bearing_deg":   float((result_df["bearing_error_deg"] ** 2).mean() ** 0.5),
        "mae_range_m":        float(result_df["range_error_m"].mean()),
        "rmse_range_m":       float((result_df["range_error_m"] ** 2).mean() ** 0.5),
        "mae_euclidean_m":    float(result_df["euclidean_error_m"].mean()),
        "rmse_euclidean_m":   float((result_df["euclidean_error_m"] ** 2).mean() ** 0.5),
        "median_bearing_error_deg":  float(result_df["bearing_error_deg"].median()),
        "median_range_error_m":      float(result_df["range_error_m"].median()),
        "median_euclidean_error_m":  float(result_df["euclidean_error_m"].median()),
    }
else:
    # random_noise: no ground truth -- report prediction distribution only
    summary = {
        "condition":               CONDITION,
        "num_samples":             len(result_df),
        "mean_bearing_deg_pred":   float(result_df["bearing_deg_pred"].mean()),
        "std_bearing_deg_pred":    float(result_df["bearing_deg_pred"].std()),
        "mean_range_m_pred":       float(result_df["range_m_pred"].mean()),
        "std_range_m_pred":        float(result_df["range_m_pred"].std()),
    }

summary_path = SAVE_DIR / f"{CONDITION}_predictions_vs_ground_truth_summary.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False)
print(f"Saved summary CSV -> {summary_path}")

print(f"\n[job {args.job_id}] Done. Condition: {CONDITION}")
