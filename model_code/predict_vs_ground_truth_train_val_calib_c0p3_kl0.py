#!/usr/bin/env python3
# predict_vs_ground_truth_train_val_calib_c0p3_kl0.py
#
# Point predictions (bearing, range) vs ground truth, for the c0p3_kl0
# checkpoint, across the three non-test splits -- all plain condition only
# (no noise variants were ever generated for train/val, only for test):
#   0 = train       (full train_dataset CSV, every ID, no conformal_role column)
#   1 = val_evaluation  (val CSV, conformal_role == "evaluation")
#   2 = val_calibration (val CSV, conformal_role == "calibration")
#
# Same format/columns as predict_vs_ground_truth_test_c0p3_kl0.py, so all
# splits (train/val_evaluation/val_calibration/test-8-conditions) are
# directly comparable side by side. Uses model() (means only) -- no NIG
# uncertainty here, that's covered separately by
# run_evidential_cal_uncertainty_quantiles_c0p3_kl0.py for the calibration
# split (which also carries the full NIG parameters, not just point
# predictions -- this script is deliberately the lighter, uniform-format
# counterpart across every split).
#
# Reuses the sibling ddl_dataloader.py's TestCustomAudioDataset /
# load_filtered_dataframe, feature_source="plain" throughout.
#
# Sharded as a 3-way array job, one split per task, each writing its own
# per-split CSVs (avoids the 3 parallel tasks racing on one file).
#
# Outputs go to ./evidential_calibration_results_c0p3_kl0/test_predictions/
# (same folder as the test-split predictions, filenames disambiguate split).

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
                    help="Job index 0-2. Selects which split this job processes.")
args = parser.parse_args()

SPLITS = ["train", "val_evaluation", "val_calibration"]
assert 0 <= args.job_id < len(SPLITS), f"job_id must be 0-{len(SPLITS)-1}, got {args.job_id}"
SPLIT_NAME = SPLITS[args.job_id]

# =========================================================
# PATHS / CONFIG
# =========================================================
CSV_ROOT     = Path("/path/to/DDLDataset")
TRAIN_CSV    = CSV_ROOT / "annotations" / "train_dataset_with_nondrone_with_9600_flag.csv"
VAL_CSV      = CSV_ROOT / "annotations" / "val_dataset_with_nondrone_with_9600_flag.csv"
FEATURE_ROOT = CSV_ROOT
MODEL_PATH   = Path("best_evidential_loc_logmel_plain_plain_c0p3_kl0.pt")

SAVE_DIR = Path("./evidential_calibration_results_c0p3_kl0/test_predictions")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cpu"
MAX_RANGE  = 250.0
EPS        = 1e-8
BATCH_SIZE = 32

print(f"Job ID : {args.job_id}")
print(f"Split  : {SPLIT_NAME}")


# =========================================================
# HELPERS
# =========================================================
def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


# =========================================================
# RESOLVE THIS JOB'S DATAFRAME
# =========================================================
if SPLIT_NAME == "train":
    df = load_filtered_dataframe(TRAIN_CSV)
elif SPLIT_NAME == "val_evaluation":
    df = load_filtered_dataframe(VAL_CSV, conformal_role="evaluation")
else:  # val_calibration
    df = load_filtered_dataframe(VAL_CSV, conformal_role="calibration")

print(f"{SPLIT_NAME}: {len(df)} rows (is_corrupt==False & sample_9600==True"
      f"{' & conformal_role==' + repr(SPLIT_NAME.split('_')[1]) if '_' in SPLIT_NAME else ''})")

# =========================================================
# LOAD MODEL
# =========================================================
print("\nLoading model ...")
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# =========================================================
# BUILD DATALOADER (plain feature_source, every ID, no noise_level filter)
# =========================================================
dataset = TestCustomAudioDataset(
    data_frame=df, wav_folder=FEATURE_ROOT, feature_combo="logmel",
    target_type="regression", feature_root=str(FEATURE_ROOT), max_range=MAX_RANGE,
    db_level="plain", split=SPLIT_NAME, feature_source="plain",
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

# =========================================================
# RUN INFERENCE
# =========================================================
rows = []

with torch.no_grad():
    for batch in tqdm(loader, desc=SPLIT_NAME):
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
        gt_bearing        = torch.rad2deg(torch.atan2(y_sin, y_cos)).cpu().numpy()
        gt_range_m        = (y_range * MAX_RANGE).cpu().numpy()
        bearing_deg_pred  = torch.rad2deg(theta_mu).cpu().numpy()
        range_m_pred      = (rng_mu * MAX_RANGE).cpu().numpy()
        bearing_error_deg = angle_diff_deg.cpu().numpy()
        range_error_m_np  = range_error_m.cpu().numpy()
        euclidean_error_m = euc_error_m.cpu().numpy()

        for b in range(audio_data.size(0)):
            rows.append({
                "ID": int(ids[b]), "split": SPLIT_NAME,
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
per_sample_path = SAVE_DIR / f"{SPLIT_NAME}_predictions_vs_ground_truth_per_sample.csv"
result_df.to_csv(per_sample_path, index=False)
print(f"\nSaved per-sample CSV -> {per_sample_path}")

# =========================================================
# SAVE SUMMARY CSV
# =========================================================
summary = {
    "split":              SPLIT_NAME,
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
summary_path = SAVE_DIR / f"{SPLIT_NAME}_predictions_vs_ground_truth_summary.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False)
print(f"Saved summary CSV -> {summary_path}")

print(f"\n[job {args.job_id}] Done. Split: {SPLIT_NAME}")
