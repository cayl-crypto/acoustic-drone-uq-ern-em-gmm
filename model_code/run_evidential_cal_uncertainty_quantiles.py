#!/usr/bin/env python3
# run_evidential_cal_uncertainty_quantiles.py
#
# Runs best_evidential_model.pt on the full calibration set (N=2912).
# All 2912 calibration IDs are loaded from the plain logmel feature folder,
# regardless of the noise_level column in the CSV.
#
# Outputs (all under ./evidential_calibration_results/):
#   plain_calibration_set_per_sample.csv
#   plain_calibration_set_summary.csv
#   plain_calibration_set_uncertainty_quantiles.csv   (13 scalars × min/q6827/q9545/q9973/max/mean/std)
#   plain_calibration_set_uncertainty_quantiles.pt    (same as dict, keyed by scalar name)

import sys
import math
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.distributions import StudentT
from tqdm import tqdm

# =========================================================
# IMPORT DATASET
# =========================================================
sys.path.insert(0, "/path/to/variance_aware_evidential_learning_ddl")
from ddl_dataloader import TestCustomAudioDataset, load_filtered_dataframe

# =========================================================
# PATHS / CONFIG
# =========================================================
VAL_CSV      = Path("/path/to/DDLDataset/annotations/"
                    "val_dataset_with_nondrone_with_9600_flag.csv")
MODEL_PATH   = Path("./best_evidential_model.pt")
FEATURE_ROOT = Path("/path/to/DDLDataset")
SAVE_DIR     = Path("./evidential_calibration_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_RANGE  = 250.0
EPS        = 1e-8
BATCH_SIZE = 32

TARGET_COVERAGES = {"q6827": 0.6827, "q9545": 0.9545, "q9973": 0.9973}
SIGMAS = [1, 2, 3]

print(f"Device   : {DEVICE}")
print(f"Save dir : {SAVE_DIR.resolve()}")


# =========================================================
# NIG HELPERS
# =========================================================
def nig_nll_from_pack(pack: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """pack: (B,4)=[mu,lam,alpha,beta], y: (B,) -> per-sample NLL (B,)"""
    mu, lam, alpha, beta = pack[:, 0], pack[:, 1], pack[:, 2], pack[:, 3]
    nu     = 2.0 * alpha
    scale2 = beta * (1.0 + 1.0 / (lam + EPS)) / (alpha + EPS)
    scale  = torch.sqrt(torch.clamp(scale2, min=EPS))
    return -StudentT(df=nu, loc=mu, scale=scale).log_prob(y)


def nig_uncertainties(pack: torch.Tensor):
    """
    Returns aleatoric_var, epistemic_var, total_var, sigma — each (B,).
        aleatoric_var = beta / (alpha - 1)
        epistemic_var = beta / (v * (alpha - 1))
        total_var     = aleatoric_var + epistemic_var
        sigma         = sqrt(total_var)
    Naming matches document_evidential_unit_norm_v2.py columns.
    """
    v, alpha, beta = pack[:, 1], pack[:, 2], pack[:, 3]
    denom         = torch.clamp(alpha - 1.0, min=EPS)
    aleatoric_var = beta / denom
    epistemic_var = beta / (torch.clamp(v, min=EPS) * denom)
    total_var     = aleatoric_var + epistemic_var
    sigma         = torch.sqrt(torch.clamp(total_var, min=EPS))
    return aleatoric_var, epistemic_var, total_var, sigma


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def conformal_quantile_idx(n: int, coverage: float) -> int:
    k = math.ceil((n + 1) * coverage)
    return min(max(k, 1), n) - 1


# =========================================================
# LOAD MODEL
# =========================================================
print("\nLoading model ...")
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")


# =========================================================
# BUILD CALIBRATION DATALOADER
# All 2912 unique calibration IDs → plain logmel features
# =========================================================
print("\nBuilding calibration dataset ...")
calib_df = load_filtered_dataframe(VAL_CSV, conformal_role="calibration")
print(f"Calibration samples : {len(calib_df)}  |  Unique IDs: {calib_df['ID'].nunique()}")

dataset = TestCustomAudioDataset(
    data_frame    = calib_df,
    wav_folder    = FEATURE_ROOT,       # not used — features loaded by ID
    feature_combo = "logmel",
    target_type   = "regression",
    feature_root  = str(FEATURE_ROOT),
    max_range     = MAX_RANGE,
    db_level      = "plain",
    split         = "val",
    feature_source= "plain",            # all IDs resolved from plain/features/logmel/
)

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=(DEVICE == "cuda"),
)


# =========================================================
# FORWARD PASS — COLLECT PER-SAMPLE DATA
# =========================================================
print("\nRunning inference ...")

rows = []

total_n         = 0
nll_sin_sum     = nll_cos_sum = nll_rng_sum = nll_tot_sum = 0.0
bearing_mae_sum = range_mae_sum = euc_mae_sum = 0.0
coverage_counts = {
    head: {k: 0.0 for k in SIGMAS}
    for head in ["sin", "cos", "range", "bearing", "overall"]
}

with torch.no_grad():
    for batch in tqdm(loader, desc="Calibration set"):
        audio_data, bearing_vec, range_to_target, sample_id = batch

        audio_data      = audio_data.to(DEVICE)
        bearing_vec     = bearing_vec.to(DEVICE)
        range_to_target = range_to_target.to(DEVICE).view(-1)

        B        = audio_data.size(0)
        total_n += B

        y_cos   = bearing_vec[:, 0]
        y_sin   = bearing_vec[:, 1]
        y_range = range_to_target

        # Forward — get all NIG parameters
        means, packs         = model.forward_all(audio_data)
        sin_mu, cos_mu, rng_mu = means
        sin_pack, cos_pack, rng_pack = packs   # each (B, 4)

        # NLLs
        nll_sin = nig_nll_from_pack(sin_pack, y_sin)
        nll_cos = nig_nll_from_pack(cos_pack, y_cos)
        nll_rng = nig_nll_from_pack(rng_pack, y_range)
        nll_tot = nll_sin + nll_cos + nll_rng

        nll_sin_sum += nll_sin.sum().item()
        nll_cos_sum += nll_cos.sum().item()
        nll_rng_sum += nll_rng.sum().item()
        nll_tot_sum += nll_tot.sum().item()

        # Uncertainties
        sin_alea_var,  sin_epi_var,  sin_tot_var,  sin_sigma  = nig_uncertainties(sin_pack)
        cos_alea_var,  cos_epi_var,  cos_tot_var,  cos_sigma  = nig_uncertainties(cos_pack)
        rng_alea_var,  rng_epi_var,  rng_tot_var,  rng_sigma  = nig_uncertainties(rng_pack)

        # Bearing std via error propagation: theta = atan2(sin, cos)
        theta_mu    = torch.atan2(sin_mu, cos_mu)
        theta_true  = torch.atan2(y_sin,  y_cos)
        denom       = sin_mu ** 2 + cos_mu ** 2 + EPS
        var_theta   = (cos_mu  / denom) ** 2 * sin_sigma ** 2 + \
                      (-sin_mu / denom) ** 2 * cos_sigma ** 2
        theta_std       = torch.sqrt(torch.clamp(var_theta, min=EPS))
        bearing_std_deg = torch.rad2deg(theta_std)

        # Point-estimate metrics (uses model helper)
        metrics = model.compute_metrics_from_means(
            sin_mu, cos_mu, rng_mu, y_sin, y_cos, y_range, max_range=MAX_RANGE
        )
        bearing_mae_sum += metrics["bearing_mae_deg"] * B
        range_mae_sum   += metrics["range_mae_m"]     * B
        euc_mae_sum     += metrics["euclidean_mae_m"] * B

        # Per-sample errors
        angle_diff_rad = wrap_angle(theta_true - theta_mu).abs()
        angle_diff_deg = torch.rad2deg(angle_diff_rad)
        range_error_m  = torch.abs(rng_mu - y_range) * MAX_RANGE
        px = rng_mu * MAX_RANGE * cos_mu;  py = rng_mu * MAX_RANGE * sin_mu
        tx = y_range * MAX_RANGE * y_cos;  ty = y_range * MAX_RANGE * y_sin
        euc_error_m = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)

        # Coverage
        for k in SIGMAS:
            kf         = float(k)
            sin_ok     = torch.abs(y_sin   - sin_mu) <= kf * sin_sigma
            cos_ok     = torch.abs(y_cos   - cos_mu) <= kf * cos_sigma
            rng_ok     = torch.abs(y_range - rng_mu) <= kf * rng_sigma
            bearing_ok = angle_diff_rad               <= kf * theta_std
            coverage_counts["sin"][k]     += sin_ok.sum().item()
            coverage_counts["cos"][k]     += cos_ok.sum().item()
            coverage_counts["range"][k]   += rng_ok.sum().item()
            coverage_counts["bearing"][k] += bearing_ok.sum().item()
            coverage_counts["overall"][k] += (rng_ok & bearing_ok).sum().item()

        # Per-sample rows
        ids        = sample_id.cpu().numpy()
        gt_bearing = torch.rad2deg(torch.atan2(y_sin, y_cos)).cpu().numpy()
        gt_range_m = (y_range * MAX_RANGE).cpu().numpy()

        for i in range(B):
            rows.append({
                "sample_id":         int(ids[i]),
                "bearing_deg_gt":    float(gt_bearing[i]),
                "range_m_gt":        float(gt_range_m[i]),

                "sin_mu":            float(sin_mu[i]),
                "cos_mu":            float(cos_mu[i]),
                "range_mu":          float(rng_mu[i]),
                "bearing_deg_pred":  float(torch.rad2deg(theta_mu[i])),
                "range_m_pred":      float(rng_mu[i] * MAX_RANGE),

                "v_sin":             float(sin_pack[i, 1]),
                "alpha_sin":         float(sin_pack[i, 2]),
                "beta_sin":          float(sin_pack[i, 3]),
                "v_cos":             float(cos_pack[i, 1]),
                "alpha_cos":         float(cos_pack[i, 2]),
                "beta_cos":          float(cos_pack[i, 3]),
                "v_range":           float(rng_pack[i, 1]),
                "alpha_range":       float(rng_pack[i, 2]),
                "beta_range":        float(rng_pack[i, 3]),

                "aleatoric_var_sin":   float(sin_alea_var[i]),
                "epistemic_var_sin":   float(sin_epi_var[i]),
                "total_var_sin":       float(sin_tot_var[i]),
                "sigma_sin":           float(sin_sigma[i]),
                "aleatoric_var_cos":   float(cos_alea_var[i]),
                "epistemic_var_cos":   float(cos_epi_var[i]),
                "total_var_cos":       float(cos_tot_var[i]),
                "sigma_cos":           float(cos_sigma[i]),
                "aleatoric_var_range": float(rng_alea_var[i]),
                "epistemic_var_range": float(rng_epi_var[i]),
                "total_var_range":     float(rng_tot_var[i]),
                "sigma_range":         float(rng_sigma[i]),
                "bearing_std_deg":     float(bearing_std_deg[i]),

                "sin_nll":           float(nll_sin[i]),
                "cos_nll":           float(nll_cos[i]),
                "range_nll":         float(nll_rng[i]),
                "total_nll":         float(nll_tot[i]),

                "bearing_error_deg": float(angle_diff_deg[i]),
                "range_error_m":     float(range_error_m[i]),
                "euclidean_error_m": float(euc_error_m[i]),
            })


# =========================================================
# OUTPUT 1: PER-SAMPLE CSV
# =========================================================
per_sample_df   = pd.DataFrame(rows)
per_sample_path = SAVE_DIR / "plain_calibration_set_per_sample.csv"
per_sample_df.to_csv(per_sample_path, index=False)
print(f"\nSaved per-sample CSV ({len(per_sample_df)} rows) : {per_sample_path}")


# =========================================================
# OUTPUT 2: AGGREGATE SUMMARY CSV
# =========================================================
summary = {"num_samples": total_n}
for key, val in [("nll_sin",   nll_sin_sum),
                 ("nll_cos",   nll_cos_sum),
                 ("nll_range", nll_rng_sum),
                 ("nll_total", nll_tot_sum)]:
    summary[key] = val / total_n
summary["mae_bearing_deg"] = bearing_mae_sum / total_n
summary["mae_range_m"]     = range_mae_sum   / total_n
summary["mae_euclidean_m"] = euc_mae_sum     / total_n
for head in ["sin", "cos", "range", "bearing", "overall"]:
    for k in SIGMAS:
        summary[f"coverage_{head}_{k}std"] = coverage_counts[head][k] / total_n

summary_path = SAVE_DIR / "plain_calibration_set_summary.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False)
print(f"Saved summary CSV                  : {summary_path}")


# =========================================================
# OUTPUT 3 & 4: CONFORMAL QUANTILES FOR 13 UNCERTAINTY SCALARS
# 4 per head (aleatoric_var, epistemic_var, total_var, sigma) × 3 heads + bearing_std_deg
# =========================================================
uncertainty_cols = [
    "aleatoric_var_sin",   "epistemic_var_sin",   "total_var_sin",   "sigma_sin",
    "aleatoric_var_cos",   "epistemic_var_cos",   "total_var_cos",   "sigma_cos",
    "aleatoric_var_range", "epistemic_var_range", "total_var_range", "sigma_range",
    "bearing_std_deg",
]

n_cal         = len(per_sample_df)
quantile_rows = []
quantile_dict = {"n_calibration": n_cal}

for col in uncertainty_cols:
    vals      = per_sample_df[col].values.astype(np.float32)
    vals_sort = np.sort(vals)

    q_vals = {}
    for q_name, coverage in TARGET_COVERAGES.items():
        idx            = conformal_quantile_idx(n_cal, coverage)
        q_vals[q_name] = float(vals_sort[idx])

    quantile_rows.append({
        "uncertainty_type": col,
        "n_calibration":    n_cal,
        "min":              float(vals_sort[0]),
        "q6827":            q_vals["q6827"],
        "q9545":            q_vals["q9545"],
        "q9973":            q_vals["q9973"],
        "max":              float(vals_sort[-1]),
        "mean":             float(vals.mean()),
        "std":              float(vals.std()),
    })

    quantile_dict[col] = {
        "min":   float(vals_sort[0]),
        "q6827": q_vals["q6827"],
        "q9545": q_vals["q9545"],
        "q9973": q_vals["q9973"],
        "max":   float(vals_sort[-1]),
    }

quantiles_csv_path = SAVE_DIR / "plain_calibration_set_uncertainty_quantiles.csv"
quantiles_pt_path  = SAVE_DIR / "plain_calibration_set_uncertainty_quantiles.pt"
pd.DataFrame(quantile_rows).to_csv(quantiles_csv_path, index=False)
torch.save(quantile_dict, quantiles_pt_path)

print(f"Saved quantiles CSV                : {quantiles_csv_path}")
print(f"Saved quantiles .pt                : {quantiles_pt_path}")
print(f"\nConformal quantiles (n={n_cal}):")
for row in quantile_rows:
    print(f"  {row['uncertainty_type']:<22s}  "
          f"min={row['min']:.4f}  q6827={row['q6827']:.4f}  "
          f"q9545={row['q9545']:.4f}  q9973={row['q9973']:.4f}  "
          f"max={row['max']:.4f}")

print(f"\nDone. All outputs saved to: {SAVE_DIR.resolve()}")
