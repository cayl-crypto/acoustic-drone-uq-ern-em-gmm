#!/usr/bin/env python3
# dump_evidential_raw_output_test_c0p3_kl0.py
#
# Raw model output -- not results/predictions-vs-ground-truth -- across all
# 8 test conditions, for the c0p3_kl0 checkpoint:
#   plain, awgn_15dB, awgn_5dB, awgn_-5dB,
#   clutter_15dB, clutter_5dB, clutter_-5dB, random_noise
#
# Per sample, dumps everything the evidential heads produce:
#   - Raw NIG parameters per head (sin/cos/range): mu, v (lambda), alpha, beta
#   - Derived point predictions (bearing_deg_pred, range_m_pred) -- a direct,
#     deterministic transform of mu, not an accuracy result
#   - Derived uncertainty scalars (aleatoric_var, epistemic_var, total_var,
#     sigma per head, bearing_std_deg) -- also just a deterministic function
#     of v/alpha/beta, not compared against anything
#
# No ground truth, no error metrics, no conformal bands -- see
# predict_vs_ground_truth_test_c0p3_kl0.py and
# apply_evidential_conformal_bands_to_test_c0p3_kl0.py for those.
#
# Since nothing here needs ground truth, no sibling-dataloader dependency
# is needed -- just glob each condition's logmel files and run the model.
# BUT: unlike awgn_*/clutter_* (which only ever hold the 11,628 test-split
# files, since extract_awgn_test_logmel.py / extract_clutter_test_logmel.py
# only ever wrote logmel for the is_corrupt==False & sample_9600==True
# filtered test rows), plain/features/logmel/ holds the FULL 58,136-file
# dataset backfill (train+val+test combined) -- a plain glob there would
# silently pull in train/val samples too, ~5x more than intended and
# inconsistent with every other condition. Same issue already hit and
# fixed in gmm_test_noise_distance_eval_c0p3_kl0.py /
# gmm_scalar_distance_eval_c0p3_kl0.py. Fixed here by intersecting plain's
# file list against the test CSV's ID set before processing; random_noise
# has no corresponding CSV rows at all, so it's globbed as-is.
#
# Sharded as an 8-way array job, one condition per task, each writing its
# own per-condition CSV.
#
# Outputs go to ./evidential_calibration_results_c0p3_kl0/test_raw_output/

import argparse
import math
from pathlib import Path
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
# PATHS / CONFIG
# =========================================================
CSV_ROOT     = Path("/path/to/DDLDataset")
TEST_CSV     = CSV_ROOT / "annotations" / "test_dataset_with_nondrone_with_9600_flag.csv"
FEATURE_DIR  = CSV_ROOT / CONDITION / "features" / "logmel"
MODEL_PATH   = Path("best_evidential_loc_logmel_plain_plain_c0p3_kl0.pt")

SAVE_DIR = Path("./evidential_calibration_results_c0p3_kl0/test_raw_output")
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
def nig_uncertainties(pack: torch.Tensor):
    v, alpha, beta = pack[:, 1], pack[:, 2], pack[:, 3]
    denom         = torch.clamp(alpha - 1.0, min=EPS)
    aleatoric_var = beta / denom
    epistemic_var = beta / (torch.clamp(v, min=EPS) * denom)
    total_var     = aleatoric_var + epistemic_var
    sigma         = torch.sqrt(torch.clamp(total_var, min=EPS))
    return aleatoric_var, epistemic_var, total_var, sigma


# =========================================================
# LOAD MODEL
# =========================================================
print("\nLoading model ...")
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

if not FEATURE_DIR.exists():
    print(f"\n[WARNING] Feature dir does not exist, nothing to do: {FEATURE_DIR}")
    raise SystemExit(0)

pt_files = sorted(FEATURE_DIR.glob("*.pt"))
print(f"\n{CONDITION}: {len(pt_files)} files found in {FEATURE_DIR}")

if CONDITION != "random_noise":
    # Restrict to the test-split IDs. Necessary for "plain" specifically
    # (its logmel dir holds the full 58,136-file dataset backfill, not
    # just the 11,628 test IDs); a no-op for awgn_*/clutter_* since their
    # dirs already only ever contain the test-split files, but applying it
    # uniformly keeps this correct regardless of what's on disk.
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df[(test_df["is_corrupt"] == False) & (test_df["sample_9600"] == True)]
    test_ids = {f"{int(v):08d}" for v in test_df["ID"]}
    before = len(pt_files)
    pt_files = [p for p in pt_files if p.stem in test_ids]
    print(f"  Restricted to test-split IDs: {before} -> {len(pt_files)} files")

rows = []

with torch.no_grad():
    for i in tqdm(range(0, len(pt_files), BATCH_SIZE), desc=CONDITION):
        batch_paths = pt_files[i:i + BATCH_SIZE]
        audio_data = torch.stack(
            [torch.load(p, map_location=DEVICE).float() for p in batch_paths]
        ).to(DEVICE)
        ids = [p.stem for p in batch_paths]

        means, packs = model.forward_all(audio_data)
        sin_mu, cos_mu, rng_mu = means
        sin_pack, cos_pack, rng_pack = packs  # each (B, 4) = [mu, v, alpha, beta]

        theta_mu = torch.atan2(sin_mu, cos_mu)
        bearing_deg_pred = torch.rad2deg(theta_mu)
        range_m_pred     = rng_mu * MAX_RANGE

        sin_alea, sin_epi, sin_tot, sin_sig = nig_uncertainties(sin_pack)
        cos_alea, cos_epi, cos_tot, cos_sig = nig_uncertainties(cos_pack)
        rng_alea, rng_epi, rng_tot, rng_sig = nig_uncertainties(rng_pack)

        denom     = sin_mu ** 2 + cos_mu ** 2 + EPS
        var_theta = (cos_mu / denom) ** 2 * sin_sig ** 2 + (-sin_mu / denom) ** 2 * cos_sig ** 2
        bearing_std_deg = torch.rad2deg(torch.sqrt(torch.clamp(var_theta, min=EPS)))

        B = audio_data.size(0)
        for b in range(B):
            rows.append({
                "ID": ids[b], "condition": CONDITION,

                # raw NIG params -- sin head
                "sin_mu":    float(sin_pack[b, 0]),
                "sin_v":     float(sin_pack[b, 1]),
                "sin_alpha": float(sin_pack[b, 2]),
                "sin_beta":  float(sin_pack[b, 3]),
                # raw NIG params -- cos head
                "cos_mu":    float(cos_pack[b, 0]),
                "cos_v":     float(cos_pack[b, 1]),
                "cos_alpha": float(cos_pack[b, 2]),
                "cos_beta":  float(cos_pack[b, 3]),
                # raw NIG params -- range head
                "range_mu":    float(rng_pack[b, 0]),
                "range_v":     float(rng_pack[b, 1]),
                "range_alpha": float(rng_pack[b, 2]),
                "range_beta":  float(rng_pack[b, 3]),

                # derived point predictions (deterministic transform of mu)
                "bearing_deg_pred": float(bearing_deg_pred[b]),
                "range_m_pred":     float(range_m_pred[b]),

                # derived uncertainty scalars (deterministic transform of v/alpha/beta)
                "aleatoric_var_sin":   float(sin_alea[b]),
                "epistemic_var_sin":   float(sin_epi[b]),
                "total_var_sin":       float(sin_tot[b]),
                "sigma_sin":           float(sin_sig[b]),
                "aleatoric_var_cos":   float(cos_alea[b]),
                "epistemic_var_cos":   float(cos_epi[b]),
                "total_var_cos":       float(cos_tot[b]),
                "sigma_cos":           float(cos_sig[b]),
                "aleatoric_var_range": float(rng_alea[b]),
                "epistemic_var_range": float(rng_epi[b]),
                "total_var_range":     float(rng_tot[b]),
                "sigma_range":         float(rng_sig[b]),
                "bearing_std_deg":     float(bearing_std_deg[b]),
            })

result_df = pd.DataFrame(rows)
out_path = SAVE_DIR / f"{CONDITION}_evidential_raw_output.csv"
result_df.to_csv(out_path, index=False)
print(f"\nSaved raw output CSV ({len(result_df)} rows) -> {out_path}")

print(f"\n[job {args.job_id}] Done. Condition: {CONDITION}")
