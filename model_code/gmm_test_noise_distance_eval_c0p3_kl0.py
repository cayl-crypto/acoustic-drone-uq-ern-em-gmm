#!/usr/bin/env python3
# gmm_test_noise_distance_eval_c0p3_kl0.py
#
# GMM-based NLL "distance from training distribution" analysis on the TEST
# split, across 8 conditions:
#   plain, awgn_15dB, awgn_5dB, awgn_-5dB,
#   clutter_15dB, clutter_5dB, clutter_-5dB, random_noise
#
# optimal_k ONLY (not gmm_k1..gmm_k10) — this replaces the earlier version
# that computed NLL under all 10 fixed-k GMMs plus a masked optimal-k pass
# (effectively ~20 full NLL computations per sample), which was too slow to
# finish within a reasonable time budget.
#
# Speed fix: the per-position optimal-k GMM parameters are GATHERED into a
# single padded (H, W, T, MAX_K) tensor ONCE at startup (unused component
# slots get weight=0, contributing ~0 probability mass in the log-sum-exp),
# so each sample only needs ONE NLL pass instead of up to 10.
#
# Sharded as an 8-way array job, one condition per task, both because the
# conditions are already independent and because generating 7 of them
# (everything but plain) is new work this run.
#
# Requires gmm_scalar_aggregate_1x1x1_c0p3_kl0.py,
# extract_features_cnn0_c0p3_kl0.py (plain), and
# extract_features_cnn0_test_noise_types_c0p3_kl0.py (the other 7) to have
# completed first.
#
# random_noise has no annotation CSV / real IDs — it's evaluated directly
# from whatever files exist in its feature dir, no CSV join.
#
# Outputs go to ./gmm_scalar_1x1x1_c0p3_kl0/test_noise_eval_results/

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
                    help="Job index 0-7. Selects which condition this job evaluates.")
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
SAVE_ROOT   = GMM_ROOT / "test_noise_eval_results"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
FULL_H = 128
FULL_W = 128
FULL_T = 31
MAX_K  = 10
EPS    = 1e-8
LOG2PI = math.log(2.0 * math.pi)

print(f"Job ID    : {args.job_id}")
print(f"Condition : {CONDITION}")


# =========================================================
# HELPERS
# =========================================================
def normalize_id(x):
    digits = "".join(ch for ch in str(x).strip() if ch.isdigit())
    return digits.zfill(8)


def build_file_index(folder: Path):
    """
    O(1)-lookup dict built once, instead of find_file_by_index()'s
    folder.glob(f"*{index}*.pt") re-scan per sample. That per-sample glob
    is what made the "plain" condition time out: plain/features/model_c0p3_kl0
    has 58,136 files (full-dataset backfill) vs ~11,628 for every other
    condition (test-split only), so a per-sample directory re-scan against
    it was ~10x slower than the other conditions -- confirmed by comparing
    completed-run throughput (awgn_15dB: 3.18 it/s) against plain's
    (3.15 s/it, i.e. ~10x slower) before this fix.
    """
    lookup = {}
    for p in sorted(folder.glob("*.pt")):
        file_id = p.stem.split("_")[0]
        lookup[normalize_id(file_id)] = p
    return lookup


def compute_nll(x, weights, means, variances):
    """
    x: (H, W, T); weights/means/variances: (H, W, T, MAX_K) — already
    gathered to each position's optimal-k parameters, zero-padded.
    Returns nll: (H, W, T)
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


def summarize_nll_map(row, nll_map, prefix):
    vals = nll_map.reshape(-1).float()
    row[f"{prefix}_mean"]        = vals.mean().item()
    row[f"{prefix}_std"]         = vals.std(unbiased=True).item() if vals.numel() > 1 else 0.0
    row[f"{prefix}_min"]         = vals.min().item()
    row[f"{prefix}_max"]         = vals.max().item()
    row[f"{prefix}_median"]      = vals.median().item()
    row[f"{prefix}_q25"]         = torch.quantile(vals, 0.25).item()
    row[f"{prefix}_q75"]         = torch.quantile(vals, 0.75).item()
    row[f"{prefix}_q90"]         = torch.quantile(vals, 0.90).item()
    row[f"{prefix}_q95"]         = torch.quantile(vals, 0.95).item()
    row[f"{prefix}_q99"]         = torch.quantile(vals, 0.99).item()
    row[f"{prefix}_top100_mean"] = vals.topk(min(100, vals.numel())).values.mean().item()
    return row


def save_dataset_summary(result_df, condition, save_root):
    valid_df   = result_df[result_df["missing_feature"] == False]
    metric_col = "gmm_optimal_k_mean"

    if metric_col not in valid_df.columns or len(valid_df) == 0:
        return

    s = valid_df[metric_col].dropna()
    summary = {
        "condition":   condition,
        "distance":    "gmm_optimal_k",
        "num_samples": len(valid_df),
        "mean":        float(s.mean())         if len(s) > 0 else None,
        "std":         float(s.std(ddof=1))    if len(s) > 1 else 0.0,
        "q25":         float(s.quantile(0.25)) if len(s) > 0 else None,
        "q50":         float(s.quantile(0.50)) if len(s) > 0 else None,
        "q75":         float(s.quantile(0.75)) if len(s) > 0 else None,
        "q90":         float(s.quantile(0.90)) if len(s) > 0 else None,
    }
    pd.DataFrame([summary]).T.to_csv(save_root / f"{condition}_gmm_optimal_k_summary.csv")


# =========================================================
# STEP 1: LOAD GMM MODELS FOR ALL k (needed to build the gathered model)
# =========================================================
print("Loading GMM aggregate models ...")
gmm_models = {}
for k in range(1, MAX_K + 1):
    gmm_models[k] = {
        "weights":   torch.load(GMM_AGG_DIR / f"weights_k{k:02d}.pt",   map_location=DEVICE).float(),
        "means":     torch.load(GMM_AGG_DIR / f"means_k{k:02d}.pt",     map_location=DEVICE).float(),
        "variances": torch.load(GMM_AGG_DIR / f"variances_k{k:02d}.pt", map_location=DEVICE).float(),
    }

optimal_k_map = torch.load(GMM_AGG_DIR / "optimal_k.pt", map_location=DEVICE).int()  # (H, W, T)
print(f"Loaded optimal_k_map shape: {tuple(optimal_k_map.shape)}")

# =========================================================
# STEP 2: BUILD THE GATHERED OPTIMAL-K MODEL ONCE
# Padded to MAX_K components; unused slots get weight=0 (negligible in the
# log-sum-exp). Done once here, not per-sample.
# =========================================================
print("Building gathered optimal-k GMM (enables single-pass NLL per sample)...")
gathered_w = torch.zeros(FULL_H, FULL_W, FULL_T, MAX_K, dtype=torch.float32)
gathered_m = torch.zeros(FULL_H, FULL_W, FULL_T, MAX_K, dtype=torch.float32)
gathered_v = torch.ones(FULL_H, FULL_W, FULL_T, MAX_K, dtype=torch.float32)

for k in range(1, MAX_K + 1):
    mask = (optimal_k_map == k)
    if not mask.any():
        continue
    # NOTE: `gathered_w[mask, :k] = ...` (combining a multi-dim boolean mask
    # with a slice in one indexing expression) computes the wrong target
    # shape for assignment in PyTorch. Rebuilt using only torch.cat (to pad
    # this k's (H,W,T,k) arrays out to (H,W,T,MAX_K)) and torch.where (to
    # select per-position via broadcasting) -- neither op has any
    # advanced-indexing ambiguity, so there's no combined-indexing case to
    # get wrong.
    mask4 = mask.unsqueeze(-1)  # (H,W,T,1), broadcasts against the last dim

    pad_shape = (FULL_H, FULL_W, FULL_T, MAX_K - k)
    w_padded = torch.cat([gmm_models[k]["weights"],   torch.zeros(pad_shape)], dim=-1)
    m_padded = torch.cat([gmm_models[k]["means"],     torch.zeros(pad_shape)], dim=-1)
    v_padded = torch.cat([gmm_models[k]["variances"], torch.ones(pad_shape)],  dim=-1)

    gathered_w = torch.where(mask4, w_padded, gathered_w)
    gathered_m = torch.where(mask4, m_padded, gathered_m)
    gathered_v = torch.where(mask4, v_padded, gathered_v)

del gmm_models  # only the gathered tensors are needed from here on
print("Gathered optimal-k GMM built.")

# =========================================================
# STEP 3: EVALUATE THIS JOB'S CONDITION
# =========================================================
if not FEATURE_DIR.exists():
    print(f"\n[WARNING] Feature dir does not exist, nothing to do: {FEATURE_DIR}")
    raise SystemExit(0)

rows = []

if CONDITION == "random_noise":
    # Synthetic samples: no annotation CSV, no real IDs — evaluate every
    # file present in the feature dir directly.
    pt_files = sorted(FEATURE_DIR.glob("*.pt"))
    print(f"\n{CONDITION}: {len(pt_files)} synthetic samples found in {FEATURE_DIR}")

    for pt_path in tqdm(pt_files, desc=CONDITION):
        row = {
            "ID": pt_path.stem, "condition": CONDITION, "feature_file": pt_path.name,
            "missing_feature": False, "error": None,
        }
        try:
            x = torch.load(pt_path, map_location=DEVICE).float()
            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")
            nll = compute_nll(x, gathered_w, gathered_m, gathered_v)
            row = summarize_nll_map(row, nll, prefix="gmm_optimal_k")
        except Exception as e:
            row["missing_feature"] = True
            row["error"] = str(e)
        rows.append(row)

else:
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df = test_df[(test_df["is_corrupt"] == False) & (test_df["sample_9600"] == True)].copy()
    test_df["ID"] = test_df["ID"].astype(str)
    meta_cols = [c for c in ["ID", "Sample Class", "noise_level"] if c in test_df.columns]

    print(f"\n{CONDITION}: {len(test_df)} test rows (is_corrupt==False & sample_9600==True)")

    file_lookup = build_file_index(FEATURE_DIR)
    print(f"  Feature files found: {len(file_lookup)}")

    for _, meta_row in tqdm(test_df.iterrows(), total=len(test_df), desc=CONDITION):
        idx = normalize_id(meta_row["ID"])
        pt_path = file_lookup.get(idx)

        row = {c: meta_row[c] for c in meta_cols}
        row["condition"]       = CONDITION
        row["feature_file"]    = None
        row["missing_feature"] = False
        row["error"]           = None

        if pt_path is None:
            row["missing_feature"] = True
            rows.append(row)
            continue

        row["feature_file"] = pt_path.name

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()
            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")
            nll = compute_nll(x, gathered_w, gathered_m, gathered_v)
            row = summarize_nll_map(row, nll, prefix="gmm_optimal_k")
        except Exception as e:
            row["missing_feature"] = True
            row["error"]           = str(e)

        rows.append(row)

result_df = pd.DataFrame(rows)

per_sample_path = SAVE_ROOT / f"{CONDITION}_gmm_distances.csv"
result_df.to_csv(per_sample_path, index=False)
print(f"  Saved per-sample CSV -> {per_sample_path}")

save_dataset_summary(result_df, CONDITION, SAVE_ROOT)

print(f"\n[job {args.job_id}] Done. Condition: {CONDITION}")
