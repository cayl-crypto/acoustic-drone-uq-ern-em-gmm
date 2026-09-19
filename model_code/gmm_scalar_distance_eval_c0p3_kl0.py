#!/usr/bin/env python3
# gmm_scalar_distance_eval_c0p3_kl0.py
#
# Documenting-results step: evaluates GMM-based NLL distances on the val
# set's "evaluation" split (conformal_role == "evaluation"), plain
# condition only, for the c0p3_kl0 checkpoint.
#
# optimal_k ONLY (not gmm_k1..gmm_k10) — same speed fix as
# gmm_test_noise_distance_eval_c0p3_kl0.py: the per-position optimal-k GMM
# parameters are gathered into a single padded (H, W, T, MAX_K) tensor once
# at startup, so each sample only needs one NLL pass instead of up to 10.
# This also sidesteps the original f"gmm_k{k_label}" prefix bug (inherited
# from the c0p1 pipeline) that silently dropped the optimal_k dataset
# summary — with only one distance type there's no ambiguous prefix left.
#
# Single job (not sharded) — only one condition (plain) and ~2,904 samples,
# well within a couple of hours even before this optimization.
#
# Outputs go to ./gmm_scalar_1x1x1_c0p3_kl0/eval_results/

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
FEATURE_DIR = Path("/path/to/DDLDataset/plain/features/model_c0p3_kl0")
CONDITION = "plain"

GMM_ROOT    = Path("./gmm_scalar_1x1x1_c0p3_kl0")
GMM_AGG_DIR = GMM_ROOT / "aggregate"
SAVE_ROOT   = GMM_ROOT / "eval_results"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

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
def normalize_id(x):
    digits = "".join(ch for ch in str(x).strip() if ch.isdigit())
    return digits.zfill(8)


def build_file_index(folder: Path):
    """
    O(1)-lookup dict built once, instead of find_file_by_index()'s
    folder.glob(f"*{index}*.pt") re-scan per sample. That per-sample glob
    against plain/features/model_c0p3_kl0 (58,136 files -- full-dataset
    backfill, not just this ~2,904-sample val split) was slow enough that
    the equivalent lookup pattern in gmm_test_noise_distance_eval_c0p3_kl0.py
    timed out its "plain" condition at ~10x the throughput of every other
    (smaller, ~11,628-file) condition dir. Same fix applied here.
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
    row[f"{prefix}_mean"]       = vals.mean().item()
    row[f"{prefix}_std"]        = vals.std(unbiased=True).item() if vals.numel() > 1 else 0.0
    row[f"{prefix}_min"]        = vals.min().item()
    row[f"{prefix}_max"]        = vals.max().item()
    row[f"{prefix}_median"]     = vals.median().item()
    row[f"{prefix}_q25"]        = torch.quantile(vals, 0.25).item()
    row[f"{prefix}_q75"]        = torch.quantile(vals, 0.75).item()
    row[f"{prefix}_q90"]        = torch.quantile(vals, 0.90).item()
    row[f"{prefix}_q95"]        = torch.quantile(vals, 0.95).item()
    row[f"{prefix}_q99"]        = torch.quantile(vals, 0.99).item()
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
        "mean":        float(s.mean())              if len(s) > 0 else None,
        "std":         float(s.std(ddof=1))         if len(s) > 1 else 0.0,
        "q25":         float(s.quantile(0.25))      if len(s) > 0 else None,
        "q50":         float(s.quantile(0.50))      if len(s) > 0 else None,
        "q75":         float(s.quantile(0.75))      if len(s) > 0 else None,
        "q90":         float(s.quantile(0.90))      if len(s) > 0 else None,
    }
    pd.DataFrame([summary]).T.to_csv(save_root / f"{condition}_gmm_optimal_k_summary.csv")


# =========================================================
# STEP 1: LOAD GMM MODELS FOR ALL k (needed to build the gathered model)
# =========================================================
print("Loading GMM models ...")
gmm_models = {}
for k in range(1, MAX_K + 1):
    gmm_models[k] = {
        "weights":   torch.load(GMM_AGG_DIR / f"weights_k{k:02d}.pt",   map_location=DEVICE).float(),
        "means":     torch.load(GMM_AGG_DIR / f"means_k{k:02d}.pt",     map_location=DEVICE).float(),
        "variances": torch.load(GMM_AGG_DIR / f"variances_k{k:02d}.pt", map_location=DEVICE).float(),
    }

optimal_k_map = torch.load(
    GMM_AGG_DIR / "optimal_k.pt", map_location=DEVICE
).int()  # (H, W, T)
print(f"Loaded optimal_k_map  shape: {tuple(optimal_k_map.shape)}")

# =========================================================
# STEP 2: BUILD THE GATHERED OPTIMAL-K MODEL ONCE
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

del gmm_models
print("Gathered optimal-k GMM built.")

# =========================================================
# STEP 3: LOAD EVALUATION IDS
# =========================================================
val_df = pd.read_csv(VAL_CSV_PATH)
val_df["ID"] = val_df["ID"].astype(str)

if "conformal_role" in val_df.columns:
    eval_df = val_df[val_df["conformal_role"] == "evaluation"].copy()
else:
    eval_df = val_df.copy()

eval_df["ID"] = eval_df["ID"].astype(str)
meta_cols = [c for c in ["ID", "label", "Label", "split", "conformal_role", "noise_level"]
             if c in eval_df.columns]

print(f"\nTotal val rows (all noise_level values)     : {len(val_df)}")
print(f"Evaluation rows (all noise_level values)     : {len(eval_df)}")

# =========================================================
# STEP 4: EVALUATE
# =========================================================
file_lookup = build_file_index(FEATURE_DIR)
print(f"Feature files found: {len(file_lookup)}")

rows = []

for _, meta_row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=CONDITION):
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

print(f"\nAll evaluation outputs saved -> {SAVE_ROOT.resolve()}")
