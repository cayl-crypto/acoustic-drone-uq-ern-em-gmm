#!/usr/bin/env python3
# gmm_scalar_distance_eval.py
#
# Evaluates GMM-based NLL distances on the val set (evaluation split)
# across all noise conditions: plain, 15dB, 5dB, -5dB.
#
# For each condition and each k in {1, ..., MAX_K} (plus optimal k):
#   - Computes per-sample NLL distance map [128, 128, 31]
#   - Summarises statistics (mean, std, quantiles) over the map values
#   - Saves per-sample CSV and dataset-level summary CSV
#
# Outputs go to ./gmm_scalar_eval_results/

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
FEATURE_DIRS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB":  Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB":   Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB":  Path("/path/to/DDLDataset/-5dB/features/model"),
}
GMM_MODEL_ROOT = Path("./gmm_scalar_models_1x1x1")
SAVE_ROOT      = Path("./gmm_scalar_eval_results")
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
def find_file_by_index(folder: Path, index: str):
    matches = sorted(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None


def compute_nll(x, weights, means, variances):
    """
    Args:
        x         : (H, W, T)    float32
        weights   : (H, W, T, k) float32
        means     : (H, W, T, k) float32
        variances : (H, W, T, k) float32

    Returns:
        nll : (H, W, T) float32
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


def save_dataset_summary(result_df, condition, k_label, save_root):
    prefix     = f"gmm_k{k_label}"
    valid_df   = result_df[result_df["missing_feature"] == False]
    metric_col = f"{prefix}_mean"

    if metric_col not in valid_df.columns or len(valid_df) == 0:
        return

    s = valid_df[metric_col].dropna()
    summary = {
        "condition":   condition,
        "k":           k_label,
        "num_samples": len(valid_df),
        "mean":        float(s.mean())              if len(s) > 0 else None,
        "std":         float(s.std(ddof=1))         if len(s) > 1 else 0.0,
        "q25":         float(s.quantile(0.25))      if len(s) > 0 else None,
        "q50":         float(s.quantile(0.50))      if len(s) > 0 else None,
        "q75":         float(s.quantile(0.75))      if len(s) > 0 else None,
        "q90":         float(s.quantile(0.90))      if len(s) > 0 else None,
    }
    pd.DataFrame([summary]).T.to_csv(
        save_root / f"{condition}_k{k_label}_summary.csv"
    )


# =========================================================
# STEP 1: LOAD GMM MODELS FOR ALL k INTO MEMORY
# =========================================================
print("Loading GMM models ...")
gmm_models = {}
for k in range(1, MAX_K + 1):
    k_dir = GMM_MODEL_ROOT / f"k{k}"
    gmm_models[k] = {
        "weights":   torch.load(k_dir / "weights.pt",   map_location=DEVICE).float(),
        "means":     torch.load(k_dir / "means.pt",     map_location=DEVICE).float(),
        "variances": torch.load(k_dir / "variances.pt", map_location=DEVICE).float(),
    }
    print(f"  Loaded k={k}")

optimal_k_map = torch.load(
    GMM_MODEL_ROOT / "optimal_k_map.pt", map_location=DEVICE
).int()  # (H, W, T)
print(f"Loaded optimal_k_map  shape: {tuple(optimal_k_map.shape)}")

# =========================================================
# STEP 2: LOAD EVALUATION IDS
# =========================================================
val_df = pd.read_csv(VAL_CSV_PATH)
val_df["ID"] = val_df["ID"].astype(str)

if "conformal_role" in val_df.columns:
    eval_df = val_df[val_df["conformal_role"] == "evaluation"].copy()
else:
    eval_df = val_df.copy()

eval_df["ID"] = eval_df["ID"].astype(str)
meta_cols = [c for c in ["ID", "label", "Label", "split", "conformal_role"] if c in eval_df.columns]

print(f"\nTotal val rows       : {len(val_df)}")
print(f"Evaluation rows      : {len(eval_df)}")

# =========================================================
# STEP 3: EVALUATE PER CONDITION
# =========================================================
for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")
    rows = []

    for _, meta_row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=condition):
        idx    = str(meta_row["ID"])
        pt_path = find_file_by_index(feature_dir, idx)

        row = {c: meta_row[c] for c in meta_cols}
        row["condition"]       = condition
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

            # NLL for each fixed k
            for k in range(1, MAX_K + 1):
                m   = gmm_models[k]
                nll = compute_nll(x, m["weights"], m["means"], m["variances"])
                row = summarize_nll_map(row, nll, prefix=f"gmm_k{k}")

            # NLL using per-position optimal k
            nll_opt = torch.zeros(FULL_H, FULL_W, FULL_T, dtype=torch.float32)
            for k in range(1, MAX_K + 1):
                mask = (optimal_k_map == k)
                if not mask.any():
                    continue
                m   = gmm_models[k]
                nll = compute_nll(x, m["weights"], m["means"], m["variances"])
                nll_opt[mask] = nll[mask]

            row = summarize_nll_map(row, nll_opt, prefix="gmm_optimal_k")

        except Exception as e:
            row["missing_feature"] = True
            row["error"]           = str(e)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Per-sample CSV
    per_sample_path = SAVE_ROOT / f"{condition}_gmm_distances.csv"
    result_df.to_csv(per_sample_path, index=False)
    print(f"  Saved per-sample CSV -> {per_sample_path}")

    # Dataset summary per k
    for k in range(1, MAX_K + 1):
        save_dataset_summary(result_df, condition, k, SAVE_ROOT)
    save_dataset_summary(result_df, condition, "optimal_k", SAVE_ROOT)

print(f"\nAll evaluation outputs saved -> {SAVE_ROOT.resolve()}")
