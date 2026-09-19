#!/usr/bin/env python3
# scalar_1x1x1_fit_and_eval.py

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
TRAIN_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/train_dataset_with_nondrone_with_9600_flag.csv"
)

VAL_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/val_dataset_with_nondrone_with_9600_flag.csv"
)

TRAIN_FEATURE_DIR = Path(
    "/path/to/DDLDataset/plain/features/model"
)

FEATURE_DIRS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB": Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB": Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB": Path("/path/to/DDLDataset/-5dB/features/model"),
}

SCALAR_MODEL_ROOT = Path("./scalar_models_1x1x1")
SAVE_ROOT = Path("./scalar_1x1x1_eval_results")

SCALAR_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
DTYPE = torch.float64

# =========================================================
# FEATURE CONFIG
# =========================================================
FULL_H = 128
FULL_W = 128
FULL_T = 31
EPS = 1e-6


# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index: str):
    matches = sorted(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None


def summarize_distance_map(row: dict, distance_map: torch.Tensor, prefix: str):
    vals = distance_map.reshape(-1).float()

    row[f"{prefix}_mean"] = vals.mean().item()
    row[f"{prefix}_std"] = vals.std(unbiased=True).item() if vals.numel() > 1 else 0.0
    row[f"{prefix}_min"] = vals.min().item()
    row[f"{prefix}_max"] = vals.max().item()
    row[f"{prefix}_median"] = vals.median().item()

    row[f"{prefix}_q25"] = torch.quantile(vals, 0.25).item()
    row[f"{prefix}_q50"] = torch.quantile(vals, 0.50).item()
    row[f"{prefix}_q75"] = torch.quantile(vals, 0.75).item()
    row[f"{prefix}_q90"] = torch.quantile(vals, 0.90).item()
    row[f"{prefix}_q95"] = torch.quantile(vals, 0.95).item()
    row[f"{prefix}_q99"] = torch.quantile(vals, 0.99).item()

    row[f"{prefix}_top5_mean"] = vals.topk(min(5, vals.numel())).values.mean().item()
    row[f"{prefix}_top10_mean"] = vals.topk(min(10, vals.numel())).values.mean().item()
    row[f"{prefix}_top100_mean"] = vals.topk(min(100, vals.numel())).values.mean().item()
    row[f"{prefix}_top1000_mean"] = vals.topk(min(1000, vals.numel())).values.mean().item()

    return row


def save_dataset_summary(result_df: pd.DataFrame, condition: str, save_root: Path):
    summary = {
        "condition": condition,
        "num_rows": len(result_df),
        "num_missing": int(result_df["missing_feature"].sum()),
    }

    valid_df = result_df[result_df["missing_feature"] == False].copy()

    metric_cols = [
        c for c in valid_df.columns
        if c.startswith(("manhattan_", "euclidean_", "mahalanobis_"))
    ]

    for metric in metric_cols:
        s = valid_df[metric].dropna()

        summary[f"{metric}_mean"] = float(s.mean()) if len(s) > 0 else None
        summary[f"{metric}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0 if len(s) == 1 else None
        summary[f"{metric}_q25"] = float(s.quantile(0.25)) if len(s) > 0 else None
        summary[f"{metric}_q50"] = float(s.quantile(0.50)) if len(s) > 0 else None
        summary[f"{metric}_q75"] = float(s.quantile(0.75)) if len(s) > 0 else None

    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = [condition]

    save_path = save_root / f"{condition}_summary.csv"
    summary_df.to_csv(save_path)
    print(f"Saved summary: {save_path}")


# =========================================================
# STEP 1: FIT SCALAR 1x1x1 MEAN AND STD FROM TRAINING SET
# =========================================================
mu_path = SCALAR_MODEL_ROOT / "mu.pt"
var_path = SCALAR_MODEL_ROOT / "var.pt"
std_path = SCALAR_MODEL_ROOT / "std.pt"

if mu_path.exists() and std_path.exists():
    print("Loading existing scalar 1x1x1 statistics...")

    mu = torch.load(mu_path, map_location=DEVICE).float()
    std = torch.load(std_path, map_location=DEVICE).float()

else:
    print("Fitting scalar 1x1x1 statistics from training set...")

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_ids = train_df["ID"].astype(str).tolist()

    print(f"Total IDs in training CSV: {len(train_ids)}")

    sum_x = torch.zeros((FULL_H, FULL_W, FULL_T), dtype=DTYPE, device=DEVICE)
    sum_x2 = torch.zeros((FULL_H, FULL_W, FULL_T), dtype=DTYPE, device=DEVICE)

    valid_sample_count = 0
    missing_train_count = 0

    for idx in tqdm(train_ids, desc="Fitting scalar mean/std"):
        pt_path = find_file_by_index(TRAIN_FEATURE_DIR, idx)

        if pt_path is None:
            missing_train_count += 1
            continue

        x = torch.load(pt_path, map_location=DEVICE).to(DTYPE)

        if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
            print(f"[WARNING] Unexpected shape for {pt_path.name}: {tuple(x.shape)}")
            continue

        sum_x += x
        sum_x2 += x * x
        valid_sample_count += 1

    if valid_sample_count < 2:
        raise ValueError("Need at least 2 valid training feature files.")

    mu = sum_x / valid_sample_count

    var = (sum_x2 - valid_sample_count * mu * mu) / (valid_sample_count - 1)
    var = torch.clamp(var, min=0.0)

    std = torch.sqrt(var)
    std = torch.clamp(std, min=EPS)

    torch.save(mu.float().cpu(), mu_path)
    torch.save(var.float().cpu(), var_path)
    torch.save(std.float().cpu(), std_path)

    scalar_summary = pd.DataFrame([{
        "model_type": "scalar_1x1x1",
        "feature_shape": f"({FULL_H}, {FULL_W}, {FULL_T})",
        "num_scalar_positions": FULL_H * FULL_W * FULL_T,
        "valid_training_samples": valid_sample_count,
        "missing_training_files": missing_train_count,
        "eps": EPS,
    }])

    scalar_summary.to_csv(SCALAR_MODEL_ROOT / "scalar_summary.csv", index=False)

    print(f"Valid training samples used: {valid_sample_count}")
    print(f"Missing training files     : {missing_train_count}")
    print(f"Saved mu : {mu_path}")
    print(f"Saved var: {var_path}")
    print(f"Saved std: {std_path}")

    mu = mu.float()
    std = std.float()


if tuple(mu.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected mu shape: {tuple(mu.shape)}")

if tuple(std.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected std shape: {tuple(std.shape)}")

std = torch.clamp(std, min=EPS)


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

print(f"\nTotal validation rows    : {len(val_df)}")
print(f"Evaluation rows selected : {len(eval_df)}")

meta_cols = [
    c for c in ["ID", "label", "Label", "split", "conformal_role"]
    if c in eval_df.columns
]

eval_meta = eval_df[meta_cols].copy()


# =========================================================
# STEP 3: EVALUATE DISTANCES FOR EACH CONDITION
# =========================================================
for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")

    rows = []

    for _, meta_row in tqdm(eval_meta.iterrows(), total=len(eval_meta), desc=condition):
        idx = str(meta_row["ID"])
        pt_path = find_file_by_index(feature_dir, idx)

        row = {col: meta_row[col] for col in meta_cols}
        row["condition"] = condition
        row["feature_file"] = None
        row["missing_feature"] = False
        row["error"] = None

        if pt_path is None:
            row["missing_feature"] = True
            rows.append(row)
            continue

        row["feature_file"] = pt_path.name

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()

            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

            diff = x - mu

            # For scalar 1D locations:
            # Manhattan distance = |x - mu|
            # Euclidean distance = |x - mu|
            # Mahalanobis distance = |x - mu| / std
            manhattan_map = torch.abs(diff)
            euclidean_map = torch.abs(diff)
            mahalanobis_map = torch.abs(diff) / std

            row = summarize_distance_map(row, manhattan_map, "manhattan")
            row = summarize_distance_map(row, euclidean_map, "euclidean")
            row = summarize_distance_map(row, mahalanobis_map, "mahalanobis")

        except Exception as e:
            row["missing_feature"] = True
            row["error"] = str(e)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    save_path = SAVE_ROOT / f"{condition}_scalar_1x1x1_distances.csv"
    result_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    save_dataset_summary(result_df, condition, SAVE_ROOT)


print(f"\nAll evaluation outputs saved to: {SAVE_ROOT.resolve()}")
print(f"Scalar models saved to       : {SCALAR_MODEL_ROOT.resolve()}")