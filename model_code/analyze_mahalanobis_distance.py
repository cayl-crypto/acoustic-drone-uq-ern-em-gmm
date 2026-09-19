from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
VAL_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/val_dataset_with_nondrone_with_9600_flag.csv"
)

FEATURE_DIRS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB": Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB": Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB": Path("/path/to/DDLDataset/-5dB/features/model"),
}

MU_PATH = Path("mu.pt")
COV_INV_PATH = Path("cov_inv.pt")

SAVE_ROOT = Path("./mahalanobis_eval_results")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index_str: str):
    matches = sorted(folder.glob(f"*{int(index_str):08d}*.pt"))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"[WARNING] Multiple matches in {folder} for index {index_str}:")
        for m in matches:
            print(f"  - {m.name}")

    return matches[0]


def feature_to_vector(x: torch.Tensor) -> torch.Tensor:
    """
    Input:
        x: [128, 128, 31]
    Output:
        v: [128]

    Uses spatial mean per channel, consistent with the
    previously suggested Mahalanobis formulation.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [128,128,31]-like tensor, got shape {tuple(x.shape)}")

    x = x.reshape(x.shape[0], -1)   # [128, 3968]
    v = x.mean(dim=1)               # [128]
    return v


def mahalanobis_distance_from_vector(v: torch.Tensor, mu: torch.Tensor, cov_inv: torch.Tensor) -> float:
    diff = v - mu
    dist_sq = diff @ cov_inv @ diff
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq).item()


def summarize_distances(df: pd.DataFrame, condition: str, save_dir: Path):
    summary = {
        "condition": condition,
        "num_samples": len(df),
        "num_missing": int(df["missing_feature"].sum()) if "missing_feature" in df.columns else 0,
    }

    valid = df[df["missing_feature"] == False].copy()

    if len(valid) > 0:
        d = valid["mahalanobis_distance"]
        summary.update({
            "min": float(d.min()),
            "max": float(d.max()),
            "mean": float(d.mean()),
            "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "median": float(d.median()),
            "q10": float(d.quantile(0.10)),
            "q25": float(d.quantile(0.25)),
            "q50": float(d.quantile(0.50)),
            "q75": float(d.quantile(0.75)),
            "q90": float(d.quantile(0.90)),
        })
    else:
        summary.update({
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
        })

    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = [condition]
    summary_df.to_csv(save_dir / f"{condition}_summary.csv")


# =========================================================
# LOAD MAHALANOBIS STATS
# =========================================================
mu = torch.load(MU_PATH, map_location=DEVICE).float()
cov_inv = torch.load(COV_INV_PATH, map_location=DEVICE).float()

print(f"Loaded mu shape     : {tuple(mu.shape)}")
print(f"Loaded cov_inv shape: {tuple(cov_inv.shape)}")


# =========================================================
# LOAD EVALUATION IDS
# =========================================================
val_df = pd.read_csv(VAL_CSV_PATH)

eval_df = val_df[val_df["conformal_role"] == "evaluation"].copy()
eval_df["ID"] = eval_df["ID"].astype(str)

eval_ids = eval_df["ID"].tolist()

print(f"Total validation rows     : {len(val_df)}")
print(f"Evaluation rows selected  : {len(eval_df)}")


# =========================================================
# PROCESS EACH CONDITION
# =========================================================
merged_frames = []

for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")
    rows = []

    for idx in tqdm(eval_ids, desc=condition):
        pt_path = find_file_by_index(feature_dir, idx)

        if pt_path is None:
            rows.append({
                "ID": idx,
                "feature_file": None,
                "missing_feature": True,
                "mahalanobis_distance": None,
            })
            continue

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()
            v = feature_to_vector(x)
            dist = mahalanobis_distance_from_vector(v, mu, cov_inv)

            rows.append({
                "ID": idx,
                "feature_file": pt_path.name,
                "missing_feature": False,
                "mahalanobis_distance": dist,
            })

        except Exception as e:
            rows.append({
                "ID": idx,
                "feature_file": pt_path.name,
                "missing_feature": True,
                "mahalanobis_distance": None,
                "error": str(e),
            })

    result_df = pd.DataFrame(rows)

    # save per-condition csv
    save_path = SAVE_ROOT / f"{condition}_mahalanobis.csv"
    result_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    # save summary
    summarize_distances(result_df, condition, SAVE_ROOT)
    print(f"Saved summary: {SAVE_ROOT / f'{condition}_summary.csv'}")

    # prepare merged
    merged_part = result_df[["ID", "mahalanobis_distance"]].rename(
        columns={"mahalanobis_distance": f"mahalanobis_{condition}"}
    )
    merged_frames.append(merged_part)


# =========================================================
# SAVE MERGED CSV
# =========================================================
if merged_frames:
    merged_df = merged_frames[0]
    for part in merged_frames[1:]:
        merged_df = merged_df.merge(part, on="ID", how="outer")

    merged_df.to_csv(SAVE_ROOT / "all_conditions_mahalanobis.csv", index=False)
    print(f"Saved merged CSV: {SAVE_ROOT / 'all_conditions_mahalanobis.csv'}")

print(f"\nAll outputs saved to: {SAVE_ROOT.resolve()}")