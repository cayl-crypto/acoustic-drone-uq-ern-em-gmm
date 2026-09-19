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

BLOCK_MODEL_ROOT = Path("./mahalanobis_blocks_1x1x31")
SAVE_ROOT = Path("./mahalanobis_block_eval_results")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"

# =========================================================
# BLOCK CONFIG
# =========================================================
BLOCK_H = 1
BLOCK_W = 1
FULL_H = 128
FULL_W = 128
FULL_T = 31

N_BLOCK_H = FULL_H // BLOCK_H   # 16
N_BLOCK_W = FULL_W // BLOCK_W   # 16
N_BLOCKS = N_BLOCK_H * N_BLOCK_W
BLOCK_DIM = BLOCK_H * BLOCK_W * FULL_T  # 1984


# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index_str: str):
    matches = sorted(folder.glob(f"*{index_str}*.pt"))
    return matches[0] if matches else None


def block_name(bi: int, bj: int) -> str:
    return f"block_r{bi:02d}_c{bj:02d}"


def extract_block(x: torch.Tensor, bi: int, bj: int) -> torch.Tensor:
    r0 = bi * BLOCK_H
    r1 = r0 + BLOCK_H
    c0 = bj * BLOCK_W
    c1 = c0 + BLOCK_W
    return x[r0:r1, c0:c1, :]


def flatten_block(block: torch.Tensor) -> torch.Tensor:
    return block.reshape(-1)


def mahalanobis_distance(vec: torch.Tensor, mu: torch.Tensor, cov_inv: torch.Tensor) -> float:
    diff = vec - mu
    dist_sq = diff @ cov_inv @ diff
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq).item()


def load_block_models(block_root: Path, device: str = "cpu"):
    block_models = {}

    for bi in range(N_BLOCK_H):
        for bj in range(N_BLOCK_W):
            name = block_name(bi, bj)
            folder = block_root / name

            mu_path = folder / "mu.pt"
            cov_inv_path = folder / "cov_inv.pt"

            if not mu_path.exists() or not cov_inv_path.exists():
                raise FileNotFoundError(f"Missing block model files in: {folder}")

            mu = torch.load(mu_path, map_location=device).float()
            cov_inv = torch.load(cov_inv_path, map_location=device).float()

            block_models[(bi, bj)] = {
                "name": name,
                "mu": mu,
                "cov_inv": cov_inv,
            }

    return block_models


def summarize_block_distances(row: dict, block_cols: list):
    vals = [row[c] for c in block_cols if pd.notna(row[c])]
    if len(vals) == 0:
        row["md_mean"] = None
        row["md_std"] = None
        row["md_min"] = None
        row["md_max"] = None
        row["md_median"] = None
        row["md_top5_mean"] = None
        row["md_top10_mean"] = None
        return row

    s = pd.Series(vals, dtype=float)
    row["md_mean"] = float(s.mean())
    row["md_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    row["md_min"] = float(s.min())
    row["md_max"] = float(s.max())
    row["md_median"] = float(s.median())
    row["md_top5_mean"] = float(s.nlargest(min(5, len(s))).mean())
    row["md_top10_mean"] = float(s.nlargest(min(10, len(s))).mean())
    return row


# =========================================================
# LOAD EVALUATION IDS
# =========================================================
val_df = pd.read_csv(VAL_CSV_PATH)
eval_df = val_df[val_df["conformal_role"] == "evaluation"].copy()
eval_df["ID"] = eval_df["ID"].astype(str)

eval_ids = eval_df["ID"].tolist()

print(f"Total validation rows    : {len(val_df)}")
print(f"Evaluation rows selected : {len(eval_df)}")

# optional metadata columns to carry into output if present
meta_cols = [c for c in ["ID", "label", "Label", "split", "conformal_role"] if c in eval_df.columns]
eval_meta = eval_df[meta_cols].copy()

# =========================================================
# LOAD BLOCK MODELS
# =========================================================
print("Loading 256 block Mahalanobis models...")
block_models = load_block_models(BLOCK_MODEL_ROOT, device=DEVICE)
print(f"Loaded {len(block_models)} block models.")

block_cols = [block_models[(bi, bj)]["name"] for bi in range(N_BLOCK_H) for bj in range(N_BLOCK_W)]

# =========================================================
# PROCESS EACH CONDITION
# =========================================================
for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")
    rows = []

    for _, meta_row in tqdm(eval_meta.iterrows(), total=len(eval_meta), desc=condition):
        idx = str(meta_row["ID"])
        pt_path = find_file_by_index(feature_dir, idx)

        row = {col: meta_row[col] for col in meta_cols}
        row["feature_file"] = None
        row["missing_feature"] = False

        # initialize block distance columns
        for col in block_cols:
            row[col] = None

        if pt_path is None:
            row["missing_feature"] = True
            row = summarize_block_distances(row, block_cols)
            rows.append(row)
            continue

        row["feature_file"] = pt_path.name

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()

            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

            for bi in range(N_BLOCK_H):
                for bj in range(N_BLOCK_W):
                    model_info = block_models[(bi, bj)]
                    col_name = model_info["name"]

                    block = extract_block(x, bi, bj)
                    vec = flatten_block(block)

                    dist = mahalanobis_distance(
                        vec=vec,
                        mu=model_info["mu"],
                        cov_inv=model_info["cov_inv"],
                    )

                    row[col_name] = dist

        except Exception as e:
            row["missing_feature"] = True
            row["error"] = str(e)

        row = summarize_block_distances(row, block_cols)
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # save one row per sample with 256 distances
    save_path = SAVE_ROOT / f"{condition}_block_mahalanobis.csv"
    result_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    # save transposed summary over dataset
    summary = {
        "condition": condition,
        "num_rows": len(result_df),
        "num_missing": int(result_df["missing_feature"].sum()),
    }

    valid_df = result_df[result_df["missing_feature"] == False].copy()

    if len(valid_df) > 0:
        for metric in ["md_mean", "md_std", "md_min", "md_max", "md_median", "md_top5_mean", "md_top10_mean"]:
            s = valid_df[metric].dropna()
            summary[f"{metric}_mean"] = float(s.mean()) if len(s) > 0 else None
            summary[f"{metric}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0 if len(s) == 1 else None
            summary[f"{metric}_q25"] = float(s.quantile(0.25)) if len(s) > 0 else None
            summary[f"{metric}_q50"] = float(s.quantile(0.50)) if len(s) > 0 else None
            summary[f"{metric}_q75"] = float(s.quantile(0.75)) if len(s) > 0 else None

    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = [condition]
    summary_df.to_csv(SAVE_ROOT / f"{condition}_summary.csv")
    print(f"Saved summary: {SAVE_ROOT / f'{condition}_summary.csv'}")

print(f"\nAll outputs saved to: {SAVE_ROOT.resolve()}")