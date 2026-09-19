#!/usr/bin/env python3
# apply_conformal_bands_to_test_1x1x1.py

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# =========================================================
# PATHS
# =========================================================
TEST_CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/test_dataset_with_nondrone_with_9600_flag.csv"
)

FEATURE_DIRS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB": Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB": Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB": Path("/path/to/DDLDataset/-5dB/features/model"),
}

SCALAR_MODEL_ROOT = Path("./scalar_models_1x1x1")

QUANTILE_ROOT = Path(
    "/path/to/DDLDataset/plain/distances/model/conformal_quantiles_1x1x1"
)

SAVE_ROOT = Path(
    "/path/to/DDLDataset/conformal_bands/test/model"
)

#DISTANCE_NAMES = ["manhattan", "euclidean", "mahalanobis"]
DISTANCE_NAMES = ["mahalanobis"]

DEVICE = "cpu"

FULL_H = 128
FULL_W = 128
FULL_T = 31
EPS = 1e-6

SAVE_PNG = False

# Band encoding:
# low       = 0
# med       = 1
# high      = 2
# very_high = 3


# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index: str):
    matches = sorted(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None


def compute_distance_map(x, mu, std, distance_name):
    diff = x - mu

    if distance_name == "manhattan":
        return torch.abs(diff)

    elif distance_name == "euclidean":
        return torch.abs(diff)

    elif distance_name == "mahalanobis":
        return torch.abs(diff) / std

    else:
        raise ValueError(f"Unknown distance name: {distance_name}")


def assign_bands(distance_map, q683, q950, q997):
    """
    Output:
        band_map shape: [128, 128, 31]
        low       = 0
        med       = 1
        high      = 2
        very_high = 3
    """
    band_map = torch.zeros_like(distance_map, dtype=torch.uint8)

    band_map[distance_map > q683] = 1
    band_map[distance_map > q950] = 2
    band_map[distance_map > q997] = 3

    return band_map


def save_band_grid_png(band_map, save_path, title=None):
    """
    Saves 31 channels as a grid of color maps.
    Green  = low
    Yellow = med
    Red    = high
    Purple = very_high
    """
    band_np = band_map.cpu().numpy()

    cmap = ListedColormap([
        "green",   # 0 low
        "yellow",  # 1 med
        "red",     # 2 high
        "purple",  # 3 very_high
    ])

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    n_rows = 4
    n_cols = 8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = axes.flatten()

    for t in range(FULL_T):
        ax = axes[t]
        ax.imshow(band_np[:, :, t], cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"ch {t:02d}", fontsize=8)
        ax.axis("off")

    # Hide unused 32nd subplot
    for k in range(FULL_T, len(axes)):
        axes[k].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_band_map(band_map):
    vals = band_map.reshape(-1)

    total = vals.numel()

    low = int((vals == 0).sum().item())
    med = int((vals == 1).sum().item())
    high = int((vals == 2).sum().item())
    very_high = int((vals == 3).sum().item())

    return {
        "num_positions": total,

        "low_count": low,
        "med_count": med,
        "high_count": high,
        "very_high_count": very_high,

        "low_pct": 100.0 * low / total,
        "med_pct": 100.0 * med / total,
        "high_pct": 100.0 * high / total,
        "very_high_pct": 100.0 * very_high / total,
    }

def normalize_id(x):
    digits = "".join(ch for ch in str(x).strip() if ch.isdigit())
    return digits.zfill(8)


def build_file_index(folder: Path):
    files = sorted(folder.glob("*.pt"))
    lookup = {}

    for p in files:
        # examples:
        # 00009689.pt
        # 00041143_snr15dB.pt
        # 00020961_snr-5dB.pt
        file_id = p.stem.split("_")[0]
        lookup[normalize_id(file_id)] = p

    return lookup
# =========================================================
# LOAD TEST IDS
# =========================================================
test_df = pd.read_csv(TEST_CSV_PATH)
test_df["ID"] = test_df["ID"].apply(normalize_id)

test_ids = test_df["ID"].tolist()

meta_cols = [
    c for c in ["ID", "label", "Label", "split", "conformal_role"]
    if c in test_df.columns
]

print(f"Total test samples: {len(test_ids)}")


# =========================================================
# LOAD SCALAR MEAN / STD
# =========================================================
mu = torch.load(SCALAR_MODEL_ROOT / "mu.pt", map_location=DEVICE).float()
std = torch.load(SCALAR_MODEL_ROOT / "std.pt", map_location=DEVICE).float()
std = torch.clamp(std, min=EPS)

if tuple(mu.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected mu shape: {tuple(mu.shape)}")

if tuple(std.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected std shape: {tuple(std.shape)}")


# =========================================================
# LOAD QUANTILES
# =========================================================
quantiles = {}

for distance_name in DISTANCE_NAMES:
    q683 = torch.load(QUANTILE_ROOT / f"{distance_name}_q683.pt", map_location=DEVICE).float()
    q950 = torch.load(QUANTILE_ROOT / f"{distance_name}_q950.pt", map_location=DEVICE).float()
    q997 = torch.load(QUANTILE_ROOT / f"{distance_name}_q997.pt", map_location=DEVICE).float()

    for q in [q683, q950, q997]:
        if tuple(q.shape) != (FULL_H, FULL_W, FULL_T):
            raise ValueError(f"Unexpected quantile shape for {distance_name}: {tuple(q.shape)}")

    quantiles[distance_name] = {
        "q683": q683,
        "q950": q950,
        "q997": q997,
    }

print("Loaded scalar models and conformal quantiles.")


# =========================================================
# APPLY BANDS TO TEST SET
# =========================================================
all_summary_rows = []

for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")
    file_lookup = build_file_index(feature_dir)
    print("Example CSV IDs :", test_df["ID"].head().tolist())
    print("Example file IDs:", list(file_lookup.keys())[:5])
    print("Matched examples:", sum(i in file_lookup for i in test_df["ID"].head(5)))
    for distance_name in DISTANCE_NAMES:
        (SAVE_ROOT / condition / distance_name / "pt").mkdir(parents=True, exist_ok=True)
        (SAVE_ROOT / condition / distance_name / "png").mkdir(parents=True, exist_ok=True)

    for _, meta_row in tqdm(test_df.iterrows(), total=len(test_df), desc=condition):
        #idx = str(meta_row["ID"])
        idx = meta_row["ID"]
        # pt_path = find_file_by_index(feature_dir, idx)
        pt_path = file_lookup.get(idx)

        base_row = {
            c: meta_row[c]
            for c in meta_cols
        }

        base_row.update({
            "condition": condition,
            "ID": idx,
            "feature_file": None,
            "missing_feature": False,
            "error": None,
        })

        if pt_path is None:
            for distance_name in DISTANCE_NAMES:
                row = dict(base_row)
                row["distance"] = distance_name
                row["missing_feature"] = True
                all_summary_rows.append(row)
            continue

        try:
            x = torch.load(pt_path, map_location=DEVICE).float()

            if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
                raise ValueError(f"Unexpected feature shape: {tuple(x.shape)}")

            for distance_name in DISTANCE_NAMES:
                distance_map = compute_distance_map(
                    x=x,
                    mu=mu,
                    std=std,
                    distance_name=distance_name,
                )

                q = quantiles[distance_name]

                band_map = assign_bands(
                    distance_map=distance_map,
                    q683=q["q683"],
                    q950=q["q950"],
                    q997=q["q997"],
                )

                out_name = pt_path.stem

                pt_save_path = (
                    SAVE_ROOT / condition / distance_name / "pt" / f"{out_name}_bands.pt"
                )

                png_save_path = (
                    SAVE_ROOT / condition / distance_name / "png" / f"{out_name}_bands.png"
                )

                torch.save(band_map.cpu(), pt_save_path)
                if SAVE_PNG:
                    save_band_grid_png(
                        band_map=band_map,
                        save_path=png_save_path,
                        title=f"{condition} | {distance_name} | ID {idx}",
                    )

                summary = summarize_band_map(band_map)

                row = dict(base_row)
                row.update(summary)
                row.update({
                    "distance": distance_name,
                    "feature_file": pt_path.name,
                    "band_pt_path": str(pt_save_path),
                    "band_png_path": str(png_save_path),
                })

                all_summary_rows.append(row)

        except Exception as e:
            for distance_name in DISTANCE_NAMES:
                row = dict(base_row)
                row["distance"] = distance_name
                row["missing_feature"] = True
                row["error"] = str(e)
                all_summary_rows.append(row)


# =========================================================
# SAVE SUMMARY
# =========================================================
summary_df = pd.DataFrame(all_summary_rows)

summary_path = SAVE_ROOT / "test_conformal_band_summary_per_sample.csv"
summary_df.to_csv(summary_path, index=False)

dataset_summary = (
    summary_df[summary_df["missing_feature"] == False]
    .groupby(["condition", "distance"])
    [
        [
            "low_pct",
            "med_pct",
            "high_pct",
            "very_high_pct",
            "low_count",
            "med_count",
            "high_count",
            "very_high_count",
        ]
    ]
    .mean()
    .reset_index()
)

dataset_summary_path = SAVE_ROOT / "test_conformal_band_summary_by_condition.csv"
dataset_summary.to_csv(dataset_summary_path, index=False)

print(f"\nSaved per-sample summary : {summary_path}")
print(f"Saved dataset summary    : {dataset_summary_path}")
print(f"All outputs saved under  : {SAVE_ROOT.resolve()}")

