#!/usr/bin/env python3
# apply_conformal_bands_to_test_1x1x1.py

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
    "/path/to/DDLDataset/plain/distances/model/"
    "conformal_quantiles_1x1x1_tim_coverage_targets_with_min_max_distances"
)

SAVE_ROOT = Path(
    "/path/to/DDLDataset/conformal_bands/test/model"
)

DISTANCE_NAMES = ["mahalanobis"]

DEVICE = "cpu"

FULL_H = 128
FULL_W = 128
FULL_T = 31
EPS = 1e-6

SAVE_PNG = False
SAVE_EXCEEDANCE_MASKS = False

TARGET_COVERAGES = {
    "q6827": 0.6827,
    "q9545": 0.9545,
    "q9973": 0.9973,
}

# Band encoding:
# low       = 0
# med       = 1
# high      = 2
# very_high = 3


# =========================================================
# HELPERS
# =========================================================
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


def assign_bands(distance_map, q6827, q9545, q9973):
    """
    Output:
        band_map shape: [128, 128, 31]

        low       = 0  distance <= q6827
        med       = 1  q6827 < distance <= q9545
        high      = 2  q9545 < distance <= q9973
        very_high = 3  distance > q9973
    """
    band_map = torch.zeros_like(distance_map, dtype=torch.uint8)

    band_map[distance_map > q6827] = 1
    band_map[distance_map > q9545] = 2
    band_map[distance_map > q9973] = 3

    return band_map


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


def summarize_range_exceedance(distance_map, calib_min_map, calib_max_map):
    """
    Checks whether each scalar distance is outside the empirical calibration range.

    below_min:
        test distance < minimum calibration distance at that scalar.

    above_max:
        test distance > maximum calibration distance at that scalar.

    outside_range:
        below_min OR above_max.
    """
    below_min = distance_map < calib_min_map
    above_max = distance_map > calib_max_map
    outside_range = below_min | above_max

    total = distance_map.numel()

    below_min_count = int(below_min.sum().item())
    above_max_count = int(above_max.sum().item())
    outside_range_count = int(outside_range.sum().item())

    lower_excess = torch.clamp(calib_min_map - distance_map, min=0.0)
    upper_excess = torch.clamp(distance_map - calib_max_map, min=0.0)

    return {
        "below_min_count": below_min_count,
        "above_max_count": above_max_count,
        "outside_range_count": outside_range_count,

        "below_min_pct": 100.0 * below_min_count / total,
        "above_max_pct": 100.0 * above_max_count / total,
        "outside_range_pct": 100.0 * outside_range_count / total,

        "mean_lower_excess_all": float(lower_excess.mean().item()),
        "mean_upper_excess_all": float(upper_excess.mean().item()),

        "max_lower_excess": float(lower_excess.max().item()),
        "max_upper_excess": float(upper_excess.max().item()),

        "mean_lower_excess_violating_only": (
            float(lower_excess[below_min].mean().item())
            if below_min_count > 0
            else 0.0
        ),

        "mean_upper_excess_violating_only": (
            float(upper_excess[above_max].mean().item())
            if above_max_count > 0
            else 0.0
        ),

        # Simple OOD indicators
        "ood_score_above_max_pct": 100.0 * above_max_count / total,
        "ood_score_upper_excess": float(upper_excess.mean().item()),
        "ood_score_above_max_x_upper_excess": (
            100.0 * above_max_count / total
        ) * float(upper_excess.mean().item()),
    }


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
        ax.imshow(
            band_np[:, :, t],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(f"ch {t:02d}", fontsize=8)
        ax.axis("off")

    for k in range(FULL_T, len(axes)):
        axes[k].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
mu = torch.load(
    SCALAR_MODEL_ROOT / "mu.pt",
    map_location=DEVICE,
).float()

std = torch.load(
    SCALAR_MODEL_ROOT / "std.pt",
    map_location=DEVICE,
).float()

std = torch.clamp(std, min=EPS)

if tuple(mu.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected mu shape: {tuple(mu.shape)}")

if tuple(std.shape) != (FULL_H, FULL_W, FULL_T):
    raise ValueError(f"Unexpected std shape: {tuple(std.shape)}")


# =========================================================
# LOAD QUANTILES + CALIBRATION MIN/MAX MAPS
# =========================================================
quantiles = {}
calib_ranges = {}

for distance_name in DISTANCE_NAMES:
    q6827_path = QUANTILE_ROOT / f"{distance_name}_q6827.pt"
    q9545_path = QUANTILE_ROOT / f"{distance_name}_q9545.pt"
    q9973_path = QUANTILE_ROOT / f"{distance_name}_q9973.pt"

    q6827 = torch.load(q6827_path, map_location=DEVICE).float()
    q9545 = torch.load(q9545_path, map_location=DEVICE).float()
    q9973 = torch.load(q9973_path, map_location=DEVICE).float()

    for q_name, q in [
        ("q6827", q6827),
        ("q9545", q9545),
        ("q9973", q9973),
    ]:
        if tuple(q.shape) != (FULL_H, FULL_W, FULL_T):
            raise ValueError(
                f"Unexpected {q_name} shape for {distance_name}: {tuple(q.shape)}"
            )

    quantiles[distance_name] = {
        "q6827": q6827,
        "q9545": q9545,
        "q9973": q9973,
        "q6827_path": str(q6827_path),
        "q9545_path": str(q9545_path),
        "q9973_path": str(q9973_path),
    }

    calib_min_path = QUANTILE_ROOT / f"{distance_name}_calib_min_map.pt"
    calib_max_path = QUANTILE_ROOT / f"{distance_name}_calib_max_map.pt"

    calib_min_map = torch.load(calib_min_path, map_location=DEVICE).float()
    calib_max_map = torch.load(calib_max_path, map_location=DEVICE).float()

    if tuple(calib_min_map.shape) != (FULL_H, FULL_W, FULL_T):
        raise ValueError(
            f"Unexpected calibration min map shape for {distance_name}: "
            f"{tuple(calib_min_map.shape)}"
        )

    if tuple(calib_max_map.shape) != (FULL_H, FULL_W, FULL_T):
        raise ValueError(
            f"Unexpected calibration max map shape for {distance_name}: "
            f"{tuple(calib_max_map.shape)}"
        )

    calib_ranges[distance_name] = {
        "min": calib_min_map,
        "max": calib_max_map,
        "min_path": str(calib_min_path),
        "max_path": str(calib_max_path),
    }

    print(f"\nLoaded {distance_name}:")
    print(f"  q6827 path    : {q6827_path}")
    print(f"  q9545 path    : {q9545_path}")
    print(f"  q9973 path    : {q9973_path}")
    print(f"  calib min path: {calib_min_path}")
    print(f"  calib max path: {calib_max_path}")

print("\nLoaded scalar models, conformal quantiles, and calibration min/max maps.")


# =========================================================
# APPLY BANDS + RANGE EXCEEDANCE TO TEST SET
# =========================================================
all_summary_rows = []

for condition, feature_dir in FEATURE_DIRS.items():
    print(f"\nProcessing condition: {condition}")

    file_lookup = build_file_index(feature_dir)

    print("Example CSV IDs :", test_df["ID"].head().tolist())
    print("Example file IDs:", list(file_lookup.keys())[:5])
    print("Matched examples:", sum(i in file_lookup for i in test_df["ID"].head(5)))

    for distance_name in DISTANCE_NAMES:
        (SAVE_ROOT / condition / distance_name / "pt").mkdir(
            parents=True,
            exist_ok=True,
        )

        (SAVE_ROOT / condition / distance_name / "png").mkdir(
            parents=True,
            exist_ok=True,
        )

        if SAVE_EXCEEDANCE_MASKS:
            (
                SAVE_ROOT
                / condition
                / distance_name
                / "range_exceedance_masks"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    for _, meta_row in tqdm(test_df.iterrows(), total=len(test_df), desc=condition):
        idx = meta_row["ID"]
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
                    q6827=q["q6827"],
                    q9545=q["q9545"],
                    q9973=q["q9973"],
                )

                calib_min_map = calib_ranges[distance_name]["min"]
                calib_max_map = calib_ranges[distance_name]["max"]

                range_summary = summarize_range_exceedance(
                    distance_map=distance_map,
                    calib_min_map=calib_min_map,
                    calib_max_map=calib_max_map,
                )

                out_name = pt_path.stem

                pt_save_path = (
                    SAVE_ROOT
                    / condition
                    / distance_name
                    / "pt"
                    / f"{out_name}_bands.pt"
                )

                png_save_path = (
                    SAVE_ROOT
                    / condition
                    / distance_name
                    / "png"
                    / f"{out_name}_bands.png"
                )

                torch.save(band_map.cpu(), pt_save_path)

                if SAVE_PNG:
                    save_band_grid_png(
                        band_map=band_map,
                        save_path=png_save_path,
                        title=f"{condition} | {distance_name} | ID {idx}",
                    )

                if SAVE_EXCEEDANCE_MASKS:
                    below_min = distance_map < calib_min_map
                    above_max = distance_map > calib_max_map
                    outside_range = below_min | above_max

                    mask_save_path = (
                        SAVE_ROOT
                        / condition
                        / distance_name
                        / "range_exceedance_masks"
                        / f"{out_name}_range_exceedance_masks.pt"
                    )

                    torch.save(
                        {
                            "below_min": below_min.cpu(),
                            "above_max": above_max.cpu(),
                            "outside_range": outside_range.cpu(),
                        },
                        mask_save_path,
                    )
                else:
                    mask_save_path = None

                band_summary = summarize_band_map(band_map)

                row = dict(base_row)
                row.update(band_summary)
                row.update(range_summary)

                row.update({
                    "distance": distance_name,
                    "feature_file": pt_path.name,

                    "band_pt_path": str(pt_save_path),
                    "band_png_path": str(png_save_path),
                    "range_exceedance_mask_path": (
                        str(mask_save_path) if mask_save_path is not None else None
                    ),

                    "q6827_path": q["q6827_path"],
                    "q9545_path": q["q9545_path"],
                    "q9973_path": q["q9973_path"],

                    "calib_min_path": calib_ranges[distance_name]["min_path"],
                    "calib_max_path": calib_ranges[distance_name]["max_path"],
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

summary_path = SAVE_ROOT / "test_conformal_band_summary_per_sample_with_range_exceedance.csv"
summary_df.to_csv(summary_path, index=False)

dataset_summary_cols = [
    "low_pct",
    "med_pct",
    "high_pct",
    "very_high_pct",

    "low_count",
    "med_count",
    "high_count",
    "very_high_count",

    "below_min_pct",
    "above_max_pct",
    "outside_range_pct",

    "below_min_count",
    "above_max_count",
    "outside_range_count",

    "mean_lower_excess_all",
    "mean_upper_excess_all",

    "max_lower_excess",
    "max_upper_excess",

    "mean_lower_excess_violating_only",
    "mean_upper_excess_violating_only",

    "ood_score_above_max_pct",
    "ood_score_upper_excess",
    "ood_score_above_max_x_upper_excess",
]

existing_dataset_summary_cols = [
    c for c in dataset_summary_cols
    if c in summary_df.columns
]

dataset_summary = (
    summary_df[summary_df["missing_feature"] == False]
    .groupby(["condition", "distance"])[existing_dataset_summary_cols]
    .mean()
    .reset_index()
)

dataset_summary_path = (
    SAVE_ROOT / "test_conformal_band_summary_by_condition_with_range_exceedance.csv"
)
dataset_summary.to_csv(dataset_summary_path, index=False)

print(f"\nSaved per-sample summary : {summary_path}")
print(f"Saved dataset summary    : {dataset_summary_path}")
print(f"All outputs saved under  : {SAVE_ROOT.resolve()}")