from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
FEATURE_DIRS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB": Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB": Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB": Path("/path/to/DDLDataset/-5dB/features/model"),
}

#INDEX_NAME = "00000002"
INDEX_NAME = "00014063"
SAMPLE_NAME = f"{INDEX_NAME}.pt"
SAVE_ROOT = Path(f"./feature_distribution_analysis_{INDEX_NAME}")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def compute_stats_dict(x: torch.Tensor, tag: str) -> dict:
    stats = {}

    # ================= BASIC =================
    stats["condition"] = tag
    stats["shape"] = str(tuple(x.shape))
    stats["dtype"] = str(x.dtype)

    # ================= VALUE STATS =================
    stats["min"] = x.min().item()
    stats["max"] = x.max().item()
    stats["mean"] = x.mean().item()
    stats["std"] = x.std().item()

    # ================= ADDITIONAL =================
    numel = x.numel()
    num_zero = (x == 0).sum().item()

    stats["num_elements"] = numel
    stats["num_zeros"] = num_zero
    stats["zero_percent"] = 100 * num_zero / numel
    stats["l1_norm"] = x.abs().sum().item()
    stats["l2_norm"] = torch.norm(x).item()

    # ================= QUANTILES =================
    quantile_points = torch.tensor(
        [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
        dtype=torch.float32,
    )
    q = torch.quantile(x.flatten(), quantile_points)

    quantile_names = [
        "q0", "q10", "q20", "q25", "q30", "q40",
        "q50", "q60", "q70", "q75", "q80", "q90", "q100"
    ]

    for name, val in zip(quantile_names, q):
        stats[name] = val.item()

    # ================= CHANNEL STATS =================
    channel_mean = x.mean(dim=(1, 2))
    stats["channel_mean_mean"] = channel_mean.mean().item()
    stats["channel_mean_std"] = channel_mean.std().item()

    return stats


def compute_and_save_stats(x: torch.Tensor, tag: str, save_dir: Path, verbose: bool = True) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_stats_dict(x, tag)

    df = pd.DataFrame([stats]).T
    df.columns = [tag]

    save_path = save_dir / f"{tag}_stats.csv"
    df.to_csv(save_path)

    if verbose:
        print(f"\nSaved stats for {tag} → {save_path}")
        print(df)

    return stats


def save_basic_histogram(values: np.ndarray, save_path: Path, title: str):
    plt.figure()
    plt.hist(values, bins=100)
    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_clipped_histogram(values: np.ndarray, save_path: Path, title: str):
    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)
    clipped = values[(values >= lower) & (values <= upper)]

    plt.figure()
    plt.hist(clipped, bins=100)
    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_overlay_histogram(all_values: dict, save_path: Path, title: str, clipped: bool = False):
    plt.figure()

    for label, values in all_values.items():
        plot_vals = values
        if clipped:
            lower = np.percentile(values, 1)
            upper = np.percentile(values, 99)
            plot_vals = values[(values >= lower) & (values <= upper)]

        plt.hist(plot_vals, bins=100, alpha=0.45, label=label)

    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_file_by_index(folder: Path, index_str: str):
    matches = list(folder.glob(f"*{index_str}*.pt"))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"[WARNING] Multiple matches in {folder} for {index_str}:")
        for m in matches:
            print(f"  - {m.name}")

    return matches[0]


# =========================================================
# LOAD SAME SAMPLE FROM ALL CONDITIONS
# =========================================================
loaded_tensors = {}
loaded_values = {}

for condition, feature_dir in FEATURE_DIRS.items():
    sample_path = find_file_by_index(feature_dir, INDEX_NAME)

    if sample_path is None:
        print(f"[WARNING] No file found for {condition} containing index {INDEX_NAME}")
        continue

    print(f"Loaded file for {condition}: {sample_path}")
    x = torch.load(sample_path, map_location="cpu")

    loaded_tensors[condition] = x
    loaded_values[condition] = x.flatten().numpy()

if len(loaded_tensors) == 0:
    raise ValueError("Could not load the sample from any folder.")


# =========================================================
# SAVE INDIVIDUAL STATS + HISTOGRAMS
# =========================================================
all_dfs = []

for condition, x in loaded_tensors.items():
    stats = compute_and_save_stats(x, condition, SAVE_ROOT, verbose=False)

    # add to merged transposed table
    df = pd.DataFrame([stats]).T
    df.columns = [condition]
    all_dfs.append(df)

    values = loaded_values[condition]

    save_basic_histogram(
        values=values,
        save_path=SAVE_ROOT / f"{INDEX_NAME}_{condition}_histogram_basic.png",
        title=f"Feature Distribution ({condition})",
    )

    save_clipped_histogram(
        values=values,
        save_path=SAVE_ROOT / f"{INDEX_NAME}_{condition}_histogram_clipped.png",
        title=f"Feature Distribution ({condition}, 1%-99% Clipped)",
    )

    print(f"Saved stats and histograms for {condition}")


# =========================================================
# SAVE MERGED TRANSPOSED CSV
# =========================================================
if all_dfs:
    merged = pd.concat(all_dfs, axis=1)
    merged.to_csv(SAVE_ROOT / "all_conditions_transposed.csv")
    print(f"Saved merged stats: {SAVE_ROOT / 'all_conditions_transposed.csv'}")


# =========================================================
# SAVE OVERLAY COMPARISON HISTOGRAMS
# =========================================================
save_overlay_histogram(
    all_values=loaded_values,
    save_path=SAVE_ROOT / f"{INDEX_NAME}_overlay_histogram_basic.png",
    title=f"Feature Distribution Comparison ({INDEX_NAME})",
    clipped=False,
)

save_overlay_histogram(
    all_values=loaded_values,
    save_path=SAVE_ROOT / f"{INDEX_NAME}_overlay_histogram_clipped.png",
    title=f"Feature Distribution Comparison ({INDEX_NAME}, 1%-99% Clipped)",
    clipped=True,
)

print(f"\nAll outputs saved to: {SAVE_ROOT.resolve()}")