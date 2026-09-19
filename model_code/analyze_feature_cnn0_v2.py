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


# choose one sample name
SAMPLE_NAME = "00000000.pt"
INDEX_NAME = "00000000"
SAVE_ROOT = Path(f"./feature_distribution_analysis_{INDEX_NAME}")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def compute_and_print_stats(x: torch.Tensor, tag: str):
    print(f"\n{'=' * 60}")
    print(f"Condition: {tag}")
    print(f"{'=' * 60}")

    print("\n===== BASIC INFO =====")
    print(f"Shape       : {tuple(x.shape)}")
    print(f"Dtype       : {x.dtype}")

    print("\n===== VALUE STATS =====")
    print(f"Min         : {x.min().item():.6f}")
    print(f"Max         : {x.max().item():.6f}")
    print(f"Mean        : {x.mean().item():.6f}")
    print(f"Std         : {x.std().item():.6f}")

    print("\n===== ADDITIONAL CHARACTERISTICS =====")
    numel = x.numel()
    num_zero = (x == 0).sum().item()

    print(f"Total elems : {numel}")
    print(f"Zeros       : {num_zero} ({100 * num_zero / numel:.2f}%)")
    print(f"L1 norm     : {x.abs().sum().item():.6f}")
    print(f"L2 norm     : {torch.norm(x).item():.6f}")

    print("\n===== DISTRIBUTION (QUANTILES) =====")
    quantile_points = torch.tensor(
        [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
        dtype=torch.float32,
    )
    q = torch.quantile(x.flatten(), quantile_points)

    print(f"Q0   (min)  : {q[0].item():.6f}")
    print(f"Q10         : {q[1].item():.6f}")
    print(f"Q20         : {q[2].item():.6f}")
    print(f"Q25         : {q[3].item():.6f}")
    print(f"Q30         : {q[4].item():.6f}")
    print(f"Q40         : {q[5].item():.6f}")
    print(f"Q50 (median): {q[6].item():.6f}")
    print(f"Q60         : {q[7].item():.6f}")
    print(f"Q70         : {q[8].item():.6f}")
    print(f"Q75         : {q[9].item():.6f}")
    print(f"Q80         : {q[10].item():.6f}")
    print(f"Q90         : {q[11].item():.6f}")
    print(f"Q100 (max)  : {q[12].item():.6f}")

    channel_mean = x.mean(dim=(1, 2))
    print("\n===== CHANNEL MEAN STATS =====")
    print(f"Mean of means : {channel_mean.mean().item():.6f}")
    print(f"Std of means  : {channel_mean.std().item():.6f}")

# =========================================================
def compute_and_save_stats(x: torch.Tensor, tag: str, save_dir: Path, verbose: bool = True):
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # =========================================================
    # SAVE CSV
    # =========================================================
    df = pd.DataFrame([stats]).T
    df.columns = ["value"]
    save_path = save_dir / f"{tag}_stats.csv"
    df.to_csv(save_path)

    # =========================================================
    # OPTIONAL PRINT
    # =========================================================
    if verbose:
        print(f"\nSaved stats for {tag} → {save_path}")
        print(df.T)  # nice vertical print

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
    # sample_path = feature_dir / SAMPLE_NAME
    sample_path = find_file_by_index(feature_dir, INDEX_NAME)

    if not sample_path.exists():
        print(f"[WARNING] File not found for {condition}: {sample_path}")
        continue

    print(f"Loaded file for {condition}: {sample_path}")
    x = torch.load(sample_path, map_location="cpu")

    loaded_tensors[condition] = x
    loaded_values[condition] = x.flatten().numpy()

# stop if nothing loaded
if len(loaded_tensors) == 0:
    raise ValueError("Could not load the sample from any folder.")


# =========================================================
# PRINT STATS + SAVE INDIVIDUAL HISTOGRAMS
# =========================================================
for condition, x in loaded_tensors.items():
    
    #compute_and_print_stats(x, condition)
    compute_and_save_stats(x, condition, SAVE_ROOT, verbose=False)

    values = loaded_values[condition]

    save_basic_histogram(
        values=values,
        save_path=SAVE_ROOT / f"{SAMPLE_NAME[:-3]}_{condition}_histogram_basic.png",
        title=f"Feature Distribution ({condition})",
    )

    save_clipped_histogram(
        values=values,
        save_path=SAVE_ROOT / f"{SAMPLE_NAME[:-3]}_{condition}_histogram_clipped.png",
        title=f"Feature Distribution ({condition}, 1%-99% Clipped)",
    )

    print(f"Saved histograms for {condition}")


# =========================================================
# SAVE OVERLAY COMPARISON HISTOGRAMS
# =========================================================
save_overlay_histogram(
    all_values=loaded_values,
    save_path=SAVE_ROOT / f"{SAMPLE_NAME[:-3]}_overlay_histogram_basic.png",
    title=f"Feature Distribution Comparison ({SAMPLE_NAME[:-3]})",
    clipped=False,
)

save_overlay_histogram(
    all_values=loaded_values,
    save_path=SAVE_ROOT / f"{SAMPLE_NAME[:-3]}_overlay_histogram_clipped.png",
    title=f"Feature Distribution Comparison ({SAMPLE_NAME[:-3]}, 1%-99% Clipped)",
    clipped=True,
)

print(f"\nAll outputs saved to: {SAVE_ROOT.resolve()}")