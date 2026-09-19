from pathlib import Path
import torch

# =========================================================
# PATH
# =========================================================
MODEL_FEATURE_DIR = Path("/path/to/DDLDataset/plain/features/model")

# =========================================================
# LOAD ONE SAMPLE
# =========================================================
pt_files = sorted(MODEL_FEATURE_DIR.glob("*.pt"))

if len(pt_files) == 0:
    raise ValueError("No .pt files found!")

sample_path = pt_files[0]
print(f"Loaded file: {sample_path.name}")

x = torch.load(sample_path, map_location="cpu")

# =========================================================
# STATS
# =========================================================
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
print(f"Zeros       : {num_zero} ({100*num_zero/numel:.2f}%)")

print(f"L1 norm     : {x.abs().sum().item():.6f}")
print(f"L2 norm     : {torch.norm(x).item():.6f}")

print("\n===== DISTRIBUTION (QUANTILES) =====")
#q = torch.quantile(x.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
q = torch.quantile(x.flatten(), torch.tensor([0.0, 0.1, 0.2, 0.25, 0.3, 0.4,0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]))
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


# channel-wise mean (important for drift analysis)
channel_mean = x.mean(dim=(1,2))
print("\nChannel mean stats:")
print(f"Mean of means : {channel_mean.mean().item():.6f}")
print(f"Std of means  : {channel_mean.std().item():.6f}")


from pathlib import Path
import torch
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================
MODEL_FEATURE_DIR = Path("/path/to/DDLDataset/plain/features/model")
SAVE_PATH = Path("./histogram_basic.png")

# =========================================================
# LOAD SAMPLE
# =========================================================
sample_path = sorted(MODEL_FEATURE_DIR.glob("*.pt"))[0]
x = torch.load(sample_path, map_location="cpu")

values = x.flatten().numpy()

# =========================================================
# PLOT & SAVE
# =========================================================
plt.figure()
plt.hist(values, bins=100)
plt.title("Feature Distribution (Conv1 Output)")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to: {SAVE_PATH}")


import numpy as np

SAVE_PATH = Path("./histogram_clipped.png")

lower = np.percentile(values, 1)
upper = np.percentile(values, 99)

clipped = values[(values >= lower) & (values <= upper)]

plt.figure()
plt.hist(clipped, bins=100)
plt.title("Feature Distribution (1%-99% Clipped)")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to: {SAVE_PATH}")