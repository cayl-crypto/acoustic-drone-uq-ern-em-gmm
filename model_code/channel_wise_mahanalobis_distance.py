from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
CSV_PATH = Path("/path/to/DDLDataset/annotations/train_dataset_with_nondrone_with_9600_flag.csv")
FEATURE_DIR = Path("/path/to/DDLDataset/plain/features/model")

DEVICE = "cpu"

# =========================================================
# LOAD IDS
# =========================================================
df = pd.read_csv(CSV_PATH)
ids = df["ID"].astype(str).tolist()

# =========================================================
# FIRST PASS: GLOBAL MEAN
# =========================================================
sum_feat = torch.zeros(128).to(DEVICE)
count = 0

def find_file_by_index(folder, index):
    matches = list(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None

print("Computing global mean...")

for idx in tqdm(ids):
    pt_path = find_file_by_index(FEATURE_DIR, idx)
    if pt_path is None:
        continue

    x = torch.load(pt_path, map_location=DEVICE)  # [128,128,31]
    x = x.reshape(128, -1)  # [128, 3968]

    sum_feat += x.sum(dim=1)
    count += x.shape[1]

mu = sum_feat / count  # [128]

# =========================================================
# SECOND PASS: COVARIANCE
# =========================================================
print("Computing covariance...")

cov = torch.zeros(128, 128).to(DEVICE)

for idx in tqdm(ids):
    pt_path = find_file_by_index(FEATURE_DIR, idx)
    if pt_path is None:
        continue

    x = torch.load(pt_path, map_location=DEVICE)
    x = x.reshape(128, -1)

    x_centered = x - mu.unsqueeze(1)
    cov += x_centered @ x_centered.T

cov = cov / (count - 1)

# =========================================================
# REGULARIZATION (IMPORTANT)
# =========================================================
eps = 1e-4
cov += eps * torch.eye(128)

cov_inv = torch.inverse(cov)

# =========================================================
# SAVE
# =========================================================
torch.save(mu, "mu.pt")
torch.save(cov, "cov.pt")
torch.save(cov_inv, "cov_inv.pt")

print("Saved mean and covariance.")