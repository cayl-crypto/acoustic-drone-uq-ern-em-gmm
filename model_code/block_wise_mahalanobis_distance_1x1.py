from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
CSV_PATH = Path(
    "/path/to/DDLDataset/annotations/train_dataset_with_nondrone_with_9600_flag.csv"
)
FEATURE_DIR = Path(
    "/path/to/DDLDataset/plain/features/model"
)

SAVE_ROOT = Path("./mahalanobis_blocks_1x1x31")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
DTYPE = torch.float64  # more stable for covariance accumulation

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

BLOCK_DIM = BLOCK_H * BLOCK_W * FULL_T  # 8*8*31 = 1984

# =========================================================
# HELPERS
# =========================================================
def find_file_by_index(folder: Path, index: str):
    matches = sorted(folder.glob(f"*{index}*.pt"))
    return matches[0] if matches else None


def extract_block(x: torch.Tensor, bi: int, bj: int) -> torch.Tensor:
    """
    x: [128, 128, 31]
    returns block: [8, 8, 31]
    """
    r0 = bi * BLOCK_H
    r1 = r0 + BLOCK_H
    c0 = bj * BLOCK_W
    c1 = c0 + BLOCK_W
    return x[r0:r1, c0:c1, :]


def flatten_block(block: torch.Tensor) -> torch.Tensor:
    """
    block: [8, 8, 31]
    returns: [1984]
    """
    return block.reshape(-1)


def block_name(bi: int, bj: int) -> str:
    return f"block_r{bi:02d}_c{bj:02d}"


# =========================================================
# LOAD IDS
# =========================================================
df = pd.read_csv(CSV_PATH)
ids = df["ID"].astype(str).tolist()

print(f"Total IDs in training CSV: {len(ids)}")
print(f"Total blocks per sample   : {N_BLOCKS}")
print(f"Block vector dimension    : {BLOCK_DIM}")

# =========================================================
# FIRST PASS: GLOBAL MEAN PER BLOCK
# =========================================================
print("\n[1/3] Computing block-wise global means...")

sum_feats = {
    (bi, bj): torch.zeros(BLOCK_DIM, dtype=DTYPE, device=DEVICE)
    for bi in range(N_BLOCK_H)
    for bj in range(N_BLOCK_W)
}

valid_sample_count = 0

for idx in tqdm(ids):
    pt_path = find_file_by_index(FEATURE_DIR, idx)
    if pt_path is None:
        continue

    x = torch.load(pt_path, map_location=DEVICE).to(DTYPE)

    if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
        print(f"[WARNING] Unexpected shape for {pt_path.name}: {tuple(x.shape)}")
        continue

    for bi in range(N_BLOCK_H):
        for bj in range(N_BLOCK_W):
            block = extract_block(x, bi, bj)
            vec = flatten_block(block)
            sum_feats[(bi, bj)] += vec

    valid_sample_count += 1

if valid_sample_count == 0:
    raise ValueError("No valid training feature files found.")

mus = {
    key: sum_feats[key] / valid_sample_count
    for key in sum_feats
}

print(f"Valid training samples used: {valid_sample_count}")

# =========================================================
# SECOND PASS: COVARIANCE PER BLOCK
# =========================================================
print("\n[2/3] Computing block-wise covariance matrices...")

covs = {
    (bi, bj): torch.zeros((BLOCK_DIM, BLOCK_DIM), dtype=DTYPE, device=DEVICE)
    for bi in range(N_BLOCK_H)
    for bj in range(N_BLOCK_W)
}

for idx in tqdm(ids):
    pt_path = find_file_by_index(FEATURE_DIR, idx)
    if pt_path is None:
        continue

    x = torch.load(pt_path, map_location=DEVICE).to(DTYPE)

    if tuple(x.shape) != (FULL_H, FULL_W, FULL_T):
        continue

    for bi in range(N_BLOCK_H):
        for bj in range(N_BLOCK_W):
            block = extract_block(x, bi, bj)
            vec = flatten_block(block)
            diff = vec - mus[(bi, bj)]
            covs[(bi, bj)] += torch.outer(diff, diff)

for key in covs:
    covs[key] /= max(valid_sample_count - 1, 1)

# =========================================================
# THIRD PASS: REGULARIZE + INVERT + SAVE
# =========================================================
print("\n[3/3] Regularizing, inverting, and saving block statistics...")

eps = 1e-3  # slightly stronger regularization for stability
eye = torch.eye(BLOCK_DIM, dtype=DTYPE, device=DEVICE)

summary_rows = []

for bi in tqdm(range(N_BLOCK_H), desc="Saving blocks"):
    for bj in range(N_BLOCK_W):
        name = block_name(bi, bj)
        save_dir = SAVE_ROOT / name
        save_dir.mkdir(parents=True, exist_ok=True)

        mu = mus[(bi, bj)]
        cov = covs[(bi, bj)] + eps * eye

        # pinv is more stable than inverse for high-dimensional covariance
        cov_inv = torch.linalg.pinv(cov)

        # save as float32 to reduce disk usage
        torch.save(mu.float().cpu(), save_dir / "mu.pt")
        torch.save(cov.float().cpu(), save_dir / "cov.pt")
        torch.save(cov_inv.float().cpu(), save_dir / "cov_inv.pt")

        summary_rows.append({
            "block_name": name,
            "block_row": bi,
            "block_col": bj,
            "block_shape": f"({BLOCK_H}, {BLOCK_W}, {FULL_T})",
            "block_dim": BLOCK_DIM,
            "num_samples": valid_sample_count,
            "regularization_eps": eps,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SAVE_ROOT / "block_summary.csv", index=False)

print(f"\nSaved all block Mahalanobis statistics to: {SAVE_ROOT.resolve()}")