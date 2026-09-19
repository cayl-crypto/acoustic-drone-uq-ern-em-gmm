#!/usr/bin/env python3
# gmm_scalar_fit_1x1x1_c0p3_kl0.py
#
# Fits a 1-D GMM at each scalar position (h, w, t) independently, for the
# c0p3_kl0 (coeff=0.3, kl_weight=0.0) checkpoint's plain-only feature cache.
# Designed to run as 8 parallel SLURM jobs, each covering 16 H rows.
#
# Usage:
#   python gmm_scalar_fit_1x1x1_c0p3_kl0.py --job_id 0   # h = 0..15
#   python gmm_scalar_fit_1x1x1_c0p3_kl0.py --job_id 1   # h = 16..31
#   ...
#   python gmm_scalar_fit_1x1x1_c0p3_kl0.py --job_id 7   # h = 112..127
#
# Each job processes: 31 T x 16 H x 128 W = 63,488 positions
# Total across 8 jobs: 507,904 positions
#
# Per position:
#   - Loads SCALAR_DIR/t{t:02d}/{h:03d}_{w:03d}.pt  shape (N,)
#   - Fits GMM for k=1..MAX_K with N_SEEDS=1 random restart
#   - Logs every seed result and BIC for every k
#   - Saves result to GMM_MODEL_ROOT/positions/t{t:02d}/{h:03d}_{w:03d}.pt
#
# All local artifacts for this checkpoint live under the consolidated
# top-level folder: ./gmm_scalar_1x1x1_c0p3_kl0/
#   models/positions/...   <- this script's output
#   aggregate/...           <- gmm_scalar_aggregate_1x1x1_c0p3_kl0.py's output
#
# After all 8 jobs finish, run gmm_scalar_aggregate_1x1x1_c0p3_kl0.py

import argparse
from pathlib import Path
import math
import numpy as np
import torch
from tqdm import tqdm

# =========================================================
# ARGS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, required=True,
                    help="Job index 0-7. Determines H row range (16 rows per job).")
args = parser.parse_args()

N_JOBS    = 8
H_PER_JOB = 128 // N_JOBS    # 16 rows per job
H_START   = args.job_id * H_PER_JOB
H_END     = H_START + H_PER_JOB

assert 0 <= args.job_id < N_JOBS, f"job_id must be 0-{N_JOBS-1}, got {args.job_id}"

# =========================================================
# PATHS
# =========================================================
SCALAR_DIR = Path(
    "/path/to/DDLDataset/plain/"
    "training_scalars_1x1x1_c0p3_kl0"
)
GMM_ROOT        = Path("./gmm_scalar_1x1x1_c0p3_kl0")
GMM_MODEL_ROOT  = GMM_ROOT / "models"
POSITION_OUTDIR = GMM_MODEL_ROOT / "positions"
GMM_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIG
# =========================================================
FULL_H   = 128
FULL_W   = 128
FULL_T   = 31

MAX_K    = 10
N_SEEDS  = 1
MAX_ITER = 100
TOL      = 1e-4
EPS      = 1e-8

LOG2PI = math.log(2.0 * math.pi)

print(f"Job ID   : {args.job_id}")
print(f"H range  : {H_START} .. {H_END - 1}")
print(f"Positions: {FULL_T} x {H_PER_JOB} x {FULL_W} = {FULL_T * H_PER_JOB * FULL_W}")
print()


# =========================================================
# 1-D GMM EM  (verbose per seed)
# =========================================================
def fit_gmm_1d(x, k, n_seeds, max_iter, tol, eps, rng, pos_label):
    """
    Fit a 1-D GMM with k components to scalar array x using EM.
    Prints log-likelihood and iteration count for every seed.

    Returns:
        weights   : (k,) float64
        means     : (k,) float64
        variances : (k,) float64
        best_ll   : float
    """
    N          = len(x)
    global_var = float(x.var()) + eps

    best_ll = -np.inf
    best_w  = np.full(k, 1.0 / k)
    best_m  = np.zeros(k)
    best_v  = np.full(k, global_var)

    for s in range(n_seeds):
        m = x[rng.integers(0, N, size=k)].copy()
        v = np.full(k, global_var)
        w = np.full(k, 1.0 / k)

        prev_ll = -np.inf
        n_iters = 0

        for i in range(max_iter):
            n_iters = i + 1

            # E-step  log_p: (k, N)
            log_p = (
                np.log(np.maximum(w, eps))[:, None]
                - 0.5 * (
                    LOG2PI
                    + np.log(v)[:, None]
                    + (x - m[:, None]) ** 2 / v[:, None]
                )
            )
            log_sum = np.logaddexp.reduce(log_p, axis=0)   # (N,)
            ll      = float(log_sum.sum())

            if np.isnan(ll):
                print(f"    [{pos_label}] k={k} seed={s+1:02d}/{n_seeds}: NaN — stopping seed")
                break

            r  = np.exp(log_p - log_sum)                   # (k, N)
            Nk = r.sum(axis=1).clip(eps)                   # (k,)

            # M-step
            w = Nk / N
            m = (r * x).sum(axis=1) / Nk
            v = ((r * (x - m[:, None]) ** 2).sum(axis=1) / Nk).clip(eps)

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        print(
            f"    [{pos_label}] k={k} seed={s+1:02d}/{n_seeds}: "
            f"ll={ll:.6f}  iters={n_iters}"
            + ("  *best*" if ll > best_ll else "")
        )

        if ll > best_ll:
            best_ll = ll
            best_w  = w.copy()
            best_m  = m.copy()
            best_v  = v.copy()

    return best_w, best_m, best_v, best_ll


def bic(ll, k, N):
    return -2.0 * ll + (3 * k - 1) * math.log(N)


# =========================================================
# VERIFY INPUTS & GET N
# =========================================================
sample_file = SCALAR_DIR / "t00" / "000_000.pt"
if not sample_file.exists():
    raise FileNotFoundError(
        f"Missing: {sample_file}\nRun extract_training_scalars_1x1x1_c0p3_kl0.py first."
    )

N = torch.load(sample_file, map_location="cpu").shape[0]
print(f"Training samples (N) : {N}")

rng = np.random.default_rng(seed=42 + args.job_id)

# =========================================================
# MAIN LOOP: FIT GMM POSITION BY POSITION
# =========================================================
for t in tqdm(range(FULL_T), desc=f"[job {args.job_id}] T-slices", position=0):
    for h in tqdm(range(H_START, H_END), desc=f"t={t:02d} H rows", position=1, leave=False):
        for w in range(FULL_W):
            pos_label = f"job={args.job_id} t={t:02d} h={h:03d} w={w:03d}"

            t_out_dir = POSITION_OUTDIR / f"t{t:02d}"
            t_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = t_out_dir / f"{h:03d}_{w:03d}.pt"

            # Resume: skip if already fitted
            if out_path.exists():
                print(f"[{pos_label}] already done, skipping.")
                continue

            # Load (N,) scalar values for this position
            data = torch.load(
                SCALAR_DIR / f"t{t:02d}" / f"{h:03d}_{w:03d}.pt",
                map_location="cpu",
            ).numpy().astype(np.float64)

            print(f"\n[{pos_label}] N={N}  mean={data.mean():.6f}  std={data.std():.6f}")

            pos_result = {}
            bics = []
            lls  = []

            for k in range(1, MAX_K + 1):
                print(f"  [{pos_label}] --- k={k} ({N_SEEDS} seeds) ---")

                wts, ms, vs, ll = fit_gmm_1d(
                    data, k, N_SEEDS, MAX_ITER, TOL, EPS, rng, pos_label
                )
                b = bic(ll, k, N)

                print(f"  [{pos_label}] k={k} RESULT: best_ll={ll:.6f}  BIC={b:.6f}")

                pos_result[k] = {
                    "weights":   wts.astype(np.float32),
                    "means":     ms.astype(np.float32),
                    "variances": vs.astype(np.float32),
                    "ll":        float(ll),
                    "bic":       float(b),
                }
                bics.append(b)
                lls.append(ll)

            optimal_k = int(np.argmin(bics)) + 1
            pos_result["bics"]      = bics
            pos_result["lls"]       = lls
            pos_result["optimal_k"] = optimal_k

            print(f"  [{pos_label}] BICs     : {[f'{b:.2f}' for b in bics]}")
            print(f"  [{pos_label}] Optimal k: {optimal_k}  ->  {out_path}")

            torch.save(pos_result, out_path)

print(f"\n[job {args.job_id}] Done. h={H_START}..{H_END-1} complete.")
print("Run gmm_scalar_aggregate_1x1x1_c0p3_kl0.py after all 8 jobs finish.")
