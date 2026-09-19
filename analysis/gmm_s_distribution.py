"""GMM plain-condition distribution of the Safe (S) group (panels A-D). Writes figures/gmm_plain_S_distribution.png."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_io import GMM_S_THRESHOLD, REPO, load_gmm

OUT_DIR = os.path.join(REPO, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

rows = list(load_gmm("plain").values())

BINS = ["below_min_pct", "low_pct", "med_pct", "high_pct", "very_high_pct", "above_max_pct"]
data = {b: np.array([float(r[b]) for r in rows]) for b in BINS}
safe_share = data["below_min_pct"] + data["low_pct"] + data["med_pct"]
is_S = safe_share >= GMM_S_THRESHOLD

# Candidate sub-bands within S, chosen from the S group's own quartiles (rounded to readable cuts)
band_edges = [95.45, 97.0, 98.5, 99.5, 100.0001]
band_names = ["S-borderline\n[95.45, 97)", "S-moderate\n[97, 98.5)", "S-tight\n[98.5, 99.5)", "S-very tight\n[99.5, 100]"]
band_colors = ["#EF6C00", "#FDD835", "#7CB342", "#2E7D32"]

safe_share_S = safe_share[is_S]
band_of_S = np.digitize(safe_share_S, band_edges[1:-1])  # 0..3

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Panel A: full safe_share histogram, S vs C, with threshold line
ax = axes[0, 0]
ax.hist(safe_share[~is_S], bins=40, color="#C62828", alpha=0.75, label=f"C (n={(~is_S).sum()})")
ax.hist(safe_share[is_S], bins=40, color="#1565C0", alpha=0.75, label=f"S (n={is_S.sum()})")
ax.axvline(GMM_S_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"S/C threshold = {GMM_S_THRESHOLD}%")
ax.set_xlabel("safe_share = below_min_pct + low_pct + med_pct  (%)")
ax.set_ylabel("number of samples")
ax.set_title("A. Full distribution of safe_share, plain condition (GMM)")
ax.legend(fontsize=8)

# --- Panel B: safe_share within S only, with candidate sub-bands shaded
ax = axes[0, 1]
ax.hist(safe_share_S, bins=40, color="#1565C0", alpha=0.85)
for i in range(len(band_edges) - 1):
    ax.axvspan(band_edges[i], band_edges[i + 1], color=band_colors[i], alpha=0.15)
    ax.axvline(band_edges[i], color=band_colors[i], linestyle=":", linewidth=1)
for i, name in enumerate(band_names):
    mid = (max(band_edges[i], safe_share_S.min()) + min(band_edges[i + 1], safe_share_S.max())) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.92, name, ha="center", va="top", fontsize=7.5, color=band_colors[i])
ax.set_xlabel("safe_share, restricted to S-labeled samples (%)")
ax.set_ylabel("number of samples")
ax.set_title("B. safe_share within S only, with candidate sub-bands")

# --- Panel C: the 3 bins that actually vary within S (low/med/high), overlaid
ax = axes[1, 0]
for b, c in [("low_pct", "#1565C0"), ("med_pct", "#F9A825"), ("high_pct", "#C62828")]:
    ax.hist(data[b][is_S], bins=40, alpha=0.55, label=b, color=c)
ax.set_xlabel("bin percentage, within S-labeled samples (%)")
ax.set_ylabel("number of samples")
ax.set_title("C. Which bins drive the deviation within S\n(below_min/very_high/above_max are ~0 and omitted)")
ax.legend(fontsize=8)

# --- Panel D: composition of S samples, sorted by safe_share ascending, stacked area
ax = axes[1, 1]
order = np.argsort(safe_share_S)
x = np.arange(len(order))
bottom = np.zeros(len(order))
stack_bins = ["low_pct", "med_pct", "high_pct", "very_high_pct", "above_max_pct", "below_min_pct"]
stack_colors = ["#1565C0", "#F9A825", "#C62828", "#8E24AA", "#546E7A", "#90A4AE"]
for b, c in zip(stack_bins, stack_colors):
    vals = data[b][is_S][order]
    ax.fill_between(x, bottom, bottom + vals, color=c, label=b, linewidth=0)
    bottom += vals
for edge in band_edges[1:-1]:
    pos = np.searchsorted(safe_share_S[order], edge)
    ax.axvline(pos, color="black", linestyle=":", linewidth=1)
ax.set_xlim(0, len(order))
ax.set_ylim(0, 100)
ax.set_xlabel("S-labeled samples, sorted by safe_share ascending  →")
ax.set_ylabel("stacked bin composition (%)")
ax.set_title("D. Composition shift across S samples\n(dotted lines = candidate sub-band cuts from panel B)")
ax.legend(fontsize=7, loc="lower right", ncol=2)

fig.suptitle("GMM 'plain' condition: where S-labeled samples deviate from each other", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(OUT_DIR, "gmm_plain_S_distribution.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("wrote", out_path)

# print the band counts for reference
print("\ncandidate sub-band counts within S (n=%d total S):" % is_S.sum())
for i, name in enumerate(band_names):
    count = (band_of_S == i).sum()
    print(f"  {name.splitlines()[0]:22s} [{band_edges[i]:.2f}, {band_edges[i+1]:.2f}): n={count} ({100*count/is_S.sum():.1f}%)")
