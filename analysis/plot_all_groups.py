"""Generate the reviewer-facing polar plots (WITH ground truth) for representative samples.

For each of the 7 ground-truth conditions and each category (ERN_S/C, GMM_S/C, JOINT_*; S = Safe, C = Critical), picks up
to 5 samples at the low/p25/median/p75/high percentiles of predicted uncertainty
(ranked by bearing_std_deg, then sigma_range) and draws prediction + 1/2/3-sigma ellipses + GT.

    python analysis/plot_all_groups.py [--out-dir figures/test_set]

Empty (condition, category) slots are skipped; 140 plots are produced.
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_io import (GT_CONDITIONS, REPO, RANGE_MAX_M, ern_label, gmm_label,
                     load_ern, load_gmm, load_pred)

N_MC = 500
CATEGORIES = ["ERN_S", "ERN_C", "GMM_S", "GMM_C",
              "JOINT_S/S", "JOINT_S/C", "JOINT_C/S", "JOINT_C/C"]
PICK_LABELS_5 = ["low", "p25", "median", "p75", "high"]

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=os.path.join(REPO, "figures", "test_set"))
OUT_DIR = ap.parse_args().out_dir
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(0)


def pick_indices(n):
    if n >= 5:
        return list(zip(PICK_LABELS_5, [0, round(0.25 * (n - 1)), round(0.5 * (n - 1)),
                                         round(0.75 * (n - 1)), n - 1]))
    labels_by_n = {1: ["only"], 2: ["low", "high"], 3: ["low", "median", "high"],
                   4: ["low", "p33", "p67", "high"]}
    return list(zip(labels_by_n[n], range(n)))


def monte_carlo_xy(sin_mu, sin_std, cos_mu, cos_std, range_mu_m, range_std_m):
    cos_samples = rng.normal(cos_mu, cos_std, size=N_MC)
    sin_samples = rng.normal(sin_mu, sin_std, size=N_MC)
    norms = np.sqrt(cos_samples**2 + sin_samples**2) + 1e-8
    range_samples = rng.normal(range_mu_m, range_std_m, size=N_MC)
    return range_samples * cos_samples / norms, range_samples * sin_samples / norms


def ellipse_contour(mean, eigvals, eigvecs, k, n_pts=150):
    t = np.linspace(0, 2 * np.pi, n_pts)
    coords = np.array([np.sqrt(max(eigvals[0], 0)) * np.cos(t),
                        np.sqrt(max(eigvals[1], 0)) * np.sin(t)]) * k
    rot = eigvecs @ coords
    rot[0, :] += mean[0]
    rot[1, :] += mean[1]
    return np.arctan2(rot[1, :], rot[0, :]), np.hypot(rot[0, :], rot[1, :])


fig = plt.figure(figsize=(6.2, 5.4))
ax = fig.add_subplot(111, projection="polar")
plotted = 0

for condition in GT_CONDITIONS:
    ern, gmm, pred = load_ern(condition), load_gmm(condition), load_pred(condition)
    records = []
    for sid in sorted(set(ern) & set(gmm) & set(pred)):
        erow, prow = ern[sid], pred[sid]
        records.append({
            "id": sid, "ern": ern_label(erow), "gmm": gmm_label(gmm[sid]),
            "bearing_gt": float(erow["bearing_deg_gt"]), "range_gt": float(erow["range_m_gt"]),
            "bearing_pred": float(prow["bearing_deg_pred"]), "range_pred": float(prow["range_m_pred"]),
            "sin_mu": float(prow["sin_mu"]), "sigma_sin": float(prow["sigma_sin"]),
            "cos_mu": float(prow["cos_mu"]), "sigma_cos": float(prow["sigma_cos"]),
            "range_mu": float(prow["range_mu"]),
            "bearing_err": float(erow["bearing_error_deg"]), "range_err": float(erow["range_error_m"]),
            "bearing_std": float(erow["bearing_std_deg"]),
            "range_std_m": float(erow["sigma_range"]) * RANGE_MAX_M,
            "euclid_err": float(erow["euclidean_error_m"]),
        })

    buckets = {c: [] for c in CATEGORIES}
    for rec in records:
        buckets[f"ERN_{rec['ern']}"].append(rec)
        buckets[f"GMM_{rec['gmm']}"].append(rec)
        buckets[f"JOINT_{rec['ern']}/{rec['gmm']}"].append(rec)

    for category in CATEGORIES:
        group = buckets[category]
        n = len(group)
        if n == 0:
            continue
        group.sort(key=lambda r: (r["bearing_std"], r["range_std_m"]))
        cat_dir = os.path.join(OUT_DIR, condition.replace("-", "m"), category.replace("/", "-"))
        os.makedirs(cat_dir, exist_ok=True)

        for label, idx in pick_indices(n):
            rec = group[idx]
            x, y = monte_carlo_xy(rec["sin_mu"], rec["sigma_sin"], rec["cos_mu"], rec["sigma_cos"],
                                  rec["range_mu"] * RANGE_MAX_M, rec["range_std_m"])
            # Anchor the ellipse at the real prediction, not the MC sample mean: with large
            # sigma_sin/cos the normalized samples scatter around the circle and their mean
            # collapses toward the origin instead of the predicted direction.
            mean_xy = np.array([rec["range_pred"] * math.cos(math.radians(rec["bearing_pred"])),
                                rec["range_pred"] * math.sin(math.radians(rec["bearing_pred"]))])
            dx, dy = x - mean_xy[0], y - mean_xy[1]
            cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            gt_rad = math.radians(rec["bearing_gt"])
            diff = np.array([rec["range_gt"] * math.cos(gt_rad), rec["range_gt"] * math.sin(gt_rad)]) - mean_xy
            try:
                mahal = math.sqrt(diff @ np.linalg.inv(cov) @ diff)
            except np.linalg.LinAlgError:
                mahal = float("inf")
            tier = 1 if mahal <= 1 else 2 if mahal <= 2 else 3 if mahal <= 3 else 4
            tier_txt = f"{tier}$\\sigma$" if tier <= 3 else "outside 3$\\sigma$"

            ax.clear()
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            colors, styles = {1: "#2E7D32", 2: "#F9A825", 3: "#C62828"}, {1: "-", 2: "--", 3: ":"}
            for k in (1, 2, 3):
                theta_e, r_e = ellipse_contour(mean_xy, eigvals, eigvecs, k)
                ax.plot(theta_e, r_e, color=colors[k], linestyle=styles[k], linewidth=2,
                        label=f"{k}$\\sigma$ contour")
            ax.plot(math.radians(rec["bearing_pred"]), rec["range_pred"], marker="*", color="black",
                    markersize=8, linestyle="none", zorder=5, label="Prediction")
            ax.plot(gt_rad, rec["range_gt"], marker="X", color="blue",
                    markersize=7, linestyle="none", zorder=5, label="Ground truth")
            ax.scatter(0, 0, marker="s", s=20, color="black", zorder=5, label="Microphone array")
            ax.set_rmax(RANGE_MAX_M + 20)
            ax.set_rticks([50, 100, 150, 200, 250])
            ax.grid(True, linestyle="--", alpha=0.5)
            std_note = " (capped)" if rec["bearing_std"] > 180.0 else ""
            ax.set_title(
                f"{condition} | {category} | {label} (n={n}, rank {idx+1}/{n})\n"
                f"ID {rec['id']}  bearing_std={rec['bearing_std']:.1f}°{std_note} "
                f"range_std={rec['range_std_m']:.1f} m\n"
                f"bearing_err={rec['bearing_err']:.1f}° range_err={rec['range_err']:.1f} m "
                f"eucl_err={rec['euclid_err']:.1f} m | covered (ellipse): {tier_txt}",
                fontsize=8.5)
            ax.legend(loc="lower right", bbox_to_anchor=(1.32, -0.05), fontsize=7)
            fig.subplots_adjust(right=0.72, top=0.82)
            fig.savefig(os.path.join(cat_dir, f"{label}_{rec['id']}.png"), dpi=130, bbox_inches="tight")
            plotted += 1

plt.close(fig)
print("total plots written:", plotted, "->", OUT_DIR)
