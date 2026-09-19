"""Plot a model prediction, its Monte-Carlo sigma-ellipses, and the ground truth on one polar axis.

Reads a predictions JSON (data/predictions/<condition>.json) and a ground-truth JSON
(ground_truth/<condition>.json), joins them on ID, and draws for each requested ID:
  star = prediction, blue X = ground truth, 1/2/3-sigma ellipses anchored on the prediction,
  square at the origin = microphone array. The title reports the errors and the smallest sigma
  ellipse that contains the ground truth (Mahalanobis distance to the drawn covariance).

    python plot_prediction_vs_gt.py \\
        --predictions ../data/predictions/plain.json \\
        --ground-truth ../ground_truth/plain.json \\
        --id 12015 --id 44189 --out-dir ./out

IDs may be given with or without zero padding (12015 == 00012015).
For the ground-truth-free version used for LLM interpretation, see plot_blind.py.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_blind import RANGE_MAX_M, ellipse_contour, monte_carlo_xy


def norm_id(raw_id):
    raw_id = str(raw_id)
    return raw_id if raw_id.startswith("rand_") else str(int(raw_id))


def load_by_id(path):
    with open(path) as f:
        return {norm_id(r["ID"]): r for r in json.load(f)}


def circular_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def plot_one(sid, pred, gt, condition, out_path, seed=0):
    rng = np.random.default_rng(seed)
    bearing_pred, range_pred = float(pred["bearing_deg_pred"]), float(pred["range_m_pred"])
    bearing_gt, range_gt = float(gt["bearing_deg_gt"]), float(gt["range_m_gt"])
    range_std_m = float(pred["sigma_range"]) * RANGE_MAX_M

    x, y = monte_carlo_xy(rng, float(pred["sin_mu"]), float(pred["sigma_sin"]),
                          float(pred["cos_mu"]), float(pred["sigma_cos"]),
                          float(pred["range_mu"]) * RANGE_MAX_M, range_std_m)
    # Anchored at the analytic prediction (see README: the raw MC mean collapses toward the
    # origin when the bearing uncertainty is very large).
    mean_xy = np.array([range_pred * math.cos(math.radians(bearing_pred)),
                        range_pred * math.sin(math.radians(bearing_pred))])
    dx, dy = x - mean_xy[0], y - mean_xy[1]
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    gt_rad = math.radians(bearing_gt)
    gt_xy = np.array([range_gt * math.cos(gt_rad), range_gt * math.sin(gt_rad)])
    diff = gt_xy - mean_xy
    try:
        mahal = math.sqrt(diff @ np.linalg.inv(cov) @ diff)
    except np.linalg.LinAlgError:
        mahal = float("inf")
    tier = 1 if mahal <= 1 else 2 if mahal <= 2 else 3 if mahal <= 3 else 4
    tier_txt = f"{tier}$\\sigma$" if tier <= 3 else "outside 3$\\sigma$"

    bearing_err = circular_diff_deg(bearing_pred, bearing_gt)
    range_err = abs(range_pred - range_gt)
    eucl = gt.get("euclidean_error_m")
    eucl_txt = f" eucl_err={float(eucl):.1f} m" if eucl is not None else ""

    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    colors, styles = {1: "#2E7D32", 2: "#F9A825", 3: "#C62828"}, {1: "-", 2: "--", 3: ":"}
    for k in (1, 2, 3):
        theta_e, r_e = ellipse_contour(mean_xy, eigvals, eigvecs, k)
        ax.plot(theta_e, r_e, color=colors[k], linestyle=styles[k], linewidth=2, label=f"{k}$\\sigma$ contour")
    ax.plot(math.radians(bearing_pred), range_pred, marker="*", color="black", markersize=8,
            linestyle="none", zorder=5, label="Prediction")
    ax.plot(gt_rad, range_gt, marker="X", color="blue", markersize=7,
            linestyle="none", zorder=5, label="Ground truth")
    ax.scatter(0, 0, marker="s", s=20, color="black", zorder=5, label="Microphone array")
    ax.set_rmax(RANGE_MAX_M + 20)
    ax.set_rticks([50, 100, 150, 200, 250])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f"{condition} | ID {sid}\n"
        f"bearing_std={float(pred['bearing_std_deg']):.1f}° range_std={range_std_m:.1f} m\n"
        f"bearing_err={bearing_err:.1f}° range_err={range_err:.1f} m{eucl_txt} | covered: {tier_txt}",
        fontsize=8.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.32, -0.05), fontsize=7)
    fig.subplots_adjust(right=0.72, top=0.82)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="data/predictions/<condition>.json")
    ap.add_argument("--ground-truth", required=True, help="ground_truth/<condition>.json")
    ap.add_argument("--id", action="append", required=True, help="sample ID (repeat for several)")
    ap.add_argument("--out-dir", default=".", help="output directory (default: current)")
    args = ap.parse_args()

    preds, gts = load_by_id(args.predictions), load_by_id(args.ground_truth)
    condition = os.path.splitext(os.path.basename(args.predictions))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    for raw in args.id:
        sid = norm_id(raw)
        if sid not in preds or sid not in gts:
            print(f"skip {raw}: not found in {'predictions' if sid not in preds else 'ground truth'}")
            continue
        out_path = os.path.join(args.out_dir, f"{condition}_{sid}.png")
        plot_one(sid, preds[sid], gts[sid], condition, out_path)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
