"""Plot a prediction + Monte-Carlo sigma-ellipse from one llm_interpretation sample JSON.

No ground truth is read or plotted -- by design, matching the blind-interpretation framing
in llm_interpretation/prompt/LLM_prompt_guide.txt. If you want a ground-truth-annotated
plot for reviewer/verification purposes instead, see plot_prediction_vs_gt.py in this
same folder.

Usage:
    python plot_blind.py --json path/to/sample.json [--out out.png]
    python plot_blind.py --json-dir path/to/folder --out-dir path/to/output
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RANGE_MAX_M = 250.0  # range_mu is normalized by this; range_m_pred = range_mu * RANGE_MAX_M
N_MC = 500


def monte_carlo_xy(rng, sin_mu, sin_std, cos_mu, cos_std, range_mu_m, range_std_m):
    cos_samples = rng.normal(cos_mu, cos_std, size=N_MC)
    sin_samples = rng.normal(sin_mu, sin_std, size=N_MC)
    norms = np.sqrt(cos_samples**2 + sin_samples**2) + 1e-8
    cos_unit, sin_unit = cos_samples / norms, sin_samples / norms
    range_samples = rng.normal(range_mu_m, range_std_m, size=N_MC)
    x = range_samples * cos_unit
    y = range_samples * sin_unit
    return x, y


def ellipse_contour(mean, eigvals, eigvecs, k, n_pts=150):
    t = np.linspace(0, 2 * np.pi, n_pts)
    coords = np.array([np.sqrt(max(eigvals[0], 0)) * np.cos(t),
                        np.sqrt(max(eigvals[1], 0)) * np.sin(t)]) * k
    coords_rot = eigvecs @ coords
    coords_rot[0, :] += mean[0]
    coords_rot[1, :] += mean[1]
    r = np.hypot(coords_rot[0, :], coords_rot[1, :])
    theta = np.arctan2(coords_rot[1, :], coords_rot[0, :])
    return theta, r


def plot_one(data, out_path, seed=0):
    rng = np.random.default_rng(seed)
    pred = data["ern_prediction"]
    unc = data["ern_uncertainty"]

    bearing_pred = float(pred["bearing_deg_pred"])
    range_pred = float(pred["range_m_pred"])
    sin_mu, cos_mu = float(pred["sin_mu"]), float(pred["cos_mu"])
    sigma_sin, sigma_cos = float(unc["sigma_sin"]), float(unc["sigma_cos"])
    range_std_m = float(unc["sigma_range"]) * RANGE_MAX_M
    range_mu_m = float(pred["range_mu"]) * RANGE_MAX_M

    x, y = monte_carlo_xy(rng, sin_mu, sigma_sin, cos_mu, sigma_cos, range_mu_m, range_std_m)

    # Anchor the ellipse at the real prediction rather than the MC sample mean: when sigma_sin/cos
    # is large relative to sin_mu/cos_mu, normalizing noisy samples to unit vectors scatters them
    # near-uniformly around the circle and their average collapses toward the origin instead of
    # the true predicted direction. Computing spread around the fixed prediction point avoids that.
    mean_xy = np.array([range_pred * math.cos(math.radians(bearing_pred)),
                         range_pred * math.sin(math.radians(bearing_pred))])
    dx, dy = x - mean_xy[0], y - mean_xy[1]
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)],
                     [np.mean(dx * dy), np.mean(dy * dy)]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    colors = {1: "#2E7D32", 2: "#F9A825", 3: "#C62828"}
    styles = {1: "-", 2: "--", 3: ":"}
    for k in (1, 2, 3):
        theta_e, r_e = ellipse_contour(mean_xy, eigvals, eigvecs, k)
        ax.plot(theta_e, r_e, color=colors[k], linestyle=styles[k], linewidth=2,
                label=f"{k}$\\sigma$ contour")

    ax.plot(math.radians(bearing_pred), range_pred, marker="*", color="black",
            markersize=8, linestyle="none", zorder=5, label="Prediction")
    ax.scatter(0, 0, marker="s", s=20, color="black", zorder=5, label="Microphone array")

    ax.set_rmax(RANGE_MAX_M + 20)
    ax.set_rticks([50, 100, 150, 200, 250])
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_title(
        f"{data.get('condition', '?')} | {data.get('category', '?')} | {data.get('pick', '?')}\n"
        f"ID {data.get('id', '?')}  bearing_std={float(unc['bearing_std_deg']):.1f}° "
        f"range_std={range_std_m:.1f} m  (no ground truth shown)",
        fontsize=9,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.32, -0.05), fontsize=7)
    fig.subplots_adjust(right=0.72, top=0.82)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="path to one llm_interpretation sample JSON")
    ap.add_argument("--json-dir", help="directory of JSON files to plot in batch (recursive)")
    ap.add_argument("--out", help="output PNG path (single-file mode; default: alongside input)")
    ap.add_argument("--out-dir", help="output directory (batch mode; default: alongside inputs)")
    args = ap.parse_args()

    if not args.json and not args.json_dir:
        ap.error("pass --json <file> or --json-dir <folder>")

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        out_path = args.out or os.path.splitext(args.json)[0] + ".png"
        plot_one(data, out_path)
        print("wrote", out_path)
        return

    paths = sorted(glob.glob(os.path.join(args.json_dir, "**", "*.json"), recursive=True))
    if not paths:
        print("no .json files found under", args.json_dir)
        return
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
        else:
            out_path = os.path.splitext(p)[0] + ".png"
        plot_one(data, out_path)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
