"""Shared loaders and labeling rules for the analysis scripts.

Loaders return {normalized_id: record}. `load_ern` merges the blind uncertainty JSON with
ground_truth/ for the 7 conditions that have ground truth, so downstream code sees one record.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
GT_DIR = os.path.join(REPO, "ground_truth")

ALL_CONDITIONS = ["plain", "awgn_-5dB", "awgn_5dB", "awgn_15dB",
                  "clutter_-5dB", "clutter_5dB", "clutter_15dB", "random_noise"]
GT_CONDITIONS = [c for c in ALL_CONDITIONS if c != "random_noise"]

RANGE_MAX_M = 250.0     # range_mu / sigma_range are normalized by this (range_m_pred = range_mu * 250)
GMM_S_THRESHOLD = 95.45  # GMM sample is S (Safe) if below_min_pct + low_pct + med_pct >= this, else C (Critical)
ERN_C_MIN_BAND = 3       # ERN sample is C (Critical) if any of sin/cos/range sigma bands is >= this, else S (Safe)


def norm_id(raw_id):
    """ERN IDs are unpadded ('7341'), GMM/prediction IDs are 8-digit padded; normalize to join."""
    raw_id = str(raw_id)
    return raw_id if raw_id.startswith("rand_") else str(int(raw_id))


def _load(path):
    with open(path) as f:
        return {norm_id(r["ID"]): r for r in json.load(f)}


def load_ern(condition):
    ern = _load(os.path.join(DATA, "ern_uncertainty", f"{condition}.json"))
    if condition in GT_CONDITIONS:
        for sid, gt in _load(os.path.join(GT_DIR, f"{condition}.json")).items():
            ern[sid].update({k: v for k, v in gt.items() if k != "ID"})
    return ern


def load_gmm(condition):
    return _load(os.path.join(DATA, "gmm_bands", f"{condition}.json"))


def load_pred(condition):
    return _load(os.path.join(DATA, "predictions", f"{condition}.json"))


def ern_label(row):
    bands = [int(row["sigma_sin_band"]), int(row["sigma_cos_band"]), int(row["sigma_range_band"])]
    return "C" if any(b >= ERN_C_MIN_BAND for b in bands) else "S"


def gmm_label(row):
    safe_share = float(row["below_min_pct"]) + float(row["low_pct"]) + float(row["med_pct"])
    return "S" if safe_share >= GMM_S_THRESHOLD else "C"
