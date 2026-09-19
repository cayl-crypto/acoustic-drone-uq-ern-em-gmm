"""Regenerate llm_interpretation/samples/**.json from the repo's data/ JSON.

    python llm_interpretation/generate_llm_json.py [--out-dir llm_interpretation/samples]

Ground truth and error fields are never read here, so the output is blind by construction.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from data_io import GT_CONDITIONS, REPO, ern_label, gmm_label, load_ern, load_gmm, load_pred  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=os.path.join(REPO, "llm_interpretation", "samples"))
OUT_DIR = ap.parse_args().out_dir
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["ERN_S", "ERN_C", "GMM_S", "GMM_C",
              "JOINT_S/S", "JOINT_S/C", "JOINT_C/S", "JOINT_C/C"]
PICK_LABELS_5 = ["low", "p25", "median", "p75", "high"]


def pick_indices(n):
    if n >= 5:
        return list(zip(PICK_LABELS_5, [0, round(0.25 * (n - 1)), round(0.5 * (n - 1)),
                                         round(0.75 * (n - 1)), n - 1]))
    labels_by_n = {1: ["only"], 2: ["low", "high"], 3: ["low", "median", "high"],
                   4: ["low", "p33", "p67", "high"]}
    return list(zip(labels_by_n[n], range(n)))


def build_sample_json(sid, condition, category, pick, erow, grow, prow):
    return {
        "id": sid,
        "condition": condition,
        "category": category,
        "pick": pick,
        "gmm": {
            "below_min_pct": float(grow["below_min_pct"]),
            "low_pct": float(grow["low_pct"]),
            "med_pct": float(grow["med_pct"]),
            "high_pct": float(grow["high_pct"]),
            "very_high_pct": float(grow["very_high_pct"]),
            "above_max_pct": float(grow["above_max_pct"]),
        },
        "ern_uncertainty": {
            "sigma_sin": float(erow["sigma_sin"]),
            "sigma_sin_band": int(erow["sigma_sin_band"]),
            "sigma_cos": float(erow["sigma_cos"]),
            "sigma_cos_band": int(erow["sigma_cos_band"]),
            "sigma_range": float(erow["sigma_range"]),
            "sigma_range_band": int(erow["sigma_range_band"]),
            "bearing_std_deg": float(erow["bearing_std_deg"]),
            "bearing_std_deg_band": int(erow["bearing_std_deg_band"]),
        },
        "ern_prediction": {
            "sin_mu": float(prow["sin_mu"]),
            "cos_mu": float(prow["cos_mu"]),
            "range_mu": float(prow["range_mu"]),
            "bearing_deg_pred": float(prow["bearing_deg_pred"]),
            "range_m_pred": float(prow["range_m_pred"]),
        },
    }


total_written = 0
for condition in GT_CONDITIONS:
    ern, gmm, pred = load_ern(condition), load_gmm(condition), load_pred(condition)
    records = [{"id": sid, "ern": ern_label(ern[sid]), "gmm": gmm_label(gmm[sid]),
                "bearing_std": float(ern[sid]["bearing_std_deg"]), "range_std": float(ern[sid]["sigma_range"])}
               for sid in sorted(set(ern) & set(gmm) & set(pred))]
    buckets = {c: [] for c in CATEGORIES}
    for rec in records:
        buckets[f"ERN_{rec['ern']}"].append(rec)
        buckets[f"GMM_{rec['gmm']}"].append(rec)
        buckets[f"JOINT_{rec['ern']}/{rec['gmm']}"].append(rec)

    for category in CATEGORIES:
        group = buckets[category]
        if not group:
            continue
        group.sort(key=lambda r: (r["bearing_std"], r["range_std"]))
        cat_dir = os.path.join(OUT_DIR, condition.replace("-", "m"), category.replace("/", "-"))
        os.makedirs(cat_dir, exist_ok=True)
        for label, idx in pick_indices(len(group)):
            sid = group[idx]["id"]
            data = build_sample_json(sid, condition, category, label, ern[sid], gmm[sid], pred[sid])
            with open(os.path.join(cat_dir, f"{label}_{sid}.json"), "w") as f:
                json.dump(data, f, indent=2)
            total_written += 1

print("total JSON files written:", total_written, "->", OUT_DIR)
