"""Build Table A (S/C band membership) and Table B (coverage + Euclidean MAE per S/C group).

S = Safe, C = Critical.

Run from anywhere:  python analysis/build_tables.py   -> writes analysis/tables/*.csv

Table A: % of samples per group, all 8 conditions.
  rows ERN, GMM (columns S, C) and ERN+GMM (columns S/S, S/C, C/S, C/C = ERN/GMM).
Table B: for the 7 conditions with ground truth, per group: n, Euclidean MAE, and cumulative
  coverage at 1/2/3 sigma. A sample is covered at k sigma only if ALL of
     |sin_mu - sin(bearing_gt)| <= k*sigma_sin
     |cos_mu - cos(bearing_gt)| <= k*sigma_cos
     |range_mu - range_gt/250|  <= k*sigma_range
  hold (range_mu / sigma_range are normalized by 250 m).
"""

import csv
import math
import os

from data_io import (ALL_CONDITIONS, GT_CONDITIONS, RANGE_MAX_M, ern_label, gmm_label,
                     load_ern, load_gmm, load_pred)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT_DIR, exist_ok=True)


def pct(count, total):
    return 100.0 * count / total if total else float("nan")


def covered_at_k(bearing_gt, range_gt, pred, k):
    sin_gt = math.sin(math.radians(bearing_gt))
    cos_gt = math.cos(math.radians(bearing_gt))
    return (abs(float(pred["sin_mu"]) - sin_gt) <= k * float(pred["sigma_sin"])
            and abs(float(pred["cos_mu"]) - cos_gt) <= k * float(pred["sigma_cos"])
            and abs(float(pred["range_mu"]) - range_gt / RANGE_MAX_M) <= k * float(pred["sigma_range"]))


GROUPS = [("ERN_S", "ERN", "S"), ("ERN_C", "ERN", "C"), ("GMM_S", "GMM", "S"), ("GMM_C", "GMM", "C"),
          ("JOINT_S/S", "ERN+GMM", "S/S"), ("JOINT_S/C", "ERN+GMM", "S/C"),
          ("JOINT_C/S", "ERN+GMM", "C/S"), ("JOINT_C/C", "ERN+GMM", "C/C")]

table_a_rows, table_b_rows = [], []

for condition in ALL_CONDITIONS:
    ern, gmm = load_ern(condition), load_gmm(condition)
    is_gt = condition in GT_CONDITIONS
    pred = load_pred(condition) if is_gt else {}
    ids = sorted(set(ern) & set(gmm) & (set(pred) if is_gt else set(ern)))
    n = len(ids)

    ern_counts, gmm_counts = {"S": 0, "C": 0}, {"S": 0, "C": 0}
    joint_counts = {"S/S": 0, "S/C": 0, "C/S": 0, "C/C": 0}
    members = {g[0]: [] for g in GROUPS}

    for sid in ids:
        el, gl = ern_label(ern[sid]), gmm_label(gmm[sid])
        ern_counts[el] += 1
        gmm_counts[gl] += 1
        joint_counts[f"{el}/{gl}"] += 1
        if is_gt:
            rec = (ern[sid], pred[sid])
            members[f"ERN_{el}"].append(rec)
            members[f"GMM_{gl}"].append(rec)
            members[f"JOINT_{el}/{gl}"].append(rec)

    table_a_rows.append({"condition": condition, "row_type": "ERN", "n_samples": n,
                         "S_pct": pct(ern_counts["S"], n), "C_pct": pct(ern_counts["C"], n)})
    table_a_rows.append({"condition": condition, "row_type": "GMM", "n_samples": n,
                         "S_pct": pct(gmm_counts["S"], n), "C_pct": pct(gmm_counts["C"], n)})
    table_a_rows.append({"condition": condition, "row_type": "ERN+GMM", "n_samples": n,
                         **{f"{k}_pct": pct(v, n) for k, v in joint_counts.items()}})

    if not is_gt:
        continue
    for key, row_type, category in GROUPS:
        recs = members[key]
        m = len(recs)
        row = {"condition": condition, "row_type": row_type, "category": category, "n_samples": m}
        if m == 0:
            row.update({"euclidean_mae_m": float("nan"), "coverage_1std_pct": float("nan"),
                        "coverage_2std_pct": float("nan"), "coverage_3std_pct": float("nan")})
        else:
            row["euclidean_mae_m"] = sum(float(e["euclidean_error_m"]) for e, _ in recs) / m
            for k in (1, 2, 3):
                hits = sum(covered_at_k(float(e["bearing_deg_gt"]), float(e["range_m_gt"]), p, k)
                           for e, p in recs)
                row[f"coverage_{k}std_pct"] = pct(hits, m)
        table_b_rows.append(row)

with open(os.path.join(OUT_DIR, "table_A_band_breakdown.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "row_type", "n_samples", "S_pct", "C_pct",
                                      "S/S_pct", "S/C_pct", "C/S_pct", "C/C_pct"])
    w.writeheader()
    w.writerows(table_a_rows)

with open(os.path.join(OUT_DIR, "table_B_coverage_and_mae.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "row_type", "category", "n_samples", "euclidean_mae_m",
                                      "coverage_1std_pct", "coverage_2std_pct", "coverage_3std_pct"])
    w.writeheader()
    w.writerows(table_b_rows)

print("wrote", OUT_DIR)
