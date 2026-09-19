"""Convert the original per-sample CSV exports into the JSON files shipped in this repo.

Provenance/documentation script: it needs the ORIGINAL project layout (ERNPredictions/,
ERNResults/, GMMResults/) as --src, which is not part of this repo. You do not need to run it
to use the repo; the JSON it produced is already in data/ and ground_truth/.

    python convert_csv_to_json.py --src "/path/to/ICASSP 2027" --dst /path/to/this/repo

What it does
- One JSON array per (dataset, condition); each element is one CSV row, typed (int/float/null).
- Ground truth and GT-derived errors are split OUT of the ERN results into ground_truth/, so
  data/ contains predictions and uncertainty only.
- GMM: drops columns that carry no information or leak cluster paths
  (band_pt_path, feature_file, missing_feature, error).
"""

import argparse
import csv
import json
import os

CONDITIONS = ["plain", "awgn_-5dB", "awgn_5dB", "awgn_15dB",
              "clutter_-5dB", "clutter_5dB", "clutter_15dB", "random_noise"]
GT_CONDITIONS = [c for c in CONDITIONS if c != "random_noise"]

STRING_COLS = {"ID", "condition", "noise_level", "distance"}
GT_COLS = ["bearing_deg_gt", "range_m_gt", "bearing_error_deg", "range_error_m", "euclidean_error_m"]
GMM_DROP = {"band_pt_path", "feature_file", "missing_feature", "error"}


def typed(col, value):
    if col in STRING_COLS:
        return value
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path):
    with open(path, newline="") as f:
        return [{k: typed(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_json(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("[\n")
        f.write(",\n".join(json.dumps(r, separators=(",", ":")) for r in records))
        f.write("\n]\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original project root")
    ap.add_argument("--dst", required=True, help="repo root")
    args = ap.parse_args()

    for cond in CONDITIONS:
        pred = read_rows(os.path.join(args.src, "ERNPredictions", f"{cond}_evidential_raw_output.csv"))
        write_json(os.path.join(args.dst, "data", "predictions", f"{cond}.json"), pred)

        ern = read_rows(os.path.join(args.src, "ERNResults", "test_conformal_bands",
                                      f"{cond}_evidential_conformal_bands_per_sample.csv"))
        if cond in GT_CONDITIONS:
            gt = [{"ID": r["ID"], **{c: r[c] for c in GT_COLS}} for r in ern]
            write_json(os.path.join(args.dst, "ground_truth", f"{cond}.json"), gt)
        ern_blind = [{k: v for k, v in r.items() if k not in GT_COLS} for r in ern]
        write_json(os.path.join(args.dst, "data", "ern_uncertainty", f"{cond}.json"), ern_blind)

        gmm = read_rows(os.path.join(args.src, "GMMResults", "conformal_bands",
                                      f"{cond}_conformal_band_summary_per_sample.csv"))
        gmm = [{k: v for k, v in r.items() if k not in GMM_DROP} for r in gmm]
        write_json(os.path.join(args.dst, "data", "gmm_bands", f"{cond}.json"), gmm)
        print("converted", cond)


if __name__ == "__main__":
    main()
