# llm_interpretation

Ground-truth-free, per-sample uncertainty summaries intended as input to an LLM (or a human
reviewer) judging how much trust to place in a model prediction, following the framing in
`prompt/LLM_prompt_guide.txt`.

## Contents

- `prompt/LLM_prompt_guide.txt` — the interpretation instructions. Deliberately GT-free:
  "You are given only the model prediction and uncertainty estimates. Ground truth is unavailable."
- `samples/<condition>/<category>/<pick>_<ID>.json` — one file per selected sample.
- `generate_llm_json.py` — regenerates all sample files from this repo's `data/` JSON (run `python llm_interpretation/generate_llm_json.py`).

## Sample selection

Same selection as the polar-plot figures in `figures/test_set/`: for each of the
7 ground-truth conditions (`plain`, `awgn_-5dB/5dB/15dB`, `clutter_-5dB/5dB/15dB`) and each of 8
categories, the 5 samples at the low/p25/median/p75/high percentiles of predicted uncertainty
(ranked by `bearing_std_deg`, then `sigma_range`) within that category. Categories:

- `ERN_S` / `ERN_C` — ERN is **C** (Critical) if any of sin/cos/range sigma falls in band 3+
  (see `analysis/build_tables.py`), otherwise **S** (Safe).
- `GMM_S` / `GMM_C` — GMM is **S** (Safe) if `below_min_pct+low_pct+med_pct >= 95.45`, otherwise **C** (Critical).
- `JOINT_S-S`, `JOINT_S-C`, `JOINT_C-S`, `JOINT_C-C` — the combined label, ERN first then GMM. Folder names
  use `-`; the JSON `category` field uses `/` (e.g. `"JOINT_S/C"`).

Most category slots are empty for the 5 noise conditions where GMM is 100% C and ERN is 100% S
(see Table A) — those combinations have no files, matching the polar-plot figures. 140 files total.

## JSON schema

```json
{
  "id": "12015", "condition": "plain", "category": "ERN_S", "pick": "low",
  "gmm": {"below_min_pct": ..., "low_pct": ..., "med_pct": ..., "high_pct": ..., "very_high_pct": ..., "above_max_pct": ...},
  "ern_uncertainty": {"sigma_sin": ..., "sigma_sin_band": ..., "sigma_cos": ..., "sigma_cos_band": ...,
                       "sigma_range": ..., "sigma_range_band": ..., "bearing_std_deg": ..., "bearing_std_deg_band": ...},
  "ern_prediction": {"sin_mu": ..., "cos_mu": ..., "range_mu": ..., "bearing_deg_pred": ..., "range_m_pred": ...}
}
```

`range_mu`/`sigma_range` are on the model's normalized 0-1 scale; multiply by 250 to get meters
(confirmed exact via `range_m_pred = range_mu * 250` across the full calibration set).

**Intentionally omitted:** `bearing_deg_gt`, `range_m_gt`, `bearing_error_deg`, `range_error_m`,
`euclidean_error_m`, and `ood_pct` (redundant with `above_max_pct`, see prior analysis).

## Plotting a sample

No pre-rendered figures are included here by design. Use `../polar_plot_tool/plot_blind.py`,
which takes one of these JSON files directly and draws the prediction + Monte-Carlo σ-ellipse,
with no ground truth shown — consistent with the blind-interpretation framing above.
