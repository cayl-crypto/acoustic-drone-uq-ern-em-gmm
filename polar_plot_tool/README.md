# polar_plot_tool

Standalone polar-plot code, independent of the S/C analysis in `../analysis/`. Needs only
`numpy` and `matplotlib`. Both scripts draw on a fixed 0-250 m polar axis (0° at the top,
bearing increasing clockwise) with the microphone array at the origin.

| script | reads | ground truth shown | intended use |
|---|---|---|---|
| `plot_prediction_vs_gt.py` | `data/predictions/<cond>.json` + `ground_truth/<cond>.json` | yes | reviewers checking model accuracy |
| `plot_blind.py` | one `llm_interpretation/samples/**/*.json` | no | blind trust interpretation (see `../llm_interpretation/`) |

## With ground truth

```bash
python plot_prediction_vs_gt.py \
    --predictions ../data/predictions/plain.json \
    --ground-truth ../ground_truth/plain.json \
    --id 12015 --id 42468 --out-dir ./out
```

Star = prediction, blue X = ground truth, green/orange/red = 1σ/2σ/3σ ellipses. The title gives
the bearing/range/Euclidean errors and the smallest σ-ellipse that contains the ground truth.
IDs work with or without zero padding (`12015` = `00012015`). Every ID in `data/predictions/`
has a match in `ground_truth/` except for `random_noise`, which has no ground truth.

## Blind (no ground truth)

```bash
# one sample
python plot_blind.py --json ../llm_interpretation/samples/plain/ERN_S/low_12015.json

# every JSON under a folder (recursive)
python plot_blind.py --json-dir ../llm_interpretation/samples/plain --out-dir ./out/plain
```

No blind figures are pre-rendered in this repo; run this on whichever samples you want.

## How the ellipse is computed

1. Draw 500 samples: `sin ~ N(sin_mu, sigma_sin)`, `cos ~ N(cos_mu, sigma_cos)`, normalize
   `(cos, sin)` to a unit direction, and scale by `range ~ N(range_mu, sigma_range)`
   (`range_mu` and `sigma_range` are stored normalized; multiply by 250 for meters).
2. Compute the 2x2 covariance of the resulting (x, y) points and draw its k·σ ellipses.
3. The ellipse is **centered on the analytic prediction** `(range_m_pred, bearing_deg_pred)`, and
   the covariance is measured around that point rather than around the sample mean. With a very
   large `sigma_sin`/`sigma_cos` the normalized samples scatter almost uniformly around the
   circle and their mean collapses toward the origin, which would pull the ellipse away from the
   prediction. In those degenerate cases (bearing_std in the thousands of degrees) a Gaussian
   ellipse is also a poor description of what is really a ring; treat those plots as "the
   bearing is essentially unknown".
