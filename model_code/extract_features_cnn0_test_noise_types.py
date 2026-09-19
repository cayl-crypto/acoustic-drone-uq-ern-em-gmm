#!/usr/bin/env python3
# extract_features_cnn0_test_noise_types.py
#
# CNN conv1 ("cnn0") feature extraction for the test-set noise-robustness
# study: plain + AWGN@{15,5,-5}dB + clutter@{15,5,-5}dB.
#
# Mirrors extract_features_cnn0.py but targets the new per-noise-type
# logmel folders produced by extract_awgn_test_logmel.py /
# extract_clutter_test_logmel.py, and is resumable (skips files whose
# output already exists).
#
# Input : {condition}/features/logmel/{id}.pt
# Output: {condition}/features/model/{id}.pt   (conv1 activation, [128,128,31])

from pathlib import Path
import torch
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path("best_evidential_model.pt")
CSV_ROOT = Path("/path/to/DDLDataset")

CONDITIONS = [
    "plain",
    "awgn_15dB", "awgn_5dB", "awgn_-5dB",
    "clutter_15dB", "clutter_5dB", "clutter_-5dB",
]

DATA_ROOTS = {c: CSV_ROOT / c / "features" / "logmel" for c in CONDITIONS}
SAVE_ROOTS = {c: CSV_ROOT / c / "features" / "model" for c in CONDITIONS}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HELPERS
# =========================================================
def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def extract_conv1_feature(model, x: torch.Tensor) -> torch.Tensor:
    conv1 = model.encoder.cnn[0]
    return conv1(x)


def process_folder(model, input_dir: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        print(f"[WARNING] No .pt files found in: {input_dir}")
        return

    print(f"\nProcessing folder: {input_dir}")
    print(f"Saving to       : {save_dir}")
    print(f"Found {len(pt_files)} files")

    n_skipped = 0
    for pt_path in tqdm(pt_files, desc=input_dir.parent.parent.name):
        save_path = save_dir / pt_path.name
        if save_path.exists():
            n_skipped += 1
            continue
        try:
            x = torch.load(pt_path, map_location=DEVICE)
            x = ensure_4d(x).to(DEVICE)

            with torch.no_grad():
                feat = extract_conv1_feature(model, x)

            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]

            torch.save(feat.cpu(), save_path)

        except Exception as e:
            print(f"[ERROR] Failed on {pt_path}: {e}")

    if n_skipped:
        print(f"  Skipped {n_skipped} already-done files")


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Using device: {DEVICE}")

    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()

    for key in CONDITIONS:
        input_dir = DATA_ROOTS[key]
        save_dir = SAVE_ROOTS[key]

        if not input_dir.exists():
            print(f"[WARNING] Input folder does not exist: {input_dir}")
            continue

        process_folder(model, input_dir, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
