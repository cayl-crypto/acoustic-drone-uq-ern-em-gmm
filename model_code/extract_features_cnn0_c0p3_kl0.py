#!/usr/bin/env python3
# extract_features_cnn0_c0p3_kl0.py
#
# CNN conv1 ("cnn0") feature extraction for the
# best_evidential_loc_logmel_plain_plain_c0p3_kl0.pt checkpoint
# (coeff=0.3, kl_weight=0.0).
#
# Plain-only pipeline: reads every logmel .pt file found in the plain
# condition's logmel folder directly (no annotation CSV, no noise_level
# filtering — every ID present in that folder gets processed).
#
# Mirrors extract_features_cnn0.py's extraction logic, but:
#   - points at the new checkpoint
#   - only processes the "plain" condition
#   - writes to a *_c0p3_kl0-tagged output dir so it never collides with
#     the original best_evidential_model.pt feature cache
#   - is resumable (skips files whose output already exists)
#
# Input : plain/features/logmel/{id}.pt        shape (8, 128, 31)
# Output: plain/features/model_c0p3_kl0/{id}.pt shape (128, 128, 31)

from pathlib import Path
import torch
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path("best_evidential_loc_logmel_plain_plain_c0p3_kl0.pt")

DATASET_ROOT = Path("/path/to/DDLDataset")

INPUT_DIR = DATASET_ROOT / "plain" / "features" / "logmel"
SAVE_DIR  = DATASET_ROOT / "plain" / "features" / "model_c0p3_kl0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HELPERS
# =========================================================
def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Converts input tensor into [B, C, F, T] format if possible.

    Supported:
    - [C, F, T]     -> [1, C, F, T]
    - [1, C, F, T]  -> unchanged
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.ndim == 3:
        return x.unsqueeze(0)

    if x.ndim == 4:
        return x

    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def extract_conv1_feature(model, x: torch.Tensor) -> torch.Tensor:
    """
    Extract output of model.encoder.cnn[0] (first Conv2d in the
    Conformer8Mic CNN frontend).

    Input:  x [B, C, F, T]
    Output: feature [B, 128, F, T]
    """
    conv1 = model.encoder.cnn[0]
    return conv1(x)


def process_folder(model, input_dir: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} files in {input_dir}")
    if not pt_files:
        print(f"[WARNING] No .pt files found in: {input_dir}")
        return

    print(f"Saving to: {save_dir}")

    n_skipped = 0
    n_done = 0
    for pt_path in tqdm(pt_files, desc="plain"):
        save_path = save_dir / pt_path.name
        if save_path.exists():
            n_skipped += 1
            continue
        try:
            x = torch.load(pt_path, map_location=DEVICE)
            x = ensure_4d(x).to(DEVICE)

            with torch.no_grad():
                feat = extract_conv1_feature(model, x)

            # remove batch dim if batch size is 1
            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]

            torch.save(feat.cpu(), save_path)
            n_done += 1

        except Exception as e:
            print(f"[ERROR] Failed on {pt_path}: {e}")

    print(f"Done: {n_done} extracted, {n_skipped} already-done skipped, "
          f"{len(pt_files) - n_done - n_skipped} failed.")


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Checkpoint  : {MODEL_PATH.resolve()}")

    # load full model object
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()

    process_folder(model, INPUT_DIR, SAVE_DIR)

    print("\nAll done.")


if __name__ == "__main__":
    main()
