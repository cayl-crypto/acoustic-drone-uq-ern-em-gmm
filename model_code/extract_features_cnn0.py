from pathlib import Path
import torch
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path("best_evidential_model.pt")

DATA_ROOTS = {
    "plain": Path("/path/to/DDLDataset/plain/features/logmel"),
    "15dB": Path("/path/to/DDLDataset/15dB/features/logmel"),
    "5dB": Path("/path/to/DDLDataset/5dB/features/logmel"),
    "-5dB": Path("/path/to/DDLDataset/-5dB/features/logmel"),
}

SAVE_ROOTS = {
    "plain": Path("/path/to/DDLDataset/plain/features/model"),
    "15dB": Path("/path/to/DDLDataset/15dB/features/model"),
    "5dB": Path("/path/to/DDLDataset/5dB/features/model"),
    "-5dB": Path("/path/to/DDLDataset/-5dB/features/model"),
}

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

    Raises error for unsupported shapes.
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
    Extract output of model.encoder.cnn[0].

    Input:
        x: [B, C, F, T]
    Output:
        feature: [B, 128, F, T]  (depends on model definition)
    """
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
    

    for pt_path in tqdm(pt_files, desc=input_dir.parent.parent.parent.name):
        try:
            x = torch.load(pt_path, map_location="cuda")

            x = ensure_4d(x).to(DEVICE)

            with torch.no_grad():
                feat = extract_conv1_feature(model, x)

            # remove batch dim if batch size is 1
            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]

            save_path = save_dir / pt_path.name
            torch.save(feat.cpu(), save_path)

        except Exception as e:
            print(f"[ERROR] Failed on {pt_path}: {e}")


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Using device: {DEVICE}")

    # load full model object
    model = torch.load(MODEL_PATH, map_location="cuda", weights_only=False)
    model = model.to(DEVICE)
    model.eval()

    for key in DATA_ROOTS:
        input_dir = DATA_ROOTS[key]
        save_dir = SAVE_ROOTS[key]

        if not input_dir.exists():
            print(f"[WARNING] Input folder does not exist: {input_dir}")
            continue

        process_folder(model, input_dir, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()