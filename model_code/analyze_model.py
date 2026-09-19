from pathlib import Path
import math
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd

from conformer_model import Conformer8Mic
from evidential_model import EvidentialLocalization
from feature_extractor import AudioFeatureExtractor


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path("best_evidential_model.pt")
WAV_PATH = Path(
    "/path/to/audio_files/00043904.wav"
)

SAVE_ROOT = Path("./feature_drift_analysis_00043904")

SAMPLE_RATE = 96000
AUDIO_LENGTH = 9600
ONLY_FEATURE = "logmel"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_LEVELS_DB = [15, 5, -5]


# =========================================================
# HELPERS
# =========================================================
def load_wave(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected SR {SAMPLE_RATE}, got {sr} for {path.name}")

    if waveform.ndim != 2:
        raise ValueError(
            f"Expected 2D waveform, got shape {waveform.shape} for {path.name}"
        )

    num_channels, num_samples = waveform.shape

    if num_samples < AUDIO_LENGTH:
        pad = AUDIO_LENGTH - num_samples
        waveform = F.pad(waveform, (0, pad))
    elif num_samples > AUDIO_LENGTH:
        waveform = waveform[:, :AUDIO_LENGTH]

    return waveform


def add_awgn(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    waveform: [C, T]
    Returns: [C, T]
    """
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise


def save_matrix_csv(x: torch.Tensor, save_path: Path):
    arr = x.detach().cpu().numpy()
    df = pd.DataFrame(arr)
    df.to_csv(save_path, index=False, header=False)


def save_single_activation(sample_tensor: torch.Tensor, layer_dir: Path):
    """
    sample_tensor is activation for one sample only.

    Supported:
    - [C, H, W] -> one CSV per channel
    - [H, W]    -> activation.csv
    - [D]       -> activation.csv as 1 x D
    - scalar    -> activation.csv
    """
    layer_dir.mkdir(parents=True, exist_ok=True)

    x = sample_tensor.detach().cpu()

    if x.ndim == 3:
        # [C, H, W]
        C, H, W = x.shape
        for ch in range(C):
            save_matrix_csv(x[ch], layer_dir / f"ch_{ch:03d}.csv")

    elif x.ndim == 2:
        # [H, W] or [T, D] or [D, T]
        save_matrix_csv(x, layer_dir / "activation.csv")

    elif x.ndim == 1:
        # [D] -> one row
        save_matrix_csv(x.unsqueeze(0), layer_dir / "activation.csv")

    elif x.ndim == 0:
        save_matrix_csv(x.view(1, 1), layer_dir / "activation.csv")

    else:
        raise ValueError(f"Unsupported activation shape: {tuple(x.shape)}")


def save_shape_metadata(activations: dict, save_path: Path):
    rows = []
    for layer_name, act in activations.items():
        rows.append({"layer": layer_name, "shape": str(tuple(act.shape))})
    pd.DataFrame(rows).to_csv(save_path, index=False)


# =========================================================
# HOOK COLLECTOR
# =========================================================
class ActivationCollector:
    def __init__(self, layer_dict):
        self.layer_dict = layer_dict
        self.activations = {}
        self.handles = []

    def _make_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            self.activations[name] = out.detach().cpu()

        return hook

    def register(self):
        for name, layer in self.layer_dict.items():
            h = layer.register_forward_hook(self._make_hook(name))
            self.handles.append(h)

    def clear(self):
        self.activations = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_layers_to_probe(model):
    return {
        "cnn_conv1": model.encoder.cnn[0],
        "cnn_conv2": model.encoder.cnn[3],
        "proj": model.encoder.proj,
        "conf_layer0": model.encoder.encoder.conformer_layers[0],
        "conf_layer5": model.encoder.encoder.conformer_layers[5],
        "pool": model.encoder.pool,
    }


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_logmel_from_waveform(
    waveform: torch.Tensor, extractor: AudioFeatureExtractor
) -> torch.Tensor:
    """
    waveform: [C, T]
    Returns model-ready logmel with batch dimension.
    """
    waveform = waveform.to(DEVICE)
    feats = extractor.extract_all(waveform)
    logmel = feats[ONLY_FEATURE]

    # ensure batch dimension exists
    if logmel.ndim == 3:
        # expected [C, F, T] -> [1, C, F, T]
        logmel = logmel.unsqueeze(0)
    elif logmel.ndim != 4:
        raise ValueError(f"Unexpected logmel shape: {tuple(logmel.shape)}")

    return logmel


# =========================================================
# OPTIONAL: SAVE MODEL OUTPUTS
# =========================================================
def save_model_outputs(output, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if isinstance(output, dict):
        for key, value in output.items():
            if torch.is_tensor(value):
                rows.append(
                    {
                        "name": key,
                        "shape": str(tuple(value.shape)),
                        "values": value.detach().cpu().reshape(-1).tolist(),
                    }
                )
    elif torch.is_tensor(output):
        rows.append(
            {
                "name": "model_output",
                "shape": str(tuple(output.shape)),
                "values": output.detach().cpu().reshape(-1).tolist(),
            }
        )
    else:
        rows.append(
            {"name": "model_output", "shape": str(type(output)), "values": str(output)}
        )

    pd.DataFrame(rows).to_csv(save_dir / "model_outputs.csv", index=False)


# =========================================================
# MAIN
# =========================================================
@torch.no_grad()
def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # ---------------- Load model ----------------
    model = torch.load(MODEL_PATH, map_location="cpu")
    model = model.to(DEVICE)
    model.eval()

    # ---------------- Extractor ----------------
    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)

    # ---------------- Load clean waveform ----------------
    clean_waveform = load_wave(WAV_PATH)

    # ---------------- Register hooks ----------------
    collector = ActivationCollector(get_layers_to_probe(model))
    collector.register()

    # ---------------- Conditions ----------------
    conditions = {
        "clean": clean_waveform,
        "noise_15dB": add_awgn(clean_waveform, 15),
        "noise_5dB": add_awgn(clean_waveform, 5),
        "noise_minus5dB": add_awgn(clean_waveform, -5),
    }

    for condition_name, waveform in conditions.items():
        print(f"Processing: {condition_name}")

        condition_dir = SAVE_ROOT / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        # save waveform too
        save_matrix_csv(waveform, condition_dir / "waveform.csv")

        # extract logmel
        logmel = extract_logmel_from_waveform(waveform, extractor)

        # save logmel input
        # remove batch dim before saving
        logmel_no_batch = logmel[0].detach().cpu()

        if logmel_no_batch.ndim == 3:
            # [C, H, W] => one csv per channel
            logmel_dir = condition_dir / "input_logmel"
            save_single_activation(logmel_no_batch, logmel_dir)
        else:
            raise ValueError(
                f"Unexpected input logmel shape after batch removal: {tuple(logmel_no_batch.shape)}"
            )

        # forward pass
        collector.clear()
        output = model(logmel.to(DEVICE))

        activations = {k: v.clone() for k, v in collector.activations.items()}

        # save shape metadata
        save_shape_metadata(activations, condition_dir / "layer_shapes.csv")

        # save model outputs
        save_model_outputs(output, condition_dir)

        # save activations
        for layer_name, act in activations.items():
            x = act.detach().cpu()

            # Case 1: standard batch-first CNN etc. [B, ...]
            if x.ndim >= 1 and x.shape[0] == 1:
                single = x[0]

            # Case 2: time-first transformer output [T, B, D]
            elif x.ndim == 3 and x.shape[1] == 1:
                single = x[:, 0, :]  # -> [T, D]

            # Case 3: possible [D, B] style
            elif x.ndim == 2 and x.shape[1] == 1:
                single = x[:, 0]  # -> [D]

            else:
                raise ValueError(
                    f"Could not infer single-sample dimension for layer {layer_name}, shape={tuple(x.shape)}"
                )

            # Optional formatting for pooled vectors like [D,1]
            if single.ndim == 2 and single.shape[-1] == 1:
                single = single.transpose(0, 1)

            layer_dir = condition_dir / layer_name
            save_single_activation(single, layer_dir)
    collector.remove()
    print(f"\nSaved all outputs to: {SAVE_ROOT.resolve()}")


if __name__ == "__main__":
    main()
