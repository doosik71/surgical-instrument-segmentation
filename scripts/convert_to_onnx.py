"""Convert the local MONAI segmentation model weights to ONNX."""

from __future__ import annotations
from app.config.settings import AppSettings

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from monai.networks.nets import FlexibleUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT_SIZE = (480, 736)
DEFAULT_OUTPUT_PATH = Path("data/model/models/model.onnx")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the ONNX model will be written.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=DEFAULT_INPUT_SIZE[0],
        help="Model input height used for export.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=DEFAULT_INPUT_SIZE[1],
        help="Model input width used for export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use for export.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export. Use cpu unless you have a reason to export on cuda.",
    )
    return parser.parse_args()


def ensure_onnx_installed() -> None:
    """Fail fast with a clear message if ONNX is not installed."""
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "The 'onnx' package is required for export but is not installed in .venv. "
            "Install it first with your environment manager."
        )

    try:
        import onnx  # noqa: F401
    except AttributeError as error:
        if "ml_dtypes" in str(error) and "float4_e2m1fn" in str(error):
            raise RuntimeError(
                "The installed ONNX package is incompatible with the current ml_dtypes package. "
                "This environment has onnx 1.19.0 with ml_dtypes 0.4.1, which breaks ONNX import on Windows. "
                "Use a compatible pair, for example by upgrading ml_dtypes or by downgrading onnx."
            ) from error
        raise


def build_model() -> FlexibleUNet:
    """Create the exact MONAI model architecture used by the application."""
    model = FlexibleUNet(
        in_channels=3,
        out_channels=2,
        backbone="efficientnet-b2",
        spatial_dims=2,
        pretrained=False,
        is_pad=False,
        pre_conv=None,
    )
    model.eval()
    return model


def load_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    """Load the local PyTorch checkpoint into the model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Local model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


def export_model(
    *,
    model: torch.nn.Module,
    output_path: Path,
    input_height: int,
    input_width: int,
    opset: int,
    device: torch.device,
) -> None:
    """Export the MONAI model to ONNX with a dynamic batch axis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, input_height, input_width, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )


def main() -> int:
    """Convert data/model/models/model.pt into an ONNX file."""
    args = parse_args()
    ensure_onnx_installed()

    device = torch.device(args.device)
    settings = AppSettings.from_env()
    model_path = settings.local_model_path
    output_path = args.output

    model = build_model().to(device)
    load_weights(model, model_path, device)

    export_model(
        model=model,
        output_path=output_path,
        input_height=args.input_height,
        input_width=args.input_width,
        opset=args.opset,
        device=device,
    )

    print(f"[INFO] Loaded weights from {model_path.resolve()}")
    print(f"[INFO] Exported ONNX model to {output_path.resolve()}")
    print(
        f"[INFO] Input tensor shape: 1x3x{args.input_height}x{args.input_width}")
    print(
        f"[INFO] Output tensor shape: 1x2x{args.input_height}x{args.input_width}")
    print(f"[INFO] Opset version: {args.opset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
