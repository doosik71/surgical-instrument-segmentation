"""Convert the exported ONNX model into a TensorRT engine."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ONNX_PATH = Path("data/model/models/model.onnx")
DEFAULT_ENGINE_PATH = Path("data/model/models/model.trt")
DEFAULT_INPUT_NAME = "image"
DEFAULT_INPUT_SIZE = (3, 480, 736)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Path to the source ONNX model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ENGINE_PATH,
        help="Path where the TensorRT engine will be written.",
    )
    parser.add_argument(
        "--trtexec",
        type=Path,
        default=None,
        help="Explicit path to trtexec.exe. If omitted, the script searches PATH and common install locations.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default=DEFAULT_INPUT_NAME,
        help="Input tensor name inside the ONNX graph.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_INPUT_SIZE[0],
        help="Input channel count used when building the optimization profile.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=DEFAULT_INPUT_SIZE[1],
        help="Input height used when building the optimization profile.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=DEFAULT_INPUT_SIZE[2],
        help="Input width used when building the optimization profile.",
    )
    parser.add_argument("--min-batch", type=int, default=1, help="Minimum batch size for the engine profile.")
    parser.add_argument("--opt-batch", type=int, default=1, help="Optimal batch size for the engine profile.")
    parser.add_argument("--max-batch", type=int, default=1, help="Maximum batch size for the engine profile.")
    parser.add_argument("--fp16", action="store_true", help="Build an FP16 TensorRT engine.")
    parser.add_argument("--int8", action="store_true", help="Build an INT8 TensorRT engine.")
    parser.add_argument(
        "--workspace",
        type=int,
        default=None,
        help="Workspace memory pool size in MiB. Passed to trtexec as --memPoolSize=workspace:<value>.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to trtexec for detailed build logs.",
    )
    return parser.parse_args()


def resolve_trtexec(explicit_path: Path | None) -> Path:
    """Resolve trtexec.exe from an explicit path, PATH, or common Windows locations."""
    candidates: list[Path] = []

    if explicit_path is not None:
        candidates.append(explicit_path)

    env_path = os.getenv("TRTEXEC_PATH")
    if env_path:
        candidates.append(Path(env_path))

    path_hit = shutil.which("trtexec.exe") or shutil.which("trtexec")
    if path_hit:
        candidates.append(Path(path_hit))

    candidates.extend(
        [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT"),
            Path(r"C:\TensorRT"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            matches = sorted(candidate.rglob("trtexec.exe"))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find trtexec.exe. Provide --trtexec <path>, set TRTEXEC_PATH, "
        "or add TensorRT's bin directory to PATH."
    )


def format_shape(batch_size: int, channels: int, input_height: int, input_width: int) -> str:
    """Format one TensorRT profile shape entry."""
    return f"{batch_size}x{channels}x{input_height}x{input_width}"


def build_command(args: argparse.Namespace, trtexec_path: Path) -> list[str]:
    """Build the trtexec command line."""
    min_shape = format_shape(args.min_batch, args.channels, args.input_height, args.input_width)
    opt_shape = format_shape(args.opt_batch, args.channels, args.input_height, args.input_width)
    max_shape = format_shape(args.max_batch, args.channels, args.input_height, args.input_width)
    shape_spec_min = f"{args.input_name}:{min_shape}"
    shape_spec_opt = f"{args.input_name}:{opt_shape}"
    shape_spec_max = f"{args.input_name}:{max_shape}"

    command = [
        str(trtexec_path),
        f"--onnx={args.onnx.resolve()}",
        f"--saveEngine={args.output.resolve()}",
        f"--minShapes={shape_spec_min}",
        f"--optShapes={shape_spec_opt}",
        f"--maxShapes={shape_spec_max}",
        "--skipInference",
    ]

    if args.fp16:
        command.append("--fp16")
    if args.int8:
        command.append("--int8")
    if args.workspace is not None:
        command.append(f"--memPoolSize=workspace:{args.workspace}")
    if args.verbose:
        command.append("--verbose")

    return command


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before invoking TensorRT."""
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if args.min_batch <= 0 or args.opt_batch <= 0 or args.max_batch <= 0:
        raise ValueError("Batch sizes must be positive integers.")
    if not (args.min_batch <= args.opt_batch <= args.max_batch):
        raise ValueError("Expected min_batch <= opt_batch <= max_batch.")
    if args.int8 and not args.fp16:
        # Keep the default explicit and predictable for common GPU deployment.
        pass


def main() -> int:
    """Convert data/model/models/model.onnx into a TensorRT engine."""
    args = parse_args()
    validate_args(args)

    trtexec_path = resolve_trtexec(args.trtexec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(args, trtexec_path)

    print(f"[INFO] Using trtexec: {trtexec_path}")
    print(f"[INFO] Source ONNX model: {args.onnx.resolve()}")
    print(f"[INFO] Output TensorRT engine: {args.output.resolve()}")
    print(
        "[INFO] Optimization profile: "
        f"min={args.min_batch} opt={args.opt_batch} max={args.max_batch} "
        f"shape={args.channels}x{args.input_height}x{args.input_width}"
    )
    if args.fp16:
        print("[INFO] FP16 mode enabled")
    if args.int8:
        print("[INFO] INT8 mode enabled")

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print(f"[INFO] Exported TensorRT engine to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
