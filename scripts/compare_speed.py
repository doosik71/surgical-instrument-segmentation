"""Compare inference speed between the PyTorch model and multiple TensorRT engines."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from monai.networks.nets import FlexibleUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import AppSettings


DEFAULT_INPUT_SIZE = (480, 736)
DEFAULT_TRT_FP32_PATH = Path("data/model/models/model-fp32.trt")
DEFAULT_TRT_FP16_PATH = Path("data/model/models/model-fp16.trt")
DEFAULT_TRT_INT8_PATH = Path("data/model/models/model-int8.trt")


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregate timing for one benchmark target."""

    name: str
    batch_size: int
    iterations: int
    total_seconds: float

    @property
    def ms_per_batch(self) -> float:
        """Average latency per batch."""
        return self.total_seconds / self.iterations * 1000.0

    @property
    def ms_per_image(self) -> float:
        """Average latency per image."""
        return self.ms_per_batch / self.batch_size

    @property
    def images_per_second(self) -> float:
        """Throughput in images per second."""
        return (self.iterations * self.batch_size) / self.total_seconds


@dataclass(slots=True)
class AccuracyResult:
    """Difference summary between the PyTorch output and one TensorRT output."""

    name: str
    output_shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_SIZE[0])
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_SIZE[1])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trt-fp32-path", type=Path, default=DEFAULT_TRT_FP32_PATH)
    parser.add_argument("--trt-fp16-path", type=Path, default=DEFAULT_TRT_FP16_PATH)
    parser.add_argument("--trt-int8-path", type=Path, default=DEFAULT_TRT_INT8_PATH)
    parser.add_argument(
        "--trt-path",
        type=Path,
        default=None,
        help="Deprecated alias for --trt-fp32-path.",
    )
    parser.add_argument("--skip-accuracy-check", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the execution device."""
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise ValueError("This benchmark expects a CUDA device because TensorRT engines run on NVIDIA GPUs.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")
    return device


def synchronize(device: torch.device) -> None:
    """Synchronize pending CUDA work before reading timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_pytorch_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the original MONAI PyTorch model from the local checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch model file not found: {model_path}")

    model = FlexibleUNet(
        in_channels=3,
        out_channels=2,
        backbone="efficientnet-b2",
        spatial_dims=2,
        pretrained=False,
        is_pad=False,
        pre_conv=None,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class TensorRTRunner:
    """Run inference against a serialized TensorRT engine using torch CUDA tensors."""

    def __init__(self, engine_path: Path, device: torch.device) -> None:
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                "Failed to deserialize the TensorRT engine. "
                "The existing .trt file was likely built with a different TensorRT version. "
                f"Rebuild {engine_path} with the currently installed TensorRT runtime ({trt.__version__})."
            )

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create a TensorRT execution context.")

        self.device = device
        self.engine = engine
        self.context = context
        self.input_name = self._find_tensor_name(trt.TensorIOMode.INPUT)
        self.output_name = self._find_tensor_name(trt.TensorIOMode.OUTPUT)
        self.output_dtype = torch.from_numpy(
            np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        ).dtype

    def _find_tensor_name(self, mode: trt.TensorIOMode) -> str:
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(tensor_name) == mode:
                return tensor_name
        raise RuntimeError(f"Could not find a tensor with mode {mode}")

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute one inference and return logits on the same CUDA device."""
        if input_tensor.device.type != "cuda":
            raise ValueError("TensorRT inference expects a CUDA input tensor.")
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"Failed to set TensorRT input shape to {input_shape}")

        output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"TensorRT returned an unresolved output shape: {output_shape}")

        output_tensor = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)
        self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))

        stream = torch.cuda.current_stream(device=self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        return output_tensor


def benchmark(name: str, iterations: int, warmup: int, device: torch.device, fn) -> BenchmarkResult:
    """Benchmark one callable after a warmup phase."""
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize(device)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        batch_size=getattr(fn, "batch_size", 1),
        iterations=iterations,
        total_seconds=elapsed,
    )


def resolve_trt_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Resolve all TensorRT engine paths to compare."""
    fp32_path = args.trt_path or args.trt_fp32_path
    return [
        ("tensorrt_fp32", fp32_path),
        ("tensorrt_fp16", args.trt_fp16_path),
        ("tensorrt_int8", args.trt_int8_path),
    ]


def validate_paths(model_path: Path, trt_paths: list[tuple[str, Path]]) -> None:
    """Fail fast when any expected input model is missing."""
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch model file not found: {model_path}")
    missing = [f"{name}: {path}" for name, path in trt_paths if not path.exists()]
    if missing:
        missing_block = "\n".join(missing)
        raise FileNotFoundError(f"Missing TensorRT engine file(s):\n{missing_block}")


def print_header(args: argparse.Namespace, model_path: Path, trt_paths: list[tuple[str, Path]], device: torch.device) -> None:
    """Print benchmark setup."""
    print("PyTorch vs TensorRT speed comparison")
    print(f"device            : {device} ({torch.cuda.get_device_name(device)})")
    print(f"pytorch_model     : {model_path.resolve()}")
    for name, path in trt_paths:
        print(f"{name:<18}: {path.resolve()}")
    print(f"input_size        : {args.input_height}x{args.input_width}")
    print(f"batch_size        : {args.batch_size}")
    print(f"warmup            : {args.warmup}")
    print(f"iterations        : {args.iterations}")
    print("")


def print_accuracy_results(results: list[AccuracyResult]) -> None:
    """Print output differences between PyTorch and TensorRT engines."""
    if not results:
        return

    print(f"{'engine':<16}  {'output_shape':<18}  {'max_abs_diff':>12}  {'mean_abs_diff':>13}")
    print("-" * 68)
    for result in results:
        print(
            f"{result.name:<16}  "
            f"{str(result.output_shape):<18}  "
            f"{result.max_abs_diff:>12.6f}  "
            f"{result.mean_abs_diff:>13.6f}"
        )
    print("")


def print_results(results: list[BenchmarkResult]) -> None:
    """Print a small benchmark table."""
    print(f"{'benchmark':<16}  {'ms/batch':>10}  {'ms/image':>10}  {'images/s':>10}")
    print("-" * 54)
    for result in results:
        print(
            f"{result.name:<16}  "
            f"{result.ms_per_batch:>10.3f}  "
            f"{result.ms_per_image:>10.3f}  "
            f"{result.images_per_second:>10.2f}"
        )


def main() -> int:
    """Compare `.pt`, `model-fp32.trt`, `model-fp16.trt`, and `model-int8.trt` speed."""
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    settings = AppSettings.from_env()
    model_path = settings.local_model_path
    trt_paths = resolve_trt_paths(args)
    validate_paths(model_path, trt_paths)
    device = resolve_device(args.device)

    print_header(args, model_path, trt_paths, device)

    pytorch_model = build_pytorch_model(model_path, device)
    trt_runners = [(name, TensorRTRunner(path, device)) for name, path in trt_paths]

    input_tensor = torch.rand(
        args.batch_size,
        3,
        args.input_height,
        args.input_width,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        pytorch_logits = pytorch_model(input_tensor)
        trt_logits = [(name, runner.infer(input_tensor)) for name, runner in trt_runners]
        synchronize(device)

    if not args.skip_accuracy_check:
        accuracy_results = [
            AccuracyResult(
                name=name,
                output_shape=tuple(logits.shape),
                max_abs_diff=torch.max(torch.abs(pytorch_logits - logits)).item(),
                mean_abs_diff=torch.mean(torch.abs(pytorch_logits - logits)).item(),
            )
            for name, logits in trt_logits
        ]
        print_accuracy_results(accuracy_results)

    def run_pytorch() -> torch.Tensor:
        with torch.no_grad():
            return pytorch_model(input_tensor)

    run_pytorch.batch_size = args.batch_size  # type: ignore[attr-defined]

    results = [benchmark("pytorch_pt", args.iterations, args.warmup, device, run_pytorch)]

    for name, runner in trt_runners:
        def run_tensorrt(current_runner: TensorRTRunner = runner) -> torch.Tensor:
            return current_runner.infer(input_tensor)

        run_tensorrt.batch_size = args.batch_size  # type: ignore[attr-defined]
        results.append(benchmark(name, args.iterations, args.warmup, device, run_tensorrt))

    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
