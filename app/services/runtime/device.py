"""GPU runtime inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DeviceStatus:
    """High-level CUDA runtime state."""

    ready: bool
    cuda_available: bool
    device_label: str
    gpu_name: str | None
    reason: str | None = None


def get_device_status(require_gpu: bool) -> DeviceStatus:
    """Inspect the current torch runtime and report GPU readiness."""
    try:
        import torch
    except ImportError as error:
        return DeviceStatus(
            ready=False,
            cuda_available=False,
            device_label="unavailable",
            gpu_name=None,
            reason=f"torch import failed: {error}",
        )

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    device_label = "cuda" if cuda_available else "cpu"

    if require_gpu and not cuda_available:
        return DeviceStatus(
            ready=False,
            cuda_available=False,
            device_label=device_label,
            gpu_name=gpu_name,
            reason="CUDA device is required but not available",
        )

    return DeviceStatus(
        ready=True,
        cuda_available=cuda_available,
        device_label=device_label,
        gpu_name=gpu_name,
        reason=None,
    )
