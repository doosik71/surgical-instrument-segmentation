"""Train the temporal interpolation model directly from source videos."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None

from app.config.settings import AppSettings
from app.services.segmentation.monai_segmenter import MonaiToolSegmenter
from app.training.datasets import TemporalVideoDataset, split_video_paths
from app.training.losses import SegmentationLoss
from app.training.models import TemporalInterpolationUNetLite


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EpochMetrics:
    """Aggregated metrics for one epoch."""

    loss: float
    dice: float
    iou: float
    precision: float
    recall: float


@dataclass(slots=True)
class ProgressReporter:
    """Wrap tqdm when available and fall back to periodic logging otherwise."""

    phase_name: str
    epoch: int
    total_epochs: int
    total_batches: int
    progress_bar: object | None = None
    last_log_time: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        phase_name: str,
        epoch: int,
        total_epochs: int,
        total_batches: int,
    ) -> "ProgressReporter":
        """Create a reporter for one epoch phase."""
        description = f"{phase_name} {epoch}/{total_epochs}"
        if tqdm is not None:
            return cls(
                phase_name=phase_name,
                epoch=epoch,
                total_epochs=total_epochs,
                total_batches=total_batches,
                progress_bar=tqdm(total=total_batches, desc=description, leave=False, dynamic_ncols=True),
            )

        LOGGER.info("Starting %s epoch %s/%s with %s batches", phase_name, epoch, total_epochs, total_batches)
        return cls(
            phase_name=phase_name,
            epoch=epoch,
            total_epochs=total_epochs,
            total_batches=total_batches,
            progress_bar=None,
            last_log_time=time.monotonic(),
        )

    def update(self, batch_index: int, loss: float, dice: float, iou: float) -> None:
        """Advance visible progress."""
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(
                loss=f"{loss:.4f}",
                dice=f"{dice:.4f}",
                iou=f"{iou:.4f}",
            )
            return

        now = time.monotonic()
        should_log = batch_index == 1 or batch_index == self.total_batches or (now - self.last_log_time) >= 5.0
        if should_log:
            LOGGER.info(
                "%s epoch %s/%s | batch %s/%s | loss %.4f | dice %.4f | iou %.4f",
                self.phase_name,
                self.epoch,
                self.total_epochs,
                batch_index,
                self.total_batches,
                loss,
                dice,
                iou,
            )
            self.last_log_time = now

    def close(self) -> None:
        """Close the progress bar if one exists."""
        if self.progress_bar is not None:
            self.progress_bar.close()


def configure_logging(log_dir: Path) -> Path:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"train_interpolation_model-{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    return log_path


def set_seed(seed: int) -> None:
    """Make training reproducible enough for experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("data/video"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temporal-gap", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=736)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("data/log"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/model/interpolation"))
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def build_datasets(args: argparse.Namespace) -> tuple[TemporalVideoDataset, TemporalVideoDataset]:
    """Create train and validation datasets from source videos."""
    all_video_paths = TemporalVideoDataset.list_default_video_paths(args.video_dir)
    if args.max_videos is not None:
        all_video_paths = all_video_paths[: args.max_videos]
    train_paths, val_paths, _ = split_video_paths(
        all_video_paths,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    input_size = (args.input_height, args.input_width)
    train_dataset = TemporalVideoDataset(
        video_paths=train_paths,
        input_size=input_size,
        temporal_gap=args.temporal_gap,
        frame_stride=args.frame_stride,
        augment=True,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset = TemporalVideoDataset(
        video_paths=val_paths,
        input_size=input_size,
        temporal_gap=args.temporal_gap,
        frame_stride=args.frame_stride,
        augment=False,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    return train_dataset, val_dataset


def collate_temporal_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """Collate tensor samples without losing video metadata."""
    return {
        "previous_rgb": torch.stack([item["previous_rgb"] for item in batch]),  # type: ignore[index]
        "current_rgb": torch.stack([item["current_rgb"] for item in batch]),  # type: ignore[index]
        "video_path": [item["video_path"] for item in batch],  # type: ignore[index]
        "previous_frame_index": [item["previous_frame_index"] for item in batch],  # type: ignore[index]
        "current_frame_index": [item["current_frame_index"] for item in batch],  # type: ignore[index]
    }


def resolve_device(device_arg: str | None) -> torch.device:
    """Resolve the training device."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute Dice, IoU, precision, and recall."""
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    true_positive = (predictions * targets).sum(dim=1)
    false_positive = (predictions * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - predictions) * targets).sum(dim=1)

    dice = ((2.0 * true_positive + 1.0) / (2.0 * true_positive + false_positive + false_negative + 1.0)).mean()
    iou = ((true_positive + 1.0) / (true_positive + false_positive + false_negative + 1.0)).mean()
    precision = ((true_positive + 1.0) / (true_positive + false_positive + 1.0)).mean()
    recall = ((true_positive + 1.0) / (true_positive + false_negative + 1.0)).mean()
    return float(dice.item()), float(iou.item()), float(precision.item()), float(recall.item())


def build_teacher_masks(
    teacher: MonaiToolSegmenter,
    previous_rgb: torch.Tensor,
    current_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate previous and current pseudo masks on the teacher device."""
    teacher_inputs = torch.cat([previous_rgb, current_rgb], dim=0)
    teacher_probabilities = teacher.predict_foreground_batch(teacher_inputs)
    previous_probabilities, current_probabilities = teacher_probabilities.chunk(2, dim=0)
    previous_mask = (previous_probabilities >= teacher.mask_threshold).float().unsqueeze(1)
    current_mask = (current_probabilities >= teacher.mask_threshold).float().unsqueeze(1)
    return previous_mask, current_mask


def run_epoch(
    *,
    model: nn.Module,
    teacher: MonaiToolSegmenter,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    total_epochs: int,
) -> EpochMetrics:
    """Run one train or validation epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_batches = 0

    autocast_enabled = device.type == "cuda"
    expected_batches = len(data_loader)
    phase_name = "train" if is_training else "val"
    progress = ProgressReporter.create(
        phase_name=phase_name,
        epoch=epoch,
        total_epochs=total_epochs,
        total_batches=expected_batches,
    )

    try:
        for batch_index, batch in enumerate(data_loader, start=1):
            previous_rgb = batch["previous_rgb"].to(device)  # type: ignore[index]
            current_rgb = batch["current_rgb"].to(device)  # type: ignore[index]

            with torch.no_grad():
                previous_mask, target_mask = build_teacher_masks(teacher, previous_rgb, current_rgb)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(previous_rgb, previous_mask, current_rgb)
                loss = criterion(logits, target_mask)

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            dice, iou, precision, recall = compute_binary_metrics(logits.detach(), target_mask.detach())
            total_loss += float(loss.item())
            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_batches += 1
            progress.update(batch_index, float(loss.item()), dice, iou)
    finally:
        progress.close()

    if total_batches == 0:
        return EpochMetrics(loss=0.0, dice=0.0, iou=0.0, precision=0.0, recall=0.0)

    return EpochMetrics(
        loss=total_loss / total_batches,
        dice=total_dice / total_batches,
        iou=total_iou / total_batches,
        precision=total_precision / total_batches,
        recall=total_recall / total_batches,
    )


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    """Save a training checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        checkpoint_path,
    )


def main() -> int:
    """Train the interpolation model."""
    args = parse_args()
    log_path = configure_logging(args.log_dir)
    set_seed(args.seed)

    settings = AppSettings.from_env()
    device = resolve_device(args.device)
    LOGGER.info("Training device: %s", device)
    LOGGER.info("Training log file: %s", log_path.resolve())

    train_dataset, val_dataset = build_datasets(args)
    LOGGER.info("Train samples: %s", len(train_dataset))
    LOGGER.info("Validation samples: %s", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_temporal_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_temporal_batch,
    )

    model = TemporalInterpolationUNetLite().to(device)
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    teacher = MonaiToolSegmenter(
        settings=settings,
        input_size=(args.input_height, args.input_width),
    )
    teacher.load()
    LOGGER.info("Teacher model loaded from: %s", teacher.model_info.weights_path if teacher.model_info else settings.local_model_path)

    metrics_csv_path = args.log_dir / f"train_interpolation_metrics-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    best_checkpoint_path = args.checkpoint_dir / "temporal_interpolation_unet_lite_best.pt"
    latest_checkpoint_path = args.checkpoint_dir / "temporal_interpolation_unet_lite_latest.pt"

    best_val_loss = float("inf")

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_dice",
                "train_iou",
                "train_precision",
                "train_recall",
                "val_loss",
                "val_dice",
                "val_iou",
                "val_precision",
                "val_recall",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                teacher=teacher,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            val_metrics = run_epoch(
                model=model,
                teacher=teacher,
                data_loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                scaler=None,
                epoch=epoch,
                total_epochs=args.epochs,
            )

            LOGGER.info(
                "Epoch %s/%s | train loss %.4f dice %.4f iou %.4f | val loss %.4f dice %.4f iou %.4f",
                epoch,
                args.epochs,
                train_metrics.loss,
                train_metrics.dice,
                train_metrics.iou,
                val_metrics.loss,
                val_metrics.dice,
                val_metrics.iou,
            )
            writer.writerow(
                [
                    epoch,
                    train_metrics.loss,
                    train_metrics.dice,
                    train_metrics.iou,
                    train_metrics.precision,
                    train_metrics.recall,
                    val_metrics.loss,
                    val_metrics.dice,
                    val_metrics.iou,
                    val_metrics.precision,
                    val_metrics.recall,
                ]
            )
            csv_file.flush()

            save_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )
            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                )
                LOGGER.info("Updated best checkpoint: %s", best_checkpoint_path.resolve())

            if args.save_every_epoch:
                epoch_checkpoint_path = args.checkpoint_dir / f"temporal_interpolation_unet_lite_epoch_{epoch:03d}.pt"
                save_checkpoint(
                    epoch_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                )

    train_dataset.close()
    val_dataset.close()
    LOGGER.info("Training finished. Metrics CSV: %s", metrics_csv_path.resolve())
    LOGGER.info("Latest checkpoint: %s", latest_checkpoint_path.resolve())
    LOGGER.info("Best checkpoint: %s", best_checkpoint_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
