"""Configuration loading and deterministic experiment settings."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


CLASSES = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet",
]
UNSEEN_CLASS = "neysan"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ProjectConfig:
    """All choices needed to reproduce the project."""

    data_dir: str = "dataset"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    classes: list[str] = field(default_factory=lambda: CLASSES.copy())
    seed: int = 42
    validation_fraction: float = 0.20
    minimum_image_side: int = 64
    cnn_image_size: int = 128
    transfer_image_size: int = 224
    epochs: int = 15
    batch_size: int = 32
    transfer_batch_size: int = 32
    learning_rate: float = 0.003
    transfer_head_lr: float = 0.001
    transfer_backbone_lr: float = 0.0001
    weight_decay: float = 0.0
    num_workers: int = 0
    target_review_coverage: float = 0.80
    imbalanced_retained_per_minority_class: int = 14
    imbalanced_classes: list[str] = field(
        default_factory=lambda: ["kamyun", "minibus", "savari", "vanet"]
    )
    device: str = "auto"
    pretrained: bool = True
    download_pretrained: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant random number generators."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
