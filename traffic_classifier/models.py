"""CNN baseline and ResNet18 transfer-learning model builders."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

PoolingKind = Literal["max", "avg"]


def pretrained_resnet18_identity(
    *, enabled: bool, download: bool
) -> dict[str, str | None | bool]:
    """Materialize and fingerprint the exact ImageNet weights used for transfer."""

    if not enabled:
        return {
            "enabled": False,
            "url": None,
            "filename": None,
            "sha256": None,
        }
    if not download:
        raise ValueError("pretrained ResNet18 identity requires download=True")
    weights = ResNet18_Weights.DEFAULT
    # Resolve the cache before the run fingerprint is created. This guarantees that
    # an interrupted first run can be resumed with the same weight identity.
    weights.get_state_dict(progress=False)
    filename = Path(urlparse(weights.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"ImageNet weights were loaded but cache file is missing: {cache_path}"
        )
    import hashlib

    digest = hashlib.sha256()
    with cache_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "enabled": True,
        "url": weights.url,
        "filename": filename,
        "sha256": digest.hexdigest(),
    }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TrafficCNN(nn.Module):
    """A compact four-block CNN suitable for the small cropped images."""

    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.0,
        pooling: PoolingKind = "max",
    ) -> None:
        super().__init__()
        if pooling not in {"max", "avg"}:
            raise ValueError("pooling must be 'max' or 'avg'")
        def make_pool() -> nn.Module:
            return (
                nn.MaxPool2d(kernel_size=2, stride=2)
                if pooling == "max"
                else nn.AvgPool2d(kernel_size=2, stride=2)
            )

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("block1", ConvBlock(3, 32)),
                    ("pool1", make_pool()),
                    ("block2", ConvBlock(32, 64)),
                    ("pool2", make_pool()),
                    ("block3", ConvBlock(64, 128)),
                    ("pool3", make_pool()),
                    ("block4", ConvBlock(128, 256)),
                    ("pool4", make_pool()),
                ]
            )
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_model(
    architecture: str,
    num_classes: int,
    *,
    dropout: float = 0.0,
    pooling: PoolingKind = "max",
    pretrained: bool = True,
    download_weights: bool = True,
) -> nn.Module:
    if architecture == "cnn":
        return TrafficCNN(
            num_classes=num_classes, dropout=dropout, pooling=pooling
        )
    if architecture == "resnet18":
        if pretrained and not download_weights:
            # Avoid constructing the weights enum before validation: doing so could
            # contact the network despite the caller explicitly disabling downloads.
            raise ValueError(
                "download_weights=False is not supported for pretrained=True in this "
                "project; set pretrained=False for an explicitly scratch model"
            )
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {architecture}")


def configure_resnet_stage(model: nn.Module, stage: Literal["head", "layer4"]) -> None:
    """Freeze the backbone for head training or unfreeze layer4 and the head."""

    if not hasattr(model, "layer4") or not hasattr(model, "fc"):
        raise TypeError("configure_resnet_stage requires a ResNet18 model")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if stage == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    elif stage != "head":
        raise ValueError("stage must be 'head' or 'layer4'")


def set_batchnorm_eval(model: nn.Module) -> None:
    """Keep frozen ImageNet BatchNorm statistics fixed."""

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percent": 100.0 * trainable / total,
    }


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    head_learning_rate: float | None = None,
    backbone_learning_rate: float | None = None,
    weight_decay: float = 0.0,
) -> list[dict[str, object]]:
    """Create AdamW groups, separating a new ResNet head from its backbone."""

    if head_learning_rate is None and backbone_learning_rate is None:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "model",
            }
        ]

    head_parameters = list(model.fc.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups: list[dict[str, object]] = []
    if head_parameters:
        groups.append(
            {
                "params": [p for p in head_parameters if p.requires_grad],
                "lr": head_learning_rate or learning_rate,
                "weight_decay": weight_decay,
                "name": "head",
            }
        )
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate or learning_rate,
                "weight_decay": weight_decay,
                "name": "pretrained_backbone",
            }
        )
    return groups
