"""Callable and CLI single-image prediction with review-aware JSON output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ProjectConfig
from .data import build_transforms
from .metrics import probabilities_from_logits
from .models import build_model
from .utils import load_checkpoint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_config_from_checkpoint(checkpoint: dict[str, Any]) -> ProjectConfig:
    values = dict(checkpoint.get("configuration", {}))
    if not values:
        values = ProjectConfig().to_dict()
    known = ProjectConfig.__dataclass_fields__
    return ProjectConfig(**{key: value for key, value in values.items() if key in known})


def load_production_model(
    checkpoint_path: str | Path = "artifacts/checkpoints/final_model.pt",
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any], ProjectConfig, torch.device]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Production checkpoint not found: {checkpoint_path}. Run the project first."
        )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    artifacts_dir = checkpoint_path.parent.parent
    ready_path = artifacts_dir / "production_ready.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    if not ready_path.is_file() or not run_manifest_path.is_file():
        raise RuntimeError(
            "Production checkpoint is not published for a complete verified run. "
            f"Expected {ready_path} and {run_manifest_path}."
        )
    with ready_path.open("r", encoding="utf-8") as handle:
        ready = json.load(handle)
    with run_manifest_path.open("r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    run_fingerprint = checkpoint.get("run_fingerprint")
    if (
        not run_fingerprint
        or ready.get("run_fingerprint") != run_fingerprint
        or run_manifest.get("run_fingerprint") != run_fingerprint
    ):
        raise RuntimeError("Production checkpoint fingerprint does not match active run metadata")
    if ready.get("final_checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise RuntimeError("Production checkpoint hash does not match publication manifest")

    results_dir = artifacts_dir / "results"
    for filename, key in (
        ("test_outputs.npz", "test_outputs_sha256"),
        ("test_errors.csv", "test_errors_sha256"),
    ):
        evidence_path = results_dir / filename
        if not evidence_path.is_file() or _sha256_file(evidence_path) != ready.get(key):
            raise RuntimeError(f"Production evidence is missing or modified: {evidence_path}")
    if checkpoint.get("source_checkpoint_sha256") != ready.get(
        "source_checkpoint_sha256"
    ):
        raise RuntimeError("Production checkpoint source hash disagrees with publication manifest")
    selected_name = str(ready.get("selected_experiment", ""))
    source_checkpoint = checkpoint_path.parent / f"{selected_name}.pt"
    if (
        not selected_name
        or not source_checkpoint.is_file()
        or _sha256_file(source_checkpoint) != checkpoint.get("source_checkpoint_sha256")
    ):
        raise RuntimeError("Selected source checkpoint is missing or has been modified")

    config = _project_config_from_checkpoint(checkpoint)
    target_device = torch.device(device)
    options = checkpoint.get("train_options", {})
    metadata = checkpoint.get("metadata", {})
    architecture = options.get("architecture") or metadata.get("architecture", "cnn")
    transform = checkpoint.get("transform", metadata.get("transform", {}))
    transform_kind = transform.get("name", "resnet" if architecture == "resnet18" else "cnn")
    _, generated_transform = build_transforms(
        config, augmentation=False, kind=transform_kind  # type: ignore[arg-type]
    )
    expected_transform = generated_transform.to_dict()
    for key in (
        "name",
        "image_size",
        "augmentation",
        "normalization_mean",
        "normalization_std",
        "operations",
    ):
        if transform.get(key) != expected_transform.get(key):
            raise RuntimeError(
                f"Serialized inference transform does not match configuration: {key}"
            )
    if "temperature" not in checkpoint or "threshold" not in checkpoint:
        raise RuntimeError("Production checkpoint is missing calibration metadata")
    class_names = list(checkpoint.get("class_names", []))
    expected_mapping = {name: index for index, name in enumerate(class_names)}
    if checkpoint.get("class_mapping") != expected_mapping or not class_names:
        raise RuntimeError("Production checkpoint has an invalid class mapping")
    model = build_model(
        architecture,
        len(class_names),
        dropout=float(metadata.get("dropout", 0.0)),
        pooling=str(metadata.get("pooling", "max")),  # type: ignore[arg-type]
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(target_device)
    model.eval()
    return model, checkpoint, config, target_device


@torch.inference_mode()
def predict_image(
    image_path: str | Path,
    *,
    checkpoint_path: str | Path = "artifacts/checkpoints/final_model.pt",
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Predict one image and return the required review-aware JSON-compatible dict."""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    model, checkpoint, config, target_device = load_production_model(
        checkpoint_path, device
    )
    transform_kind = checkpoint.get("transform", {}).get(
        "name", "resnet" if checkpoint.get("train_options", {}).get("architecture") == "resnet18" else "cnn"
    )
    transform, _ = build_transforms(
        config, augmentation=False, kind=transform_kind  # type: ignore[arg-type]
    )
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            tensor = transform(rgb).unsqueeze(0).to(target_device)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Unsupported or unreadable image: {path}") from error
    logits = model(tensor)
    temperature = float(checkpoint.get("temperature", 1.0))
    probabilities = probabilities_from_logits(
        logits.cpu().numpy(), kind="softmax", temperature=temperature
    )[0]
    class_names = list(checkpoint["class_names"])
    predicted_index = int(probabilities.argmax())
    confidence = float(probabilities[predicted_index])
    threshold = float(checkpoint.get("threshold", 0.70))
    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(confidence, 6),
        "probabilities": {
            class_name: round(float(probabilities[index]), 6)
            for index, class_name in enumerate(class_names)
        },
        "needs_review": bool(confidence < threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a traffic-vehicle image as JSON")
    parser.add_argument("image", help="Path to one JPEG/PNG image")
    parser.add_argument(
        "--checkpoint", default="artifacts/checkpoints/final_model.pt"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = predict_image(
        args.image, checkpoint_path=args.checkpoint, device=args.device
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
