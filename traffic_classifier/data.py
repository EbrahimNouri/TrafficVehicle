"""Dataset loading, leakage-safe splitting, transforms, and sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms

from .config import ProjectConfig, dataloader_generator


TransformKind = Literal["cnn", "resnet"]


@dataclass(frozen=True)
class ImageTransformSpec:
    name: str
    image_size: int
    augmentation: bool
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    interpolation: str = "bilinear"
    operations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_size": self.image_size,
            "augmentation": self.augmentation,
            "normalization_mean": list(self.normalization_mean),
            "normalization_std": list(self.normalization_std),
            "interpolation": self.interpolation,
            "operations": list(self.operations),
        }


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Every extension torchvision's default ImageFolder extension list accepts, plus
# the lossless formats commonly found in traffic-camera exports. Kept as a single
# constant so "does this split hold any image?" and "is this class folder
# empty?" can never disagree about what counts as an image.
IMAGE_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def build_transforms(
    config: ProjectConfig,
    *,
    augmentation: bool,
    kind: TransformKind = "cnn",
) -> tuple[transforms.Compose, ImageTransformSpec]:
    """Build and describe the exact transform used by a model run."""

    image_size = (
        config.cnn_image_size if kind == "cnn" else config.transfer_image_size
    )
    interpolation = transforms.InterpolationMode.BILINEAR
    operations: list[Any] = []
    if augmentation:
        if kind == "cnn":
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.78, 1.0),
                        ratio=(0.70, 1.45),
                        interpolation=interpolation,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.20, 0.20, 0.12, 0.04)], p=0.7
                    ),
                    transforms.RandomRotation(degrees=8),
                ]
            )
        else:
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.80, 1.0),
                        ratio=(0.75, 1.33),
                        interpolation=interpolation,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
    else:
        operations.append(
            transforms.Resize(
                (image_size, image_size), interpolation=interpolation, antialias=True
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if augmentation and kind == "cnn":
        operation_names = (
            "RandomResizedCrop(size=96 placeholder, scale=(0.78, 1.0), ratio=(0.70, 1.45), bilinear)",
            "RandomHorizontalFlip(p=0.5)",
            "RandomApply(ColorJitter(brightness=0.20, contrast=0.20, saturation=0.12, hue=0.04), p=0.7)",
            "RandomRotation(degrees=8)",
            "ToTensor()",
            f"Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})",
        )
    elif augmentation and kind == "resnet":
        operation_names = (
            "RandomResizedCrop(size=160 placeholder, scale=(0.80, 1.0), ratio=(0.75, 1.33), bilinear)",
            "RandomHorizontalFlip(p=0.5)",
            "ToTensor()",
            f"Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})",
        )
    else:
        operation_names = (
            f"Resize(size=({image_size}, {image_size}), bilinear, antialias=True)",
            "ToTensor()",
            f"Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})",
        )
    operation_names = tuple(
        item.replace("size=96 placeholder", f"size={image_size}")
        .replace("size=160 placeholder", f"size={image_size}")
        for item in operation_names
    )
    spec = ImageTransformSpec(
        name=kind,
        image_size=image_size,
        augmentation=augmentation,
        normalization_mean=IMAGENET_MEAN,
        normalization_std=IMAGENET_STD,
        operations=operation_names,
    )
    return transforms.Compose(operations), spec


def count_images(root: str | Path) -> int:
    """Count image files under a directory, or 0 if the directory is absent.

    A directory that exists but holds no image (for example `unclean` after a
    `--mode move` export drained it) reports 0 rather than raising, so callers
    can skip it instead of failing the whole run.
    """

    path = Path(root)
    if not path.is_dir():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def split_image_count(root: str | Path, split: str) -> int:
    """Count image files under `<root>/<split>`, or 0 if the split is absent."""

    return count_images(Path(root) / split)


def load_image_folder(
    root: str | Path,
    split: str,
    transform: transforms.Compose,
    expected_classes: Sequence[str] | None = None,
    remove_empty_classes: bool = True,
    allow_empty: bool = False,
) -> datasets.ImageFolder:
    """Load one split separately, avoiding the incorrect `ImageFolder(dataset)`.

    Args:
        root: Root directory of the dataset.
        split: Name of the split (e.g. 'train', 'val', 'test').
        transform: Transform to apply to images.
        expected_classes: Optional sequence of expected class names. When given,
            the folder's class order must match this list exactly.
        remove_empty_classes: If True (default), delete class folders that contain
            no image with a torchvision-supported extension.
        allow_empty: Passed to `ImageFolder`. If True, class folders with zero
            images are kept, so the class mapping still matches
            `expected_classes`. Required for evaluation splits where a class may
            have no test images (e.g. `ambulance` in this project's test split).
    """

    path = Path(root) / split
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")

    if remove_empty_classes:
        import shutil

        for class_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            has_image = any(
                item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
                for item in class_dir.rglob("*")
            )
            if not has_image:
                print(f"[load_image_folder] removing empty class dir: {class_dir}")
                shutil.rmtree(class_dir, ignore_errors=True)

    dataset = datasets.ImageFolder(
        path, transform=transform, allow_empty=allow_empty
    )
    if expected_classes is not None:
        if dataset.classes != list(expected_classes):
            raise ValueError(
                f"Class mapping mismatch in {split}: {dataset.class_to_idx} != "
                f"{dict(zip(expected_classes, range(len(expected_classes))))}"
            )
    return dataset


def canonical_pixel_hash(path: str | Path) -> str:
    """Hash decoded, orientation-normalized RGB pixels rather than JPEG metadata.

    Raw file SHA-256 can miss visually identical images when source metadata or
    encoding differs. Hashing a version tag, dimensions, and decoded RGB bytes is
    independent of source-file encoding and catches these cross-split duplicates.
    """

    import hashlib

    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256()
        digest.update(b"canonical-rgb-v1\0")
        digest.update(rgb.width.to_bytes(4, byteorder="big"))
        digest.update(rgb.height.to_bytes(4, byteorder="big"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def raw_file_hash(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_split(
    root: str | Path, split: str, minimum_image_side: int
) -> list[dict[str, Any]]:
    """Collect machine-readable image records and quality issues."""

    records: list[dict[str, Any]] = []
    split_path = Path(root) / split
    for path in sorted(item for item in split_path.rglob("*") if item.is_file()):
        relative = path.relative_to(split_path).as_posix()
        record: dict[str, Any] = {
            "path": relative,
            "absolute_path": path.as_posix(),
            "split": split,
            "label": path.parent.name if path.parent != split_path else None,
            "suffix": path.suffix.lower(),
            "file_bytes": path.stat().st_size,
            "raw_sha256": None,
            "pixel_sha256": None,
            "format": None,
            "mode": None,
            "width": None,
            "height": None,
            "issues": [],
        }
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            record["issues"].append("unsupported_format")
        try:
            record["raw_sha256"] = raw_file_hash(path)
            record["pixel_sha256"] = canonical_pixel_hash(path)
            with Image.open(path) as image:
                image.load()
                record["format"] = image.format
                record["mode"] = image.mode
                record["width"], record["height"] = image.size
            if min(record["width"], record["height"]) < minimum_image_side:
                record["issues"].append("unusually_small")
            aspect_ratio = record["width"] / record["height"]
            if not 0.2 <= aspect_ratio <= 5.0:
                record["issues"].append("unusual_aspect_ratio")
        except Exception as error:  # pragma: no cover - exercised with corrupt fixtures
            record["issues"].append(f"unreadable:{type(error).__name__}")
        records.append(record)
    return records


def pixel_duplicate_groups(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        digest = record.get("pixel_sha256")
        if digest:
            groups.setdefault(digest, []).append(record)
    return [group for group in groups.values() if len(group) > 1]


def cleaned_unclean_indices(
    train_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
    unclean_dataset: datasets.ImageFolder,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Remove unclean copies of frozen train/test content and document conflicts."""

    protected_hashes: dict[str, list[tuple[str, str, str]]] = {}
    for split_name, dataset in (("train", train_dataset), ("test", test_dataset)):
        for path, label in dataset.samples:
            digest = canonical_pixel_hash(path)
            protected_hashes.setdefault(digest, []).append(
                (split_name, label, Path(path).name)
            )

    retained: list[int] = []
    exclusions: list[dict[str, Any]] = []
    for index, (path, label) in enumerate(unclean_dataset.samples):
        digest = canonical_pixel_hash(path)
        if digest not in protected_hashes:
            retained.append(index)
            continue
        sources = protected_hashes[digest]
        exclusions.append(
            {
                "unclean_path": Path(path).as_posix(),
                "unclean_label": label,
                "pixel_sha256": digest,
                "duplicate_of": [
                    {"split": source_split, "label": source_label, "filename": filename}
                    for source_split, source_label, filename in sources
                ],
                "label_conflict": any(source_label != label for source_split, source_label, _ in sources),
            }
        )
    return retained, exclusions


def stratified_validation_indices(
    dataset: datasets.ImageFolder,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Split sample indices by class with a fixed NumPy seed."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(dataset.targets):
        by_class.setdefault(target, []).append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    records: list[dict[str, Any]] = []
    for class_index in sorted(by_class):
        indices = np.asarray(by_class[class_index], dtype=int)
        rng.shuffle(indices)
        validation_count = int(round(len(indices) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(indices) - 1)
        val = sorted(indices[:validation_count].tolist())
        train = sorted(indices[validation_count:].tolist())
        train_indices.extend(train)
        validation_indices.extend(val)
        records.append(
            {
                "class_index": class_index,
                "class_name": dataset.classes[class_index],
                "train_indices": train,
                "validation_indices": val,
                "validation_paths": [dataset.samples[index][0] for index in val],
            }
        )
    return sorted(train_indices), sorted(validation_indices), records


def simulated_imbalance_indices(
    train_indices: Sequence[int],
    targets: Sequence[int],
    class_to_idx: dict[str, int],
    minority_classes: Sequence[str],
    retained_minority: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Retain a reproducible subset from selected training classes."""

    rng = np.random.default_rng(seed)
    selected = set(class_to_idx[name] for name in minority_classes)
    indices_by_class: dict[int, list[int]] = {}
    for index in train_indices:
        indices_by_class.setdefault(targets[index], []).append(index)

    retained: list[int] = []
    details: dict[str, Any] = {
        "seed": seed,
        "minority_classes": list(minority_classes),
        "requested_minority_retained": retained_minority,
        "retained_indices": [],
        "class_counts": {},
    }
    for class_index in sorted(indices_by_class):
        candidates = np.asarray(indices_by_class[class_index], dtype=int)
        rng.shuffle(candidates)
        count = (
            min(retained_minority, len(candidates))
            if class_index in selected
            else len(candidates)
        )
        kept = sorted(candidates[:count].tolist())
        retained.extend(kept)
        details["class_counts"][str(class_index)] = count
        details["retained_indices"].extend(kept)
    details["retained_indices"] = sorted(details["retained_indices"])
    details["total_retained"] = len(retained)
    return sorted(retained), details


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield batches containing an equal count from every class.

    The sampler oversamples each class to the largest class count. It can sample
    with replacement for smaller classes and is deterministic for a given seed and
    epoch. Pass this as `batch_sampler`; do not also provide `batch_size`, `shuffle`,
    or `sampler` to `DataLoader`.
    """

    def __init__(
        self,
        targets: Sequence[int],
        indices: Sequence[int],
        num_classes: int,
        batch_size: int,
        seed: int = 42,
        num_samples: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size % num_classes != 0:
            raise ValueError("batch_size must be divisible by the number of classes")
        if drop_last:
            raise ValueError("drop_last=True is incompatible with exact class balance")
        self.targets = list(targets)
        self.indices = list(indices)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.samples_per_class = batch_size // num_classes
        self.seed = seed
        self.epoch = 0
        self.indices_by_class: list[list[int]] = [[] for _ in range(num_classes)]
        for index in self.indices:
            target = int(self.targets[index])
            if not 0 <= target < num_classes:
                raise ValueError(f"Invalid target {target}")
            self.indices_by_class[target].append(index)
        if any(not values for values in self.indices_by_class):
            missing = [i for i, values in enumerate(self.indices_by_class) if not values]
            raise ValueError(f"BalancedBatchSampler has no examples for classes {missing}")
        largest_class = max(len(values) for values in self.indices_by_class)
        self.num_samples = int(num_samples or largest_class)
        if self.num_samples % self.samples_per_class != 0:
            self.num_samples = math.ceil(
                self.num_samples / self.samples_per_class
            ) * self.samples_per_class
        self.num_batches = self.num_samples // self.samples_per_class
        self.total_size = self.num_batches * self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        class_streams: list[list[int]] = []
        class_generators: list[torch.Generator] = []
        for class_index, values in enumerate(self.indices_by_class):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + self.epoch * self.num_classes + class_index
            )
            class_generators.append(generator)
            order = torch.as_tensor(values, dtype=torch.long)[
                torch.randperm(len(values), generator=generator)
            ].tolist()
            class_streams.append(order)

        for _ in range(self.num_batches):
            batch: list[int] = []
            for class_index, stream in enumerate(class_streams):
                for _ in range(self.samples_per_class):
                    if not stream:
                        values = self.indices_by_class[class_index]
                        stream[:] = torch.as_tensor(values, dtype=torch.long)[
                            torch.randperm(
                                len(values), generator=class_generators[class_index]
                            )
                        ].tolist()
                    batch.append(stream.pop())
            yield batch


def make_loader(
    dataset: Dataset[Any],
    indices: Sequence[int],
    *,
    training: bool,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    balanced_sampler: BalancedBatchSampler | None = None,
    steps_per_epoch: int | None = None,
) -> DataLoader[Any]:
    if balanced_sampler is not None:
        # The sampler owns the exact parent-dataset indices. Do not wrap the parent
        # in a second Subset, which would require translating sampler indices.
        return DataLoader(
            dataset,
            batch_sampler=balanced_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    subset = dataset if isinstance(dataset, Subset) else Subset(dataset, list(indices))

    generator = dataloader_generator(seed)
    if steps_per_epoch is not None:
        sampler = torch.utils.data.RandomSampler(
            subset, replacement=True, num_samples=steps_per_epoch * batch_size,
            generator=generator,
        )
    else:
        sampler = torch.utils.data.RandomSampler(
            subset, generator=generator
        ) if training else torch.utils.data.SequentialSampler(subset)
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def split_record(
    dataset: datasets.ImageFolder,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    transform_spec: ImageTransformSpec,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    def describe(indices: Sequence[int]) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "label_index": int(dataset.targets[index]),
                "label": dataset.classes[int(dataset.targets[index])],
                "path": Path(dataset.samples[index][0]).as_posix(),
            }
            for index in indices
        ]

    return {
        "seed": seed,
        "validation_fraction": float(validation_fraction),
        "class_to_idx": dataset.class_to_idx,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "train": describe(train_indices),
        "validation": describe(validation_indices),
        "transform": transform_spec.to_dict(),
    }


def _all_dataset_paths(dataset: Dataset[Any]) -> list[Path]:
    if isinstance(dataset, Subset):
        parent_paths = _all_dataset_paths(dataset.dataset)
        if any(index < 0 or index >= len(parent_paths) for index in dataset.indices):
            raise IndexError("Nested Subset contains an invalid parent index")
        return [parent_paths[index] for index in dataset.indices]
    if isinstance(dataset, datasets.ImageFolder):
        return [Path(path) for path, _ in dataset.samples]
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def paths_for_subset(dataset: Dataset[Any], indices: Sequence[int]) -> list[Path]:
    """Resolve paths in dataset order.

    `indices` are positions in the supplied dataset (including a possible
    ``Subset``), not parent-dataset indices. Passing an empty sequence returns all
    stored samples in deterministic loader order.
    """

    positions = list(range(len(dataset))) if len(indices) == 0 else list(indices)
    if any(position < 0 or position >= len(dataset) for position in positions):
        raise IndexError("Dataset position is outside the supplied dataset")
    stored_paths = _all_dataset_paths(dataset)
    return [stored_paths[position] for position in positions]
