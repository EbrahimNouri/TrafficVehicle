"""Central registry of every project file path.

`configs/paths.json` is the single source of truth for where the project reads
and writes. Keeping the registry in one place stops scripts and modules from
each inventing their own path strings, which is how stale copies end up in
untracked directories.

All values in the registry are relative to the repository root and written in
posix style. `load_paths()` resolves them to absolute paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.json"

_RESERVED_PREFIXES = ("_",)


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten the nested registry into dotted keys."""

    flat: dict[str, str] = {}
    for key, value in values.items():
        if key.startswith("_"):
            continue
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{dotted}."))
        elif isinstance(value, str):
            flat[dotted] = value
        else:
            raise TypeError(f"path entry {dotted} must be a string, got {value!r}")
    return flat


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved access to the registry."""

    root: Path
    entries: dict[str, str]
    source: Path

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def get(self, key: str, default: str | Path | None = None) -> Path:
        """Return one path, or `default` when the key is unknown."""

        if key in self.entries:
            return self.resolve(self.entries[key])
        if default is None:
            raise KeyError(
                f"unknown path key {key!r}; known keys: {sorted(self.entries)}"
            )
        return Path(default)

    def require(self, key: str) -> Path:
        """Return one path and fail loudly when the file or directory is absent."""

        resolved = self.get(key)
        if not resolved.exists():
            raise FileNotFoundError(f"{key} points to a missing path: {resolved}")
        return resolved

    def group(self, prefix: str) -> dict[str, Path]:
        """Return every entry under a dotted prefix, e.g. `results`."""

        marker = f"{prefix}."
        return {
            key[len(marker) :]: self.resolve(value)
            for key, value in self.entries.items()
            if key.startswith(marker)
        }

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate

    def relative(self, path: str | Path) -> str:
        """Render a path relative to the root in posix style, when possible."""

        candidate = Path(path).resolve()
        try:
            return candidate.relative_to(self.root).as_posix()
        except ValueError:
            return candidate.as_posix()

    def ensure_dir(self, key: str) -> Path:
        """Return the directory for `key`, creating it when missing."""

        directory = self.get(key)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def missing(self) -> dict[str, str]:
        """Return entries whose path does not exist yet, for reporting."""

        return {
            key: value
            for key, value in sorted(self.entries.items())
            if not self.resolve(value).exists()
        }

    def verify(self) -> dict[str, Any]:
        """Summarise the registry for a report."""

        return {
            "source": self.relative(self.source),
            "root": self.root.as_posix(),
            "total_entries": len(self.entries),
            "directories": sum(
                1 for value in self.entries.values() if not Path(value).suffix
            ),
            "files": sum(
                1 for value in self.entries.values() if Path(value).suffix
            ),
            "missing": self.missing(),
        }


def load_paths(path: str | Path | None = None) -> ProjectPaths:
    """Load the registry, defaulting to `configs/paths.json` at the root."""

    source = Path(path) if path is not None else PATHS_FILE
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    root_value = raw.get("root", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (source.parent.parent / root).resolve()
    return ProjectPaths(root=root, entries=flatten(raw), source=source)