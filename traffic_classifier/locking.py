"""Cross-platform single-writer lock for a project artifacts directory."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any


class RunLockError(RuntimeError):
    """Raised when another process already owns the artifacts directory."""


class ProjectRunLock:
    """Acquire an advisory OS lock that is released automatically on process exit."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.directory = Path(artifacts_dir)
        self.path = self.directory / ".run.lock"
        self._handle: Any = None

    def __enter__(self) -> "ProjectRunLock":
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        try:
            self._handle.seek(0)
            if self._handle.read(1) == b"":
                self._handle.write(b"0")
                self._handle.flush()
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        except OSError as error:
            owner = self._read_owner()
            self._handle.close()
            self._handle = None
            raise RunLockError(
                f"Artifacts directory is already locked: {self.directory}. "
                f"Owner: {owner}"
            ) from error
        metadata = (
            f"pid={os.getpid()}\nhost={socket.gethostname()}\n"
            f"started={datetime.now(timezone.utc).isoformat()}\n"
        ).encode("utf-8")
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(metadata)
        self._handle.flush()
        return self

    def _read_owner(self) -> str:
        try:
            with self.path.open("rb") as handle:
                return handle.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return "unknown"

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        try:
            if os.name != "nt":
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            # Closing a Windows handle releases an msvcrt byte-range lock. An explicit
            # LK_UNLCK can itself fail after the locked region is rewritten.
        finally:
            self._handle.close()
            self._handle = None
