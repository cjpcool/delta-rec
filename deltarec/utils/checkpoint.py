from __future__ import annotations

import os

from pathlib import Path

import shutil

import tempfile

import math

from typing import Any, Mapping

class AtomicCheckpointError(RuntimeError):
    pass

DEFAULT_EXPECTED_CHECKPOINT_BYTES = 1024**3

DEFAULT_SPACE_RESERVE_BYTES = 1024**3

DEFAULT_SPACE_MULTIPLIER = 1.25

def checkpoint_space_required(expected_bytes: int) -> int:
    """Return write headroom without inspecting or validating checkpoint payloads."""

    multiplier = float(
        os.environ.get("DELTAREC_CHECKPOINT_SPACE_MULTIPLIER", DEFAULT_SPACE_MULTIPLIER)
    )
    reserve = int(
        os.environ.get("DELTAREC_CHECKPOINT_SPACE_RESERVE_BYTES", DEFAULT_SPACE_RESERVE_BYTES)
    )
    if expected_bytes < 1 or multiplier < 1.0 or reserve < 0:
        raise AtomicCheckpointError("invalid checkpoint capacity configuration")
    return math.ceil(expected_bytes * multiplier) + reserve

def require_checkpoint_space(path: Path, expected_bytes: int, *, label: str) -> None:
    """Fail before a save when the filesystem lacks conservative write headroom."""

    probe = path.expanduser().resolve()
    while not probe.exists():
        if probe == probe.parent:
            break
        probe = probe.parent
    available = shutil.disk_usage(probe).free
    required = checkpoint_space_required(expected_bytes)
    if available < required:
        raise AtomicCheckpointError(
            f"insufficient {label} checkpoint space at {probe}: "
            f"available={available} required={required} expected_write={expected_bytes}"
        )

def recent_file_footprint(directory: Path, *, suffix: str = ".pt") -> int | None:
    candidates = [
        path.stat().st_size
        for path in directory.glob(f"*{suffix}")
        if path.is_file() and not path.name.startswith(".")
    ] if directory.is_dir() else []
    return max(candidates, default=None)

def atomic_torch_save(
    torch: Any,
    payload: Mapping[str, Any],
    destination: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Spool locally, fsync, copy to an NFS temporary, then atomically replace."""

    target = destination.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise AtomicCheckpointError(f"refusing to overwrite checkpoint: {target}")
    spool = Path(os.environ.get("DELTAREC_CHECKPOINT_SPOOL", tempfile.gettempdir()))
    spool.mkdir(parents=True, exist_ok=True)
    expected_bytes = recent_file_footprint(target.parent)
    if expected_bytes is None:
        expected_bytes = int(
            os.environ.get(
                "DELTAREC_CHECKPOINT_EXPECTED_BYTES", DEFAULT_EXPECTED_CHECKPOINT_BYTES
            )
        )
    require_checkpoint_space(spool, expected_bytes, label="local spool")
    require_checkpoint_space(target.parent, expected_bytes, label="destination")
    descriptor, local_name = tempfile.mkstemp(
        prefix=f"deltarec-{target.name}-", suffix=".pt", dir=spool
    )
    os.close(descriptor)
    local = Path(local_name)
    remote = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), local)
        with local.open("r+b") as handle:
            os.fsync(handle.fileno())
        with local.open("rb") as source, remote.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=16 * 1024 * 1024)
            sink.flush()
            os.fsync(sink.fileno())
        if target.exists() and not overwrite:
            raise AtomicCheckpointError(f"refusing to overwrite checkpoint: {target}")
        os.replace(remote, target)
        if os.name != 'nt':
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return target
    finally:
        local.unlink(missing_ok=True)
        remote.unlink(missing_ok=True)
