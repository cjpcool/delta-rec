from __future__ import annotations

import hashlib

import json

import os

import tempfile

from pathlib import Path

from typing import Any, Mapping

SCHEMA = "deltarec-headline-v1"

SEEDS = (0, 1, 2)

def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def protocol_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    # Self-hash fields are excluded so the same helper can validate protocol,
    # lock, run, and aggregate artifacts after their digest has been attached.
    for field_name in (
        "protocol_sha256",
        "lock_sha256",
        "run_sha256",
        "result_sha256",
        "audit_sha256",
        "manifest_sha256",
        "evidence_sha256",
    ):
        clean.pop(field_name, None)
    return hashlib.sha256(_canonical_json(clean)).hexdigest()

def validate_protocol_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema") != SCHEMA:
        raise RuntimeError("headline protocol lock has the wrong schema")
    if lock.get("state") != "budget-locked-test-enabled":
        raise RuntimeError("headline test evaluation is not unlocked")
    observed = str(lock.get("lock_sha256", ""))
    expected = protocol_hash(lock)
    if observed != expected:
        raise RuntimeError("headline protocol lock checksum mismatch")
    protocol = lock.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("headline protocol lock is missing protocol data")
    protocol_record = dict(protocol)
    protocol_digest = protocol_record.pop("protocol_sha256", None)
    if protocol_digest != protocol_hash(protocol_record):
        raise RuntimeError("embedded headline protocol checksum mismatch")

def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a canonical JSON artifact without exposing a partial manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
