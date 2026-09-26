"""Backbone-independent contracts for DeltaRec Trajectory-Late runs.

The HSTU runner predates the two non-HSTU ports and keeps some of its
contract code local to the official selector implementation.  This module is
the small, dependency-free part shared by the FuXiLinear and Blossom ports.
It deliberately contains no model or data-loader code: those remain
backbone-specific seams.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TRAJECTORY_LATE_SCHEMA = "deltarec-trajectory-late-v2"
TRAJECTORY_LATE_CHECKPOINT_SCHEMA = "deltarec-full-gdr-trajectory-checkpoint-v2"
TRAJECTORY_LATE_CWI_MANIFEST_SCHEMA = "deltarec-trajectory-late-cwi-manifest-v2"
TRAJECTORY_LATE_SELECTOR_SCHEMA = "deltarec-trajectory-late-selector-v2"
TARGET_POLICY = "checkpoint-specific-observations-no-label-averaging"
SAMPLING_POLICY = "seeded-minibatch-teacher-sampling-late-biased"
FOUR_TEACHER_WEIGHTS = (0.1, 0.2, 0.3, 0.4)
_SELECTOR_LINEAGE_KEYS = (
    "schema",
    "protocol_schema",
    "dataset",
    "backbone",
    "seed",
    "teacher_checkpoints",
    "teacher_weights",
    "normalized_trajectory_weights",
    "target_policy",
    "sampling_policy",
    "final_theta_T_checkpoint",
    "final_theta_T_checkpoint_sha256",
    "frozen_embedding_snapshot_source",
    "frozen_embedding_snapshot_sha256",
    "trajectory_cwi_manifest",
    "legacy_selector_resume",
    "selector_checkpoint",
)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Mapping[str, Any], field: str = "content_sha256") -> str:
    clean = dict(value)
    clean.pop(field, None)
    return hashlib.sha256(canonical_json(clean)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash a state mapping without relying on pickle implementation details."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            raise TypeError(f"state value {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": int(stat.st_size),
    }


def verify_file_record(record: Mapping[str, Any], *, name: str = "artifact") -> Path:
    """Verify a recorded path/size/digest and return its resolved path."""

    if not isinstance(record, Mapping):
        raise ValueError(f"{name} file record is missing")
    source = Path(str(record.get("path", ""))).expanduser().resolve()
    if (
        not source.is_file()
        or record.get("sha256") != sha256_file(source)
        or int(record.get("size_bytes", -1)) != source.stat().st_size
    ):
        raise ValueError(f"{name} file record does not match its artifact")
    return source


def normalized_trajectory_weights(
    count: int, explicit: Sequence[float] | None = None
) -> list[float]:
    """Return positive, late-biased probabilities for an ordered trajectory.

    Four checkpoints use the registered decimal spelling.  For another
    explicitly recorded trajectory length the default remains the monotone
    ``1..T`` contract, which is the same rule used by HSTU for non-four-teacher
    runs.  Explicit weights always win and are never silently replaced.
    """

    if isinstance(count, bool) or count < 1:
        raise ValueError("a trajectory needs at least one checkpoint")
    if explicit is None:
        raw = list(FOUR_TEACHER_WEIGHTS) if count == 4 else list(range(1, count + 1))
    else:
        raw = [float(value) for value in explicit]
    if len(raw) != count or any(
        not math.isfinite(value) or value <= 0 for value in raw
    ):
        raise ValueError("trajectory weights must be finite, positive, and match teachers")
    total = float(sum(raw))
    return [float(value) / total for value in raw]


def trajectory_contract_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the scientific fields that make a dense checkpoint reusable."""

    required = (
        "dataset",
        "backbone",
        "seed",
        "variant",
        "architecture",
        "grouping_sha256",
        "group_count",
        "objective",
        "negatives",
        "temperature",
        "learning_rate",
        "effective_batch",
        "kernel",
        "protocol_sha256",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"trajectory config is missing contract fields: {missing}")
    return {field: config[field] for field in required}


def immutable_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create an immutable JSON artifact, accepting only byte-identical reuse."""

    target = Path(path).expanduser().resolve()
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"refusing to overwrite immutable artifact: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"stale immutable-artifact temporary exists: {temporary}")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_ordered_trajectory(
    paths: Sequence[str | Path],
    *,
    loader: Callable[[Path], Mapping[str, Any]],
    expected_contract: Mapping[str, Any],
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Verify contiguous, immutable checkpoint order and scientific identity."""

    if not paths:
        raise ValueError("trajectory must contain at least one checkpoint")
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(
            f"trajectory has {len(paths)} checkpoints; expected {expected_count}"
        )
    records: list[dict[str, Any]] = []
    previous_epoch = 0
    previous_step = -1
    for index, raw_path in enumerate(paths, 1):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = dict(loader(path))
        contract = payload.get("contract")
        if contract != dict(expected_contract):
            changed = sorted(
                key
                for key in set(expected_contract) | set(contract or {})
                if (contract or {}).get(key) != expected_contract.get(key)
            )
            raise ValueError(f"incompatible trajectory checkpoint {path}; changed={changed}")
        if payload.get("trajectory_index") != index:
            raise ValueError(f"trajectory index is not contiguous at {path}")
        epoch = int(payload.get("epoch", -1))
        step = int(payload.get("global_step", payload.get("step", -1)))
        if epoch <= previous_epoch or step < previous_step:
            raise ValueError(f"trajectory checkpoint order is not increasing at {path}")
        if index > 1 and payload.get("parent_checkpoint") != str(records[-1]["path"]):
            raise ValueError(f"trajectory parent does not point to the previous checkpoint: {path}")
        records.append(
            {
                "trajectory_index": index,
                "path": str(path),
                "sha256": sha256_file(path),
                "epoch": epoch,
                "global_step": step,
                "parent_checkpoint": payload.get("parent_checkpoint"),
                "payload": payload,
            }
        )
        previous_epoch = epoch
        previous_step = step
    return records


def require_trajectory_protocol(config: Mapping[str, Any]) -> None:
    if config.get("schema") != TRAJECTORY_LATE_SCHEMA:
        raise ValueError(
            "the requested stage is not a Trajectory-Late run; legacy selectors "
            "must be launched through their legacy entrypoint"
        )
    if config.get("target_policy") != TARGET_POLICY:
        raise ValueError("Trajectory-Late target policy is missing or changed")
    if config.get("trajectory_sampling") != "late":
        raise ValueError("Trajectory-Late sampling must be late-biased")


def full_gdr_stage_plan(
    *,
    saved: Mapping[str, Any] | None,
    completed: int,
    reused_count: int,
    continuation_parent: Any = None,
) -> dict[str, Any]:
    """Separate exact local resume from intentional model-only continuation.

    ``completed`` counts the immutable trajectory prefix already present on
    shared storage.  A node-local checkpoint at that same epoch may resume the
    current optimizer/RNG state; a reused prefix or explicit parent otherwise
    starts a fresh optimizer/schedule from the selected model state.
    """

    if completed < 0 or reused_count < 0 or completed < reused_count:
        raise ValueError("invalid completed/reused Full-GDR trajectory counts")
    saved_epoch = int(saved.get("epoch", 0)) if saved else 0
    saved_semantics = str(saved.get("resume_semantics", "")) if saved else ""
    resume_local = bool(
        saved
        and saved.get("stage") == "full_gdr"
        and saved_epoch == completed
        and (
            saved_semantics.startswith("true-resume")
            or saved_semantics.startswith("continuation-model-only")
        )
    )
    continuation = bool(
        not resume_local and (reused_count or continuation_parent or completed)
    )
    return {
        "saved_epoch": saved_epoch,
        "saved_semantics": saved_semantics,
        "resume_local": resume_local,
        "continuation": continuation,
    }


def selector_lineage(
    *,
    config: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    cwi_manifest: str | Path,
    final_snapshot_sha256: str,
    selector_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Build the common selector provenance record used by both ports."""

    if not trajectory:
        raise ValueError("selector lineage needs a nonempty trajectory")
    weights = normalized_trajectory_weights(
        len(trajectory), config.get("trajectory_weights")
    )
    value: dict[str, Any] = {
        "schema": TRAJECTORY_LATE_SELECTOR_SCHEMA,
        "protocol_schema": TRAJECTORY_LATE_SCHEMA,
        "dataset": config["dataset"],
        "backbone": config["backbone"],
        "seed": int(config["seed"]),
        "teacher_checkpoints": [
            {
                "trajectory_index": int(record["trajectory_index"]),
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
            for record in trajectory
        ],
        "teacher_weights": weights,
        "normalized_trajectory_weights": weights,
        "target_policy": TARGET_POLICY,
        "sampling_policy": SAMPLING_POLICY,
        "final_theta_T_checkpoint": str(trajectory[-1]["path"]),
        "final_theta_T_checkpoint_sha256": str(trajectory[-1]["sha256"]),
        "frozen_embedding_snapshot_source": str(trajectory[-1]["path"]),
        "frozen_embedding_snapshot_sha256": final_snapshot_sha256,
        "trajectory_cwi_manifest": file_record(cwi_manifest),
        "legacy_selector_resume": False,
    }
    if selector_checkpoint is not None:
        value["selector_checkpoint"] = file_record(selector_checkpoint)
    value["lineage_content_sha256"] = selector_lineage_content_sha256(value)
    return value


def selector_lineage_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash only the canonical lineage fields, excluding backend extensions."""

    payload = {
        key: value[key]
        for key in _SELECTOR_LINEAGE_KEYS
        if key in value
    }
    return content_sha256(payload, "lineage_content_sha256")


def verify_selector_lineage_artifacts(
    binding: Mapping[str, Any],
    *,
    trajectory_manifest_path: str | Path,
    final_checkpoint: str | Path,
    expected_count: int,
    expected_weights: Sequence[float],
) -> dict[str, Any]:
    """Verify selector lineage files before restoring a sparse model."""

    if binding.get("schema") != TRAJECTORY_LATE_SELECTOR_SCHEMA:
        raise ValueError("selector binding is not a Trajectory-Late lineage record")
    if binding.get("legacy_selector_resume") is not False:
        raise ValueError("legacy selector lineage cannot be resumed")
    if binding.get("lineage_content_sha256") != selector_lineage_content_sha256(binding):
        raise ValueError("selector lineage checksum changed")
    final_path = Path(final_checkpoint).expanduser().resolve()
    if (
        binding.get("final_theta_T_checkpoint") != str(final_path)
        or binding.get("final_theta_T_checkpoint_sha256") != sha256_file(final_path)
        or binding.get("frozen_embedding_snapshot_source") != str(final_path)
    ):
        raise ValueError("selector lineage is not bound to the final theta_T checkpoint")
    trajectory_path = Path(trajectory_manifest_path).expanduser().resolve()
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory.get("manifest_content_sha256") != content_sha256(
        trajectory, "manifest_content_sha256"
    ):
        raise ValueError("trajectory lineage checksum changed")
    checkpoints = trajectory.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != expected_count:
        raise ValueError("selector trajectory lineage has the wrong checkpoint count")
    for expected_index, record in enumerate(checkpoints, 1):
        if not isinstance(record, Mapping):
            raise ValueError("trajectory lineage checkpoint record is invalid")
        checkpoint_path = Path(str(record.get("path", ""))).expanduser().resolve()
        if (
            int(record.get("trajectory_index", -1)) != expected_index
            or not checkpoint_path.is_file()
            or record.get("sha256") != sha256_file(checkpoint_path)
        ):
            raise ValueError("trajectory lineage checkpoint digest or order changed")
    expected_teachers = [
        {
            "trajectory_index": int(record["trajectory_index"]),
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
        }
        for record in checkpoints
    ]
    if binding.get("teacher_checkpoints") != expected_teachers:
        raise ValueError("selector teacher checkpoint lineage changed")
    if expected_teachers[-1]["path"] != str(final_path):
        raise ValueError("selector final checkpoint is not the last trajectory teacher")
    weights = [float(value) for value in expected_weights]
    if binding.get("teacher_weights") != weights or binding.get(
        "normalized_trajectory_weights"
    ) != weights:
        raise ValueError("selector teacher weights changed")
    cwi_path = verify_file_record(
        binding.get("trajectory_cwi_manifest"), name="trajectory CWI manifest"
    )
    cwi = json.loads(cwi_path.read_text(encoding="utf-8"))
    if (
        cwi.get("manifest_content_sha256")
        != content_sha256(cwi, "manifest_content_sha256")
        or cwi.get("target_policy") != TARGET_POLICY
        or cwi.get("normalized_trajectory_weights") != weights
        or cwi.get("final_theta_T_checkpoint") != str(final_path)
        or cwi.get("final_theta_T_checkpoint_sha256") != sha256_file(final_path)
    ):
        raise ValueError("trajectory CWI lineage changed")
    teachers = cwi.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != expected_count:
        raise ValueError("trajectory CWI teacher manifest list is incomplete")
    for expected_index, (teacher, expected_teacher) in enumerate(
        zip(teachers, expected_teachers), 1
    ):
        if not isinstance(teacher, Mapping):
            raise ValueError("trajectory CWI teacher manifest is invalid")
        if (
            int(teacher.get("trajectory_index", -1)) != expected_index
            or teacher.get("target_policy") != TARGET_POLICY
        ):
            raise ValueError("trajectory CWI teacher identity changed")
        parent = teacher.get("parent_checkpoint")
        if not isinstance(parent, Mapping):
            raise ValueError("trajectory CWI teacher parent record is missing")
        if (
            parent.get("path") != expected_teacher["path"]
            or parent.get("sha256") != expected_teacher["sha256"]
            or sha256_file(parent["path"]) != parent.get("sha256")
        ):
            raise ValueError("trajectory CWI teacher parent digest changed")
        verify_file_record(teacher.get("shard"), name="trajectory CWI teacher shard")
    return cwi


__all__ = [
    "FOUR_TEACHER_WEIGHTS",
    "SAMPLING_POLICY",
    "TARGET_POLICY",
    "TRAJECTORY_LATE_CHECKPOINT_SCHEMA",
    "TRAJECTORY_LATE_CWI_MANIFEST_SCHEMA",
    "TRAJECTORY_LATE_SCHEMA",
    "TRAJECTORY_LATE_SELECTOR_SCHEMA",
    "canonical_json",
    "content_sha256",
    "file_record",
    "full_gdr_stage_plan",
    "immutable_json",
    "normalized_trajectory_weights",
    "require_trajectory_protocol",
    "selector_lineage",
    "selector_lineage_content_sha256",
    "sha256_file",
    "tensor_state_sha256",
    "trajectory_contract_fields",
    "verify_file_record",
    "verify_selector_lineage_artifacts",
    "verify_ordered_trajectory",
]
