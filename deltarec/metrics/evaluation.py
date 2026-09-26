from __future__ import annotations

import csv

import hashlib

import json

import os

from pathlib import Path

import tempfile

from typing import Any, Iterator, Mapping, Sequence

from deltarec.metrics.ranking import KUAI_TASKS, kuai_metrics, ranking_metrics

from deltarec.data.exposure import kuai_exposure_identity

from deltarec.utils.io import sha256_file

from deltarec.utils.protocols import ModelMode, RerankingBatch

from deltarec.adaptors.recbole import COMMON_MAX_HISTORY_LENGTH, RecBoleBridgeError

RATING_MANIFEST_SCHEMA = "deltarec-frozen-hstu-top100-manifest-v1"

RATING_SHARD_SCHEMA = "deltarec-frozen-hstu-top100-shard-v1"

POSITIVE_RATING_MANIFEST_SCHEMA = (
    "deltarec-positive-injected-hstu-top100-manifest-v1"
)

POSITIVE_RATING_SHARD_SCHEMA = "deltarec-positive-injected-hstu-top100-shard-v1"

KUAI_SPLIT_SCHEMA = "deltarec-headline-v1-frozen-splits-v1"

RATING_EVIDENCE_SCHEMA = "deltarec-recbole-ranking-request-evidence-v1"

KUAI_EVIDENCE_SCHEMA = "deltarec-recbole-kuai-slate-evidence-v1"

RATING_CANDIDATE_COUNT = 100

KUAI_CANDIDATE_COUNT = 32

KUAI_REQUIRED_COLUMNS = {
    "slate_id",
    "user_id",
    "history_item_ids",
    "history_timestamps",
    "candidate_item_ids",
    "candidate_timestamps",
    "candidate_play_time_ms",
    "candidate_duration_ms",
    *(f"label_{task}" for task in KUAI_TASKS),
}

class RecBoleEvaluationError(RecBoleBridgeError):
    """Raised when frozen evaluation evidence is inconsistent."""

def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )

class AtomicJsonlWriter:
    """Stream JSONL to a temporary file and expose it only after success."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._handle: Any | None = None
        self._temporary: str | None = None

    def __enter__(self) -> "AtomicJsonlWriter":
        if self.path.exists():
            raise RecBoleEvaluationError(
                f"refusing to overwrite per-unit evidence: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, self._temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JSONL writer is not open")
        self._handle.write(_canonical_json(row) + "\n")

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        del error, traceback
        assert self._handle is not None and self._temporary is not None
        try:
            if error_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()
            if error_type is None:
                os.replace(self._temporary, self.path)
            else:
                os.unlink(self._temporary)
        except FileNotFoundError:
            if error_type is None:
                raise
        finally:
            self._handle = None
            self._temporary = None

def _load_json_object(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecBoleEvaluationError(f"JSON root must be an object: {resolved}")
    return payload

def _as_int_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(entry) for entry in value]

def _validate_rating_manifest(
    path: Path, *, dataset: str, split_role: str
) -> tuple[Mapping[str, Any], Path, str]:
    manifest_path = path.expanduser().resolve()
    manifest = _load_json_object(manifest_path)
    positive_injection = manifest.get("positive_injection") is True
    expected = {
        "schema": (
            POSITIVE_RATING_MANIFEST_SCHEMA
            if positive_injection
            else RATING_MANIFEST_SCHEMA
        ),
        "dataset": dataset,
        "split_role": split_role,
        "method": "hstu",
        "top_k": RATING_CANDIDATE_COUNT,
        "positive_injection": positive_injection,
        "target_used_by_retriever": False,
        "tie_break": "score-desc-item-id-asc",
        "history_seen_item_mask": True,
    }
    if positive_injection:
        expected.update(
            {
                "target_used_by_candidate_construction": True,
                "injection_policy": (
                    "missing-target-replace-rank100-resort-by-hstu-score"
                ),
                "candidate_target_coverage": 1.0,
                "evaluation_scope": "reranking-only-not-end-to-end-retrieval",
            }
        )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RecBoleEvaluationError(
                f"candidate manifest requires {key}={value!r}; "
                f"observed {manifest.get(key)!r}"
            )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RecBoleEvaluationError("candidate manifest has no shards")
    declared_rows = int(manifest.get("rows", -1))
    if declared_rows <= 0 or sum(int(shard.get("rows", -1)) for shard in shards) != declared_rows:
        raise RecBoleEvaluationError("candidate manifest row count is inconsistent")
    names = [str(shard.get("filename", "")) for shard in shards]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise RecBoleEvaluationError("candidate shard filenames are invalid or duplicated")
    return manifest, manifest_path.parent, sha256_file(manifest_path)

def _load_rating_shard(
    torch: Any,
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    positive_injection: bool = False,
) -> Mapping[str, Any]:
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise RecBoleEvaluationError(f"candidate shard checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_schema = (
        POSITIVE_RATING_SHARD_SCHEMA if positive_injection else RATING_SHARD_SCHEMA
    )
    if not isinstance(payload, Mapping) or payload.get("schema") != expected_schema:
        raise RecBoleEvaluationError(f"unexpected candidate shard schema: {path}")
    required = {
        "user_ids",
        "history_item_ids",
        "history_lengths",
        "target_item_ids",
        "candidate_item_ids",
        "target_indices",
        "labels",
    }
    if positive_injection:
        required.update({"natural_retrieval_hit_at_100", "forced_injection"})
    missing = sorted(required - set(payload))
    if missing:
        raise RecBoleEvaluationError(f"candidate shard lacks fields: {missing}")
    rows = int(payload["user_ids"].shape[0])
    if rows != expected_rows:
        raise RecBoleEvaluationError(f"candidate shard row mismatch: {path}")
    shapes = {
        "history_item_ids": (rows, payload["history_item_ids"].shape[1]),
        "history_lengths": (rows,),
        "target_item_ids": (rows,),
        "candidate_item_ids": (rows, RATING_CANDIDATE_COUNT),
        "target_indices": (rows,),
        "labels": (rows, RATING_CANDIDATE_COUNT),
    }
    for name, expected in shapes.items():
        if tuple(payload[name].shape) != expected:
            raise RecBoleEvaluationError(
                f"candidate shard {name} shape {tuple(payload[name].shape)} != {expected}"
            )
    if "timestamps" in payload and tuple(payload["timestamps"].shape) != tuple(payload["history_item_ids"].shape):
        raise RecBoleEvaluationError("candidate shard timestamps do not match histories")
    if payload["history_item_ids"].shape[1] > COMMON_MAX_HISTORY_LENGTH:
        raise RecBoleEvaluationError("candidate history exceeds L=1024")
    return payload

def _validate_rating_rows(
    torch: Any, payload: Mapping[str, Any], *, positive_injection: bool = False
) -> None:
    histories = payload["history_item_ids"]
    lengths = payload["history_lengths"]
    candidates = payload["candidate_item_ids"]
    targets = payload["target_item_ids"]
    target_indices = payload["target_indices"]
    labels = payload["labels"].to(dtype=torch.bool)
    rows, width = candidates.shape
    for row in range(rows):
        length = int(lengths[row])
        if not 0 < length <= histories.shape[1]:
            raise RecBoleEvaluationError("rating history length is outside the tensor")
        active = _as_int_list(histories[row, :length])
        padding = histories[row, length:]
        candidate_row = _as_int_list(candidates[row])
        if any(item <= 0 for item in active) or bool((padding != 0).any()):
            raise RecBoleEvaluationError("rating histories must be positive then right-padded")
        if any(item <= 0 for item in candidate_row) or len(set(candidate_row)) != width:
            raise RecBoleEvaluationError("rating candidate row is not positive and unique")
        historical_candidates = set(active).intersection(candidate_row)
        if historical_candidates:
            allowed = {int(targets[row])} if positive_injection else set()
            if historical_candidates != allowed:
                raise RecBoleEvaluationError("frozen candidates contain a historical item")
        found = labels[row].nonzero(as_tuple=False).flatten().tolist()
        index = int(target_indices[row])
        expected_index = int(found[0]) if found else -1
        if len(found) > 1 or index != expected_index:
            raise RecBoleEvaluationError("target index/label metadata is inconsistent")
        target = int(targets[row])
        if (index >= 0) != (target in candidate_row):
            raise RecBoleEvaluationError("retrieval-hit metadata is inconsistent")
        if index >= 0 and candidate_row[index] != target:
            raise RecBoleEvaluationError("target index points to the wrong candidate")
        if positive_injection:
            natural = bool(payload["natural_retrieval_hit_at_100"][row])
            forced = bool(payload["forced_injection"][row])
            if natural == forced or index < 0:
                raise RecBoleEvaluationError(
                    "positive-injection flags or target coverage are inconsistent"
                )

def evaluate_rating_stream(
    *,
    torch: Any,
    adapter: Any,
    candidate_manifest: Path,
    dataset: str,
    split_role: str,
    protocol_lock_hash: str | None,
    microbatch_size: int,
    evidence: AtomicJsonlWriter,
) -> tuple[Mapping[str, float], Mapping[str, Any], str]:
    """Stream rating shards and aggregate only common ``ranking_metrics`` outputs."""

    manifest, root, candidate_sha = _validate_rating_manifest(
        candidate_manifest, dataset=dataset, split_role=split_role
    )
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    seen_users: set[int] = set()
    total = 0
    hr_sum = 0.0
    ndcg_sum = 0.0
    retrieval_sum = 0.0
    for shard_spec in manifest["shards"]:
        shard_name = str(shard_spec["filename"])
        payload = _load_rating_shard(
            torch,
            root / shard_name,
            expected_sha256=str(shard_spec.get("sha256", "")),
            expected_rows=int(shard_spec.get("rows", -1)),
            positive_injection=bool(manifest["positive_injection"]),
        )
        _validate_rating_rows(
            torch,
            payload,
            positive_injection=bool(manifest["positive_injection"]),
        )
        rows = int(payload["user_ids"].shape[0])
        for start in range(0, rows, microbatch_size):
            stop = min(rows, start + microbatch_size)
            users = payload["user_ids"][start:stop]
            for user in _as_int_list(users):
                if user in seen_users:
                    raise RecBoleEvaluationError(f"duplicate rating user_id {user}")
                seen_users.add(user)
            batch = RerankingBatch(
                user_ids=users,
                history_item_ids=payload["history_item_ids"][start:stop],
                history_lengths=payload["history_lengths"][start:stop],
                candidate_item_ids=payload["candidate_item_ids"][start:stop],
                target_indices=payload["target_indices"][start:stop],
                labels=payload["labels"][start:stop],
                timestamps=(None if "timestamps" not in payload else payload["timestamps"][start:stop]),
                metadata={
                    "history_order": "chronological",
                    "split_role": split_role,
                    "protocol_lock_hash": protocol_lock_hash,
                    "dataset": dataset,
                    "candidate_manifest_sha256": candidate_sha,
                },
            )
            output = adapter.run(batch, ModelMode.EVAL)
            scores = output.ranking_scores.detach().cpu()
            candidates = batch.candidate_item_ids
            targets = payload["target_item_ids"][start:stop]
            metrics = ranking_metrics(
                candidate_ids=candidates,
                scores=scores,
                target_ids=targets,
                k=10,
            )
            count = stop - start
            hr_sum += float(metrics.per_request_hr.sum())
            ndcg_sum += float(metrics.per_request_ndcg.sum())
            if manifest["positive_injection"]:
                natural_hits = payload["natural_retrieval_hit_at_100"][start:stop]
                forced = payload["forced_injection"][start:stop]
                retrieval_sum += float(natural_hits.sum())
            else:
                natural_hits = payload["target_indices"][start:stop] >= 0
                forced = torch.zeros_like(natural_hits, dtype=torch.bool)
                retrieval_sum += metrics.retrieval_recall * count
            for offset in range(count):
                evidence.write(
                    {
                        "schema": RATING_EVIDENCE_SCHEMA,
                        "dataset": dataset,
                        "split": split_role,
                        "unit_index": total + offset,
                        "candidate_shard": shard_name,
                        "shard_row": start + offset,
                        "user_id": int(users[offset]),
                        "target_item_id": int(targets[offset]),
                        "target_index_at_100": int(
                            payload["target_indices"][start + offset]
                        ),
                        "retrieval_hit_at_100": bool(natural_hits[offset]),
                        "forced_injection": bool(forced[offset]),
                        "hr_at_10": float(metrics.per_request_hr[offset]),
                        "ndcg_at_10": float(metrics.per_request_ndcg[offset]),
                    }
                )
            total += count
    if total != int(manifest["rows"]) or total == 0:
        raise RecBoleEvaluationError("evaluated rating row count differs from manifest")
    return (
        {"hr_at_10": hr_sum / total, "ndcg_at_10": ndcg_sum / total},
        {
            "requests": total,
            "retrieval_recall_at_100": retrieval_sum / total,
            "candidate_target_coverage": (
                float(manifest.get("candidate_target_coverage", 0.0))
                if manifest["positive_injection"]
                else retrieval_sum / total
            ),
            "forced_injection_requests": int(
                manifest.get("forced_injection_requests", 0)
            ),
            "evaluation_scope": manifest.get(
                "evaluation_scope", "end-to-end-frozen-retrieval-and-reranking"
            ),
            "metric_implementation": "deltarec.metrics.ranking.ranking_metrics",
        },
        candidate_sha,
    )

def _parse_ints(value: str, *, field: str) -> tuple[int, ...]:
    try:
        return tuple(int(entry) for entry in value.split(",") if entry)
    except ValueError as error:
        raise RecBoleEvaluationError(f"{field} contains a non-integer") from error

def _validate_split_manifest(
    split_manifest: Path, *, slate_file: Path, split_role: str
) -> str:
    manifest = _load_json_object(split_manifest)
    if manifest.get("schema") != KUAI_SPLIT_SCHEMA or manifest.get("dataset") != "kuairand-1k":
        raise RecBoleEvaluationError("unexpected Kuai split manifest")
    observed_content = str(manifest.get("manifest_content_sha256", ""))
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    expected_content = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    if observed_content != expected_content:
        raise RecBoleEvaluationError("Kuai split manifest content checksum mismatch")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping) or parameters.get("candidate_count") != KUAI_CANDIDATE_COUNT:
        raise RecBoleEvaluationError("Kuai split manifest does not freeze K=32")
    artifacts = manifest.get("artifacts")
    key = f"{split_role}_slates"
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(key), Mapping):
        raise RecBoleEvaluationError(f"Kuai split manifest lacks {key}")
    record = artifacts[key]
    resolved = slate_file.expanduser().resolve()
    if resolved.name != record.get("filename") or resolved.name != f"{split_role}_slates.csv":
        raise RecBoleEvaluationError("Kuai slate filename/split role mismatch")
    digest = sha256_file(resolved)
    if digest != record.get("sha256") or resolved.stat().st_size != int(record.get("size_bytes", -1)):
        raise RecBoleEvaluationError("Kuai slate artifact checksum/size mismatch")
    return digest

def _iter_kuai_slates(path: Path, *, split_role: str) -> Iterator[Mapping[str, Any]]:
    resolved = path.expanduser().resolve()
    seen_slates: set[str] = set()
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(KUAI_REQUIRED_COLUMNS - columns)
        if missing:
            raise RecBoleEvaluationError(f"Kuai slate columns are missing {missing}")
        for row_number, row in enumerate(reader, 2):
            slate_id = row["slate_id"]
            if not slate_id.startswith(f"{split_role}-") or slate_id in seen_slates:
                raise RecBoleEvaluationError(
                    f"invalid or duplicate Kuai slate_id at row {row_number}"
                )
            seen_slates.add(slate_id)
            history = _parse_ints(row["history_item_ids"], field="history_item_ids")
            history_times = _parse_ints(
                row["history_timestamps"], field="history_timestamps"
            )
            candidates = _parse_ints(
                row["candidate_item_ids"], field="candidate_item_ids"
            )
            candidate_times = _parse_ints(
                row["candidate_timestamps"], field="candidate_timestamps"
            )
            candidate_plays = _parse_ints(
                row["candidate_play_time_ms"], field="candidate_play_time_ms"
            )
            candidate_durations = _parse_ints(
                row["candidate_duration_ms"], field="candidate_duration_ms"
            )
            task_major = tuple(
                _parse_ints(row[f"label_{task}"], field=f"label_{task}")
                for task in KUAI_TASKS
            )
            if not 0 < len(history) <= COMMON_MAX_HISTORY_LENGTH:
                raise RecBoleEvaluationError("Kuai history length must lie in [1,1024]")
            if len(history_times) != len(history) or any(
                left > right for left, right in zip(history_times, history_times[1:])
            ):
                raise RecBoleEvaluationError("Kuai history timestamps are not aligned/sorted")
            if len(candidates) != KUAI_CANDIDATE_COUNT:
                raise RecBoleEvaluationError("Kuai candidates must be K=32 logged exposures")
            if (
                len(candidate_times) != KUAI_CANDIDATE_COUNT
                or len(candidate_plays) != KUAI_CANDIDATE_COUNT
                or len(candidate_durations) != KUAI_CANDIDATE_COUNT
                or any(
                    left > right
                    for left, right in zip(candidate_times, candidate_times[1:])
                )
                or max(history_times) >= min(candidate_times)
                or any(value < 0 for value in (*candidate_plays, *candidate_durations))
            ):
                raise RecBoleEvaluationError(
                    "Kuai candidate exposure fields must be aligned, chronological K=32"
                )
            if any(item <= 0 for item in (*history, *candidates)):
                raise RecBoleEvaluationError("Kuai mapped item IDs must be positive")
            if any(len(values) != KUAI_CANDIDATE_COUNT for values in task_major):
                raise RecBoleEvaluationError("Kuai task labels must each have K=32")
            if any(value not in (0, 1) for values in task_major for value in values):
                raise RecBoleEvaluationError("Kuai task labels must be binary")
            user_id = int(row["user_id"])
            if user_id < 0:
                raise RecBoleEvaluationError("Kuai user ID must be nonnegative")
            yield {
                "slate_id": slate_id,
                "user_id": user_id,
                "history": history,
                "candidates": candidates,
                "candidate_timestamps": candidate_times,
                "candidate_play_time_ms": candidate_plays,
                "candidate_duration_ms": candidate_durations,
                "labels": tuple(zip(*task_major)),
            }

def _kuai_batch(torch: Any, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    histories = torch.zeros(
        (len(rows), COMMON_MAX_HISTORY_LENGTH), dtype=torch.long
    )
    lengths = torch.empty(len(rows), dtype=torch.long)
    candidates = torch.empty((len(rows), KUAI_CANDIDATE_COUNT), dtype=torch.long)
    labels = torch.empty(
        (len(rows), KUAI_CANDIDATE_COUNT, len(KUAI_TASKS)), dtype=torch.float32
    )
    users = torch.empty(len(rows), dtype=torch.long)
    for index, row in enumerate(rows):
        history = row["history"]
        histories[index, : len(history)] = torch.as_tensor(history, dtype=torch.long)
        lengths[index] = len(history)
        candidates[index] = torch.as_tensor(row["candidates"], dtype=torch.long)
        labels[index] = torch.as_tensor(row["labels"], dtype=torch.float32)
        users[index] = row["user_id"]
    return {
        "histories": histories,
        "lengths": lengths,
        "candidates": candidates,
        "labels": labels,
        "users": users,
    }

def evaluate_kuai_stream(
    *,
    torch: Any,
    adapter: Any,
    slate_file: Path,
    split_manifest: Path,
    split_role: str,
    protocol_lock_hash: str | None,
    microbatch_size: int,
    evidence: AtomicJsonlWriter,
) -> tuple[Mapping[str, float], Mapping[str, Any], str]:
    """Stream K=32 slates, then call the common GAUC/loss implementation once."""

    candidate_sha = _validate_split_manifest(
        split_manifest, slate_file=slate_file, split_role=split_role
    )
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    logits_chunks: list[Any] = []
    labels_chunks: list[Any] = []
    user_chunks: list[Any] = []
    pending: list[Mapping[str, Any]] = []
    total = 0

    def consume(rows: Sequence[Mapping[str, Any]]) -> None:
        nonlocal total
        batch = _kuai_batch(torch, rows)
        reranking = RerankingBatch(
            user_ids=batch["users"],
            history_item_ids=batch["histories"],
            history_lengths=batch["lengths"],
            candidate_item_ids=batch["candidates"],
            task_labels={
                task: batch["labels"][..., index]
                for index, task in enumerate(KUAI_TASKS)
            },
            metadata={
                "history_order": "chronological",
                "split_role": split_role,
                "protocol_lock_hash": protocol_lock_hash,
                "dataset": "kuairand-1k",
                "candidate_sha256": candidate_sha,
            },
        )
        output = adapter.run(reranking, ModelMode.EVAL)
        logits = output.ranking_scores.detach().cpu().to(dtype=torch.float32)
        if tuple(logits.shape) != tuple(batch["labels"].shape):
            raise RecBoleEvaluationError(
                f"Kuai adapter score shape {tuple(logits.shape)} != "
                f"{tuple(batch['labels'].shape)}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise RecBoleEvaluationError("Kuai adapter produced non-finite logits")
        logits_chunks.append(logits)
        labels_chunks.append(batch["labels"])
        user_chunks.append(batch["users"])
        for offset, row in enumerate(rows):
            identity = kuai_exposure_identity(
                slate_id=str(row["slate_id"]),
                user_id=int(row["user_id"]),
                candidate_item_ids=row["candidates"],
                candidate_timestamps=row["candidate_timestamps"],
                candidate_play_time_ms=row["candidate_play_time_ms"],
                candidate_duration_ms=row["candidate_duration_ms"],
                labels=row["labels"],
            )
            evidence.write(
                {
                    "schema": KUAI_EVIDENCE_SCHEMA,
                    "dataset": "kuairand-1k",
                    "split": split_role,
                    "unit_index": total + offset,
                    "slate_id": row["slate_id"],
                    "user_id": row["user_id"],
                    "candidate_item_ids": list(row["candidates"]),
                    **identity,
                    "task_order": list(KUAI_TASKS),
                    "logits": logits[offset].tolist(),
                    "labels": batch["labels"][offset].to(dtype=torch.int8).tolist(),
                }
            )
        total += len(rows)

    for row in _iter_kuai_slates(slate_file, split_role=split_role):
        pending.append(row)
        if len(pending) == microbatch_size:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    if total == 0:
        raise RecBoleEvaluationError("Kuai evaluation contains no slates")
    metrics = kuai_metrics(
        logits=torch.cat(logits_chunks),
        labels=torch.cat(labels_chunks),
        user_ids=torch.cat(user_chunks),
    )
    return (
        {
            "macro_gauc": metrics.macro_gauc,
            "multitask_loss": metrics.multitask_loss,
        },
        {
            "slates": total,
            "candidate_examples": total * KUAI_CANDIDATE_COUNT,
            "task_gauc": dict(metrics.task_gauc),
            "task_loss": dict(metrics.task_loss),
            "task_prevalence": dict(metrics.task_prevalence),
            "eligible_users": dict(metrics.eligible_users),
            "eligible_examples": dict(metrics.eligible_examples),
            "metric_implementation": "deltarec.metrics.ranking.kuai_metrics",
        },
        candidate_sha,
    )
