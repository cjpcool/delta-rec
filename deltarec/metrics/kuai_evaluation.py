from __future__ import annotations

from dataclasses import asdict

from pathlib import Path

from typing import Any

import json

import uuid

METRIC_ID = "kuai-user-pooled-raw-logit-gauc-v1"

def evaluate_user_gauc(*, model: Any, dataloader: Any, metric_logger: Any,
                       device: Any, torch: Any, batch_observer: Any = None) -> dict:
    import numpy as np
    from torch.utils.data import SequentialSampler
    from deltarec.metrics.ranking import kuai_metrics
    from deltarec.data.kuai_loader import _find_multitask_module
    from deltarec.metrics.kuai_evidence import write_kuai_per_slate_evidence, write_kuai_per_user_evidence

    if not isinstance(dataloader.sampler, SequentialSampler) or dataloader.drop_last:
        raise ValueError("user-GAUC requires complete sequential validation slates")
    dataset = dataloader.dataset
    model.eval()
    captured = []
    hook = _find_multitask_module(model)._prediction_module.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach()))
    logits_parts, label_parts, weight_parts = [], [], []
    count = 0
    try:
        with torch.no_grad():
            for sample in dataloader:
                sample.to(device)
                captured.clear()
                outputs = model.forward(sample.uih_features_kjt, sample.candidates_features_kjt)
                if batch_observer is not None:
                    batch_observer(model=model, sample=sample)
                lengths = sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1)[0]
                if not bool((lengths == 32).all()) or len(captured) != 1:
                    raise ValueError("invalid K=32 slate batch or raw-logit hook count")
                n = len(lengths)
                raw = captured[0].float()
                predictions, labels, weights = outputs[3:]
                if tuple(raw.shape) != (n * 32, 8) or tuple(labels.shape) != (8, n * 32):
                    raise ValueError("unexpected Kuai task layout")
                if not torch.allclose(raw.sigmoid().T, predictions.float(), atol=2e-3, rtol=2e-3):
                    raise ValueError("raw logits do not reproduce official predictions")
                for value in (raw, labels, weights):
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("non-finite Kuai validation tensor")
                if bool(((labels != 0) & (labels != 1)).any()) or bool((weights < 0).any()):
                    raise ValueError("invalid Kuai labels or supervision weights")
                logits_parts.append(raw.reshape(n, 32, 8).cpu().numpy())
                label_parts.append(labels.T.reshape(n, 32, 8).float().cpu().numpy())
                weight_parts.append(weights.T.reshape(n, 32, 8).float().cpu().numpy())
                count += n
                if len(logits_parts) % 250 == 0:
                    print(f"user-GAUC validation: {count}/{len(dataset)} slates", flush=True)
    finally:
        hook.remove()
    if count != len(dataset) or not count:
        raise ValueError("validation did not cover every frozen slate exactly once")
    logits, labels, weights = [np.concatenate(parts) for parts in
                              (logits_parts, label_parts, weight_parts)]
    # Read identities from the same indexed dataset, never infer users from
    # batch position or collapse duplicate candidate exposures.
    fields = ("user_id", "slate_id", "candidate_item_ids", "candidate_timestamps",
              "candidate_play_time_ms", "candidate_duration_ms")
    rows = [{key: row[key] for key in fields} for row in
            (dataset._row(i) for i in range(count))]
    users = np.asarray([int(row["user_id"]) for row in rows], dtype=np.int64)
    aggregate = kuai_metrics(logits=logits, labels=labels, weights=weights, user_ids=users)
    root = Path(getattr(dataloader, "metric_evidence_root",
                        "outputs/kuai-evaluation")) / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    common = dict(logits=logits, labels=labels, weights=weights, user_ids=users,
                  dataset="kuairand-1k", split=getattr(dataloader, "metric_split_role", "validation"))
    per_user = write_kuai_per_user_evidence(**common, path=root / "per_user.jsonl")
    for key in ("task_gauc", "task_loss"):
        expected = getattr(aggregate, key)
        if set(per_user[key]) != set(expected) or any(
            not np.isclose(per_user[key][task], value, rtol=1e-12, atol=1e-12)
            for task, value in expected.items()
        ):
            raise ValueError("per-user evidence does not reconstruct aggregate metrics")
    candidate_fields = ("candidate_item_ids", "candidate_timestamps",
                        "candidate_play_time_ms", "candidate_duration_ms")
    candidates = {field: np.asarray([[int(value) for value in row[field].split(",")] for row in rows], dtype=np.int64)
                  for field in candidate_fields}
    write_kuai_per_slate_evidence(**common, **candidates,
        slate_ids=[row["slate_id"] for row in rows], path=root / "per_slate.jsonl")
    result = {**asdict(aggregate), "metric_id": METRIC_ID,
              "evaluation_batches": len(logits_parts), "candidate_examples": count * 32,
              "slates": count, "per_user_evidence": str(Path(root.name) / "per_user.jsonl"),
              "per_slate_evidence": str(Path(root.name) / "per_slate.jsonl"),
              "evidence_reconstruction_verified": True}
    (root / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"user-GAUC validation complete: {aggregate.macro_gauc:.10f}", flush=True)
    return result
