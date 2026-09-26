"""Opt-in Kuai Task2 execution improvements, preserving the frozen 5C API."""
from __future__ import annotations

import types
import torch

REVISION = "kuai-frozen-selector-shared-first-projection-candidate-pooled-gc-v2"


def candidate_pooled_group_scores(scores, group_ids, prototypes, lengths):
    """Candidate-count normalized pooling; only absent groups use prototypes."""
    from deltarec.adaptors.kuai.research.modeling.sequential.deltarec.group_pooling import pool_group_scores
    groups = prototypes.shape[1]
    pooled = pool_group_scores(scores.float(), group_ids, group_count=groups,
        pool="logmeanexp", history_lengths=lengths)
    counts = torch.zeros(scores.shape[0], groups, dtype=torch.int64, device=scores.device)
    counts.scatter_add_(1, group_ids, torch.ones_like(group_ids))
    return torch.where(counts[:, :, None] > 0, pooled, prototypes.float())


def configure_candidate_pooled_gc(stack):
    """Scope the headline pooling correction to this train/eval instance.

    The old global-prefix serving caches are deliberately unavailable: their
    keys omit the slate, which now participates in group selection semantics.
    """
    if stack.config.grouping_policy != "fixed_category_prototype_v1" or stack.config.group_pool != "logmeanexp":
        raise ValueError("headline GC requires fixed categories and logmeanexp")
    original_forward = stack.forward_group_shared
    original_prototype_scores = stack._fixed_category_scores
    def pooled(_self, history_embeddings, history_lengths):
        candidate_ids, candidate_embeddings = _self._kuai_gc_request
        batch = history_embeddings.shape[0]
        ids = candidate_ids.reshape(batch, -1)
        embeddings = candidate_embeddings.reshape(batch, ids.shape[1], -1)
        scores, chunks = _self._selector_scores(history_embeddings, embeddings, history_lengths)
        fallback, fallback_chunks = original_prototype_scores(history_embeddings, history_lengths)
        groups = _self._fixed_category_group_ids(ids)
        return candidate_pooled_group_scores(scores, groups, fallback, history_lengths), chunks + fallback_chunks
    def forward(_self, **kwargs):
        if getattr(_self, "_kuai_gc_request", None) is not None:
            raise RuntimeError("concurrent GC forward is unsupported in this single-stream training entry")
        if kwargs.get("candidate_selector_embeddings") is None:
            raise ValueError("GC pooling needs exact shared candidate embeddings")
        _self._kuai_gc_request = (kwargs["candidate_item_ids"], kwargs["candidate_selector_embeddings"])
        try:
            return original_forward(**kwargs)
        finally:
            _self._kuai_gc_request = None
    def unsupported(_self, *args, **kwargs):
        raise RuntimeError("prototype-only serving caches are invalid for candidate-pooled headline GC")
    stack._fixed_category_scores = types.MethodType(pooled, stack)
    stack.forward_group_shared = types.MethodType(forward, stack)
    for name in ("cached_forward", "cached_state_forward", "global_cached_state_forward",
                 "materialize_selection_cache", "materialize_state_cache", "materialize_global_group_state_cache"):
        if hasattr(stack, name):
            setattr(stack, name, types.MethodType(unsupported, stack))


def configure_sparse_stack(stack, *, method):
    if method not in {"delta_pc", "delta_gc"}:
        raise ValueError("runtime improvements apply only to DeltaRec PC/GC")
    selector = stack.selector
    if any(parameter.requires_grad for parameter in selector.parameters()):
        raise ValueError("discrete ranker selection requires a frozen CWI MLP")
    if not getattr(selector, "requires_shared_embeddings", False):
        raise ValueError("Kuai requires the model's shared lookup tensors")
    if getattr(stack, "_kuai_runtime_revision", None) is not None:
        raise ValueError("Kuai runtime is already configured")
    score = selector.score_embeddings
    def frozen_scores(_self, *args, **kwargs):
        # Hard Top-B membership has no gradient. Avoid constructing the large
        # score graph through the otherwise trainable shared embedding table.
        with torch.no_grad():
            return score(*args, **kwargs)
    selector.score_embeddings = types.MethodType(frozen_scores, selector)
    if method == "delta_pc":
        stack.activate_subplan5b_frozen_backend({
            "executor": "project_once_then_gather",
            "projection_policy": "project_full_history_once",
            "candidate_chunk_size": 32, "gdr_backend": "fla", "recent_floor": 32,
            "projection_dtype": "bfloat16", "state_dtype": "float32",
            "retention_ratio": stack.retention_ratio})
    else:
        configure_candidate_pooled_gc(stack)
    stack._kuai_runtime_revision = REVISION
    return {"revision": REVISION, "frozen_selector_grad_enabled": False,
            "first_layer_projection": "project_once_then_gather" if method == "delta_pc" else "existing-packed-group-projection",
            "budget_formula": "max(ceil(rho*n),min(32,n))",
            "gc_pooling": "candidate-logmeanexp-empty-prototype" if method == "delta_gc" else None}
