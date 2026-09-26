"""Opt-in paper implementation: offline utility groups, online shared queries.

This module does not change any registered training or frozen-artifact default.
Preparation is exact-slate specific and runs on CPU. Only group assignments and
chronological event indices are cached; recurrent states are built on every
request. The official rating host supplies its own GDR layers and scoring head.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import torch
from deltarec.layers.recent_selection import recent_topk_indices


SCHEMA = "deltarec-utility-groups-v1"


@dataclass(frozen=True)
class UtilityGroupingConfig:
    groups: int = 4
    retention_ratio: float = 0.25
    minimum_count: int = 32
    lloyd_iterations: int = 8
    selector_transform: str = "asinh"

    def __post_init__(self):
        for name in ("groups", "minimum_count", "lloyd_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0,1]")
        if self.selector_transform not in {"asinh", "identity"}:
            raise ValueError("selector_transform must be asinh or identity")


@dataclass(frozen=True)
class UtilityGroupPlan:
    candidate_group_ids: torch.Tensor  # CPU int64 [K], exposure positions retained
    selected_indices: torch.Tensor  # CPU int64 [G,B], chronological original positions
    history_length: int
    group_utilities: torch.Tensor | None = None  # offline diagnostics; never cached
    binding: dict[str, Any] | None = None
    content_sha256: str | None = None  # execution rejects mutation after preparation

    @property
    def group_count(self) -> int:
        return int(self.selected_indices.shape[0])

    @property
    def budget(self) -> int:
        return int(self.selected_indices.shape[1])


def _budget(length: int, config: UtilityGroupingConfig) -> int:
    # Match the official rating protocol, including its FP32 rounding boundary.
    count = int(torch.ceil(torch.tensor(length, dtype=torch.float32)
                           * torch.tensor(config.retention_ratio, dtype=torch.float32)))
    return max(min(length, max(32, config.minimum_count)), count)


@torch.no_grad()
def prepare_utility_groups(utility_scores: torch.Tensor,
                           config: UtilityGroupingConfig) -> UtilityGroupPlan:
    """Partition raw predicted [K,L] utility profiles by squared Euclidean distance.

    Deterministic farthest-first seeds and at most eight Lloyd iterations are
    used by default. Ties prefer the lower candidate/exposure position. Empty
    clusters take the largest-residual member of a cluster with >1 members.
    All G groups are nonempty, including duplicate profiles; require G <= K.
    Arithmetic means preserve each event's member-utility range. Stable Top-B
    ties prefer earlier original history positions. The latest min(32,L) events are mandatory inside the budget.
    """
    if utility_scores.ndim != 2 or not torch.is_floating_point(utility_scores):
        raise ValueError("utilities must be floating point [K,L]")
    profiles = utility_scores.detach().to(device="cpu", dtype=torch.float64).contiguous()
    candidates, length = profiles.shape
    if length < 1 or candidates < config.groups or not bool(torch.isfinite(profiles).all()):
        raise ValueError("require finite nonempty profiles and 1 <= G <= K")
    groups = config.groups
    if groups == candidates:
        assignments = torch.arange(candidates)
    elif groups == 1:
        assignments = torch.zeros(candidates, dtype=torch.long)
    else:
        # Explicit reductions avoid implementation-dependent cdist/GEMM switches.
        first = int(((profiles - profiles.mean(0)).square().sum(1)).argmax())
        seeds = [first]
        nearest = (profiles - profiles[first]).square().sum(1)
        while len(seeds) < groups:
            eligible = nearest.clone()
            eligible[seeds] = -1
            chosen = int(eligible.argmax())
            seeds.append(chosen)
            nearest = torch.minimum(nearest, (profiles - profiles[chosen]).square().sum(1))
        centers = profiles[seeds].clone()
        previous = None
        for _ in range(config.lloyd_iterations):
            distances = torch.stack([(profiles - center).square().sum(1) for center in centers], 1)
            assignments = distances.argmin(1)
            counts = torch.bincount(assignments, minlength=groups)
            residual = distances[torch.arange(candidates), assignments]
            for empty in torch.where(counts == 0)[0].tolist():
                eligible = residual.masked_fill(counts[assignments] <= 1, -1)
                moved = int(eligible.argmax())
                counts[assignments[moved]] -= 1
                assignments[moved] = empty
                counts[empty] += 1
            centers = torch.stack([profiles[assignments == group].mean(0) for group in range(groups)])
            if previous is not None and torch.equal(previous, assignments):
                break
            previous = assignments.clone()
    pooled = torch.stack([profiles[assignments == group].mean(0) for group in range(groups)])
    budget = _budget(length, config)
    selected = recent_topk_indices(pooled, budget)
    return UtilityGroupPlan(assignments, selected, length, pooled)


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _ids(value: torch.Tensor, name: str) -> list[int]:
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64) or value.numel() < 1:
        raise ValueError(f"{name} must be nonempty integer [N]")
    result = value.detach().cpu().tolist()
    if min(result) < 1:
        raise ValueError(f"{name} must contain valid nonpadding IDs")
    return result


def selection_binding(history_ids: torch.Tensor, candidate_ids: torch.Tensor, *,
                      model_revision: str, selector_revision: str,
                      config: UtilityGroupingConfig, feature_revision: str = "id-original-position-v1") -> dict:
    """Bind immutable model/selector SHA256s, complete history and exact slate.

    The slate includes order and repeated exposure IDs. An unseen/reordered slate
    is a miss. Hosts with extra event features must supply their content revision.
    Offline serving requires this exact slate to be available in advance; this
    implementation does not claim arbitrary-slate reuse or incremental updates.
    """
    for name, value in (("model_revision", model_revision), ("selector_revision", selector_revision)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be a lowercase SHA256 content identity")
    if not isinstance(feature_revision, str) or not feature_revision:
        raise ValueError("feature_revision must be nonempty")
    histories, candidates = _ids(history_ids, "history_ids"), _ids(candidate_ids, "candidate_ids")
    return dict(schema=SCHEMA, model_revision=model_revision, selector_revision=selector_revision,
                history_sha256=_digest(histories), history_length=len(histories),
                candidate_slate_sha256=_digest(candidates), candidate_count=len(candidates),
                feature_revision=feature_revision, config=asdict(config))


class SelectedIndexCache:
    """Bounded memory cache plus optional atomic JSON files; no model tensors."""

    def __init__(self, directory: Path | str | None = None, *, max_entries: int = 128):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.directory = None if directory is None else Path(directory)
        self.max_entries = max_entries
        self._memory: OrderedDict[str, dict] = OrderedDict()

    def _remember(self, key: str, payload: dict):
        self._memory[key] = payload
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    @staticmethod
    def _decode(payload: dict, binding: dict) -> UtilityGroupPlan:
        unsigned = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload.get("content_sha256") != _digest(unsigned) or payload.get("binding") != binding:
            raise ValueError("selected-index cache checksum or binding mismatch")
        assignments = torch.tensor(payload["candidate_group_ids"], dtype=torch.long)
        indices = torch.tensor(payload["selected_indices"], dtype=torch.long)
        config = UtilityGroupingConfig(**binding["config"])
        length = binding["history_length"]
        if (assignments.shape != (binding["candidate_count"],)
                or indices.shape != (config.groups, _budget(length, config))
                or bool((assignments < 0).any()) or bool((assignments >= config.groups).any())
                or bool((torch.bincount(assignments, minlength=config.groups) == 0).any())
                or bool((indices < 0).any()) or bool((indices >= length).any())
                or bool((indices.diff(dim=1) <= 0).any())):
            raise ValueError("invalid cached group partition or chronological budget")
        return UtilityGroupPlan(assignments, indices, length, binding=binding,
                                content_sha256=payload["content_sha256"])

    def get(self, binding: dict) -> UtilityGroupPlan | None:
        key = _digest(binding)
        payload = self._memory.get(key)
        if payload is None:
            path = None if self.directory is None else self.directory / f"{key}.json"
            if path is None or not path.is_file():
                return None
            payload = json.loads(path.read_text())
        plan = self._decode(payload, binding)
        self._remember(key, payload)
        return plan

    def put(self, binding: dict, plan: UtilityGroupPlan) -> UtilityGroupPlan:
        unsigned = dict(binding=binding, candidate_group_ids=plan.candidate_group_ids.tolist(),
                        selected_indices=plan.selected_indices.tolist())
        payload = dict(unsigned, content_sha256=_digest(unsigned))
        validated = self._decode(payload, binding)
        key = _digest(binding)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, prefix=".indices-", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(payload, stream, sort_keys=True, allow_nan=False)
                    stream.flush()
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                temporary.replace(self.directory / f"{key}.json")
            finally:
                temporary.unlink(missing_ok=True)
        self._remember(key, payload)
        return validated


@dataclass(frozen=True)
class UtilityScoreResult:
    scores: torch.Tensor
    cache_hit: bool
    lookup_ms: float
    prepare_ms: float
    execution_ms: float
    groups: int
    retained_writes: int


class UtilityGroupedRatingScorer:
    """Actual official rating GDR integration, one request per call, eval only.

    ``prepare`` calls the trained selector offline. ``score`` resolves the exact
    cached slate or rebuilds it on a miss, reporting rebuild time separately.
    ``execute`` measures/reuses an already prepared index plan without selector
    work. All history indices keep original positional embeddings. Each group's
    final retained history hidden is the host readout R(S_g), then the official
    similarity head scores each candidate with its corresponding group query.
    Candidates never become recurrent tokens. Deep layers recompute projections
    on the selected history; fixed-transition deletion equivalence is only local.

    Revision arguments must identify the supplied immutable model and selector
    contents, not filenames. Construct a new adapter after changing either one.
    This adapter supports the pinned ID + original-position rating host only;
    timestamps, user context and additional dynamic event features are unsupported.
    Content-addressed histories can therefore be shared across identical ID rows.
    Candidates are sorted by ID before grouping and restored to slate order after
    preparation. Duplicate IDs retain their exposure order (the rating selector
    gives duplicate IDs identical profiles); the exact-slate cache preserves all
    duplicate exposure positions and still misses on a reordered slate.
    The official selector learns asinh(CWI); its outputs are converted back with
    sinh before raw-utility clustering and aggregation. Use selector_transform=
    'identity' for a selector already predicting raw utilities. Explicit supplied
    utility_scores are always raw utility, so no inverse transform is applied.
    These overrides create separately bound diagnostic plans: only explicit
    execute(plan, ...) uses them; normal score(...) cannot hit their cache keys.
    """

    def __init__(self, scorer, selector, *, model_revision: str, selector_revision: str,
                 config: UtilityGroupingConfig | None = None,
                 cache: SelectedIndexCache | None = None):
        from deltarec.models.hstu_shared import SharedSparseScoring

        class GroupEncoder(SharedSparseScoring):
            def _score_head(self, query, candidates):
                return query

        self.scorer, self.selector = scorer, selector
        self.config = config or UtilityGroupingConfig()
        self.cache = cache or SelectedIndexCache()
        self.model_revision, self.selector_revision = model_revision, selector_revision
        # Reuse only the existing direct packed execution. The proxy's method is
        # never consulted for selection: utility-derived indices enter directly.
        proxy = SimpleNamespace(model=scorer.model, model_config=scorer.model_config,
                                method="random", retention_ratio=self.config.retention_ratio)
        self.encoder = GroupEncoder(proxy)
        self._versions = self._version_signature()

    def _version_signature(self):
        # The official selector holds this embedding through a weak reference,
        # so selector.parameters() alone cannot establish or track its identity.
        embedding_module = getattr(self.scorer.model, "_embedding_module", None)
        host_embedding = getattr(embedding_module, "_item_emb", None)
        selector_embedding = getattr(self.selector, "official_embedding", None)
        if (not isinstance(host_embedding, torch.nn.Embedding)
                or selector_embedding is not host_embedding):
            raise ValueError("selector must share the official model _item_emb instance")
        modules = [self.scorer.model]
        if isinstance(self.selector, torch.nn.Module):
            modules.append(self.selector)
        registered = tuple((id(tensor), tensor._version) for module in modules
                           for tensor in (*module.parameters(), *module.buffers()))
        return registered + ((id(selector_embedding.weight), selector_embedding.weight._version),)

    def _validate_frozen(self):
        if self.scorer.model.training or (isinstance(self.selector, torch.nn.Module) and self.selector.training):
            raise ValueError("utility cache serving requires model and selector eval mode")
        if self._versions != self._version_signature():
            raise ValueError("model/selector changed; construct an adapter with new content revisions")

    def _binding(self, history_ids, candidate_ids):
        return selection_binding(history_ids, candidate_ids, model_revision=self.model_revision,
                                 selector_revision=self.selector_revision, config=self.config) | {
                                     "utility_source": "selector-prediction"}

    @torch.inference_mode()
    def prepare(self, history_ids: torch.Tensor, candidate_ids: torch.Tensor,
                utility_scores: torch.Tensor | None = None) -> UtilityGroupPlan:
        self._validate_frozen()
        binding = self._binding(history_ids, candidate_ids)
        if (utility_scores is None and self.config.groups == 1
                and self.config.retention_ratio == 1.0):
            # Full-GDR has a known single full history and needs no selector.
            plan = UtilityGroupPlan(torch.zeros(candidate_ids.numel(), dtype=torch.long),
                                    torch.arange(history_ids.numel())[None], history_ids.numel())
            return self.cache.put(binding, plan)
        diagnostic = utility_scores is not None
        if utility_scores is None:
            device = next(self.scorer.model.parameters()).device
            utility_scores = self.selector.score_ids(
                history_ids.to(device)[None], candidate_ids.to(device)[None],
                torch.tensor([history_ids.numel()], device=device))[0]
            if self.config.selector_transform == "asinh":
                utility_scores = torch.sinh(utility_scores.float())
            if not bool(torch.isfinite(utility_scores).all()):
                raise ValueError("selector inverse transform produced non-finite utilities")
        if utility_scores.shape != (candidate_ids.numel(), history_ids.numel()):
            raise ValueError("selector output must match exact candidate slate and history")
        if diagnostic:
            # Bind the actual raw utility values supplied to the grouping
            # algorithm, independently of the trained selector's identity.
            raw = utility_scores.detach().to(device="cpu", dtype=torch.float64).contiguous()
            utility_sha256 = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
            binding = binding | {"utility_source": "external-diagnostic",
                                 "utility_sha256": utility_sha256}
        canonical = candidate_ids.detach().cpu().argsort(stable=True)
        grouped = prepare_utility_groups(utility_scores.detach().cpu()[canonical], self.config)
        assignments = torch.empty_like(grouped.candidate_group_ids)
        assignments[canonical] = grouped.candidate_group_ids
        plan = UtilityGroupPlan(assignments, grouped.selected_indices,
                                grouped.history_length, grouped.group_utilities)
        return self.cache.put(binding, plan)

    @torch.inference_mode()
    def execute(self, plan: UtilityGroupPlan, history_ids: torch.Tensor,
                candidate_ids: torch.Tensor) -> torch.Tensor:
        from deltarec.models.hstu_packed import BatchedSparseScoring
        self._validate_frozen()
        expected = self._binding(history_ids, candidate_ids)
        observed = None if plan.binding is None else dict(plan.binding)
        if observed is not None and observed.get("utility_source") == "external-diagnostic":
            digest = observed.pop("utility_sha256", None)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("diagnostic plan has invalid utility content binding")
            observed["utility_source"] = "selector-prediction"
        if observed != expected:
            raise ValueError("prepared selection does not match model/history/selector/exact slate")
        # frozen dataclasses do not freeze tensor contents. Verify the immutable
        # checksum and structural budget before indexing any device tensors.
        plan = SelectedIndexCache._decode(dict(binding=plan.binding,
            candidate_group_ids=plan.candidate_group_ids.tolist(),
            selected_indices=plan.selected_indices.tolist(), content_sha256=plan.content_sha256),
            plan.binding)
        device = next(self.scorer.model.parameters()).device
        groups, budget = plan.selected_indices.shape
        streams = torch.arange(groups, device=device).repeat_interleave(budget)
        positions = plan.selected_indices.flatten().to(device)
        offsets = torch.arange(groups + 1, dtype=torch.long) * budget
        queries = self.encoder.packed_scores(
            history_ids.to(device)[None], torch.zeros(groups, dtype=torch.long),
            torch.full((groups,), plan.history_length, dtype=torch.long), streams,
            positions, offsets, None)
        # Reuse the unmodified host head, with K queries selected from only G
        # recurrent histories. There are exactly K scalar scores, not G*K scores.
        return BatchedSparseScoring._score_head(
            self.encoder, queries[plan.candidate_group_ids.to(device)],
            candidate_ids.to(device)[:, None]).reshape(-1)

    def _synchronize(self):
        device = next(self.scorer.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def score(self, history_ids: torch.Tensor, candidate_ids: torch.Tensor) -> UtilityScoreResult:
        self._validate_frozen()
        start = time.perf_counter()
        plan = self.cache.get(self._binding(history_ids, candidate_ids))
        lookup_ms = 1000 * (time.perf_counter() - start)
        hit = plan is not None
        prepare_ms = 0.0
        if plan is None:
            self._synchronize()
            start = time.perf_counter()
            plan = self.prepare(history_ids, candidate_ids)
            self._synchronize()
            prepare_ms = 1000 * (time.perf_counter() - start)
        self._synchronize()
        start = time.perf_counter()
        scores = self.execute(plan, history_ids, candidate_ids)
        self._synchronize()
        return UtilityScoreResult(scores, hit, lookup_ms, prepare_ms,
                                  1000 * (time.perf_counter() - start), plan.group_count,
                                  plan.group_count * plan.budget)
