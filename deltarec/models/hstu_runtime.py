from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping
import torch
import torch.nn.functional as F
from torch import nn
from deltarec.adaptors.hstu_model import MetaBridgeError
from types import SimpleNamespace as OfficialResearchGDRLoad
from deltarec.layers.hstu_gdr import ResearchHSTUGDRLayer, _research_layers
from deltarec.models.hstu_selector import RatingGCSelector, RatingPCSelector, exact_budget_write_mask
SPARSE_RATING_METHODS = frozenset(('full-gdr', 'deltarec-pc', 'deltarec-gc'))
CANDIDATE_STATE_METHODS = frozenset(('deltarec-pc',))
RUNTIME_SCHEMA = 'deltarec-meta-research-sparse-runtime-v1'

def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f'{tuple(tensor.shape)}:{tensor.dtype}'.encode('ascii'))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def _stable_budget_mask(scores: torch.Tensor, lengths: torch.Tensor, *, retention_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable exact Top-B for score tensors ``[B,S,L]``."""
    if scores.ndim != 3 or lengths.shape != (scores.shape[0],):
        raise ValueError('selection scores/lengths must be [B,S,L]/[B]')
    selected = exact_budget_write_mask(scores, lengths, retention_ratio=retention_ratio, recent_floor=32)
    return (selected.write_mask, selected.budgets[:, 0])

@dataclass(frozen=True)
class ResearchGDRStateCache:
    schema: str
    method: str
    state: torch.Tensor
    history_lengths: torch.Tensor
    stream_count: int
    candidate_key: torch.Tensor | None
    candidate_group_ids: torch.Tensor | None
    selected_counts: torch.Tensor
    realized_write_ratio: float
    model_binding_sha256: str
    selector_binding_sha256: str | None
    grouping_binding_sha256: str | None
    retention_ratio: float = 1.0
    recent_floor: int = 32

    def validate(self) -> None:
        if self.schema != RUNTIME_SCHEMA or self.method not in SPARSE_RATING_METHODS:
            raise MetaBridgeError('invalid research GDR state-cache schema/method')
        if self.state.ndim != 6 or self.state.dtype != torch.float32:
            raise MetaBridgeError('research GDR cache state must be FP32 [B,S,layers,H,Dk,Dv]')
        if self.state.shape[1] != self.stream_count:
            raise MetaBridgeError('research GDR cache stream count mismatch')
        if self.history_lengths.shape != (self.state.shape[0],):
            raise MetaBridgeError('research GDR cache history lengths mismatch')
        if self.selected_counts.shape != self.state.shape[:2]:
            raise MetaBridgeError('research GDR cache selected counts mismatch')
        if self.method in CANDIDATE_STATE_METHODS and self.candidate_key is None:
            raise MetaBridgeError('candidate-specific GDR cache lacks candidate key')
        if self.method == 'deltarec-gc' and self.candidate_group_ids is None:
            raise MetaBridgeError('GC cache lacks candidate-to-global-group key')

@dataclass(frozen=True)
class PreparedResearchHistory:
    """Non-differentiable selection reused by activation recomputation.

    This is ephemeral per-forward data, never a persisted model parameter,
    buffer, or a cross-optimizer-step state cache.
    """
    batch_indices: torch.Tensor
    positions: torch.Tensor
    offsets: torch.Tensor
    offsets_cpu: torch.Tensor
    selected_counts: torch.Tensor
    candidate_groups: torch.Tensor | None
    selected_writes: int
    eligible_writes: int

class OfficialResearchSparseScorer(nn.Module):
    """Candidate scorer and cache builder on one official research HSTU."""

    def __init__(self, loaded: OfficialResearchGDRLoad, *, method: str, retention_ratio: float=1.0, recent_floor: int=32, selector: nn.Module | None=None, selector_binding_sha256: str | None=None, grouping_binding_sha256: str | None=None) -> None:
        super().__init__()
        if method not in SPARSE_RATING_METHODS:
            raise ValueError(f'unsupported official rating sparse method {method!r}')
        if method == 'full-gdr':
            if retention_ratio != 1.0:
                raise ValueError('Full GDR must retain every history event')
        elif retention_ratio not in (0.25, 0.5):
            raise ValueError('sparse GDR method requires retention ratio 0.25/0.50')
        if recent_floor != 32:
            raise ValueError('headline sparse recent floor is fixed to 32')
        layers = _research_layers(loaded.model)
        if not all((isinstance(layer, ResearchHSTUGDRLayer) for layer in layers)):
            raise MetaBridgeError('official research GDR replacement is not installed')
        if not all((layer.mode == 'gdr' for layer in layers)):
            raise MetaBridgeError('official sparse runtime requires GDR mode')
        preprocessor = loaded.model._input_features_preproc
        if type(preprocessor).__qualname__ != 'LearnablePositionalEmbeddingInputFeaturesPreprocessor':
            raise MetaBridgeError('rating sparse runtime requires official positional preprocessor')
        self.model = loaded.model
        self.model_config = loaded.model_config
        self.method = method
        self.retention_ratio = float(retention_ratio)
        self.recent_floor = recent_floor
        self.selector = selector
        self.selector_binding_sha256 = selector_binding_sha256
        self.grouping_binding_sha256 = grouping_binding_sha256
        model_binding = loaded.evidence.get('binding')
        if not isinstance(model_binding, Mapping):
            raise MetaBridgeError('loaded official model lacks warm-start binding evidence')
        self.model_binding_sha256 = str(model_binding['binding_file_sha256'])
        item_embedding = self.model._embedding_module._item_emb
        if method == 'deltarec-pc':
            if not isinstance(selector, RatingPCSelector) or selector.official_embedding is not item_embedding or (not selector_binding_sha256):
                raise MetaBridgeError('DeltaRec-PC requires a bound selector on the exact winner embedding')
        elif method == 'deltarec-gc':
            if not isinstance(selector, RatingGCSelector) or selector.pc_selector.official_embedding is not item_embedding or (not selector_binding_sha256):
                raise MetaBridgeError('DeltaRec-GC requires the bound same-seed PC selector')
            if not grouping_binding_sha256:
                raise MetaBridgeError('DeltaRec-GC requires bound grouping evidence')
            if selector.grouping_evidence.get('binding_sha256') != grouping_binding_sha256:
                raise MetaBridgeError('DeltaRec-GC grouping evidence SHA mismatch')
            if selector.item_to_category_group.shape != (item_embedding.num_embeddings,):
                raise MetaBridgeError('GC item-to-group mapping is not exact item space')
            if selector.category_group_prototypes.shape[1] != item_embedding.embedding_dim or selector.category_group_prototypes.dtype != torch.float32:
                raise MetaBridgeError('GC prototypes are not FP32 [G,D] in winner space')
        elif selector is not None:
            raise MetaBridgeError(f'{method} cannot consume a candidate selector')

    @property
    def item_embedding(self) -> nn.Embedding:
        return self.model._embedding_module._item_emb

    def _validate_batch(self, history_item_ids: torch.Tensor, history_lengths: torch.Tensor, candidate_item_ids: torch.Tensor) -> None:
        if history_item_ids.ndim != 2 or history_lengths.shape != (history_item_ids.shape[0],):
            raise ValueError('official sparse scorer expects history [B,L]/lengths [B]')
        if candidate_item_ids.ndim != 2 or candidate_item_ids.shape[0] != history_item_ids.shape[0]:
            raise ValueError('official sparse scorer expects candidate IDs [B,K]')
        if bool((history_lengths < 1).any()) or bool((history_lengths > history_item_ids.shape[1]).any()):
            raise ValueError('history lengths must lie within the dense history width')
        positions = torch.arange(history_item_ids.shape[1], device=history_item_ids.device)[None]
        if bool(((positions < history_lengths[:, None]) & history_item_ids.eq(0)).any()):
            raise ValueError('valid history prefix contains padding ID 0')
        if bool(((positions >= history_lengths[:, None]) & history_item_ids.ne(0)).any()):
            raise ValueError('official sparse history must be right-zero padded')
        maximum = self.item_embedding.num_embeddings - 1
        if candidate_item_ids.numel() and (int(candidate_item_ids.min()) < 1 or int(candidate_item_ids.max()) > maximum):
            raise ValueError('candidate ID is outside the bound winner item map')

    def _preprocess_history(self, history_item_ids: torch.Tensor, history_lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.get_item_embeddings(history_item_ids)
        _, output, _ = self.model._input_features_preproc(past_lengths=history_lengths, past_ids=history_item_ids, past_embeddings=embeddings, past_payloads={'timestamps': torch.zeros_like(history_item_ids)})
        return output

    def _selector_scores(self, history_item_ids: torch.Tensor, history_lengths: torch.Tensor, candidate_item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, width = history_item_ids.shape
        if self.method == 'full-gdr':
            scores = torch.zeros(batch, 1, width, device=history_item_ids.device)
            return (scores, None)
        raise MetaBridgeError('candidate selector methods use their frozen select API')

    def _selection(self, history_item_ids: torch.Tensor, history_lengths: torch.Tensor, candidate_item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.method in ('deltarec-pc',):
            assert isinstance(self.selector, RatingPCSelector)
            selected = self.selector.select(history_item_ids, candidate_item_ids, history_lengths, retention_ratio=self.retention_ratio, recent_floor=self.recent_floor)
            return (selected.write_mask, selected.budgets[:, 0], None)
        if self.method == 'deltarec-gc':
            assert isinstance(self.selector, RatingGCSelector)
            selected = self.selector.select(history_item_ids, candidate_item_ids, history_lengths, retention_ratio=self.retention_ratio, recent_floor=self.recent_floor)
            return (selected.write_mask, selected.budgets[:, 0], selected.candidate_group_ids)
        scores, candidate_groups = self._selector_scores(history_item_ids, history_lengths, candidate_item_ids)
        if self.method == 'full-gdr':
            positions = torch.arange(scores.shape[-1], device=scores.device)
            mask = positions[None, None] < history_lengths[:, None, None]
            budgets = history_lengths
        else:
            mask, budgets = _stable_budget_mask(scores, history_lengths, retention_ratio=self.retention_ratio)
        return (mask, budgets, candidate_groups)

    @staticmethod
    def _compact(history_x: torch.Tensor, selection_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length_tensor = selection_mask.sum(-1, dtype=torch.int64).reshape(-1)
        if not length_tensor.numel() or bool((length_tensor < 1).any()):
            raise MetaBridgeError('every official GDR logical stream needs history')
        batch_indices, _, positions = selection_mask.nonzero(as_tuple=True)
        packed = history_x[batch_indices, positions]
        offsets = torch.cat((length_tensor.new_zeros(1), length_tensor.cumsum(0)), dim=0)
        return (packed, offsets)

    @torch.no_grad()
    def prepare_history(self, history_item_ids: torch.Tensor, history_lengths: torch.Tensor, candidate_item_ids: torch.Tensor) -> PreparedResearchHistory:
        self._validate_batch(history_item_ids, history_lengths, candidate_item_ids)
        selection, _, groups = self._selection(history_item_ids, history_lengths, candidate_item_ids)
        counts = selection.sum(-1, dtype=torch.int64)
        batch_indices, _, positions = selection.nonzero(as_tuple=True)
        lengths = counts.reshape(-1)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        offsets_cpu = offsets.cpu()
        if bool((offsets_cpu[1:] <= offsets_cpu[:-1]).any()):
            raise MetaBridgeError('every official GDR logical stream needs history')
        return PreparedResearchHistory(batch_indices=batch_indices, positions=positions, offsets=offsets, offsets_cpu=offsets_cpu, selected_counts=counts, candidate_groups=groups, selected_writes=int(offsets_cpu[-1]), eligible_writes=int(history_lengths.sum()) * counts.shape[1])

    def build_state_cache(self, *, history_item_ids: torch.Tensor, history_lengths: torch.Tensor, candidate_item_ids: torch.Tensor, prepared_history: PreparedResearchHistory | None=None) -> ResearchGDRStateCache:
        plan = prepared_history
        if plan is None:
            plan = self.prepare_history(history_item_ids, history_lengths, candidate_item_ids)
        history_x = self._preprocess_history(history_item_ids, history_lengths)
        packed = history_x[plan.batch_indices, plan.positions]
        offsets = plan.offsets
        candidate_groups = plan.candidate_groups
        states: list[torch.Tensor] = []
        for layer in _research_layers(self.model):
            assert isinstance(layer, ResearchHSTUGDRLayer)
            packed, final_state = layer.forward_gdr_streams(x=packed, x_offsets=offsets, x_offsets_cpu=plan.offsets_cpu, event_gate=torch.ones(len(packed), dtype=packed.dtype, device=packed.device), initial_state=None, return_final_state=True)
            if final_state is None:
                raise MetaBridgeError('research GDR layer omitted final state')
            states.append(final_state)
        batch, streams = plan.selected_counts.shape
        state = torch.stack(states, dim=1).reshape(batch, streams, len(states), *states[0].shape[1:])
        selected_counts = plan.selected_counts
        realized = plan.selected_writes / max(plan.eligible_writes, 1)
        candidate_key = candidate_item_ids.detach().clone() if self.method in CANDIDATE_STATE_METHODS else None
        cache = ResearchGDRStateCache(schema=RUNTIME_SCHEMA, method=self.method, state=state, history_lengths=history_lengths.detach().clone(), stream_count=streams, candidate_key=candidate_key, candidate_group_ids=None if candidate_groups is None else candidate_groups.detach().clone(), selected_counts=selected_counts, realized_write_ratio=realized, model_binding_sha256=self.model_binding_sha256, selector_binding_sha256=self.selector_binding_sha256, grouping_binding_sha256=self.grouping_binding_sha256, retention_ratio=self.retention_ratio, recent_floor=self.recent_floor)
        cache.validate()
        return cache

    def _candidate_input(self, candidate_item_ids: torch.Tensor, history_lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.get_item_embeddings(candidate_item_ids)
        preprocessor = self.model._input_features_preproc
        positions = history_lengths[:, None].expand_as(candidate_item_ids)
        if int(positions.max()) >= preprocessor._pos_emb.num_embeddings:
            raise MetaBridgeError('candidate position exceeds official positional table')
        output = embeddings * self.model_config.item_embedding_dim ** 0.5
        output = output + preprocessor._pos_emb(positions)
        return preprocessor._emb_dropout(output)

    def serve_cache_hit(self, *, candidate_item_ids: torch.Tensor, cache: ResearchGDRStateCache) -> torch.Tensor:
        cache.validate()
        if cache.method != self.method:
            raise MetaBridgeError('research GDR cache method/version mismatch')
        if cache.retention_ratio != self.retention_ratio or cache.recent_floor != self.recent_floor:
            raise MetaBridgeError('research GDR cache retention budget mismatch')
        if cache.model_binding_sha256 != self.model_binding_sha256:
            raise MetaBridgeError('research GDR cache belongs to another official model')
        if cache.selector_binding_sha256 != self.selector_binding_sha256 or cache.grouping_binding_sha256 != self.grouping_binding_sha256:
            raise MetaBridgeError('research GDR cache selector/grouping version mismatch')
        batch, candidates = candidate_item_ids.shape
        if batch != cache.state.shape[0]:
            raise ValueError('research GDR cache batch size mismatch')
        if self.method in CANDIDATE_STATE_METHODS:
            if cache.candidate_key is None or not torch.equal(cache.candidate_key, candidate_item_ids):
                raise MetaBridgeError('candidate-specific GDR cache key mismatch')
            stream_indices = torch.arange(candidates, device=candidate_item_ids.device)[None].expand(batch, -1)
        elif self.method == 'deltarec-gc':
            assert isinstance(self.selector, RatingGCSelector)
            observed_groups = self.selector.group_ids(candidate_item_ids)
            if cache.candidate_group_ids is None or not torch.equal(cache.candidate_group_ids, observed_groups):
                raise MetaBridgeError('GC cache global-group key mismatch')
            stream_indices = observed_groups
        else:
            stream_indices = torch.zeros_like(candidate_item_ids)
        if int(stream_indices.max()) >= cache.stream_count:
            raise MetaBridgeError('candidate addresses an absent GDR cache stream')
        x = self._candidate_input(candidate_item_ids, cache.history_lengths).reshape(batch * candidates, -1)
        flattened_stream = (stream_indices + torch.arange(batch, device=stream_indices.device)[:, None] * cache.stream_count).reshape(-1)
        offsets = torch.arange(batch * candidates + 1, device=x.device, dtype=torch.int64)
        for layer_index, layer in enumerate(_research_layers(self.model)):
            assert isinstance(layer, ResearchHSTUGDRLayer)
            layer_state = cache.state[:, :, layer_index].reshape(batch * cache.stream_count, *cache.state.shape[3:])
            if self.method == 'deltarec-gc':
                x = layer.forward_gdr_readonly(x=x.reshape(batch, candidates, -1), state=cache.state[:, :, layer_index], group_indices=stream_indices).reshape(batch * candidates, -1)
                continue
            initial = layer_state.index_select(0, flattened_stream).contiguous()
            x, _ = layer.forward_gdr_streams(x=x, x_offsets=offsets, event_gate=torch.zeros(len(x), dtype=x.dtype, device=x.device), initial_state=initial, return_final_state=False)
        query = self.model._output_postproc(x)
        candidate_embeddings = self.model.get_item_embeddings(candidate_item_ids.reshape(-1, 1))
        if self.model_config.item_l2_norm:
            candidate_embeddings = candidate_embeddings / torch.clamp(torch.linalg.norm(candidate_embeddings, ord=2, dim=-1, keepdim=True), min=self.model_config.l2_norm_eps)
        scores, _ = self.model.similarity_fn(query_embeddings=query, item_ids=candidate_item_ids.reshape(-1, 1), item_embeddings=candidate_embeddings)
        return scores.reshape(batch, candidates)

    def score(self, *, history_item_ids: torch.Tensor, history_lengths: torch.Tensor, candidate_item_ids: torch.Tensor) -> tuple[torch.Tensor, ResearchGDRStateCache, Mapping[str, Any]]:
        cache = self.build_state_cache(history_item_ids=history_item_ids, history_lengths=history_lengths, candidate_item_ids=candidate_item_ids)
        scores = self.serve_cache_hit(candidate_item_ids=candidate_item_ids, cache=cache)
        return (scores, cache, {'schema': RUNTIME_SCHEMA, 'method': self.method, 'state_layout': list(cache.state.shape), 'state_dtype': str(cache.state.dtype), 'stream_count': cache.stream_count, 'selected_counts': cache.selected_counts.detach().cpu().tolist(), 'realized_write_ratio': cache.realized_write_ratio, 'candidate_transition': 'read-only', 'model_binding_sha256': self.model_binding_sha256, 'selector_binding_sha256': self.selector_binding_sha256, 'grouping_binding_sha256': self.grouping_binding_sha256, 'cache_state_sha256': _tensor_sha256(cache.state)})
