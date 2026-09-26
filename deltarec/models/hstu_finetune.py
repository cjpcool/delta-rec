from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

import torch.nn.functional as F

from torch import nn

from torch.utils.checkpoint import checkpoint

from deltarec.adaptors.hstu_model import MetaBridgeError

from deltarec.models.hstu_runtime import OfficialResearchSparseScorer

class OfficialCandidateConditionedSampledSoftmaxLoss(nn.Module):
    """Published local sampled-softmax with one GDR query per candidate."""

    def __init__(
        self,
        *,
        num_to_sample: int,
        softmax_temperature: float,
        model: nn.Module,
        scorer: OfficialResearchSparseScorer,
        context_owner: Any,
        train_catalog_ids: Sequence[int],
        supervision_chunk_size: int,
        candidate_chunk_size: int,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if activation_checkpoint:
            raise MetaBridgeError(
                "published rating configs do not enable loss activation checkpointing"
            )
        if num_to_sample < 1 or softmax_temperature <= 0:
            raise MetaBridgeError("invalid published sampled-softmax parameters")
        if scorer.model is not model:
            raise MetaBridgeError("sparse scorer must own the exact official trainer model")
        self._num_to_sample = int(num_to_sample)
        self._softmax_temperature = float(softmax_temperature)
        self._model = model
        self._scorer = scorer
        object.__setattr__(self, "_context_owner", context_owner)
        self.register_buffer(
            "_bound_train_catalog",
            torch.tensor(tuple(train_catalog_ids), dtype=torch.int64),
            persistent=False,
        )
        self._supervision_chunk_size = int(supervision_chunk_size)
        self._candidate_chunk_size = int(candidate_chunk_size)
        self._calls = 0
        self._supervision_tokens = 0
        self._sampled_candidates = 0
        self._selected_writes = 0
        self._eligible_writes = 0
        object.__setattr__(self, "_bound_negative_sampler", None)
        object.__setattr__(self, "_bound_sampler_catalog", None)
        self._bound_sampler_catalog_version = None

    def _verify_sampler(self, sampler: Any) -> None:
        if type(sampler).__qualname__ != "LocalNegativesSampler" or type(
            sampler
        ).__module__ != (
            "deltarec.adaptors.hstu.modeling.sequential."
            "autoregressive_losses"
        ):
            raise MetaBridgeError("sparse fine-tuning requires official LocalNegativesSampler")
        observed = getattr(sampler, "_all_item_ids", None)
        if sampler is self._bound_negative_sampler:
            if (
                observed is not self._bound_sampler_catalog
                or observed._version != self._bound_sampler_catalog_version
                or getattr(sampler, "_item_emb", None)
                is not self._model._embedding_module._item_emb
            ):
                raise MetaBridgeError("bound negative sampler changed during training")
            return
        if not isinstance(observed, torch.Tensor) or not torch.equal(
            observed.detach().cpu().to(torch.int64),
            self._bound_train_catalog.detach().cpu(),
        ):
            raise MetaBridgeError(
                "official local sampler catalog differs from the bound train-only catalog"
            )
        if getattr(sampler, "_item_emb", None) is not self._model._embedding_module._item_emb:
            raise MetaBridgeError("negative sampler does not reference the exact official table")
        object.__setattr__(self, "_bound_negative_sampler", sampler)
        object.__setattr__(self, "_bound_sampler_catalog", observed)
        self._bound_sampler_catalog_version = observed._version

    @staticmethod
    def _prefix_rows(
        full_ids: torch.Tensor,
        row_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        width = int(prefix_lengths.max())
        rows = full_ids.index_select(0, row_indices)[:, :width].clone()
        positions = torch.arange(width, device=rows.device)[None]
        rows.masked_fill_(positions >= prefix_lengths[:, None], 0)
        return rows

    def _score_candidates(
        self,
        histories: torch.Tensor,
        history_lengths: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if self._scorer.method == "deltarec-gc":
            # Selector/Top-B are deterministic and have no gradients. Retain
            # only packed indices; recompute differentiable history and reads
            # with the original dropout RNG state during backward.
            plan = self._scorer.prepare_history(histories, history_lengths, candidates)
            if torch.is_grad_enabled():
                scores = checkpoint(
                    self._score_prepared_candidate_slice, histories, history_lengths,
                    candidates, plan, use_reentrant=False, preserve_rng_state=True,
                )
            else:
                scores = self._score_prepared_candidate_slice(histories, history_lengths, candidates, plan)
            self._selected_writes += plan.selected_writes
            self._eligible_writes += plan.eligible_writes
            return scores
        else:
            slices = tuple(
                candidates[:, start : start + self._candidate_chunk_size]
                for start in range(0, candidates.shape[1], self._candidate_chunk_size)
            )
        outputs: list[torch.Tensor] = []
        for candidate_slice in slices:
            # Recomputation prevents all candidate/prefix activation graphs
            # from remaining resident until the final sampled-softmax backward.
            # Integer inputs are supported by non-reentrant checkpointing;
            # parameter and history-state gradients remain connected.
            if torch.is_grad_enabled():
                scores, selected, eligible = checkpoint(
                    self._score_candidate_slice, histories, history_lengths,
                    candidate_slice, use_reentrant=False, preserve_rng_state=True,
                )
            else:
                scores, selected, eligible = self._score_candidate_slice(
                    histories, history_lengths, candidate_slice
                )
            outputs.append(scores)
            self._selected_writes += int(selected.detach())
            self._eligible_writes += int(eligible.detach())
        return torch.cat(outputs, dim=1)

    def _score_prepared_candidate_slice(self, histories, history_lengths, candidates, plan):
        cache = self._scorer.build_state_cache(
            history_item_ids=histories, history_lengths=history_lengths,
            candidate_item_ids=candidates, prepared_history=plan,
        )
        return self._scorer.serve_cache_hit(candidate_item_ids=candidates, cache=cache)

    def _score_candidate_slice(self, histories, history_lengths, candidate_slice):
        # Serving-only evidence would copy the entire state to CPU for hashing.
        cache = self._scorer.build_state_cache(
            history_item_ids=histories, history_lengths=history_lengths,
            candidate_item_ids=candidate_slice,
        )
        scores = self._scorer.serve_cache_hit(candidate_item_ids=candidate_slice, cache=cache)
        return scores, cache.selected_counts.sum(), history_lengths.sum() * cache.stream_count

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        self._verify_sampler(negatives_sampler)
        context = self._context_owner.consume_training_context()
        full_ids = context["past_ids"]
        full_lengths = context["past_lengths"]
        if not torch.equal(full_lengths, lengths) or full_ids.shape[0] != lengths.shape[0]:
            raise MetaBridgeError("sparse loss context does not match official model forward")
        if output_embeddings.shape != supervision_embeddings.shape or (
            supervision_ids.shape != supervision_weights.shape
            or supervision_ids.shape != supervision_embeddings.shape[:-1]
        ):
            raise MetaBridgeError("official sampled-softmax tensor shapes changed")
        positions = torch.arange(supervision_ids.shape[1], device=lengths.device)[None]
        valid = positions < lengths[:, None]
        if bool((supervision_ids.masked_select(valid) <= 0).any()):
            raise MetaBridgeError("valid next-item supervision contains padding")
        row_indices = torch.arange(lengths.numel(), device=lengths.device).repeat_interleave(
            lengths.to(torch.int64)
        )
        prefix_lengths = torch.cat(
            [
                torch.arange(1, int(length) + 1, device=lengths.device)
                for length in lengths.detach().cpu().tolist()
            ]
        )
        positive_ids = supervision_ids.masked_select(valid)
        weights = supervision_weights.masked_select(valid)
        if positive_ids.numel() != row_indices.numel() or float(weights.sum()) <= 0:
            raise MetaBridgeError("sparse sampled-softmax supervision is empty")
        if self._scorer.method == 'deltarec-gc':
            # Exact LocalNegativesSampler randint shape/dtype/device and ID
            # lookup, without constructing embeddings the loss never uses.
            output_shape = positive_ids.size() + (self._num_to_sample,)
            sampled_offsets = torch.randint(
                low=0, high=negatives_sampler._num_items, size=output_shape,
                dtype=positive_ids.dtype, device=positive_ids.device,
            )
            sampled_ids = negatives_sampler._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        else:
            sampled_ids, sampled_embeddings = negatives_sampler(
                positive_ids=positive_ids, num_to_sample=self._num_to_sample,
            )
            del sampled_embeddings
        if sampled_ids.shape != (positive_ids.numel(), self._num_to_sample):
            raise MetaBridgeError("official local sampler output shape changed")
        candidates = torch.cat((positive_ids[:, None], sampled_ids), dim=1)
        numerator = output_embeddings.new_zeros(())
        for start in range(0, positive_ids.numel(), self._supervision_chunk_size):
            stop = min(start + self._supervision_chunk_size, positive_ids.numel())
            chunk_rows = row_indices[start:stop]
            chunk_lengths = prefix_lengths[start:stop]
            histories = self._prefix_rows(full_ids, chunk_rows, chunk_lengths)
            chunk_candidates = candidates[start:stop]
            scores = self._score_candidates(
                histories, chunk_lengths, chunk_candidates
            )
            positive_logits = scores[:, :1] / self._softmax_temperature
            negative_logits = torch.where(
                chunk_candidates[:, 1:].eq(chunk_candidates[:, :1]),
                scores[:, 1:].new_full((), -5e4),
                scores[:, 1:] / self._softmax_temperature,
            )
            per_token = -F.log_softmax(
                torch.cat((positive_logits, negative_logits), dim=1), dim=1
            )[:, 0]
            numerator = numerator + (
                per_token * weights[start:stop].to(per_token.dtype)
            ).sum()
        denominator = weights.sum()
        self._calls += 1
        self._supervision_tokens += int(positive_ids.numel())
        self._sampled_candidates += int(candidates.numel())
        return numerator / denominator, {}

    def evidence(self) -> Mapping[str, Any]:
        return {
            "schema": "deltarec-official-candidate-conditioned-sampled-softmax-v1",
            "candidate_source": "official-local-negative-sampler-bound-training-catalog",
            "positive_source": "next item strictly after each history prefix",
            "validation_top100_gradient_visible": False,
            "test_data_gradient_visible": False,
            "negative_history_collisions": "retained exactly as official sampler",
            "positive_equivalent_negative": "logit-masked-minus-5e4",
            "loss_denominator": "sum(supervision_weights) per microbatch",
            "optimizer_window_denominator": (
                "sum(ar_mask) across token-exact accumulation window"
            ),
            "num_negatives": self._num_to_sample,
            "temperature": self._softmax_temperature,
            "supervision_chunk_size": self._supervision_chunk_size,
            "candidate_chunk_size": self._candidate_chunk_size,
            "candidate_activation_recomputation": "non-reentrant-preserve-rng",
            "calls": self._calls,
            "supervision_tokens": self._supervision_tokens,
            "sampled_candidates": self._sampled_candidates,
            "selected_writes": self._selected_writes,
            "eligible_writes": self._eligible_writes,
            "actual_write_ratio": (
                None
                if not self._eligible_writes
                else self._selected_writes / self._eligible_writes
            ),
        }
