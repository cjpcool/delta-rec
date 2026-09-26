import torch

from torch.utils.checkpoint import checkpoint

from deltarec.models.hstu_runtime import OfficialResearchSparseScorer, PreparedResearchHistory

from deltarec.models.hstu_finetune import OfficialCandidateConditionedSampledSoftmaxLoss

from deltarec.layers.hstu_gdr import _research_layers

class AmazonScorer(OfficialResearchSparseScorer):
    def prepare_history(self, history_item_ids, history_lengths, candidate_item_ids):
        if self.method in ('deltarec-pc','deltarec-gc') and bool((history_lengths <= 32).all()):
            self._validate_batch(history_item_ids, history_lengths, candidate_item_ids)
            valid = torch.arange(history_item_ids.shape[1], device=history_lengths.device)[None] < history_lengths[:, None]
            rows, positions = valid.nonzero(as_tuple=True)
            offsets = torch.cat((history_lengths.new_zeros(1), history_lengths.cumsum(0)))
            cpu = offsets.cpu()
            return PreparedResearchHistory(rows, positions, offsets, cpu,
                history_lengths[:, None],
                self.selector.group_ids(candidate_item_ids) if self.method == 'deltarec-gc' else None,
                int(cpu[-1]), int(cpu[-1]))
        return super().prepare_history(history_item_ids, history_lengths, candidate_item_ids)

    def serve_cache_hit(self, *, candidate_item_ids, cache):
        # One physical state for short PC histories; its candidate key is not a
        # history dependency. A full-gdr view retains all lineage/budget checks.
        shared = cache.stream_count == 1 and bool((cache.history_lengths <= 32).all())
        if self.method == 'full-gdr' or (self.method in ('deltarec-pc','deltarec-gc') and shared):
            cache.validate()
            if (cache.method != self.method or cache.grouping_binding_sha256 != self.grouping_binding_sha256 or
                cache.model_binding_sha256 != self.model_binding_sha256 or
                cache.selector_binding_sha256 != self.selector_binding_sha256 or
                cache.retention_ratio != self.retention_ratio or cache.recent_floor != 32):
                raise ValueError('short-history cache lineage mismatch')
            if candidate_item_ids.shape[0] != cache.state.shape[0]:
                raise ValueError('short-history cache batch mismatch')
            batch, count = candidate_item_ids.shape
            x = self._candidate_input(candidate_item_ids, cache.history_lengths)
            groups = torch.zeros_like(candidate_item_ids)
            for i, layer in enumerate(_research_layers(self.model)):
                x = layer.forward_gdr_readonly(x=x, state=cache.state[:, :, i], group_indices=groups)
            query = self.model._output_postproc(x.reshape(batch * count, -1))
            embeddings = self.model.get_item_embeddings(candidate_item_ids.reshape(-1, 1))
            if self.model_config.item_l2_norm:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(self.model_config.l2_norm_eps)
            scores, _ = self.model.similarity_fn(query_embeddings=query,
                item_ids=candidate_item_ids.reshape(-1, 1), item_embeddings=embeddings)
            return scores.reshape(batch, count)
        return super().serve_cache_hit(candidate_item_ids=candidate_item_ids, cache=cache)

class AmazonLoss(OfficialCandidateConditionedSampledSoftmaxLoss):
    """Same all-position local sampled softmax; PC avoids selector recomputation."""
    def _score_candidates(self, histories, history_lengths, candidates):
        # Split mixed prefix chunks so every short request gets a single state.
        short = history_lengths <= 32
        if bool(short.any()) and not bool(short.all()):
            out = None
            for mask in (short, ~short):
                index = mask.nonzero().flatten()
                lengths = history_lengths[index]
                values = self._score_candidates(histories[index, :int(lengths.max())], lengths, candidates[index])
                if out is None:
                    out = values.new_empty(candidates.shape)
                out = out.index_copy(0, index, values)
            return out
        slices = [candidates] if bool(short.all()) or self._scorer.method == 'deltarec-gc' else candidates.split(self._candidate_chunk_size, dim=1)
        outputs = []
        for ids in slices:
            plan = self._scorer.prepare_history(histories, history_lengths, ids)
            if torch.is_grad_enabled():
                score = checkpoint(self._score_prepared_candidate_slice, histories,
                    history_lengths, ids, plan, use_reentrant=False, preserve_rng_state=True)
            else:
                score = self._score_prepared_candidate_slice(histories, history_lengths, ids, plan)
            outputs.append(score)
            self._selected_writes += plan.selected_writes
            self._eligible_writes += plan.eligible_writes
        return torch.cat(outputs, dim=1)

class IDOnlyLocalSampler:
    """Proxy is not used: install this exact sampler forward on a local instance."""
    @staticmethod
    def forward(sampler, positive_ids, num_to_sample):
        shape = positive_ids.size() + (num_to_sample,)
        positions = torch.randint(0, sampler._num_items, shape,
            dtype=positive_ids.dtype, device=positive_ids.device)
        ids = sampler._all_item_ids[positions.reshape(-1)].reshape(shape)
        return ids, None
