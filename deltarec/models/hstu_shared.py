import torch
from torch.nn import functional as F
from deltarec.models.hstu_packed import BatchedSparseScoring, finish_layer

class SharedSparseScoring(BatchedSparseScoring):
    def packed_scores(self, full_ids, rows, ends, streams, positions, offsets_cpu, candidates):
        """Gather IDs/original positions BEFORE embedding; isolate prefix states."""
        preproc = self.model._input_features_preproc
        ids = full_ids[rows.to(full_ids.device)[streams], positions]
        x = self.model.get_item_embeddings(ids) * preproc._embedding_dim**.5 + preproc._pos_emb(positions)
        if preproc._emb_dropout.training and preproc._dropout_rate:
            # Keep v8's exact dense-prefix dropout draw layout/RNG consumption.
            # No embedding graph for discarded events; the mask is non-grad.
            mask = preproc._emb_dropout(torch.ones(len(rows), int(ends.max()), x.shape[-1], device=x.device, dtype=x.dtype))
            x = x * mask[streams, positions]
        offsets = offsets_cpu.to(x.device)
        last = offsets[1:] - 1
        self.counts.history_streams += len(rows)
        self.counts.history_model_calls += 1
        self.counts.first_layer_projection_tokens += len(x)
        for index, (layer, kernel) in enumerate(zip(self.layers, self.kernels)):
            u, q, k, v, decay, beta = layer._project(x)
            projected = layer._kernel_input_type(q=q, k=k, v=v, decay_logits=decay,
                beta_logits=beta, log_decay_scale=layer.gdr_log_decay_scale,
                decay_bias=layer.gdr_decay_bias, offsets=offsets, offsets_cpu=offsets_cpu,
                event_gate=torch.ones(len(x), device=x.device, dtype=x.dtype))
            result = kernel(projected, initial_state=None, return_final_state=False)
            if index + 1 == len(self.layers):
                # Every retained transition still executes. Only the unused
                # final-layer output linear/residual is pruned, AFTER dropout
                # so training masks and global RNG remain unchanged.
                base = layer.base
                attention = result.context.reshape(-1, base._num_heads * base._linear_dim)
                normalized = base._norm_attn_output(attention)
                value = torch.cat((u, attention, u * normalized), -1) if base._concat_ua else u * normalized
                value = F.dropout(value, p=float(base._dropout_ratio), training=layer.training)
                x = base._o(value[last]) + x[last]
            else:
                x = finish_layer(layer, x, u, result.context)
        return self._score_head(self.model._output_postproc(x), candidates)
