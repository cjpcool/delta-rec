from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from deltarec.layers.paper_cwi import official_rating_cwi

from deltarec.models.hstu_runtime import OfficialResearchSparseScorer

def label_contexts(
    scorer: OfficialResearchSparseScorer, official_loss_from_queries: Any,
    contexts: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not contexts:
        raise ValueError('HSTU CWI requires at least one official training context')
    device = next(scorer.model.parameters()).device
    lengths = torch.tensor([len(row['history']) for row in contexts],
                           dtype=torch.int64, device=device)
    width = int(lengths.max())
    histories = torch.zeros((len(contexts), width), dtype=torch.int64, device=device)
    for index, row in enumerate(contexts):
        histories[index, :lengths[index]] = torch.tensor(
            row['history'], dtype=torch.int64, device=device)
    positives = torch.tensor([[int(row['target'])] for row in contexts],
                             dtype=torch.int64, device=device)
    _, labels, _ = official_rating_cwi(
        scorer, histories, lengths, positives,
        official_loss_from_queries=official_loss_from_queries,
    )
    return histories.detach().cpu(), lengths.detach().cpu(), positives.detach().cpu(), labels.detach().cpu()
