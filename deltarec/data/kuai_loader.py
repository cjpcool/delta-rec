from __future__ import annotations

from pathlib import Path

from typing import Any

MetaBridgeError = ValueError

from deltarec.data.kuai_slates import COMMON_CANDIDATE_COUNT, COMMON_TOTAL_SEQUENCE_LENGTH, HeadlineKuaiSlateDataset, KUAI_CANDIDATE_KEYS, KUAI_TASK_NAMES, KUAI_UIH_KEYS

def _validate_and_apply_protocol_shape(
    config: Any, variant: str = "standard"
) -> Any:
    """Validate published Standard or apply the audited deterministic Large scale."""

    fixed = {
        "hstu_embedding_table_dim": 512,
        "hstu_preprocessor_hidden_dim": 256,
        "hstu_transducer_embedding_dim": 512,
        "hstu_group_norm": False,
        "hstu_input_dropout_ratio": 0.2,
        "hstu_linear_dropout_rate": 0.1,
        "causal_multitask_weights": 0.2,
    }
    for name, value in fixed.items():
        if getattr(config, name) != value:
            raise MetaBridgeError(
                f"unexpected pinned DLRMv3-HSTU architecture {name}="
                f"{getattr(config, name)!r}"
            )
    scale_fields = (
        "hstu_num_heads",
        "hstu_attn_linear_dim",
        "hstu_attn_qk_dim",
        "hstu_attn_num_layers",
    )
    standard = (4, 128, 128, 5)
    variants = {
        "standard": standard,
        "large": (8, 64, 64, 20),
    }
    if variant not in variants:
        raise MetaBridgeError(f"unknown Kuai HSTU variant {variant!r}")
    observed = tuple(getattr(config, name) for name in scale_fields)
    expected = variants[variant]
    if observed not in {standard, expected}:
        raise MetaBridgeError(
            f"unexpected Kuai HSTU scale fields {observed}; expected raw Standard "
            f"{standard} or resolved {variant} {expected}"
        )
    for name, value in zip(scale_fields, expected):
        setattr(config, name, value)
    if tuple(config.hstu_uih_feature_names) != KUAI_UIH_KEYS:
        raise MetaBridgeError("pinned Kuai UIH feature order changed")
    if tuple(config.hstu_candidate_feature_names) != KUAI_CANDIDATE_KEYS:
        raise MetaBridgeError("pinned Kuai candidate feature order changed")
    if tuple(task.task_name for task in config.multitask_configs) != KUAI_TASK_NAMES:
        raise MetaBridgeError("pinned Kuai eight-task order changed")
    if tuple(config.action_weights or ()) != (1, 2, 4, 8, 16, 32, 64, 128):
        raise MetaBridgeError("pinned Kuai action weights changed")
    config.max_seq_len = COMMON_TOTAL_SEQUENCE_LENGTH
    config.max_num_candidates = COMMON_CANDIDATE_COUNT
    config.max_num_candidates_inference = COMMON_CANDIDATE_COUNT
    return config

def _make_loader(
    dataset: HeadlineKuaiSlateDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    shuffle: bool,
    train_utils: Any,
    torch: Any,
) -> Any:
    from deltarec.adaptors.kuai.dlrm_v3.datasets.dataset import collate_fn

    # Preserve the published ChunkDistributedSampler/drop-last behavior for
    # optimization.  The upstream chunk sampler asserts drop_last=True even
    # for one replica, so it cannot represent the frozen evaluation
    # denominator.  Evaluation is single-replica and ordered; a sequential
    # sampler therefore preserves every slate exactly once, including the
    # final partial batch.
    drop_last = bool(shuffle)
    if shuffle:
        sampler = train_utils.ChunkDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            drop_last=True,
            shuffle=True,
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "sampler": sampler,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(**kwargs)

def make_frozen_kuai_dataloaders(
    *,
    train_slates: Path | None,
    evaluation_slates: Path,
    user_features: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    train_utils: Any,
    torch: Any,
) -> tuple[Any | None, Any]:
    evaluation_dataset = HeadlineKuaiSlateDataset(evaluation_slates, user_features)
    evaluation_loader = _make_loader(
        evaluation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        train_utils=train_utils,
        torch=torch,
    )
    if train_slates is None:
        return None, evaluation_loader
    training_dataset = HeadlineKuaiSlateDataset(train_slates, user_features)
    training_loader = _make_loader(
        training_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
        train_utils=train_utils,
        torch=torch,
    )
    return training_loader, evaluation_loader

def _find_multitask_module(model: Any) -> Any:
    root = getattr(model, "module", model)
    matches = [
        module
        for module in root.modules()
        if module.__class__.__name__ == "DefaultMultitaskModule"
    ]
    if len(matches) != 1:
        raise MetaBridgeError(
            f"expected one official DefaultMultitaskModule, found {len(matches)}"
        )
    return matches[0]

def _kuai_video_embedding_state(model: Any) -> tuple[str, Any]:
    """Locate only the pinned TorchRec ``video_id`` table in a DMP state."""

    candidates: list[tuple[str, Any]] = []
    for name, value in model.state_dict().items():
        if (
            name.endswith("_embedding_collection.embeddings.video_id.weight")
            or name.endswith("embedding_collection.embeddings.video_id.weight")
        ):
            candidates.append((name, value))
    if len(candidates) != 1:
        raise MetaBridgeError(
            "expected exactly one official Kuai video_id embedding table, "
            f"found {[name for name, _ in candidates]}"
        )
    name, table = candidates[0]
    try:
        shape = tuple(int(value) for value in table.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise MetaBridgeError("official Kuai video table has no global shape") from error
    if len(shape) != 2 or shape[1] != 512 or shape[0] < 2:
        raise MetaBridgeError(
            f"official Kuai video table shape changed: {shape} != [N,512]"
        )
    return name, table
