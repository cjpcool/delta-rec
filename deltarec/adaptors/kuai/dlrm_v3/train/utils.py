# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.
import logging

import os

from collections.abc import Iterator

from datetime import timedelta

from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Type, Union

import gin

import torch

import torchrec

from deltarec.adaptors.kuai.dlrm_v3.configs import get_embedding_table_config, get_hstu_configs

from deltarec.adaptors.kuai.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig

from torch import distributed as dist

from torch.distributed.optim import _apply_optimizer_in_backward as apply_optimizer_in_backward

from torch.optim.optimizer import Optimizer

from torch.utils.data import Dataset as TorchDataset

from torch.utils.data.distributed import _T_co, DistributedSampler

from torchrec.distributed.model_parallel import DistributedModelParallel

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology

from torchrec.distributed.sharding_plan import get_default_sharders

from torchrec.distributed.types import ShardedTensor, ShardingEnv

from torchrec.modules.embedding_configs import EmbeddingConfig

from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingCollection

from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper

from torchrec.optim.optimizers import in_backward_optimizer_filter

logger: logging.Logger = logging.getLogger(__name__)

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

def setup(
    rank: int, world_size: int, master_port: int, device: torch.device
) -> dist.ProcessGroup:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    BACKEND = dist.Backend.NCCL
    TIMEOUT = 1800

    # initialize the process group
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )

    # set device
    torch.cuda.set_device(device)

    return pg

def cleanup() -> None:
    dist.destroy_process_group()

class ChunkDistributedSampler(DistributedSampler[_T_co]):
    """
    Each rank reads a contiguous chunk (trunk) of the input data
    """

    def __init__(
        self,
        dataset: TorchDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 1001 + int(self.rank))
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))
        assert self.drop_last is True, (
            "drop_last must be True for ChunkDistributedSampler"
        )
        indices = [i + self.num_samples * self.rank for i in indices]

        assert len(indices) == self.num_samples
        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        logger.warning(f"Setting epoch to {epoch}")
        self.epoch = epoch

@gin.configurable
def make_model(
    dataset: str,
) -> Tuple[torch.nn.Module, DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:
    hstu_config = get_hstu_configs(dataset)
    table_config = get_embedding_table_config(dataset)

    model = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=False,
    )

    return (
        model,
        hstu_config,
        table_config,
    )

@gin.configurable()
def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory

@gin.configurable()
def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory

def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
    learning_rate_multiplier: float = 1.0,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    if learning_rate_multiplier not in (0.03, 0.1, 0.3, 1.0):
        raise ValueError("unsupported optimizer learning-rate multiplier")
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class()
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class()
    )
    # Scale before registering the in-backward optimizer. TorchRec copies the
    # sparse kwargs into the fused FBGEMM kernel during DMP construction.
    dense_opt_args["lr"] = float(dense_opt_args["lr"]) * learning_rate_multiplier
    sparse_opt_args["lr"] = float(sparse_opt_args["lr"]) * learning_rate_multiplier
    # Fuse sparse optimizer to backward step
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
    sharders = get_default_sharders()
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=world_size,
            world_size=world_size,
            compute_device="cuda",
            hbm_cap=160 * 1024 * 1024 * 1024,
            ddr_cap=32 * 1024 * 1024 * 1024,
        )
    )
    pg = dist.GroupMember.WORLD
    env = ShardingEnv.from_process_group(pg)  # pyre-ignore [6]
    pg = env.process_group

    plan = planner.collective_plan(model, sharders, pg)

    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
    )
    # Create keyed optimizer
    all_optimizers = []
    # DMP keeps the in-backward optimizer separately. Include it so LR evidence
    # and durable checkpoints cover its RowWiseAdagrad state as well as dense
    # Adam state.
    fused_optimizer = getattr(model, "fused_optimizer", None)
    if fused_optimizer is None:
        raise RuntimeError("DMP did not expose the fused sparse optimizer")
    all_optimizers.append(("sparse_fused", fused_optimizer))
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_opt_factory,
                ),
            )
        )
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer
