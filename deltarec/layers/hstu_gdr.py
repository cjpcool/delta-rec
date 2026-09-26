from __future__ import annotations

import hashlib

import importlib

import math

from typing import Any, Mapping, Sequence

import torch

import torch.nn.functional as F

from torch import nn

MetaBridgeError = ValueError

REPLACEMENT_SCHEMA = "deltarec-meta-research-hstu-full-gdr-v1"

SUPPORTED_KERNELS = frozenset(("reference", "fla"))

def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(
        f"{tuple(tensor.shape)}:{tensor.dtype}".encode("ascii")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor_digest = _tensor_sha256(value)
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(tensor_digest))
    return digest.hexdigest()

def _canonical_official_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Remove only the wrapper namespace and newly introduced GDR tensors."""

    canonical: dict[str, torch.Tensor] = {}
    for raw_name, value in model.state_dict().items():
        if any(
            raw_name.endswith(suffix)
            for suffix in (
                ".gdr_log_decay_scale",
                ".gdr_decay_bias",
                ".gdr_gate_weight",
            )
        ):
            continue
        name = raw_name.replace(".base.", ".")
        if name in canonical:
            raise MetaBridgeError(f"official state canonicalization collision: {name}")
        canonical[name] = value.detach().cpu()
    return canonical

def _research_layers(model: nn.Module) -> Sequence[nn.Module]:
    hstu_stack = getattr(model, "_hstu", None)
    layers = getattr(hstu_stack, "_attention_layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise MetaBridgeError("model is not the pinned research HSTU stack")
    return tuple(layers)

def audit_research_hstu_contract(model: nn.Module) -> dict[str, Any]:
    """Audit the exact official layer tensors and the incompatible DLRMv3 seam."""

    layers = _research_layers(model)
    layer_records: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        module_name = type(layer).__module__
        class_name = type(layer).__qualname__
        if module_name != "deltarec.adaptors.hstu.modeling.sequential.hstu" or (
            class_name != "SequentialTransductionUnitJagged"
        ):
            raise MetaBridgeError(
                f"unexpected official HSTU layer {index}: {module_name}.{class_name}"
            )
        embedding_dim = int(layer._embedding_dim)
        heads = int(layer._num_heads)
        dqk = int(layer._attention_dim)
        dv = int(layer._linear_dim)
        expected_uvqk = (embedding_dim, heads * (2 * dv + 2 * dqk))
        expected_o_weight = (
            embedding_dim,
            heads * dv * (3 if bool(layer._concat_ua) else 1),
        )
        if tuple(layer._uvqk.shape) != expected_uvqk:
            raise MetaBridgeError(f"research HSTU UVQK shape changed in layer {index}")
        if tuple(layer._o.weight.shape) != expected_o_weight or tuple(
            layer._o.bias.shape
        ) != (embedding_dim,):
            raise MetaBridgeError(f"research HSTU output projection changed in layer {index}")
        if any(name.startswith("_input_norm_") for name, _ in layer.named_parameters()):
            raise MetaBridgeError("research HSTU unexpectedly gained affine input norm")
        relative = getattr(layer, "_rel_attn_bias", None)
        relative_tensors = (
            {}
            if relative is None
            else {
                name: {
                    "shape": list(value.shape),
                    "sha256": _tensor_sha256(value),
                }
                for name, value in relative.named_parameters()
            }
        )
        layer_records.append(
            {
                "index": index,
                "embedding_dim": embedding_dim,
                "heads": heads,
                "dqk": dqk,
                "dv": dv,
                "uvqk": {
                    "shape": list(layer._uvqk.shape),
                    "bias": False,
                    "split_order": ["u", "v", "q", "k"],
                },
                "input_norm": "non-affine-layer-norm",
                "relative_attention_bias": relative_tensors,
                "output": {
                    "weight_shape": list(layer._o.weight.shape),
                    "bias_shape": list(layer._o.bias.shape),
                    "concat_ua": bool(layer._concat_ua),
                    "formula": (
                        "Linear([u,a,u*a])+x"
                        if bool(layer._concat_ua)
                        else "Linear(u*LN(a))+x"
                    ),
                },
            }
        )
    return {
        "schema": "deltarec-research-vs-dlrm-hstu-contract-audit-v1",
        "research_backbone": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "layers": layer_records,
            "embedding": "official LocalEmbeddingModule",
            "preprocessor": "official learnable positional embedding",
            "postprocessor": type(model._output_postproc).__qualname__,
            "scorer": type(model._ndp_module).__qualname__,
        },
        "dlrm_modules_stu": {
            "input_norm": "affine-layer-norm(weight,bias)",
            "uvqk_bias": True,
            "projection_split_order": ["u", "q", "k", "v"],
            "output": "3*H*Dv-to-D weight, no bias; concat(u,a,u*a)",
            "target_aware_attention": True,
            "relative_time_position_bias": False,
        },
        "lossless_state_mapping_possible": False,
        "blocking_differences": [
            "research input LayerNorm has no affine tensors; modules.stu has weight+bias",
            "research UVQK has no bias; modules.stu requires UVQK bias",
            "published research concat_ua=False uses H*Dv-to-D output weight plus bias",
            "modules.stu uses 3*H*Dv-to-D output weight without output bias",
            "research dense attention consumes learned relative time/position bias",
            "modules.stu uses a different target-aware packed attention contract",
            "former headline model replaces official embedding/preprocessor/postprocessor/scorer",
        ],
        "required_bridge": "strict-load-official-then-replace-attention-only",
    }

class ResearchHSTUGDRLayer(nn.Module):
    """Canonical GDR attention inside one untouched research HSTU layer."""

    def __init__(
        self,
        base: nn.Module,
        *,
        seed: int,
        kernel_backend: str,
        mode: str = "gdr",
    ) -> None:
        super().__init__()
        if kernel_backend not in SUPPORTED_KERNELS:
            raise ValueError("research HSTU GDR kernel must be 'reference' or 'fla'")
        if mode not in ("passthrough", "gdr"):
            raise ValueError("research HSTU replacement mode must be passthrough or gdr")
        if type(base).__module__ != (
            "deltarec.adaptors.hstu.modeling.sequential.hstu"
        ) or type(base).__qualname__ != "SequentialTransductionUnitJagged":
            raise MetaBridgeError("GDR replacement requires an official research HSTU layer")
        self.base = base
        self.kernel_backend = kernel_backend
        self.mode = mode
        kernels = importlib.import_module(
            "deltarec.layers.gdr_kernels"
        )
        selective = importlib.import_module(
            "deltarec.layers.selective_gdr"
        )
        self._kernel_input_type = selective.GDRKernelInput
        self.reference_kernel = kernels.ReferenceGDRKernel()
        self.fla_kernel = kernels.FLAGDRKernel(
            validate_inputs=True, assume_binary_event_gate=True
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        heads = int(base._num_heads)
        decay_scale = torch.empty(heads).uniform_(0, 16, generator=generator)
        self.gdr_log_decay_scale = nn.Parameter(
            torch.log(decay_scale.clamp_min(1e-4))
        )
        dt = torch.exp(
            torch.rand(heads, generator=generator)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.gdr_decay_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.gdr_gate_weight = nn.Parameter(
            torch.zeros(int(base._embedding_dim), 2 * heads)
        )
        self.last_final_state: torch.Tensor | None = None
        self.train(base.training)

    def set_mode(self, mode: str) -> None:
        if mode not in ("passthrough", "gdr"):
            raise ValueError("research HSTU replacement mode must be passthrough or gdr")
        self.mode = mode

    def set_decay_timescales(self, interval: Sequence[float], *, seed: int) -> None:
        if len(interval) != 2:
            raise ValueError("decay timescale interval requires two values")
        low, high = float(interval[0]), float(interval[1])
        if not 0 < low <= high:
            raise ValueError("invalid decay timescale interval")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        heads = int(self.base._num_heads)
        dt = torch.exp(
            torch.rand(heads, generator=generator)
            * (math.log(high) - math.log(low))
            + math.log(low)
        ).clamp_min(1e-5)
        bias = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.gdr_decay_bias.copy_(bias.to(self.gdr_decay_bias))

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.base
        normed = base._norm_input(x)
        projected = torch.mm(normed, base._uvqk)
        if base._linear_activation == "silu":
            projected = F.silu(projected)
        elif base._linear_activation != "none":
            raise MetaBridgeError(
                f"unsupported official HSTU linear activation {base._linear_activation!r}"
            )
        heads = int(base._num_heads)
        dqk = int(base._attention_dim)
        dv = int(base._linear_dim)
        u, v, q, k = torch.split(
            projected,
            (heads * dv, heads * dv, heads * dqk, heads * dqk),
            dim=1,
        )
        gates = torch.mm(normed, self.gdr_gate_weight.to(normed.dtype))
        decay_logits, beta_logits = gates.chunk(2, dim=-1)
        return (
            u,
            q.view(-1, heads, dqk),
            k.view(-1, heads, dqk),
            v.view(-1, heads, dv),
            decay_logits,
            beta_logits,
        )

    def forward_gdr_readonly(
        self,
        *,
        x: torch.Tensor,
        state: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Read grouped FP32 states without executing a zero-write recurrence.

        x is [B,C,D], state is [B,G,H,Dk,Dv]. Computing all G read
        products and gathering the chosen group avoids replicating the much
        larger recurrent matrix per candidate. Autograd sums candidate
        contributions into their original group states.
        """
        base = self.base
        batch, candidates, dimension = x.shape
        heads, dqk, dv = int(base._num_heads), int(base._attention_dim), int(base._linear_dim)
        if self.mode != 'gdr' or state.dtype != torch.float32:
            raise MetaBridgeError('read-only GDR requires canonical FP32 state')
        if state.shape[0] != batch or state.shape[2:] != (heads, dqk, dv):
            raise ValueError('read-only GDR state shape mismatch')
        if group_indices.shape != (batch, candidates):
            raise ValueError('read-only GDR group shape mismatch')
        residual = x.reshape(-1, dimension)
        normed = base._norm_input(residual)
        # Keep the original parameter tensor and optimizer slots. K/V and
        # transition gates cannot affect a read with gamma=1 and beta=0.
        weights = torch.cat((base._uvqk[:, :heads * dv],
                             base._uvqk[:, 2 * heads * dv:2 * heads * dv + heads * dqk]), dim=1)
        uq = torch.mm(normed, weights)
        if base._linear_activation == 'silu':
            uq = F.silu(uq)
        elif base._linear_activation != 'none':
            raise MetaBridgeError('unsupported read-only GDR activation')
        u, q = uq.split((heads * dv, heads * dqk), dim=-1)
        q = q.reshape(batch, candidates, heads, dqk).float()
        # This is the canonical/FLA normalization: sqrt(sum(q^2) + eps),
        # not F.normalize's clamp(norm, eps).
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        groups = state.shape[1]
        matrices = state.permute(0, 2, 3, 1, 4).reshape(batch * heads, dqk, groups * dv)
        queries = q.permute(0, 2, 1, 3).reshape(batch * heads, candidates, dqk)
        products = torch.bmm(queries, matrices).reshape(batch, heads, candidates, groups, dv)
        indices = group_indices[:, None, :, None, None].expand(batch, heads, candidates, 1, dv)
        attention = products.gather(3, indices).squeeze(3).permute(0, 2, 1, 3)
        attention = (attention * (dqk ** -0.5)).to(u.dtype).reshape(-1, heads * dv)
        self.last_final_state = None
        if bool(base._concat_ua):
            output_input = torch.cat((u, attention, u * base._norm_attn_output(attention)), dim=-1)
        else:
            output_input = u * base._norm_attn_output(attention)
        # The dropout call has exactly the original [B*C,D] shape/order.
        return (base._o(F.dropout(output_input, p=float(base._dropout_ratio),
                                  training=self.training)) + residual).reshape(batch, candidates, dimension)

    def forward_gdr_streams(
        self,
        *,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        event_gate: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = True,
        x_offsets_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run packed logical streams with explicit canonical GDR state.

        This is the sparse/cache seam used by rating Recent/Rank/Target/PC/GC.
        It deliberately bypasses the research HSTU K/V cache schema: each
        initial/final state is FP32 ``[streams,H,Dk,Dv]`` and event gating is
        applied only to decay and beta inside the canonical kernel.
        """

        if self.mode != "gdr":
            raise MetaBridgeError("explicit GDR streams require replacement mode=gdr")
        if x.ndim != 2 or x_offsets.ndim != 1:
            raise ValueError("research GDR streams expect packed x [M,D]")
        if int(x_offsets[0]) != 0 or int(x_offsets[-1]) != len(x) or bool(
            (x_offsets[1:] < x_offsets[:-1]).any()
        ):
            raise ValueError("research GDR stream offsets do not span packed x")
        if event_gate.shape != (len(x),) or not torch.is_floating_point(event_gate):
            raise ValueError("research GDR event gate must be floating [M]")
        if not bool(torch.isfinite(event_gate).all()) or bool(
            ((event_gate != 0) & (event_gate != 1)).any()
        ):
            raise ValueError("headline GDR stream event gate must be binary")
        streams = len(x_offsets) - 1
        expected_state = (
            streams,
            int(self.base._num_heads),
            int(self.base._attention_dim),
            int(self.base._linear_dim),
        )
        if initial_state is not None:
            if initial_state.shape != expected_state:
                raise ValueError(
                    f"research GDR initial state must have shape {expected_state}"
                )
            if initial_state.dtype != torch.float32:
                raise ValueError("research GDR initial state must be FP32")
        u, q, k, v, decay_logits, beta_logits = self._project(x)
        projected = self._kernel_input_type(
            q=q,
            k=k,
            v=v,
            decay_logits=decay_logits,
            beta_logits=beta_logits,
            log_decay_scale=self.gdr_log_decay_scale,
            decay_bias=self.gdr_decay_bias,
            offsets=x_offsets,
            offsets_cpu=x_offsets_cpu,
            event_gate=event_gate,
        )
        kernel = (
            self.reference_kernel
            if self.kernel_backend == "reference"
            else self.fla_kernel
        )
        result = kernel(
            projected,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        if result.final_state is not None and result.final_state.dtype != torch.float32:
            raise MetaBridgeError("canonical research GDR final state is not FP32")
        self.last_final_state = result.final_state
        attention = result.context.reshape(
            -1, int(self.base._num_heads) * int(self.base._linear_dim)
        )
        if bool(self.base._concat_ua):
            normalized = self.base._norm_attn_output(attention)
            output_input = torch.cat((u, attention, u * normalized), dim=-1)
        else:
            output_input = u * self.base._norm_attn_output(attention)
        output = self.base._o(
            F.dropout(
                output_input,
                p=float(self.base._dropout_ratio),
                training=self.training,
            )
        ) + x
        return output, result.final_state

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor | None,
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Any = None,
        return_cache_states: bool = False,
        gdr_event_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if gdr_event_gate is not None:
            raise ValueError("CWI event intervention uses forward_gdr_streams with its explicit gradient contract")
        if self.mode == "passthrough":
            return self.base(
                x=x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                delta_x_offsets=delta_x_offsets,
                cache=cache,
                return_cache_states=return_cache_states,
            )
        if delta_x_offsets is not None or cache is not None:
            raise NotImplementedError(
                "research Full-GDR uses its versioned state cache, never HSTU K/V delta cache"
            )
        if return_cache_states:
            raise NotImplementedError(
                "research Full-GDR cache export must use the versioned GDR cache adapter"
            )
        if x.ndim != 2 or x_offsets.ndim != 1:
            raise ValueError("research Full-GDR expects jagged x [M,D] and offsets [B+1]")
        if int(x_offsets[0]) != 0 or int(x_offsets[-1]) != len(x) or bool(
            (x_offsets[1:] < x_offsets[:-1]).any()
        ):
            raise ValueError("research Full-GDR offsets do not span packed x")
        if invalid_attn_mask.ndim != 2 or invalid_attn_mask.shape[0] != (
            invalid_attn_mask.shape[1]
        ):
            raise ValueError("official HSTU attention mask must be square")
        del all_timestamps  # GDR replaces the relative-time attention aggregation.
        output, _ = self.forward_gdr_streams(
            x=x,
            x_offsets=x_offsets,
            event_gate=torch.ones(len(x), dtype=x.dtype, device=x.device),
            initial_state=None,
            return_final_state=False,
        )
        # HSTUJagged ignores the cache tuple unless explicitly requested.
        return output, (None, None, None, output)

def install_research_hstu_gdr(
    model: nn.Module,
    *,
    seed: int,
    kernel_backend: str,
    decay_timescale_range: Sequence[float],
    mode: str = "gdr",
) -> dict[str, Any]:
    """Replace attention in place and prove every official tensor is retained."""

    contract = audit_research_hstu_contract(model)
    before = _canonical_official_state(model)
    before_hash = _state_sha256(before)
    layers = _research_layers(model)
    wrappers = nn.ModuleList()
    for index, layer in enumerate(layers):
        wrapper = ResearchHSTUGDRLayer(
            layer,
            seed=seed + index,
            kernel_backend=kernel_backend,
            mode=mode,
        )
        wrapper.set_decay_timescales(
            decay_timescale_range, seed=seed + 71 + index
        )
        wrappers.append(wrapper)
    model._hstu._attention_layers = wrappers
    after = _canonical_official_state(model)
    if set(before) != set(after):
        raise MetaBridgeError(
            "official tensor set changed while installing research HSTU GDR"
        )
    mismatched = [name for name in before if not torch.equal(before[name], after[name])]
    if mismatched:
        raise MetaBridgeError(
            f"official tensors changed while installing research HSTU GDR: {mismatched[:3]}"
        )
    after_hash = _state_sha256(after)
    if before_hash != after_hash:
        raise MetaBridgeError("official state hash changed during GDR replacement")
    new_parameters = {
        name: {
            "shape": list(value.shape),
            "sha256": _tensor_sha256(value),
        }
        for name, value in model.named_parameters()
        if any(
            name.endswith(suffix)
            for suffix in (
                ".gdr_log_decay_scale",
                ".gdr_decay_bias",
                ".gdr_gate_weight",
            )
        )
    }
    if len(new_parameters) != 3 * len(layers):
        raise MetaBridgeError("research HSTU GDR did not register exactly three tensors/layer")
    return {
        "schema": REPLACEMENT_SCHEMA,
        "mode": mode,
        "kernel_backend": kernel_backend,
        "decay_timescale_range": [float(x) for x in decay_timescale_range],
        "official_tensor_count": len(before),
        "official_state_sha256_before": before_hash,
        "official_state_sha256_after": after_hash,
        "official_state_bitwise_preserved": True,
        "new_gdr_parameters": new_parameters,
        "contract_audit": contract,
    }

def set_research_hstu_replacement_mode(model: nn.Module, mode: str) -> None:
    layers = _research_layers(model)
    if not all(isinstance(layer, ResearchHSTUGDRLayer) for layer in layers):
        raise MetaBridgeError("research HSTU GDR replacement is not installed")
    for layer in layers:
        layer.set_mode(mode)
