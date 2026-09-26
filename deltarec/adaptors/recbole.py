"""LinRec and Blossom network components required by DeltaRec."""
import copy
from types import SimpleNamespace
from dataclasses import dataclass, asdict
import torch
from torch import nn
from deltarec.layers.linrec_attention import TransformerEncoder
from deltarec.layers.blossom_feedforward import FeedForward
from deltarec.metrics.ranking import KUAI_TASKS
COMMON_MAX_HISTORY_LENGTH = 1024
RecBoleBridgeError = ValueError

@dataclass(frozen=True)
class RecBoleModelConfig:
    method: str
    n_layers: int = 2
    n_heads: int = 8
    hidden_size: int = 128
    inner_size: int = 256
    hidden_dropout_prob: float = .5
    attn_dropout_prob: float = .5
    hidden_act: str = 'gelu'
    layer_norm_eps: float = 1e-12
    initializer_range: float = .02
    loss_type: str = 'CE'
    max_history_length: int = 1024
    multitask: bool = False
    def fingerprint(self):
        from deltarec.utils.io import protocol_hash
        return protocol_hash(asdict(self))

class DeltaRecShell(nn.Module):
    def __init__(self, c, max_item_id):
        super().__init__()
        self.n_heads = c.n_heads
        self.item_embedding = nn.Embedding(max_item_id + 1, c.hidden_size, padding_idx=0)
        if c.method == 'linrec':
            self.position_embedding = nn.Embedding(c.max_history_length, c.hidden_size)
        if c.method == 'linrec':
            self.trm_encoder = TransformerEncoder(c.n_layers,c.n_heads,c.hidden_size,c.inner_size,c.hidden_dropout_prob,c.attn_dropout_prob,c.hidden_act,c.layer_norm_eps)
        elif c.method == 'blossomrec':
            attention = nn.Module()
            attention.dense = nn.Linear(c.hidden_size, c.hidden_size)
            attention.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
            attention.out_dropout = nn.Dropout(c.hidden_dropout_prob)
            attention.sparse_attention = _DiscardedBlossomInitialization(c.hidden_size, c.n_heads)
            layer = nn.Module()
            layer.blossom_attention = attention
            layer.feed_forward = FeedForward(c.hidden_size,c.inner_size,c.hidden_dropout_prob,c.hidden_act,c.layer_norm_eps)
            self.trm_encoder = nn.Module()
            self.trm_encoder.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(c.n_layers)])
        else:
            raise ValueError('Unsupported DeltaRec shell')
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.dropout = nn.Dropout(c.hidden_dropout_prob)
        def init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0., std=c.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_(); m.weight.data.fill_(1.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        self.apply(init)

def build_upstream_model(config, *, max_item_id, source_checkout=None, device='cpu', fixture_only=False):
    return DeltaRecShell(config, max_item_id).to(device)


class _DiscardedBlossomInitialization(nn.Module):
    """Preserve RNG draws of the replaced upstream attention during initialization.

    These parameters are dropped when DeltaRec takes the residual/FFN shell.
    Their constructors and traversal order match the pinned upstream constructor;
    no sparse-attention forward, trainer or evaluator is included.
    """
    def __init__(self, width, heads):
        super().__init__()
        dim=width//heads
        self.to_qkv=nn.Linear(width,width+4*dim,bias=False)
        compress=nn.Sequential(nn.Identity(),nn.Linear(32*dim,32*dim),nn.ReLU(),nn.Linear(32*dim,dim))
        self.k_compress=copy.deepcopy(compress)
        self.v_compress=copy.deepcopy(compress)
        strategy=nn.Linear(width,2*heads)
        nn.init.zeros_(strategy.weight)
        strategy.bias.data.copy_(torch.tensor([-2.,2.]*heads))
        self.to_strategy_combine=nn.Sequential(strategy,nn.Sigmoid(),nn.Identity())
        self.combine_heads=nn.Linear(width,width,bias=False)
