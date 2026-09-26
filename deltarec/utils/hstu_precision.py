from __future__ import annotations

def ieee_gdr_solve():
    from deltarec.layers.gdr_kernels import _import_fla_chunk_gated_delta_rule
    _import_fla_chunk_gated_delta_rule()
    import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
    import triton.language as tl
    chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')
