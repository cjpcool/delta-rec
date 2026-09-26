def nonfinite_gradient_names(named_parameters):
    import torch
    groups = {}
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        groups.setdefault(values.device, []).append((name, torch.isfinite(values).all()))
    bad = []
    for entries in groups.values():
        flags = torch.stack([flag for _, flag in entries])
        if not bool(flags.all()):
            bad.extend(name for (name, _), finite in zip(entries, flags.cpu().tolist()) if not finite)
    return bad
