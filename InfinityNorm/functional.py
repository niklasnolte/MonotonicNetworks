from torch import nn
import torch

kinds = ["one", "inf", "one-inf"]


def divide_norm(m: nn.Module, kind="one", always_norm=True,
                alpha=None, name="weight") -> nn.Module:
    if kind not in kinds:
        raise ValueError(f"kind {kind} not recognized. Choose one of {kinds}")

    def normalize_weight(m: nn.Module, _) -> None:
        weight = getattr(m, name + "_orig")
        weight = _get_normed_weights(weight, kind, always_norm, alpha)
        setattr(m, name, weight)

    w = m._parameters[name]
    delattr(m, name)
    m.register_parameter(name + "_orig", w)
    setattr(m, name, w.data)
    m.register_forward_pre_hook(normalize_weight)
    return m


def project_norm(m: nn.Module, kind="one", always_norm=True,
                 alpha=None, name="weight") -> nn.Module:
    if kind not in kinds:
        raise ValueError(f"kind {kind} not recognized. Choose one of {kinds}")

    @torch.no_grad()
    def normalize_weight(m: nn.Module, _) -> None:
        weight = getattr(m, name).detach()
        weight = _get_normed_weights(weight, kind, always_norm, alpha)
        getattr(m, name).copy_(weight)

    m.register_forward_pre_hook(normalize_weight)
    return m


def _get_normed_weights(weight, kind, always_norm, alpha):
    if kind == "one":
        norms = weight.abs().sum(axis=0)
    elif kind == "inf":
        norms = weight.abs().sum(axis=1)
    elif kind == "one-inf":
        norms = weight.abs().max()
    if alpha is None:
        alpha = 1
    if always_norm:
        weight = weight / norms
    else:
        weight = weight / torch.max(torch.ones_like(norms), norms / alpha)
    return weight
