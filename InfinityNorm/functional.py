from torch import nn
import torch

kinds = ["one", "inf", "one-inf"]


def direct_norm(
    m: nn.Module,
    kind="one",
    always_norm=True,
    alpha=None,
    name="weight",
    vectorwise=True,
) -> nn.Module:
    if kind not in kinds:
        raise ValueError(f"kind {kind} not recognized. Choose one of {kinds}")

    def normalize_weight(m: nn.Module, _) -> None:
        weight = getattr(m, name + "_orig")
        weight = _get_normed_weights(weight, kind, always_norm, alpha, vectorwise)
        setattr(m, name, weight)

    w = m._parameters[name]
    delattr(m, name)
    m.register_parameter(name + "_orig", w)
    setattr(m, name, w.data)
    m.register_forward_pre_hook(normalize_weight)
    return m


def project_norm(
    m: nn.Module,
    kind="one",
    always_norm=True,
    alpha=None,
    name="weight",
    vectorwise=True,
) -> nn.Module:
    if kind not in kinds:
        raise ValueError(f"kind {kind} not recognized. Choose one of {kinds}")

    @torch.no_grad()
    def normalize_weight(m: nn.Module, _) -> None:
        weight = getattr(m, name).detach()
        weight = _get_normed_weights(weight, kind, always_norm, alpha, vectorwise)
        getattr(m, name).copy_(weight)

    m.register_forward_pre_hook(normalize_weight)
    return m


def _get_normed_weights(weight, kind, always_norm, alpha, vectorwise):
    if kind == "one":
        norms = weight.abs().sum(axis=0)
    elif kind == "inf":
        norms = weight.abs().sum(axis=1)
    elif kind == "one-inf":
        norms = weight.abs().amax(dim=0)
    if not vectorwise:
        norms = norms.max()

    alpha = alpha or 1

    if not always_norm:
        norms = torch.max(torch.ones_like(norms), norms / alpha)
    else:
        norms = norms / alpha

    if kind == "inf" and vectorwise:
        # if the sum was over axis 1, the shape is different
        weight = (weight.T / norms).T
    else:
        weight = weight / norms

    return weight
