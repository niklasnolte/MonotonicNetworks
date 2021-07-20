from torch import nn
import torch


def infnorm(m: nn.Module, always_norm=True, name="weight") -> nn.Module:
    def absi(m: nn.Module, _) -> None:
        weight = getattr(m, name + "_orig")
        norms = weight.abs().sum(axis=0)
        if always_norm:
            weight = weight / norms
        else:
            weight = weight / torch.max(torch.ones_like(norms), norms)
        setattr(m, name, weight)

    w = m._parameters[name]
    delattr(m, name)
    m.register_parameter(name + "_orig", w)
    setattr(m, name, w.data)
    m.register_forward_pre_hook(absi)
    return m
