from torch import nn


def infnorm(m: nn.Module, name='weight') -> nn.Module:
    def absi(m: nn.Module, _) -> None:
        weight = getattr(m, name + '_orig')
        weight = weight / weight.abs().sum(axis=0)
        setattr(m, name, weight)

    w = m._parameters[name]
    delattr(m, name)
    m.register_parameter(name + "_orig", w)
    setattr(m, name, w.data)
    m.register_forward_pre_hook(absi)
    return m
