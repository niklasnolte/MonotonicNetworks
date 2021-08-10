import torch
from torch import nn
import typing as T


class SigmaNet(nn.Module):
    def __init__(
            self,
            nn: nn.Module,  # Must already be sigma lipschitz
            sigma: float,
            monotone_constraints: T.Optional[T.Iterable] = None,
            gamma: T.Optional[float] = 1
    ):
        super().__init__()
        self.nn = nn
        self.register_buffer("gamma", torch.Tensor([gamma]))
        self.register_buffer("sigma", torch.Tensor([sigma]))
        self.monotone_constraint = monotone_constraints or [1]
        self.monotone_constraint = torch.tensor(self.monotone_constraint).float()

    def forward(self, x):
        return (self.nn(x) + self.sigma * self.gamma * (x * self.monotone_constraint.to(x.device)).sum(
            axis=-1, keepdim=True))
