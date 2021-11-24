import torch
import typing as T


class SigmaNet(torch.nn.Module):
    def __init__(
        self,
        nn: torch.nn.Module,  # Must already be sigma lipschitz
        sigma: float,
        monotone_constraints: T.Optional[T.Iterable] = None,
    ):
        super().__init__()
        self.nn = nn
        self.register_buffer("sigma", torch.Tensor([sigma]))
        self.monotone_constraint = monotone_constraints or [1]
        self.monotone_constraint = torch.tensor(self.monotone_constraint).float()

    def forward(self, x: torch.Tensor):
        return self.nn(x) + self.sigma * (
            x * self.monotone_constraint.to(x.device)
        ).sum(axis=-1, keepdim=True)
