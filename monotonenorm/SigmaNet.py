import torch
import typing as T


class SigmaNet(torch.nn.Module):
    def __init__(
        self,
        nn: torch.nn.Module,  # Must already be sigma lipschitz
        sigma: float,
        monotone_constraints: T.Optional[T.Iterable] = None,
    ):
        """ Implementation of a monotone network with a sigma lipschitz constraint.

        Args:
            nn (torch.nn.Module): Lipschitz-constrained network with Lipschitz
                constant sigma.
            monotone_constraints (T.Optional[T.Iterable], optional): Iterable of the
                monotonic features. For example, if a network
                which takes a vector of size 3 is meant to be monotonic in the last
                feature only, then monotone_constraints should be [0, 0, 1].
                Defaults to all features (i.e. a vector of ones everywhere).
                """
        super().__init__()
        self.nn = nn
        self.register_buffer("sigma", torch.Tensor([sigma]))
        self.monotone_constraint = monotone_constraints or [1]
        self.monotone_constraint = torch.tensor(self.monotone_constraint).float()

    def forward(self, x: torch.Tensor):
        return self.nn(x) + self.sigma * (
            x * self.monotone_constraint.to(x.device)
        ).sum(axis=-1, keepdim=True)
