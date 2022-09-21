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
                Montonically deacreasing features should have value -1 instead of 1.
            sigma (float, optional): Lipschitz constant of the network given in nn. Defaults to 1.
                """
        super().__init__()
        self.nn = nn
        self.register_buffer("sigma", torch.Tensor([sigma]))
        if monotone_constraints is None:
          monotone_constraints = [1]
        self.register_buffer(
            "monotone_constraints", torch.tensor(monotone_constraints).float()
        )

    def forward(self, x: torch.Tensor):
        return self.nn(x) + self.sigma * (x * self.monotone_constraints).sum(
            axis=-1, keepdim=True
        )
