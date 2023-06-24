import torch
import typing as T
from .functional import direct_norm


class MonotonicWrapper(torch.nn.Module):
    def __init__(
        self,
        lipschitz_module: torch.nn.Module,  # Must already be sigma lipschitz
        sigma: float = 1,
        monotone_constraints: T.Optional[T.Iterable] = None,
    ):
        """Implementation of a monotone network with a sigma lipschitz constraint.

        Args:
            lipschitz_module (torch.nn.Module): Lipschitz-constrained nn.Module with Lipschitz
                constant sigma.
            sigma (float, optional): Lipschitz constant of the network given in nn. Defaults to 1.
            monotone_constraints (T.Optional[T.Iterable], optional): Iterable of the
                monotonic features. For example, if a network
                which takes a vector of size 3 is meant to be monotonic in the last
                feature only, then monotone_constraints should be [0, 0, 1].
                Defaults to all features (i.e. a vector of ones everywhere).
                Montonically deacreasing features should have value -1 instead of 1.
        """
        super().__init__()
        self.nn = lipschitz_module
        self.register_buffer("sigma", torch.Tensor([sigma]))
        if monotone_constraints is None:
            monotone_constraints = [1]
        monotone_constraints = torch.Tensor(monotone_constraints)
        self.register_buffer("monotone_constraints", monotone_constraints)

    def forward(self, x: torch.Tensor):
        return self.nn(x) + self.sigma * (x * self.monotone_constraints).sum(  # type: ignore
            axis=-1, keepdim=True
        )


class LipschitzLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma=1):
        super().__init__()
        self.register_buffer("sigma", torch.Tensor([sigma]))
        # Directly enforce Lipschitz constraint
        self.lipschitz_linear = direct_norm(
            torch.nn.Linear(in_features, out_features, bias=bias),
            max_norm=sigma,
        )

    def forward(self, x):
        return self.lipschitz_linear(x)


class MonotonicLayer(LipschitzLayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        sigma=1,
        monotone_constraints: T.Optional[T.Iterable] = None,
    ):
        super().__init__(in_features, out_features, bias, sigma)
        self.register_buffer("sigma", torch.Tensor([sigma]))
        if monotone_constraints is None:
            monotone_constraints = [1] * in_features
        monotone_constraints = torch.Tensor(monotone_constraints)
        if monotone_constraints.ndim == 1:
            monotone_constraints = monotone_constraints.repeat(
                out_features, 1
            ).T
        if monotone_constraints.shape[0] != in_features:
            raise ValueError(
                f"monotone_constraints must be of length {in_features},"
                f" got {monotone_constraints.shape[0]}"
            )
        if monotone_constraints.shape[1] != out_features:
            raise ValueError(
                "monotone_constraints must be of shape (in_features, out_features),"
                f" got {monotone_constraints.shape}"
            )

        self.register_buffer("monotone_constraints", monotone_constraints)

    def forward(self, x: torch.Tensor):
        residual = self.sigma * (
            x.unsqueeze(-1) * self.monotone_constraints
        ).sum(axis=1)
        return (super().forward(x) + residual) / 2
