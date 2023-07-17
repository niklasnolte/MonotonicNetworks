import torch
import typing as T
from .functional import direct_norm
import warnings
from torch.nn import Module, Parameter, Linear


class MonotonicWrapper(Module):
    def __init__(
        self,
        lipschitz_module: Module,  # Must already be lipschitz
        lipschitz_const: float = 1.0,
        monotonic_constraints: T.Optional[T.Iterable] = None,
    ):
        """This is a wrapper around a module with a lipschitz_const lipschitz constant. It
            adds a term to the output of the module which enforces monotonicity constraints
            given by monotonic_constraints.
            Returns a module which is monotonic and Lipschitz with constant lipschitz_const.

        Args:
            lipschitz_module (torch.nn.Module): Lipschitz-constrained nn.Module with Lipschitz
                constant lipschitz_const.
            lipschitz_const (float, optional): Lipschitz constant of the network given in nn.
                Defaults to 1.
            monotonic_constraints (torch.Tensor, optional): Monotonicity constraints on the inputs.
                If None, all inputs are assumed to be monotonically increasing.
                If monotonic_constraints[i] = 1, the i-th input is constrained to be non-decreasing.
                If monotonic_constraints[i] = -1, the i-th input is constrained to be
                non-increasing.
                If monotonic_constraints[i] = 0, the i-th input is unconstrained.

                If a 1D tensor, the same constraint is applied to all ouputs.
                If a 2D tensor, the (i, j)-th element specifies the constraint on the j-th output
                with respect to the i-th input.
                Default: ``None``
        """
        super().__init__()
        self.nn = lipschitz_module
        self.register_buffer(
            "lipschitz_const", torch.Tensor([lipschitz_const])
        )
        if monotonic_constraints is None:
            monotonic_constraints = [1]
        monotonic_constraints = torch.Tensor(monotonic_constraints)
        if monotonic_constraints.ndim == 1:
            monotonic_constraints = monotonic_constraints.unsqueeze(-1)
        self.register_buffer("monotonic_constraints", monotonic_constraints)

    def forward(self, x: torch.Tensor):
        mc = self.monotonic_constraints.expand(x.shape[1], -1)  # type: ignore
        residual = self.lipschitz_const * x @ mc
        return self.nn(x) + residual


class LipschitzLinear(Linear):
    """
    A linear layer with a Lipschitz constraint on its weights.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        lipschitz_const (float, optional): Lipschitz constant of the layer.
            Default: ``1``
        kind (str, optional): Type of Lipschitz constraint to enforce. Options are
            - "one",  # |W|_1 constraint
            - "inf",  # |W|_inf constraint
            - "one-inf",  # |W|_1,inf constraint
            - "two-inf",  # |W|_2,inf constraint
            Will be passed to :func:`direct_norm`, see its documentation for more details.
            Defaults to "one".
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lipschitz_const: float = 1.0,
        kind: str = "one-inf",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer(
            "lipschitz_const", torch.Tensor([lipschitz_const])
        )
        # Directly enforce Lipschitz constraint
        self = direct_norm(self, max_norm=lipschitz_const, kind=kind)


class MonotonicLayer(LipschitzLinear):
    """
    A linear layer with a Lipschitz constraint on its weights and monotonicity constraints.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        lipschitz_const (float, optional): Lipschitz constant of the layer.
            Default: ``1``
        monotonic_constraints (torch.Tensor, optional): Monotonicity constraints on the inputs.
            If None, all inputs are assumed to be monotonically increasing.
            If monotonic_constraints[i] = 1, the i-th input is constrained to be non-decreasing.
            If monotonic_constraints[i] = -1, the i-th input is constrained to be non-increasing.
            If monotonic_constraints[i] = 0, the i-th input is unconstrained.

            If a 1D tensor, the same constraint is applied to all ouputs.
            If a 2D tensor, the (i, j)-th element specifies the constraint on the j-th output
            with respect to the i-th input.
            Default: ``None``
        kind (str, optional): Type of Lipschitz constraint to enforce. Default: ``"one-inf"``
            Options are
            - "one",  # |W|_1 constraint
            - "inf",  # |W|_inf constraint
            - "one-inf",  # |W|_1,inf constraint
            - "two-inf",  # |W|_2,inf constraint
            Will be passed to :func:`direct_norm`, see its documentation for more details.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lipschitz_const: float = 1.0,
        monotonic_constraints: T.Optional[T.Iterable] = None,
        kind: str = "one-inf",
    ):
        super().__init__(
            in_features, out_features, bias, lipschitz_const, kind
        )
        self.register_buffer(
            "lipschitz_const", torch.tensor([lipschitz_const])
        )
        if monotonic_constraints is None:
            monotonic_constraints = torch.ones(in_features)
        else:
            monotonic_constraints = torch.Tensor(monotonic_constraints)

        if monotonic_constraints.shape not in [
            (in_features, out_features),
            (in_features, 1),
            (in_features,),
        ]:
            raise ValueError(
                "monotonic_constraints must be of shape (in_features, out_features),"
                " ,(in_features, 1), or (in_features,)"
                f" got {monotonic_constraints.shape}"
            )

        if monotonic_constraints.ndim == 1:
            monotonic_constraints = monotonic_constraints.unsqueeze(-1)
        self.register_buffer("monotonic_constraints", monotonic_constraints)

    def forward(self, x: torch.Tensor):
        residual = self.lipschitz_const * x @ self.monotonic_constraints
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        return (x + residual) / 2


class RMSNorm(Module):
    def __init__(
        self, norm_shape: T.Union[T.Iterable, int], affine: bool = True
    ):
        super().__init__()
        self.register_buffer("norm_shape", torch.tensor(norm_shape))
        weights = torch.ones(norm_shape) / self.norm_shape.sqrt()  # type: ignore
        self.register_parameter("weight", Parameter(weights, affine))
        bias = torch.zeros(norm_shape)  # type: ignore
        self.register_parameter("bias", Parameter(bias, affine))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)).clip(min=1)
        return (x / rms) * self.weight.pow(2) + self.bias  # type: ignore


class SigmaNet(MonotonicWrapper):
    def __init__(
        self,
        nn: Module,  # Must already be sigma lipschitz
        sigma: float,
        monotonic_constraints: T.Optional[T.Iterable] = None,
    ):
        """Implementation of a monotone network with a sigma lipschitz constraint.

        Args:
            nn (torch.nn.Module): Lipschitz-constrained network with Lipschitz
                constant sigma.
            monotonic_constraints (T.Optional[T.Iterable], optional): Iterable of the
                monotonic features. For example, if a network
                which takes a vector of size 3 is meant to be monotonic in the last
                feature only, then monotonic_constraints should be [0, 0, 1].
                Defaults to all features (i.e. a vector of ones everywhere).
                Montonically deacreasing features should have value -1 instead of 1.
            sigma (float, optional): Lipschitz constant of the network given in nn. Defaults to 1.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("default", DeprecationWarning)
            warnings.warn(
                "SigmaNet is deprecated, use MonotonicWrapper instead",
                DeprecationWarning,
            )
        super().__init__(nn, sigma, monotonic_constraints)
