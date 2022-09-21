from torch import nn
import torch
from torch.nn.utils.parametrize import register_parametrization
import typing as t

kinds = [
    "one",  # |W|_1 constraint
    "inf",  # |W|_inf constraint
    "one-inf",  # |W|_1,inf constraint
    "two-inf",  # |W|_2,inf constraint
]


def direct_norm(
    layer: nn.Linear,
    kind: str = "one",
    always_norm: bool = False,
    max_norm: float = None,
    parameter_name: str = "weight",
    vectorwise: bool = True,
) -> nn.Linear:
    """
    Constrain the norm of a layer's weight matrix. This function is meant to be
    used as a wrapper around `torch.nn.Linear`. It adds a hook to the forward pass 
    to constrain the norm of the weight matrix. The hook is registered as a forward pre-hook.
    This means that the weights are normalized before the forward pass and that the gradients
    are backpropagated through normalization back to the original weights.


    Args:
        layer (nn.Linear): The layer to constrain.
        kind (str, optional): The kind of norm to constrain options are
             [
                "one",  # |W|_1 constraint
                "inf",  # |W|_inf constraint
                "one-inf",  # |W|_1,inf constraint
                "two-inf",  # |W|_2,inf constraint
            ]
            Defaults to "one".
        always_norm (bool, optional): Always normalize the weights to max_norm or allow them 
            to have a norm smaller than 1. Defaults to False.
        max_norm (float, optional): The maximum norm to constrain the weights to. Defaults to 1. 
        parameter_name (str, optional): Name of the weight matrix in the layer. Defaults to "weight".
        vectorwise (bool, optional): Normalize the matrix vectors instead of the matrix itself. 
            The vector norm is a weaker constraint and gives better results. See the paper for details.
            Defaults to True.
    Raises:
        ValueError: If kind is not one of the options.

    Returns:
        nn.Linear: The layer with the hook registered. Matrix multiplication will be done with 
            normalized version of the weights.
    """
    if kind not in kinds:
        raise ValueError(f"kind {kind} not recognized. Choose one of {kinds}")

    class Normalize(nn.Module):
      def forward(self, W):
        return get_normed_weights(W, kind, always_norm, max_norm, vectorwise)

    register_parametrization(layer, parameter_name, Normalize())

    return layer


def project_norm(
    layer: nn.Linear,
    kind: str = "one",
    always_norm: bool = True,
    max_norm: t.Optional[float] = None,
    parameter_name: str = "weight",
    vectorwise: bool = True,
) -> nn.Linear:
    """
    Constrain the norm of a layer's weight matrix by projecting the weight matrix to the correct norm.
    This function is meant to be used as a wrapper around `torch.nn.Linear`. It adds a hook to the forward pass
    which projects to the current weight matrix to a normalized matrix but the gradients are not backpropagated 
    through normalization. 

    Args:
        layer (nn.Linear): The layer to constrain.
        kind (str, optional): The kind of norm to constrain options are
             [
                "one",  # |W|_1 constraint
                "inf",  # |W|_inf constraint
                "one-inf",  # |W|_1,inf constraint
                "two-inf",  # |W|_2,inf constraint
            ]
            Defaults to "one".
        always_norm (bool, optional): Always normalize the weights to max_norm or allow them 
            to have a norm smaller than 1. Defaults to False.
        max_norm (float, optional): The maximum norm to constrain the weights to. Defaults to 1. 
        parameter_name (str, optional): Name of the weight matrix in the layer. Defaults to "weight".
        vectorwise (bool, optional): Normalize the matrix vectors instead of the matrix itself. 
            The vector norm is a weaker constraint and gives better results. See the paper for details.
            Defaults to True.
    Raises:
        ValueError: If kind is not one of the options.

    Returns:
        nn.Linear: The layer with the hook registered. Matrix multiplication will be done with 
            normalized version of the weights.
    """
    if kind not in kinds:
        raise ValueError(f"kind {kind} not recognized. Choose one of {kinds}")

    @torch.no_grad()
    def normalize_weight(layer: nn.Linear, _) -> None:
        weight = getattr(layer, parameter_name).detach()
        weight = get_normed_weights(weight, kind, always_norm, max_norm, vectorwise)
        getattr(layer, parameter_name).data.copy_(weight)

    layer.register_forward_pre_hook(normalize_weight)
    return layer


def get_normed_weights(
    weight: torch.Tensor,
    kind: str,
    always_norm: bool,
    max_norm: t.Optional[float],
    vectorwise: bool,
) -> torch.Tensor:
    """
    Normalize weight matrix to a given norm.

    Args:
        weight (torch.Tensor): The weight matrix to normalize.
        kind (str): The kind of norm to constrain options are
                [
                    "one",  # |W|_1 constraint
                    "inf",  # |W|_inf constraint
                    "one-inf",  # |W|_1,inf constraint
                    "two-inf",  # |W|_2,inf constraint
                ]
        always_norm (bool): Always normalize the weights to max_norm or allow them
            to have a norm smaller than 1.
        max_norm (t.Optional[float]): The maximum norm to constrain the weights to.
        vectorwise (bool): Normalize the matrix vectors instead of the matrix itself.

    Returns:
        torch.Tensor: The normalized weight matrix.
    """
    if kind == "one":
        norms = weight.abs().sum(axis=0)
    elif kind == "inf":
        norms = weight.abs().sum(axis=1, keepdim=True)
    elif kind == "one-inf":
        norms = weight.abs()
    elif kind == "two-inf":
        norms = torch.norm(weight, p=2, dim=1, keepdim=True)
    if not vectorwise:
        norms = norms.max()

    max_norm = max_norm or 1

    if not always_norm:
        norms = torch.max(torch.ones_like(norms), norms / max_norm)
    else:
        norms = norms / max_norm

    weight = weight / torch.max(norms, torch.ones_like(norms)*1e-10)

    return weight
