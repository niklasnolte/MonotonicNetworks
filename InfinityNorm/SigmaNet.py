import torch
from torch import nn
import typing as T


class SigmaNet(nn.Module):
    def __init__(
            self,
            nn: nn.Module,
            sigma: float,
            monotonic_in: T.Optional[T.Iterable] = None,
            nfeatures: T.Optional[int] = None,
    ):
        super().__init__()
        self.nn = nn
        self.bound = sigma
        if monotonic_in is not None:
            assert (
                    nfeatures is not None
            ), "if you want custom monotonicity, define how many features your data has via the nfeatures argument"
            # only be monotonic in these indices
            self.mask = torch.zeros(nfeatures).float()
            self.mask[monotonic_in] = 1.0
        else:
            # no mask then, monotonic in all features
            self.mask = torch.tensor([1]).float()

    def forward(self, x):
        return self.bound * (
                self.nn(x) + (x * self.mask.to(x.device)).sum(axis=-1, keepdim=True)
        )
