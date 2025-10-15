"""
Adapted from https://github.com/Lezcano/geotorch/blob/master/geotorch/so.py
"""

import torch
from torch import nn
from .param import param


class he(param):
    def __init__(self, size):
        """
        Heisenberg(n) lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (...,...,n,n).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__(size)

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """parametrise Heisenberg lie algebra from the general linear matrix X

        Args:
            X (torch.tensor): (...,n,n)


        Returns:
            torch.tensor: (...,n,n)
        """

        X = X.triu(1)

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError("weights has dimension < 2")
        if X.size(-2) != X.size(-1):
            raise ValueError("not sqaured matrix")
        return self.frame(X)
