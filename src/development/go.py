"""
General Orthogonal group : 
A in GL_n such that : exists P orthogonal s.t. (PA)^T PA = D, where D is diagonal.
"""
import torch
from torch import nn


class go(nn.Module):
    def __init__(self, size):
        """
        go(n) lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (...,...,n,n).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """ parametrise general orthogonal lie algebra from the general linear matrix X

        Args:
            X (torch.tensor): (...,n,n)
        

        Returns:
            torch.tensor: (...,n,n)
        """

        X = X.tril()
        X = X - X.tril(-1).transpose(-2, -1)

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not sqaured matrix')
        return self.frame(X)

    @ staticmethod
    def in_lie_algebra(X, eps=1e-5):
        X = X - torch.diagonal(X, dim1=-2, dim2=-1)
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(X.transpose(-2, -1), -X, atol=eps))

