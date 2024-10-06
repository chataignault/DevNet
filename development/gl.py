import torch
from torch import nn


class gl(nn.Module):
    def __init__(self, size):
        """
        gl(n) lie algebra matrices, parametrized 
        by a general linear matrix with shape (...,...,n,n).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """ parametrise geleral linear lie algebra from the gneal linear matrix X

        Args:
            X (torch.tensor): (...,n,n)

        Returns:
            torch.tensor: (...,n,n)
        """
        N, C, m, m = X.shape
    
        return X


    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not squared matrix')
        return self.frame(X)
    