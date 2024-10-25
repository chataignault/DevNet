from torch import nn
from torch import Tensor
from abc import ABC, abstractmethod


class param(nn.Module):
    @abstractmethod
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    @abstractmethod
    def frame(self, X: Tensor) -> Tensor:
        pass
