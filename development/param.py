from torch import nn
from torch import Tensor
from abc import ABC, abstractmethod

class param(nn.Module, ABC):
    @abstractmethod
    def __init__(self, size: int):
        super().__init__()
        pass

    @abstractmethod
    def frame(self, X: Tensor) -> Tensor:
        pass