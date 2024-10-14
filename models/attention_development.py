import sys
import os

sys.path.append(os.path.join(os.getcwd(), ".."))

from development.nn import development_layer
from development.param import param
from development.so import so


from torch import nn
from torch import Tensor
from torch.nn import MultiheadAttention

class AttentionDevelopment(nn.Module):

    def __init__(self, 
                 embed_dim:int,
                 dropout:float,
                 num_heads:int,
                 input_size:int,
                 hidden_size:int,
                 channels:int,
                 param:param
                 ):
        self.attention = MultiheadAttention(
            embed_dim=embed_dim,
            dropout=dropout,
            num_heads=num_heads,
        )
        self.development = development_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            channels=channels,
            param=param
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.attention(x)
        x = self.development(x)
        return x

if __name__ == "__main__":
    # TODO run some tests to instantiate and forward / backward of the model
    model = AttentionDevelopment(
        dropout=.1,
        embed_dim=4,
        num_heads=2,
        input_size=3,
        hidden_size=2,
        channels=2,
        param=so,
    )
