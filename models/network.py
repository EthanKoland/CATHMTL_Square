import torch
import torch.nn as nn

from .modules import *


class ODSNet(nn.Module):

    def __init__(self, in_channels, heads) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.encoder = Encoder()
        self.decoder = Decoder(heads)

    def forward(self, f):
        f16, f8, f4 = self.encoder(f)
        out = self.decoder(f16, f8, f4)

        return out
