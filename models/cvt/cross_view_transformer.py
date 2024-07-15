import torch.nn as nn

from models.cvt.encoder import *

class CrossViewTransformer(nn.Module):
    def __init__(self ):
        super().__init__()
        self.encoder = Encoder()


    def forward(self, images, intrinsics, extrinsics):
        x, _ = self.encoder(images, intrinsics, extrinsics)

        return x
