import torch.nn as nn

from models.cvt.encoder import *

class Shrink(nn.Module):
    def __init__(self):
        super(Shrink, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        dim_last=64,
        n_classes=2,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.encoder = Encoder()


    def forward(self, images, intrinsics, extrinsics):
        x, atts = self.encoder(images, intrinsics, extrinsics)

        return x
