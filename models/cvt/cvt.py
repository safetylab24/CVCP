import torch.nn as nn


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

    def forward(self, images, intrinsics, extrinsics):
        x = self.encoder(images, intrinsics, extrinsics)
        y = self.decoder(x)

        return y