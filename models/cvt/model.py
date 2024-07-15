import torch
import torch.nn as nn

from models.cvt.cross_view_transformer import CrossViewTransformer

class CVTModel(nn.Module):
    def __init__(self, devices, weights=None):
        super(CVTModel, self).__init__()

        self.device = devices[0]
        self.devices = devices

        self.weights = weights

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        self.backbone = None
        self.gamma = .1
        self.tsne = False
        
        self.m_in = -23.0
        self.m_out = -5.0

        # TODO: maybe change this to DistributedDataParallel
        self.backbone = nn.DataParallel(
                CrossViewTransformer().to(self.device),
                output_device=self.device,
                device_ids=self.devices)

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def train_step(self, images, intrinsics, extrinsics, labels):
        outs = self(images, intrinsics, extrinsics)
        preds = self.activate(outs)
        return outs, preds

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images, intrinsics, extrinsics)
