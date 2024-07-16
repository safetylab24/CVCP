import logging
from models.centerpoint.det3d.models.bbox_heads.center_head import CenterHead

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from models.cvt.encoder import CVTEncoder

from collections import OrderedDict

torch.set_float32_matmul_precision('high')

# creates CenterHead part of the model
def head(in_channels, tasks, dataset, weight, code_weights, common_heads, share_conv_channel, dcn_head=False):
    # Initialize the logger
    logger = logging.getLogger("CenterHead")
    logging.basicConfig(level=logging.INFO)

    # Initialize the CenterHead model
    center = CenterHead(
        in_channels=sum(in_channels),
        tasks=tasks,
        code_weights=code_weights,
        common_heads=common_heads,
        share_conv_channel=share_conv_channel,
        logger=logger,
        dataset=dataset,
        weight=weight,
        dcn_head=dcn_head
    )

    return center


class CVCPModel(L.LightningModule):
    def __init__(self, cvt_encoder: CVTEncoder, center_head_model: CenterHead, resize_shape, cfg):
        super().__init__()
        self.cvt_encoder = cvt_encoder
        self.head_model = center_head_model
        self.resize_shape = resize_shape
        self.cfg = cfg

    # expects data as dict of images, intrinsics, extrinsics, labels
    def training_step(self, batch, batch_idx):
        # (1, 128, 25, 25), (B, dim, H, W)
        cvt_out = self.cvt_encoder(
            batch['images'], batch['intrinsics'], batch['extrinsics'])
        # Resize the output of cvt_model
        cvt_out = F.interpolate(
            cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        # Pass the resized output through the head model
        # Returns predictions for each task
        preds = self.head_model(cvt_out)

        # Compute the loss
        losses = self.head_model.loss(batch['labels'], preds)
        loss, log_vars = self.parse_second_losses(losses)
        self.log("train_loss", loss)
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch, batch_idx):
        cvt_out = self.cvt_encoder(
            batch['images'], batch['intrinsics'], batch['extrinsics'])
        cvt_out = F.interpolate(
            cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        preds = self.head_model(cvt_out)

        # Compute the loss
        losses = self.head_model.loss(batch['labels'], preds)
        loss, log_vars = self.parse_second_losses(losses)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg['lr'], total_steps=self.trainer.estimated_stepping_batches, max_momentum=self.cfg['max_momentum'], base_momentum=self.cfg['base_momentum']),
                'monitor': 'val_loss',
                'frequency': 1
            }
        }

    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses["loss"])
        for loss_name, loss_value in losses.items():
            if loss_name == "loc_loss_elem":
                log_vars[loss_name] = [[i.item() for i in j]
                                       for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars
