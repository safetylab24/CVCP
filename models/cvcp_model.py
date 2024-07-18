import logging
from models.centerpoint.center_head import CenterHead

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from models.cvt.encoder import CVTEncoder

from collections import OrderedDict

from metrics.iou_3d import IoU3D
from metrics.mAP_3d import MAP3D

torch.set_float32_matmul_precision('high')

# creates CenterHead part of the model


def head(in_channels, tasks, dataset, weight, code_weights, common_heads, share_conv_channel, dcn_head=False):
    # Initialize the logger
    logger = logging.getLogger('CenterHead')
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
        self.max_conf_val = 0

        self.iou = IoU3D(num_classes=len(cfg['centerpoint']['tasks']))
        # self.mAP = MAP3D(iou_threshold=0.1)

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
        if (batch_idx + 1) % self.cfg['log_every_n_steps'] == 0:
            self.log('train_loss', loss, sync_dist=True,
                     prog_bar=True, on_step=True, logger=True)
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
        if (batch_idx + 1) % self.cfg['log_every_n_steps'] == 0:
            self.log('val_loss', loss, sync_dist=True,
                     prog_bar=True, on_step=True, logger=True)

        # TODO: fix this
        #prediction = self.head_model.predict(batch['labels'], preds, self.cfg['test'])
        #for task in prediction:
         #   if task['scores'].max() > self.max_conf_val:
          #      self.max_conf_val = task['scores'].max().item()
        #self.iou.update(prediction, batch['labels_original'])
        # self.mAP.update(prediction, batch['labels_original'])
        #self.iou.compute()

        return loss

    def test_step(self, batch, batch_idx):
        cvt_out = self.cvt_encoder(
            batch['images'], batch['intrinsics'], batch['extrinsics'])
        cvt_out = F.interpolate(
            cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        preds_raw = self.head_model(cvt_out)

        preds_final = self.head_model.predict(preds_raw, self.cfg['test'])

        self.iou.update(preds_final, batch['labels_original'])
        self.log('iou', self.iou.compute())
        # self.log('mAP', self.mAP.compute())
        # self.log('ciou', self.ciou.compute())
        losses = self.head_model.loss(batch['labels'], preds_raw)
        loss, log_vars = self.parse_second_losses(losses)
        self.log('test_loss', loss, sync_dist=True,
                 batch_size=batch['images'].shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.cfg['lr'],
                    total_steps=self.trainer.estimated_stepping_batches,
                    max_momentum=self.cfg['max_momentum'],
                    base_momentum=self.cfg['base_momentum']),
                'monitor': 'val_loss',
                'frequency': 1,
                'interval': 'step',
            }
        }

    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses['loss'])
        for loss_name, loss_value in losses.items():
            if loss_name == 'loc_loss_elem':
                log_vars[loss_name] = [[i.item() for i in j]
                                       for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars
