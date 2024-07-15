import logging
from models.centerpoint.det3d.models.bbox_heads.center_head import CenterHead

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer

from models.cvt.model import CVTModel

from collections import OrderedDict

torch.set_float32_matmul_precision('medium')

# creates CVT head of the model
def cvt(devices=[0], loss_type='ce'):
    model = CVTModel(
        devices=devices,
        weights=torch.tensor([2., 1.])
    )

    return model

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


class CVCPModel(nn.Module):
    def __init__(self, cvt_model: CVTModel, head_model: CenterHead, resize_shape, opt: Optimizer=None, scheduler: lr_scheduler=lr_scheduler.OneCycleLR):
        super(CVCPModel, self).__init__()
        self.cvt_model = cvt_model.cuda(3)
        self.head_model = head_model.cuda(3)
        self.resize_shape = resize_shape
        self.opt = opt
        self.scheduler = scheduler

    def forward(self, images, intrinsics, extrinsics):
        # Pass input through the cvt model
        # (1, 128, 25, 25), (B, dim, H, W)
        cvt_out = self.cvt_model(images, intrinsics, extrinsics)

        # Resize the output of cvt_model
        cvt_out = F.interpolate(cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)

        # Pass the resized output through the head model
        # Returns predictions for each task, and a shared convolution output
        head_out = self.head_model(cvt_out)

        return head_out

    def train_step(self, data: dict):
        # expects data as dict of images, intrinsics, extrinsics, labels

        # Switch to training mode
        self.train()

        # Forward pass
        preds = self.forward(
            data['images'], data['intrinsics'], data['extrinsics'])

        # Compute the loss
        losses = self.head_model.loss(data['labels'], preds)
        
        loss, log_vars = self.parse_second_losses(losses)

        # Clear gradients
        self.opt.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the parameters + learning rate
        self.opt.step()
        self.scheduler.step()
        

        return loss.item()

    def val_step(self, data: dict):
        self.eval()

        preds = self.forward(data['images'], data['intrinsics'], data['extrinsics'])

        losses = self.head_model.loss(data['labels'], preds)
        
        loss, log_vars = self.parse_second_losses(losses)
        
        return loss.item()
    
    def predict(self, data, test_config) -> dict:
        self.eval()
        with torch.inference_mode():
            preds = self.forward(data['images'], data['intrinsics'], data['extrinsics'])
            return self.head_model.predict(data['labels'], preds, test_config)
        
    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses["loss"])
        for loss_name, loss_value in losses.items():
            if loss_name == "loc_loss_elem":
                log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars

class CVCPModelL(L.LightningModule):
    def __init__(self, cvt_model: CVTModel, center_head_model: CenterHead, resize_shape):
        super().__init__()
        self.cvt_model = cvt_model
        self.head_model = center_head_model
        self.resize_shape = resize_shape
    
    # expects data as dict of images, intrinsics, extrinsics, labels
    def training_step(self, batch, batch_idx):
        # (1, 128, 25, 25), (B, dim, H, W)
        cvt_out = self.cvt_model(batch['images'], batch['intrinsics'], batch['extrinsics'])
        # Resize the output of cvt_model
        cvt_out = F.interpolate(cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        # Pass the resized output through the head model
        # Returns predictions for each task
        preds = self.head_model(cvt_out)

        # Compute the loss
        losses = self.head_model.loss(batch['labels'], preds)
        loss, log_vars = self.parse_second_losses(losses)
        return loss
    
    def validation_step(self, batch, batch_idx):
        cvt_out = self.cvt_model(batch['images'], batch['intrinsics'], batch['extrinsics'])
        cvt_out = F.interpolate(cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        preds = self.head_model(cvt_out)

        # Compute the loss
        losses = self.head_model.loss(batch['labels'], preds)
        loss, log_vars = self.parse_second_losses(losses)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5),
                'monitor': 'val_loss',
                'frequency': 1
                }
            }
        
    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses["loss"])
        for loss_name, loss_value in losses.items():
            if loss_name == "loc_loss_elem":
                log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars