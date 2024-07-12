import logging
from CVCP.models.centerpoint.det3d.models.bbox_heads.center_head import CenterHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cvt.model import CVTModel


# creates CVT head of the model
def cvt(devices=[0], n_classes=2, loss_type='ce'):
    model = CVTModel(
        devices=devices,
        n_classes=n_classes,
        loss_type=loss_type,
        weights=torch.tensor([2., 1.])
    )

    return model

# creates CenterHead part of the model


def head(in_channels, tasks, dataset, weight, code_weights, common_heads, share_conv_channel, dcn_head=False):
    # Define tasks
    # tasks = [
    #     dict(num_class=1, class_names=["car"]),
    #     # dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    #     # dict(num_class=2, class_names=["bus", "trailer"]),
    #     # dict(num_class=1, class_names=["barrier"]),
    #     # dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    #     # dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    # ]

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
    def __init__(self, cvt_model, head_model: CenterHead, resize_shape, opt: torch.optim.Optimizer=None):
        super(CVCPModel, self).__init__()
        self.cvt_model = cvt_model
        self.head_model = head_model
        self.resize_shape = resize_shape
        self.opt = opt
        # self.loss = loss

    def forward(self, images, intrinsics, extrinsics):
        # Pass input through the cvt model
        # (1, 128, 25, 25), (B, dim, H, W)
        cvt_out = self.cvt_model(images, intrinsics, extrinsics).cuda()

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
        
        # for key in data['labels'].keys():
        #     for k in data['labels'][key]:
        #         k = k.cuda()

        # Compute the loss
        losses = self.head_model.loss(data['labels'], preds)
        
        loss, log_vars = parse_second_losses(losses)

        # Clear gradients
        self.opt.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the parameters
        self.opt.step()

        return loss.item()

    def val_step(self, data: dict):
        self.eval()

        preds = self.forward(
            data['images'], data['intrinsics'], data['extrinsics']).detach().cpu()

        losses = self.head_model.loss(data['labels'], preds)
        
        loss, log_vars = parse_second_losses(losses)

def parse_second_losses(losses):
    from collections import OrderedDict

    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars