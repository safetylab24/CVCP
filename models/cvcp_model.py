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
from .centerpoint.two_stage_detector import TwoStageDetectorUtils as utils
from .centerpoint.roi_head import RoIHead
from .centerpoint.BEVFeatureExtractor import BEVFeatureExtractor

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


model_cfg = dict(
    CLASS_AGNOSTIC=True,
    SHARED_FC=[256, 256],
    CLS_FC=[256, 256],
    REG_FC=[256, 256],
    DP_RATIO=0.3,

    TARGET_CONFIG=dict(
        ROI_PER_IMAGE=128,
        FG_RATIO=0.5,
        SAMPLE_ROI_BY_EACH_CLASS=True,
        CLS_SCORE_TYPE='roi_iou',
        CLS_FG_THRESH=0.75,
        CLS_BG_THRESH=0.25,
        CLS_BG_THRESH_LO=0.1,
        HARD_BG_RATIO=0.8,
        REG_FG_THRESH=0.55
    ),
    LOSS_CONFIG=dict(
        CLS_LOSS='BinaryCrossEntropy',
        REG_LOSS='L1',
        LOSS_WEIGHTS={
            'rcnn_cls_weight': 1.0,
            'rcnn_reg_weight': 1.0,
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        }
    )
)


class CVCPModel(L.LightningModule):
    """
    CVCPModel is a LightningModule subclass that represents the CVCP model.

    Args:
        cvt_encoder (CVTEncoder): The CVT encoder model.
        center_head_model (CenterHead): The center head model.
        resize_shape (tuple): The shape to resize the output of the CVT model.
        cfg (dict): The configuration dictionary.

    Attributes:
        cvt_encoder (CVTEncoder): The CVT encoder model.
        center_head (CenterHead): The center head model.
        resize_shape (tuple): The shape to resize the output of the CVT model.
        cfg (dict): The configuration dictionary.
        max_conf_val (int): The maximum confidence value.
        roi_head (RoIHead): The RoI head model.
        bev_feature_extractor (BEVFeatureExtractor): The BEV feature extractor model.
        iou (IoU3D): The IoU3D metric.

    Methods:
        _forward_stage_one: Performs the forward pass for stage one of the model.
        _forward_stage_two: Performs the forward pass for stage two of the model.
        training_step: Defines the training step for the model.
        validation_step: Defines the validation step for the model.
        test_step: Defines the test step for the model.
        configure_optimizers: Configures the optimizers for the model.
        parse_second_losses: Parses the second losses for logging.

    """

    def __init__(self, cvt_encoder: CVTEncoder, center_head_model: CenterHead, resize_shape, cfg):
        super().__init__()
        self.cvt_encoder = cvt_encoder
        self.center_head = center_head_model
        self.resize_shape = resize_shape
        self.cfg = cfg
        self.max_conf_val = 0
        self.roi_head = RoIHead(
            input_channels=650, model_cfg=model_cfg, code_size=9, add_box_param=True)
        self.bev_feature_extractor = BEVFeatureExtractor()

        self.iou = IoU3D(num_classes=len(cfg['centerpoint']['tasks']))
        # self.mAP = MAP3D(iou_threshold=0.1)

    def _forward_stage_one(self, batch):
        """
        Performs the forward pass for stage one of the model.

        Args:
            batch (dict): The input batch.

        Returns:
            tuple: A tuple containing the predicted boxes, the output of the CVT model, and the loss.

        """
        cvt_out = self.cvt_encoder(
            batch['images'], batch['intrinsics'], batch['extrinsics'])
        cvt_out = F.interpolate(
            cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        preds = self.center_head(cvt_out)

        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.center_head.predict(new_preds, test_cfg=self.cfg['test'])
        loss = self.center_head.loss(batch['labels'], preds)

        return boxes, cvt_out, loss

    def _forward_stage_two(self, stage_one_boxes, bev, batch, train=False):
        """
        Performs the forward pass for stage two of the model.

        Args:
            stage_one_boxes: The predicted boxes from stage one.
            bev: The BEV input.
            batch: The input batch.
            train (bool): Whether to perform training or not.

        Returns:
            tuple: A tuple containing the ROI loss, the tensorboard dictionary.

        """
        centers_vehicle_frame = utils.get_box_center(stage_one_boxes)
        feature = self.bev_feature_extractor.forward(
            centers_vehicle_frame, num_point=5, bev=bev)
        features = []
        features.append(feature)

        label = utils.reorder_first_stage_pred_and_feature(
            first_pred=stage_one_boxes, label=batch['labels'], features=features)

        pred_stage_two = self.roi_head(label, training=train)
        roi_loss, tb_dict = self.roi_head.get_loss()

        if train: 
            return roi_loss, tb_dict
        else:
            return pred_stage_two, roi_loss, tb_dict

    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            float: The loss value.

        """
        stage_one_boxes, bev, stage_one_loss = self._forward_stage_one(batch)
        roi_loss, tb_dict = self._forward_stage_two(
            stage_one_boxes, bev, batch, train=True)

        loss_out = utils.combine_loss(stage_one_loss, roi_loss, tb_dict)

        loss_out, log_vars = self.parse_second_losses(loss_out)
        if (batch_idx + 1) % self.cfg['log_every_n_steps'] == 0:
            self.log('train_loss', loss_out, sync_dist=True,
                     prog_bar=True, on_step=True, logger=True)
        return loss_out

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step for the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            float: The loss value.

        """
        stage_one_boxes, bev, stage_one_loss = self._forward_stage_one(batch)
        preds_final, roi_loss, tb_dict = self._forward_stage_two(
            stage_one_boxes, bev, batch)

        boxes_out = utils.post_process(preds_final)
        loss_out = utils.combine_loss(stage_one_loss, roi_loss, tb_dict)

        self.iou(
            preds_final['rois'][..., :7],
            preds_final['roi_labels'],
            preds_final['gt_boxes_and_cls'][..., :7],
            preds_final['gt_boxes_and_cls'][..., -1]
        )
        
        loss_out, log_vars = self.parse_second_losses(loss_out)
        if (batch_idx + 1) % self.cfg['log_every_n_steps'] == 0:
            self.log('train_loss', loss_out, sync_dist=True,
                     prog_bar=True, on_step=True, logger=True)
            
        self.log('iou', self.iou, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_out

    def test_step(self, batch, batch_idx):
        """
        Defines the test step for the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            float: The loss value.

        """
        stage_one_boxes, bev, stage_one_loss = self._forward_stage_one(batch)
        boxes_final, roi_loss, tb_dict = self._forward_stage_two(
            stage_one_boxes, bev, batch)

        boxes_out = utils.post_process(boxes_final)
        loss_out = utils.combine_loss(stage_one_loss, roi_loss, tb_dict)
        
        loss_out, log_vars = self.parse_second_losses(loss_out)
        if (batch_idx + 1) % self.cfg['log_every_n_steps'] == 0:
            self.log('train_loss', loss_out, sync_dist=True,
                     prog_bar=True, on_step=True, logger=True)
        return loss_out

    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.

        """
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
        """
        Parses the second losses for logging.

        Args:
            losses: The losses to parse.

        Returns:
            tuple: A tuple containing the total loss and the log variables.

        """
        log_vars = OrderedDict()
        loss = sum(losses['loss'])
        for loss_name, loss_value in losses.items():
            if loss_name == 'loc_loss_elem':
                log_vars[loss_name] = [[i.item() for i in j]
                                       for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars
