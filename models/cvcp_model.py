import logging
from .centerpoint.first_stage.center_head import CenterHead
from .MLP import MLP

import torch
import torch.nn.functional as F

from models.cvt.encoder import CVTEncoder

from collections import OrderedDict

from metrics.iou_3d import IoU3D
from .centerpoint.second_stage.two_stage_detector import TwoStageDetectorUtils as utils
from .centerpoint.second_stage.roi_head import RoIHead
from .centerpoint.second_stage.BEVFeatureExtractor import BEVFeatureExtractor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
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


class CVCPModel(nn.Module):
    """
    CVCPModel is a Module subclass that represents the CVCP model.

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
        self.mlp = MLP(num_hidden=3)
        self.resize_shape = resize_shape
        self.center_head = center_head_model
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
            batch (dict): Mg5g41BfR5mRThe input batch.

        Returns:
            tuple: A tuple containing the predicted boxes, the output of the CVT model, and the loss.

        """
        cvt_out = self.cvt_encoder(
            batch['images'], batch['intrinsics'], batch['extrinsics'])
        # print(cvt_out.shape)
        cvt_out = self.mlp(cvt_out)
        # print(cvt_out.shape)
        cvt_out = F.interpolate(
            cvt_out, size=self.resize_shape, mode='bilinear', align_corners=False)
        # print(cvt_out.shape)
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

    def _forward_stage_two(self, stage_one_boxes, bev, batch):
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

        pred_stage_two = self.roi_head(label)
        roi_loss, tb_dict = self.roi_head.get_loss()

        # #iou loss

        # pred_boxes = pred_stage_two['rois'][..., :7]
        # pred_classes = pred_stage_two['roi_labels']
        # label_boxes = pred_stage_two['gt_boxes_and_cls'][..., :7]
        # label_classes = pred_stage_two['gt_boxes_and_cls'][..., -1]

        # intersections = []
        # unions = []
        # for k in label_classes.unique():
        #     pred_boxes_mask = (pred_classes == k)
        #     label_boxes_mask = (label_classes == k)
        #     if pred_boxes_mask.sum() > 0 and label_boxes_mask.sum() > 0:
        #         cur_pred = pred_boxes[pred_boxes_mask]
        #         cur_label = label_boxes[label_boxes_mask]

        # if train:
        #     return roi_loss, tb_dict
        # else:
        return pred_stage_two, roi_loss, tb_dict

    def forward(self, batch):
        """
        Performs the forward pass for the model.

        Args:
            batch: The input batch.

        Returns:
            dict: The predicted boxes.

        """
        boxes, cvt_out, loss = self._forward_stage_one(batch)
        pred_stage_two, roi_loss, tb_dict = self._forward_stage_two(
            boxes, cvt_out, batch)

        loss = utils.combine_loss(loss, roi_loss, tb_dict)
        loss, log_vars = self.parse_second_losses(loss)

        return pred_stage_two, loss

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

    def visualize(self, pred_boxes_3d, label_boxes_3d):
        pred_boxes_2d = []
        label_boxes_2d = []
        for box in pred_boxes_3d:
            x, y, z, w, l, h, vel_x, vel_y, yaw = box
            pred_boxes_2d.append((x, y, w, l))
        for box in label_boxes_3d:
            x, y, z, w, l, h, vel_x, vel_y, yaw = box
            label_boxes_2d.append((x, y, w, l))
        # Create a plot
        fig, ax = plt.subplots()

        # Plot each pred bounding box
        for bbox in pred_boxes_2d:
            c_x, c_y, w, l = bbox
            lower_left_x = c_x - l / 2
            lower_left_y = c_y - w / 2
            rect = patches.Rectangle((lower_left_x.cpu().item(), lower_left_y.cpu().item()), l.cpu(
            ).item(), w.cpu().item(), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Plot each label bounding box
        for bbox in label_boxes_2d:
            c_x, c_y, w, l = bbox
            lower_left_x = c_x - w / 2
            lower_left_y = c_y - l / 2
            rect = patches.Rectangle((lower_left_x.cpu().item(), lower_left_y.cpu().item()), l.cpu(
            ).item(), w.cpu().item(), linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Plot the ego vehicle at the origin
        ax.plot(0, 0, 'bo')  # blue dot

        # Set plot limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        # Set labels and title
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('2D Bounding Boxes Relative to Ego Vehicle')

        # Save the plot as an image
        plt.savefig('bounding_boxes_plot.png')
        plt.close()
