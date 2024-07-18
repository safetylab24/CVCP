import torch
from torchmetrics import Metric
from models.centerpoint.utils.box_torch_ops import center_to_corner_box3d
from pytorch3d.ops import box3d_overlap
# import MAP metric from torchmetrics


class IoU3D(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.num_samples = 0
        self.iou_sum = 0
        self.add_state('preds', default=torch.zeros(
            num_classes), dist_reduce_fx='cat')
        self.add_state('labels', default=torch.zeros(
            num_classes), dist_reduce_fx='cat')

    def update(self, preds: dict, labels: dict):
        self.preds = preds
        self.labels = labels

    def compute(self):
        # calculate iou for each class, then sum them up
        for i, (pred, label) in enumerate(zip(self.preds, self.labels)):
            iou = self.compute_iou_3d(pred, label)
            self.iou_sum[i] += iou
            self.num_samples += 1
        return self.iou_sum / self.num_samples

    def compute_iou_3d(self, pred_bbox, label_bbox):
        origin = (0.5, 1.0, 0.5)
        gt_box_corners_pred = center_to_corner_box3d(
            pred_bbox[:, :3], pred_bbox[:,
                                        3:6], pred_bbox[:, 6], origin=origin, axis=2
        )
        gt_box_corners_label = center_to_corner_box3d(
            label_bbox[:, :3], label_bbox[:,
                                          3:6], label_bbox[:, 6], origin=origin, axis=2
        )
        _, iou = box3d_overlap(gt_box_corners_pred, gt_box_corners_label)
        return iou
