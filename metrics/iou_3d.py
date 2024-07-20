import torch
from torchmetrics import Metric
from models.centerpoint.utils.box_torch_ops import center_to_corner_box3d
# from pytorch3d.ops import box3d_overlap
# import MAP metric from torchmetrics
from models.centerpoint.ops.iou3d_nms.iou3d_nms_utils import to_pcdet
from models.centerpoint.ops.iou3d_nms import iou3d_nms_cuda


class IoU3D(Metric):
    """
    Intersection over Union (IoU) metric for 3D object detection.

    Args:
        num_classes (int): Number of classes.
        dist_sync_on_step (bool): Whether to synchronize metric state across processes during training. Default is False.
    """

    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.num_samples = 0
        self.iou_sum = 0
        self.add_state('pred_boxes', default=torch.tensor([]),
                       dist_reduce_fx='cat')
        self.add_state('pred_classes', default=torch.tensor([]),
                       dist_reduce_fx='cat')
        self.add_state('label_boxes', default=torch.tensor([]),
                       dist_reduce_fx='cat')
        self.add_state('label_classes', default=torch.tensor([]),
                       dist_reduce_fx='cat')

    def update(self, pred_boxes, pred_classes, label_boxes, label_classes):
        """
        Update the metric state with predicted and label boxes and classes.

        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes.
            pred_classes (torch.Tensor): Predicted class labels.
            label_boxes (torch.Tensor): Label bounding boxes.
            label_classes (torch.Tensor): Label class labels.
        """
        self.pred_boxes = pred_boxes
        self.pred_classes = pred_classes
        self.label_boxes = label_boxes
        self.label_classes = label_classes

    def compute(self):
        """
        Computes the Complete Intersection over Union (CIoU) metric for 3D bounding boxes.

        Returns:
            torch.Tensor: The computed IoU metric.
        """
        pred_boxes = self.pred_boxes
        pred_classes = self.pred_classes
        label_boxes = self.label_boxes
        label_classes = self.label_classes

        intersections = []
        unions = []
        for k in label_classes.unique():
            pred_boxes_mask = (pred_classes == k)
            label_boxes_mask = (label_classes == k)
            if pred_boxes_mask.sum() > 0 and label_boxes_mask.sum() > 0:
                cur_pred = pred_boxes[pred_boxes_mask]
                cur_label = label_boxes[label_boxes_mask]

            intersection, union = boxes_iou3d_gpu(
                cur_pred, cur_label)  # (M, N)
            intersections.append(intersection.sum())
            unions.append(union.sum())

        return sum(intersections) / sum(unions) if unions else torch.tensor(0.0)


def boxes_iou3d_gpu(boxes_a, boxes_b):
    '''
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    '''
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # transform back to pcdet's coordinate
    boxes_a = to_pcdet(boxes_a)
    boxes_b = to_pcdet(boxes_b)

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.zeros(size=(
        boxes_a.shape[0], boxes_b.shape[0]), dtype=torch.float, device='cuda')  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(
        boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    return overlaps_3d, torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
