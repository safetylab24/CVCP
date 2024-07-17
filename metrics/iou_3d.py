import torch
from torchmetrics import Metric

class IoU3D(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("iou_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for c in range(self.num_classes):
            pred_mask = preds[:, :, 0] == c
            target_mask = target[:, :, 0] == c
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            if union > 0:
                self.iou_sum[c] += intersection / union
                self.num_samples[c] += 1

    def compute(self):
        return self.iou_sum / self.num_samples

    def compute_iou_3d(self, pred_bbox, target_bbox):
        # Implement the IoU calculation for 3D bounding boxes
        # pred_bbox and target_bbox are tensors of shape (N, 7): (x, y, z, dx, dy, dz, class)
        # Return tensor of shape (N,) with IoU values
        pass
    
    # def compute_iou_3d(self, pred_bbox, target_bbox):
    #     # Calculate the intersection volume
    #     max_xyz = torch.min(pred_bbox[:, :3] + pred_bbox[:, 3:6] / 2, target_bbox[:, :3] + target_bbox[:, 3:6] / 2)
    #     min_xyz = torch.max(pred_bbox[:, :3] - pred_bbox[:, 3:6] / 2, target_bbox[:, :3] - target_bbox[:, 3:6] / 2)
    #     intersection = torch.prod(torch.clamp(max_xyz - min_xyz, min=0), dim=1)
        
    #     # Calculate the volume of the predicted and target bounding boxes
    #     pred_volume = torch.prod(pred_bbox[:, 3:6], dim=1)
    #     target_volume = torch.prod(target_bbox[:, 3:6], dim=1)
        
    #     # Calculate the union volume
    #     union = pred_volume + target_volume - intersection
        
    #     # Calculate the IoU
    #     iou = intersection / union
    #     return iou

