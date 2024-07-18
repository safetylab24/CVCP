import torch
from torchmetrics import Metric

class MAP3D(Metric):
    def __init__(self, iou_threshold=0.1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        self.add_state('map', default=torch.tensor(0.0), dist_reduce_fx='mean')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Compute IoU for 3D bounding boxes
        iou = self.compute_iou_3d(preds, target)
        # Calculate True Positives, False Positives, False Negatives
        tp, fp, fn = self.compute_tp_fp_fn(iou, self.iou_threshold)
        # Update states
        self.map += tp / (tp + fp + fn + 1e-6)  # Add small epsilon to avoid division by zero
        self.total += 1

    def compute(self):
        return self.map / self.total

    def compute_iou_3d(self, preds, target):
        # Implement the IoU calculation for 3D bounding boxes
        inter_vol = self.compute_intersection(preds, target)
        vol_preds = self.compute_volume(preds)
        vol_target = self.compute_volume(target)
        union_vol = vol_preds + vol_target - inter_vol
        iou = inter_vol / (union_vol + 1e-6)  # Add small epsilon to avoid division by zero
        return iou

    def compute_intersection(self, box1, box2):
        # Compute the intersection volume of two 3D boxes
        max_xyz = torch.min(box1[:, :, 3:], box2[:, :, 3:])
        min_xyz = torch.max(box1[:, :, :3], box2[:, :, :3])
        inter_dims = (max_xyz - min_xyz).clamp(min=0)
        inter_vol = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]
        return inter_vol

    def compute_volume(self, box):
        # Compute the volume of a 3D box
        dims = box[:, :, 3:] - box[:, :, :3]
        vol = dims[:, :, 0] * dims[:, :, 1] * dims[:, :, 2]
        return vol

    def compute_tp_fp_fn(self, iou, threshold, preds: torch.Tensor, labels: torch.Tensor):
        # Compute True Positives, False Positives, False Negatives based on IoU and threshold
        tp = 0
        fn = 0
        fp = 0
        temp_iou  = 0
        
        
        for label in labels:
            for pred in preds:
                _, temp_iou = compute_iou_3d(pred, target)
                if(temp_iou>threshold):
                    tp+=1
                elif(temp_iou==0):
                    fp+=1
                
        return tp, fp, fn

# Example usage
# preds = torch.tensor([[[0, 0, 0, 2, 2, 2]]])  # Example predicted 3D bounding box
# target = torch.tensor([[[1, 1, 1, 3, 3, 3]]])  # Example ground truth 3D bounding box

# metric = MAP3D()
# metric.update(preds, target)
# print(metric.compute())

