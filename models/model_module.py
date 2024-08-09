
import torch
import lightning as L
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR


from metrics.iou_3d import IoU3D
from models.cvcp_model import CVCPModel
from .centerpoint.second_stage.two_stage_detector import TwoStageDetectorUtils as utils

class CVCPModule(L.LightningModule):
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

    def __init__(self, model: CVCPModel, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

        # self.iou = IoU3D(num_classes=len(self.cfg['centerpoint']['tasks']))
        # self.mAP = MAP3D(iou_threshold=0.1)

    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            float: The loss value.

        """
        preds, loss = self.model(batch)
        
        # self.iou(
        #     preds['rois'][..., :7],
        #     preds['roi_labels'],
        #     preds['gt_boxes_and_cls'][..., :7],
        #     preds['gt_boxes_and_cls'][..., -1]
        # )
        
        
        self.log('train_loss', loss, sync_dist=True,
                    prog_bar=True, on_step=True, logger=True)
        # self.log('train_iou', self.iou, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the valdation step for the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            float: The loss value.

        """
        preds, loss = self.model(batch)
        
        # self.iou(
        #     preds['rois'][..., :7],
        #     preds['roi_labels'],
        #     preds['gt_boxes_and_cls'][..., :7],
        #     preds['gt_boxes_and_cls'][..., -1]
        # )
        
        
        self.log('val_loss', loss, sync_dist=True,
                    prog_bar=True, on_step=True, logger=True)
        # self.log('train_iou', self.iou, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    # def test_step(self, batch, batch_idx):
    #     """
    #     Defines the test step for the model.

    #     Args:
    #         batch: The input batch.
    #         batch_idx: The index of the current batch.

    #     Returns:
    #         float: The loss value.

    #     """
    #     preds, loss = self.model(batch)
        
    #     self.iou(
    #         preds['rois'][..., :7],
    #         preds['roi_labels'],
    #         preds['gt_boxes_and_cls'][..., :7],
    #         preds['gt_boxes_and_cls'][..., -1]
    #     )
        
    #     boxes_processed = utils.post_process(preds)        
        
    #     self.log('test_loss', loss, sync_dist=True,
    #                 prog_bar=True, on_step=True, logger=True)
    #     self.log('test_iou', self.iou, on_step=True, on_epoch=False, prog_bar=True)

    #     self.model.visualize(pred_boxes_3d=boxes_processed[0]['box3d_lidar'], label_boxes_3d=preds['gt_boxes_and_cls'][0, :, :-1])
        
    #     return loss

    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay']) 
        scheduler = OneCycleLR(optimizer, max_lr=self.cfg['lr'], total_steps=self.trainer.estimated_stepping_batches)

        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'frequency': 1,
            'interval': 'step',
        }
    }