import torch
import torch.nn as nn

from models.cvt.tools.loss import ce_loss, focal_loss
from models.cvt.cross_view_transformer import CrossViewTransformer

class CVTModel(nn.Module):
    def __init__(self, devices, n_classes=4, opt=None, scaler=None, loss_type=None, weights=None):
        super(CVTModel, self).__init__()

        self.device = devices[0]
        self.devices = devices

        self.weights = weights

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        self.backbone = None

        self.loss_type = loss_type
        self.n_classes = n_classes
        self.opt = opt
        self.scaler = scaler
        self.gamma = .1
        self.tsne = False
        
        self.m_in = -23.0
        self.m_out = -5.0

        self.backbone = nn.DataParallel(
                CrossViewTransformer(n_classes=self.n_classes).to(self.device),
                output_device=self.device,
                device_ids=self.devices)

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)
    
    @staticmethod
    def loss(self, logits, target, reduction='mean'):
        if self.loss_type == 'ce':
            A = ce_loss(logits, target, weights=self.weights)
        elif self.loss_type == 'focal':
            A = focal_loss(logits, target, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def state_dict(self, epoch=-1):
        return {
            'model_state_dict': super().state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt is not None else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'epoch': epoch
        }

    def load(self, state_dict):
        self.load_state_dict(state_dict['model_state_dict'])

        if self.opt is not None:
            self.opt.load_state_dict(state_dict['optimizer_state_dict'])

        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def train_step(self, images, intrinsics, extrinsics, labels):
        self.opt.zero_grad(set_to_none=True)

        if self.scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outs = self(images, intrinsics, extrinsics)
                loss = self.loss(outs, labels.to(self.device))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            outs = self(images, intrinsics, extrinsics)
            loss = self.loss(outs, labels.to(self.device))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images, intrinsics, extrinsics)
