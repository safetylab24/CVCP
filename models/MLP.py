import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_hidden):
        from torchvision.ops import MLP
        super().__init__()
        self.out_dim = 512
        self.mlp = MLP(in_channels=128, hidden_channels=[512, 512, self.out_dim])
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, output_padding=0),
        #     nn.ReLU()
        # )

    def forward(self, x):
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        # Flatten the spatial dimensions: BxCxHW
        x = x.view(B, C, H * W).permute(0, 2, 1)  # BxHWxC
        x = self.mlp(x)  # Apply MLP: Bx625x128
        x = x.permute(0, 2, 1).view(B, self.out_dim, H, W)  # BxC'xHXW
        # Upsample spatial dimensions to 128x128
        # x = self.upsample(x)
        return x
