"""Parts for Unet construction similar to https://github.com/milesial/Pytorch-UNet"""
from torch import nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3) -> None:
        super().__init__()

        pad_mode = "replicate"
        pad = (kernel_size - 1) // 2
        self.f = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode=pad_mode),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode=pad_mode),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
