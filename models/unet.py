import torch
from torch import nn, Tensor

from models.unet_parts import DoubleConv, Down, Up


def conv_out(
    h_in: int, kernel_size: int, padding: int = 0, dilation: int = 1, stride: int = 1
) -> int:
    return (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def max_pool_out(
    h_in: int,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
    dilation: int = 1,
):
    if stride is None:
        stride = kernel_size
    return (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(size, size),
            nn.LeakyReLU()
        )

    def forward(self, X: Tensor) -> Tensor:
        return X + self.f(X)



class UnetConverter(nn.Module):
    def __init__(
        self, channel_factor: int = 8, kernel_sizes: list[int] = [19, 19, 19],
        in_size: int = 1024, inner_linear: bool = False,
    ) -> None:
        super().__init__()
        f = channel_factor

        self.input = DoubleConv(1, f * 2**0, kernel_size=kernel_sizes[0])
        chs = [2**i for i in range(len(kernel_sizes) + 1)]

        self.down = nn.ModuleList(
            [
                Down(f * i0, f * i1, kernel_size=ks)
                for i0, i1, ks in zip(chs[:-1], chs[1:], kernel_sizes)
            ]
        )

        inner_shape = in_size 
        for _ in kernel_sizes:
            inner_shape //= 2

        if inner_linear:
            self.inner_linear = ResBlock(inner_shape)
        else:
            self.inner_linear = nn.Identity()

        up_chs = list(reversed(chs))
        self.up = nn.ModuleList(
            [
                Up(f * i0, f * i1, kernel_size=ks) for i0, i1, ks in zip(up_chs[:-1], up_chs[1:], reversed(kernel_sizes))
            ]
        )

        self.out = nn.Sequential(
            nn.Conv1d(f, 1, 1, padding=0),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [self.input(x)]
        for i, d in enumerate(self.down):
            xs.append(d(xs[i]))

        x = self.inner_linear(xs[-1])
        for i, u in enumerate(self.up):
            x = u(x, xs[-(2 + i)])

        del xs
        return self.out(x)
