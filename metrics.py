from typing import Callable
from torch import Tensor
import torch

class Metrics:
    def __init__(self, dev: str, **funcs: Callable[[Tensor, Tensor], Tensor]):
        self.acc: dict[str, list[float]] = {name: [] for name in funcs.keys()}
        self.funcs = funcs
        self.dev = dev
        self.n: list[int] = []

    def reset(self):
        self.n: list[int] = []
        keys = list(self.acc.keys())
        for k in keys:
            del self.acc[k]
            self.acc[k] = []

    def update(self, y: Tensor, Y: Tensor) -> dict[str, Tensor]:
        self.n.append(Y.shape[0])
        ret = {k: self.funcs[k](y, Y) for k in self.acc.keys()}
        for k, v in ret.items():
            self.acc[k].append(v.item())

        return ret
    
    
    def finalize(self) -> dict[str, float]:
        total_samples = sum(self.n)
        ret = {
            k: sum((a * float(n) for a, n in zip(self.acc[k], self.n))) / total_samples for k in self.acc.keys()
        }

        return ret

def mean_maxloss(x: Tensor, y: Tensor) -> Tensor:
    diff = (x - y).abs().max(dim=-1).values
    return diff.mean()

def metrics_to_float(metric_dict: dict[str, Tensor]) -> dict[str, float]:
    return {k: v.item() for k, v in metric_dict.items()}

def combined_loss(inputs: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (
        torch.nn.functional.mse_loss(inputs, target, reduction="mean")
        + torch.nn.functional.l1_loss(inputs, target, reduction="mean")
    )



