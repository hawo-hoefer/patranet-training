import json
import logging
import os
import sys
from functools import partial

import torch
from numpy.typing import NDArray
from torch import Generator, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from chunkds import ChunkDS
from metrics import Metrics, combined_loss, mean_maxloss
from models.unet import UnetConverter

logging.captureWarnings(True)
TORCH_DTYPE = torch.float32

logger = logging.getLogger("train_unet.py")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)7s] %(name)s | %(message)s",
)


def add_noise(
    XY: tuple[NDArray, NDArray], noise_amt: float, generator: Generator | None = None
) -> tuple[Tensor, Tensor]:
    X, Y = XY

    X = torch.tensor(X, dtype=TORCH_DTYPE).unsqueeze(1)
    Y = torch.tensor(Y, dtype=TORCH_DTYPE).unsqueeze(1)
    noise_scale = torch.rand(size=[X.shape[0], 1, 1]) * noise_amt
    noise = (torch.rand(size=X.shape, generator=generator) - 0.5) * noise_scale

    X += noise
    # Scale by X range
    Y -= X.min(dim=-1, keepdim=True).values
    Y /= X.max(dim=-1, keepdim=True).values

    X -= X.min(dim=-1, keepdim=True).values
    X /= X.max(dim=-1, keepdim=True).values

    return X, Y


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, v0_path: str, v1_path: str, name: str):
        self.v0 = ChunkDS(v0_path, num_threads=5, name=f"{name}-inputs")
        self.v1 = ChunkDS(v1_path, num_threads=5, name=f"{name}-targets")
        assert self.v0.num_samples == self.v1.num_samples

    def __len__(self) -> int:
        return self.v0.num_samples

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.v0.inputs[0][i], self.v1.inputs[0][i]

    def __getitems__(self, i: list[int]) -> tuple[Tensor, Tensor]:
        return self.v0.inputs[0][i], self.v1.inputs[0][i]


def get_dataloaders(ds_path: str, batch_size: int):
    train = TranslationDataset(
        os.path.join(ds_path, "cukab", "train"),
        os.path.join(ds_path, "cuka", "train"),
        "train",
    )
    val = TranslationDataset(
        os.path.join(ds_path, "cukab", "val"),
        os.path.join(ds_path, "cuka", "val"),
        "val",
    )
    gen = torch.Generator(device="cpu").manual_seed(1234)

    dl_cfg = {
        "batch_size": batch_size,
        #                                        25% of reference pattern height
        "collate_fn": partial(add_noise, noise_amt=0.25 * 420, generator=gen),
        "shuffle": True,
        "prefetch_factor": 2,
        "num_workers": 16,
        "persistent_workers": True,
        "generator": gen,
        "pin_memory": True,
    }
    train_loader = DataLoader(train, **dl_cfg)
    val_loader = DataLoader(val, **dl_cfg)
    return train_loader, val_loader, len(train), len(val)


class EarlyStopping:
    def __init__(self, patience: int, rel_thresh: float = 1e-3):
        self.patience = patience
        self.best = 1e100
        self.count = 0
        self.threshold = rel_thresh

    def __call__(self, score):
        if (self.best - score) > self.threshold * self.best:
            self.count = 0
            self.best = score
        else:
            self.count += 1

        return self.count >= self.patience


def train_model(
    learning_rate: float = 1e-3,
    epochs: int = 50,
    kernel_sizes: list[int] = [19, 19, 19],
    channel_factor: int = 2,
    inner_linear: bool = True,
):
    workdir = os.path.dirname(__file__)
    ds_path = os.path.join(workdir, "data")
    training_results_path = os.path.join(workdir, "training_results")
    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)

    train_loader, val_loader, train_len, val_len = get_dataloaders(ds_path, 1024)
    model = (
        UnetConverter(
            kernel_sizes=kernel_sizes,
            channel_factor=channel_factor,
            inner_linear=inner_linear,
            in_size=2048,
        )
        .cuda()
        .to(TORCH_DTYPE)
    )

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, threshold=1e-2, threshold_mode="rel", patience=3
    )
    should_stop = EarlyStopping(patience=10, rel_thresh=1e-3)

    cfg = {
        "learning_rate": learning_rate,
        "kernel_sizes": kernel_sizes,
        "channel_factor": channel_factor,
        "epochs": epochs,
        "model_size": sum(int(torch.numel(p)) for p in model.parameters()),
    }

    with open(os.path.join(training_results_path, "cfg.json"), "w") as file:
        json.dump(cfg, file)

    logger.info("Model cfg:")
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"Training on {train_len:,} samples.")
    logger.info(f"Validating on {val_len:,} samples.")
    logger.info(
        f"Model with {sum(torch.numel(p) for p in model.parameters()):,} parameters."
    )

    m = Metrics("cuda", comb=combined_loss, mean_max=mean_maxloss)
    logger.info(f"{'epoch':5} {'lr':12} {'loss':12} {'val_loss':12} {'mean_max':12}")

    metrics: dict[str, list[dict[str, float]]] = {"val": [], "train": []}

    for e in range(epochs):
        progress = tqdm(total=len(train_loader) + len(val_loader), leave=False)
        m.reset()
        model.train()
        for X, Y in train_loader:
            X, Y = X.cuda().to(TORCH_DTYPE), Y.to(torch.float32).cuda()
            Y_hat = model.forward(X)
            m_dict = m.update(Y_hat, Y)
            l = m_dict["comb"]

            opt.zero_grad()
            l.backward()
            opt.step()
            progress.update()

            del m_dict, l, Y_hat, Y, X

        tm = m.finalize()
        metrics["train"].append(tm)
        tl = tm["comb"]


        m.reset()

        model.eval()
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.cuda(), Y.cuda()
                Y_hat = model.forward(X)
                m_dict = m.update(Y_hat, Y)["comb"]
                del m_dict, Y_hat, Y, X
                progress.update()

        vm = m.finalize()
        metrics["val"].append(vm)

        vl = vm["comb"]
        vlmax = vm["mean_max"]
        lr = sched.get_last_lr()[0]

        progress.clear()
        logger.info(f"{e:5} {lr:12.6e} {tl:12.6e} {vl:12.6e} {vlmax:12.6e}")
        sys.stdout.flush()

        sched.step(tl)

        metrics["val"][-1]["epoch"] = e
        metrics["train"][-1]["epoch"] = e

        with open(os.path.join(training_results_path, "metrics.json"), "w") as file:
            json.dump(metrics, file)

        if should_stop(tm["comb"]):
            break

        if should_stop.count == 0:
            torch.save(model, os.path.join(training_results_path, "model.pt"))

    del model


if __name__ == "__main__":
    train_model(
        learning_rate=5e-2,
        epochs=20,
        kernel_sizes=[3, 3, 3, 3, 3],
        inner_linear=True,
        channel_factor=1,
    )
