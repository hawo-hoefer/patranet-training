# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import json

import torch
import os
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from tqdm.auto import tqdm

from numpy.typing import NDArray
from torch import Generator
import pandas as pd
from train_unet import TranslationDataset
from matplotlib import pyplot as plt
from train_unet import add_noise
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from plot_utils import *


# %%
def mapr(X, x0, x1, y0, y1):
    return (X - x0) / (x1 - x0) * (y1 - y0) + y0

def norm(X):
    return (X - X.min()) / (X.max() - X.min())

def clip(X, x0, x1):
    X = X.copy()
    lo = X < x0
    X[lo] = x0
    hi = X > x1
    X[hi] = x1
    return X

def draw_my_rect(r, larger_fac = 1.1, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x0, x1, y0, y1 = r
    dx = x1 - x0
    dy = y1 - y0

    off = (larger_fac - 1) / 2
    
    rect = Rectangle((x0 - dx * off, y0 - dy * off), dx * larger_fac, dy * larger_fac, **kwargs)
    ax.add_artist(rect)


def draw_zoom_lines(origin_rect, dst_rect, ax = None):
    if ax is None:
        ax = plt.gca()

    
    draw_my_rect(dst_rect, facecolor='#fff', edgecolor='black', ax=ax, zorder=100)
    draw_my_rect(origin_rect, facecolor='#0000', edgecolor='black', ax=ax, zorder=100)  
    
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        x0, x1, y0, y1 = origin_rect
        dx0, dx1, dy0, dy1 = dst_rect
        ddx = dx1 - dx0
        ddy = dy1 - dy0

        x0idx = np.argmin(np.abs(x - x0))
        x1idx = np.argmin(np.abs(x - x1))
        x, y = x[x0idx:x1idx], clip(y[x0idx:x1idx], y0, y1)

        xrange_rel = (x.max() - x.min()) / (x1 - x0)
        yrange_rel = (y.max() - y.min()) / (y1 - y0)
        
        map_y = mapr(y, y0, y1, dy0, dy1)
        map_x = np.linspace(dx0, dx1, map_y.shape[0])
        
        l2 = Line2D(map_x, map_y, zorder=101)
        l2.update_from(line)
        l2.set_label(None)
        ax.add_artist(l2)
    
def add_noise_exact(
    XY: tuple[NDArray, NDArray], noise_amt: float, generator: Generator | None = None
) -> tuple[Tensor, Tensor]:
    X, Y = XY

    X = torch.tensor(X, dtype=torch.float32, device="cpu").unsqueeze(1)
    Y = torch.tensor(Y, dtype=torch.float32, device="cpu").unsqueeze(1)
    noise = torch.randn(size=X.shape, generator=generator, device="cpu") * noise_amt 

    X += noise
    # Scale by X range    xmin = X.min(dim=-1, keepdim=True).values
    Y -= X.min(dim=-1, keepdim=True).values   
    X -= X.min(dim=-1, keepdim=True).values
    
    Y /= X.max(dim=-1, keepdim=True).values
    X /= X.max(dim=-1, keepdim=True).values

    return X, Y

def normalize(X: Tensor):
    xmin = X.min(dim=-1, keepdims=True).values
    xmax = X.max(dim=-1, keepdims=True).values
    return (X - xmin) / (xmax - xmin)


# %%
mpl.rc("text", usetex = False)
mpl.rc("font", size=7)

# %%
with np.load("./experimental_data/data.npz") as archive:
    patterns = archive["inputs"]
    composition = archive["targets"]
    
with open("./experimental_data/meta.json", "r") as file:
    meta = json.load(file)
sample_info = pd.DataFrame(meta["samples"])

# %%
unfiltered = sample_info[~sample_info["filter"]]["idx"]
unfiltered_patterns = patterns[unfiltered]
thetas = np.linspace(10, 70, 2048)

# %%
sample_info.loc[unfiltered].iloc[0]

# %%
ds = TranslationDataset("./data/cukab/test", "./data/cuka/test", "test")

# %%
m = torch.load("./training_results/model.pt", weights_only=False).eval().cpu()

# %%
X, Y = ds[10:40]
generator = torch.Generator().manual_seed(1234)
X, Y = add_noise_exact((X, Y), noise_amt=0.125 * 420, generator=generator)

with torch.no_grad():
    Y_ = m(X)

idx = 6

fig = plt.figure(figsize=(3.29, 3.29 / 16 * 9), layout="constrained")

X = X[idx].numpy().squeeze()
Y = Y[idx].numpy().squeeze()
Y_ = Y_[idx].numpy().squeeze()
plt.plot(thetas, X, label=r"$\mathrm{CuK}\alpha_{1/2}/\beta$", color=kit_blue)
plt.plot(thetas, Y, label=r"$\mathrm{CuK}\alpha_{1/2}$", color=kit_red)
plt.plot(thetas, Y_, label="translated", color=kit_orange)
plt.xlabel(r"$2\theta\ [\mathrm{deg}]$")
plt.ylabel("I / a.u.")
plt.yticks([], [])
plt.ylim(-0.05, 1.4)
plt.legend()
orig_rect = (31, 40, 0.3, 1.0)
dst_rect = (52, 70, 0.2, 1.25)
draw_zoom_lines(orig_rect, dst_rect, plt.gca())
plt.annotate("", xy=(dst_rect[0] - 1, dst_rect[3] - 0.05), xytext=(orig_rect[1] + 1, 0.975),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc,rad=-0.2"))
plt.savefig("example_synthetic_translation.pdf", bbox_inches="tight")

# %%
with torch.no_grad():
    translated_exp = m(torch.tensor(unfiltered_patterns).unsqueeze(1).float()).numpy().squeeze()
thetas = np.linspace(10, 70, 2048)
fig = plt.figure(figsize=(3.29, 3.29 / 16 * 9), layout="constrained")
plt.plot(thetas, unfiltered_patterns[0], color=kit_blue, label=r"$\mathrm{Fe}_3\mathrm{O}_4$ @ 0.1s/step $\mathrm{CuK}\alpha{1/2}/\beta$")
plt.plot(thetas, translated_exp[0], color=kit_orange, label=r"model output")

plt.legend(loc="lower right")
plt.ylabel(r"$\mathrm{I}\ [\mathrm{a.u.}]$%")
plt.xlabel(r"$2\theta\ [\mathrm{deg}]$")
plt.yticks([])

orig_rect = (34, 38, 0.5, 1.00)
dst_rect = (10, 25, 0.4, 1.00)
draw_zoom_lines(orig_rect, dst_rect, plt.gca())
plt.annotate("", xy=(dst_rect[1] + 1, dst_rect[3] - 0.05), xytext=(orig_rect[0] - 1, 0.975),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc,rad=-0.2"))

plt.savefig("./example_exp_translation.pdf", bbox_inches="tight")

# %%
generator = torch.Generator().manual_seed(342342)
dl = DataLoader(ds, batch_size=256, collate_fn=partial(add_noise, noise_amt=0.25 * 420, generator=generator))

maes = []
mmax = []
untranslated_maes = []
untranslated_mmax = []
with torch.no_grad():
    m.cuda()
    for X, Y in tqdm(dl):
        Y_ = m(X.cuda()).cpu()

        # assert ((Y_ >= 0.0) & (Y_ <= 1.0)).all()
        mmax.append((Y - Y_).abs().max(dim=-1).values.squeeze())
        maes.append((Y - Y_).abs().mean(dim=-1).squeeze())
        untranslated_maes.append((X - Y).abs().mean(dim=-1).squeeze())
        untranslated_mmax.append((X - Y).abs().max(dim=-1).values.squeeze())
    m.cpu()

maes = torch.cat(maes)
mmax = torch.cat(mmax)
untranslated_maes = torch.cat(untranslated_maes)
untranslated_mmax = torch.cat(untranslated_mmax)

# %%
print("mmax-% over 0.2", (mmax > 0.2).float().mean() * 100)
print("mmax-quantile 0.05, 0.95", mmax.quantile(0.05), mmax.quantile(0.95))
print("mmax-mean", mmax.mean())
print("mmax-max", mmax.max())
print("mmean-% over 0.2", (maes > 0.02).float().mean() * 100)
print("mmean-quantile 0.05, 0.95", maes.quantile(0.05), maes.quantile(0.95))
print("mmean-mean", maes.mean())

print("mmean-untranslated-mean", untranslated_maes.mean())
print("mmean-untranslated-quantile 0.05, 0.95", untranslated_maes.quantile(0.05), untranslated_maes.quantile(0.95))
print("mmax-untranslated-mean", untranslated_mmax.mean())
print("mmax-untranslated-quantile 0.05, 0.95", untranslated_mmax.quantile(0.05), untranslated_mmax.quantile(0.95))

# %%
fig, ax = plt.subplots(2, 1, layout="constrained", figsize=(3.29, 2.29 / 4 * 3))
ax[0].hist(maes, bins=250, color=kit_blue, density=True)

y0, y1 = ax[0].get_ylim()
boxplot_props = lambda color: {"color": color}

bp = boxplot_props(kit_orange)

ax[0].boxplot(maes, orientation="horizontal", positions=[(y1 - y0) / 2],
              showfliers=False, widths=[(y1 - y0) / 3], 
              boxprops=bp, 
              whiskerprops=bp,
              capprops=bp,
              medianprops=bp)
ax[0].vlines([maes.mean()], ymin=y0, ymax=y0 + (y1 - y0) * 0.8, color=kit_red, label="mean")

bp = boxplot_props(kit_blue)
ax[1].hist(mmax, bins=2500, color=kit_orange, density=True)
y0, y1 = ax[1].get_ylim()
ax[1].boxplot(mmax, orientation="horizontal", positions=[(y1 - y0) / 2],
              showfliers=False, widths=[(y1 - y0) / 3], 
              boxprops=bp, 
              whiskerprops=bp,
              capprops=bp,
              medianprops=bp)
ax[1].vlines([mmax.mean()], ymin=y0, ymax=y0 + (y1 - y0) * 0.8, color=kit_cyan, label="mean")

ax[0].set_yticks([])
ax[0].set_xlim(0, 0.02)
ax[0].set_ylabel(r"$\rho$")
ax[0].legend(loc="upper right")
ax[0].set_xlabel(r"$M_\mathrm{mean}$")
ax[0].set_xticks([0.0, 0.005, 0.01, 0.015, 0.02])

ax[1].set_yticks([])
ax[1].set_xlim(0, 0.2)
ax[1].set_ylabel(r"$\rho$")
ax[1].set_xlabel(r"$M_\mathrm{max}$")
ax[1].legend(loc="upper right")
ax[1].set_xticks([0.0, 0.05, 0.1, 0.15, 0.2])
plt.savefig("patranet_err_dist.pdf", bbox_inches="tight")
