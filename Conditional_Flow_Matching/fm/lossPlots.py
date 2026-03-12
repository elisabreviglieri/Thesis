# fm/lossPlots.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_overlay(
    results_dict: dict,
    title: str,
    which: str = "val",          # "train" or "val"
    logy: bool = False,
    max_epochs: int | None = None,
    legend_loc: str = "upper right",
):
    """
    Overlay loss curves for many modes on the same plot.

    results_dict: {mode: res}
      res["loss_history"]["train"] and ["val"] must exist
    """
    if which not in {"train", "val"}:
        raise ValueError("which must be 'train' or 'val'")

    plt.figure(figsize=(9, 5))

    for mode, res in results_dict.items():
        hist = res["loss_history"][which]
        y = np.asarray(hist, dtype=float)

        if max_epochs is not None:
            y = y[:max_epochs]

        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, linewidth=2.0, label=mode)

    plt.xlabel("Epoch")
    plt.ylabel(f"{which} loss")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.legend(fontsize=8, loc=legend_loc)
    plt.tight_layout()
    plt.show()


def plot_loss_grid(
    results_dict: dict,
    title: str,
    order: list | None = None,
    logy: bool = False,
    max_epochs: int | None = None,
    ncols: int = 3,
):
    """
    Small multiples: one subplot per mode, showing train+val.
    """
    modes = list(results_dict.keys()) if order is None else list(order)
    n = len(modes)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i, mode in enumerate(modes):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        res = results_dict[mode]
        tr = np.asarray(res["loss_history"]["train"], dtype=float)
        va = np.asarray(res["loss_history"]["val"], dtype=float)

        if max_epochs is not None:
            tr = tr[:max_epochs]
            va = va[:max_epochs]

        x_tr = np.arange(1, len(tr) + 1)
        x_va = np.arange(1, len(va) + 1)

        ax.plot(x_tr, tr, linewidth=1.5, label="train")
        ax.plot(x_va, va, linewidth=2.2, label="val")
        ax.set_title(mode, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")

    # turn off unused axes
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()
    
    
    
# ========================================
# Per quando faccio fine-tuning -> Tanti plot!
# ========================================
def plot_val_bal_vs_unbal_per_mode(
    results_bal: dict,
    results_unbal: dict,
    title_prefix: str,
    modes: list,
    logy: bool = False,
):
    """
    For each mode, plot validation loss:
        BALANCED vs UNBALANCED

    results_bal / results_unbal:
        dict[mode] -> res (must contain res["loss_history"]["val"])
    """

    for m in modes:

        if (m not in results_bal) or (m not in results_unbal):
            continue

        yb = np.asarray(results_bal[m]["loss_history"]["val"], dtype=float)
        yu = np.asarray(results_unbal[m]["loss_history"]["val"], dtype=float)

        x1 = np.arange(1, len(yb) + 1)
        x2 = np.arange(1, len(yu) + 1)

        plt.figure(figsize=(7, 4))

        plt.plot(x1, yb, linewidth=2.2, label="BAL")
        plt.plot(x2, yu, linewidth=2.2, label="UNBAL")

        plt.title(f"{title_prefix} — {m}")
        plt.xlabel("Epoch")
        plt.ylabel("val loss")

        if logy:
            plt.yscale("log")

        plt.legend()
        plt.tight_layout()
        plt.show()