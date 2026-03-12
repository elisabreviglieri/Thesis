# fm/training.py
from __future__ import annotations

import os
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import model


# ============================================================
# Load prepared splits
# ============================================================
def load_prepared(mode_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load prepared (already scaled) splits + scalers saved by apply_and_save_all_scalings().
    """
    df_tr = pd.read_pickle(os.path.join(mode_dir, "train.pkl"))
    df_va = pd.read_pickle(os.path.join(mode_dir, "val.pkl"))
    df_te = pd.read_pickle(os.path.join(mode_dir, "test.pkl"))
    scalers = joblib.load(os.path.join(mode_dir, "scalers.joblib"))
    return df_tr, df_va, df_te, scalers


# ============================================================
# Train one scaling mode  (supports fine-tuning)
# ============================================================
def train_one_mode(
    mode_dir: str,
    train_cols: list,
    y_cols: list,
    device: str | torch.device = None,
    epochs: int = 1000,
    batch_size: int = 4096,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    patience: int = 30,
    min_delta: float = 1e-4,
    verbose_every: int = 10,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    init_state_dict: dict | None = None,   # NEW (fine-tuning)
    strict_init: bool = True,              # NEW
) -> Dict[str, Any]:
    """
    Train one CFM model on ONE scaling mode folder (prepared splits already scaled).

    loss_type:
        - "mse"   (default)
        - "huber"

    Fine-tuning:
        init_state_dict: state_dict() to load before training
        strict_init: passed to net.load_state_dict(...)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # Load prepared (already scaled)
    # -----------------------
    df_tr, df_va, df_te, scalers = load_prepared(mode_dir)

    Xtr = df_tr[train_cols].to_numpy(np.float32)
    ytr = df_tr[y_cols].to_numpy(np.float32)
    Xva = df_va[train_cols].to_numpy(np.float32)
    yva = df_va[y_cols].to_numpy(np.float32)

    # -----------------------
    # Dataset / loaders
    # -----------------------
    ds_tr = model.TabularFMDataset(Xtr, ytr)
    ds_va = model.TabularFMDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    # -----------------------
    # Model
    # -----------------------
    net = model.ConditionalVelocityField(
        x_dim=len(y_cols),
        context_dim=len(train_cols),
    ).to(device)

    # ---- Fine-tuning init (optional)
    if init_state_dict is not None:
        missing, unexpected = net.load_state_dict(init_state_dict, strict=strict_init)
        if (not strict_init) and (missing or unexpected):
            print(f"[init_state_dict] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # -----------------------
    # Training bookkeeping
    # -----------------------
    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    bad = 0

    # ============================================================
    # Training loop
    # ============================================================
    for ep in range(1, epochs + 1):

        # -----------------------
        # TRAIN
        # -----------------------
        net.train()
        tr_losses = []

        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)

            if loss_type == "mse":
                loss = model.fm_loss(net, xb, yb, device)
            elif loss_type == "huber":
                loss = model.fm_loss_huber(net, xb, yb, device, delta=huber_delta)
            else:
                raise ValueError("loss_type must be 'mse' or 'huber'")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())

        tr_mean = float(np.mean(tr_losses))

        # -----------------------
        # VALIDATION
        # -----------------------
        net.eval()
        va_losses = []

        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)

                if loss_type == "mse":
                    loss = model.fm_loss(net, xb, yb, device)
                elif loss_type == "huber":
                    loss = model.fm_loss_huber(net, xb, yb, device, delta=huber_delta)
                else:
                    raise ValueError("loss_type must be 'mse' or 'huber'")

                va_losses.append(loss.item())

        va_mean = float(np.mean(va_losses))

        hist["train"].append(tr_mean)
        hist["val"].append(va_mean)

        # -----------------------
        # Early stopping
        # -----------------------
        improved = (best_val - va_mean) > min_delta

        if improved:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose_every and (ep == 1 or ep % verbose_every == 0):
            print(
                f"[ep {ep:4d}] train={tr_mean:.6f}  "
                f"val={va_mean:.6f}  best={best_val:.6f}  "
                f"bad={bad}/{patience}"
            )

        if bad >= patience:
            print(f"Early stopping at ep {ep} (best val={best_val:.6f})")
            break

    # Restore best model
    if best_state is not None:
        net.load_state_dict(best_state)

    # -----------------------
    # Result dict
    # -----------------------
    res = {
        "model": net,
        "device": str(device),
        "train_cols": list(train_cols),
        "y_cols": list(y_cols),
        "loss_history": hist,
        "df_splits_scaled": (df_tr, df_va, df_te),
        "scalers": scalers,
        "mode_dir": mode_dir,
        "hyperparams": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "min_delta": min_delta,
            "loss_type": loss_type,
            "huber_delta": huber_delta,
            "strict_init": strict_init,
            "used_init_state": (init_state_dict is not None),
        },
    }

    return res
# ============================================================
# Save run
# ============================================================
def save_run(res: Dict[str, Any], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    torch.save(res["model"].state_dict(), os.path.join(save_dir, "model_state.pt"))
    joblib.dump(res.get("scalers", None), os.path.join(save_dir, "scalers.joblib"))

    meta = {
        "train_cols": res.get("train_cols", None),
        "y_cols": res.get("y_cols", None),
        "device": res.get("device", None),
        "mode_dir": res.get("mode_dir", None),
        "loss_history": res.get("loss_history", None),
        "hyperparams": res.get("hyperparams", None),
    }
    joblib.dump(meta, os.path.join(save_dir, "meta.joblib"))


# ============================================================
# Load run
# ============================================================
def load_run(save_dir: str, device: str | torch.device = None) -> Dict[str, Any]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = joblib.load(os.path.join(save_dir, "meta.joblib"))
    scalers = joblib.load(os.path.join(save_dir, "scalers.joblib"))

    train_cols = meta["train_cols"]
    y_cols = meta["y_cols"]

    net = model.ConditionalVelocityField(
        x_dim=len(y_cols),
        context_dim=len(train_cols),
    ).to(device)

    state = torch.load(os.path.join(save_dir, "model_state.pt"), map_location="cpu")
    net.load_state_dict(state)

    res = {
        "model": net,
        "device": str(device),
        "train_cols": train_cols,
        "y_cols": y_cols,
        "scalers": scalers,
        "mode_dir": meta.get("mode_dir", None),
        "loss_history": meta.get("loss_history", None),
        "hyperparams": meta.get("hyperparams", None),
        "save_dir": save_dir,
    }

    return res