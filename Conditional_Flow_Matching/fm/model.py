# fm/model.py
from __future__ import annotations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# -----------------------
# Dataset
# -----------------------
class TabularFMDataset(Dataset): # serve per far funzionare il DataLoader
    # Costruttore:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32) # context features
        self.y = torch.tensor(y, dtype=torch.float32) # target features

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]

# -> il modello impara y = f(X) tramite flow matching generativo

# -----------------------
# Conditional velocity field v_theta(t, x_t, context) 
# (t as scalar)
# -----------------------
class ConditionalVelocityField(nn.Module): # -> Rete neurale 

    DEFAULT_HIDDEN = (256, 256, 256) # quindi la rete è : input -> Linear(in_dim->256) SiLU -> Linear(256->256) SiLU -> Linear(256->256) SiLU -> Linear(256->x_dim)
    DEFAULT_DROPOUT = 0.0

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        hidden=None,
        dropout=None,
    ):
        super().__init__()

        if hidden is None:
            hidden = self.DEFAULT_HIDDEN
        if dropout is None:
            dropout = self.DEFAULT_DROPOUT
        
        # Architettura:
        in_dim = x_dim + context_dim + 1  # +1 for scalar t

        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0.0:
                layers += [nn.Dropout(dropout)]
            d = h

        layers += [nn.Linear(d, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x_t: torch.Tensor, context: torch.Tensor):
        # t: (B,)
        # x_t: (B, x_dim)
        # context: (B, context_dim)
        t = t[:, None]  # (B, 1)
        # Concatenazione input:
        inp = torch.cat([x_t, context, t], dim=-1) # input della rete
        return self.net(inp) # output: v_pred, cioè v_theta(t, x_t, context)


# -----------------------
# Flow Matching loss (linear path)
# -----------------------
def fm_loss(model: nn.Module, context: torch.Tensor, x1: torch.Tensor, device):
    """
    context: (B, C)
    x1:      (B, x_dim)
    """
    B, x_dim = x1.shape

    t = torch.rand(B, device=device)
    x0 = torch.randn(B, x_dim, device=device)

    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    v_star = x1 - x0

    v_pred = model(t, x_t, context)
    return torch.mean((v_pred - v_star) ** 2)


# -----------------------
# Sampling via Euler integration
# -----------------------
@torch.no_grad()
def sample_y(model: nn.Module, context: torch.Tensor, n_steps: int = 50, device="cpu"):
    """
    context: (B, C)
    returns: (B, x_dim)
    """
    model.eval()

    B = context.shape[0]
    x_dim = model.net[-1].out_features

    x = torch.randn(B, x_dim, device=device)
    dt = 1.0 / n_steps

    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device)
        v = model(t, x, context)
        x = x + dt * v

    return x


# -----------------------
# Prediction helper for ANY scaling mode (uses res["scalers"])
# -----------------------
@torch.no_grad()
def predict_tau_corr_auto(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    n_steps: int = 60,
    batch_size: int = 8192,
):
    model_ = res["model"]
    device = res.get("device", "cpu")

    scalers = res.get("scalers", None) or {
        "mode": "X_NONE__Y_NONE",
        "class_col": "class",
        "x_scaler": None,
        "y_scaler": None,
    }
    mode = scalers.get("mode", "X_NONE__Y_NONE")
    class_col = scalers.get("class_col", "class")

    if mode not in {
        "X_NONE__Y_NONE",
        "X_GLOBAL__Y_LOCAL",
        "X_GLOBAL__Y_GLOBAL",
        "X_LOCAL__Y_LOCAL",
        "X_NONE__Y_LOCAL",
    }:
        raise ValueError(f"Unknown scaling mode in res['scalers']: {mode}")

    if class_col not in df.columns:
        raise ValueError(f"Missing class column '{class_col}' in df.")

    x_scaler = scalers.get("x_scaler", None)
    y_scaler = scalers.get("y_scaler", None)

    def scale_X(df_chunk: pd.DataFrame) -> np.ndarray:
        X = df_chunk[train_cols].to_numpy(dtype=np.float32)

        if x_scaler is None:
            return X

        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        if isinstance(x_scaler, StandardScaler):
            return x_scaler.transform(X).astype(np.float32, copy=False)

        if isinstance(x_scaler, dict):
            Xs = np.empty_like(X, dtype=np.float32)
            if m0.any():
                Xs[m0] = x_scaler[0].transform(X[m0])
            if m1.any():
                Xs[m1] = x_scaler[1].transform(X[m1])
            return Xs

        raise ValueError("x_scaler must be None, a StandardScaler, or a dict {0,1}.")

    def inverse_Y(df_chunk: pd.DataFrame, y_hat_np: np.ndarray) -> np.ndarray:
        if y_scaler is None:
            return y_hat_np.astype(np.float32, copy=False)

        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        if isinstance(y_scaler, StandardScaler):
            return y_scaler.inverse_transform(y_hat_np).astype(np.float32, copy=False)

        if isinstance(y_scaler, dict):
            yp = np.empty_like(y_hat_np, dtype=np.float32)
            if m0.any():
                yp[m0] = y_scaler[0].inverse_transform(y_hat_np[m0])
            if m1.any():
                yp[m1] = y_scaler[1].inverse_transform(y_hat_np[m1])
            return yp

        raise ValueError("y_scaler must be None, a StandardScaler, or a dict {0,1}.")

    N = len(df)
    preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        df_b = df.iloc[start:end]

        X_scaled = scale_X(df_b)
        ctx = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        y_hat = sample_y(model_, ctx, n_steps=n_steps, device=device)
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float32, copy=False)

        y_phys = inverse_Y(df_b, y_hat_np)
        preds.append(y_phys)

    y_pred = np.concatenate(preds, axis=0)

    out = pd.DataFrame(
        y_pred,
        columns=[f"{c}_pred" for c in y_cols],
        index=df.index,
    )

    if len(y_cols) >= 2:
        out.rename(
            columns={
                f"{y_cols[0]}_pred": "tau1_corr_pred",
                f"{y_cols[1]}_pred": "tau2_corr_pred",
            },
            inplace=True,
        )

    return out


# ===========================
# Huber Loss Model
# ===========================

def fm_loss_huber(
    model: nn.Module,
    context: torch.Tensor,
    x1: torch.Tensor,
    device,
    delta: float = 1.0,
):
    """
    Same as fm_loss but using Huber loss instead of MSE.
    """
    B, x_dim = x1.shape

    t = torch.rand(B, device=device)
    x0 = torch.randn(B, x_dim, device=device)

    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    v_star = x1 - x0

    v_pred = model(t, x_t, context)

    huber = nn.HuberLoss(delta=delta)
    return huber(v_pred, v_star)




# Per applicare nell'inference gli scaler di Higgs anche a Drell-Yan:
@torch.no_grad()
def predict_tau_corr_auto_new(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    n_steps: int = 60,
    batch_size: int = 8192,
    forced_scaler_class: int | None = None,
):
    model_ = res["model"]
    device = res.get("device", "cpu")

    scalers = res.get("scalers", None) or {
        "mode": "X_NONE__Y_NONE",
        "class_col": "class",
        "x_scaler": None,
        "y_scaler": None,
    }
    mode = scalers.get("mode", "X_NONE__Y_NONE")
    class_col = scalers.get("class_col", "class")

    if mode not in {
        "X_NONE__Y_NONE",
        "X_GLOBAL__Y_LOCAL",
        "X_GLOBAL__Y_GLOBAL",
        "X_LOCAL__Y_LOCAL",
        "X_NONE__Y_LOCAL",
    }:
        raise ValueError(f"Unknown scaling mode in res['scalers']: {mode}")

    if class_col not in df.columns:
        raise ValueError(f"Missing class column '{class_col}' in df.")

    if forced_scaler_class is not None and forced_scaler_class not in {0, 1}:
        raise ValueError("forced_scaler_class must be None, 0, or 1.")

    x_scaler = scalers.get("x_scaler", None)
    y_scaler = scalers.get("y_scaler", None)

    def scale_X(df_chunk: pd.DataFrame) -> np.ndarray:
        X = df_chunk[train_cols].to_numpy(dtype=np.float32)

        if x_scaler is None:
            return X

        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        if isinstance(x_scaler, StandardScaler):
            return x_scaler.transform(X).astype(np.float32, copy=False)

        if isinstance(x_scaler, dict):
            Xs = np.empty_like(X, dtype=np.float32)

            if forced_scaler_class is not None:
                Xs[:] = x_scaler[forced_scaler_class].transform(X)
                return Xs

            if m0.any():
                Xs[m0] = x_scaler[0].transform(X[m0])
            if m1.any():
                Xs[m1] = x_scaler[1].transform(X[m1])
            return Xs

        raise ValueError("x_scaler must be None, a StandardScaler, or a dict {0,1}.")

    def inverse_Y(df_chunk: pd.DataFrame, y_hat_np: np.ndarray) -> np.ndarray:
        if y_scaler is None:
            return y_hat_np.astype(np.float32, copy=False)

        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        if isinstance(y_scaler, StandardScaler):
            return y_scaler.inverse_transform(y_hat_np).astype(np.float32, copy=False)

        if isinstance(y_scaler, dict):
            yp = np.empty_like(y_hat_np, dtype=np.float32)

            if forced_scaler_class is not None:
                yp[:] = y_scaler[forced_scaler_class].inverse_transform(y_hat_np)
                return yp

            if m0.any():
                yp[m0] = y_scaler[0].inverse_transform(y_hat_np[m0])
            if m1.any():
                yp[m1] = y_scaler[1].inverse_transform(y_hat_np[m1])
            return yp

        raise ValueError("y_scaler must be None, a StandardScaler, or a dict {0,1}.")

    N = len(df)
    preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        df_b = df.iloc[start:end]

        X_scaled = scale_X(df_b)
        ctx = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        y_hat = sample_y(model_, ctx, n_steps=n_steps, device=device)
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float32, copy=False)

        y_phys = inverse_Y(df_b, y_hat_np)
        preds.append(y_phys)

    y_pred = np.concatenate(preds, axis=0)

    out = pd.DataFrame(
        y_pred,
        columns=[f"{c}_pred" for c in y_cols],
        index=df.index,
    )

    if len(y_cols) >= 2:
        out.rename(
            columns={
                f"{y_cols[0]}_pred": "tau1_corr_pred",
                f"{y_cols[1]}_pred": "tau2_corr_pred",
            },
            inplace=True,
        )

    return out



# to accept that only some variables are scaled
# -----------------------
# Prediction helper for ANY scaling mode (uses res["scalers"])
# -----------------------
@torch.no_grad()
def predict_tau_corr_auto_new(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    n_steps: int = 60,
    batch_size: int = 8192,
):
    model_ = res["model"]
    device = res.get("device", "cpu")

    scalers = res.get("scalers", None) or {
        "mode": "X_NONE__Y_NONE",
        "class_col": "class",
        "x_scaler": None,
        "y_scaler": None,
        "x_scale_cols": list(train_cols),
    }
    mode = scalers.get("mode", "X_NONE__Y_NONE")
    class_col = scalers.get("class_col", "class")
    x_scale_cols = scalers.get("x_scale_cols", list(train_cols))

    if mode not in {
        "X_NONE__Y_NONE",
        "X_GLOBAL__Y_LOCAL",
        "X_GLOBAL__Y_GLOBAL",
        "X_LOCAL__Y_LOCAL",
        "X_NONE__Y_LOCAL",
        "X_LOCAL__Y_GLOBAL",

    }:
        raise ValueError(f"Unknown scaling mode in res['scalers']: {mode}")

    if class_col not in df.columns:
        raise ValueError(f"Missing class column '{class_col}' in df.")

    x_scaler = scalers.get("x_scaler", None)
    y_scaler = scalers.get("y_scaler", None)

    def scale_X(df_chunk: pd.DataFrame) -> np.ndarray:
        # full context seen by the model
        X_full = df_chunk[train_cols].to_numpy(dtype=np.float32)

        if x_scaler is None:
            return X_full

        # subset of columns that must actually be scaled
        X_sub = df_chunk[x_scale_cols].to_numpy(dtype=np.float32)

        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        # start from original full array, then overwrite only scaled cols
        Xs_full = X_full.copy()

        if isinstance(x_scaler, StandardScaler):
            Xs_sub = x_scaler.transform(X_sub).astype(np.float32, copy=False)
            for j, col in enumerate(x_scale_cols):
                idx = train_cols.index(col)
                Xs_full[:, idx] = Xs_sub[:, j]
            return Xs_full

        if isinstance(x_scaler, dict):
            Xs_sub = X_sub.copy()

            if m0.any():
                Xs_sub[m0] = x_scaler[0].transform(X_sub[m0]).astype(np.float32, copy=False)
            if m1.any():
                Xs_sub[m1] = x_scaler[1].transform(X_sub[m1]).astype(np.float32, copy=False)

            for j, col in enumerate(x_scale_cols):
                idx = train_cols.index(col)
                Xs_full[:, idx] = Xs_sub[:, j]

            return Xs_full

        raise ValueError("x_scaler must be None, a StandardScaler, or a dict {0,1}.")

    def inverse_Y(df_chunk: pd.DataFrame, y_hat_np: np.ndarray) -> np.ndarray:
        if y_scaler is None:
            return y_hat_np.astype(np.float32, copy=False)

        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        if isinstance(y_scaler, StandardScaler):
            return y_scaler.inverse_transform(y_hat_np).astype(np.float32, copy=False)

        if isinstance(y_scaler, dict):
            yp = np.empty_like(y_hat_np, dtype=np.float32)
            if m0.any():
                yp[m0] = y_scaler[0].inverse_transform(y_hat_np[m0])
            if m1.any():
                yp[m1] = y_scaler[1].inverse_transform(y_hat_np[m1])
            return yp

        raise ValueError("y_scaler must be None, a StandardScaler, or a dict {0,1}.")

    N = len(df)
    preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        df_b = df.iloc[start:end]

        X_scaled = scale_X(df_b)
        ctx = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        y_hat = sample_y(model_, ctx, n_steps=n_steps, device=device)
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float32, copy=False)

        y_phys = inverse_Y(df_b, y_hat_np)
        preds.append(y_phys)

    y_pred = np.concatenate(preds, axis=0)

    out = pd.DataFrame(
        y_pred,
        columns=[f"{c}_pred" for c in y_cols],
        index=df.index,
    )

    if len(y_cols) >= 2:
        out.rename(
            columns={
                f"{y_cols[0]}_pred": "tau1_corr_pred",
                f"{y_cols[1]}_pred": "tau2_corr_pred",
            },
            inplace=True,
        )

    return out



# scaling utils: per controllare le predizioni prima e dopo l'inferenza
@torch.no_grad()
def predict_tau_corr_scaled_only(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    n_steps: int = 60,
    batch_size: int = 8192,
):
    model_ = res["model"]
    device = res.get("device", "cpu")

    scalers = res.get("scalers", None) or {
        "mode": "X_NONE__Y_NONE",
        "class_col": "class",
        "x_scaler": None,
        "y_scaler": None,
        "x_scale_cols": list(train_cols),
    }

    class_col = scalers.get("class_col", "class")
    x_scale_cols = scalers.get("x_scale_cols", list(train_cols))
    x_scaler = scalers.get("x_scaler", None)

    def scale_X(df_chunk: pd.DataFrame) -> np.ndarray:
        X_full = df_chunk[train_cols].to_numpy(dtype=np.float32)

        if x_scaler is None:
            return X_full

        X_sub = df_chunk[x_scale_cols].to_numpy(dtype=np.float32)
        c = df_chunk[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        Xs_full = X_full.copy()

        if isinstance(x_scaler, StandardScaler):
            Xs_sub = x_scaler.transform(X_sub).astype(np.float32, copy=False)
            for j, col in enumerate(x_scale_cols):
                idx = train_cols.index(col)
                Xs_full[:, idx] = Xs_sub[:, j]
            return Xs_full

        if isinstance(x_scaler, dict):
            Xs_sub = X_sub.copy()
            if m0.any():
                Xs_sub[m0] = x_scaler[0].transform(X_sub[m0]).astype(np.float32, copy=False)
            if m1.any():
                Xs_sub[m1] = x_scaler[1].transform(X_sub[m1]).astype(np.float32, copy=False)

            for j, col in enumerate(x_scale_cols):
                idx = train_cols.index(col)
                Xs_full[:, idx] = Xs_sub[:, j]
            return Xs_full

        raise ValueError("x_scaler must be None, a StandardScaler, or a dict {0,1}.")

    N = len(df)
    preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        df_b = df.iloc[start:end]

        X_scaled = scale_X(df_b)
        ctx = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        y_hat = sample_y(model_, ctx, n_steps=n_steps, device=device)
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float32, copy=False)
        preds.append(y_hat_np)

    y_pred_scaled = np.concatenate(preds, axis=0)

    out = pd.DataFrame(
        y_pred_scaled,
        columns=[f"{c}_pred_scaled" for c in y_cols],
        index=df.index,
    )

    return out



class DirectRegressor(nn.Module):
    """
    Regressore diretto: predice (tau1_corr, tau2_corr) da context x.
    Stessa architettura del CFM ma senza t e senza x_t in input.
    Input:  context x  (dim = context_dim)
    Output: y_pred     (dim = y_dim)
    """
    def __init__(
        self,
        y_dim: int,
        context_dim: int,
        hidden: tuple = (256, 256, 256, 256, 256),
    ):
        super().__init__()

        dims = [context_dim] + list(hidden) + [y_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


def regressor_loss_huber(
    net: DirectRegressor,
    context: torch.Tensor,
    y: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    y_pred = net(context)
    return nn.HuberLoss(delta=delta)(y_pred, y)


def predict_regressor(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """
    Inferenza del regressore diretto.

    - Scala solo le feature X.
    - Predice y nello spazio di training.
    - Applica inverse scaling a y_pred se necessario.
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    net = res["model"]
    scalers = res["scalers"]
    device = res.get("device", "cpu")

    net.eval()

    # --------------------------------------------------
    # Scaling solo di X
    # --------------------------------------------------
    X = df[train_cols].to_numpy(np.float32)

    x_scaler = scalers.get("x_scaler", None)

    if x_scaler is None:
        X_scaled = X

    elif isinstance(x_scaler, StandardScaler):
        X_scaled = x_scaler.transform(X).astype(np.float32)

    elif isinstance(x_scaler, dict):

        if "class" not in df.columns:
            raise ValueError(
                "Local x_scaler requires a 'class' column in df."
            )

        X_scaled = np.empty_like(X, dtype=np.float32)
        classes = df["class"].to_numpy()

        for cls, scaler in x_scaler.items():
            mask = classes == cls
            if mask.any():
                X_scaled[mask] = scaler.transform(X[mask]).astype(np.float32)

    else:
        raise TypeError(f"Unknown x_scaler type: {type(x_scaler)}")

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):

            xb = torch.tensor(
                X_scaled[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            yb = net(xb).cpu().numpy()
            preds.append(yb)

    preds = np.concatenate(preds, axis=0)

    # --------------------------------------------------
    # Inverse transform delle predizioni Y
    # --------------------------------------------------
    y_scaler = scalers.get("y_scaler", None)

    if y_scaler is None:
        preds_phys = preds

    elif isinstance(y_scaler, StandardScaler):
        preds_phys = y_scaler.inverse_transform(preds)

    elif isinstance(y_scaler, dict):

        if "class" not in df.columns:
            raise ValueError(
                "Local y_scaler requires a 'class' column in df."
            )

        preds_phys = np.empty_like(preds, dtype=np.float32)
        classes = df["class"].to_numpy()

        for cls, scaler in y_scaler.items():
            mask = classes == cls
            if mask.any():
                preds_phys[mask] = scaler.inverse_transform(preds[mask])

    else:
        raise TypeError(f"Unknown y_scaler type: {type(y_scaler)}")

    out = pd.DataFrame(
        {
            y_cols[0].replace("_corr", "_corr_pred"): preds_phys[:, 0],
            y_cols[1].replace("_corr", "_corr_pred"): preds_phys[:, 1],
        },
        index=df.index,
    )

    return out


def predict_regressor_from_prepared(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """
    Inferenza del regressore su dataframe già preparato/scalato.
    NON scala X.
    Fa solo inverse scaling delle predizioni y.
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    net = res["model"]
    scalers = res["scalers"]
    device = res.get("device", "cpu")

    net.eval()

    X = df[train_cols].to_numpy(np.float32)

    preds = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            yb = net(xb).cpu().numpy()
            preds.append(yb)

    preds = np.concatenate(preds, axis=0)

    y_scaler = scalers.get("y_scaler", None)

    if y_scaler is None:
        preds_phys = preds

    elif isinstance(y_scaler, StandardScaler):
        preds_phys = y_scaler.inverse_transform(preds)

    elif isinstance(y_scaler, dict):
        preds_phys = np.empty_like(preds)

        classes = df["class"].to_numpy()

        for cls, scaler in y_scaler.items():
            mask = classes == cls
            if mask.any():
                preds_phys[mask] = scaler.inverse_transform(preds[mask])

    else:
        raise TypeError(f"Unknown y_scaler type: {type(y_scaler)}")

    return pd.DataFrame(
        {
            y_cols[0].replace("_corr", "_corr_pred"): preds_phys[:, 0],
            y_cols[1].replace("_corr", "_corr_pred"): preds_phys[:, 1],
        },
        index=df.index,
    )
    
    
# =================================
# MULTICLASS (Per TTbar)
# =================================

@torch.no_grad()
def predict_tau_corr_auto_multiclass(
    df: pd.DataFrame,
    res: dict,
    train_cols: list,
    y_cols: list,
    n_steps: int = 60,
    batch_size: int = 8192,
):
    model_ = res["model"]
    device = res.get("device", "cpu")

    scalers = res.get("scalers", None) or {
        "mode": "X_NONE__Y_NONE",
        "class_col": "class",
        "x_scaler": None,
        "y_scaler": None,
        "x_scale_cols": list(train_cols),
    }

    mode = scalers.get("mode", "X_NONE__Y_NONE")
    class_col = scalers.get("class_col", "class")
    x_scale_cols = scalers.get("x_scale_cols", list(train_cols))

    if class_col not in df.columns:
        raise ValueError(f"Missing class column '{class_col}' in df.")

    x_scaler = scalers.get("x_scaler", None)
    y_scaler = scalers.get("y_scaler", None)

    def scale_X(df_chunk: pd.DataFrame) -> np.ndarray:
        X_full = df_chunk[train_cols].to_numpy(dtype=np.float32)

        if x_scaler is None:
            return X_full

        X_sub = df_chunk[x_scale_cols].to_numpy(dtype=np.float32)
        Xs_full = X_full.copy()

        if isinstance(x_scaler, StandardScaler):
            Xs_sub = x_scaler.transform(X_sub).astype(np.float32, copy=False)

            for j, col in enumerate(x_scale_cols):
                idx = train_cols.index(col)
                Xs_full[:, idx] = Xs_sub[:, j]

            return Xs_full

        if isinstance(x_scaler, dict):
            Xs_sub = np.empty_like(X_sub, dtype=np.float32)
            classes = df_chunk[class_col].to_numpy()

            for cls, scaler in x_scaler.items():
                mask = classes == cls
                if mask.any():
                    Xs_sub[mask] = scaler.transform(
                        X_sub[mask]
                    ).astype(np.float32, copy=False)

            for j, col in enumerate(x_scale_cols):
                idx = train_cols.index(col)
                Xs_full[:, idx] = Xs_sub[:, j]

            return Xs_full

        raise ValueError("x_scaler must be None, a StandardScaler, or a dict.")

    def inverse_Y(df_chunk: pd.DataFrame, y_hat_np: np.ndarray) -> np.ndarray:

        if y_scaler is None:
            return y_hat_np.astype(np.float32, copy=False)

        if isinstance(y_scaler, StandardScaler):
            return y_scaler.inverse_transform(
                y_hat_np
            ).astype(np.float32, copy=False)

        if isinstance(y_scaler, dict):
            yp = np.empty_like(y_hat_np, dtype=np.float32)
            classes = df_chunk[class_col].to_numpy()

            for cls, scaler in y_scaler.items():
                mask = classes == cls
                if mask.any():
                    yp[mask] = scaler.inverse_transform(
                        y_hat_np[mask]
                    ).astype(np.float32, copy=False)

            return yp

        raise ValueError("y_scaler must be None, a StandardScaler, or a dict.")

    N = len(df)
    preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        df_b = df.iloc[start:end]

        X_scaled = scale_X(df_b)
        ctx = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        y_hat = sample_y(model_, ctx, n_steps=n_steps, device=device)
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float32, copy=False)

        y_phys = inverse_Y(df_b, y_hat_np)
        preds.append(y_phys)

    y_pred = np.concatenate(preds, axis=0)

    out = pd.DataFrame(
        y_pred,
        columns=[f"{c}_pred" for c in y_cols],
        index=df.index,
    )

    if len(y_cols) >= 2:
        out.rename(
            columns={
                f"{y_cols[0]}_pred": "tau1_corr_pred",
                f"{y_cols[1]}_pred": "tau2_corr_pred",
            },
            inplace=True,
        )

    return out