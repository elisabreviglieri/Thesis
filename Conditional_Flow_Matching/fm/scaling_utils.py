import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from . import model



def transform_X_global(
    df_in: pd.DataFrame,
    train_cols: list,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply a global StandardScaler to the X features.
    The same scaler is applied to all events.
    """

    df_out = df_in.copy()

    X_scaled = scaler.transform(df_out[train_cols].to_numpy())
    df_out.loc[:, train_cols] = X_scaled

    return df_out


def transform_X_local(
    df_in: pd.DataFrame,
    train_cols: list,
    scaler_dict: dict,
    class_col: str = "class",
) -> pd.DataFrame:
    """
    Apply class-conditional scaling to X features.

    scaler_dict must be:
        {0: scaler_for_DY, 1: scaler_for_Higgs}
    """

    df_out = df_in.copy()

    m0 = df_out[class_col].to_numpy() == 0
    m1 = df_out[class_col].to_numpy() == 1

    if m0.any():
        X0 = scaler_dict[0].transform(df_out.loc[m0, train_cols].to_numpy())
        df_out.loc[m0, train_cols] = X0

    if m1.any():
        X1 = scaler_dict[1].transform(df_out.loc[m1, train_cols].to_numpy())
        df_out.loc[m1, train_cols] = X1

    return df_out


# Per i plot:
def plot_feature_raw_global_local(
    df_raw,
    df_global,
    df_local,
    feature: str,
    class_col: str = "class",
    bins: int = 80,
    density: bool = True,
    figsize=(9, 5),
):
    """
    Plot one feature comparing:
      - raw
      - global scaling
      - local scaling

    separately for:
      - DY   (class = 0)
      - Higgs (class = 1)
    """

    # masks
    m_raw_dy = df_raw[class_col] == 0
    m_raw_h  = df_raw[class_col] == 1

    m_glob_dy = df_global[class_col] == 0
    m_glob_h  = df_global[class_col] == 1

    m_loc_dy = df_local[class_col] == 0
    m_loc_h  = df_local[class_col] == 1

    plt.figure(figsize=figsize)

    # DY
    plt.hist(
        df_raw.loc[m_raw_dy, feature].dropna().to_numpy(),
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label="DY raw",
    )
    plt.hist(
        df_global.loc[m_glob_dy, feature].dropna().to_numpy(),
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label="DY global",
    )
    plt.hist(
        df_local.loc[m_loc_dy, feature].dropna().to_numpy(),
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label="DY local",
    )

    # Higgs
    plt.hist(
        df_raw.loc[m_raw_h, feature].dropna().to_numpy(),
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label="H raw",
    )
    plt.hist(
        df_global.loc[m_glob_h, feature].dropna().to_numpy(),
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label="H global",
    )
    plt.hist(
        df_local.loc[m_loc_h, feature].dropna().to_numpy(),
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label="H local",
    )

    plt.xlabel(feature)
    plt.ylabel("Density" if density else "Counts")
    plt.title(f"{feature} — raw vs global vs local")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()
    
    

# Training con una sola context feature scaled, le altre raw:
def transform_one_feature(
    df_in: pd.DataFrame,
    feature: str,
    mode: str,
    global_scaler=None,
    local_scaler_dict=None,
    class_col: str = "class",
) -> pd.DataFrame:
    """
    Return a copy of df_in where only one feature is transformed.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe (raw).
    feature : str
        Name of the feature to transform.
    mode : str
        "global" or "local".
    global_scaler : StandardScaler, optional
        Fitted scaler for the chosen feature, used if mode == "global".
    local_scaler_dict : dict, optional
        Dict {0: scaler_DY, 1: scaler_H}, used if mode == "local".
    class_col : str
        Name of the class column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with only the selected feature transformed.
    """

    if feature not in df_in.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe.")

    if mode not in {"global", "local"}:
        raise ValueError("mode must be 'global' or 'local'.")

    df_out = df_in.copy()

    # ---------- global scaling ----------
    if mode == "global":
        if global_scaler is None:
            raise ValueError("global_scaler must be provided when mode='global'.")

        vals = df_out[[feature]].to_numpy()
        vals_scaled = global_scaler.transform(vals)
        df_out.loc[:, feature] = vals_scaled[:, 0]
        return df_out

    # ---------- local scaling ----------
    if mode == "local":
        if local_scaler_dict is None:
            raise ValueError("local_scaler_dict must be provided when mode='local'.")
        if class_col not in df_out.columns:
            raise ValueError(f"Missing class column '{class_col}' in dataframe.")

        m0 = df_out[class_col].to_numpy() == 0
        m1 = df_out[class_col].to_numpy() == 1

        if m0.any():
            vals0 = df_out.loc[m0, [feature]].to_numpy()
            vals0_scaled = local_scaler_dict[0].transform(vals0)
            df_out.loc[m0, feature] = vals0_scaled[:, 0]

        if m1.any():
            vals1 = df_out.loc[m1, [feature]].to_numpy()
            vals1_scaled = local_scaler_dict[1].transform(vals1)
            df_out.loc[m1, feature] = vals1_scaled[:, 0]

        return df_out
    
    
# Per salvare gli split per i training
def save_custom_splits_for_training(
    splits_dict,
    out_root,
    run_name,
    scalers=None,
):
    """
    Save custom train/val/test dataframes in a folder compatible with training.train_one_mode.
    """
    mode_dir = os.path.join(out_root, run_name)
    os.makedirs(mode_dir, exist_ok=True)

    splits_dict["train"].to_pickle(os.path.join(mode_dir, "train.pkl"))
    splits_dict["val"].to_pickle(os.path.join(mode_dir, "val.pkl"))
    splits_dict["test"].to_pickle(os.path.join(mode_dir, "test.pkl"))

    if scalers is None:
        scalers = {
            "mode": f"CUSTOM_{run_name}",
            "class_col": "class",
            "x_scaler": None,
            "y_scaler": None,
        }

    joblib.dump(scalers, os.path.join(mode_dir, "scalers.joblib"))

    return mode_dir


# Una funzione completa per tutto (training compreso):

def train_one_feature_variant(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature: str,
    variant: str,                     # "global" or "local"
    train_cols: list,
    y_cols: list,
    class_col: str = "class",
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
    init_state_dict: dict | None = None,
    strict_init: bool = True,
):
    """
    Train a model starting from RAW splits, transforming ONLY one feature.

    - all X remain raw
    - only `feature` is transformed
    - y remains raw

    variant:
        - "global": fit one scaler on train[feature]
        - "local": fit one scaler per class on train[feature]
    """

    if variant not in {"global", "local"}:
        raise ValueError("variant must be 'global' or 'local'.")

    if feature not in train_cols:
        raise ValueError(f"Feature '{feature}' is not in train_cols.")

    if class_col not in df_train.columns:
        raise ValueError(f"Missing class column '{class_col}' in df_train.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Fit scaler(s) ONLY on TRAIN for the chosen feature
    # --------------------------------------------------
    if variant == "global":
        feature_scaler = StandardScaler().fit(df_train[[feature]].to_numpy())
    else:
        df_train_dy = df_train[df_train[class_col] == 0]
        df_train_h  = df_train[df_train[class_col] == 1]

        feature_scaler = {
            0: StandardScaler().fit(df_train_dy[[feature]].to_numpy()),
            1: StandardScaler().fit(df_train_h[[feature]].to_numpy()),
        }

    # --------------------------------------------------
    # Transform ONLY the chosen feature
    # --------------------------------------------------
    df_tr = transform_one_feature(
        df_in=df_train,
        feature=feature,
        mode=variant,
        global_scaler=feature_scaler if variant == "global" else None,
        local_scaler_dict=feature_scaler if variant == "local" else None,
        class_col=class_col,
    )

    df_va = transform_one_feature(
        df_in=df_val,
        feature=feature,
        mode=variant,
        global_scaler=feature_scaler if variant == "global" else None,
        local_scaler_dict=feature_scaler if variant == "local" else None,
        class_col=class_col,
    )

    df_te = transform_one_feature(
        df_in=df_test,
        feature=feature,
        mode=variant,
        global_scaler=feature_scaler if variant == "global" else None,
        local_scaler_dict=feature_scaler if variant == "local" else None,
        class_col=class_col,
    )

    # --------------------------------------------------
    # Build arrays
    # --------------------------------------------------
    Xtr = df_tr[train_cols].to_numpy(np.float32)
    ytr = df_tr[y_cols].to_numpy(np.float32)

    Xva = df_va[train_cols].to_numpy(np.float32)
    yva = df_va[y_cols].to_numpy(np.float32)

    # --------------------------------------------------
    # Dataset / loaders
    # --------------------------------------------------
    ds_tr = model.TabularFMDataset(Xtr, ytr)
    ds_va = model.TabularFMDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    net = model.ConditionalVelocityField(
        x_dim=len(y_cols),
        context_dim=len(train_cols),
    ).to(device)

    if init_state_dict is not None:
        missing, unexpected = net.load_state_dict(init_state_dict, strict=strict_init)
        if (not strict_init) and (missing or unexpected):
            print(f"[init_state_dict] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):

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

        improved = (best_val - va_mean) > min_delta

        if improved:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose_every and (ep == 1 or ep % verbose_every == 0):
            print(
                f"[{feature} | {variant} | ep {ep:4d}] "
                f"train={tr_mean:.6f}  val={va_mean:.6f}  "
                f"best={best_val:.6f}  bad={bad}/{patience}"
            )

        if bad >= patience:
            print(f"Early stopping at ep {ep} (best val={best_val:.6f})")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    res = {
        "model": net,
        "device": str(device),
        "train_cols": list(train_cols),
        "y_cols": list(y_cols),
        "feature_tested": feature,
        "feature_variant": variant,
        "feature_scaler": feature_scaler,
        "loss_history": hist,
        "df_splits_scaled": (df_tr, df_va, df_te),
        "scalers": {
            "mode": f"ONE_FEATURE_{variant.upper()}",
            "class_col": class_col,
            "x_scaler": None,
            "y_scaler": None,
            "feature_tested": feature,
            "feature_variant": variant,
        },
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


# Per plottare tutte le variabili in una griglia:
def plot_feature_grid_raw_global_local(
    df_raw,
    df_global,
    df_local,
    features,
    class_col="class",
    bins=80,
    density=True,
    ncols=4,
    figsize=(16, 10),
):
    n = len(features)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    mask_dy = df_raw[class_col] == 0
    mask_h  = df_raw[class_col] == 1

    for i, feat in enumerate(features):
        ax = axes[i]

        ax.hist(
            df_raw.loc[mask_dy, feat],
            bins=bins,
            density=density,
            histtype="step",
            label="raw DY",
        )

        ax.hist(
            df_global.loc[mask_dy, feat],
            bins=bins,
            density=density,
            histtype="step",
            label="global DY",
        )

        ax.hist(
            df_local.loc[mask_dy, feat],
            bins=bins,
            density=density,
            histtype="step",
            label="local DY",
        )

        ax.set_title(feat)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

# Per plottare Drell-Yan e Higgs separatamente:
def plot_feature_grid_raw_global_local(
    df_raw,
    df_global,
    df_local,
    features,
    class_value=0,
    class_col="class",
    bins=80,
    density=True,
    ncols=4,
    figsize=(16, 10),
):
    n = len(features)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    mask = df_raw[class_col] == class_value

    class_name = "DY" if class_value == 0 else "Higgs"

    for i, feat in enumerate(features):
        ax = axes[i]

        ax.hist(
            df_raw.loc[mask, feat],
            bins=bins,
            density=density,
            histtype="step",
            label=f"raw {class_name}",
        )

        ax.hist(
            df_global.loc[mask, feat],
            bins=bins,
            density=density,
            histtype="step",
            label=f"global {class_name}",
        )

        ax.hist(
            df_local.loc[mask, feat],
            bins=bins,
            density=density,
            histtype="step",
            label=f"local {class_name}",
        )

        ax.set_title(feat)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    
    
# Per plottare Drell-Yan e Higgs insieme:
def plot_feature_grid_both_classes(
    df_raw,
    df_global,
    df_local,
    features,
    class_col="class",
    bins=60,
    density=True,
    ncols=4,
    figsize=(16, 12),
    qlow=0.01,
    qhigh=0.99,
):
    import numpy as np
    import matplotlib.pyplot as plt

    mask_dy = df_raw[class_col] == 0
    mask_h  = df_raw[class_col] == 1

    n = len(features)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(features):
        ax = axes[i]

        x_raw_dy  = df_raw.loc[mask_dy, feat].dropna().to_numpy()
        x_glob_dy = df_global.loc[mask_dy, feat].dropna().to_numpy()
        x_loc_dy  = df_local.loc[mask_dy, feat].dropna().to_numpy()

        x_raw_h  = df_raw.loc[mask_h, feat].dropna().to_numpy()
        x_glob_h = df_global.loc[mask_h, feat].dropna().to_numpy()
        x_loc_h  = df_local.loc[mask_h, feat].dropna().to_numpy()

        all_vals = np.concatenate([x_raw_dy, x_glob_dy, x_loc_dy,
                                   x_raw_h, x_glob_h, x_loc_h])

        if len(all_vals) == 0:
            ax.set_title(feat)
            ax.axis("off")
            continue

        xmin = np.quantile(all_vals, qlow)
        xmax = np.quantile(all_vals, qhigh)

        if xmin == xmax:
            xmin -= 1e-6
            xmax += 1e-6

        # DY
        ax.hist(x_raw_dy,  bins=bins, range=(xmin, xmax), density=density,
                histtype="step", linewidth=1.4, label="DY raw")
        ax.hist(x_glob_dy, bins=bins, range=(xmin, xmax), density=density,
                histtype="step", linewidth=1.4, label="DY global")
        ax.hist(x_loc_dy,  bins=bins, range=(xmin, xmax), density=density,
                histtype="step", linewidth=1.4, label="DY local")

        # Higgs
        ax.hist(x_raw_h,  bins=bins, range=(xmin, xmax), density=density,
                histtype="step", linewidth=1.4, linestyle="--", label="H raw")
        ax.hist(x_glob_h, bins=bins, range=(xmin, xmax), density=density,
                histtype="step", linewidth=1.4, linestyle="--", label="H global")
        ax.hist(x_loc_h,  bins=bins, range=(xmin, xmax), density=density,
                histtype="step", linewidth=1.4, linestyle="--", label="H local")

        ax.set_title(feat, fontsize=10)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
    
# Confronto raw-local:
def plot_feature_grid_raw_local(
    df_raw,
    df_local,
    features,
    class_value=0,
    class_col="class",
    bins=80,
    density=True,
    ncols=4,
    figsize=(16, 6),
):

    import numpy as np
    import matplotlib.pyplot as plt

    n = len(features)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    mask = df_raw[class_col] == class_value
    class_name = "DY" if class_value == 0 else "Higgs"

    for i, feat in enumerate(features):
        ax = axes[i]

        ax.hist(
            df_raw.loc[mask, feat],
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"raw {class_name}",
        )

        ax.hist(
            df_local.loc[mask, feat],
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"local {class_name}",
        )

        ax.set_title(feat)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    
    
# Per la griglia dei plot delle masse:
def plot_mass_grid_vs_baseline(
    masses_feature_dict,
    baseline_mass,
    features,
    label_feature="only local",
    label_baseline="baseline",
    bins=120,
    range_=(0, 300),
    density=True,
    ncols=3,
    figsize=(15, 10),
):
    """
    Plot a grid of invariant-mass histograms:
      - baseline
      - one feature variant

    Parameters
    ----------
    masses_feature_dict : dict
        Dict like:
            masses_feature_dict[feat]["local"] = np.ndarray
        or
            masses_feature_dict[feat] = np.ndarray
    baseline_mass : np.ndarray
        Baseline mass distribution.
    features : list
        List of features to show in the grid.
    label_feature : str
        Label for the feature-based curve.
    label_baseline : str
        Label for the baseline curve.
    """

    n = len(features)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(features):
        ax = axes[i]

        # support both:
        # masses_feature_dict[feat]["local"]
        # masses_feature_dict[feat]
        vals = masses_feature_dict[feat]
        if isinstance(vals, dict):
            mass_feat = vals["local"]
        else:
            mass_feat = vals

        ax.hist(
            baseline_mass,
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=1.8,
            label=label_baseline,
        )

        ax.hist(
            mass_feat,
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=1.8,
            label=label_feature,
        )

        ax.set_title(feat, fontsize=10)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    
    
# Per training cumulativo: 
def train_selected_features_variant(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_features: list,
    variant: str,                     # "global" or "local"
    train_cols: list,
    y_cols: list,
    class_col: str = "class",
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
    init_state_dict: dict | None = None,
    strict_init: bool = True,
):
    """
    Train a model starting from RAW splits, transforming ONLY the features
    listed in `selected_features`.

    - all X remain raw by default
    - only selected_features are transformed
    - y remains raw

    variant:
        - "global": fit one scaler on train[feature] for each selected feature
        - "local": fit one scaler per class on train[feature] for each selected feature
    """

    if variant not in {"global", "local"}:
        raise ValueError("variant must be 'global' or 'local'.")

    if class_col not in df_train.columns:
        raise ValueError(f"Missing class column '{class_col}' in df_train.")

    for feat in selected_features:
        if feat not in train_cols:
            raise ValueError(f"Feature '{feat}' is not in train_cols.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_features = list(selected_features)

    # --------------------------------------------------
    # Fit scaler(s) ONLY on TRAIN for each chosen feature
    # --------------------------------------------------
    feature_scalers = {}

    if variant == "global":
        for feat in selected_features:
            feature_scalers[feat] = StandardScaler().fit(df_train[[feat]].to_numpy())

    else:
        df_train_dy = df_train[df_train[class_col] == 0]
        df_train_h  = df_train[df_train[class_col] == 1]

        for feat in selected_features:
            feature_scalers[feat] = {
                0: StandardScaler().fit(df_train_dy[[feat]].to_numpy()),
                1: StandardScaler().fit(df_train_h[[feat]].to_numpy()),
            }

    # --------------------------------------------------
    # Transform ONLY the chosen features
    # --------------------------------------------------
    df_tr = df_train.copy()
    df_va = df_val.copy()
    df_te = df_test.copy()

    for feat in selected_features:
        feat_scaler = feature_scalers[feat]

        df_tr = transform_one_feature(
            df_in=df_tr,
            feature=feat,
            mode=variant,
            global_scaler=feat_scaler if variant == "global" else None,
            local_scaler_dict=feat_scaler if variant == "local" else None,
            class_col=class_col,
        )

        df_va = transform_one_feature(
            df_in=df_va,
            feature=feat,
            mode=variant,
            global_scaler=feat_scaler if variant == "global" else None,
            local_scaler_dict=feat_scaler if variant == "local" else None,
            class_col=class_col,
        )

        df_te = transform_one_feature(
            df_in=df_te,
            feature=feat,
            mode=variant,
            global_scaler=feat_scaler if variant == "global" else None,
            local_scaler_dict=feat_scaler if variant == "local" else None,
            class_col=class_col,
        )

    # --------------------------------------------------
    # Build arrays
    # --------------------------------------------------
    Xtr = df_tr[train_cols].to_numpy(np.float32)
    ytr = df_tr[y_cols].to_numpy(np.float32)

    Xva = df_va[train_cols].to_numpy(np.float32)
    yva = df_va[y_cols].to_numpy(np.float32)

    # --------------------------------------------------
    # Dataset / loaders
    # --------------------------------------------------
    ds_tr = model.TabularFMDataset(Xtr, ytr)
    ds_va = model.TabularFMDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    net = model.ConditionalVelocityField(
        x_dim=len(y_cols),
        context_dim=len(train_cols),
    ).to(device)

    if init_state_dict is not None:
        missing, unexpected = net.load_state_dict(init_state_dict, strict=strict_init)
        if (not strict_init) and (missing or unexpected):
            print(f"[init_state_dict] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    bad = 0

    tag = f"{len(selected_features)}feat"

    for ep in range(1, epochs + 1):

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

        improved = (best_val - va_mean) > min_delta

        if improved:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose_every and (ep == 1 or ep % verbose_every == 0):
            print(
                f"[{tag} | {variant} | ep {ep:4d}] "
                f"train={tr_mean:.6f}  val={va_mean:.6f}  "
                f"best={best_val:.6f}  bad={bad}/{patience}"
            )

        if bad >= patience:
            print(f"Early stopping at ep {ep} (best val={best_val:.6f})")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    res = {
        "model": net,
        "device": str(device),
        "train_cols": list(train_cols),
        "y_cols": list(y_cols),
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "feature_variant": variant,
        "feature_scalers": feature_scalers,
        "loss_history": hist,
        "df_splits_scaled": (df_tr, df_va, df_te),
        "scalers": {
            "mode": f"{len(selected_features)}_FEATURES_{variant.upper()}",
            "class_col": class_col,
            "x_scaler": None,
            "y_scaler": None,
            "selected_features": selected_features,
            "feature_variant": variant,
        },
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



def train_selected_features_variant_with_y(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_features: list,
    x_variant: str,                  # "global" or "local"
    y_variant: str,                  # "none", "global", "local"
    train_cols: list,
    y_cols: list,
    class_col: str = "class",
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
    init_state_dict: dict | None = None,
    strict_init: bool = True,
):
    """
    Train from RAW splits:
      - only selected X features are transformed with x_variant
      - Y is transformed with y_variant
      - all other X remain raw

    x_variant: "global" or "local"
    y_variant: "none", "global", or "local"
    """

    if x_variant not in {"global", "local"}:
        raise ValueError("x_variant must be 'global' or 'local'.")
    if y_variant not in {"none", "global", "local"}:
        raise ValueError("y_variant must be 'none', 'global', or 'local'.")
    if class_col not in df_train.columns:
        raise ValueError(f"Missing class column '{class_col}' in df_train.")

    selected_features = list(selected_features)

    for feat in selected_features:
        if feat not in train_cols:
            raise ValueError(f"Feature '{feat}' is not in train_cols.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Fit X scalers only on TRAIN for selected features
    # --------------------------------------------------
    x_feature_scalers = {}

    if x_variant == "global":
        for feat in selected_features:
            x_feature_scalers[feat] = StandardScaler().fit(df_train[[feat]].to_numpy())
    else:
        df_train_dy = df_train[df_train[class_col] == 0]
        df_train_h  = df_train[df_train[class_col] == 1]

        for feat in selected_features:
            x_feature_scalers[feat] = {
                0: StandardScaler().fit(df_train_dy[[feat]].to_numpy()),
                1: StandardScaler().fit(df_train_h[[feat]].to_numpy()),
            }

    # --------------------------------------------------
    # Fit Y scaler(s) only on TRAIN
    # --------------------------------------------------
    if y_variant == "none":
        y_scaler = None

    elif y_variant == "global":
        y_scaler = StandardScaler().fit(df_train[y_cols].to_numpy())

    else:  # local
        df_train_dy = df_train[df_train[class_col] == 0]
        df_train_h  = df_train[df_train[class_col] == 1]

        y_scaler = {
            0: StandardScaler().fit(df_train_dy[y_cols].to_numpy()),
            1: StandardScaler().fit(df_train_h[y_cols].to_numpy()),
        }

    # --------------------------------------------------
    # Helper: transform Y only
    # --------------------------------------------------
    def transform_y_only(df_in, y_scaler, y_variant, y_cols, class_col):
        out = df_in.copy()

        def _assign_cols(mask, cols, arr_2d):
            for j, col in enumerate(cols):
                target_dtype = out[col].dtype
                out.loc[mask, col] = arr_2d[:, j].astype(target_dtype, copy=False)

        if y_variant == "none":
            return out

        if y_variant == "global":
            Y = y_scaler.transform(out[y_cols].to_numpy())
            _assign_cols(slice(None), y_cols, Y)
            return out

        # local
        c = out[class_col].to_numpy()
        m0 = (c == 0)
        m1 = (c == 1)

        if m0.any():
            Y0 = y_scaler[0].transform(out.loc[m0, y_cols].to_numpy())
            _assign_cols(m0, y_cols, Y0)

        if m1.any():
            Y1 = y_scaler[1].transform(out.loc[m1, y_cols].to_numpy())
            _assign_cols(m1, y_cols, Y1)

        return out

    # --------------------------------------------------
    # Transform selected X + transform Y
    # --------------------------------------------------
    df_tr = df_train.copy()
    df_va = df_val.copy()
    df_te = df_test.copy()

    for feat in selected_features:
        feat_scaler = x_feature_scalers[feat]

        df_tr = transform_one_feature(
            df_in=df_tr,
            feature=feat,
            mode=x_variant,
            global_scaler=feat_scaler if x_variant == "global" else None,
            local_scaler_dict=feat_scaler if x_variant == "local" else None,
            class_col=class_col,
        )

        df_va = transform_one_feature(
            df_in=df_va,
            feature=feat,
            mode=x_variant,
            global_scaler=feat_scaler if x_variant == "global" else None,
            local_scaler_dict=feat_scaler if x_variant == "local" else None,
            class_col=class_col,
        )

        df_te = transform_one_feature(
            df_in=df_te,
            feature=feat,
            mode=x_variant,
            global_scaler=feat_scaler if x_variant == "global" else None,
            local_scaler_dict=feat_scaler if x_variant == "local" else None,
            class_col=class_col,
        )

    df_tr = transform_y_only(df_tr, y_scaler, y_variant, y_cols, class_col)
    df_va = transform_y_only(df_va, y_scaler, y_variant, y_cols, class_col)
    df_te = transform_y_only(df_te, y_scaler, y_variant, y_cols, class_col)

    # --------------------------------------------------
    # Build arrays
    # --------------------------------------------------
    Xtr = df_tr[train_cols].to_numpy(np.float32)
    ytr = df_tr[y_cols].to_numpy(np.float32)

    Xva = df_va[train_cols].to_numpy(np.float32)
    yva = df_va[y_cols].to_numpy(np.float32)

    # --------------------------------------------------
    # Dataset / loaders
    # --------------------------------------------------
    ds_tr = model.TabularFMDataset(Xtr, ytr)
    ds_va = model.TabularFMDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    net = model.ConditionalVelocityField(
        x_dim=len(y_cols),
        context_dim=len(train_cols),
    ).to(device)

    if init_state_dict is not None:
        missing, unexpected = net.load_state_dict(init_state_dict, strict=strict_init)
        if (not strict_init) and (missing or unexpected):
            print(f"[init_state_dict] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    bad = 0

    tag = f"{len(selected_features)}feat_X{x_variant}_Y{y_variant}"

    for ep in range(1, epochs + 1):

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

        improved = (best_val - va_mean) > min_delta

        if improved:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose_every and (ep == 1 or ep % verbose_every == 0):
            print(
                f"[{tag} | ep {ep:4d}] "
                f"train={tr_mean:.6f}  val={va_mean:.6f}  "
                f"best={best_val:.6f}  bad={bad}/{patience}"
            )

        if bad >= patience:
            print(f"Early stopping at ep {ep} (best val={best_val:.6f})")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    # mode standard per permettere predict_tau_corr_auto
    if y_variant == "none":
        mode_for_predict = "X_NONE__Y_NONE"
    elif y_variant == "global":
        mode_for_predict = "X_NONE__Y_GLOBAL"
    else:
        mode_for_predict = "X_NONE__Y_LOCAL"

    res = {
        "model": net,
        "device": str(device),
        "train_cols": list(train_cols),
        "y_cols": list(y_cols),
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "x_variant": x_variant,
        "y_variant": y_variant,
        "x_feature_scalers": x_feature_scalers,
        "y_scaler": y_scaler,
        "loss_history": hist,
        "df_splits_scaled": (df_tr, df_va, df_te),
        "scalers": {
            "mode": mode_for_predict,
            "class_col": class_col,
            "x_scaler": None,       # X già preparata nel df_splits_scaled
            "y_scaler": y_scaler,   # serve per inverse-transform delle prediction
            "selected_features": selected_features,
            "x_variant": x_variant,
            "y_variant": y_variant,
        },
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


