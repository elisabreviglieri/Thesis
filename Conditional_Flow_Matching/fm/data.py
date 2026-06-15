# fm/data.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Physics utility: balance DY/H using class label
# ============================================================
def make_balanced_df(
    df: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create a balanced dataset knowing that:
        class = 0 -> DY
        class = 1 -> Higgs

    Downsamples the majority class to match the minority.
    Returns a shuffled balanced DataFrame.
    """
    if "class" not in df.columns:
        raise ValueError("Column 'class' not found in DataFrame.")

    df_dy = df[df["class"] == 0]
    df_h  = df[df["class"] == 1]

    if len(df_dy) == 0 or len(df_h) == 0:
        raise ValueError("One of the classes is empty. Cannot balance.")

    n = min(len(df_dy), len(df_h))

    df_dy_sample = df_dy.sample(n=n, random_state=seed, replace=False)
    df_h_sample  = df_h.sample(n=n, random_state=seed, replace=False)

    df_balanced = pd.concat([df_dy_sample, df_h_sample])
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_balanced


# ============================================================
# Split utilities
# ============================================================
def split_df_once(
    df: pd.DataFrame,
    seed: int = 0,
):
    """
    Perform a single 80/10/10 split using sklearn train_test_split (no stratification).
    Suitable when you either have no labels or you don't need stratification.
    """
    df_train, df_temp = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def split_df_flat(
    df: pd.DataFrame,
    seed: int = 0,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """
    80/10/10 random split (no stratification).
    Works for flat samples without class labels.
    """
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("Fractions must sum to 1.")

    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_frac),
        random_state=seed,
        shuffle=True,
    )

    relative_test_frac = test_frac / (val_frac + test_frac)

    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_frac,
        random_state=seed,
        shuffle=True,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ------------------------------------------------------------
# Scaling utilities (modes)
# ------------------------------------------------------------
VALID_SCALING_MODES = {
    "X_NONE__Y_NONE",       # universal: works with or without class
    "X_GLOBAL__Y_GLOBAL",   # universal: works with or without class
    "X_GLOBAL__Y_LOCAL",    # requires class
    "X_LOCAL__Y_LOCAL",     # requires class
    "X_NONE__Y_LOCAL",      # requires class
    "X_LOCAL__Y_GLOBAL"     # requires class    
}


def fit_scalers(
    df_train: pd.DataFrame,
    train_cols: list,
    y_cols: list,
    mode: str = "X_NONE__Y_NONE",
    class_col: str = "class",
    x_scale_cols: list | None = None,
) -> dict:

    if mode not in VALID_SCALING_MODES:
        raise ValueError(f"Unknown scaling mode '{mode}'. Valid: {sorted(VALID_SCALING_MODES)}")

    for c in train_cols:
        if c not in df_train.columns:
            raise ValueError(f"Missing train column '{c}' in df_train.")
    for c in y_cols:
        if c not in df_train.columns:
            raise ValueError(f"Missing target column '{c}' in df_train.")

    if x_scale_cols is None:
        x_scale_cols = list(train_cols)

    for c in x_scale_cols:
        if c not in train_cols:
            raise ValueError(f"x_scale_cols contains '{c}', which is not in train_cols.")
        if c not in df_train.columns:
            raise ValueError(f"Missing x-scale column '{c}' in df_train.")

    scalers = {
        "mode": mode,
        "class_col": class_col,
        "x_scale_cols": list(x_scale_cols),
    }

    if mode == "X_NONE__Y_NONE":
        scalers["x_scaler"] = None
        scalers["y_scaler"] = None
        return scalers

    if mode == "X_GLOBAL__Y_GLOBAL":
        sx = StandardScaler().fit(df_train[x_scale_cols].to_numpy())
        sy = StandardScaler().fit(df_train[y_cols].to_numpy())
        scalers["x_scaler"] = sx
        scalers["y_scaler"] = sy
        return scalers

    if class_col not in df_train.columns:
        raise ValueError(f"Missing class column '{class_col}' in df_train (required for mode {mode}).")

    df_dy = df_train[df_train[class_col] == 0]
    df_h  = df_train[df_train[class_col] == 1]
    if len(df_dy) == 0 or len(df_h) == 0:
        raise ValueError("One of the classes is empty in df_train; cannot fit class-conditional scalers.")

    def fit_y_local():
        s0 = StandardScaler().fit(df_dy[y_cols].to_numpy())
        s1 = StandardScaler().fit(df_h[y_cols].to_numpy())
        return {0: s0, 1: s1}
    
    def fit_y_global():
        return StandardScaler().fit(df_train[y_cols].to_numpy())

    def fit_x_global():
        return StandardScaler().fit(df_train[x_scale_cols].to_numpy())

    def fit_x_local():
        s0 = StandardScaler().fit(df_dy[x_scale_cols].to_numpy())
        s1 = StandardScaler().fit(df_h[x_scale_cols].to_numpy())
        return {0: s0, 1: s1}

    if mode == "X_GLOBAL__Y_LOCAL":
        scalers["x_scaler"] = fit_x_global()
        scalers["y_scaler"] = fit_y_local()
        return scalers

    if mode == "X_LOCAL__Y_LOCAL":
        scalers["x_scaler"] = fit_x_local()
        scalers["y_scaler"] = fit_y_local()
        return scalers

    if mode == "X_NONE__Y_LOCAL":
        scalers["x_scaler"] = None
        scalers["y_scaler"] = fit_y_local()
        return scalers
    
    if mode == "X_LOCAL__Y_GLOBAL":
        scalers["x_scaler"] = fit_x_local()
        scalers["y_scaler"] = fit_y_global()
        return scalers

    raise RuntimeError("Unhandled scaling mode.")


def transform_df(
    df: pd.DataFrame,
    train_cols: list,
    y_cols: list,
    scalers: dict,
) -> pd.DataFrame:

    mode = scalers.get("mode", None)
    class_col = scalers.get("class_col", "class")
    x_scale_cols = scalers.get("x_scale_cols", list(train_cols))

    if mode not in VALID_SCALING_MODES:
        raise ValueError(f"Invalid scalers['mode'] = {mode}")

    out = df.copy()

    if mode == "X_NONE__Y_NONE":
        return out

    x_scaler = scalers.get("x_scaler", None)
    y_scaler = scalers.get("y_scaler", None)

    def assign_cols(mask, cols, arr):
        arr = arr.astype(np.float32)

        for j, col in enumerate(cols):
            out.loc[mask, col] = pd.Series(
                arr[:, j],
                index=out.loc[mask].index,
                dtype=np.float32,
        )

    def _get_class_masks():
        if class_col not in out.columns:
            raise ValueError(f"Missing class column '{class_col}' in df (required for mode {mode}).")
        c = out[class_col].to_numpy()
        return (c == 0), (c == 1)

    if x_scaler is None:
        pass
    elif isinstance(x_scaler, StandardScaler):
        X = x_scaler.transform(out[x_scale_cols].to_numpy())
        _assign_cols(slice(None), x_scale_cols, X)
    elif isinstance(x_scaler, dict):
        m0, m1 = _get_class_masks()
        if m0.any():
            X0 = x_scaler[0].transform(out.loc[m0, x_scale_cols].to_numpy())
            _assign_cols(m0, x_scale_cols, X0)
        if m1.any():
            X1 = x_scaler[1].transform(out.loc[m1, x_scale_cols].to_numpy())
            _assign_cols(m1, x_scale_cols, X1)
    else:
        raise ValueError("x_scaler must be None, a StandardScaler, or a dict {0,1}.")

    if y_scaler is None:
        pass
    elif isinstance(y_scaler, StandardScaler):
        Y = y_scaler.transform(out[y_cols].to_numpy())
        _assign_cols(slice(None), y_cols, Y)
    elif isinstance(y_scaler, dict):
        m0, m1 = _get_class_masks()
        if m0.any():
            Y0 = y_scaler[0].transform(out.loc[m0, y_cols].to_numpy())
            _assign_cols(m0, y_cols, Y0)
        if m1.any():
            Y1 = y_scaler[1].transform(out.loc[m1, y_cols].to_numpy())
            _assign_cols(m1, y_cols, Y1)
    else:
        raise ValueError("y_scaler must be None, a StandardScaler, or a dict {0,1}.")

    return out

def prepare_splits_with_scaling(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    train_cols: list,
    y_cols: list,
    mode: str = "X_NONE__Y_NONE",
    class_col: str = "class",
    x_scale_cols: list | None = None,
):
    """
    Convenience helper:
      - fit scalers on df_train only
      - transform train/val/test
      - return (df_train_s, df_val_s, df_test_s, scalers)
    """
    scalers = fit_scalers(
        df_train=df_train,
        train_cols=train_cols,
        y_cols=y_cols,
        mode=mode,
        class_col=class_col,
        x_scale_cols=x_scale_cols,
    )

    df_train_s = transform_df(df_train, train_cols, y_cols, scalers)
    df_val_s   = transform_df(df_val,   train_cols, y_cols, scalers)
    df_test_s  = transform_df(df_test,  train_cols, y_cols, scalers)

    return (
        df_train_s.reset_index(drop=True),
        df_val_s.reset_index(drop=True),
        df_test_s.reset_index(drop=True),
        scalers,
    )

# ============================================================
# Prepared splits saver (used by training.load_prepared)
# ============================================================
SCALING_MODES = [
    "X_NONE__Y_NONE",
    "X_GLOBAL__Y_LOCAL",
    "X_GLOBAL__Y_GLOBAL",
    "X_LOCAL__Y_LOCAL",
    "X_NONE__Y_LOCAL",
]

def apply_and_save_all_scalings(
    df_train, df_val, df_test,
    dataset_tag: str,             # "unbalanced" o "balanced"
    train_cols, y_cols,
    outdir: str,
):
    base = os.path.join(outdir, dataset_tag)
    os.makedirs(base, exist_ok=True)

    saved_paths = {}

    for mode in SCALING_MODES:
        df_tr_s, df_va_s, df_te_s, scalers = prepare_splits_with_scaling(
            df_train, df_val, df_test,
            train_cols=train_cols,
            y_cols=y_cols,
            mode=mode,
            class_col="class",
        )

        mode_dir = os.path.join(base, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # DataFrames scalati
        df_tr_s.to_pickle(os.path.join(mode_dir, "train.pkl"))
        df_va_s.to_pickle(os.path.join(mode_dir, "val.pkl"))
        df_te_s.to_pickle(os.path.join(mode_dir, "test.pkl"))

        # Scalers
        joblib.dump(scalers, os.path.join(mode_dir, "scalers.joblib"))

        saved_paths[mode] = mode_dir
        print(f"[{dataset_tag}] saved {mode} -> {mode_dir}")

    return saved_paths

def apply_and_save_selected_scalings(
    df: pd.DataFrame,
    out_root: str,
    train_cols: list,
    y_cols: list,
    modes: list,
    seed: int = 42,
    *,
    split_kind: str = "auto",
    class_col: str = "class",
) -> dict:
    """
    Split ONCE, then for each mode:
      - fit scalers on train only
      - transform train/val/test
      - save train.pkl, val.pkl, test.pkl (scaled)
      - ALSO save raw_train.pkl, raw_val.pkl, raw_test.pkl
      - save scalers.joblib

    split_kind:
      - "flat":   use split_df_flat()
      - "physics":use split_df_once()
      - "auto":   if class_col in df -> physics else flat
    """

    os.makedirs(out_root, exist_ok=True)

    if split_kind not in {"auto", "flat", "physics"}:
        raise ValueError("split_kind must be 'auto', 'flat', or 'physics'.")

    # ---- split ONCE
    if split_kind == "flat":
        df_tr, df_va, df_te = split_df_flat(df, seed=seed)
    elif split_kind == "physics":
        df_tr, df_va, df_te = split_df_once(df, seed=seed)
    else:
        if class_col in df.columns:
            df_tr, df_va, df_te = split_df_once(df, seed=seed)
        else:
            df_tr, df_va, df_te = split_df_flat(df, seed=seed)

    mode_dirs = {}

    for mode in modes:

        if mode not in VALID_SCALING_MODES:
            raise ValueError(f"Unknown scaling mode '{mode}'")

        mode_dir = os.path.join(out_root, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # ---- SAVE RAW (unscaled) SPLITS
        df_tr.to_pickle(os.path.join(mode_dir, "raw_train.pkl"))
        df_va.to_pickle(os.path.join(mode_dir, "raw_val.pkl"))
        df_te.to_pickle(os.path.join(mode_dir, "raw_test.pkl"))

        # ---- prepare scaled versions
        df_tr_s, df_va_s, df_te_s, scalers = prepare_splits_with_scaling(
            df_train=df_tr,
            df_val=df_va,
            df_test=df_te,
            train_cols=train_cols,
            y_cols=y_cols,
            mode=mode,
            class_col=class_col,
        )

        # ---- SAVE SCALED
        df_tr_s.to_pickle(os.path.join(mode_dir, "train.pkl"))
        df_va_s.to_pickle(os.path.join(mode_dir, "val.pkl"))
        df_te_s.to_pickle(os.path.join(mode_dir, "test.pkl"))
        joblib.dump(scalers, os.path.join(mode_dir, "scalers.joblib"))

        mode_dirs[mode] = mode_dir

    return mode_dirs


# ===================================
# MULTICLASS FUNCTIONS (TTbar added)
# ===================================

def fit_scalers_multiclass(
    df_train: pd.DataFrame,
    train_cols: list,
    y_cols: list,
    mode: str = "X_NONE__Y_NONE",
    class_col: str = "class",
    x_scale_cols: list | None = None,
) -> dict:

    if mode not in VALID_SCALING_MODES:
        raise ValueError(f"Unknown scaling mode '{mode}'.")

    if x_scale_cols is None:
        x_scale_cols = list(train_cols)

    classes = sorted(df_train[class_col].dropna().unique())

    scalers = {
        "mode": mode,
        "class_col": class_col,
        "x_scale_cols": list(x_scale_cols),
        "classes": classes,
    }

    if mode == "X_NONE__Y_NONE":
        scalers["x_scaler"] = None
        scalers["y_scaler"] = None
        return scalers

    if mode == "X_GLOBAL__Y_GLOBAL":
        scalers["x_scaler"] = StandardScaler().fit(df_train[x_scale_cols].to_numpy())
        scalers["y_scaler"] = StandardScaler().fit(df_train[y_cols].to_numpy())
        return scalers

    def fit_local(cols):
        out = {}
        for cls in classes:
            df_cls = df_train[df_train[class_col] == cls]
            if len(df_cls) == 0:
                raise ValueError(f"Class {cls} is empty.")
            out[cls] = StandardScaler().fit(df_cls[cols].to_numpy())
        return out

    def fit_global(cols):
        return StandardScaler().fit(df_train[cols].to_numpy())

    if mode == "X_GLOBAL__Y_LOCAL":
        scalers["x_scaler"] = fit_global(x_scale_cols)
        scalers["y_scaler"] = fit_local(y_cols)
        return scalers

    if mode == "X_LOCAL__Y_LOCAL":
        scalers["x_scaler"] = fit_local(x_scale_cols)
        scalers["y_scaler"] = fit_local(y_cols)
        return scalers

    if mode == "X_NONE__Y_LOCAL":
        scalers["x_scaler"] = None
        scalers["y_scaler"] = fit_local(y_cols)
        return scalers

    if mode == "X_LOCAL__Y_GLOBAL":
        scalers["x_scaler"] = fit_local(x_scale_cols)
        scalers["y_scaler"] = fit_global(y_cols)
        return scalers

    raise RuntimeError("Unhandled scaling mode.")

def transform_df_multiclass(
    df: pd.DataFrame,
    train_cols: list,
    y_cols: list,
    scalers: dict,
) -> pd.DataFrame:

    mode = scalers["mode"]
    class_col = scalers.get("class_col", "class")
    x_scale_cols = scalers.get("x_scale_cols", list(train_cols))

    out = df.copy()

    if mode == "X_NONE__Y_NONE":
        return out

    def assign_cols(mask, cols, arr):
        for j, col in enumerate(cols):
            out.loc[mask, col] = arr[:, j]

    def apply_scaler_to_classes(scaler_dict, cols):
        for cls, scaler in scaler_dict.items():
            mask = out[class_col] == cls
            if mask.any():
                arr = scaler.transform(out.loc[mask, cols].to_numpy())
                assign_cols(mask, cols, arr)

    x_scaler = scalers.get("x_scaler")
    y_scaler = scalers.get("y_scaler")

    if isinstance(x_scaler, StandardScaler):
        X = x_scaler.transform(out[x_scale_cols].to_numpy())
        assign_cols(slice(None), x_scale_cols, X)
    elif isinstance(x_scaler, dict):
        apply_scaler_to_classes(x_scaler, x_scale_cols)

    if isinstance(y_scaler, StandardScaler):
        Y = y_scaler.transform(out[y_cols].to_numpy())
        assign_cols(slice(None), y_cols, Y)
    elif isinstance(y_scaler, dict):
        apply_scaler_to_classes(y_scaler, y_cols)

    return out

def prepare_splits_with_scaling_multiclass(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    train_cols: list,
    y_cols: list,
    mode: str = "X_NONE__Y_NONE",
    class_col: str = "class",
    x_scale_cols: list | None = None,
):

    scalers = fit_scalers_multiclass(
        df_train=df_train,
        train_cols=train_cols,
        y_cols=y_cols,
        mode=mode,
        class_col=class_col,
        x_scale_cols=x_scale_cols,
    )

    df_train_s = transform_df_multiclass(df_train, train_cols, y_cols, scalers)
    df_val_s   = transform_df_multiclass(df_val,   train_cols, y_cols, scalers)
    df_test_s  = transform_df_multiclass(df_test,  train_cols, y_cols, scalers)

    return (
        df_train_s.reset_index(drop=True),
        df_val_s.reset_index(drop=True),
        df_test_s.reset_index(drop=True),
        scalers,
    )
    

def apply_and_save_all_scalings_multiclass(
    df_train, df_val, df_test,
    dataset_tag: str,
    train_cols, y_cols,
    outdir: str,
):
    base = os.path.join(outdir, dataset_tag)
    os.makedirs(base, exist_ok=True)

    saved_paths = {}

    for mode in SCALING_MODES:
        df_tr_s, df_va_s, df_te_s, scalers = prepare_splits_with_scaling_multiclass(
            df_train, df_val, df_test,
            train_cols=train_cols,
            y_cols=y_cols,
            mode=mode,
            class_col="class",
        )

        mode_dir = os.path.join(base, mode)
        os.makedirs(mode_dir, exist_ok=True)

        df_tr_s.to_pickle(os.path.join(mode_dir, "train.pkl"))
        df_va_s.to_pickle(os.path.join(mode_dir, "val.pkl"))
        df_te_s.to_pickle(os.path.join(mode_dir, "test.pkl"))

        joblib.dump(scalers, os.path.join(mode_dir, "scalers.joblib"))

        saved_paths[mode] = mode_dir
        print(f"[{dataset_tag}] saved {mode} -> {mode_dir}")

    return saved_paths

