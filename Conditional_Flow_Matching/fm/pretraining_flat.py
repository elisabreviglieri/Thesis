# fm/pretraining_flat.py

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import torch

from . import data, training


DEFAULT_FLAT_MODES = ("X_GLOBAL__Y_GLOBAL", "X_NONE__Y_NONE")


def pretrain_flat(
    df_flat: pd.DataFrame,
    *,
    train_cols: list,
    y_cols: list,
    out_prepared: str,
    out_runs: str,
    modes: Iterable[str] = DEFAULT_FLAT_MODES,
    seed: int = 42,
    device: Optional[str] = None,
    overwrite: bool = True,
    **train_kwargs,
) -> Dict[str, Any]:
    """
    Pre-train on flat samples with the requested modes:
      - X_GLOBAL__Y_GLOBAL
      - X_NONE__Y_NONE

    Saves:
      out_prepared/<MODE>/{train,val,test}.pkl + scalers.joblib
      out_runs/<MODE>/{model_state.pt, meta.joblib, scalers.joblib}

    Notes:
      - No timestamps/seed in folder names (by design).
      - Training hyperparameters are NOT duplicated here:
        pass overrides via **train_kwargs, otherwise training.train_one_mode defaults apply.
      - If overwrite=False and a run folder exists, raises an error.
    """
    os.makedirs(out_prepared, exist_ok=True)
    os.makedirs(out_runs, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    modes = list(modes)

    # sanity: flat doesn't require class, but needs all columns
    missing = [c for c in (train_cols + y_cols) if c not in df_flat.columns]
    if missing:
        raise ValueError(f"Missing columns in df_flat: {missing}")

    # 1) prepared splits (split once, then per-mode scaling)
    mode_dirs = data.apply_and_save_selected_scalings(
        df=df_flat,
        out_root=out_prepared,
        train_cols=train_cols,
        y_cols=y_cols,
        modes=modes,
        seed=seed,
        split_kind="flat",
    )

    # 2) train + save per mode
    run_dirs: Dict[str, str] = {}

    for mode in modes:
        save_dir = os.path.join(out_runs, mode)

        if (not overwrite) and os.path.exists(save_dir):
            raise FileExistsError(
                f"Run folder already exists and overwrite=False:\n  {save_dir}"
            )

        print("\n" + "=" * 80)
        print(f"PRETRAIN FLAT | mode={mode}")
        print(f"prepared: {mode_dirs[mode]}")
        print(f"save_dir:  {save_dir}")
        print("=" * 80)

        res = training.train_one_mode(
            mode_dir=mode_dirs[mode],
            train_cols=train_cols,
            y_cols=y_cols,
            device=device,
            **train_kwargs,
        )
        training.save_run(res, save_dir)
        run_dirs[mode] = save_dir

    return {
        "mode_dirs": mode_dirs,
        "run_dirs": run_dirs,
        "modes": modes,
        "device": str(device),
        "train_kwargs": dict(train_kwargs),
        "out_prepared": out_prepared,
        "out_runs": out_runs,
    }