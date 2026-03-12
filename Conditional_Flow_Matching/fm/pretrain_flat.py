# fm/pretrain_flat.py

from __future__ import annotations

import os
import time
import joblib
import pandas as pd
import torch

from . import data, training


# ============================================================
# CONFIG
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODES = ["X_GLOBAL__Y_GLOBAL", "X_NONE__Y_NONE"]

HP = dict(
    epochs=300,
    batch_size=4096,
    lr=2e-4,
    weight_decay=0.0,
    patience=30,
    min_delta=1e-4,
    verbose_every=10,
    loss_type="mse",       # or "huber"
    huber_delta=1.0,
)

# paths relative to project root
FLAT_PKL_REL = os.path.join("data", "flat", "df_flat.pkl")
TRAIN_COLS_REL = os.path.join("config", "train_cols.joblib")
Y_COLS_REL     = os.path.join("config", "y_cols.joblib")


def _project_root() -> str:
    # this file is in <root>/fm/pretrain_flat.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    root = _project_root()

    flat_pkl = os.path.join(root, FLAT_PKL_REL)
    train_cols = joblib.load(os.path.join(root, TRAIN_COLS_REL))
    y_cols     = joblib.load(os.path.join(root, Y_COLS_REL))

    out_prepared = os.path.join(root, "results", "prepared_splits", "flat")
    out_runs     = os.path.join(root, "results", "runs_prepared", "flat")
    os.makedirs(out_prepared, exist_ok=True)
    os.makedirs(out_runs, exist_ok=True)

    df = pd.read_pickle(flat_pkl)

    # quick sanity: flat -> no 'class' assumed
    missing = [c for c in (train_cols + y_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in flat df: {missing}")

    # 1) prepare splits once, save in results/prepared_splits/flat/<MODE>/
    mode_dirs = data.apply_and_save_selected_scalings(
        df=df,
        out_root=out_prepared,
        train_cols=train_cols,
        y_cols=y_cols,
        modes=MODES,
        seed=SEED,
        split_kind="flat",
        class_col="class",  # irrelevant for these modes
    )

    # 2) train + save weights in results/runs_prepared/flat/<TAG>/
    run_id = time.strftime("%Y%m%d_%H%M%S")

    for mode in MODES:
        tag = f"flat_pretrain__{mode}__{HP['loss_type']}__seed{SEED}__{run_id}"
        save_dir = os.path.join(out_runs, tag)

        print("\n" + "=" * 80)
        print(f"PRETRAIN FLAT | mode={mode}")
        print(f"prepared: {mode_dirs[mode]}")
        print(f"save_dir:  {save_dir}")
        print("=" * 80)

        res = training.train_one_mode(
            mode_dir=mode_dirs[mode],
            train_cols=train_cols,
            y_cols=y_cols,
            device=DEVICE,
            **HP,
        )
        training.save_run(res, save_dir)

    print("\nDone.")
    print("Prepared splits:", out_prepared)
    print("Runs:", out_runs)


if __name__ == "__main__":
    main()