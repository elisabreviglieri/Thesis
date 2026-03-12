# fm/massInference.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import model, physics, training


def build_fm_masses_from_runs(
    run_root: str,
    prepared_root: str,
    modes: list,
    train_cols: list,
    y_cols: list,
    *,
    pt1_col: str = "tau1_pt_reco_corrPNet",
    pt2_col: str = "tau2_pt_reco_corrPNet",
    device: str = "cpu",
    n_steps: int = 60,
    batch_size: int = 8192,
):
    masses_by_mode = {}
    raw_tests_by_mode = {}

    for mode_name in modes:
        run_dir = os.path.join(run_root, mode_name)
        prep_dir = os.path.join(prepared_root, mode_name)

        df_test_raw = pd.read_pickle(os.path.join(prep_dir, "raw_test.pkl"))
        raw_tests_by_mode[mode_name] = df_test_raw

        # quick column checks (fail fast)
        needed = [
            pt1_col, "tau1_eta", "tau1_phi", "tau1_mass",
            pt2_col, "tau2_eta", "tau2_phi", "tau2_mass",
        ]
        missing = [c for c in needed if c not in df_test_raw.columns]
        if missing:
            raise ValueError(f"[{mode_name}] Missing columns in raw_test.pkl: {missing}")

        res = training.load_run(run_dir, device=device)
        res["device"] = device

        df_pred = model.predict_tau_corr_auto(
            df_test_raw,
            res,
            train_cols=train_cols,
            y_cols=y_cols,
            n_steps=n_steps,
            batch_size=batch_size,
        )

        m_fm = physics.inv_mass_two_taus_corrected_ratio(
            df_test_raw[pt1_col].to_numpy(),
            df_test_raw["tau1_eta"].to_numpy(),
            df_test_raw["tau1_phi"].to_numpy(),
            df_test_raw["tau1_mass"].to_numpy(),
            df_pred["tau1_corr_pred"].to_numpy(),
            df_test_raw[pt2_col].to_numpy(),
            df_test_raw["tau2_eta"].to_numpy(),
            df_test_raw["tau2_phi"].to_numpy(),
            df_test_raw["tau2_mass"].to_numpy(),
            df_pred["tau2_corr_pred"].to_numpy(),
        )

        masses_by_mode[mode_name] = m_fm
        print(f"[OK] {os.path.basename(run_root)} | {mode_name} | N={len(m_fm)}")

    return masses_by_mode, raw_tests_by_mode