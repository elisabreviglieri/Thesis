# fm.analysis

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_S_over_B(
    df,
    mass_col,
    window=(110, 130),
    class_col="class",
    dy_label=0,
    h_label=1,
    return_significance=True,
):
    """
    Compute S/M inside a mass window.

    S = number of signal events (default: class=1, Higgs)
    B= number of background events (default: class=0, DY)

    Returns:
        dict with counts and ratios
    """

    lo, hi = window

    m = df[mass_col].to_numpy()
    c = df[class_col].to_numpy()

    in_win = (m >= lo) & (m <= hi)

    N_S = np.sum(in_win & (c == h_label))
    N_B = np.sum(in_win & (c == dy_label))

    S_over_B = (N_S / N_B) if N_B > 0 else np.inf

    result = {
        "N_S": int(N_S),
        "N_B": int(N_B),
        "S_over_B": float(S_over_B),
        "window": window,
        "mass_col": mass_col,
    }

    if return_significance:
        S_over_sqrtB = (N_S / np.sqrt(N_B)) if N_B > 0 else np.inf
        result["S_over_sqrtB"] = float(S_over_sqrtB)

    return result



def compute_S_over_B_all_modes(
    masses_bal,
    masses_unbal,
    df_test_bal,
    df_test_unbal,
    scaling_modes,
    window=(110, 130),
    class_col="class",
    dy_label=0,
    h_label=1,
    save_csv=False,
    csv_path="S_over_B_results.csv",
):
    """
    Compute S/B for balanced and unbalanced datasets,
    for each scaling mode.

    Parameters
    ----------
    masses_bal : dict
        {mode: np.array} masses for balanced dataset
    masses_unbal : dict
        {mode: np.array} masses for unbalanced dataset
    df_test_bal : DataFrame
    df_test_unbal : DataFrame
    scaling_modes : list of modes
    window : tuple
        mass window (default: 110–130 GeV)
    save_csv : bool
        if True, saves results to CSV
    """

    rows = []

    for mode in scaling_modes:

        # -----------------------
        # Sanity checks
        # -----------------------
        if len(masses_bal[mode]) != len(df_test_bal):
            raise ValueError(f"Length mismatch BALANCED for mode {mode}")

        if len(masses_unbal[mode]) != len(df_test_unbal):
            raise ValueError(f"Length mismatch UNBALANCED for mode {mode}")

        # -----------------------
        # Balanced
        # -----------------------
        tmp_bal = df_test_bal[[class_col]].copy()
        tmp_bal["_mass_tmp"] = np.asarray(masses_bal[mode])

        res_bal = compute_S_over_B(
            tmp_bal,
            mass_col="_mass_tmp",
            window=window,
            class_col=class_col,
            dy_label=dy_label,
            h_label=h_label,
        )

        rows.append({
            "dataset": "balanced",
            "mode": mode,
            "window": f"{window[0]}-{window[1]}",
            "N_S": res_bal["N_S"],
            "N_B": res_bal["N_B"],
            "S_over_B": res_bal["S_over_B"],
            "S_over_sqrtB": res_bal.get("S_over_sqrtB", None),
        })

        # -----------------------
        # Unbalanced
        # -----------------------
        tmp_un = df_test_unbal[[class_col]].copy()
        tmp_un["_mass_tmp"] = np.asarray(masses_unbal[mode])

        res_un = compute_S_over_B(
            tmp_un,
            mass_col="_mass_tmp",
            window=window,
            class_col=class_col,
            dy_label=dy_label,
            h_label=h_label,
        )

        rows.append({
            "dataset": "unbalanced",
            "mode": mode,
            "window": f"{window[0]}-{window[1]}",
            "N_S": res_un["N_S"],
            "N_B": res_un["N_B"],
            "S_over_B": res_un["S_over_B"],
            "S_over_sqrtB": res_un.get("S_over_sqrtB", None),
        })

    df_results = pd.DataFrame(rows)

    # Pivot comodo per confronto
    df_pivot = df_results.pivot(
        index="mode",
        columns="dataset",
        values="S_over_B"
    ).reset_index()

    df_pivot.columns.name = None

    if save_csv:
        df_results.to_csv(csv_path, index=False)

    return df_results, df_pivot


# S/B when comparing non-flat & flat trainings with split masses:
def compute_SB_from_split_masses(masses_h, masses_dy, window=(110, 140)):
    df_tmp = pd.DataFrame({
        "mass": np.concatenate([np.asarray(masses_h), np.asarray(masses_dy)]),
        "class": np.concatenate([
            np.ones(len(masses_h), dtype=int),
            np.zeros(len(masses_dy), dtype=int),
        ])
    })

    return compute_S_over_B(
        df_tmp,
        mass_col="mass",
        window=window,
        class_col="class",
        dy_label=0,
        h_label=1,
        return_significance=True,
    )
    

# Per calcolare il picco di un istogramma:
def find_peak_position(masses, bins=120, range_=(0,300)):
    hist, edges = np.histogram(masses, bins=bins, range=range_)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_bin = np.argmax(hist)
    return centers[peak_bin]



def fit_signal_window(masses):

    mu, sigma = norm.fit(masses)

    low = mu - 2*sigma
    high = mu + 2*sigma

    return mu, sigma, low, high

def compute_SB(signal, background, window):

    low, high = window

    S = ((signal > low) & (signal < high)).sum()
    B = ((background > low) & (background < high)).sum()

    SB = S / B if B > 0 else np.nan
    SsqrtB = S / np.sqrt(B) if B > 0 else np.nan

    return S, B, SB, SsqrtB
