# fm/preprocessing.py
from __future__ import annotations # Mi permette di usare annotazioni tipo pd.DataFrame senza problemi di forward reference

from dataclasses import dataclass # Mi permette di creare oggetti "config" senza dover scrivere __init__
from typing import Optional, Tuple, Literal, Dict

import numpy as np
import pandas as pd


# ============================================================
# Config:
# - Come voglio fare il cut in pt:
#    * pt_min = soglia di 30 GeV
#    * space : mi dice se la variabile è in log o già pt
#    * tau*_pt_col: nome delle colonne pt da usare se space="pt"
#    * tau*_logpt_col: nome delle colonne logpt da usare se space="logpt"
#    -> No hardcoding del nome delle variabili 
# ============================================================
@dataclass
class PtCutConfig:
    pt_min: float = 30.0
    space: Literal["pt", "logpt"] = "pt"  
    tau1_pt_col: str = "tau1_pt_reco_corrPNet"  
    tau2_pt_col: str = "tau2_pt_reco_corrPNet"
    tau1_logpt_col: str = "tau1_logpt"
    tau2_logpt_col: str = "tau2_logpt"


@dataclass
class TauIDConfig:
    tau1_id_col: str = "tau1_rawPNetVSjet"
    tau2_id_col: str = "tau2_rawPNetVSjet"
    invalid_value: float = -1.0

    # working point
    wp_mode: Literal["fixed", "target_eff"] = "target_eff"
    thr_tau1: Optional[float] = None
    thr_tau2: Optional[float] = None
    target_eff: float = 0.90

    # event-level requirement
    require_both: bool = True

    # reference sample used to compute thresholds if wp_mode="target_eff"
    # typical choice: "higgs"
    reference: Literal["higgs", "dy"] = "higgs"


# ============================================================
# pT cut
# ============================================================
def apply_pt_cut(df: pd.DataFrame, cfg: PtCutConfig) -> pd.DataFrame:
    """
    Keep events with both taus passing pT cut.
    Works either in 'pt' or 'logpt' space.
    """
    out = df.copy()

    if cfg.space == "logpt":
        log_pt_min = np.log(cfg.pt_min)
        m = (out[cfg.tau1_logpt_col] > log_pt_min) & (out[cfg.tau2_logpt_col] > log_pt_min)
        return out.loc[m].copy()

    # pt space
    m = (out[cfg.tau1_pt_col] > cfg.pt_min) & (out[cfg.tau2_pt_col] > cfg.pt_min)
    return out.loc[m].copy()


# ============================================================
# TauID cleaning + WP
# ============================================================
def filter_valid_tauid(df: pd.DataFrame, cfg: TauIDConfig) -> pd.DataFrame:
    """
    Drop events where tauID score is invalid (e.g. -1 for either tau).
    """
    out = df.copy()
    m = (out[cfg.tau1_id_col] != cfg.invalid_value) & (out[cfg.tau2_id_col] != cfg.invalid_value)
    return out.loc[m].copy()


def threshold_for_target_eff(scores: np.ndarray, target_eff: float) -> float:
    """
    Choose threshold t such that fraction passing is ~ target_eff.
    Implemented via quantile at (1 - target_eff).
    """
    s = np.asarray(scores)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("Empty/invalid scores array.")
    return float(np.quantile(s, 1.0 - float(target_eff)))


def compute_tauid_thresholds(
    df_dy: pd.DataFrame,
    df_h: pd.DataFrame,
    cfg: TauIDConfig,
) -> Tuple[float, float]:
    """
    Compute (thr_tau1, thr_tau2) depending on cfg.wp_mode.

    - fixed: uses cfg.thr_tau1/thr_tau2
    - target_eff: computes thresholds on reference sample
    """
    if cfg.wp_mode == "fixed":
        if cfg.thr_tau1 is None or cfg.thr_tau2 is None:
            raise ValueError("wp_mode='fixed' requires thr_tau1 and thr_tau2.")
        return float(cfg.thr_tau1), float(cfg.thr_tau2)

    if cfg.wp_mode == "target_eff":
        ref = df_h if cfg.reference == "higgs" else df_dy
        thr1 = threshold_for_target_eff(ref[cfg.tau1_id_col].to_numpy(), cfg.target_eff)
        thr2 = threshold_for_target_eff(ref[cfg.tau2_id_col].to_numpy(), cfg.target_eff)
        return thr1, thr2

    raise ValueError("wp_mode must be 'fixed' or 'target_eff'.")


def apply_tauid_wp(df: pd.DataFrame, cfg: TauIDConfig, thr_tau1: float, thr_tau2: float) -> pd.DataFrame:
    """
    Apply tauID working point at event level.
    """
    out = df.copy()
    m1 = out[cfg.tau1_id_col] > thr_tau1
    m2 = out[cfg.tau2_id_col] > thr_tau2

    m = (m1 & m2) if cfg.require_both else (m1 | m2)
    return out.loc[m].copy()


# ============================================================
# Volendo metto tutto in una funzione sola, ma nel notebook
# vorrei controllare step by step che sia tutto ok 
# ============================================================
def preprocess_dy_higgs(
    df_dy: pd.DataFrame,
    df_h: pd.DataFrame,
    pt_cfg: Optional[PtCutConfig] = None,
    tauid_cfg: Optional[TauIDConfig] = None,
) -> Dict[str, object]:
    """
    Run preprocessing on DY and Higgs consistently.

    Returns dict:
      - df_dy, df_h
      - thresholds (if tauid_cfg provided)
      - counts (before/after)
    """
    if pt_cfg is None:
        pt_cfg = PtCutConfig()
    if tauid_cfg is None:
        tauid_cfg = TauIDConfig()

    # --- pT cut
    df_dy1 = apply_pt_cut(df_dy, pt_cfg)
    df_h1  = apply_pt_cut(df_h,  pt_cfg)

    # --- tauID valid
    df_dy2 = filter_valid_tauid(df_dy1, tauid_cfg)
    df_h2  = filter_valid_tauid(df_h1,  tauid_cfg)

    # --- thresholds
    thr1, thr2 = compute_tauid_thresholds(df_dy2, df_h2, tauid_cfg)

    # --- apply WP
    df_dy3 = apply_tauid_wp(df_dy2, tauid_cfg, thr1, thr2)
    df_h3  = apply_tauid_wp(df_h2,  tauid_cfg, thr1, thr2)

    return {
        "df_dy": df_dy3,
        "df_h": df_h3,
        "thr_tau1": thr1,
        "thr_tau2": thr2,
        "counts": {
            "dy_before": len(df_dy),
            "dy_after": len(df_dy3),
            "h_before": len(df_h),
            "h_after": len(df_h3),
        },
    }
    
    
# =================================
# PER I SAMPLE FLAT
# =================================

# Definisco le features 

tau_features = [
    "logpt","eta","phi","mass","dxy","dz",
    "ptCorrPNet","rawPNetVSjet","rawDeepTau2018v2p5VSjet",
    "charge",
    "dM_0","dM_1","dM_2","dM_10","dM_11",
    "leadTkDeltaEta","leadTkDeltaPhi","leadTkPtOverTauPt"
]

gen_tau_features = ["pt","eta","phi","mass"]

met_features = [
    "MET_logpt","MET_phi","MET_covXX","MET_covXY",
    "MET_covYY","MET_significance","MET_sumEt","MET_sumPtUnclustered"
]

jet_features = ["logpt","eta","phi","mass"]


# Estraggo i blocchi principali dal .npz :
def extract_blocks(npz):
    """
    Estrae i blocchi principali in modo uniforme da un file .npz.
    Ritorna un dict con array e masse scalari.
    """
    out = {}

    # --- features blocks
    out["x_taus"] = npz["x_taus"]        # (N, 2, 18)
    out["x_gen"]  = npz["x_gen"]         # (N, 2, 4)
    out["x_met"]  = npz["x_met"]         # (N, 1, 8)
    out["x_jets"] = npz["x_jets"]        # (N, 3, 4)

    # opzionale: non so ancora se devo usarlo 
    if "x_tauprod" in npz.files:
        out["x_tauprod"] = npz["x_tauprod"]  # (N, 10, 10)

    # --- masses / targets
    out["m_reco"]    = npz["m_vis_ptcorr"]   # m_reco
    out["m_target"]  = npz["m_gen"]          # target
    out["m_vis"]     = npz["m_vis"]
    out["m_fastmtt"] = npz["m_fastmtt"]

    # --- N + sanity
    N = out["x_taus"].shape[0]
    for k in ["x_gen", "x_met", "x_jets", "m_reco", "m_target", "m_vis", "m_fastmtt"]:
        if out[k].shape[0] != N:
            raise ValueError(f"Length mismatch: {k} has {out[k].shape[0]} but x_taus has {N}")
    out["N"] = N

    return out


# Funzione per costruire i DataFrames uguali:

def build_df_from_npz(npz, sample: str):
    taus     = npz["x_taus"]         # (N,2,18)
    tau_gen  = npz["x_gen"]          # (N,2,4)
    met      = npz["x_met"]          # (N,1,8)
    jets     = npz["x_jets"]         # (N,3,4)

    m_vis        = npz["m_vis"]
    m_vis_ptcorr = npz["m_vis_ptcorr"]
    m_fastmtt    = npz["m_fastmtt"]
    m_target     = npz["m_gen"]

    N = taus.shape[0]

    # sanity
    assert tau_gen.shape[0] == N
    assert met.shape[0] == N
    assert jets.shape[0] == N
    assert m_vis.shape[0] == N
    assert m_vis_ptcorr.shape[0] == N
    assert m_fastmtt.shape[0] == N
    assert m_target.shape[0] == N

    df = {}

    # ---------- TAU 1 & TAU 2 ----------
    for itau in [0, 1]:
        prefix = f"tau{itau+1}_"
        for i, feat in enumerate(tau_features):
            df[prefix + feat] = taus[:, itau, i]

    # ---------- GEN TAU 1 & 2 ----------
    for itau in [0, 1]:
        prefix = f"gen_tau{itau+1}_"
        for i, feat in enumerate(gen_tau_features):
            df[prefix + feat] = tau_gen[:, itau, i]

    # ---------- MET ----------
    for i, feat in enumerate(met_features):
        df[feat] = met[:, 0, i]

    # ---------- JETS (1..3) ----------
    for j in range(3):
        for i, feat in enumerate(jet_features):
            df[f"jet{j+1}_{feat}"] = jets[:, j, i]

    # ---------- MASSE ----------
    df["m_vis"]        = m_vis
    df["m_vis_ptcorr"] = m_vis_ptcorr
    df["m_fastmtt"]    = m_fastmtt
    df["m_gen"]        = m_target

    # ---------- TAG SAMPLE ----------
    df["sample"] = sample

    df_flat = pd.DataFrame(df)
    return df_flat


# Processing di pT e targets :
def add_pt_and_targets(df: pd.DataFrame, use_clip: bool = True) -> pd.DataFrame:
    """
    Adds:
      - tau*_pt_reco
      - tau*_pt_reco_corrPNet
      - tau*_corr (target)

    Returns a new dataframe (copy).
    """
    out = df.copy()

    # --- pt reco
    out["tau1_pt_reco"] = np.exp(out["tau1_logpt"])
    out["tau2_pt_reco"] = np.exp(out["tau2_logpt"])

    # --- pt reco corrected (PNet factor)
    out["tau1_pt_reco_corrPNet"] = out["tau1_pt_reco"] * out["tau1_ptCorrPNet"]
    out["tau2_pt_reco_corrPNet"] = out["tau2_pt_reco"] * out["tau2_ptCorrPNet"]

    # --- remove unphysical zeros
    mask = (
        (out["tau1_pt_reco_corrPNet"] > 0) &
        (out["tau2_pt_reco_corrPNet"] > 0)
    )
    out = out.loc[mask].copy()

    # --- targets
    eps = 1e-8
    if use_clip:
        out["tau1_corr"] = out["gen_tau1_pt"] / np.clip(out["tau1_pt_reco_corrPNet"], eps, None)
        out["tau2_corr"] = out["gen_tau2_pt"] / np.clip(out["tau2_pt_reco_corrPNet"], eps, None)
    else:
        out["tau1_corr"] = out["gen_tau1_pt"] / out["tau1_pt_reco_corrPNet"]
        out["tau2_corr"] = out["gen_tau2_pt"] / out["tau2_pt_reco_corrPNet"]

    return out