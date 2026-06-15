# fm/features.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import (
    inv_mass_two_objects,
    delta_phi,
    delta_R,
    pt_two_objects,
)


def add_derived_bdt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived kinematic features used by the BDT classifier.

    Requires the following columns to already be present in df:
        - tau1_pt_reco_corrPNet, tau2_pt_reco_corrPNet
        - tau1_eta, tau1_phi, tau1_mass
        - tau2_eta, tau2_phi, tau2_mass
        - jet1_logpt, jet1_eta, jet1_phi, jet1_mass
        - jet2_logpt, jet2_eta, jet2_phi, jet2_mass
        - jet3_logpt, jet3_eta, jet3_phi, jet3_mass
        - MET_logpt, MET_phi

    These are added by preprocessing.add_pt_and_targets() beforehand.

    Returns a new DataFrame (copy) with the additional columns.
    """

    df = df.copy()

    # -----------------------
    # pT reconstruction
    # -----------------------
    df["jet1_pt"] = np.exp(df["jet1_logpt"])
    df["jet2_pt"] = np.exp(df["jet2_logpt"])
    df["jet3_pt"] = np.exp(df["jet3_logpt"])
    df["MET_pt"]  = np.exp(df["MET_logpt"])

    df["tau1_pt_for_features"] = df["tau1_pt_reco_corrPNet"]
    df["tau2_pt_for_features"] = df["tau2_pt_reco_corrPNet"]

    # -----------------------
    # visible ditau mass
    # -----------------------
    df["m_vis"] = inv_mass_two_objects(
        df["tau1_pt_for_features"],
        df["tau1_eta"],
        df["tau1_phi"],
        df["tau1_mass"],
        df["tau2_pt_for_features"],
        df["tau2_eta"],
        df["tau2_phi"],
        df["tau2_mass"],
    )

    # -----------------------
    # tau-tau variables
    # -----------------------
    df["deltaEta_tautau"] = df["tau1_eta"] - df["tau2_eta"]

    df["deltaPhi_tautau"] = delta_phi(
        df["tau1_phi"],
        df["tau2_phi"],
    )
    df["abs_deltaPhi_tautau"] = np.abs(df["deltaPhi_tautau"])

    df["dR_tautau"] = delta_R(
        df["tau1_eta"], df["tau1_phi"],
        df["tau2_eta"], df["tau2_phi"],
    )

    df["pT_tautau"] = pt_two_objects(
        df["tau1_pt_for_features"], df["tau1_eta"], df["tau1_phi"], df["tau1_mass"],
        df["tau2_pt_for_features"], df["tau2_eta"], df["tau2_phi"], df["tau2_mass"],
    )

    # -----------------------
    # tau-jet dR variables
    # -----------------------
    for tau in [1, 2]:
        for jet in [1, 2, 3]:
            df[f"dR_tau{tau}jet{jet}"] = delta_R(
                df[f"tau{tau}_eta"],
                df[f"tau{tau}_phi"],
                df[f"jet{jet}_eta"],
                df[f"jet{jet}_phi"],
            )

    # -----------------------
    # jet-jet variables: jet1-jet2
    # -----------------------
    df["m_jj"] = inv_mass_two_objects(
        df["jet1_pt"], df["jet1_eta"], df["jet1_phi"], df["jet1_mass"],
        df["jet2_pt"], df["jet2_eta"], df["jet2_phi"], df["jet2_mass"],
    )

    df["deltaEta_jj"] = df["jet1_eta"] - df["jet2_eta"]

    df["deltaPhi_jj"] = delta_phi(
        df["jet1_phi"],
        df["jet2_phi"],
    )
    df["abs_deltaPhi_jj"] = np.abs(df["deltaPhi_jj"])

    df["dR_jj"] = delta_R(
        df["jet1_eta"], df["jet1_phi"],
        df["jet2_eta"], df["jet2_phi"],
    )

    df["pT_jj"] = pt_two_objects(
        df["jet1_pt"], df["jet1_eta"], df["jet1_phi"], df["jet1_mass"],
        df["jet2_pt"], df["jet2_eta"], df["jet2_phi"], df["jet2_mass"],
    )

    # -----------------------
    # tau-MET variables
    # -----------------------
    df["deltaPhi_tau1MET"] = delta_phi(
        df["tau1_phi"],
        df["MET_phi"],
    )
    df["deltaPhi_tau2MET"] = delta_phi(
        df["tau2_phi"],
        df["MET_phi"],
    )
    df["abs_deltaPhi_tau1MET"] = np.abs(df["deltaPhi_tau1MET"])
    df["abs_deltaPhi_tau2MET"] = np.abs(df["deltaPhi_tau2MET"])

    df["METoverpTtautau"] = df["MET_pt"] / (df["pT_tautau"] + 1e-6)

    # -----------------------
    # log variables
    # -----------------------
    df["log_pT_tautau"]       = np.log1p(df["pT_tautau"])
    df["log_m_jj"]            = np.log1p(df["m_jj"])
    df["log_METoverpTtautau"] = np.log1p(df["METoverpTtautau"])

    return df