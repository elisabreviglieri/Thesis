# BDT_FeaturesEngineering.py

import numpy as np


tau1_cols = [
    "tau1_logpt", "tau1_eta", "tau1_phi", "tau1_mass",
    "tau1_dxy", "tau1_dz",
    "tau1_ptCorrPNet", "tau1_rawPNetVSjet", "tau1_rawDeepTau2018v2p5VSjet",
    "tau1_charge",
    "tau1_dM_0", "tau1_dM_1", "tau1_dM_2", "tau1_dM_10", "tau1_dM_11",
    "tau1_leadTkDeltaEta", "tau1_leadTkDeltaPhi", "tau1_leadTkPtOverTauPt",
]

tau2_cols = [
    "tau2_logpt", "tau2_eta", "tau2_phi", "tau2_mass",
    "tau2_dxy", "tau2_dz",
    "tau2_ptCorrPNet", "tau2_rawPNetVSjet", "tau2_rawDeepTau2018v2p5VSjet",
    "tau2_charge",
    "tau2_dM_0", "tau2_dM_1", "tau2_dM_2", "tau2_dM_10", "tau2_dM_11",
    "tau2_leadTkDeltaEta", "tau2_leadTkDeltaPhi", "tau2_leadTkPtOverTauPt",
]

jet_cols = [
    "jet1_logpt", "jet1_eta", "jet1_phi", "jet1_mass",
    "jet2_logpt", "jet2_eta", "jet2_phi", "jet2_mass",
    "jet3_logpt", "jet3_eta", "jet3_phi", "jet3_mass",
]

met_cols = [
    "MET_pt", "MET_phi", "MET_significance", "MET_sumEt"
]

base_bdt_cols = tau1_cols + tau2_cols + jet_cols + met_cols


derived_bdt_cols = [
    "m_vis",
    "deltaEta_tautau",
    "deltaPhi_tautau",
    "abs_deltaPhi_tautau",
    "dR_tautau",
    "pT_tautau",
    "m_jj",
    "deltaEta_jj",
    "deltaPhi_jj",
    "abs_deltaPhi_jj",
    "dR_jj",
    "pT_jj",
    "deltaPhi_tau1MET",
    "deltaPhi_tau2MET",
    "abs_deltaPhi_tau1MET",
    "abs_deltaPhi_tau2MET",
    "METoverpTtautau",
    "log_pT_tautau",
    "log_m_jj",
    "log_METoverpTtautau",
]

tau_jet_cols = [
    f"dR_tau{tau}jet{jet}"
    for tau in [1, 2]
    for jet in [1, 2, 3]
]

bdt_cols = base_bdt_cols + derived_bdt_cols + tau_jet_cols


def delta_phi(phi1, phi2):
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def delta_R(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)


def inv_mass_two_objects(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    E = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    mass2 = E**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(mass2, 0.0))


def pt_two_objects(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)

    px = px1 + px2
    py = py1 + py2

    return np.sqrt(px**2 + py**2)


def add_derived_bdt_features(df):
    df = df.copy()

    df["jet1_pt"] = np.exp(df["jet1_logpt"])
    df["jet2_pt"] = np.exp(df["jet2_logpt"])
    df["jet3_pt"] = np.exp(df["jet3_logpt"])

    df["tau1_pt_for_features"] = df["tau1_pt_reco_corrPNet"]
    df["tau2_pt_for_features"] = df["tau2_pt_reco_corrPNet"]

    df["m_vis"] = inv_mass_two_objects(
        df["tau1_pt_for_features"], df["tau1_eta"], df["tau1_phi"], df["tau1_mass"],
        df["tau2_pt_for_features"], df["tau2_eta"], df["tau2_phi"], df["tau2_mass"],
    )

    df["deltaEta_tautau"] = df["tau1_eta"] - df["tau2_eta"]
    df["deltaPhi_tautau"] = delta_phi(df["tau1_phi"], df["tau2_phi"])
    df["abs_deltaPhi_tautau"] = np.abs(df["deltaPhi_tautau"])

    df["dR_tautau"] = delta_R(
        df["tau1_eta"], df["tau1_phi"],
        df["tau2_eta"], df["tau2_phi"],
    )

    df["pT_tautau"] = pt_two_objects(
        df["tau1_pt_for_features"], df["tau1_eta"], df["tau1_phi"], df["tau1_mass"],
        df["tau2_pt_for_features"], df["tau2_eta"], df["tau2_phi"], df["tau2_mass"],
    )

    for tau in [1, 2]:
        for jet in [1, 2, 3]:
            df[f"dR_tau{tau}jet{jet}"] = delta_R(
                df[f"tau{tau}_eta"],
                df[f"tau{tau}_phi"],
                df[f"jet{jet}_eta"],
                df[f"jet{jet}_phi"],
            )

    df["m_jj"] = inv_mass_two_objects(
        df["jet1_pt"], df["jet1_eta"], df["jet1_phi"], df["jet1_mass"],
        df["jet2_pt"], df["jet2_eta"], df["jet2_phi"], df["jet2_mass"],
    )

    df["deltaEta_jj"] = df["jet1_eta"] - df["jet2_eta"]
    df["deltaPhi_jj"] = delta_phi(df["jet1_phi"], df["jet2_phi"])
    df["abs_deltaPhi_jj"] = np.abs(df["deltaPhi_jj"])

    df["dR_jj"] = delta_R(
        df["jet1_eta"], df["jet1_phi"],
        df["jet2_eta"], df["jet2_phi"],
    )

    df["pT_jj"] = pt_two_objects(
        df["jet1_pt"], df["jet1_eta"], df["jet1_phi"], df["jet1_mass"],
        df["jet2_pt"], df["jet2_eta"], df["jet2_phi"], df["jet2_mass"],
    )

    df["deltaPhi_tau1MET"] = delta_phi(df["tau1_phi"], df["MET_phi"])
    df["deltaPhi_tau2MET"] = delta_phi(df["tau2_phi"], df["MET_phi"])

    df["abs_deltaPhi_tau1MET"] = np.abs(df["deltaPhi_tau1MET"])
    df["abs_deltaPhi_tau2MET"] = np.abs(df["deltaPhi_tau2MET"])

    df["METoverpTtautau"] = df["MET_pt"] / (df["pT_tautau"] + 1e-6)

    df["log_pT_tautau"] = np.log1p(df["pT_tautau"])
    df["log_m_jj"] = np.log1p(df["m_jj"])
    df["log_METoverpTtautau"] = np.log1p(df["METoverpTtautau"])

    return df


def check_bdt_features(df, features=bdt_cols):
    missing = [col for col in features if col not in df.columns]

    if missing:
        raise ValueError(f"Missing BDT features: {missing}")

    return True