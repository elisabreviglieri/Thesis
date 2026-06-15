# fm/physics.py
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------
# 4-vector helpers
# ------------------------------------------------------------

def pt_eta_phi_m_to_p4(pt, eta, phi, m):
    """
    Convert (pt, eta, phi, m) to Cartesian 4-momentum components.

    Returns:
        E, px, py, pz  (numpy arrays)
    """
    pt  = np.asarray(pt)
    eta = np.asarray(eta)
    phi = np.asarray(phi)
    m   = np.asarray(m)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    p2 = px**2 + py**2 + pz**2
    E  = np.sqrt(np.maximum(p2 + m**2, 0.0))

    return E, px, py, pz


def inv_mass_two_objects(pt1, eta1, phi1, m1,
                         pt2, eta2, phi2, m2):
    """
    Compute invariant mass of two objects given (pt, eta, phi, m).
    Returns numpy array.
    """
    E1, px1, py1, pz1 = pt_eta_phi_m_to_p4(pt1, eta1, phi1, m1)
    E2, px2, py2, pz2 = pt_eta_phi_m_to_p4(pt2, eta2, phi2, m2)

    E  = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_tot = E**2 - (px**2 + py**2 + pz**2)

    return np.sqrt(np.maximum(m2_tot, 0.0))


# ------------------------------------------------------------
# Tau-specific helpers (my target is ratio = pt_gen/pt_reco)
# ------------------------------------------------------------

def corrected_pt_ratio(pt, ratio):
    """
    Apply tau correction where ratio = pt_gen / pt_reco.
    So: pt_corr = pt_reco * ratio
    """
    pt = np.asarray(pt)
    ratio = np.asarray(ratio)
    return pt * ratio


def inv_mass_two_taus_corrected_ratio(
    pt1, eta1, phi1, m1, ratio1,
    pt2, eta2, phi2, m2, ratio2,
):
    """
    Invariant mass after applying pt corrections defined as ratios.
    """
    pt1_corr = corrected_pt_ratio(pt1, ratio1)
    pt2_corr = corrected_pt_ratio(pt2, ratio2)

    return inv_mass_two_objects(
        pt1_corr, eta1, phi1, m1,
        pt2_corr, eta2, phi2, m2,
    )
    
    
    
def delta_phi(phi1, phi2):
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)

    return np.arctan2(
        np.sin(phi1 - phi2),
        np.cos(phi1 - phi2),
    )


def delta_R(eta1, phi1, eta2, phi2):
    eta1 = np.asarray(eta1)
    eta2 = np.asarray(eta2)

    d_eta = eta1 - eta2
    d_phi = delta_phi(phi1, phi2)

    return np.sqrt(d_eta**2 + d_phi**2)


def pt_two_objects(pt1, eta1, phi1, m1,
                   pt2, eta2, phi2, m2):
    """
    Transverse momentum of the system made by two objects.
    """
    E1, px1, py1, pz1 = pt_eta_phi_m_to_p4(pt1, eta1, phi1, m1)
    E2, px2, py2, pz2 = pt_eta_phi_m_to_p4(pt2, eta2, phi2, m2)

    px = px1 + px2
    py = py1 + py2

    return np.sqrt(px**2 + py**2)

