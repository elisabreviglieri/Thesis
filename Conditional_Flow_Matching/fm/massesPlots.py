# fm/massesPlots.py

import numpy as np
import matplotlib.pyplot as plt


def plot_mass_overlay(masses_dict, title,
                      bins=120,
                      range_=None,
                      density=True,
                      alpha=0.6,
                      logy=False):

    plt.figure(figsize=(9,5))

    for mode, m in masses_dict.items():
        m = np.asarray(m)
        plt.hist(
            m,
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2.0,
            label=mode,
        )

    plt.xlabel("inv_mass_FM")
    plt.ylabel("density" if density else "counts")
    plt.title(title)
    plt.legend(fontsize=8, loc="upper right")

    if logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()
    
def plot_balanced_vs_unbalanced_by_mode(
    masses_bal: dict,
    masses_unbal: dict,
    df_test_bal,
    df_test_unbal,
    scaling_modes: list,
    bins: int = 120,
    range_: tuple | None = (0, 300),
    density: bool = True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    c_bal = df_test_bal["class"].to_numpy()
    c_unb = df_test_unbal["class"].to_numpy()

    m_dy_bal = (c_bal == 0)
    m_h_bal  = (c_bal == 1)

    m_dy_unb = (c_unb == 0)
    m_h_unb  = (c_unb == 1)

    for mode in scaling_modes:

        mb = np.asarray(masses_bal[mode])
        mu = np.asarray(masses_unbal[mode])

        dy_bal = mb[m_dy_bal]
        dy_unb = mu[m_dy_unb]

        h_bal  = mb[m_h_bal]
        h_unb  = mu[m_h_unb]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        # --- DY
        axes[0].hist(dy_bal, bins=bins, range=range_, density=density,
                     histtype="step", linewidth=2, label="balanced")
        axes[0].hist(dy_unb, bins=bins, range=range_, density=density,
                     histtype="step", linewidth=2, label="unbalanced")
        axes[0].set_title("Drell-Yan (class=0)")
        axes[0].set_xlabel("inv_mass_FM")
        axes[0].set_ylabel("density" if density else "counts")
        axes[0].legend(fontsize=8)

        # --- Higgs
        axes[1].hist(h_bal, bins=bins, range=range_, density=density,
                     histtype="step", linewidth=2, label="balanced")
        axes[1].hist(h_unb, bins=bins, range=range_, density=density,
                     histtype="step", linewidth=2, label="unbalanced")
        axes[1].set_title("Higgs (class=1)")
        axes[1].set_xlabel("inv_mass_FM")
        axes[1].legend(fontsize=8)

        fig.suptitle(f"FM invariant mass — {mode} — balanced vs unbalanced")
        plt.tight_layout()
        plt.show()
        
        
        

# Post flat samples training & inference:
# mi serve per fare il confronto anche con i diversi scaling fatti in fase pre-training 

def plot_mass_A_vs_B_by_class(
    masses_A: dict,
    masses_B: dict,
    df_test_raw,
    modes: list,
    title_prefix: str,
    bins: int = 120,
    range_: tuple = (0, 300),
    density: bool = True,
    color_A: str = "green",
    color_B: str = "red",
):
    c = df_test_raw["class"].to_numpy()
    m_dy = (c == 0)
    m_h  = (c == 1)

    for mode in modes:
        if mode not in masses_A or mode not in masses_B:
            continue

        mA = np.asarray(masses_A[mode])
        mB = np.asarray(masses_B[mode])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        # DY
        axes[0].hist(
            mA[m_dy],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            color=color_A,
            label="A (init NO-SCALE)",
        )
        axes[0].hist(
            mB[m_dy],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            color=color_B,
            label="B (init GLOBAL)",
        )
        axes[0].set_title("Drell–Yan (class=0)")
        axes[0].set_xlabel("m(ττ) [GeV]")
        axes[0].set_ylabel("density" if density else "counts")
        axes[0].legend(fontsize=8)

        # Higgs
        axes[1].hist(
            mA[m_h],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            color=color_A,
            label="A (init NO-SCALE)",
        )
        axes[1].hist(
            mB[m_h],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            color=color_B,
            label="B (init GLOBAL)",
        )
        axes[1].set_title("Higgs (class=1)")
        axes[1].set_xlabel("m(ττ) [GeV]")
        axes[1].legend(fontsize=8)

        fig.suptitle(f"{title_prefix} — {mode}")
        plt.tight_layout()
        plt.show()
        
        
# Griglia di plot per comparison : balanced vs unbalanced, patience30 vs patience100 per tutti i mode
# separando Higgs e Drell-Yan
def plot_mass_grid_by_mode(
    masses_a_pat30,
    masses_a_pat100,
    masses_b_pat30,
    masses_b_pat100,
    scaling_modes,
    main_title,
    label_a="MSE",
    label_b="HUBER",
    bins=120,
    range_=(0, 300),
    density=True,
    add_higgs_line=False,
):
    for mode in scaling_modes:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

        # 1) A pat30 vs A pat100
        axes[0].hist(
            masses_a_pat30[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_a} pat30",
        )
        axes[0].hist(
            masses_a_pat100[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_a} pat100",
        )
        axes[0].set_title(f"{label_a} pat30 vs pat100")
        axes[0].legend()

        # 2) B pat30 vs B pat100
        axes[1].hist(
            masses_b_pat30[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_b} pat30",
        )
        axes[1].hist(
            masses_b_pat100[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_b} pat100",
        )
        axes[1].set_title(f"{label_b} pat30 vs pat100")
        axes[1].legend()

        # 3) A vs B at pat30
        axes[2].hist(
            masses_a_pat30[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_a} pat30",
        )
        axes[2].hist(
            masses_b_pat30[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_b} pat30",
        )
        axes[2].set_title(f"{label_a} vs {label_b} (pat30)")
        axes[2].legend()

        # 4) A vs B at pat100
        axes[3].hist(
            masses_a_pat100[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_a} pat100",
        )
        axes[3].hist(
            masses_b_pat100[mode],
            bins=bins,
            range=range_,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label_b} pat100",
        )
        axes[3].set_title(f"{label_a} vs {label_b} (pat100)")
        axes[3].legend()

        if add_higgs_line:
            for ax in axes:
                ax.axvline(125, linestyle="--", linewidth=1)

        fig.suptitle(f"{main_title} — {mode}", fontsize=14)
        plt.tight_layout()
        plt.show()