#!/usr/bin/env python3
# compton_app.py
"""
Interactive Compton scattering mini-tutor.

Requirements:
    pip install streamlit numpy matplotlib pandas

Run:
    streamlit run compton_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- CONSTANTS ----------
m_e_c2_keV = 510.99895  # electron rest mass energy, keV
r_e_cm = 2.8179403262e-13  # classical electron radius, cm
r_e = r_e_cm  # keep symbolic name
# photon energies domain
E_MIN_KEV, E_MAX_KEV = 1.0, 2.0e4  # 1 keV to 20 MeV


# ---------- PHYSICS FUNCTIONS ----------
def klein_nishina_dsigma_domega(E_keV, theta):
    """
    Klein-Nishina differential cross section (per electron) in cm^2/sr.
    E_keV: incident photon energy (keV)
    theta: scattering angle (radians)
    """
    alpha = E_keV / m_e_c2_keV
    # Scattered photon energy ratio
    Eprime_over_E = 1.0 / (1.0 + alpha * (1 - np.cos(theta)))
    term1 = Eprime_over_E**2
    term2 = Eprime_over_E + 1.0 / Eprime_over_E - np.sin(theta) ** 2
    return 0.5 * r_e**2 * term1 * term2


def scattered_energy(E_keV, theta):
    """Scattered photon energy (keV) given incident E and angle theta (rad)."""
    alpha = E_keV / m_e_c2_keV
    return E_keV / (1.0 + alpha * (1 - np.cos(theta)))


def wavelength_shift(theta):
    """
    Compton wavelength shift Δλ = λ' - λ = (h / (m_e c))(1 - cos θ)
    Returned in pm (picometers).
    λ_c = 2.42631023867 pm.
    """
    lambda_c_pm = 2.42631023867
    return lambda_c_pm * (1 - np.cos(theta))


def total_kn_cross_section(E_keV, n_theta=10000):
    """
    Numerically integrate Klein-Nishina over solid angle to get total per-electron
    cross section (cm^2). Fast enough for interactive use.
    """
    # integrate over theta from 0 to pi: dΩ = 2π sinθ dθ
    thetas = np.linspace(0, np.pi, n_theta)
    dsigma = klein_nishina_dsigma_domega(E_keV, thetas)
    integrand = dsigma * 2 * np.pi * np.sin(thetas)
    return np.trapz(integrand, thetas)


# ---------- MATERIAL DATABASE ----------
# Simple Z list; you can extend/replace with full periodic table as needed.
MATERIALS = {
    "Water (H2O)": 10,   # effective Z ~10
    "Soft Tissue": 7.4,  # rough effective Z
    "Bone": 13.8,        # cortical bone effective Z
    "Aluminum": 13,
    "Iron": 26,
    "Copper": 29,
    "Silver": 47,
    "Iodine": 53,
    "Tungsten": 74,
    "Lead": 82,
}

# Compton scattering is ~proportional to electron density not Z per se,
# but users often vary Z to visualize trend. We'll scale by Z (proxy) here.


# ---------- PLOTTING HELPERS ----------
plt.style.use("dark_background")

def make_kn_angle_plot(E_keV, Z_eff, n_points=361):
    theta = np.linspace(0, np.pi, n_points)
    dsigma = klein_nishina_dsigma_domega(E_keV, theta) * Z_eff  # scale by Z_eff
    Eprime = scattered_energy(E_keV, theta)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(np.degrees(theta), dsigma, label=r"$\frac{d\sigma}{d\Omega}$ (scaled by $Z_\mathrm{eff}$)")
    ax1.set_xlabel(r"$\theta$ (deg)")
    ax1.set_ylabel(r"$\frac{d\sigma}{d\Omega}$ (cm$^2$/sr)")

    ax2 = ax1.twinx()
    ax2.plot(np.degrees(theta), Eprime, linestyle="--", label=r"$E'(\theta)$")
    ax2.set_ylabel(r"$E'_\gamma$ (keV)")

    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, pd.DataFrame({
        "theta_deg": np.degrees(theta),
        "dsigma_domega_cm2_sr_scaled": dsigma,
        "Eprime_keV": Eprime,
        "dLambda_pm": wavelength_shift(theta),
    })


def make_total_cs_vs_energy_plot(Z_eff, n_E=300):
    energies = np.logspace(np.log10(E_MIN_KEV), np.log10(E_MAX_KEV), n_E)
    sigma_tot = np.array([total_kn_cross_section(E) * Z_eff for E in energies])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(energies, sigma_tot)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel(r"Total $\sigma_\mathrm{KN}$ (cm$^2$) × $Z_\mathrm{eff}$")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig, pd.DataFrame({"E_keV": energies, "sigma_tot_cm2_scaled": sigma_tot})


# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="Compton Scattering Tutor", layout="wide")
    st.title("Compton Scattering Interactive Tutor for Medical Physics at Penn")

    # Sidebar controls
    st.sidebar.header("Controls")

    material = st.sidebar.selectbox("Material (effective Z)", list(MATERIALS.keys()))
    Z_eff = MATERIALS[material]

    E_keV = st.sidebar.slider("Incident photon energy (keV)", float(E_MIN_KEV), float(E_MAX_KEV), 100.0, step=1.0)
    show_total_curve = st.sidebar.checkbox("Show total cross section vs energy", value=True)
    export_csv = st.sidebar.checkbox("Allow CSV download", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Angle Resolution**")
    n_points = st.sidebar.slider("Points in angle plot", 90, 1441, 361, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Extra Features**")
    show_wavelength_shift = st.sidebar.checkbox("Display wavelength shift table segment", value=False)

    # Equations section
    st.subheader("Key Equations")
    st.latex(r"""E' = \frac{E}{1 + \frac{E}{m_ec^2}(1-\cos\theta)}""")
    st.latex(r"""\Delta \lambda = \lambda' - \lambda = \frac{h}{m_ec}(1-\cos\theta)""")
    st.latex(r"""\frac{d\sigma}{d\Omega} = \frac{r_e^2}{2} \left(\frac{E'}{E}\right)^2
    \left(\frac{E'}{E} + \frac{E}{E'} - \sin^2\theta\right)""")
    st.caption("All cross sections shown are per electron and then scaled by the chosen effective Z for visualization.")

    # Angle plot and data
    fig1, df1 = make_kn_angle_plot(E_keV, Z_eff, n_points=n_points)
    st.subheader("Angular Distribution and Scattered Energy")
    st.pyplot(fig1, use_container_width=True)

    if show_wavelength_shift:
        st.markdown("**Sample of wavelength shifts (Δλ, pm) at select angles:**")
        sample = df1.loc[df1['theta_deg'].round().isin([0, 30, 60, 90, 120, 150, 180])]
        st.dataframe(sample[['theta_deg', 'dLambda_pm']].set_index('theta_deg'))

    # Total cross section vs energy
    if show_total_curve:
        fig2, df2 = make_total_cs_vs_energy_plot(Z_eff)
        st.subheader("Total Klein–Nishina Cross Section vs Energy")
        st.pyplot(fig2, use_container_width=True)

    # Download CSVs
    if export_csv:
        st.download_button("Download angle-resolved data (CSV)", df1.to_csv(index=False), file_name="angle_data.csv")
        if show_total_curve:
            st.download_button("Download total σ(E) data (CSV)", df2.to_csv(index=False), file_name="total_sigma.csv")

    # Notes / caveats
    with st.expander("Notes and Optional Extensions"):
        st.markdown(
            "- Scaling by Z is a heuristic; true macroscopic attenuation scales with electron density and form factors.\n"
            "- At low energies (< tens of keV), binding effects and Rayleigh scatter matter.\n"
            "- At high Z/MeV, pair production competes; you could add μ/ρ curves from NIST XCOM.\n"
            "- Add polar plots of dσ/dΩ or energy-transfer to electrons if you want more depth."
        )

    st.markdown("---")
    st.markdown(
        "Built for quick exploration: vary energy and Z to see how angular distribution and total cross section change."
    )


if __name__ == "__main__":
    main()
