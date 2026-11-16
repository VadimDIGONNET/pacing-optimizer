"""Tableau de bord Streamlit pour l'optimiseur de pacing.

Lancer avec :  streamlit run app.py

On ajuste l'athlete et le parcours dans la barre laterale et le tableau,
puis on compare pacing constant et pacing optimise : temps, puissance par
segment, vitesse, et bilan de W' le long du parcours.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pacing_model import (
    Rider, Environment, Course,
    simulate, constant_power_target, optimize_pacing,
)

st.set_page_config(page_title="Optimiseur de pacing", layout="wide")

# Couleurs des deux strategies (une teinte par entite, jamais recyclee)
C_CONST = "#6B7280"   # gris : pacing constant (reference)
C_OPT = "#0F766E"     # teal : pacing optimise
C_PROFILE = "#CBD5E1" # gris clair : profil du parcours

st.title("Optimiseur de pacing")
st.caption("Modeles Martin (1998) pour la puissance, Critical Power / W' "
           "(Skiba) pour la contrainte physiologique. A budget energetique "
           "identique, le pacing optimise minimise le temps.")

# --- Barre laterale : parametres athlete et environnement ---
with st.sidebar:
    st.header("Athlete")
    mass_rider = st.slider("Masse athlete (kg)", 45.0, 95.0, 68.0, 0.5)
    mass_bike = st.slider("Masse velo + equipement (kg)", 6.0, 12.0, 8.0, 0.1)
    cda = st.slider("CdA, trainee aero (m2)", 0.18, 0.40, 0.28, 0.005)
    crr = st.slider("Crr, roulement", 0.002, 0.008, 0.004, 0.0005, format="%.4f")
    cp = st.slider("Critical Power (W)", 180.0, 450.0, 300.0, 5.0)
    w_prime = st.slider("W', reserve anaerobie (kJ)", 8.0, 35.0, 20.0, 0.5) * 1000
    p_max = st.slider("Puissance plafond (W)", 400.0, 900.0, 600.0, 10.0)

    st.header("Environnement")
    rho = st.slider("Densite de l'air (kg/m3)", 0.95, 1.30, 1.225, 0.005)
    wind = st.slider("Vent de face (m/s)", -8.0, 8.0, 0.0, 0.5)

rider = Rider(mass_rider=mass_rider, mass_bike=mass_bike, cda=cda, crr=crr,
              cp=cp, w_prime=w_prime, p_max=p_max)
env = Environment(rho=rho, wind_ms=wind)

# --- Parcours editable ---
st.subheader("Parcours")
st.caption("Modifiez les segments (longueur en metres, pente en %). "
           "Ajoutez ou supprimez des lignes librement.")

if "course_df" not in st.session_state:
    demo = Course.rolling_demo()
    st.session_state.course_df = pd.DataFrame({
        "longueur_m": demo.length_m.astype(int),
        "pente_%": (demo.grade * 100).round(1),
    })

edited = st.data_editor(st.session_state.course_df, num_rows="dynamic",
                        use_container_width=True, key="editor")

course = Course(length_m=edited["longueur_m"].to_numpy(float),
                grade=(edited["pente_%"].to_numpy(float) / 100.0))

# --- Calculs ---
try:
    p_const = constant_power_target(course, rider, env)
    naive = simulate(np.full(course.n, p_const), course, rider, env)
    optimal = optimize_pacing(course, rider, env)
except Exception as exc:  # parcours ou parametres incoherents
    st.error(f"Calcul impossible avec ces parametres : {exc}")
    st.stop()

gain = naive.time_s - optimal.time_s

# --- Indicateurs cles ---
col1, col2, col3 = st.columns(3)
col1.metric("Temps pacing constant", naive.time_str, f"{p_const:.0f} W")
col2.metric("Temps pacing optimise", optimal.time_str,
            f"{optimal.avg_speed_kmh:.1f} km/h")
col3.metric("Gain", f"{gain:.1f} s", f"{gain / naive.time_s * 100:.2f} %")

# --- Graphiques : profil, puissance, bilan de W' ---
dist_km = np.concatenate([[0], np.cumsum(course.length_m)]) / 1000.0
elev = np.concatenate([[0], np.cumsum(course.length_m * course.grade)])


def steps(values):
    """Prolonge une valeur par segment pour un trace en marches d'escalier."""
    return np.concatenate([values, [values[-1]]])


fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(11, 8), sharex=True,
    gridspec_kw={"height_ratios": [1, 2, 1.3]},
)

ax1.fill_between(dist_km, elev, color=C_PROFILE)
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Profil du parcours")

ax2.step(dist_km, steps(np.full(course.n, p_const)), where="post",
         color=C_CONST, lw=2, label=f"Constant ({p_const:.0f} W)")
ax2.step(dist_km, steps(optimal.powers), where="post",
         color=C_OPT, lw=2.5, label="Optimise")
ax2.axhline(rider.cp, color="#B91C1C", ls="--", lw=1, label=f"CP ({rider.cp:.0f} W)")
ax2.set_ylabel("Puissance (W)")
ax2.legend(loc="upper right", fontsize=8)
ax2.set_title("Strategie de puissance")

ax3.step(dist_km, steps(naive.wbal_j / 1000), where="post", color=C_CONST, lw=2)
ax3.step(dist_km, steps(optimal.wbal_j / 1000), where="post", color=C_OPT, lw=2.5)
ax3.axhline(0, color="#B91C1C", ls="--", lw=1)
ax3.set_ylabel("Bilan W' (kJ)")
ax3.set_xlabel("Distance (km)")
ax3.set_title("Reserve anaerobie restante")

fig.tight_layout()
st.pyplot(fig)

# --- Detail par segment ---
st.subheader("Detail par segment")
table = pd.DataFrame({
    "longueur_m": course.length_m.astype(int),
    "pente_%": (course.grade * 100).round(1),
    "P optimise (W)": optimal.powers.round(0).astype(int),
    "vitesse (km/h)": (optimal.speed_ms * 3.6).round(1),
    "temps (s)": optimal.seg_time_s.round(1),
})
st.dataframe(table, use_container_width=True)
