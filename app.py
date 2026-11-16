"""Pacing Optimizer v2 : tableau de bord Streamlit.

Lancer :  streamlit run app.py

Fonctions :
  - parcours : GPX importe, ou mode demo (Tour de France 2026, etape 16),
    ou tableau de segments saisis a la main (comme la v1) ;
  - meteo : Open-Meteo a la date et l'heure choisies (densite de l'air et
    vent par segment), ou valeurs manuelles ;
  - athlete : saisie manuelle, ou CP/W' ajustes depuis un fichier de
    puissance (CSV secs,power ou export GoldenCheetah) ;
  - resultats : strategies constante / optimale (et reelle en mode demo),
    bilan W', temps par section, carte avec puissance cible et fleches de
    vent, export CSV et .zwo.
"""

from __future__ import annotations

import io
import sys
from datetime import date, time as dtime, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pacing_model import Rider, Environment, Course  # noqa: E402
from course_io import load_gpx, smooth_elevation, segment_track, SegmentedCourse  # noqa: E402
from weather import (fetch_hourly, sample_at, segment_environment, WeatherSample)  # noqa: E402
from dynamics import DynamicsParams, simulate_dynamic  # noqa: E402
from optimizer import optimize_dynamic, plan_table  # noqa: E402
from physiology import mmp_curve, fit_cp_wprime  # noqa: E402

st.set_page_config(page_title="Pacing Optimizer v2", layout="wide")

C_CONST, C_OPT, C_REAL, C_PROFILE = "#6B7280", "#0F766E", "#B45309", "#CBD5E1"
DEMO_GPX = ROOT / "data/raw/gpx/tdf2026_stage16.gpx"
DEMO_SPLITS = ROOT / "data/processed/tdf2026_stage16_splits.csv"
DEMO = {"date": date(2026, 7, 21), "time": dtime(17, 13), "mass": 63.0, "cda": 0.21, "cp": 430.0,
        "w_prime": 25.0, "breakpoints": "9.64, 10.9, 17.0", "a_lat": 8.0,
        "checkpoints": [(4.8, 488), (9.7, 942), (18.1, 1410), (26.1, 1939)]}

st.title("Pacing Optimizer v2")
st.caption("Parcours GPX, meteo Open-Meteo, physique dynamique (inertie, virages, freinage), "
           "contrainte W'bal. Les parametres athlete sont des hypotheses tant qu'ils ne viennent pas "
           "d'un capteur de puissance.")


# ---------------------------------------------------------------------------
# Chargement (cache)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_segmented(gpx_bytes: bytes, step_m: float) -> SegmentedCourse:
    tmp = ROOT / "data/raw/gpx/_upload.gpx"
    tmp.write_bytes(gpx_bytes)
    track = load_gpx(tmp)
    return segment_track(track, step_m=step_m, ele=smooth_elevation(track))


@st.cache_data(show_spinner=False)
def weather_env(lat_mid, lon_mid, ele_mid, bearing, dist_mid, n, day: str, when: str, lats, lons, dists):
    samples = []
    for la, lo, d in zip(lats, lons, dists):
        payload = fetch_hourly(float(la), float(lo), day, cache_dir=ROOT / "data/raw/weather")
        samples.append((float(d), sample_at(payload, datetime.fromisoformat(f"{day}T{when}"))))

    class _S:   # vue minimale pour segment_environment
        pass
    s = _S(); s.n = n; s.mid_dist_m = np.asarray(dist_mid); s.mid_ele = np.asarray(ele_mid)
    s.bearing_deg = np.asarray(bearing)
    env, df = segment_environment(s, samples)
    return env.rho, env.wind_ms, df, samples[0][1]


def fmt(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


# ---------------------------------------------------------------------------
# Barre laterale
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parcours")
    mode = st.radio("Source", ["Demo : Tour 2026, etape 16", "Fichier GPX", "Segments manuels"],
                    index=0 if DEMO_GPX.exists() else 2)
    step_m = st.select_slider("Resolution (m)", [50, 100], value=50)
    seg = None
    if mode.startswith("Demo"):
        if DEMO_GPX.exists():
            seg = load_segmented(DEMO_GPX.read_bytes(), float(step_m))
        else:
            st.error("GPX de demo absent (data/raw/gpx/tdf2026_stage16.gpx).")
    elif mode == "Fichier GPX":
        up = st.file_uploader("GPX", type=["gpx"])
        if up is not None:
            seg = load_segmented(up.getvalue(), float(step_m))
    demo = mode.startswith("Demo")

    st.header("Athlete")
    d = DEMO if demo else {"mass": 68.0, "cda": 0.28, "cp": 300.0, "w_prime": 20.0, "a_lat": 6.0}
    mass_rider = st.number_input("Masse athlete (kg)", 40.0, 110.0, d["mass"], 0.5)
    mass_bike = st.number_input("Masse velo + equipement (kg)", 6.0, 14.0, 8.5, 0.1)
    cda = st.number_input("CdA (m2)", 0.15, 0.45, d["cda"], 0.005, format="%.3f")
    crr = st.number_input("Crr", 0.002, 0.010, 0.0032, 0.0002, format="%.4f")
    power_file = st.file_uploader("Fichier de puissance (CSV secs,power) pour ajuster CP et W'", type=["csv"])
    cp_default, wp_default = d["cp"], d["w_prime"]
    if power_file is not None:
        try:
            pf = pd.read_csv(power_file)
            cols = {c.lower(): c for c in pf.columns}
            secs = pf[cols.get("secs", list(pf.columns)[0])].to_numpy(float)
            pw = pf[cols.get("power", list(pf.columns)[1])].to_numpy(float)
            grid = np.arange(0, secs[-1] + 1); p1 = np.interp(grid, secs, pw)
            cp_fit, wp_fit, r2 = fit_cp_wprime(mmp_curve(p1))
            if not np.isnan(cp_fit):
                cp_default, wp_default = float(cp_fit), float(wp_fit) / 1000.0
                st.success(f"CP {cp_fit:.0f} W, W' {wp_fit / 1000:.1f} kJ (R2 {r2:.3f}) ajustes sur 3-20 min")
        except Exception as exc:   # fichier illisible
            st.warning(f"Fichier de puissance non lu : {exc}")
    cp = st.number_input("Critical Power (W)", 100.0, 600.0, cp_default, 5.0)
    w_prime = st.number_input("W' (kJ)", 5.0, 45.0, wp_default, 0.5) * 1000.0
    p_max = st.number_input("Puissance plafond (W)", 300.0, 1200.0, 750.0 if demo else 600.0, 10.0)

    st.header("Meteo")
    wmode = st.radio("Source meteo", ["Open-Meteo (date et heure)", "Manuelle"], index=0)
    if wmode.startswith("Open"):
        day = st.date_input("Date", d.get("date", date.today()))
        when = st.time_input("Heure de depart (locale)", d.get("time", dtime(10, 0)))
    else:
        rho_manual = st.slider("Densite de l'air (kg/m3)", 0.95, 1.30, 1.15, 0.005)
        wind_manual = st.slider("Vent a 10 m (m/s)", 0.0, 15.0, 0.0, 0.5)
        wind_dir_manual = st.slider("Direction du vent (deg, d'ou il vient)", 0, 359, 0, 5)

    st.header("Physique et optimiseur")
    a_lat = st.slider("Acceleration laterale limite (m/s2)", 3.0, 10.0, d.get("a_lat", 6.0), 0.5,
                      help="6 = prudent ; 8 = valeur calee sur l'etape 16 (0,82 g)")
    width = st.slider("Largeur de route utilisable (m)", 0.0, 10.0, 6.0, 0.5)
    block_m = st.select_slider("Taille des blocs de puissance (m)", [250, 500, 1000], value=500)
    bp_text = st.text_input("Coupures de section (km, separees par des virgules ; vide = auto)",
                            d.get("breakpoints", ""))
    smooth = st.select_slider("Lissage du plan", ["faible", "moyen", "fort"], value="moyen")

# ---------------------------------------------------------------------------
# Parcours manuel (v1)
# ---------------------------------------------------------------------------

if mode == "Segments manuels":
    st.subheader("Segments")
    if "course_df" not in st.session_state:
        demo_c = Course.rolling_demo()
        st.session_state.course_df = pd.DataFrame({"longueur_m": demo_c.length_m.astype(int),
                                                   "pente_%": (demo_c.grade * 100).round(1)})
    edited = st.data_editor(st.session_state.course_df, num_rows="dynamic", use_container_width=True)
    L = edited["longueur_m"].to_numpy(float); g = edited["pente_%"].to_numpy(float) / 100.0
    nodes = np.concatenate([[0.0], np.cumsum(L)])
    ele = np.concatenate([[0.0], np.cumsum(L * g)])
    seg = SegmentedCourse(nodes, np.full(len(nodes), 46.0), np.full(len(nodes), 6.0), ele, L, g,
                          np.zeros(len(L)), np.full(len(L), np.inf), np.zeros(len(L)), corners=[])

if seg is None:
    st.info("Chargez un GPX ou choisissez le mode demo.")
    st.stop()

rider = Rider(mass_rider=mass_rider, mass_bike=mass_bike, crr=crr, cda=cda, cp=cp,
              w_prime=w_prime, p_max=p_max, p_min=0.0)
params = DynamicsParams(a_lat_max=a_lat, road_width_m=width)

# ---------------------------------------------------------------------------
# Meteo
# ---------------------------------------------------------------------------

wdf = None
if wmode.startswith("Open") and mode != "Segments manuels":
    try:
        idx = np.linspace(0, len(seg.dist_m) - 1, 3).round().astype(int)
        rho_arr, wind_arr, wdf, s0 = weather_env(
            None, None, tuple(seg.mid_ele), tuple(seg.bearing_deg), tuple(seg.mid_dist_m), seg.n,
            day.isoformat(), when.strftime("%H:%M:%S"),
            tuple(seg.lat[idx]), tuple(seg.lon[idx]), tuple(seg.dist_m[idx]))
        env = Environment(rho=np.asarray(rho_arr), wind_ms=np.asarray(wind_arr))
        st.caption(f"Meteo Open-Meteo au depart : {s0.temp_c:.1f} C, HR {s0.rh_pct:.0f} %, "
                   f"{s0.p_surface_hpa:.0f} hPa, vent {s0.wind10_ms:.1f} m/s de {s0.wind_dir_deg:.0f} deg "
                   f"(a 10 m) ; densite moyenne {np.mean(rho_arr):.3f} kg/m3, vent de face moyen "
                   f"{np.average(wind_arr, weights=seg.length_m):+.2f} m/s a 1 m.")
    except Exception as exc:
        st.warning(f"Meteo indisponible ({exc}) : valeurs manuelles par defaut.")
        env = Environment(rho=1.15, wind_ms=0.0)
else:
    from weather import wind_at_height, wind_components
    if wmode.startswith("Open"):
        env = Environment(rho=1.15, wind_ms=0.0)
    else:
        w1 = wind_at_height(wind_manual)
        head, _ = wind_components(w1, wind_dir_manual, seg.bearing_deg)
        env = Environment(rho=rho_manual, wind_ms=np.asarray(head))

# ---------------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------------

breakpoints = None
if bp_text.strip():
    try:
        breakpoints = [float(x) for x in bp_text.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        st.warning("Coupures illisibles, decoupage automatique utilise.")
lam = {"faible": 5e-5, "moyen": 2e-4, "fort": 1e-3}[smooth]

with st.spinner("Optimisation..."):
    try:
        opt = optimize_dynamic(seg, rider, env, params, block_m=float(block_m),
                               breakpoints_km=breakpoints, smooth_lambda=lam)
    except Exception as exc:
        st.error(f"Calcul impossible avec ces parametres : {exc}")
        st.stop()

real = None
if demo and DEMO_SPLITS.exists():
    from calibration import invert_sector_powers
    try:
        _, p_real = invert_sector_powers(seg, rider, env, DEMO["checkpoints"], params)
        real = simulate_dynamic(p_real, seg, rider, env, params)
    except Exception:
        real = None

# ---------------------------------------------------------------------------
# Indicateurs
# ---------------------------------------------------------------------------

cols = st.columns(4 if real is not None else 3)
cols[0].metric("Puissance constante", fmt(opt.constant.time_s), f"{opt.p_constant:.0f} W")
cols[1].metric("Pacing optimise", fmt(opt.result.time_s), f"{opt.result.avg_speed_kmh:.1f} km/h")
cols[2].metric("Gain", f"{opt.gain_s:.1f} s", f"{opt.gain_s / opt.constant.time_s * 100:.2f} %")
if real is not None:
    cols[3].metric("Strategie reelle (reconstruite)", fmt(real.time_s),
                   f"{real.time_s - opt.result.time_s:+.1f} s vs optimal")
st.caption(f"{len(opt.block_powers)} blocs, {opt.n_iter} iterations, {opt.elapsed_s:.1f} s de calcul ; "
           f"min W'bal {opt.result.min_wbal_j / 1000:.2f} kJ.")

# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

km_nodes = seg.dist_m / 1000.0
km_mid = seg.mid_dist_m / 1000.0
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2, 1.3]})
ax1.fill_between(km_nodes, seg.ele, seg.ele.min() - 20, color=C_PROFILE)
ax1.set_ylabel("Altitude (m)"); ax1.set_title("Profil")
ax2.plot(km_mid, opt.constant.applied_power, color=C_CONST, lw=1.5, label=f"Constante ({opt.p_constant:.0f} W)")
ax2.plot(km_mid, opt.result.powers, color=C_OPT, lw=2.2, label="Optimale (cible)")
ax2.plot(km_mid, opt.result.applied_power, color=C_OPT, lw=1, ls=":", label="Optimale (appliquee, freinages)")
if real is not None:
    ax2.plot(km_mid, real.powers, color=C_REAL, lw=1.5, label="Reelle (estimee des pointages)")
ax2.axhline(rider.cp, color="#B91C1C", ls="--", lw=1, label=f"CP {rider.cp:.0f} W")
ax2.set_ylabel("Puissance (W)"); ax2.legend(loc="lower left", fontsize=8, ncol=3); ax2.set_title("Strategies")
ax3.plot(km_mid, opt.constant.wbal_j / 1000, color=C_CONST, lw=1.5)
ax3.plot(km_mid, opt.result.wbal_j / 1000, color=C_OPT, lw=2.2)
if real is not None:
    ax3.plot(km_mid, real.wbal_j / 1000, color=C_REAL, lw=1.5)
ax3.axhline(0, color="#B91C1C", ls="--", lw=1); ax3.set_ylabel("W'bal (kJ)"); ax3.set_xlabel("Distance (km)")
fig.tight_layout()
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Carte
# ---------------------------------------------------------------------------

if mode != "Segments manuels":
    st.subheader("Carte : puissance cible par bloc et vent")
    tab = plan_table(opt, seg)
    pmin, pmax = tab.power_w.min(), tab.power_w.max()

    def color(p):
        t = 0.0 if pmax == pmin else (p - pmin) / (pmax - pmin)
        return [int(37 + (185 - 37) * t), int(99 + (28 - 99) * t), int(235 + (28 - 235) * t), 200]

    paths = []
    for row in tab.itertuples():
        m = (km_nodes >= row.km_start - 1e-9) & (km_nodes <= row.km_end + 1e-9)
        pts = [[float(lo), float(la)] for lo, la in zip(seg.lon[m], seg.lat[m])]
        if len(pts) >= 2:
            paths.append({"path": pts, "color": color(row.power_w),
                          "label": f"km {row.km_start:.1f}-{row.km_end:.1f} : {row.power_w:.0f} W, "
                                   f"{row.speed_kmh:.0f} km/h, pente {row.grade_pct:+.1f} %"})
    layers = [pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=4,
                        pickable=True)]
    if wdf is not None and len(wdf):
        arrows = []
        step = max(seg.n // 25, 1)
        for i in range(0, seg.n, step):
            wd = float(wdf.wind_dir_deg.iloc[i]); spd = float(wdf.wind_rider_ms.iloc[i])
            # le vent souffle VERS wd + 180 ; fleche de 300 m pour 5 m/s
            ang = np.radians(wd + 180.0); L = 60.0 * max(spd, 0.5)
            la0, lo0 = float(seg.lat[i]), float(seg.lon[i])
            la1 = la0 + L * np.cos(ang) / 111320.0
            lo1 = lo0 + L * np.sin(ang) / (111320.0 * np.cos(np.radians(la0)))
            arrows.append({"path": [[lo0, la0], [lo1, la1]], "label": f"vent {spd:.1f} m/s de {wd:.0f} deg"})
        layers.append(pdk.Layer("PathLayer", data=arrows, get_path="path", get_color=[120, 120, 120, 180],
                                width_min_pixels=2, pickable=True))
    view = pdk.ViewState(latitude=float(seg.lat.mean()), longitude=float(seg.lon.mean()), zoom=11.5)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{label}"},
                             map_style=None))
    st.caption("Bleu = puissance basse, rouge = puissance haute. Fleches grises : vent a hauteur de coureur.")
else:
    tab = plan_table(opt, seg)

# ---------------------------------------------------------------------------
# Tables et exports
# ---------------------------------------------------------------------------

st.subheader("Plan de puissance par bloc")
show = tab.rename(columns={"km_start": "km debut", "km_end": "km fin", "grade_pct": "pente %",
                           "power_w": "P cible (W)", "applied_w": "P appliquee (W)",
                           "speed_kmh": "vitesse (km/h)", "time_s": "temps (s)",
                           "wbal_end_kj": "W'bal fin (kJ)", "braking": "freinage"})
st.dataframe(show.round(1), use_container_width=True, hide_index=True)

if breakpoints:
    st.subheader("Temps par section")
    edges = [0.0] + sorted(breakpoints) + [seg.total_km]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = seg.slice_km(a, b)
        if len(idx) == 0:
            continue
        row = {"section": f"km {a:.1f} - {b:.1f}", "constante (s)": opt.constant.seg_time_s[idx].sum(),
               "optimale (s)": opt.result.seg_time_s[idx].sum()}
        if real is not None:
            row["reelle (s)"] = real.seg_time_s[idx].sum()
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(1), use_container_width=True, hide_index=True)

csv = tab.round(1).to_csv(index=False).encode()
c1, c2 = st.columns(2)
c1.download_button("Exporter le plan (CSV)", csv, "pacing_plan.csv", "text/csv")


def to_zwo(tab, name="Pacing plan"):
    """Seance Zwift (.zwo) en blocs de puissance absolue : un bloc par ligne du plan.
    Les durees sont les temps prevus par bloc (un compteur ne sait pas declencher
    sur la distance en .zwo)."""
    parts = [f'<workout_file><author>Pacing Optimizer</author><name>{name}</name>'
             '<description>Plan de puissance par bloc</description><sportType>bike</sportType><workout>']
    for row in tab.itertuples():
        parts.append(f'<SteadyState Duration="{max(int(round(row.time_s)), 1)}" '
                     f'Power="{row.power_w:.0f}" />')
    parts.append("</workout></workout_file>")
    return "\n".join(parts).encode()


c2.download_button("Exporter en seance (.zwo, puissance absolue)", to_zwo(tab), "pacing_plan.zwo", "application/xml")
st.caption("Hypotheses : masse, CdA, Crr, W' et limite de virage sont des entrees, pas des mesures. "
           "En mode demo, la strategie reelle est reconstruite a partir des pointages publies, ce ne sont "
           "pas des watts mesures.")
