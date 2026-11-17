"""Pacing Intelligence : tableau de bord d'analyse pour staff performance.

Lancer :  streamlit run app.py

Contenu :
  - modele predictif hybride (physique de Martin + gradient boosting
    entraine sur les segments GoldenCheetah OpenData) : temps atteignable
    sur un parcours pour un profil de coureur, avec incertitude validee ;
  - plan de puissance optimal par section (simulateur dynamique, W'bal) ;
  - conditions reelles (Open-Meteo), carte, exports ;
  - validation : erreurs mesurees sur sorties et athletes jamais vus, et sur
    les pointages du Tour de France 2026 (etape 16).
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pacing_model import Rider, Environment  # noqa: E402
from course_io import load_gpx, smooth_elevation, segment_track  # noqa: E402
from weather import fetch_hourly, sample_at, segment_environment  # noqa: E402
from dynamics import DynamicsParams, simulate_dynamic  # noqa: E402
from optimizer import optimize_dynamic, plan_table  # noqa: E402
from ml_model import HybridModel, predict_course_fast, MODEL_DIR  # noqa: E402

st.set_page_config(page_title="Pacing Intelligence", page_icon="⏱", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
TEAL, AMBER, SLATE, RED, BLUE = "#2DD4BF", "#F59E0B", "#94A3B8", "#F87171", "#60A5FA"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid #1F2A44; padding-bottom:12px; margin-bottom:18px; }
.hero .t { font-size:30px; font-weight:700; color:#F8FAFC; }
.hero .s { color:#94A3B8; font-size:14px; margin-top:2px; }
.hero .tag { background:#0F2A2B; color:#2DD4BF; border:1px solid #1E4A47; border-radius:999px; padding:4px 12px; font-size:12px; font-weight:600; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin: 6px 0 18px 0; }
.kpi { background:linear-gradient(180deg,#121C33 0%,#0F172A 100%); border:1px solid #1F2A44; border-radius:12px; padding:14px 16px; }
.kpi .l { color:#94A3B8; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
.kpi .v { font-family:'JetBrains Mono', monospace; font-size:22px; font-weight:500; color:#F8FAFC; margin-top:4px; white-space:nowrap; }
.kpi .d { font-size:12px; color:#2DD4BF; margin-top:2px; }
.kpi .d.warn { color:#F59E0B; }
.kpi .d.muted { color:#94A3B8; }
.sec { color:#F8FAFC; font-weight:600; font-size:17px; margin: 14px 0 6px 0; }
.note { background:#111A2E; border-left:3px solid #F59E0B; padding:10px 14px; border-radius:6px; color:#CBD5E1; font-size:13px; }
.src { color:#64748B; font-size:12px; }
[data-testid="stSidebar"] { background:#0B1220; border-right:1px solid #1F2A44; }
div[data-testid="stDataFrame"] { border:1px solid #1F2A44; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#0B1220", plot_bgcolor="#0F172A",
                   font=dict(family="Inter", color="#CBD5E1", size=12), margin=dict(l=50, r=20, t=40, b=40),
                   legend=dict(orientation="h", y=-0.22, x=0), hovermode="x unified")


def kpi(label, value, delta="", cls=""):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="d {cls}">{delta}</div></div>'


def fmt(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


# ---------------------------------------------------------------------------
# Chargement (cache)
# ---------------------------------------------------------------------------
DEMO_GPX = ROOT / "data/raw/gpx/tdf2026_stage16.gpx"
DEMO = {"date": date(2026, 7, 21), "time": dtime(17, 13), "mass": 63.0, "cda": 0.21, "cp": 430.0, "w_prime": 25.0,
        "breakpoints": "9.64, 10.9, 17.0", "a_lat": 8.0, "checkpoints": [(4.8, 488), (9.7, 942), (18.1, 1410), (26.1, 1939)]}


@st.cache_resource(show_spinner=False)
def load_model():
    p = MODEL_DIR / "hybrid.joblib"
    if not p.exists():
        return None, {}
    m = HybridModel.load(p)
    metrics = json.load(open(MODEL_DIR / "metrics.json")) if (MODEL_DIR / "metrics.json").exists() else {}
    return m, metrics


@st.cache_data(show_spinner=False)
def load_segmented(gpx_bytes: bytes, step_m: float):
    tmp = ROOT / "data/raw/gpx/_upload.gpx"; tmp.write_bytes(gpx_bytes)
    track = load_gpx(tmp)
    return segment_track(track, step_m=step_m, ele=smooth_elevation(track))


@st.cache_data(show_spinner=False)
def weather_arrays(day: str, when: str, lats, lons, dists, mid_ele, bearing, mid_dist, n):
    samples = []
    for la, lo, d in zip(lats, lons, dists):
        payload = fetch_hourly(float(la), float(lo), day, cache_dir=ROOT / "data/raw/weather")
        samples.append((float(d), sample_at(payload, datetime.fromisoformat(f"{day}T{when}"))))

    class S:
        pass
    s = S(); s.n = n; s.mid_dist_m = np.asarray(mid_dist); s.mid_ele = np.asarray(mid_ele); s.bearing_deg = np.asarray(bearing)
    env, df = segment_environment(s, samples)
    return np.asarray(env.rho), np.asarray(env.wind_ms), df, samples[0][1]


@st.cache_data(show_spinner=False)
def load_validation_tables():
    out = {}
    for name in ("stage16_loo", "stage16_strategies", "stage16_sectors", "stage16_loo_hybrid"):
        p = ROOT / f"reports/tables/{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    p = ROOT / "data/processed/opendata_segments/athletes.csv"
    if p.exists():
        out["athletes"] = pd.read_csv(p)
    return out


model, metrics = load_model()
tables = load_validation_tables()


@st.cache_data(show_spinner=False)
def training_domain():
    p = MODEL_DIR / "profiles.json"
    if not p.exists():
        return None
    prof = pd.DataFrame(json.load(open(p))).T
    prof = prof[prof.cp_w.notna() & (prof.mass_kg > 0)]
    wkg = prof.cp_w / prof.mass_kg
    return {"wkg_max": float(wkg.quantile(0.95)), "cda_min": float(prof.cda.quantile(0.05)), "n": int(len(prof))}


domain = training_domain()

# ---------------------------------------------------------------------------
# Barre laterale : entrees
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Parcours")
    mode = st.radio("Source", ["Tour de France 2026, étape 16", "Fichier GPX"], label_visibility="collapsed")
    seg = None
    if mode.startswith("Tour"):
        if DEMO_GPX.exists():
            seg = load_segmented(DEMO_GPX.read_bytes(), 50.0)
    else:
        up = st.file_uploader("GPX", type=["gpx"])
        if up is not None:
            seg = load_segmented(up.getvalue(), 50.0)
    demo = mode.startswith("Tour")
    d = DEMO if demo else {"mass": 70.0, "cda": 0.23, "cp": 380.0, "w_prime": 22.0, "a_lat": 6.0, "breakpoints": ""}

    st.markdown("### Coureur")
    c1, c2 = st.columns(2)
    mass_rider = c1.number_input("Masse (kg)", 40.0, 110.0, d["mass"], 0.5)
    mass_bike = c2.number_input("Vélo (kg)", 6.0, 14.0, 8.5, 0.1)
    cda = c1.number_input("CdA (m²)", 0.15, 0.60, d["cda"], 0.005, format="%.3f")
    crr = c2.number_input("Crr", 0.002, 0.012, 0.0032, 0.0002, format="%.4f")
    cp = c1.number_input("CP (W)", 100.0, 600.0, d["cp"], 5.0)
    w_prime = c2.number_input("W' (kJ)", 5.0, 45.0, d["w_prime"], 0.5) * 1000.0
    p_max = st.number_input("Puissance plafond (W)", 300.0, 1200.0, 750.0 if demo else 650.0, 10.0)

    st.markdown("### Conditions")
    wmode = st.radio("Météo", ["Open-Meteo à la date", "Manuelle"], label_visibility="collapsed")
    if wmode.startswith("Open"):
        day = st.date_input("Date", d.get("date", date.today()))
        when = st.time_input("Heure de départ", d.get("time", dtime(10, 0)))
    else:
        rho_manual = st.slider("Densité de l'air (kg/m³)", 0.95, 1.30, 1.15, 0.005)
        wind_manual = st.slider("Vent à 10 m (m/s)", 0.0, 15.0, 0.0, 0.5)
        wind_dir_manual = st.slider("Direction du vent (°, provenance)", 0, 359, 0, 5)

    with st.expander("Physique et optimiseur"):
        a_lat = st.slider("Limite latérale en virage (m/s²)", 3.0, 10.0, d.get("a_lat", 6.0), 0.5)
        width = st.slider("Largeur de route utilisable (m)", 0.0, 10.0, 6.0, 0.5)
        block_m = st.select_slider("Blocs de puissance (m)", [250, 500, 1000], value=500)
        bp_text = st.text_input("Coupures de section (km)", d.get("breakpoints", ""))
        smooth = st.select_slider("Lissage du plan", ["faible", "moyen", "fort"], value="moyen")

# ---------------------------------------------------------------------------
# En-tete
# ---------------------------------------------------------------------------
n_seg = metrics.get("n_segments", 0); n_ath = metrics.get("n_athletes", 0)
ho = metrics.get("held_out_athletes", {})
st.markdown(f"""
<div class="hero"><div><div class="t">Pacing Intelligence</div>
<div class="s">Prédiction du temps atteignable et plan de puissance optimal, à partir d'un modèle entraîné sur des sorties réelles.</div></div>
<div><span class="tag">modèle : {n_seg:,} segments, {n_ath} athlètes</span></div></div>
""".replace(",", " "), unsafe_allow_html=True)

if seg is None:
    st.info("Chargez un GPX ou choisissez l'étape 16.")
    st.stop()

rider = Rider(mass_rider=mass_rider, mass_bike=mass_bike, crr=crr, cda=cda, cp=cp, w_prime=w_prime, p_max=p_max, p_min=0.0)
params = DynamicsParams(a_lat_max=a_lat, road_width_m=width)
profile = {"mass_kg": mass_rider, "cda": cda, "crr": crr}

# Meteo
wdf, s0 = None, None
if wmode.startswith("Open"):
    try:
        idx = np.linspace(0, len(seg.dist_m) - 1, 3).round().astype(int)
        rho_arr, wind_arr, wdf, s0 = weather_arrays(day.isoformat(), when.strftime("%H:%M:%S"),
                                                     tuple(seg.lat[idx]), tuple(seg.lon[idx]), tuple(seg.dist_m[idx]),
                                                     tuple(seg.mid_ele), tuple(seg.bearing_deg), tuple(seg.mid_dist_m), seg.n)
        env = Environment(rho=rho_arr, wind_ms=wind_arr)
    except Exception as exc:
        st.warning(f"Météo indisponible ({exc}) : densité 1,15 kg/m³, vent nul.")
        env = Environment(rho=1.15, wind_ms=0.0)
else:
    from weather import wind_at_height, wind_components
    head, _ = wind_components(wind_at_height(wind_manual), wind_dir_manual, seg.bearing_deg)
    env = Environment(rho=rho_manual, wind_ms=np.asarray(head))

breakpoints = None
if bp_text.strip():
    try:
        breakpoints = [float(x) for x in bp_text.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        breakpoints = None
lam = {"faible": 1e-5, "moyen": 5e-5, "fort": 2e-4}[smooth]

@st.cache_data(show_spinner=False, max_entries=32)
def cached_optimisation(gpx_key, rider_tuple, rho_key, wind_key, a_lat, width, block_m, breakpoints, lam):
    """Le plan optimal prend 10 a 30 s : on le met en cache sur l'ensemble des entrees."""
    r = Rider(*rider_tuple)
    return optimize_dynamic(seg, r, env, DynamicsParams(a_lat_max=a_lat, road_width_m=width),
                            block_m=float(block_m), breakpoints_km=breakpoints, smooth_lambda=lam)


with st.spinner("Optimisation du plan de puissance…"):
    rider_tuple = (rider.mass_rider, rider.mass_bike, rider.crr, rider.cda, rider.cp, rider.w_prime, rider.p_max, rider.p_min)
    opt = cached_optimisation(str(hash(seg.dist_m.tobytes()) ^ hash(seg.ele.tobytes())), rider_tuple,
                              float(np.round(np.mean(env.rho), 4)), float(np.round(np.mean(np.atleast_1d(env.wind_ms)), 3)),
                              a_lat, width, block_m, tuple(breakpoints) if breakpoints else None, lam)

# Prediction hybride du plan optimal et du plan constant
hyb = None
if model is not None:
    t_h, v_h = predict_course_fast(model, seg, opt.result.powers, profile, cp_w=cp)
    t_hc, _ = predict_course_fast(model, seg, opt.constant.powers, profile, cp_w=cp)
    hyb = {"time_s": float(t_h.sum()), "seg_time_s": t_h, "speed_ms": v_h, "const_time_s": float(t_hc.sum())}

real = None
if demo:
    from calibration import invert_sector_powers
    try:
        _, p_real = invert_sector_powers(seg, rider, env, DEMO["checkpoints"], params)
        real = simulate_dynamic(p_real, seg, rider, env, params)
    except Exception:
        real = None

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
err_pct = ho.get("mae_pct_hybrid")
cards = [
    kpi("Temps prédit, plan optimal", fmt(opt.result.time_s), f"physique dynamique · {opt.result.avg_speed_kmh:.1f} km/h"),
]
out_of_domain = bool(domain and (cp / mass_rider > domain["wkg_max"] or cda < domain["cda_min"]))
if hyb:
    p90 = ho.get("p90_pct_hybrid")
    rng = f"{fmt(hyb['time_s'] * (1 - p90 / 100))} – {fmt(hyb['time_s'] * (1 + p90 / 100))}" if p90 else ""
    cards.append(kpi("Temps prédit, modèle appris", fmt(hyb["time_s"]),
                     (f"intervalle 90 % {rng} · limite du domaine" if out_of_domain else f"intervalle 90 % {rng}") if rng else "modèle hybride",
                     "warn" if out_of_domain else "muted"))
cards.append(kpi("Gain vs puissance constante", f"{opt.gain_s:+.1f} s", f"constante {opt.p_constant:.0f} W → {fmt(opt.constant.time_s)}"))
if real is not None:
    cards.append(kpi("Stratégie réelle reconstruite", fmt(real.time_s), f"{real.time_s - opt.result.time_s:+.1f} s vs plan optimal", "warn"))
cards.append(kpi("Puissance moyenne du plan", f"{opt.result.avg_power:.0f} W", f"{opt.result.avg_power / mass_rider:.2f} W/kg · CP {cp:.0f} W", "muted"))
if s0 is not None:
    cards.append(kpi("Conditions", f"{s0.temp_c:.0f} °C · {s0.wind10_ms:.1f} m/s", f"ρ {np.mean(env.rho):.3f} · face {np.average(env.wind_ms, weights=seg.length_m):+.1f} m/s", "muted"))
st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tab_plan, tab_pred, tab_map, tab_val, tab_data = st.tabs(["Plan de puissance", "Prédiction", "Carte", "Validation", "Données et hypothèses"])
km_nodes = seg.dist_m / 1000.0; km_mid = seg.mid_dist_m / 1000.0
tab = plan_table(opt, seg)

with tab_plan:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km_nodes, y=seg.ele, fill="tozeroy", name="Altitude (m)", line=dict(color="#1E293B", width=1),
                             fillcolor="rgba(148,163,184,0.18)", yaxis="y2", hovertemplate="%{y:.0f} m"))
    fig.add_trace(go.Scatter(x=km_mid, y=opt.constant.powers, name=f"Constante {opt.p_constant:.0f} W", line=dict(color=SLATE, width=1.5, dash="dot")))
    if real is not None:
        fig.add_trace(go.Scatter(x=km_mid, y=real.powers, name="Réelle (estimée des pointages)", line=dict(color=AMBER, width=1.8)))
    fig.add_trace(go.Scatter(x=km_mid, y=opt.result.powers, name="Plan optimal (cible)", line=dict(color=TEAL, width=2.6, shape="hv")))
    fig.add_trace(go.Scatter(x=km_mid, y=opt.result.applied_power, name="Appliquée (freinages)", line=dict(color=TEAL, width=1, dash="dot"), opacity=0.7))
    fig.add_hline(y=cp, line=dict(color=RED, dash="dash", width=1), annotation_text=f"CP {cp:.0f} W", annotation_position="top left")
    fig.update_layout(**PLOT_LAYOUT, height=420, title="Puissance cible par bloc",
                      yaxis=dict(title="W", gridcolor="#1F2A44"), xaxis=dict(title="km", gridcolor="#1F2A44"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="m", range=[seg.ele.min() - 20, seg.ele.max() * 1.6]))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=km_mid, y=opt.constant.wbal_j / 1000, name="Constante", line=dict(color=SLATE, width=1.5, dash="dot")))
        if real is not None:
            fig2.add_trace(go.Scatter(x=km_mid, y=real.wbal_j / 1000, name="Réelle", line=dict(color=AMBER, width=1.8)))
        fig2.add_trace(go.Scatter(x=km_mid, y=opt.result.wbal_j / 1000, name="Plan optimal", line=dict(color=TEAL, width=2.4), fill="tozeroy", fillcolor="rgba(45,212,191,0.10)"))
        fig2.add_hline(y=0, line=dict(color=RED, dash="dash", width=1))
        fig2.update_layout(**PLOT_LAYOUT, height=300, title="Réserve anaérobie W'bal (kJ)", yaxis=dict(gridcolor="#1F2A44"), xaxis=dict(title="km", gridcolor="#1F2A44"))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown('<div class="sec">Temps par section</div>', unsafe_allow_html=True)
        edges = [0.0] + sorted(breakpoints or []) + [seg.total_km]
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            idx = seg.slice_km(a, b)
            if len(idx) == 0:
                continue
            row = {"Section": f"km {a:.1f} – {b:.1f}", "Pente": f"{np.average(seg.grade[idx], weights=seg.length_m[idx]) * 100:+.1f} %",
                   "P cible": f"{np.average(opt.result.powers[idx], weights=seg.length_m[idx]):.0f} W",
                   "Optimal": fmt(opt.result.seg_time_s[idx].sum()), "Constante": fmt(opt.constant.seg_time_s[idx].sum())}
            if real is not None:
                row["Réelle"] = fmt(real.seg_time_s[idx].sum())
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"{len(opt.block_powers)} blocs de {block_m} m · {opt.elapsed_s:.0f} s de calcul · min W'bal {opt.result.min_wbal_j / 1000:.2f} kJ")

    st.markdown('<div class="sec">Plan bloc par bloc</div>', unsafe_allow_html=True)
    show = tab.rename(columns={"km_start": "km début", "km_end": "km fin", "grade_pct": "pente %", "power_w": "P cible (W)",
                               "applied_w": "P appliquée (W)", "speed_kmh": "vitesse (km/h)", "time_s": "temps (s)",
                               "wbal_end_kj": "W'bal fin (kJ)", "braking": "freinage"}).round(1)
    st.dataframe(show, hide_index=True, use_container_width=True, height=300)
    e1, e2 = st.columns(2)
    e1.download_button("Exporter le plan (CSV)", tab.round(1).to_csv(index=False).encode(), "pacing_plan.csv", "text/csv", use_container_width=True)

    def to_zwo(tab, name="Pacing plan"):
        parts = [f'<workout_file><author>Pacing Intelligence</author><name>{name}</name><description>Plan de puissance par bloc</description><sportType>bike</sportType><workout>']
        parts += [f'<SteadyState Duration="{max(int(round(r.time_s)), 1)}" Power="{r.power_w:.0f}" />' for r in tab.itertuples()]
        parts.append("</workout></workout_file>")
        return "\n".join(parts).encode()
    e2.download_button("Exporter en séance (.zwo)", to_zwo(tab), "pacing_plan.zwo", "application/xml", use_container_width=True)

with tab_pred:
    if hyb is None:
        st.markdown('<div class="note">Modèle appris absent : lancez <code>python examples/train_hybrid.py</code>. Seule la physique dynamique est affichée.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=km_mid, y=opt.result.speed_ms * 3.6, name="Physique dynamique", line=dict(color=TEAL, width=2)))
        if hyb:
            fig3.add_trace(go.Scatter(x=km_mid, y=hyb["speed_ms"] * 3.6, name="Modèle appris", line=dict(color=BLUE, width=2)))
        if real is not None:
            fig3.add_trace(go.Scatter(x=km_mid, y=real.speed_ms * 3.6, name="Réelle reconstruite", line=dict(color=AMBER, width=1.5)))
        fig3.update_layout(**PLOT_LAYOUT, height=380, title="Vitesse prédite sur le plan optimal (km/h)", yaxis=dict(gridcolor="#1F2A44"), xaxis=dict(title="km", gridcolor="#1F2A44"))
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.markdown('<div class="sec">Deux modèles, un même plan</div>', unsafe_allow_html=True)
        rows = [{"Modèle": "Physique dynamique (Martin + inertie, virages, freinage)", "Plan optimal": fmt(opt.result.time_s), "Constante": fmt(opt.constant.time_s)}]
        if hyb:
            rows.append({"Modèle": "Hybride appris (physique + boosting, OpenData)", "Plan optimal": fmt(hyb["time_s"]), "Constante": fmt(hyb["const_time_s"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if hyb:
            st.markdown(f"""<div class="note">Le modèle appris corrige la physique de ce qu'elle ignore (variabilité de l'effort,
            inertie entre segments, routes réelles). Sur des athlètes jamais vus, son erreur sur 10 km est de
            <b>{ho.get('mae_pct_hybrid', float('nan')):.2f} %</b> contre {ho.get('mae_pct_physics', float('nan')):.2f} % pour la physique seule
            (p90 {ho.get('p90_pct_hybrid', float('nan')):.2f} %). Il ne connaît ni le vent du jour ni les virages : la physique dynamique reste la
            référence pour le plan, le modèle appris donne l'intervalle réaliste.</div>""", unsafe_allow_html=True)
        if hyb and out_of_domain:
            st.markdown(f"""<div class="note"><b>Coureur à la limite du domaine d'entraînement.</b> Le modèle appris a vu des athlètes jusqu'à
            {domain['wkg_max']:.1f} W/kg de CP et un CdA minimal de {domain['cda_min']:.2f} m² ({domain['n']} profils). À {cp / mass_rider:.1f} W/kg
            et {cda:.2f} m², sa prédiction extrapole : sur les pointages pro de l'étape 16 elle est biaisée de
            {tables['stage16_loo_hybrid'][tables['stage16_loo_hybrid'].model == 'hybride'].err_s.mean():+.0f} s par pointage
            (physique dynamique : {tables['stage16_loo_hybrid'][tables['stage16_loo_hybrid'].model == 'physique'].err_s.mean():+.0f} s).
            Le plan et le temps de référence restent ceux de la physique dynamique.</div>""" if "stage16_loo_hybrid" in tables else
            "<div class='note'>Coureur hors du domaine d'entraînement du modèle appris : prédiction indicative.</div>", unsafe_allow_html=True)
        elif hyb and real is not None:
            st.caption(f"Écart entre les deux modèles : {hyb['time_s'] - opt.result.time_s:+.1f} s.")

with tab_map:
    pmin, pmax = tab.power_w.min(), tab.power_w.max()

    def color(p):
        t = 0.0 if pmax == pmin else (p - pmin) / (pmax - pmin)
        return [int(96 + (248 - 96) * t), int(165 + (113 - 165) * t), int(250 + (113 - 250) * t), 220]
    paths = []
    for row in tab.itertuples():
        m = (km_nodes >= row.km_start - 1e-9) & (km_nodes <= row.km_end + 1e-9)
        pts = [[float(lo), float(la)] for lo, la in zip(seg.lon[m], seg.lat[m])]
        if len(pts) >= 2:
            paths.append({"path": pts, "color": color(row.power_w),
                          "label": f"km {row.km_start:.1f}-{row.km_end:.1f} · {row.power_w:.0f} W · {row.speed_kmh:.0f} km/h · {row.grade_pct:+.1f} %"})
    layers = [pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=5, pickable=True)]
    if wdf is not None and len(wdf):
        arrows = []
        for i in range(0, seg.n, max(seg.n // 25, 1)):
            wd = float(wdf.wind_dir_deg.iloc[i]); spd = float(wdf.wind_rider_ms.iloc[i])
            ang = np.radians(wd + 180.0); L = 70.0 * max(spd, 0.5)
            la0, lo0 = float(seg.lat[i]), float(seg.lon[i])
            arrows.append({"path": [[lo0, la0], [lo0 + L * np.sin(ang) / (111320.0 * np.cos(np.radians(la0))), la0 + L * np.cos(ang) / 111320.0]],
                           "label": f"vent {spd:.1f} m/s de {wd:.0f}°"})
        layers.append(pdk.Layer("PathLayer", data=arrows, get_path="path", get_color=[148, 163, 184, 200], width_min_pixels=2, pickable=True))
    view = pdk.ViewState(latitude=float(seg.lat.mean()), longitude=float(seg.lon.mean()), zoom=11.6, pitch=35)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{label}"},
                             map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"), use_container_width=True, height=560)
    st.caption("Bleu = puissance basse, rose = puissance haute. Flèches : vent à hauteur de coureur.")

with tab_val:
    st.markdown('<div class="sec">Modèle appris : erreurs mesurées sur des données jamais vues à l\'entraînement</div>', unsafe_allow_html=True)
    if metrics:
        rows = []
        for key, lab in (("held_out_rides", "Sorties inconnues (athlètes connus)"), ("held_out_athletes", "Athlètes inconnus")):
            m = metrics.get(key, {})
            rows.append({"Jeu de test": lab, "Segments": f"{m.get('n_segments', 0):,}".replace(",", " "), "Fenêtres 10 km": m.get("n_windows", 0),
                         "Erreur temps, physique seule": f"{m.get('mae_pct_physics', float('nan')):.2f} %",
                         "Erreur temps, modèle appris": f"{m.get('mae_pct_hybrid', float('nan')):.2f} %",
                         "p90": f"{m.get('p90_pct_hybrid', float('nan')):.2f} %", "Biais": f"{m.get('bias_pct_hybrid', float('nan')):+.2f} %"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Entraîné le {metrics.get('trained_at', '?')} sur {n_seg:,} segments de 200 m, {n_ath} athlètes GoldenCheetah OpenData (CC0).".replace(",", " "))
    else:
        st.info("Pas encore de métriques : entraînez le modèle.")

    st.markdown('<div class="sec">Tour de France 2026, étape 16 : prédiction d\'un pointage omis</div>', unsafe_allow_html=True)
    if "stage16_loo" in tables:
        loo = tables["stage16_loo"]
        c1, c2 = st.columns([2, 3])
        with c1:
            by = loo.groupby("km").agg(MAE_s=("err_s", lambda e: e.abs().mean()), biais_s=("err_s", "mean"), n=("err_s", "size")).reset_index().round(1)
            by.columns = ["Pointage (km)", "MAE (s)", "Biais (s)", "n"]
            st.dataframe(by, hide_index=True, use_container_width=True)
            st.markdown(f'<div class="kpis">{kpi("Erreur moyenne", f"{loo.err_s.abs().mean():.1f} s", f"{loo.err_pct_sector.abs().mean():.2f} % du temps de secteur · n = {len(loo)}")}</div>', unsafe_allow_html=True)
        with c2:
            fig4 = go.Figure()
            for r_name, g in loo.groupby("rider"):
                fig4.add_trace(go.Bar(x=g.km.astype(str) + " km", y=g.err_s, name=r_name))
            fig4.update_layout(**PLOT_LAYOUT, height=320, barmode="group", title="Erreur de prédiction par coureur et pointage (s)", yaxis=dict(gridcolor="#1F2A44"))
            st.plotly_chart(fig4, use_container_width=True)
    if "stage16_loo_hybrid" in tables:
        lh = tables["stage16_loo_hybrid"]
        st.markdown('<div class="sec">Étape 16 : physique dynamique contre modèle appris, même protocole</div>', unsafe_allow_html=True)
        comp = lh.groupby("model").agg(MAE_s=("err_s", lambda e: e.abs().mean()), MAE_pct=("err_pct_sector", lambda e: e.abs().mean()),
                                       biais_s=("err_s", "mean"), n=("err_s", "size")).round(2).reset_index()
        comp.columns = ["Modèle", "MAE (s)", "MAE (% secteur)", "Biais (s)", "n"]
        c1, c2 = st.columns([2, 3])
        c1.dataframe(comp, hide_index=True, use_container_width=True)
        c2.markdown(f"""<div class="note">Même protocole, mêmes coureurs : le modèle appris atteint
        {comp.iloc[0, 1] if comp.iloc[0, 0] == 'hybride' else comp.iloc[1, 1]:.1f} s d'erreur sans connaître le vent ni les virages du jour,
        contre {comp.iloc[1, 1] if comp.iloc[0, 0] == 'hybride' else comp.iloc[0, 1]:.1f} s pour la physique calibrée sur cette étape.
        Il a été entraîné sur des athlètes de 2 à 6 W/kg : les professionnels sont à la limite de son domaine, et son erreur a été
        divisée par presque trois en passant de 17 à 241 athlètes. Plus de données, en particulier des fichiers de puissance
        d'équipe, le rapprochent encore.</div>""", unsafe_allow_html=True)

    if "stage16_strategies" in tables:
        s16 = tables["stage16_strategies"]
        st.markdown('<div class="sec">Étape 16 : stratégie réelle, constante et optimale (top 10)</div>', unsafe_allow_html=True)
        d = s16[["rider", "mass_kg", "cp_w", "n_checkpoints", "real_s", "constant_s", "optimal_s", "gain_vs_real_s", "gain_vs_constant_s"]].copy()
        for c in ("real_s", "constant_s", "optimal_s"):
            d[c] = d[c].map(fmt)
        d.columns = ["Coureur", "Masse", "CP calée (W)", "Pointages", "Réelle", "Constante", "Optimale", "Gain vs réelle (s)", "Gain vs constante (s)"]
        st.dataframe(d.round(1), hide_index=True, use_container_width=True)
        st.caption("Les watts des pros sont déduits des temps sous hypothèses (CdA, Crr, W'), jamais mesurés.")

with tab_data:
    st.markdown('<div class="sec">Hypothèses de cette analyse</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Paramètre": "CdA", "Valeur": f"{cda:.3f} m²", "Statut": "hypothèse (saisie)"},
        {"Paramètre": "Crr", "Valeur": f"{crr:.4f}", "Statut": "hypothèse (saisie)"},
        {"Paramètre": "Masse coureur + vélo", "Valeur": f"{mass_rider:.1f} + {mass_bike:.1f} kg", "Statut": "saisie"},
        {"Paramètre": "CP, W'", "Valeur": f"{cp:.0f} W, {w_prime / 1000:.1f} kJ", "Statut": "saisie ou calée sur les pointages"},
        {"Paramètre": "Limite de virage, largeur", "Valeur": f"{a_lat:.1f} m/s², {width:.0f} m", "Statut": "calée sur l'étape 16 (8 m/s²)"},
        {"Paramètre": "Météo", "Valeur": "Open-Meteo horaire, vent ramené à 1 m (z0 = 0,10 m)" if wmode.startswith("Open") else "manuelle", "Statut": "mesurée (réanalyse)"},
        {"Paramètre": "Parcours", "Valeur": f"{seg.total_km:.2f} km, {seg.elevation_gain_m:.0f} m D+, {len(seg.corners or [])} virages", "Statut": "GPX"},
    ]), hide_index=True, use_container_width=True)
    if "athletes" in tables:
        a = tables["athletes"]
        st.markdown('<div class="sec">Jeu d\'entraînement</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpis">{kpi("Athlètes traités", f"{len(a)}", "GoldenCheetah OpenData, CC0")}{kpi("Sorties retenues", f"{int(a.n_rides_kept.sum()):,}".replace(",", " "), f"sur {int(a.n_rides.sum()):,} avec capteur".replace(",", " "))}{kpi("Segments de 200 m", f"{int(a.n_segments.sum()):,}".replace(",", " "), f"{a.n_segments.sum() * 0.2 / 1000:,.0f} 000 km roulés".replace(",", " "))}</div>', unsafe_allow_html=True)
    st.markdown('<div class="src">Sources : cyclingstage.com (GPX, pointages), bikeraceinfo.com, bikeradar.com (départs), Wikipedia (masses, origine citée), Open-Meteo, OSF 10.17605/OSF.IO/6HFPZ. Détail dans data/processed/README.md.</div>', unsafe_allow_html=True)
