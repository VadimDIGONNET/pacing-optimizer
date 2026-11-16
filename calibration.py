"""Calage et validation sur les pointages reels (phase 4).

Entrees (data/processed/, toutes sourcees) :
  - tdf2026_stage16_splits.csv : pointages cumules par coureur, heure de depart
  - tdf2026_stage16_riders.csv : masse publiee (qualite notee)
  - meteo Open-Meteo (weather.py) a l'heure de passage de chaque coureur

Hypotheses (jamais presentees comme des mesures) :
  - CdA : valeur de reference CDA_REF pour le coureur de reference, mise a
    l'echelle par la surface corporelle de DuBois (m^0,425 h^0,725) quand la
    taille est connue, sinon par m^(2/3). Une position de CLM optimisee a un
    CdA de 0,19 a 0,23 m2 chez les pros ; 0,21 est le milieu de la plage.
  - Crr = 0,0032 (boyaux de CLM sur asphalte sec, Crr de 0,0025 a 0,0040).
  - Masse du velo de CLM = 8,5 kg (minimum UCI 6,8 kg + roues lenticulaires
    et prolongateurs ; mesures publiques 8 a 9,5 kg).
  - W' = 25 kJ ; sert au calage de CP (effort maximal : min W'bal = 0).
  - Virages : a_lat et largeur de route de dynamics.DynamicsParams.

Methode :
  1. `invert_sector_powers` : pour chaque coureur, puissance constante par
     secteur entre deux pointages consecutifs, trouvee par dichotomie avec le
     simulateur dynamique (l'etat de vitesse se propage d'un secteur au
     suivant). Les secteurs d'un coureur sont definis par SES pointages
     disponibles.
  2. `calibrate_cp` : CP telle que l'effort reel vide exactement W'.
  3. `leave_one_out` : on omet un pointage, on recale les puissances sur les
     autres (hypothese : puissance constante sur le secteur fusionne), et on
     predit le pointage omis. L'erreur melange donc la physique et
     l'hypothese de pacing constant : c'est la seule validation possible
     sans capteur de puissance, et elle est bornee par la variabilite du
     pacing reel (voir README).
  4. `descent_tradeoff` : sur le secteur de descente, puissance impliquee en
     fonction de a_lat, car les deux ne sont pas identifiables separement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pacing_model import Rider, Environment
from dynamics import simulate_dynamic, DynamicsParams
from weather import rider_weather, segment_environment

ROOT = Path(__file__).resolve().parent
CDA_REF = 0.21            # m2, coureur de reference (Evenepoel, 63 kg, 1,71 m)
CRR = 0.0032
MASS_BIKE = 8.5           # kg
W_PRIME = 25000.0         # J
P_MAX = 900.0

# Parametres de virage cales sur l'etape 16 : la validation par pointage omis
# est minimale pour a_lat = 8 m/s2 (0,82 g, limite d'adherence sur sec), voir
# examples/phase4_calibration.py. A utiliser pour toutes les analyses de
# l'etape 16 ; la valeur par defaut de DynamicsParams (6 m/s2) reste plus
# prudente pour un parcours inconnu.
STAGE16_PARAMS = DynamicsParams(a_lat_max=8.0, road_width_m=6.0)

# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

def load_splits(path=ROOT / "data/processed/tdf2026_stage16_splits.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["rider", "checkpoint_km"]).reset_index(drop=True)


def load_riders(path=ROOT / "data/processed/tdf2026_stage16_riders.csv") -> pd.DataFrame:
    return pd.read_csv(path).set_index("rider")


def body_surface_area(mass_kg: float, height_m: float) -> float:
    """DuBois & DuBois (1916), m2."""
    return 0.007184 * mass_kg ** 0.425 * (height_m * 100) ** 0.725


def scaled_cda(mass_kg, height_m, ref_mass=63.0, ref_height=1.71, cda_ref=CDA_REF) -> float:
    """CdA par profil morphologique (hypothese documentee ci-dessus)."""
    if height_m is not None and not np.isnan(height_m):
        return cda_ref * body_surface_area(mass_kg, height_m) / body_surface_area(ref_mass, ref_height)
    return cda_ref * (mass_kg / ref_mass) ** (2.0 / 3.0)


def make_rider(name: str, riders: pd.DataFrame, cda_override: float | None = None) -> Rider | None:
    if name not in riders.index or np.isnan(riders.loc[name, "mass_kg"]):
        return None
    m = float(riders.loc[name, "mass_kg"])
    h = riders.loc[name, "height_m"] if "height_m" in riders.columns else np.nan
    h = float(h) if not pd.isna(h) else None
    cda = cda_override if cda_override is not None else scaled_cda(m, h)
    return Rider(mass_rider=m, mass_bike=MASS_BIKE, crr=CRR, cda=cda, cp=400.0,
                 w_prime=W_PRIME, p_max=P_MAX, p_min=0.0)


# ---------------------------------------------------------------------------
# Inversion des pointages
# ---------------------------------------------------------------------------

@dataclass
class SectorFit:
    rider: str
    km_start: float
    km_end: float
    time_s: float             # temps de secteur reel
    power_w: float            # puissance constante impliquee
    speed_kmh: float
    wkg: float
    segments: np.ndarray      # indices de segments


def _sector_indices(seg, km_start, km_end):
    idx = seg.slice_km(km_start, km_end)
    if len(idx) == 0:
        raise ValueError(f"secteur vide {km_start}-{km_end}")
    return idx


def invert_sector_powers(seg, rider: Rider, env: Environment, checkpoints,
                         params: DynamicsParams | None = None, p_lo=0.0, p_hi=1000.0):
    """checkpoints : liste de (km, temps cumule s), croissante. Renvoie la
    liste des SectorFit et le vecteur de puissance par segment.

    Chaque secteur est resolu dans l'ordre, les precedents etant figes, pour
    que la vitesse d'entree dans le secteur soit celle de la simulation.
    """
    params = params or DynamicsParams()
    powers = np.zeros(seg.n)
    fits, prev_km, prev_t = [], 0.0, 0.0
    for km, t_cum in checkpoints:
        idx = _sector_indices(seg, prev_km, km)
        last = idx[-1]

        def cum_time(p):
            powers[idx] = p
            r = simulate_dynamic(powers, seg, rider, env, params)
            return r.seg_time_s[: last + 1].sum() - t_cum

        if cum_time(p_hi) > 0:
            p = p_hi
        elif cum_time(p_lo) < 0:
            p = p_lo
        else:
            p = brentq(cum_time, p_lo, p_hi, xtol=0.5)
        powers[idx] = p
        dt = t_cum - prev_t
        length = seg.length_m[idx].sum()
        fits.append(SectorFit(rider="", km_start=prev_km, km_end=km, time_s=dt, power_w=p,
                              speed_kmh=length / dt * 3.6, wkg=p / rider.mass_rider, segments=idx))
        prev_km, prev_t = km, t_cum
    return fits, powers


def calibrate_cp(powers, seg, rider: Rider, env: Environment, params=None,
                 lo=200.0, hi=650.0) -> float:
    """CP telle que min W'bal = 0 sur l'effort reel (effort maximal suppose)."""
    params = params or DynamicsParams()

    def f(cp):
        return simulate_dynamic(powers, seg, replace(rider, cp=cp), env, params).min_wbal_j

    if f(hi) < 0:
        return hi
    if f(lo) > 0:
        return lo
    return brentq(f, lo, hi, xtol=0.2)


# ---------------------------------------------------------------------------
# Validation croisee par pointage omis
# ---------------------------------------------------------------------------

def leave_one_out(seg, rider: Rider, env: Environment, checkpoints, params=None):
    """Pour chaque pointage intermediaire omis : recalage sur les autres puis
    prediction du temps cumule au pointage omis. Renvoie un DataFrame
    (km, reel, predit, erreur s, erreur % du temps de secteur)."""
    params = params or DynamicsParams()
    rows = []
    if len(checkpoints) < 2:
        return pd.DataFrame(rows)
    for k in range(len(checkpoints) - 1):      # on ne peut pas omettre l'arrivee
        kept = [c for j, c in enumerate(checkpoints) if j != k]
        _, powers = invert_sector_powers(seg, rider, env, kept, params)
        r = simulate_dynamic(powers, seg, rider, env, params)
        km, t_real = checkpoints[k]
        idx = seg.slice_km(0.0, km)
        t_pred = r.seg_time_s[: idx[-1] + 1].sum()
        prev_t = checkpoints[k - 1][1] if k > 0 else 0.0
        next_t = checkpoints[k + 1][1]
        rows.append({"km": km, "t_real_s": t_real, "t_pred_s": t_pred,
                     "err_s": t_pred - t_real,
                     "err_pct_sector": (t_pred - t_real) / (next_t - prev_t) * 100.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline etape 16
# ---------------------------------------------------------------------------

def rider_checkpoints(splits: pd.DataFrame, name: str):
    d = splits[splits.rider == name]
    return [(float(km), float(t)) for km, t in zip(d.checkpoint_km, d.time_cum_s)]


def rider_env(seg, splits, name, date="2026-07-21", cache_dir=ROOT / "data/raw/weather"):
    d = splits[splits.rider == name].iloc[0]
    finish = splits[(splits.rider == name)].time_cum_s.max()
    samples = rider_weather(seg, date, d.start_time_local, elapsed_s=finish / 2.0,
                            n_points=3, cache_dir=cache_dir)
    env, df = segment_environment(seg, samples)
    return env, df


def run_stage16(seg, params: DynamicsParams | None = None, cda_override=None,
                verbose=True):
    params = params or DynamicsParams()
    splits, riders = load_splits(), load_riders()
    sector_rows, loo_rows, rider_rows = [], [], []
    for name in splits.rider.unique():
        rider = make_rider(name, riders, cda_override)
        if rider is None:
            if verbose:
                print(f"  {name:<22}: masse inconnue, ignore")
            continue
        cps = rider_checkpoints(splits, name)
        env, wdf = rider_env(seg, splits, name)
        fits, powers = invert_sector_powers(seg, rider, env, cps, params)
        cp = calibrate_cp(powers, seg, rider, env, params) if len(cps) >= 2 else np.nan
        rider.cp = cp if not np.isnan(cp) else rider.cp
        for f in fits:
            f.rider = name
            sector_rows.append({"rider": name, "km_start": f.km_start, "km_end": f.km_end,
                                "time_s": f.time_s, "power_w": f.power_w, "wkg": f.wkg,
                                "speed_kmh": f.speed_kmh})
        loo = leave_one_out(seg, rider, env, cps, params)
        loo["rider"] = name
        loo_rows.append(loo)
        rider_rows.append({"rider": name, "mass_kg": rider.mass_rider, "cda": rider.cda,
                           "n_checkpoints": len(cps), "cp_w": cp,
                           "cp_wkg": cp / rider.mass_rider if not np.isnan(cp) else np.nan,
                           "avg_power_w": np.average(powers, weights=seg.length_m),
                           "headwind_mean_ms": float(np.average(wdf.headwind_ms, weights=seg.length_m)),
                           "rho_mean": float(wdf.rho.mean())})
    sectors = pd.DataFrame(sector_rows)
    loo = pd.concat([d for d in loo_rows if len(d)], ignore_index=True) if any(len(d) for d in loo_rows) else pd.DataFrame()
    return sectors, loo, pd.DataFrame(rider_rows)


def descent_tradeoff(seg, rider: Rider, env: Environment, checkpoints, params: DynamicsParams,
                     a_lat_values=(4.0, 5.0, 6.0, 8.0, 10.0, 99.0)):
    """Puissance impliquee sur chaque secteur en fonction de a_lat."""
    rows = []
    for a in a_lat_values:
        fits, _ = invert_sector_powers(seg, rider, env, checkpoints, replace(params, a_lat_max=a))
        rows.append({"a_lat": a, **{f"P {f.km_start:.1f}-{f.km_end:.1f}": f.power_w for f in fits}})
    return pd.DataFrame(rows)
