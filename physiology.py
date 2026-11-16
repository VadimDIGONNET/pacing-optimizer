"""Physiologie (phase 5) : courbes puissance-duree reelles, CP et W', test du W'bal.

Donnees : GoldenCheetah OpenData (OSF, DOI 10.17605/OSF.IO/6HFPZ), licence
CC0 1.0 Universal (verifiee par l'API OSF le 2026-11-16). Un zip par
athlete : un JSON (athlete, liste des sorties et metriques, dont le poids
declare) et un CSV par sorties (secs, km, power, hr, cad, alt). Ces athletes
sont des amateurs et des cyclistes entraines, pas des professionnels : ils
servent a verifier le modele et a borner CP/kg et W', pas a representer le
peloton.

Methode :
  1. Puissance maximale moyenne (MMP) par duree, sur toutes les sorties avec
     capteur de puissance, par sommes cumulees (exact, pas d'approximation).
  2. CP et W' : modele a deux parametres de Monod & Scherrer, travail = CP t +
     W', ajuste par moindres carres sur les durees de 3 a 20 min (plage
     recommandee ; en dessous la composante anaerobie alactique fausse W',
     au-dessus la derive de fatigue fausse CP).
  3. Test du W'bal de Skiba : sur les sorties qui contiennent une MMP de
     saison entre 3 et 20 min (efforts presque maximaux), on calcule le W'bal
     differentiel avec les CP/W' ajustes. Un modele coherent donne un minimum
     de W'bal proche de 0 sur ces sorties ; on reporte la distribution
     min(W'bal)/W'.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DURATIONS_S = [5, 10, 20, 30, 60, 120, 180, 240, 300, 360, 480, 600, 720, 900, 1200, 1500, 1800, 2400, 3600]
FIT_RANGE_S = (180, 1200)


def mmp_curve(power: np.ndarray, durations=DURATIONS_S) -> dict:
    """Puissance maximale moyenne pour chaque duree (echantillonnage 1 Hz)."""
    p = np.nan_to_num(np.asarray(power, float), nan=0.0)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    out = {}
    for d in durations:
        if len(p) >= d:
            out[d] = float((cs[d:] - cs[:-d]).max() / d)
    return out


def fit_cp_wprime(mmp: dict, fit_range=FIT_RANGE_S):
    """Regression lineaire travail = CP t + W' sur les durees de la plage."""
    pts = [(d, p * d) for d, p in mmp.items() if fit_range[0] <= d <= fit_range[1]]
    if len(pts) < 3:
        return np.nan, np.nan, np.nan
    t = np.array([a for a, _ in pts]); w = np.array([b for _, b in pts])
    cp, wp = np.polyfit(t, w, 1)
    resid = w - (cp * t + wp)
    r2 = 1 - resid.var() / w.var()
    return float(cp), float(wp), float(r2)


def wbal_skiba(power: np.ndarray, cp: float, w_prime: float) -> np.ndarray:
    """W'bal differentiel (Skiba 2015) a 1 Hz : depletion lineaire au-dessus de CP,
    recuperation exponentielle en dessous, constante de temps W'/(CP - P)."""
    p = np.nan_to_num(np.asarray(power, float), nan=0.0)
    wb = np.empty(len(p)); w = w_prime
    for i, pi in enumerate(p):
        if pi > cp:
            w -= (pi - cp)
        else:
            w = w_prime - (w_prime - w) * np.exp(-(cp - pi) / w_prime)
        wb[i] = w
    return wb


def _resample_1hz(df: pd.DataFrame) -> np.ndarray:
    secs = df["secs"].to_numpy(float); power = df["power"].to_numpy(float)
    if len(secs) < 2:
        return power
    grid = np.arange(0, secs[-1] + 1)
    # trous > 10 s : puissance nulle (arret), pas interpolee
    p = np.interp(grid, secs, power)
    gaps = np.diff(secs); starts = secs[:-1][gaps > 10]; ends = secs[1:][gaps > 10]
    for a, b in zip(starts, ends):
        p[(grid > a) & (grid < b)] = 0.0
    return p


def analyse_athlete(zip_path, min_ride_s: int = 1200, max_rides: int | None = None) -> dict:
    """Courbe MMP, CP/W' et test W'bal pour un athlete. Renvoie un dict."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        meta = json.loads(zf.read([n for n in names if n.endswith(".json")][0]))
        rides = meta.get("RIDES", [])
        weights = [float(r["METRICS"].get("athlete_weight", "nan")) for r in rides if "METRICS" in r]
        weights = [w for w in weights if 30 < w < 150]
        csvs = sorted(n for n in names if n.endswith(".csv"))
        if max_rides:
            csvs = csvs[:max_rides]
        best = {d: (0.0, None) for d in DURATIONS_S}
        rides_power = {}
        n_power = 0
        for name in csvs:
            try:
                df = pd.read_csv(io.BytesIO(zf.read(name)), usecols=["secs", "power"])
            except Exception:
                continue
            if len(df) < min_ride_s or df["power"].fillna(0).max() <= 0:
                continue
            p = _resample_1hz(df)
            n_power += 1
            m = mmp_curve(p)
            for d, v in m.items():
                if v > best[d][0]:
                    best[d] = (v, name)
            rides_power[name] = p
    mmp = {d: v for d, (v, _) in best.items() if v > 0}
    cp, wp, r2 = fit_cp_wprime(mmp)
    # test W'bal sur les sorties contenant une MMP 3-20 min
    maximal = {best[d][1] for d in DURATIONS_S if FIT_RANGE_S[0] <= d <= FIT_RANGE_S[1] and best[d][1]}
    wb_min = []
    if not np.isnan(cp) and wp > 0:
        for name in maximal:
            wb = wbal_skiba(rides_power[name], cp, wp)
            wb_min.append(wb.min() / wp)
    mass = float(np.median(weights)) if weights else np.nan
    return {"athlete": zip_path.stem, "gender": meta["ATHLETE"].get("gender"),
            "yob": meta["ATHLETE"].get("yob"), "n_rides_power": n_power, "mass_kg": mass,
            "cp_w": cp, "w_prime_j": wp, "fit_r2": r2,
            "cp_wkg": cp / mass if mass else np.nan,
            "mmp": mmp, "wbal_min_frac": wb_min, "n_maximal_rides": len(maximal)}
