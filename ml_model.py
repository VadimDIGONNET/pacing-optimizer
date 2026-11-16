"""Modele predictif hybride physique + apprentissage (entraine sur OpenData).

Idee : le modele de Martin (1998) explique l'essentiel de la vitesse a partir
de la puissance, de la pente, de l'altitude et de la masse. Ce qu'il ignore
(variabilite de la puissance dans le segment, roue libre, inertie entre
segments, vent moyen, position, qualite de la route, imprecision de la
pente barometrique) laisse un residu systematique. On l'apprend sur des
millions de segments reels :

  1. Par athlete, CdA et Crr sont ajustes par moindres carres sur ses
     segments (physique seule) : c'est le "profil" de l'athlete, et c'est
     ce qu'une equipe mesure en soufflerie ou en test de roulement.
  2. Un gradient boosting (HistGradientBoosting, scikit-learn) apprend
        log(v_reel) - log(v_physique)
     a partir des features du segment et du profil : l'athlete n'est jamais
     une variable (le modele doit generaliser a un coureur jamais vu).
  3. Prediction : v = v_physique * exp(residu predit). Le temps d'un
     parcours est la somme des temps de segment.

Validation :
  - par sorties : 20 % des sorties de chaque athlete mises de cote ;
  - par athletes : 20 % des athletes jamais vus a l'entrainement
    (c'est le cas d'usage : un coureur pro inconnu du jeu).
  Erreurs rapportees : MAE sur log(v) (= erreur relative de temps) par
  segment, et erreur de temps sur des fenetres de 10 km.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pacing_model import G, DRIVETRAIN_EFF

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
# Seulement des variables connues au moment de planifier (pas de FC ni de
# cadence) : un plan de puissance, un profil physique, un parcours.
FEATURES = ["power_w", "grade", "alt_m", "v_in_ms", "power_sd", "pedal_frac",
            "mass_kg", "cda", "crr", "wkg", "v_phys", "p_over_cp", "elapsed_s"]
RHO0 = 1.225


def rho_isa(alt_m):
    """Densite ISA en fonction de l'altitude (pas de meteo dans OpenData)."""
    return RHO0 * (1.0 - 2.25577e-5 * np.asarray(alt_m, float)) ** 4.2559


def physics_speed(power_w, grade, alt_m, mass_kg, cda, crr, bike_kg=9.0):
    """Vitesse d'equilibre de Martin (vectorisee, Newton sur le cube)."""
    power = np.asarray(power_w, float); grade = np.asarray(grade, float)
    m = np.asarray(mass_kg, float) + bike_kg
    theta = np.arctan(grade)
    f_rg = m * G * (np.sin(theta) + np.asarray(crr, float) * np.cos(theta))
    k = 0.5 * rho_isa(alt_m) * np.asarray(cda, float)
    p = np.maximum(power, 1.0) * DRIVETRAIN_EFF
    v = np.full_like(p, 8.0)
    for _ in range(30):                                   # Newton : k v^3 + f v - p = 0
        f = k * v ** 3 + f_rg * v - p
        df = 3 * k * v ** 2 + f_rg
        v = np.clip(v - f / np.maximum(df, 1e-6), 0.3, 30.0)
    return v


# ---------------------------------------------------------------------------
# 1. Profil physique par athlete
# ---------------------------------------------------------------------------

def fit_athlete_profile(seg: pd.DataFrame, mass_fallback: float = 75.0) -> dict:
    """CdA et Crr par moindres carres sur log(v) pour les segments ou le
    coureur pedale (puissance > 50 W, pedalage > 80 %), pente < 8 %.
    Bornes physiques : CdA 0,15 a 0,60, Crr 0,002 a 0,012."""
    d = seg[(seg.power_w > 50) & (seg.pedal_frac > 0.8) & (seg.grade.abs() < 0.08) & (seg.speed_ms > 3)]
    mass = float(seg.mass_kg.dropna().median()) if seg.mass_kg.notna().any() else mass_fallback
    if len(d) < 200:
        return {"cda": 0.32, "crr": 0.005, "mass_kg": mass, "n_fit": len(d), "fit_rmse": np.nan}
    d = d.sample(min(len(d), 20000), random_state=0)
    y = np.log(d.speed_ms.to_numpy())

    def resid(x):
        return np.log(physics_speed(d.power_w, d.grade, d.alt_m, mass, x[0], x[1])) - y

    r = least_squares(resid, x0=[0.32, 0.005], bounds=([0.15, 0.002], [0.60, 0.012]), loss="soft_l1", f_scale=0.1)
    return {"cda": float(r.x[0]), "crr": float(r.x[1]), "mass_kg": mass, "n_fit": int(len(d)),
            "fit_rmse": float(np.sqrt(np.mean(r.fun ** 2)))}


def add_features(seg: pd.DataFrame, profile: dict, cp_w: float | None = None) -> pd.DataFrame:
    d = seg.copy()
    d["mass_kg"] = profile["mass_kg"]; d["cda"] = profile["cda"]; d["crr"] = profile["crr"]
    d["wkg"] = d.power_w / d.mass_kg
    d["v_phys"] = physics_speed(d.power_w, d.grade, d.alt_m, d.mass_kg, d.cda, d.crr)
    cp = cp_w if cp_w else np.nan
    d["p_over_cp"] = d.power_w / cp if cp_w else np.nan
    d["cad"] = d.cad.replace(0, np.nan); d["hr"] = d.hr.replace(0, np.nan)
    return d


def athlete_cp(seg: pd.DataFrame) -> float:
    """CP approchee : 95 % de la meilleure puissance moyenne sur ~20 min
    (fenetre de 100 segments consecutifs d'une meme sortie)."""
    best = 0.0
    for _, r in seg.groupby("ride"):
        if len(r) < 100:
            continue
        p = r.sort_values("k").power_w.to_numpy(); t = r.sort_values("k").time_s.to_numpy()
        w = np.convolve(p * t, np.ones(100), "valid") / np.convolve(t, np.ones(100), "valid")
        best = max(best, float(w.max()))
    return 0.95 * best if best > 0 else np.nan


# ---------------------------------------------------------------------------
# 2. Modele de residu
# ---------------------------------------------------------------------------

@dataclass
class HybridModel:
    booster: object = None
    features: list = field(default_factory=lambda: list(FEATURES))
    profiles: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def predict_speed(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(float)
        resid = self.booster.predict(X) if self.booster is not None else 0.0
        return df.v_phys.to_numpy() * np.exp(resid)

    def save(self, path: Path = MODEL_DIR / "hybrid.joblib"):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path = MODEL_DIR / "hybrid.joblib"):
        import joblib
        return joblib.load(path)


def prepare(segments: pd.DataFrame, profiles: dict | None = None):
    """Ajoute profil + features a tout le jeu ; renvoie (df, profiles)."""
    profiles = dict(profiles or {})
    out = []
    for a, s in segments.groupby("athlete"):
        if a not in profiles:
            profiles[a] = fit_athlete_profile(s)
            profiles[a]["cp_w"] = athlete_cp(s)
        out.append(add_features(s, profiles[a], profiles[a].get("cp_w")))
    return pd.concat(out, ignore_index=True), profiles


def train(df: pd.DataFrame, max_iter: int = 400, learning_rate: float = 0.08, seed: int = 0) -> HybridModel:
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.log(df.speed_ms.to_numpy()) - np.log(df.v_phys.to_numpy())
    X = df[FEATURES].to_numpy(float)
    b = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=63,
                                      min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    b.fit(X, y)
    return HybridModel(booster=b)


def split_rides(df: pd.DataFrame, frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    rides = df[["athlete", "ride"]].drop_duplicates()
    test = set(map(tuple, rides.sample(frac=frac, random_state=seed).to_numpy()))
    m = np.array([(a, r) in test for a, r in zip(df.athlete, df.ride)])
    return df[~m], df[m]


def evaluate(model: HybridModel, df: pd.DataFrame, window_km: float = 10.0) -> dict:
    """Erreurs par segment et par fenetre de `window_km` dans chaque sortie."""
    v_hat = model.predict_speed(df)
    logerr = np.log(v_hat) - np.log(df.speed_ms.to_numpy())
    phys = np.log(df.v_phys.to_numpy()) - np.log(df.speed_ms.to_numpy())
    out = {"n_segments": int(len(df)), "mae_log_hybrid": float(np.abs(logerr).mean()),
           "mae_log_physics": float(np.abs(phys).mean()), "bias_log_hybrid": float(logerr.mean()),
           "bias_log_physics": float(phys.mean())}
    d = df[["athlete", "ride", "k", "length_m", "time_s"]].copy()
    d["t_hat"] = d.length_m / v_hat; d["t_phys"] = d.length_m / df.v_phys.to_numpy()
    n_win = int(window_km * 1000 / 200)
    errs, errs_p = [], []
    for _, r in d.groupby(["athlete", "ride"]):
        r = r.sort_values("k")
        if len(r) < n_win:
            continue
        for i in range(0, len(r) - n_win + 1, n_win):
            w = r.iloc[i:i + n_win]
            if not np.all(np.diff(w.k) == 1):
                continue
            errs.append((w.t_hat.sum() - w.time_s.sum()) / w.time_s.sum())
            errs_p.append((w.t_phys.sum() - w.time_s.sum()) / w.time_s.sum())
    if errs:
        e, ep = np.array(errs), np.array(errs_p)
        out.update({"n_windows": len(e), "window_km": window_km,
                    "mae_pct_hybrid": float(np.abs(e).mean() * 100), "mae_pct_physics": float(np.abs(ep).mean() * 100),
                    "p90_pct_hybrid": float(np.percentile(np.abs(e), 90) * 100),
                    "bias_pct_hybrid": float(e.mean() * 100), "bias_pct_physics": float(ep.mean() * 100)})
    return out


# ---------------------------------------------------------------------------
# 3. Prediction sur un parcours (chainage de la vitesse d'entree)
# ---------------------------------------------------------------------------

def predict_course(model: HybridModel, seg, powers, profile: dict, cp_w: float | None = None,
                   power_sd_w: float = 15.0, v0: float = 1.0):
    """Temps par segment d'un parcours (SegmentedCourse ou Course) pour un plan
    de puissance, avec le modele hybride. La vitesse d'entree de chaque
    segment est la vitesse moyenne predite du precedent (chainage), la
    puissance est supposee tenue (ecart-type `power_sd_w`, pedalage complet
    sauf roue libre quand P = 0), le temps ecoule s'accumule.

    Renvoie (time_s par segment, speed_ms par segment, DataFrame des features).
    """
    powers = np.asarray(powers, float)
    n = len(powers)
    length = np.asarray(seg.length_m, float); grade = np.asarray(seg.grade, float)
    alt = np.asarray(getattr(seg, "mid_ele", np.zeros(n)), float)
    mass, cda, crr = profile["mass_kg"], profile["cda"], profile["crr"]
    v_phys = physics_speed(powers, grade, alt, mass, cda, crr)
    v_in, elapsed = v0, 0.0
    times = np.zeros(n); speeds = np.zeros(n); rows = []
    for i in range(n):
        row = {"power_w": powers[i], "grade": grade[i], "alt_m": alt[i], "v_in_ms": v_in,
               "power_sd": power_sd_w if powers[i] > 0 else 0.0, "pedal_frac": 1.0 if powers[i] > 0 else 0.0,
               "mass_kg": mass, "cda": cda, "crr": crr, "wkg": powers[i] / mass, "v_phys": v_phys[i],
               "p_over_cp": powers[i] / cp_w if cp_w else np.nan, "elapsed_s": elapsed}
        X = np.array([[row[f] for f in model.features]], float)
        v = float(v_phys[i] * np.exp(model.booster.predict(X)[0])) if model.booster is not None else float(v_phys[i])
        v = max(v, 0.5)
        times[i] = length[i] / v; speeds[i] = v
        elapsed += times[i]; v_in = v
        rows.append(row)
    return times, speeds, pd.DataFrame(rows)


def predict_course_fast(model: HybridModel, seg, powers, profile: dict, cp_w: float | None = None,
                        power_sd_w: float = 15.0, n_pass: int = 3):
    """Variante vectorisee : v_in et elapsed sont pris de la passe precedente
    (point fixe en `n_pass` passes, ~1 ms par passe sur 500 segments)."""
    powers = np.asarray(powers, float); n = len(powers)
    length = np.asarray(seg.length_m, float); grade = np.asarray(seg.grade, float)
    alt = np.asarray(getattr(seg, "mid_ele", np.zeros(n)), float)
    mass, cda, crr = profile["mass_kg"], profile["cda"], profile["crr"]
    v_phys = physics_speed(powers, grade, alt, mass, cda, crr)
    v = v_phys.copy()
    for _ in range(n_pass):
        v_in = np.concatenate([[1.0], v[:-1]])
        t = length / v
        elapsed = np.concatenate([[0.0], np.cumsum(t)[:-1]])
        X = np.column_stack([powers, grade, alt, v_in, np.where(powers > 0, power_sd_w, 0.0),
                             (powers > 0).astype(float), np.full(n, mass), np.full(n, cda), np.full(n, crr),
                             powers / mass, v_phys, powers / cp_w if cp_w else np.full(n, np.nan), elapsed])
        v = np.maximum(v_phys * np.exp(model.booster.predict(X)), 0.5)
    return length / v, v
