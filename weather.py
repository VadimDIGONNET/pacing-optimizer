"""Meteo reelle par segment (phase 2) : Open-Meteo, densite de l'air, vent.

Sources et conventions :
  - Archive horaire Open-Meteo (reanalyse ERA5 / modeles nationaux, grille
    1 a 11 km) : temperature a 2 m, humidite relative, pression de surface,
    vent a 10 m (vitesse, direction meteorologique = d'ou vient le vent).
    Usage non commercial, cache JSON dans data/raw/weather/, jamais modifie.
  - Densite de l'air : air humide = melange air sec + vapeur d'eau,
    rho = (p - p_v) / (R_d T) + p_v / (R_v T), avec p_v = HR * e_s(T) et
    e_s selon la formule de Magnus (Alduchov & Eskridge 1996). Ecart avec la
    formule CIPM-2007 inferieur a 0,1 % dans les conditions d'une course.
  - Correction d'altitude : la pression de surface est celle de la maille du
    modele (altitude renvoyee par l'API). On la ramene a l'altitude de chaque
    segment par la formule hypsometrique, avec un gradient thermique de
    6,5 K/km pour la temperature.
  - Vent a hauteur de cycliste : profil logarithmique
    u(z) = u10 * ln(z / z0) / ln(10 / z0). Avec z = 1 m (hauteur moyenne du
    couple coureur + velo, ou s'exerce la trainee) et z0 = 0,10 m (terrain
    "a peu pres ouvert" : champs, haies, hameaux ; classes de rugosite de
    Wieringa 1992 / Davenport), le facteur vaut 0,50. Sensibilite : z0 = 0,03
    (plaine nue) donne 0,60 ; z0 = 0,25 (village) donne 0,38. Ce facteur est
    une hypothese, pas une mesure, et il est expose en parametre.
  - Composante de vent : vent de face = u * cos(dir_vent - cap) (positif
    quand le vent vient de l'avant), vent de travers = u * sin(dir_vent - cap).
    Le modele de Martin ne consomme que la composante de face ; le travers est
    conserve pour affichage et pour une future dependance CdA(angle de lacet).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from pacing_model import Environment, G

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
               "pressure_msl", "wind_speed_10m", "wind_direction_10m",
               "wind_gusts_10m"]
R_DRY = 287.058      # J/(kg K)
R_VAP = 461.495      # J/(kg K)
LAPSE_K_PER_M = 0.0065


# ---------------------------------------------------------------------------
# 1. Acces aux donnees
# ---------------------------------------------------------------------------

def fetch_hourly(lat: float, lon: float, date: str, cache_dir="data/raw/weather",
                 timezone: str = "Europe/Paris", timeout: float = 30.0) -> dict:
    """Meteo horaire Open-Meteo pour un point et un jour (AAAA-MM-JJ), en cache.

    Renvoie le JSON brut de l'API (cle `elevation` = altitude de la maille,
    `hourly` = series). Le fichier cache porte les coordonnees et la date.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"open_meteo_archive_{lat:.4f}_{lon:.4f}_{date}.json"
    if cache.exists():
        with open(cache) as fh:
            return json.load(fh)

    import requests
    params = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
              "start_date": date, "end_date": date,
              "hourly": ",".join(HOURLY_VARS), "timezone": timezone,
              "wind_speed_unit": "ms"}
    r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    payload["source_url"] = r.url
    payload["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    with open(cache, "w") as fh:
        json.dump(payload, fh)
    return payload


@dataclass
class WeatherSample:
    """Conditions a un instant et un point : unites SI sauf mention."""
    time: datetime
    temp_c: float
    rh_pct: float
    p_surface_hpa: float
    p_msl_hpa: float
    wind10_ms: float
    wind_dir_deg: float       # direction meteorologique (d'ou vient le vent)
    gust10_ms: float
    grid_elevation_m: float
    lat: float
    lon: float


def sample_at(payload: dict, when: datetime) -> WeatherSample:
    """Interpole lineairement les series horaires a l'instant `when` (heure
    locale du fuseau demande). La direction du vent est interpolee en
    vecteur pour eviter le saut 359 -> 0 degres."""
    h = payload["hourly"]
    times = [datetime.fromisoformat(t) for t in h["time"]]
    t0 = times[0]
    xs = np.array([(t - t0).total_seconds() for t in times])
    x = (when.replace(tzinfo=None) - t0).total_seconds()
    if not (xs[0] <= x <= xs[-1]):
        raise ValueError(f"{when} hors de la plage meteo {times[0]} - {times[-1]}")

    def interp(key):
        return float(np.interp(x, xs, np.array(h[key], float)))

    spd = np.array(h["wind_speed_10m"], float)
    ang = np.radians(np.array(h["wind_direction_10m"], float))
    u = float(np.interp(x, xs, spd * np.sin(ang)))
    v = float(np.interp(x, xs, spd * np.cos(ang)))
    direction = float(np.round(np.degrees(np.arctan2(u, v)) % 360.0, 6) % 360.0)

    return WeatherSample(
        time=when, temp_c=interp("temperature_2m"),
        rh_pct=interp("relative_humidity_2m"),
        p_surface_hpa=interp("surface_pressure"), p_msl_hpa=interp("pressure_msl"),
        wind10_ms=interp("wind_speed_10m"), wind_dir_deg=direction,
        gust10_ms=interp("wind_gusts_10m"),
        grid_elevation_m=float(payload["elevation"]),
        lat=float(payload["latitude"]), lon=float(payload["longitude"]),
    )


# ---------------------------------------------------------------------------
# 2. Physique de l'air
# ---------------------------------------------------------------------------

def saturation_vapour_pressure_hpa(temp_c):
    """Formule de Magnus, coefficients d'Alduchov & Eskridge (1996)."""
    temp_c = np.asarray(temp_c, float)
    return 6.1094 * np.exp(17.625 * temp_c / (temp_c + 243.04))


def air_density(temp_c, pressure_hpa, rh_pct):
    """Densite de l'air humide (kg/m3)."""
    t_k = np.asarray(temp_c, float) + 273.15
    p_pa = np.asarray(pressure_hpa, float) * 100.0
    p_v = np.asarray(rh_pct, float) / 100.0 * saturation_vapour_pressure_hpa(temp_c) * 100.0
    return (p_pa - p_v) / (R_DRY * t_k) + p_v / (R_VAP * t_k)


def temperature_at_altitude(temp_ref_c, z_ref_m, z_m, lapse=LAPSE_K_PER_M):
    return np.asarray(temp_ref_c, float) - lapse * (np.asarray(z_m, float) - z_ref_m)


def pressure_at_altitude(p_ref_hpa, z_ref_m, z_m, temp_ref_c, lapse=LAPSE_K_PER_M):
    """Formule hypsometrique avec temperature moyenne de la couche."""
    z_m = np.asarray(z_m, float)
    t_mean_k = np.asarray(temp_ref_c, float) + 273.15 - 0.5 * lapse * (z_m - z_ref_m)
    return np.asarray(p_ref_hpa, float) * np.exp(-G * (z_m - z_ref_m) / (R_DRY * t_mean_k))


def wind_at_height(wind10_ms, z_m: float = 1.0, z0_m: float = 0.10):
    """Profil logarithmique neutre : ramene le vent a 10 m a la hauteur z."""
    return np.asarray(wind10_ms, float) * np.log(z_m / z0_m) / np.log(10.0 / z0_m)


def wind_components(wind_ms, wind_dir_deg, bearing_deg):
    """(vent de face, vent de travers) en m/s pour un cap donne."""
    rel = np.radians(np.asarray(wind_dir_deg, float) - np.asarray(bearing_deg, float))
    w = np.asarray(wind_ms, float)
    return w * np.cos(rel), w * np.sin(rel)


# ---------------------------------------------------------------------------
# 3. Environnement par segment
# ---------------------------------------------------------------------------

def segment_environment(seg, samples, z_rider_m: float = 1.0, z0_m: float = 0.10):
    """Construit un `Environment` vectoriel (rho et vent par segment).

    seg     : SegmentedCourse (course_io).
    samples : WeatherSample ou liste de (distance_m, WeatherSample) ; dans le
              second cas chaque variable est interpolee le long du parcours.
    Renvoie (Environment, DataFrame par segment).
    """
    import pandas as pd

    if isinstance(samples, WeatherSample):
        samples = [(0.0, samples)]
    samples = sorted(samples, key=lambda s: s[0])
    xs = np.array([d for d, _ in samples], float)
    mid = seg.mid_dist_m

    def field(name):
        vals = np.array([getattr(s, name) for _, s in samples], float)
        return np.interp(mid, xs, vals) if len(xs) > 1 else np.full(seg.n, vals[0])

    temp_ref = field("temp_c")
    z_ref = field("grid_elevation_m")
    p_ref = field("p_surface_hpa")
    rh = field("rh_pct")
    wind10 = field("wind10_ms")
    # direction interpolee en vecteur
    spd = np.array([s.wind10_ms for _, s in samples]); ang = np.radians([s.wind_dir_deg for _, s in samples])
    if len(xs) > 1:
        u = np.interp(mid, xs, spd * np.sin(ang)); v = np.interp(mid, xs, spd * np.cos(ang))
    else:
        u = np.full(seg.n, spd[0] * np.sin(ang[0])); v = np.full(seg.n, spd[0] * np.cos(ang[0]))
    wind_dir = np.degrees(np.arctan2(u, v)) % 360.0

    z = seg.mid_ele
    temp = temperature_at_altitude(temp_ref, z_ref, z)
    p = pressure_at_altitude(p_ref, z_ref, z, temp_ref)
    rho = air_density(temp, p, rh)
    w_rider = wind_at_height(wind10, z_rider_m, z0_m)
    head, cross = wind_components(w_rider, wind_dir, seg.bearing_deg)

    df = pd.DataFrame({
        "km": mid / 1000.0, "ele_m": z, "bearing_deg": seg.bearing_deg,
        "temp_c": temp, "pressure_hpa": p, "rh_pct": rh, "rho": rho,
        "wind10_ms": wind10, "wind_rider_ms": w_rider, "wind_dir_deg": wind_dir,
        "headwind_ms": head, "crosswind_ms": cross,
    })
    return Environment(rho=rho, wind_ms=head), df


def rider_weather(seg, date: str, start_time: str, elapsed_s: float = 0.0,
                  n_points: int = 3, cache_dir="data/raw/weather"):
    """Meteo d'un coureur : echantillons a n_points le long du parcours, a
    l'instant start_time + elapsed_s (heure locale). Renvoie la liste
    (distance_m, WeatherSample) attendue par `segment_environment`."""
    when = datetime.fromisoformat(f"{date}T{start_time}") + timedelta(seconds=elapsed_s)
    idx = np.linspace(0, len(seg.dist_m) - 1, n_points).round().astype(int)
    out = []
    for i in idx:
        payload = fetch_hourly(float(seg.lat[i]), float(seg.lon[i]), date, cache_dir=cache_dir)
        out.append((float(seg.dist_m[i]), sample_at(payload, when)))
    return out
