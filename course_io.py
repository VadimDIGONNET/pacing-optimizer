"""Lecture de traces GPX et construction d'un parcours segmente (phase 1).

Pipeline :
  1. `load_gpx`          : points (lat, lon, altitude), distance cumulee geodesique
                           sur l'ellipsoide WGS84 (pyproj).
  2. `smooth_elevation`  : filtre de Savitzky-Golay sur une grille reguliere.
  3. `segment_track`     : reechantillonnage tous les 50 a 100 m ; pour chaque
                           segment : longueur, pente, cap, rayon de courbure
                           minimal. Le resultat expose un `Course` compatible
                           avec `pacing_model`, plus les attributs geometriques
                           dont les phases 2 (vent) et 3 (virages) ont besoin.
  4. `fetch_elevation_open_meteo` : altitude de controle (API Open-Meteo,
                           modele de terrain a 90 m), cache JSON dans data/raw.

Choix de parametres (justifies dans les docstrings) :
  - lissage altitude : fenetre 250 m, polynome d'ordre 2 ;
  - courbure : derivees de Savitzky-Golay sur une grille de 5 m, fenetre 35 m.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import gpxpy
import numpy as np
from pyproj import Geod
from scipy.signal import savgol_filter

from pacing_model import Course

GEOD = Geod(ellps="WGS84")
EARTH_R = 6371008.8   # rayon moyen (m), pour la projection locale


# ---------------------------------------------------------------------------
# 1. Lecture
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Trace brute : un point par echantillon GPX, distance cumulee en metres."""
    lat: np.ndarray
    lon: np.ndarray
    ele_raw: np.ndarray
    dist_m: np.ndarray
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.lat)

    @property
    def total_km(self) -> float:
        return float(self.dist_m[-1] / 1000.0)


def load_gpx(path) -> Track:
    """Charge un GPX (tracks puis routes) et calcule la distance geodesique.

    Les points consecutifs confondus (distance nulle) sont supprimes : ils
    rendraient la pente indefinie.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)

    pts = [p for t in gpx.tracks for s in t.segments for p in s.points]
    if not pts:
        pts = [p for r in gpx.routes for p in r.points]
    if len(pts) < 2:
        raise ValueError(f"{path} : moins de deux points")

    lat = np.array([p.latitude for p in pts], float)
    lon = np.array([p.longitude for p in pts], float)
    ele = np.array([p.elevation if p.elevation is not None else np.nan
                    for p in pts], float)
    if np.isnan(ele).any():
        raise ValueError(f"{path} : altitudes manquantes dans le GPX")

    _, _, step = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    keep = np.concatenate([[True], step > 0.0])
    lat, lon, ele = lat[keep], lon[keep], ele[keep]
    _, _, step = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dist = np.concatenate([[0.0], np.cumsum(step)])
    return Track(lat, lon, ele, dist, source=str(path))


# ---------------------------------------------------------------------------
# 2. Lissage de l'altitude
# ---------------------------------------------------------------------------

def _odd_window(length_m: float, step_m: float, minimum: int = 5) -> int:
    w = int(round(length_m / step_m))
    w = max(w, minimum)
    return w if w % 2 == 1 else w + 1


def smooth_elevation(track: Track, window_m: float = 250.0, polyorder: int = 2,
                     step_m: float = 5.0) -> np.ndarray:
    """Altitude lissee aux points de la trace (Savitzky-Golay).

    Pourquoi ces valeurs : le GPX de cyclingstage a une altitude entiere
    (quantification de 1 m) et un espacement median de 7 m. Une fenetre de
    250 m supprime le bruit de quantification tout en conservant les ruptures
    de pente reelles, qui s'etalent sur plus de 200 m sur une route. L'ordre 2
    preserve la courbure du profil (sommets et bas de descente). Le filtre est
    applique sur une grille reguliere de 5 m pour que la fenetre soit homogene
    en metres, puis reinterpole aux points d'origine.
    """
    grid = np.arange(0.0, track.dist_m[-1] + step_m, step_m)
    grid = grid[grid <= track.dist_m[-1]]
    ele_grid = np.interp(grid, track.dist_m, track.ele_raw)
    w = _odd_window(window_m, step_m, minimum=polyorder + 2)
    w = min(w, len(grid) - (1 - len(grid) % 2))
    ele_s = savgol_filter(ele_grid, w, polyorder, mode="interp")
    return np.interp(track.dist_m, grid, ele_s)


def elevation_gain(ele: np.ndarray) -> float:
    """Denivele positif (m) : somme des montees entre points consecutifs."""
    d = np.diff(np.asarray(ele, float))
    return float(d[d > 0].sum())


# ---------------------------------------------------------------------------
# 3. Segmentation
# ---------------------------------------------------------------------------

@dataclass
class SegmentedCourse:
    """Parcours reechantillonne : n segments, n+1 noeuds.

    Attributs par noeud (n+1) : dist_m, lat, lon, ele.
    Attributs par segment (n) : length_m, grade, bearing_deg (cap geographique
    0 = nord, 90 = est), radius_m (rayon de courbure minimal rencontre dans
    le segment ; np.inf en ligne droite), turn_deg (changement de cap cumule).
    """
    dist_m: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    ele: np.ndarray
    length_m: np.ndarray
    grade: np.ndarray
    bearing_deg: np.ndarray
    radius_m: np.ndarray
    turn_deg: np.ndarray
    source: str = ""
    corners: list = None       # liste de dict : s_start_m, s_end_m, turn_deg, r_min_m, sign

    @property
    def n(self) -> int:
        return len(self.length_m)

    @property
    def total_km(self) -> float:
        return float(self.dist_m[-1] / 1000.0)

    @property
    def mid_dist_m(self) -> np.ndarray:
        return 0.5 * (self.dist_m[:-1] + self.dist_m[1:])

    @property
    def mid_ele(self) -> np.ndarray:
        return 0.5 * (self.ele[:-1] + self.ele[1:])

    @property
    def course(self) -> Course:
        """Vue compatible avec l'API v1 (`simulate`, `optimize_pacing`)."""
        return Course(self.length_m.copy(), self.grade.copy())

    @property
    def elevation_gain_m(self) -> float:
        return elevation_gain(self.ele)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({
            "km_start": self.dist_m[:-1] / 1000.0,
            "km_end": self.dist_m[1:] / 1000.0,
            "length_m": self.length_m,
            "grade_pct": self.grade * 100.0,
            "ele_m": self.mid_ele,
            "bearing_deg": self.bearing_deg,
            "radius_m": self.radius_m,
            "turn_deg": self.turn_deg,
            "lat": 0.5 * (self.lat[:-1] + self.lat[1:]),
            "lon": 0.5 * (self.lon[:-1] + self.lon[1:]),
        })

    def slice_km(self, km_start: float, km_end: float) -> np.ndarray:
        """Indices des segments dont le milieu est dans [km_start, km_end[."""
        mid = self.mid_dist_m / 1000.0
        return np.where((mid >= km_start) & (mid < km_end))[0]


def _local_xy(lat, lon, lat0, lon0):
    """Projection equirectangulaire locale en metres (parcours < 100 km)."""
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def curvature_profile(track: Track, step_m: float = 5.0, window_m: float = 40.0,
                      smooth_m: float = 25.0):
    """Courbure effective kappa(s) (1/m) sur une grille reguliere.

    Methode : cap theta(s) calcule sur les positions lissees (Savitzky-Golay,
    fenetre `smooth_m`, ordre 2), deroule, puis courbure = variation NETTE de
    cap sur une fenetre glissante de `window_m` divisee par cette longueur :
        kappa(s) = [theta(s + W/2) - theta(s - W/2)] / W.
    Pourquoi : deriver deux fois une trace GPS bruitee (1 a 3 m) fabrique des
    virages fantomes de quelques metres de rayon. Le changement net de cap
    sur 40 m est insensible a ce bruit (les zigzags s'annulent) mais capte
    bien un lacet : 180 degres sur 40 m donnent r = 12,7 m. Un virage large
    de rayon 100 m tourne de 23 degres sur 40 m, bien au-dessus du bruit
    residuel (environ 3 degres). La fenetre est donc un compromis entre
    resolution (epingles) et robustesse. Renvoie (grille_s, kappa).
    """
    grid = np.arange(0.0, track.dist_m[-1], step_m)
    lat = np.interp(grid, track.dist_m, track.lat)
    lon = np.interp(grid, track.dist_m, track.lon)
    x, y = _local_xy(lat, lon, track.lat.mean(), track.lon.mean())
    w = _odd_window(smooth_m, step_m, minimum=5)
    dx = savgol_filter(x, w, 2, deriv=1, delta=step_m, mode="interp")
    dy = savgol_filter(y, w, 2, deriv=1, delta=step_m, mode="interp")
    theta = np.unwrap(np.arctan2(dy, dx))
    half = max(int(round(window_m / step_m / 2)), 1)
    theta_pad = np.pad(theta, half, mode="edge")
    dtheta = theta_pad[2 * half:] - theta_pad[:-2 * half]
    kappa = dtheta / (2 * half * step_m)
    return grid, kappa


def segment_track(track: Track, step_m: float = 50.0, ele=None,
                  min_radius_m: float = 8.0, straight_radius_m: float = 2000.0,
                  corner_radius_m: float = 150.0, **curv_kwargs) -> SegmentedCourse:
    """Reechantillonne la trace en segments de `step_m` (le dernier plus court).

    ele : altitude par point de la trace (lissee en general) ; par defaut
    `smooth_elevation(track)`.
    min_radius_m : plancher du rayon (en dessous, bruit ou demi-tour).
    straight_radius_m : au-dela, le segment est considere droit (rayon inf).
    """
    if ele is None:
        ele = smooth_elevation(track)
    total = track.dist_m[-1]
    nodes = np.arange(0.0, total, step_m)
    if total - nodes[-1] < 0.25 * step_m and len(nodes) > 1:
        nodes = nodes[:-1]
    nodes = np.append(nodes, total)

    lat = np.interp(nodes, track.dist_m, track.lat)
    lon = np.interp(nodes, track.dist_m, track.lon)
    ele_n = np.interp(nodes, track.dist_m, ele)
    length = np.diff(nodes)
    grade = np.diff(ele_n) / length
    az, _, _ = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    bearing = np.mod(az, 360.0)

    grid, kappa = curvature_profile(track, **curv_kwargs)
    idx = np.clip(np.searchsorted(nodes, grid, side="right") - 1, 0, len(length) - 1)
    kmax = np.zeros(len(length))
    np.maximum.at(kmax, idx, np.abs(kappa))
    radius = np.where(kmax > 0, 1.0 / np.maximum(kmax, 1e-9), np.inf)
    radius = np.maximum(radius, min_radius_m)
    radius[radius > straight_radius_m] = np.inf

    dtheta = np.abs(kappa) * (grid[1] - grid[0] if len(grid) > 1 else 0.0)
    turn = np.zeros(len(length))
    np.add.at(turn, idx, np.degrees(dtheta))

    corners = detect_corners(grid, kappa, min_radius_m=min_radius_m,
                             corner_radius_m=corner_radius_m)
    return SegmentedCourse(nodes, lat, lon, ele_n, length, grade, bearing,
                           radius, turn, source=track.source, corners=corners)


def detect_corners(grid, kappa, corner_radius_m: float = 150.0,
                   min_radius_m: float = 8.0, min_turn_deg: float = 15.0):
    """Regroupe les echantillons de courbure en virages.

    Un virage = suite d'echantillons consecutifs de meme signe dont le rayon
    est inferieur a `corner_radius_m` (150 m : au-dela, un velo de CLM passe
    sans ralentir a 80 km/h, a_lat = 3,3 m/s2). On conserve les virages dont
    l'angle total depasse `min_turn_deg`. Chaque virage garde son angle total
    (degres), son rayon minimal (axe de la route) et son etendue en metres.
    """
    ds = grid[1] - grid[0] if len(grid) > 1 else 0.0
    sharp = np.abs(kappa) > 1.0 / corner_radius_m
    sign = np.sign(kappa)
    corners, i, n = [], 0, len(kappa)
    while i < n:
        if not sharp[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and sharp[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        k = np.abs(kappa[i:j + 1])
        turn = float(np.degrees(k.sum() * ds))
        if turn >= min_turn_deg:
            corners.append({"s_start_m": float(grid[i]), "s_end_m": float(grid[j] + ds),
                            "turn_deg": turn, "r_min_m": float(max(1.0 / k.max(), min_radius_m)),
                            "sign": int(sign[i])})
        i = j + 1
    return corners


def load_course(path, step_m: float = 50.0, **kwargs) -> SegmentedCourse:
    """Raccourci : GPX -> trace -> altitude lissee -> parcours segmente."""
    track = load_gpx(path)
    return segment_track(track, step_m=step_m, **kwargs)


# ---------------------------------------------------------------------------
# 4. Altitude de controle (Open-Meteo)
# ---------------------------------------------------------------------------

OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"


def fetch_elevation_open_meteo(lat, lon, cache_dir="data/raw/elevation",
                               batch: int = 100, timeout: float = 30.0) -> np.ndarray:
    """Altitude du modele de terrain Open-Meteo (Copernicus DEM 90 m).

    Les reponses sont mises en cache (JSON) dans `cache_dir`, cle = hash des
    coordonnees, pour ne jamais rappeler l'API sur la meme trace. Usage non
    commercial, moins de 10 000 appels par jour : conditions Open-Meteo.
    """
    import requests

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(np.round(np.c_[lat, lon], 6).tobytes()).hexdigest()[:16]
    cache = cache_dir / f"open_meteo_elevation_{key}.json"
    if cache.exists():
        with open(cache) as fh:
            return np.array(json.load(fh)["elevation"], float)

    out = []
    for i in range(0, len(lat), batch):
        params = {"latitude": ",".join(f"{v:.6f}" for v in lat[i:i + batch]),
                  "longitude": ",".join(f"{v:.6f}" for v in lon[i:i + batch])}
        r = requests.get(OPEN_METEO_ELEVATION, params=params, timeout=timeout)
        r.raise_for_status()
        out.extend(r.json()["elevation"])
    payload = {"source_url": OPEN_METEO_ELEVATION, "n": len(out),
               "latitude": lat.tolist(), "longitude": lon.tolist(),
               "elevation": out}
    with open(cache, "w") as fh:
        json.dump(payload, fh)
    return np.array(out, float)
