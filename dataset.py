"""Jeu d'entrainement : segments reels issus de GoldenCheetah OpenData.

Chaque sortie avec capteur de puissance (secs, km, power, alt a 1 Hz) est
decoupee en segments de `SEG_M` metres. Pour chaque segment on garde ce que
le modele physique utilise (puissance moyenne, pente, altitude, vitesse
d'entree) et ce qu'il ignore (variabilite de la puissance, cadence,
frequence cardiaque, contexte de la sortie), ainsi que la cible : le temps
du segment, donc la vitesse moyenne.

Traitement en flux : une archive est telechargee, reduite en parquet, puis
supprimee (disque limite). Rien d'autre que le parquet ne reste.

Filtres (documentes, appliques par sortie puis par segment) :
  - sortie >= 20 min, puissance presente, distance > 5 km, GPS present
    (km qui augmente) : exclut le home-trainer ;
  - altitude presente et non constante ; secondes continues (trous > 10 s
    coupent la sortie en morceaux independants) ;
  - segment : longueur entre 150 et 400 m (un saut de distance signale un
    trou GPS), vitesse moyenne entre 1,5 et 25 m/s, vitesse d'entree < 30 m/s,
    pente entre -25 et 25 %, puissance moyenne entre 0 et 1500 W.
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEG_M = 200.0
MIN_RIDE_S = 1200
MIN_RIDE_KM = 5.0
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data/processed/opendata_segments"


def _pieces(df: pd.DataFrame):
    """Coupe la sortie aux trous temporels (> 10 s) et renvoie des morceaux continus."""
    secs = df["secs"].to_numpy(float)
    cuts = np.where(np.diff(secs) > 10)[0] + 1
    for part in np.split(np.arange(len(df)), cuts):
        if len(part) >= 120:
            yield df.iloc[part]


def ride_segments(df: pd.DataFrame, ride_id: str, athlete: str, mass_kg: float) -> pd.DataFrame | None:
    """Segments de SEG_M metres d'une sortie. None si la sortie est rejetee."""
    need = {"secs", "km", "power", "alt"}
    if not need.issubset(df.columns):
        return None
    df = df[["secs", "km", "power", "alt"] + [c for c in ("hr", "cad") if c in df.columns]].copy()
    df = df.dropna(subset=["secs", "km"])
    if len(df) < MIN_RIDE_S or df["km"].iloc[-1] < MIN_RIDE_KM:
        return None
    if df["power"].fillna(0).max() <= 0 or df["alt"].std() < 1.0:
        return None
    if (np.diff(df["km"].to_numpy()) < -1e-6).any():
        return None
    rows = []
    ride_elapsed0 = df["secs"].iloc[0]
    for piece in _pieces(df):
        secs = piece["secs"].to_numpy(float); dist = piece["km"].to_numpy(float) * 1000.0
        power = piece["power"].fillna(0).to_numpy(float); alt = piece["alt"].to_numpy(float)
        hr = piece["hr"].to_numpy(float) if "hr" in piece else np.full(len(piece), np.nan)
        cad = piece["cad"].to_numpy(float) if "cad" in piece else np.full(len(piece), np.nan)
        # altitude lissee sur ~30 s (le barometre des compteurs est bruite au metre pres)
        alt_s = pd.Series(alt).rolling(31, center=True, min_periods=1).mean().to_numpy()
        edges = np.arange(dist[0], dist[-1], SEG_M)
        if len(edges) < 3:
            continue
        idx = np.searchsorted(dist, edges)
        # vitesse instantanee (m/s) sur 5 s pour la vitesse d'entree
        v_inst = np.gradient(dist, secs)
        v_inst = pd.Series(v_inst).rolling(5, center=True, min_periods=1).mean().to_numpy()
        cum_work = np.concatenate([[0.0], np.cumsum(power)])
        for k in range(len(idx) - 1):
            i0, i1 = idx[k], idx[k + 1]
            if i1 - i0 < 3:
                continue
            dt = secs[i1] - secs[i0]; dd = dist[i1] - dist[i0]
            if dt <= 0 or dd <= 0:
                continue
            v = dd / dt
            grade = (alt_s[i1] - alt_s[i0]) / dd
            p = power[i0:i1]
            p_mean = (cum_work[i1] - cum_work[i0]) / (i1 - i0)
            v_in = float(v_inst[i0])
            if not (1.5 <= v <= 25.0 and -0.25 <= grade <= 0.25 and 0.0 <= p_mean <= 1500.0
                    and 150.0 <= dd <= 400.0 and 0.0 <= v_in <= 30.0):
                continue
            with np.errstate(all="ignore"):
                hr_m, cad_m = float(np.nanmean(hr[i0:i1])), float(np.nanmean(cad[i0:i1]))
            rows.append((athlete, ride_id, k, float(dist[i0]), dd, dt, v, grade, float(alt_s[i0]),
                         p_mean, float(p.std()), float((p > 0).mean()), v_in,
                         hr_m, cad_m,
                         float(secs[i0] - ride_elapsed0), mass_kg))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=[
        "athlete", "ride", "k", "dist_m", "length_m", "time_s", "speed_ms", "grade", "alt_m",
        "power_w", "power_sd", "pedal_frac", "v_in_ms", "hr", "cad", "elapsed_s", "mass_kg"]).astype(
        {"length_m": "float32", "time_s": "float32", "speed_ms": "float32", "grade": "float32",
         "alt_m": "float32", "power_w": "float32", "power_sd": "float32", "pedal_frac": "float32",
         "v_in_ms": "float32", "hr": "float32", "cad": "float32", "elapsed_s": "float32", "mass_kg": "float32"})


def process_zip(zip_path: Path, out_dir: Path = OUT_DIR, max_rides: int | None = None) -> dict:
    """Une archive athlete -> un parquet de segments + une ligne de resume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    athlete = zip_path.stem
    n_rides, n_kept, frames = 0, 0, []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        meta = json.loads(zf.read([n for n in names if n.endswith(".json")][0]))
        weights = {}
        for r in meta.get("RIDES", []):
            w = float(r.get("METRICS", {}).get("athlete_weight", "nan") or "nan")
            if 30 < w < 150:
                weights[r["date"][:10].replace("/", "_")] = w
        mass_default = float(np.median(list(weights.values()))) if weights else np.nan
        csvs = sorted(n for n in names if n.endswith(".csv"))[:max_rides]
        for name in csvs:
            n_rides += 1
            try:
                df = pd.read_csv(io.BytesIO(zf.read(name)))
            except Exception:
                continue
            mass = weights.get(name[:10], mass_default)
            seg = ride_segments(df, name[:-4], athlete, mass)
            if seg is not None:
                frames.append(seg); n_kept += 1
    info = {"athlete": athlete, "gender": meta["ATHLETE"].get("gender"), "yob": meta["ATHLETE"].get("yob"),
            "n_rides": n_rides, "n_rides_kept": n_kept, "n_segments": 0, "mass_kg": mass_default}
    if frames:
        allseg = pd.concat(frames, ignore_index=True)
        allseg.to_parquet(out_dir / f"{athlete}.parquet", index=False)
        info["n_segments"] = len(allseg)
    return info


def stream_osf(n_athletes: int, out_dir: Path = OUT_DIR, min_mb: float = 3.0, max_mb: float = 60.0,
               tmp_dir: Path = ROOT / "data/raw/opendata", skip_existing: bool = True, pause_s: float = 2.0):
    """Telecharge, traite et supprime les archives OSF une par une (disque limite).
    Les archives deja en cache dans tmp_dir sont traitees puis conservees."""
    import requests
    S = requests.Session(); S.headers["User-Agent"] = "pacing-optimizer-research/0.1 (python-requests)"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "athletes.csv"
    done = set(pd.read_csv(summary_path).athlete) if summary_path.exists() else set()
    rows = []
    url = "https://api.osf.io/v2/nodes/6hfpz/files/osfstorage/?page[size]=100"
    n = 0
    while url and n < n_athletes:
        r = S.get(url, timeout=60); r.raise_for_status(); d = r.json()
        for f in d["data"]:
            if n >= n_athletes:
                break
            a = f["attributes"]
            if a.get("kind") != "file" or not a["name"].endswith(".zip"):
                continue
            athlete = a["name"][:-4]
            if skip_existing and athlete in done:
                n += 1; continue
            size_mb = (a.get("size") or 0) / 1e6
            if not (min_mb <= size_mb <= max_mb):
                continue
            dest = tmp_dir / a["name"]; cached = dest.exists()
            if not cached:
                with S.get(f["links"]["download"], stream=True, timeout=300) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_content(1 << 20):
                            fh.write(chunk)
            t0 = time.time()
            try:
                info = process_zip(dest, out_dir)
            except Exception as exc:
                info = {"athlete": athlete, "error": str(exc)[:80]}
            info["size_mb"] = round(size_mb, 1); info["proc_s"] = round(time.time() - t0, 1)
            rows.append(info); n += 1
            if not cached:
                dest.unlink()
            print(f"{n:4d} {athlete[:8]} {size_mb:5.1f} Mo -> {info.get('n_segments', 0):7d} segments "
                  f"({info.get('n_rides_kept', 0)}/{info.get('n_rides', 0)} sorties) {info['proc_s']} s", flush=True)
            df = pd.DataFrame(rows)
            old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
            pd.concat([old, df], ignore_index=True).drop_duplicates("athlete", keep="last").to_csv(summary_path, index=False)
            rows = []
            time.sleep(pause_s)
        url = d["links"].get("next")
    return summary_path


def load_segments(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    files = sorted(out_dir.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    stream_osf(n)
