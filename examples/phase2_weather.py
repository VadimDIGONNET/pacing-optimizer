"""Phase 2 : meteo reelle de l'etape 16 par segment (Open-Meteo archive).

Produit :
  - data/processed/tdf2026_stage16_weather_segments.csv (rho et vent par segment,
    pour l'heure de passage d'Evenepoel, depart 17:13 CEST)
  - reports/figures/stage16_weather.png

Sources :
  - https://archive-api.open-meteo.com/v1/archive (cache data/raw/weather/)
  - Date et premier depart : https://www.intervals.icu/tour-de-france-2026/stage-16/
  - Horaires de depart : https://www.bikeradar.com/news/2026-tour-de-france-stage-16-time-trial-start-times
  - Observation independante : bikeraceinfo.com, "26C, sunny, wind NNE 11 km/h at Thonon 2:40 PM"
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from course_io import load_course  # noqa: E402
from weather import rider_weather, segment_environment, fetch_hourly  # noqa: E402

DATE = "2026-07-21"
START_EVENEPOEL = "17:13:00"     # CEST, bikeradar
HALF_RACE_S = 970                # ~ mi-course (32'19" / 2)


def main():
    seg = load_course(ROOT / "data/raw/gpx/tdf2026_stage16.gpx")
    samples = rider_weather(seg, DATE, START_EVENEPOEL, elapsed_s=HALF_RACE_S,
                            cache_dir=ROOT / "data/raw/weather")
    env, df = segment_environment(seg, samples)
    df.insert(0, "segment", np.arange(seg.n))
    df["source_url"] = "https://archive-api.open-meteo.com/v1/archive"
    out = ROOT / "data/processed/tdf2026_stage16_weather_segments.csv"
    df.round(4).to_csv(out, index=False)

    s0 = samples[0][1]
    print(f"Meteo a {s0.time:%H:%M} CEST (depart Evenepoel + {HALF_RACE_S // 60} min) :")
    for d, s in samples:
        print(f"  km {d / 1000:4.1f} (maille {s.grid_elevation_m:.0f} m) : {s.temp_c:.1f} C, "
              f"HR {s.rh_pct:.0f} %, {s.p_surface_hpa:.1f} hPa, vent 10 m {s.wind10_ms:.1f} m/s "
              f"de {s.wind_dir_deg:.0f} deg, rafales {s.gust10_ms:.1f} m/s")
    print(f"Densite par segment : {df.rho.min():.3f} a {df.rho.max():.3f} kg/m3 (moyenne {df.rho.mean():.3f})")
    print(f"Vent a 1 m : {df.wind_rider_ms.mean():.2f} m/s ; composante de face : "
          f"{df.headwind_ms.min():+.2f} a {df.headwind_ms.max():+.2f} m/s, moyenne {df.headwind_ms.mean():+.2f}")
    for lab, a, b in [("montee 0-10.9", 0, 10.9), ("descente 10.9-17", 10.9, 17.0), ("plat 17-26.1", 17.0, 26.2)]:
        idx = seg.slice_km(a, b)
        print(f"  {lab:<16}: vent de face moyen {np.average(df.headwind_ms[idx], weights=seg.length_m[idx]):+.2f} m/s")

    # evolution sur l'apres-midi au depart
    payload = fetch_hourly(float(seg.lat[0]), float(seg.lon[0]), DATE, cache_dir=ROOT / "data/raw/weather")
    h = payload["hourly"]
    hours = [int(t[11:13]) for t in h["time"]]
    sel = [i for i, hh in enumerate(hours) if 12 <= hh <= 18]
    print("Evolution au depart (Evian) :")
    for i in sel:
        print(f"  {h['time'][i][11:]} : {h['temperature_2m'][i]:.1f} C, vent {h['wind_speed_10m'][i]:.1f} m/s "
              f"de {h['wind_direction_10m'][i]:.0f} deg")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    km = df.km
    axes[0].fill_between(seg.dist_m / 1000, seg.ele, 360, color="#CBD5E1")
    axes[0].set_ylabel("Altitude (m)"); axes[0].set_title("Etape 16 : meteo par segment (Open-Meteo, 17:29 CEST)")
    axes[1].plot(km, df.rho, color="#0F766E", lw=2); axes[1].set_ylabel("rho (kg/m3)")
    axes[2].bar(km, df.headwind_ms, width=0.05, color=np.where(df.headwind_ms > 0, "#B91C1C", "#2563EB"))
    axes[2].axhline(0, color="k", lw=0.5); axes[2].set_ylabel("Vent de face (m/s)\nrouge = face, bleu = dos")
    axes[2].set_xlabel("Distance (km)")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/stage16_weather.png", dpi=130)
    print("Table : data/processed/tdf2026_stage16_weather_segments.csv ; figure : reports/figures/stage16_weather.png")


if __name__ == "__main__":
    main()
