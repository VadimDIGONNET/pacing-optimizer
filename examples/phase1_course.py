"""Phase 1 : controle du parcours GPX de l'etape 16 contre le profil publie.

Produit reports/figures/stage16_profile.png et affiche les controles :
distance, denivele positif, pente moyenne de la cote de Larringes, ecart
avec l'altitude de controle Open-Meteo.

Sources :
  - GPX : https://cdn.cyclingstage.com/images/tour-de-france/2026/stage-16-parcours.gpx
  - Profil publie : https://www.intervals.icu/tour-de-france-2026/stage-16/
    (26,1 km, 500 m D+, Cote de Larringes 9,54 km a 4,4 %, +403 m, sommet km 9,64,
    altitudes affichees 385 m et 394 m)
  - Pointages : https://www.cyclingstage.com/remo-evenepoel/stage-16-results-tdf-2026
    (km 4,8 ; 9,7 ; 18,1)
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from course_io import (load_gpx, smooth_elevation, segment_track,  # noqa: E402
                       elevation_gain, fetch_elevation_open_meteo)

GPX = ROOT / "data/raw/gpx/tdf2026_stage16.gpx"
PUBLISHED = {
    "distance_km": 26.1, "gain_m": 500.0,
    "larringes": {"length_km": 9.54, "grade_pct": 4.4, "gain_m": 403.0, "summit_km": 9.64},
    "checkpoints_km": [4.8, 9.7, 18.1],
    "altitudes": {"start": 385.0, "finish": 394.0},
}


def checks(seg, track):
    lar = PUBLISHED["larringes"]
    i1 = np.searchsorted(seg.dist_m, lar["summit_km"] * 1000)
    i0 = np.searchsorted(seg.dist_m, (lar["summit_km"] - lar["length_km"]) * 1000)
    gain = seg.ele[i1] - seg.ele[i0]
    grade = gain / (seg.dist_m[i1] - seg.dist_m[i0]) * 100
    return {
        "distance_km": seg.total_km,
        "distance_err_pct": (seg.total_km / PUBLISHED["distance_km"] - 1) * 100,
        "gain_raw_m": elevation_gain(track.ele_raw),
        "gain_smooth_m": seg.elevation_gain_m,
        "larringes_gain_m": gain, "larringes_grade_pct": grade,
        "summit_km": seg.dist_m[np.argmax(seg.ele)] / 1000, "summit_m": seg.ele.max(),
        "start_m": seg.ele[0], "finish_m": seg.ele[-1],
    }


def main():
    track = load_gpx(GPX)
    ele_s = smooth_elevation(track)
    seg = segment_track(track, step_m=50.0, ele=ele_s)
    c = checks(seg, track)
    dem = fetch_elevation_open_meteo(seg.lat, seg.lon, cache_dir=ROOT / "data/raw/elevation")
    diff = dem - seg.ele

    print(f"Distance GPX        : {c['distance_km']:.3f} km ({c['distance_err_pct']:+.2f} % vs 26,1 km)")
    print(f"D+ brut / lisse     : {c['gain_raw_m']:.0f} m / {c['gain_smooth_m']:.0f} m (publie : ~500 m)")
    print(f"Larringes (km 0,10 -> 9,64) : +{c['larringes_gain_m']:.0f} m, {c['larringes_grade_pct']:.2f} % "
          f"(publie : +403 m, 4,4 %)")
    print(f"Sommet GPX          : km {c['summit_km']:.2f}, {c['summit_m']:.0f} m "
          f"(publie : sommet de la cote au km 9,64)")
    print(f"Depart / arrivee    : {c['start_m']:.0f} / {c['finish_m']:.0f} m (publie : 385 / 394 m)")
    print(f"Controle Open-Meteo : biais {diff.mean():+.1f} m, ecart-type {diff.std():.1f} m, "
          f"max {np.abs(diff).max():.0f} m (MNT 90 m, bruite sur les versants)")
    print(f"Virages r < 40 m    : {(seg.radius_m < 40).sum()} segments de 50 m")

    km = seg.dist_m / 1000.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    ax.fill_between(track.dist_m / 1000, track.ele_raw, 360, color="#CBD5E1", label="GPX brut")
    ax.plot(km, seg.ele, color="#0F766E", lw=2, label="GPX lisse (Savitzky-Golay 250 m)")
    ax.plot(km, dem, color="#B45309", lw=1, alpha=0.7, label="Open-Meteo (MNT 90 m), controle")
    lar = PUBLISHED["larringes"]
    ax.plot([lar["summit_km"] - lar["length_km"], lar["summit_km"]],
            [PUBLISHED["altitudes"]["start"] + 8, PUBLISHED["altitudes"]["start"] + 8 + lar["gain_m"]],
            "--", color="#B91C1C", lw=1.5, label="Cote de Larringes publiee (9,54 km, +403 m)")
    for x in PUBLISHED["checkpoints_km"]:
        ax.axvline(x, color="#6B7280", ls=":", lw=1)
        ax.text(x, 830, f"km {x}", ha="center", fontsize=8, color="#6B7280")
    ax.set_ylabel("Altitude (m)"); ax.set_ylim(360, 850); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Tour de France 2026, etape 16 : profil GPX contre profil publie")
    ax2.bar(seg.mid_dist_m / 1000, seg.grade * 100, width=0.05, color="#0F766E")
    ax2.set_ylabel("Pente (%)"); ax2.set_xlabel("Distance (km)"); ax2.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    out = ROOT / "reports/figures/stage16_profile.png"
    fig.savefig(out, dpi=130)
    print(f"Figure : {out.relative_to(ROOT)}")
    return c


if __name__ == "__main__":
    main()
