"""Phase 5 : CP, W' et test W'bal sur l'echantillon GoldenCheetah OpenData.

Sorties : data/processed/opendata_cp_wprime.csv, reports/figures/opendata_physiology.png
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
from physiology import analyse_athlete, DURATIONS_S  # noqa: E402


def main():
    zips = sorted((ROOT / "data/raw/opendata").glob("*.zip"))
    rows, curves, wb_all = [], {}, []
    for z in zips:
        r = analyse_athlete(z)
        print(f"{r['athlete'][:8]}  {r['gender']} {r['yob']}  sorties {r['n_rides_power']:3d}  masse {r['mass_kg']:5.1f} kg  "
              f"CP {r['cp_w']:5.0f} W ({r['cp_wkg']:.2f} W/kg)  W' {r['w_prime_j'] / 1000:5.1f} kJ  R2 {r['fit_r2']:.3f}  "
              f"W'bal min/W' sur {r['n_maximal_rides']} sorties max : "
              + (f"{np.mean(r['wbal_min_frac']):+.2f} (min {np.min(r['wbal_min_frac']):+.2f})" if r["wbal_min_frac"] else "n/a"))
        rows.append({k: v for k, v in r.items() if k not in ("mmp", "wbal_min_frac")}
                    | {"wbal_min_frac_mean": np.mean(r["wbal_min_frac"]) if r["wbal_min_frac"] else np.nan,
                       "source_url": "https://osf.io/6hfpz/", "license": "CC0 1.0"})
        curves[r["athlete"]] = r["mmp"]
        wb_all += r["wbal_min_frac"]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/processed/opendata_cp_wprime.csv", index=False)
    ok = df[(df.fit_r2 > 0.98) & (df.n_rides_power >= 20)]
    print(f"\nAthletes retenus (R2 > 0,98, >= 20 sorties) : {len(ok)} / {len(df)}")
    print(f"CP/kg : mediane {ok.cp_wkg.median():.2f}, plage {ok.cp_wkg.min():.2f} - {ok.cp_wkg.max():.2f} W/kg")
    print(f"W'    : mediane {ok.w_prime_j.median() / 1000:.1f} kJ, plage {ok.w_prime_j.min() / 1000:.1f} - {ok.w_prime_j.max() / 1000:.1f} kJ "
          f"({(ok.w_prime_j / ok.mass_kg).median():.0f} J/kg median)")
    wb = np.array(wb_all)
    print(f"W'bal min / W' sur les sorties maximales : mediane {np.median(wb):+.2f}, "
          f"IQR [{np.percentile(wb, 25):+.2f}, {np.percentile(wb, 75):+.2f}], n = {len(wb)}")
    print("Lecture : 0 = le modele vide exactement W' a l'effort maximal ; < 0 = W' sous-estime ou CP surestime.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, m in curves.items():
        d = sorted(m); axes[0].plot(d, [m[k] for k in d], color="#0F766E", alpha=0.5, lw=1)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Duree (s)"); axes[0].set_ylabel("MMP (W)")
    axes[0].set_title(f"Courbes puissance-duree ({len(curves)} athletes OpenData)")
    axes[1].scatter(ok.cp_wkg, ok.w_prime_j / 1000, color="#0F766E")
    axes[1].set_xlabel("CP (W/kg)"); axes[1].set_ylabel("W' (kJ)"); axes[1].set_title("CP et W' ajustes (3 a 20 min)")
    axes[2].hist(wb, bins=15, color="#0F766E", alpha=0.8); axes[2].axvline(0, color="#B91C1C", ls="--")
    axes[2].set_xlabel("min W'bal / W' sur les sorties maximales"); axes[2].set_title("Test du W'bal de Skiba")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/opendata_physiology.png", dpi=130)
    print("Figure : reports/figures/opendata_physiology.png")


if __name__ == "__main__":
    main()
