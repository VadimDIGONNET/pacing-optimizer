"""Phase 6 : strategie reelle, puissance constante et strategie optimale,
etape 16 du Tour 2026, pour les coureurs du top 10 dont on a la masse.

Pour chaque coureur : CP calee en phase 4 (min W'bal = 0 sur l'effort reel,
W' = 25 kJ), CdA par morphologie, meteo a son heure de passage, limite de
virage calee (a_lat = 8 m/s2). "Reelle" = puissances par secteur impliquees
par ses pointages (ce ne sont pas des watts mesures). Les coureurs sans
pointage intermediaire n'ont qu'une puissance moyenne : leur strategie
reelle est inconnue, on ne compare que constante et optimale.

Sorties : reports/tables/stage16_strategies.csv, reports/figures/stage16_strategies.png
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
from dynamics import simulate_dynamic  # noqa: E402
from optimizer import optimize_dynamic, plan_table  # noqa: E402
from calibration import (load_splits, load_riders, make_rider, rider_checkpoints, rider_env,  # noqa: E402
                         invert_sector_powers, calibrate_cp, STAGE16_PARAMS)

BREAKPOINTS = [9.64, 10.9, 17.0]
TOP10 = ["Remco Evenepoel", "Tadej Pogacar", "Mattias Skjelmose", "Paul Seixas", "Isaac del Toro",
         "Mattia Cattaneo", "Thymen Arensman", "Bruno Armirail", "Pablo Castrillo", "Joshua Tarling"]


def fmt(t):
    return f"{int(t // 60)}'{t % 60:04.1f}\""


def main():
    seg = load_course(ROOT / "data/raw/gpx/tdf2026_stage16.gpx")
    splits, rdf = load_splits(), load_riders()
    rows, plans = [], {}
    for name in TOP10:
        rider = make_rider(name, rdf)
        if rider is None:
            print(f"{name}: masse inconnue, ignore"); continue
        cps = rider_checkpoints(splits, name)
        env, _ = rider_env(seg, splits, name)
        fits, p_real = invert_sector_powers(seg, rider, env, cps, STAGE16_PARAMS)
        rider.cp = calibrate_cp(p_real, seg, rider, env, STAGE16_PARAMS)
        real = simulate_dynamic(p_real, seg, rider, env, STAGE16_PARAMS)
        opt = optimize_dynamic(seg, rider, env, STAGE16_PARAMS, block_m=500.0, breakpoints_km=BREAKPOINTS)
        plans[name] = (opt, real)
        row = {"rider": name, "mass_kg": rider.mass_rider, "cda": round(rider.cda, 3), "cp_w": rider.cp,
               "n_checkpoints": len(cps), "real_s": real.time_s, "constant_w": opt.p_constant,
               "constant_s": opt.constant.time_s, "optimal_s": opt.result.time_s,
               "gain_vs_real_s": real.time_s - opt.result.time_s,
               "gain_vs_constant_s": opt.gain_s, "calc_s": opt.elapsed_s}
        for a, b in zip([0.0] + BREAKPOINTS, BREAKPOINTS + [seg.total_km]):
            idx = seg.slice_km(a, b)
            row[f"P_opt {a:.1f}-{b:.1f}"] = np.average(opt.result.powers[idx], weights=seg.length_m[idx])
            row[f"P_real {a:.1f}-{b:.1f}"] = np.average(p_real[idx], weights=seg.length_m[idx])
        rows.append(row)
        print(f"{name:<20} CP {rider.cp:.0f} W | reel {fmt(real.time_s)} | constant {opt.p_constant:.0f} W {fmt(opt.constant.time_s)} | "
              f"optimal {fmt(opt.result.time_s)} | gain vs reel {row['gain_vs_real_s']:+.1f} s, vs constant {opt.gain_s:+.1f} s "
              f"({len(cps)} pointages, {opt.elapsed_s:.0f} s calcul)")
    df = pd.DataFrame(rows)
    (ROOT / "reports/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "reports/tables/stage16_strategies.csv", index=False)
    full = df[df.n_checkpoints >= 4]
    print(f"\nCoureurs avec pointages complets : gain moyen vs strategie reelle {full.gain_vs_real_s.mean():+.1f} s, "
          f"vs constante {full.gain_vs_constant_s.mean():+.1f} s")
    print("Secteurs (W, moyenne ponderee) :")
    cols = [c for c in df.columns if c.startswith("P_")]
    print(full[["rider"] + cols].round(0).to_string(index=False))

    # figure Evenepoel
    opt, real = plans["Remco Evenepoel"]
    km = seg.mid_dist_m / 1000
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                        gridspec_kw={"height_ratios": [1, 2, 1.3]})
    ax1.fill_between(seg.dist_m / 1000, seg.ele, 360, color="#CBD5E1"); ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Etape 16 : strategies pour Evenepoel (CP calee, CdA 0,21, W' 25 kJ)")
    ax2.plot(km, real.powers, color="#B45309", lw=1.5, label=f"Reelle (estimee des pointages) {fmt(real.time_s)}")
    ax2.plot(km, opt.constant.powers, color="#6B7280", lw=1.5, label=f"Constante {opt.p_constant:.0f} W, {fmt(opt.constant.time_s)}")
    ax2.plot(km, opt.result.powers, color="#0F766E", lw=2.2, label=f"Optimale {fmt(opt.result.time_s)}")
    ax2.plot(km, opt.result.applied_power, color="#0F766E", lw=0.8, ls=":", label="Optimale appliquee (freinages)")
    ax2.set_ylabel("Puissance (W)"); ax2.legend(fontsize=8, loc="lower left", ncol=2)
    ax3.plot(km, real.wbal_j / 1000, color="#B45309", lw=1.5)
    ax3.plot(km, opt.constant.wbal_j / 1000, color="#6B7280", lw=1.5)
    ax3.plot(km, opt.result.wbal_j / 1000, color="#0F766E", lw=2.2)
    ax3.axhline(0, color="#B91C1C", ls="--", lw=1); ax3.set_ylabel("W'bal (kJ)"); ax3.set_xlabel("Distance (km)")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/stage16_strategies.png", dpi=130)
    plan_table(opt, seg).round(1).to_csv(ROOT / "reports/tables/stage16_plan_evenepoel.csv", index=False)
    print("Figure : reports/figures/stage16_strategies.png ; plan : reports/tables/stage16_plan_evenepoel.csv")


if __name__ == "__main__":
    main()
