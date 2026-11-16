"""Phase 4 : calage et validation sur les pointages de l'etape 16.

Sorties : reports/tables/stage16_sectors.csv, stage16_loo.csv, stage16_riders.csv,
          stage16_descent_tradeoff.csv, et un resume console.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from course_io import load_course  # noqa: E402
from dynamics import DynamicsParams  # noqa: E402
from calibration import (run_stage16, descent_tradeoff, load_splits, load_riders,  # noqa: E402
                         make_rider, rider_checkpoints, rider_env, STAGE16_PARAMS)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:.1f}")


def main():
    out = ROOT / "reports/tables"; out.mkdir(parents=True, exist_ok=True)
    seg = load_course(ROOT / "data/raw/gpx/tdf2026_stage16.gpx")
    params = STAGE16_PARAMS
    print(f"Parametres virages : a_lat {params.a_lat_max} m/s2, largeur {params.road_width_m} m\n")

    sectors, loo, riders = run_stage16(seg, params)
    sectors.to_csv(out / "stage16_sectors.csv", index=False)
    loo.to_csv(out / "stage16_loo.csv", index=False)
    riders.to_csv(out / "stage16_riders.csv", index=False)

    print("=== Puissances impliquees par secteur (hypotheses : CdA par morphologie, Crr 0,0032, velo 8,5 kg) ===")
    full = sectors[sectors.rider.isin(sectors.groupby("rider").size()[lambda s: s >= 3].index)]
    print(full.pivot_table(index="rider", columns=["km_start", "km_end"], values="power_w").round(0))
    print("\n=== Coureurs : CP calee (min W'bal = 0, W' 25 kJ), puissance moyenne, meteo ===")
    print(riders.sort_values("avg_power_w", ascending=False).to_string(index=False))

    if len(loo):
        print("\n=== Validation par pointage omis (hypothese : puissance constante sur le secteur fusionne) ===")
        print(loo[["rider", "km", "t_real_s", "t_pred_s", "err_s", "err_pct_sector"]].to_string(index=False))
        print(f"\nMAE : {loo.err_s.abs().mean():.1f} s, {loo.err_pct_sector.abs().mean():.2f} % du temps de secteur "
              f"(n = {len(loo)})")
        print("MAE par pointage :")
        print(loo.groupby("km").agg(mae_s=("err_s", lambda e: e.abs().mean()),
                                    mae_pct=("err_pct_sector", lambda e: e.abs().mean()),
                                    bias_s=("err_s", "mean"), n=("err_s", "size")).round(2))

    # compromis descente pour le coureur de reference
    splits, rdf = load_splits(), load_riders()
    ev = make_rider("Remco Evenepoel", rdf)
    env, _ = rider_env(seg, splits, "Remco Evenepoel")
    tr = descent_tradeoff(seg, ev, env, rider_checkpoints(splits, "Remco Evenepoel"), params)
    tr.to_csv(out / "stage16_descent_tradeoff.csv", index=False)
    print("\n=== Evenepoel : puissance impliquee par secteur selon a_lat (non identifiable sur la descente) ===")
    print(tr.round(0).to_string(index=False))

    # choix de a_lat par la validation croisee
    print("\n=== Choix de a_lat : erreur de validation selon la limite de virage ===")
    for a in (5.0, 6.0, 8.0, 10.0, 99.0):
        _, loo_a, riders_a = run_stage16(seg, DynamicsParams(a_lat_max=a), verbose=False)
        ev_cp = riders_a[riders_a.rider == "Remco Evenepoel"].cp_w.iloc[0]
        print(f"  a_lat {a:5.1f} : MAE {loo_a.err_s.abs().mean():5.1f} s ({loo_a.err_pct_sector.abs().mean():.2f} %), "
              f"biais km 9,7 {loo_a[loo_a.km == 9.7].err_s.mean():+5.1f} s, CP Evenepoel {ev_cp:.0f} W")

    # sensibilite CdA sur le coureur de reference
    print("\n=== Sensibilite au CdA (Evenepoel, a_lat par defaut) ===")
    from calibration import invert_sector_powers, calibrate_cp
    for cda in (0.19, 0.21, 0.23):
        r = make_rider("Remco Evenepoel", rdf, cda_override=cda)
        fits, powers = invert_sector_powers(seg, r, env, rider_checkpoints(splits, "Remco Evenepoel"), params)
        cp = calibrate_cp(powers, seg, r, env, params)
        print(f"  CdA {cda:.2f} : " + ", ".join(f"{f.power_w:.0f} W" for f in fits) + f" ; CP {cp:.0f} W ({cp / r.mass_rider:.2f} W/kg)")


if __name__ == "__main__":
    main()
