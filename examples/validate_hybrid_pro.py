"""Validation du modele appris sur les pros : etape 16, pointage omis.

Pour chaque coureur avec pointages, les puissances de secteur sont recalees
sur les autres pointages avec le modele appris (pas la physique), puis le
pointage omis est predit. Meme protocole que la phase 4, donc comparable.
Le modele appris n'a jamais vu un coureur a 6,5 W/kg ni cette route : c'est
un test d'extrapolation, rapporte tel quel.

Sortie : reports/tables/stage16_loo_hybrid.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from course_io import load_course  # noqa: E402
from ml_model import HybridModel  # noqa: E402
from calibration import (load_splits, load_riders, make_rider, rider_checkpoints, rider_env,  # noqa: E402
                         invert_sector_powers, calibrate_cp, leave_one_out, hybrid_simulate_fn, STAGE16_PARAMS)


def main():
    seg = load_course(ROOT / "data/raw/gpx/tdf2026_stage16.gpx")
    model = HybridModel.load()
    splits, rdf = load_splits(), load_riders()
    rows = []
    for name in splits.rider.unique():
        rider = make_rider(name, rdf)
        cps = rider_checkpoints(splits, name)
        if rider is None or len(cps) < 2:
            continue
        env, _ = rider_env(seg, splits, name)
        # CP du coureur : calee avec la physique (phase 4), reutilisee pour p_over_cp
        _, p_phys = invert_sector_powers(seg, rider, env, cps, STAGE16_PARAMS)
        rider.cp = calibrate_cp(p_phys, seg, rider, env, STAGE16_PARAMS)
        fn = hybrid_simulate_fn(model, seg, rider)
        fits_h, _ = invert_sector_powers(seg, rider, env, cps, STAGE16_PARAMS, simulate_fn=fn)
        loo_p = leave_one_out(seg, rider, env, cps, STAGE16_PARAMS); loo_p["model"] = "physique"
        loo_h = leave_one_out(seg, rider, env, cps, STAGE16_PARAMS, simulate_fn=fn); loo_h["model"] = "hybride"
        for d in (loo_p, loo_h):
            d["rider"] = name
        rows += [loo_p, loo_h]
        print(f"{name:<20} P impliquees hybride : " + ", ".join(f"{f.power_w:.0f}" for f in fits_h)
              + f" W | physique : " + ", ".join(f"{p:.0f}" for p in [np.average(p_phys[seg.slice_km(a, b)]) for (a, b) in zip([0] + [c[0] for c in cps[:-1]], [c[0] for c in cps])]))
    loo = pd.concat(rows, ignore_index=True)
    loo.to_csv(ROOT / "reports/tables/stage16_loo_hybrid.csv", index=False)
    summ = loo.groupby("model").agg(mae_s=("err_s", lambda e: e.abs().mean()), mae_pct=("err_pct_sector", lambda e: e.abs().mean()),
                                    bias_s=("err_s", "mean"), n=("err_s", "size")).round(2)
    print("\nPointage omis, etape 16 (memes coureurs, memes pointages) :"); print(summ)
    print(loo.pivot_table(index="rider", columns="model", values="err_s", aggfunc=lambda e: e.abs().mean()).round(1))


if __name__ == "__main__":
    main()
