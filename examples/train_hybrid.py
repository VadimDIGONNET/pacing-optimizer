"""Entraine et valide le modele hybride sur les segments OpenData disponibles.

Validation : (a) sorties jamais vues des memes athletes, (b) athletes jamais vus.
Sorties : models/hybrid.joblib, models/profiles.json, models/metrics.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import load_segments  # noqa: E402
from ml_model import prepare, train, split_rides, evaluate, tt_like, MODEL_DIR  # noqa: E402


def main(max_segments_per_athlete: int | None = None):
    t0 = time.time()
    seg = load_segments()
    if max_segments_per_athlete:
        seg = seg.groupby("athlete", group_keys=False).apply(lambda s: s.sample(min(len(s), max_segments_per_athlete), random_state=0))
    print(f"{len(seg):,} segments, {seg.athlete.nunique()} athletes, {seg.groupby(['athlete', 'ride']).ngroups:,} sorties, "
          f"{seg.length_m.sum() / 1e3:,.0f} km")
    df, profiles = prepare(seg)
    n_all = len(df); df = tt_like(df)
    print(f"segments type CLM : {len(df):,} / {n_all:,} ({len(df) / n_all * 100:.0f} %)")
    prof = pd.DataFrame(profiles).T
    print(f"profils : CdA mediane {prof.cda.median():.3f} [{prof.cda.quantile(.1):.3f}-{prof.cda.quantile(.9):.3f}], "
          f"Crr mediane {prof.crr.median():.4f}, CP/kg mediane {(prof.cp_w / prof.mass_kg).median():.2f}")

    # (b) athletes jamais vus : 20 %
    rng = np.random.default_rng(0)
    athletes = np.array(sorted(df.athlete.unique())); rng.shuffle(athletes)
    held = set(athletes[: max(1, len(athletes) // 5)])
    df_seen, df_unseen = df[~df.athlete.isin(held)], df[df.athlete.isin(held)]
    # (a) sorties jamais vues parmi les athletes vus
    tr, te = split_rides(df_seen)
    model = train(tr)
    m_rides = evaluate(model, te); m_unseen = evaluate(model, df_unseen); tr_rides = tr[["athlete", "ride"]].drop_duplicates().sample(min(200, tr.ride.nunique()), random_state=0)
    m_train = evaluate(model, tr.merge(tr_rides, on=["athlete", "ride"]))
    for name, m in (("entrainement", m_train), ("sorties inconnues", m_rides), ("athletes inconnus", m_unseen)):
        print(f"{name:<18}: segments MAE log-v physique {m['mae_log_physics']:.4f} -> hybride {m['mae_log_hybrid']:.4f} ; "
              f"fenetres {m.get('window_km')} km ({m.get('n_windows')}): temps MAE physique {m.get('mae_pct_physics', np.nan):.2f} % "
              f"-> hybride {m.get('mae_pct_hybrid', np.nan):.2f} % (p90 {m.get('p90_pct_hybrid', np.nan):.2f} %, biais {m.get('bias_pct_hybrid', np.nan):+.2f} %)")
    # importance par permutation (rapide, sur un echantillon)
    from sklearn.inspection import permutation_importance
    s = te.sample(min(len(te), 20000), random_state=0)
    y = np.log(s.speed_ms) - np.log(s.v_phys)
    imp = permutation_importance(model.booster, s[model.features].to_numpy(float), y, n_repeats=3, random_state=0)
    order = np.argsort(-imp.importances_mean)
    print("importance :", ", ".join(f"{model.features[i]} {imp.importances_mean[i]:.4f}" for i in order[:8]))

    # modele final sur tout, sauvegarde
    final = train(df)
    final.profiles = profiles
    final.metrics = {"train": m_train, "held_out_rides": m_rides, "held_out_athletes": m_unseen,
                     "n_segments": int(len(df)), "n_athletes": int(df.athlete.nunique()), "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    final.save()
    MODEL_DIR.mkdir(exist_ok=True)
    json.dump(profiles, open(MODEL_DIR / "profiles.json", "w"), indent=1)
    json.dump(final.metrics, open(MODEL_DIR / "metrics.json", "w"), indent=1)
    print(f"modele sauve dans models/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
