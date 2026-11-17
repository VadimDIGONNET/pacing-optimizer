# Model card: hybrid pacing model

**Purpose.** Predict the time a cyclist needs on a course segment given a power plan, a physical profile (mass, CdA, Crr) and the course geometry, more accurately than the steady-state physics alone. Used by the dashboard to predict finish times and intervals, next to the dynamic physics simulator that produces the power plan.

**Trained on.** 6,131,613 segments of 200 m from 234 GoldenCheetah OpenData athletes (DOI 10.17605/OSF.IO/6HFPZ, licence CC0 1.0), 50,644 rides, 2.36 million km, power meter at 1 Hz. Only time-trial-like segments are used (continuous pedalling, steady power, above 100 W and 5 m/s). Athletes are amateurs and trained cyclists: CP from 2.4 to 6.0 W/kg, CdA fitted between 0.32 and 0.60 m² (road position, not time-trial position).

**Inputs (all known when planning).** Mean power, grade, altitude, entry speed, power variability, pedalling fraction, elapsed time, rider mass, CdA, Crr, W/kg, physics speed, power over CP. The athlete identity is not an input.

**Architecture.** Martin et al. (1998) steady-state speed with per-athlete CdA and Crr, plus a scikit-learn HistGradientBoostingRegressor on the residual log(observed speed) - log(physics speed). 400 iterations, 63 leaves, 200 samples per leaf.

**Validation (never seen in training).**

| Test set | 10 km windows | Physics only | Hybrid | p90 | Bias |
|---|---|---|---|---|---|
| Unseen rides of known athletes | 1064 | 9.57 % | 1.17 % | 2.70 % | +0.39 % |
| Unseen athletes | 1146 | 9.87 % | 1.74 % | 3.43 % | +0.43 % |
| Tour de France 2026 stage 16, leave-one-checkpoint-out (7 pros) | 15 checkpoints | 5.3 s (0.45 %) | 6.2 s (0.53 %) | | +4.8 s |

**Known limits.** No wind and no cornering in the learned model (the physics simulator handles both). Professional riders sit at the edge of the training domain (CP above 6 W/kg, CdA below 0.3 m²); the dashboard flags it. Air density in training uses the ISA altitude profile, not the weather of the day. Segment time is predicted from mean power: a plan with the same mean but more variability rides slower in reality, which the power_sd input captures only within the range seen in training.

**Reproduce.** `python dataset.py 250` (streams OSF archives, deletes them after processing), `python examples/train_hybrid.py`, `python examples/validate_hybrid_pro.py`. Trained on 2026-11-17 02:41.
