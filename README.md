# Pacing Optimizer

Power pacing for cycling time trials and triathlon bike legs, validated on real race data. Give it a GPX file, a date, and a rider profile; it returns a block-by-block power plan that minimises time while keeping the anaerobic reserve (W') above zero everywhere, with real air density, wind, corners and braking.

Version 2 replays the Tour de France 2026 stage 16 time trial (Évian to Thonon, 26.1 km) from public data and measures how well the model predicts withheld split times. Full report: `reports/tdf2026_stage16.html`.

## What it does

1. **Course from GPX** (`course_io.py`). Geodesic distance, Savitzky-Golay elevation smoothing (250 m window), 50 m segments with grade, heading and curvature. Corners are detected from the net heading change over 40 m, which is robust to GPS noise. An Open-Meteo terrain query serves as an independent elevation check.
2. **Real weather** (`weather.py`). Open-Meteo archive, cached. Humid-air density from temperature, pressure and humidity, corrected to each segment's altitude. Wind brought from 10 m to rider height with a log profile (roughness 0.10 m, factor 0.50), then projected on each segment's heading.
3. **Dynamic physics** (`dynamics.py`). Integration of the equation of motion with inertia (1.5 kg of rotating mass), cornering speed limits on the racing line, braking by a backward pass, and the applied power used for the W' balance. Numba kernel: 0.8 ms per simulation of a 26 km course.
4. **Calibration and validation** (`calibration.py`). Sector powers inverted from published splits, CP fitted so that the real effort empties W', leave-one-checkpoint-out validation with errors in seconds and in percent of sector time.
5. **Physiology** (`physiology.py`). Mean-maximal power curves, CP and W' by the two-parameter model, and a test of Skiba's W'bal on GoldenCheetah OpenData (CC0).
6. **Optimiser v2** (`optimizer.py`). Equilibrium block optimiser for the structure, then SLSQP polish on the dynamic simulator with a smoothing penalty. Under 20 s for a 26 km course at 500 m blocks.
7. **Streamlit app** (`app.py`). GPX upload or stage 16 demo, date and time for the weather, rider profile (manual or fitted from a power file), map with target power and wind arrows, strategy comparison, CSV and .zwo export.

The v1 engine (`pacing_model.py`: `Rider`, `Environment`, `Course`, `simulate`, `optimize_pacing`) is unchanged and still used for the initial plan. `Environment` now also accepts per-segment arrays.

## Results on Tour de France 2026, stage 16

Data: GPX from cyclingstage.com, hourly weather from Open-Meteo, splits at km 4.8, 9.7, 18.1 and the finish (top 5 per checkpoint and top 20 at the finish, from cyclingstage.com and bikeraceinfo.com), start times from bikeradar.com, published masses from Wikipedia infoboxes with their cited origin. Every real number has a `source_url` in `data/processed/`.

Course check: 26.075 km in the GPX against 26.1 official, 472 m of climbing against about 500 m announced, Côte de Larringes +398 m at 4.2 % against +403 m at 4.4 % published.

Weather at 17:30 CEST: 25 °C, 33 % humidity, 976 hPa, wind 4.4 m/s from the NNE at 10 m. Air density 1.09 to 1.14 kg/m³ along the course.

**Validation.** Each intermediate checkpoint is predicted after refitting on the others, for the riders with splits (Evenepoel, Pogačar, Skjelmose, Seixas, del Toro, Lipowitz, Cattaneo):

| Checkpoint | Mean absolute error | Bias | n |
|---|---|---|---|
| km 4.8 | 4.4 s | +4.2 s | 5 |
| km 9.7 | 6.7 s | +0.2 s | 6 |
| km 18.1 | 2.0 s | +1.1 s | 4 |
| All | **5.3 s (0.45 % of sector time)** | | 15 |

The error includes the assumption of constant power over the merged sector, not only the physics. The cornering limit (8 m/s², 0.82 g) was chosen because it minimises this error; at 6 m/s² the km 9.7 bias is -10 s and without corners the error triples.

**Strategies** (CP fitted per rider, same W' budget, 500 m blocks):

| Rider | Real (reconstructed) | Constant power | Optimal | Gain vs real | Gain vs constant |
|---|---|---|---|---|---|
| Evenepoel | 32'19.0" | 32'16.3" (451 W) | 32'02.4" | 16.6 s | 13.9 s |
| Pogačar | 32'47.0" | 32'36.2" (459 W) | 32'21.9" | 25.1 s | 14.3 s |
| Skjelmose | 33'23.0" | 33'18.4" (438 W) | 33'03.4" | 19.6 s | 15.0 s |

The optimal plan climbs at about the real power, eases to 370 to 390 W in the descent where corners cap the speed anyway, and spends the recovered W' on the lakeside flat. Evenepoel paced closer to that than Pogačar, which accounts for about a third of his 28 s margin in the model. Implied powers are estimates derived from times under the stated assumptions, never measured watts. Computation: about 30 s per rider at 500 m blocks.

## Assumptions, stated

| Parameter | Value | Why |
|---|---|---|
| CdA | 0.21 m² for a 63 kg, 1.71 m rider, scaled by DuBois body surface area | Middle of the 0.19 to 0.23 range of elite TT positions; ±0.02 moves the climb power by ±10 W and the flat power by ±45 W |
| Crr | 0.0032 | TT tubulars on dry asphalt |
| Bike mass | 8.5 kg | UCI minimum plus disc wheel and extensions |
| W' | 25 kJ | Upper end of the OpenData range (median 17 kJ, strongest athlete 36 kJ) |
| Lateral acceleration limit | 8 m/s² (calibrated), 6 m/s² default | Dry-tyre adhesion limit; descent power and cornering limit are not separately identifiable from splits |
| Road width for the racing line | 6 m | Closed road |
| Wind roughness length | 0.10 m | "Roughly open" terrain; 0.03 gives a factor 0.60, 0.25 gives 0.38 |

## Physiology check

15 GoldenCheetah OpenData athletes (CC0 1.0, verified through the OSF API): CP 2.4 to 6.0 W/kg, W' 6 to 37 kJ. On rides containing a season-best 3 to 20 min effort, Skiba's W'bal reaches a median minimum of -31 % of W', so a CP fitted with "W'bal = 0 at exhaustion" is conservative.

## Limits

- Full intermediate splits exist for all riders on ProCyclingStats and on the official site, but both exclude automated access. The pipeline reads a manual export placed in `data/raw/`.
- The GPX is a reconstruction, not a race recording. Corner radii in the descent are approximate, which is why the cornering limit is calibrated rather than physical.
- Implied CP values of 6.4 to 7.0 W/kg are high, which points to a slightly optimistic reference CdA or to residual descent cost.
- One stage validated so far. Candidate stages for the multi-race validation are listed in `reports/phase7_candidates.md`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # 44 tests, about 1 minute
streamlit run app.py                    # dashboard with the stage 16 demo
python examples/phase1_course.py        # course checks and profile figure
python examples/phase2_weather.py       # weather table and figure
python examples/phase4_calibration.py   # sector powers, validation, sensitivity
python examples/phase5_physiology.py    # OpenData CP, W', W'bal test
python examples/phase6_optimizer.py     # real, constant and optimal strategies
python reports/build_report.py          # reports/tdf2026_stage16.html
```

The OpenData sample (16 athlete archives, 214 MB) is not committed; `examples/phase5_physiology.py` documents the download from OSF.

## Load your own course

```python
from course_io import load_course
from weather import rider_weather, segment_environment
from pacing_model import Rider
from dynamics import DynamicsParams
from optimizer import optimize_dynamic, plan_table

seg = load_course("my_race.gpx")
env, weather = segment_environment(seg, rider_weather(seg, "2026-07-21", "17:13:00", elapsed_s=900))
rider = Rider(mass_rider=70, mass_bike=8.5, cda=0.23, crr=0.0032, cp=350, w_prime=20000, p_max=700, p_min=0)
plan = optimize_dynamic(seg, rider, env, DynamicsParams(), block_m=500)
print(plan.result.time_str, plan.gain_s)
print(plan_table(plan, seg))
```

## Files

```
pacing_model.py     v1 engine, API unchanged
course_io.py        GPX, smoothing, heading, curvature, corners
weather.py          Open-Meteo, air density, wind per segment
dynamics.py         dynamic simulator (numba kernel)
calibration.py      split inversion, CP fit, leave-one-out validation
physiology.py       MMP curves, CP/W' fit, W'bal test
optimizer.py        block optimiser v2
app.py              Streamlit v2 (v1 dashboard kept in examples/app_v1.py)
examples/           phase scripts and the original stage 16 replay
data/raw/           cached downloads, never edited
data/processed/     cleaned data with sources (see its README)
reports/            figures, tables, HTML report, phase 7 candidates
tests/              44 tests
PLAN.md             specification (French)
```

## References

- Martin, J. C. et al. (1998). Validation of a mathematical model for road cycling power. *Journal of Applied Biomechanics*.
- Monod, H., Scherrer, J. (1965). The work capacity of a synergic muscular group. *Ergonomics*.
- Skiba, P. F. et al. (2012, 2015). W' balance model.
- Alduchov, O., Eskridge, R. (1996). Improved Magnus form approximation of saturation vapor pressure.
- Wieringa, J. (1992). Updating the Davenport roughness classification.
- GoldenCheetah OpenData, DOI 10.17605/OSF.IO/6HFPZ, CC0 1.0.

## Author

Vadim Digonnet, data scientist for endurance sport.
[LinkedIn](https://www.linkedin.com/in/vadim-digonnet/) · v.digonnet.pro@gmail.com
