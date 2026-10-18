# Pacing Optimizer

Power pacing for cycling time trials and triathlon bike legs. Give it a course and a rider profile, and it returns the power for each segment that minimises race time while spending exactly the same anaerobic energy (W') as a constant-power effort.

## Demo result

Hilly 10 km time trial, rider with CP 300 W, W' 20 kJ, 76 kg system mass, CdA 0.28:

| Strategy | Power | Time | Avg speed |
|---|---|---|---|
| Constant power | 318 W | 18:19.3 | 32.7 km/h |
| Optimised pacing | 100 to 453 W by segment | 17:34.7 | 34.1 km/h |

**Gain: 44.6 s (4.1 %) for the same W' budget.** The optimiser pushes above CP on the climbs and backs off on the descents, as the pacing literature predicts.

| Segment | Grade | Optimised power | Speed |
|---|---|---|---|
| 1 | 0.0 % | 300 W | 41.3 km/h |
| 2 | +6.0 % | 453 W | 27.6 km/h |
| 3 | -4.0 % | 100 W | 50.5 km/h |
| 4 | 0.0 % | 300 W | 41.3 km/h |
| 5 | +8.0 % | 344 W | 18.1 km/h |
| 6 | -5.0 % | 101 W | 55.4 km/h |
| 7 | +2.0 % | 364 W | 36.2 km/h |
| 8 | 0.0 % | 300 W | 41.3 km/h |

## How it works

1. **Power to speed.** The road cycling power model of Martin et al. (1998) converts power into steady-state speed on each segment, from gravity, rolling resistance and aerodynamic drag (wind included).
2. **Physiological limit.** The Critical Power / W' model tracks the anaerobic reserve along the course with Skiba's W' balance: linear depletion above CP, exponential recovery below.
3. **Optimisation.** SciPy's SLSQP minimises total time under one constraint: W' balance stays at or above zero everywhere. The benchmark is the constant power that empties W' exactly at the finish line.

## Files

- `pacing_model.py`: the engine (`Rider`, `Environment`, `Course`, `simulate`, `constant_power_target`, `optimize_pacing`). Runs on its own or as an import.
- `app.py`: Streamlit dashboard. Sliders for the rider and conditions, an editable course table, and charts for the power plan and the W' balance.
- `requirements.txt`: dependencies.

## Run it

```bash
pip install -r requirements.txt
python pacing_model.py      # demo in the terminal
streamlit run app.py        # interactive dashboard
```

## Load a real course

```python
from pacing_model import Course, Rider, Environment, optimize_pacing

course = Course.from_elevation(distance_m, elevation_m)  # from a GPX or CSV export
result = optimize_pacing(course, Rider(cp=320, w_prime=22000), Environment())
print(result.time_str, result.powers.round())
```

## Next steps

- GPX import and calibration on real race files
- CP and W' fitted from an athlete's power data
- Cornering, braking, speed-dependent CdA and directional wind per segment
- Export of a pacing plan for a bike computer

## References

- Martin, J. C., Milliken, D. L., Cobb, J. E., McFadden, K. L., Coggan, A. R. (1998). *Validation of a mathematical model for road cycling power.* Journal of Applied Biomechanics.
- Monod, H., Scherrer, J. (1965). *The work capacity of a synergic muscular group.* Ergonomics.
- Skiba, P. F. et al. (2012, 2015). W' balance model.

## Author

Vadim Digonnet, data scientist for endurance sport.
[LinkedIn](https://www.linkedin.com/in/vadim-digonnet/) · v.digonnet.pro@gmail.com
