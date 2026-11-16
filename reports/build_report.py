"""Assemble reports/tdf2026_stage16.html from the tables and figures produced
by the phase scripts (examples/phase1_course.py ... phase6_optimizer.py).

Run from the repo root:  python reports/build_report.py
"""

import base64
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports/tables"
FIG = ROOT / "reports/figures"


def img(name):
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{name}">'


def table(df, floatfmt="{:.1f}"):
    return df.to_html(index=False, border=0, classes="tbl", float_format=lambda v: floatfmt.format(v))


def fmt_t(s):
    return f"{int(s // 60)}'{s % 60:04.1f}\""


def main():
    sectors = pd.read_csv(TAB / "stage16_sectors.csv")
    loo = pd.read_csv(TAB / "stage16_loo.csv")
    riders = pd.read_csv(TAB / "stage16_riders.csv")
    strat = pd.read_csv(TAB / "stage16_strategies.csv")
    trade = pd.read_csv(TAB / "stage16_descent_tradeoff.csv")
    plan = pd.read_csv(TAB / "stage16_plan_evenepoel.csv")
    phys = pd.read_csv(ROOT / "data/processed/opendata_cp_wprime.csv")

    full = sectors[sectors.rider.isin(sectors.groupby("rider").size()[lambda s: s >= 3].index)]
    piv = full.pivot_table(index="rider", columns=["km_start", "km_end"], values="power_w").round(0)
    piv.columns = [f"km {a:.1f}-{b:.1f}" for a, b in piv.columns]
    piv = piv.reset_index()

    loo_tab = loo[["rider", "km", "t_real_s", "t_pred_s", "err_s", "err_pct_sector"]].rename(columns={
        "rider": "Rider", "km": "Checkpoint (km)", "t_real_s": "Real (s)", "t_pred_s": "Predicted (s)",
        "err_s": "Error (s)", "err_pct_sector": "Error (% of sector)"})
    mae_s, mae_pct = loo.err_s.abs().mean(), loo.err_pct_sector.abs().mean()
    by_cp = loo.groupby("km").agg(MAE_s=("err_s", lambda e: e.abs().mean()), bias_s=("err_s", "mean"),
                                  n=("err_s", "size")).reset_index().rename(columns={"km": "Checkpoint (km)"})

    strat_tab = strat[["rider", "mass_kg", "cp_w", "n_checkpoints", "real_s", "constant_w", "constant_s",
                       "optimal_s", "gain_vs_real_s", "gain_vs_constant_s"]].copy()
    for c in ("real_s", "constant_s", "optimal_s"):
        strat_tab[c] = strat_tab[c].map(fmt_t)
    strat_tab = strat_tab.rename(columns={
        "rider": "Rider", "mass_kg": "Mass (kg)", "cp_w": "CP fitted (W)", "n_checkpoints": "Checkpoints",
        "real_s": "Real (reconstructed)", "constant_w": "Constant (W)", "constant_s": "Constant time",
        "optimal_s": "Optimal time", "gain_vs_real_s": "Gain vs real (s)", "gain_vs_constant_s": "Gain vs constant (s)"})
    ev = strat[strat.rider == "Remco Evenepoel"].iloc[0]
    phys_ok = phys[(phys.fit_r2 > 0.98) & (phys.n_rides_power >= 20)]

    plan_tab = plan[["km_start", "km_end", "grade_pct", "power_w", "speed_kmh", "time_s", "wbal_end_kj", "braking"]].rename(
        columns={"km_start": "km start", "km_end": "km end", "grade_pct": "Grade (%)", "power_w": "Target (W)",
                 "speed_kmh": "Speed (km/h)", "time_s": "Time (s)", "wbal_end_kj": "W'bal end (kJ)", "braking": "Braking"})

    css = """
    body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#1F2937;line-height:1.5}
    h1{font-size:28px;margin-bottom:0} h2{margin-top:40px;border-bottom:2px solid #0F766E;padding-bottom:4px;font-size:21px}
    .sub{color:#6B7280;margin-top:4px} img{width:100%;border:1px solid #E5E7EB;border-radius:6px;margin:12px 0}
    .tbl{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0} .tbl th{background:#F3F4F6;text-align:left;padding:6px 8px}
    .tbl td{padding:5px 8px;border-top:1px solid #E5E7EB} .kpi{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}
    .kpi div{flex:1;min-width:180px;background:#F0FDFA;border-left:4px solid #0F766E;padding:10px 14px;border-radius:4px}
    .kpi b{display:block;font-size:22px} .note{background:#FFFBEB;border-left:4px solid #B45309;padding:10px 14px;border-radius:4px;margin:14px 0}
    .src{font-size:12px;color:#6B7280}
    """
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 16 pacing report</title><style>{css}</style></head><body>
<h1>Tour de France 2026, stage 16 ITT: pacing analysis</h1>
<p class="sub">Évian-les-Bains to Thonon-les-Bains, 26.1 km, 21 July 2026. Pacing Optimizer v2, generated from public data.</p>
<div class="kpi">
<div><b>{mae_s:.1f} s</b>mean error when predicting a withheld checkpoint ({mae_pct:.2f} % of sector time, n = {len(loo)})</div>
<div><b>{ev.gain_vs_real_s:+.0f} s</b>Evenepoel: optimal plan versus his reconstructed strategy</div>
<div><b>{ev.gain_vs_constant_s:+.0f} s</b>Evenepoel: optimal plan versus constant power at the same W'</div>
<div><b>{ev.cp_w:.0f} W</b>Evenepoel: CP implied by his splits ({ev.cp_w / ev.mass_kg:.2f} W/kg), under the stated assumptions</div>
</div>
<div class="note"><b>What is measured and what is assumed.</b> Measured: the GPX track, the hourly weather, the published
split times, start times and published rider masses. Assumed: CdA (0.21 m² for the reference rider, scaled by body surface
area), Crr 0.0032, bike mass 8.5 kg, W' 25 kJ, cornering limit (calibrated on this stage). No rider power was measured:
every watt in this report is inferred from times.</div>

<h2>1. Course</h2>
<p>GPX from cyclingstage.com, 26.075 km (-0.1 % versus the official 26.1 km), 472 m of climbing after smoothing
(official: about 500 m). The Côte de Larringes measures +398 m over 9.54 km in the GPX (4.2 %), against +403 m and 4.4 %
published. The true high point is at km 10.9 (807 m), 1.3 km after the categorised summit. Open-Meteo's 90 m terrain model
agrees within a 3 m bias. 37 segments of 50 m have a centre-line radius under 40 m, most of them in the descent between
km 14.5 and 17.</p>
{img("stage16_profile.png")}

<h2>2. Weather</h2>
<p>Open-Meteo archive at three points along the course, interpolated to each 50 m segment at each rider's passage time.
Around 17:30 CEST: 25 °C, 33 % humidity, 976 hPa at the start, wind 4.4 m/s from the NNE at 10 m, i.e. 2.2 m/s at rider
height (log profile, roughness 0.10 m). Air density 1.09 to 1.14 kg/m³ along the course. The climb has a slight
tailwind, the lakeside finish a slight headwind. An independent observation (bikeraceinfo: 26 °C, NNE 11 km/h at the
finish) agrees.</p>
{img("stage16_weather.png")}

<h2>3. Implied power per sector</h2>
<p>For each rider with intermediate splits, the constant power on each sector that reproduces their split with the
dynamic simulator (inertia, cornering, braking), using their own weather. These are estimates, not measurements, and
they move with the assumed CdA: ±0.02 m² changes the climb power by about ±10 W and the flat power by about ±45 W.</p>
{table(piv, "{:.0f}")}
<p class="src">Sources: cyclingstage.com (checkpoints, top 5 per checkpoint), bikeraceinfo.com (finish, Cattaneo's km 9.7),
bikeradar.com (start times), Wikipedia infoboxes (masses, with the cited origin recorded in data/processed).</p>

<h2>4. Validation: predicting a withheld checkpoint</h2>
<p>For each intermediate checkpoint, the model is refitted on the other checkpoints (constant power over the merged
sector) and predicts the withheld one. The error mixes the physics with the assumption of constant pacing across the
merged sector, which is why the km 9.7 checkpoint (merging the second half of the climb with the descent) is the
hardest. Mean absolute error: <b>{mae_s:.1f} s</b>, <b>{mae_pct:.2f} %</b> of the sector time.</p>
{table(by_cp)}
{table(loo_tab)}
<p>The cornering limit was chosen with this test: with a lateral acceleration limit of 8 m/s² (0.82 g, the dry-tyre
adhesion limit) the error is minimal and the km 9.7 bias vanishes; at 6 m/s² the bias is -10 s, and without any cornering
limit the error triples (18.7 s). The descent power and the cornering limit are not separately identifiable from split
times, as the table below shows for Evenepoel: the descent sector only pins down a combination of the two.</p>
{table(trade.rename(columns={"a_lat": "a_lat (m/s²)"}), "{:.0f}")}

<h2>5. Physiology check (GoldenCheetah OpenData)</h2>
<p>{len(phys_ok)} amateur and trained athletes (CC0 data) give CP between {phys_ok.cp_wkg.min():.1f} and
{phys_ok.cp_wkg.max():.1f} W/kg and W' between {phys_ok.w_prime_j.min() / 1000:.0f} and {phys_ok.w_prime_j.max() / 1000:.0f} kJ
(median {phys_ok.w_prime_j.median() / 1000:.0f} kJ, {(phys_ok.w_prime_j / phys_ok.mass_kg).median():.0f} J/kg). On rides that
contain a season-best 3 to 20 min effort, Skiba's W'bal reaches a median minimum of -31 % of W': the model overstates
W' depletion, so a CP fitted with "W'bal = 0 at exhaustion" is conservative. The 25 kJ assumed for the pros sits at the
upper end of the amateur range, consistent with the 36 kJ of the strongest athlete in the sample (6.0 W/kg).</p>
{img("opendata_physiology.png")}

<h2>6. Strategies: real, constant and optimal</h2>
<p>For each rider: CP fitted so that their reconstructed effort empties W' exactly; "constant" is the highest constant
power that keeps W'bal positive everywhere (the binding point is the summit, not the finish); "optimal" is the block
plan (500 m blocks, sections cut at km 9.64, 10.9 and 17.0) that minimises time under the same constraint. Riders without
intermediate splits only have a finish time, so their real strategy is unknown.</p>
{table(strat_tab)}
{img("stage16_strategies.png")}
<p>Evenepoel's plan, block by block (target power; applied power is lower in braking zones):</p>
{table(plan_tab)}

<h2>7. Limits</h2>
<ul>
<li>Intermediate splits are available for the top 5 at each checkpoint only. ProCyclingStats and the official site hold
the full set but forbid automated access; the pipeline accepts a manual export.</li>
<li>CdA, Crr, bike mass and W' are assumptions. The implied CP values (6.4 to 7.0 W/kg) are high, which suggests the
reference CdA of 0.21 is slightly optimistic or the descent costs are still underestimated.</li>
<li>The GPX is a reconstruction by cyclingstage.com, not a race-day recording: corner radii in the descent are
approximate, hence the calibrated rather than physical cornering limit.</li>
<li>Weather is hourly model data on a 1 to 10 km grid; gusts and local shelter along the lake are not resolved.</li>
<li>One stage so far. The multi-stage validation (phase 7) is pending the choice of courses.</li>
</ul>
<p class="src">Code: pacing_model.py (v1 engine), course_io.py, weather.py, dynamics.py, calibration.py, physiology.py,
optimizer.py, app.py. Data and sources in data/processed/README.md.</p>
</body></html>"""
    out = ROOT / "reports/tdf2026_stage16.html"
    out.write_text(html, encoding="utf-8")
    print(f"written {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
