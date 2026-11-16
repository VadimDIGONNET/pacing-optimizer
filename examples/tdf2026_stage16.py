"""Cas reel : CLM de la 16e etape du Tour de France 2026 (Evian - Thonon, 26,1 km).

Vainqueur : Remco Evenepoel en 32'19". Splits officiels :
    km 4,8  -> 8'08"
    km 9,7  -> 15'42"   (sommet de la cote de Larringes)
    km 18,1 -> 23'30"
    km 26,1 -> 32'19"

Demarche :
  1. On reconstruit le parcours a partir du profil publie (montee de 9,6 km a
     4,4 %, descente d'environ 7 km, final plat le long du lac).
  2. On inverse le modele physique pour estimer la puissance d'Evenepoel sur
     chaque secteur a partir de ses temps intermediaires reels.
  3. On cale CP pour que cette strategie vide exactement W' (effort a bloc).
  4. A physiologie identique, on cherche la repartition optimale et on mesure
     l'ecart avec ce qu'il a reellement fait.

Les parametres de l'athlete (masse, CdA, W') sont des hypotheses publiques
plausibles, pas des mesures. Lancer depuis la racine du repo :
    python examples/tdf2026_stage16.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacing_model import (  # noqa: E402
    Rider, Environment, Course, simulate, speed_from_power, optimize_pacing,
)

# Parcours reconstruit depuis le profil publie (altitude depart ~385 m,
# sommet ~805 m, arrivee ~394 m). Le segment 3 est le petit plateau sommital.
SEGMENTS = [
    (4800, 0.044),    # km 0    -> 4,8   montee (1er pointage)
    (4840, 0.044),    # km 4,8  -> 9,64  montee jusqu'au sommet
    (300, 0.000),     # plateau
    (6700, -0.060),   # descente vers le lac
    (1460, -0.001),   # debut du plat jusqu'au 3e pointage (km 18,1)
    (8000, -0.001),   # final plat jusqu'a l'arrivee
]

# Splits officiels (distance cumulee en m, temps cumule en s)
SPLITS = [(4800, 488), (9700, 942), (18100, 1410), (26100, 1939)]

# Secteurs de pointage exprimes en indices de segments
SECTORS = [[0], [1], [2, 3, 4], [5]]


def fmt(t):
    return f"{int(t // 60)}'{t % 60:04.1f}\""


def sector_power(seg_idx, target_time, course, rider, env):
    """Puissance constante qui fait parcourir les segments en target_time."""
    def gap(p):
        t = sum(course.length_m[i] / speed_from_power(p, course.grade[i], rider, env)
                for i in seg_idx)
        return t - target_time
    return brentq(gap, 50, 1500, xtol=0.1)


def analyse():
    """Reconstruit l'effort reel, cale CP et optimise. Renvoie un dict.

    Separe du main() pour que les tests de non-regression puissent verifier
    les chiffres sans passer par l'affichage.
    """
    course = Course.from_segments(SEGMENTS)
    # 26 C, ~400-800 m d'altitude : densite de l'air ~1,11 kg/m3.
    # Vent leger (11 km/h NNE) neglige faute de trace GPS precis.
    env = Environment(rho=1.11, wind_ms=0.0)
    rider = Rider(mass_rider=61.0, mass_bike=8.0, cda=0.21, crr=0.0032,
                  w_prime=25000.0, p_max=750.0, p_min=100.0)

    # 1. Puissances estimees par secteur a partir des splits reels
    sector_times = np.diff([0] + [t for _, t in SPLITS])
    actual = np.zeros(course.n)
    for seg_idx, t in zip(SECTORS, sector_times):
        actual[seg_idx] = sector_power(seg_idx, t, course, rider, env)

    # 2. Calage de CP : la strategie reelle vide exactement W'
    def min_wbal(cp):
        r = Rider(**{**rider.__dict__, "cp": cp})
        return simulate(actual, course, r, env).min_wbal_j
    rider.cp = brentq(min_wbal, 250, 600, xtol=0.1)

    real = simulate(actual, course, rider, env)
    # Plateau, descente et debut du plat restent a la puissance reelle : le
    # modele ignore virages et freinages, il surestimerait le gain possible.
    fixed = set(SECTORS[2])
    bounds = [(actual[i], actual[i]) if i in fixed else (rider.p_min, rider.p_max)
              for i in range(course.n)]
    best = optimize_pacing(course, rider, env, bounds=bounds)
    return {"course": course, "rider": rider, "env": env,
            "actual": actual, "real": real, "best": best}


def main():
    res = analyse()
    course, rider, env = res["course"], res["rider"], res["env"]
    actual, real, best = res["actual"], res["real"], res["best"]
    gain = real.time_s - best.time_s

    print("CLM Tour de France 2026, etape 16 (Evian - Thonon, 26,1 km)")
    print(f"Hypotheses : {rider.mass_rider:.0f} kg + {rider.mass_bike:.0f} kg, "
          f"CdA {rider.cda}, W' {rider.w_prime / 1000:.0f} kJ, rho {env.rho}")
    print(f"CP calee sur l'effort reel : {rider.cp:.0f} W "
          f"({rider.cp / rider.mass_rider:.2f} W/kg)\n")

    print("Secteur              | P estimee | P optimale")
    labels = ["km 0-4,8 (montee)", "km 4,8-9,7 (montee)",
              "km 9,7-18,1 (desc.)", "km 18,1-26,1 (plat)"]
    for lab, seg_idx in zip(labels, SECTORS):
        w = course.length_m[seg_idx]
        p_real = np.average(actual[seg_idx], weights=w)
        p_opt = np.average(best.powers[seg_idx], weights=w)
        print(f"{lab:<20} | {p_real:6.0f} W  | {p_opt:6.0f} W")

    print("\nPointage | Reel     | Optimise")
    cum_real = np.cumsum(real.seg_time_s)
    cum_best = np.cumsum(best.seg_time_s)
    ends = [0, 1, 4, 5]  # dernier segment de chaque secteur
    for (d, _), i in zip(SPLITS, ends):
        print(f"km {d / 1000:4.1f}  | {fmt(cum_real[i])} | {fmt(cum_best[i])}")

    print(f"\nTemps reel reconstruit : {fmt(real.time_s)}")
    print(f"Temps optimal          : {fmt(best.time_s)}")
    print(f"Gain potentiel         : {gain:.1f} s ({gain / real.time_s * 100:.2f} %)")


if __name__ == "__main__":
    main()
