"""Optimiseur de pacing pour sports d'endurance (cyclisme, triathlon).

Le moteur repose sur deux modeles publies et largement utilises :

  - Modele de puissance mecanique (Martin et al., 1998) : relie la puissance
    developpee a la vitesse, en tenant compte de la gravite (pente), de la
    resistance au roulement et de la trainee aerodynamique.

  - Modele Critical Power / W' (Monod & Scherrer 1965 ; bilan de W' de
    Skiba et al. 2012/2015) : CP est la puissance theoriquement soutenable
    indefiniment, W' la reserve d'energie disponible au-dessus de CP. Le
    "bilan de W'" se vide quand P > CP et se recharge quand P < CP.

A partir de la, on compare deux strategies a budget energetique (W') identique :
  - puissance constante (le reflexe naturel) ;
  - pacing optimise segment par segment (minimise le temps total).

Le resultat attendu, conforme a la litterature : pousser dans les montees,
relacher dans les descentes, pour gagner du temps sans depenser plus d'energie.

Dependances : numpy, scipy. Lancer `python pacing_model.py` pour la demo.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, brentq

# Constantes physiques
G = 9.81                 # acceleration de la pesanteur (m/s^2)
RHO_SEA_LEVEL = 1.225    # densite de l'air au niveau de la mer, 15 C (kg/m^3)
DRIVETRAIN_EFF = 0.976   # rendement de la transmission


@dataclass
class Rider:
    """Parametres de l'athlete et de sa machine.

    cp (W) et w_prime (J) se mesurent via un test (ex. 3 min all-out, ou
    plusieurs efforts maximaux de durees differentes). cda et crr peuvent
    etre estimes ou mesures (test de deceleration, capteur de puissance).
    """
    mass_rider: float = 68.0   # masse de l'athlete (kg)
    mass_bike: float = 8.0     # masse du velo + equipement (kg)
    crr: float = 0.004         # coefficient de resistance au roulement
    cda: float = 0.28          # produit Cd*A, trainee aerodynamique (m^2)
    cp: float = 300.0          # Critical Power (W)
    w_prime: float = 20000.0   # reserve anaerobie W' (J)
    p_max: float = 600.0       # puissance plafond soutenable sur un segment (W)
    p_min: float = 100.0       # puissance plancher (roue libre exclue) (W)

    @property
    def mass(self) -> float:
        return self.mass_rider + self.mass_bike


@dataclass
class Environment:
    """Conditions exterieures. wind_ms > 0 = vent de face."""
    rho: float = RHO_SEA_LEVEL
    wind_ms: float = 0.0


@dataclass
class Course:
    """Parcours discretise en segments de longueur et de pente constantes.

    length_m : longueur de chaque segment (m).
    grade    : pente de chaque segment (fraction ; 0.06 = 6 %).
    """
    length_m: np.ndarray
    grade: np.ndarray

    def __post_init__(self):
        self.length_m = np.asarray(self.length_m, dtype=float)
        self.grade = np.asarray(self.grade, dtype=float)

    @property
    def total_km(self) -> float:
        return float(self.length_m.sum() / 1000.0)

    @property
    def n(self) -> int:
        return len(self.length_m)

    @classmethod
    def from_segments(cls, segments) -> "Course":
        """segments : iterable de couples (longueur_m, pente)."""
        length, grade = zip(*segments)
        return cls(np.array(length, float), np.array(grade, float))

    @classmethod
    def from_elevation(cls, distance_m, elevation_m) -> "Course":
        """Construit le parcours a partir d'un profil (distance, altitude).

        Les pentes sont calculees entre points consecutifs. Pratique pour
        charger un vrai parcours exporte d'un GPS ou d'un fichier GPX/CSV.
        """
        distance_m = np.asarray(distance_m, dtype=float)
        elevation_m = np.asarray(elevation_m, dtype=float)
        seg_len = np.diff(distance_m)
        seg_grade = np.diff(elevation_m) / seg_len
        return cls(seg_len, seg_grade)

    @classmethod
    def rolling_demo(cls) -> "Course":
        """Contre-la-montre vallonne de 10 km servant de demonstration."""
        return cls.from_segments([
            (1500, 0.00), (1000, 0.06), (1500, -0.04), (1000, 0.00),
            (1500, 0.08), (1000, -0.05), (1500, 0.02), (1000, 0.00),
        ])


@dataclass
class PacingResult:
    """Resultat d'une simulation : strategie de puissance et ses consequences."""
    powers: np.ndarray       # puissance par segment (W)
    time_s: float            # temps total (s)
    seg_time_s: np.ndarray   # temps par segment (s)
    speed_ms: np.ndarray     # vitesse par segment (m/s)
    wbal_j: np.ndarray       # bilan de W' a la fin de chaque segment (J)
    min_wbal_j: float        # bilan de W' minimal atteint (doit rester >= 0)
    course: Course

    @property
    def avg_power(self) -> float:
        return float(np.mean(self.powers))

    @property
    def avg_speed_kmh(self) -> float:
        return self.course.total_km / (self.time_s / 3600.0)

    @property
    def time_str(self) -> str:
        return f"{int(self.time_s // 60)}:{self.time_s % 60:04.1f}"


def speed_from_power(power: float, grade: float, rider: Rider, env: Environment) -> float:
    """Vitesse d'equilibre (m/s) pour une puissance et une pente donnees.

    Resout numeriquement l'equation de Martin et al. (1998) :
        P_rider = [ (m g (sin + Crr cos)) v + 0.5 rho CdA (v + vent)^2 v ] / rendement
    On cherche la racine en v (fonction monotone croissante de la puissance).
    """
    theta = np.arctan(grade)
    f_roll_grav = rider.mass * G * (np.sin(theta) + rider.crr * np.cos(theta))

    def net(v):
        f_aero = 0.5 * env.rho * rider.cda * (v + env.wind_ms) ** 2
        return (f_roll_grav + f_aero) * v / DRIVETRAIN_EFF - power

    lo, hi = 0.01, 30.0
    # En forte montee avec faible puissance, meme la vitesse plancher "coute"
    # deja plus que la puissance fournie : on renvoie cette vitesse plancher.
    if net(lo) > 0:
        return lo
    return brentq(net, lo, hi, xtol=1e-4)


def simulate(powers, course: Course, rider: Rider, env: Environment) -> PacingResult:
    """Deroule une strategie de puissance sur le parcours et renvoie le detail.

    Le bilan de W' suit le modele differentiel de Skiba : depletion lineaire
    au-dessus de CP, recuperation exponentielle en dessous.
    """
    powers = np.asarray(powers, dtype=float)
    wbal = rider.w_prime
    min_wbal = rider.w_prime
    seg_time, speeds, wbal_track = [], [], []

    for p, length, grade in zip(powers, course.length_m, course.grade):
        v = speed_from_power(p, grade, rider, env)
        t = length / v
        if p > rider.cp:
            wbal -= (p - rider.cp) * t
        else:
            wbal = rider.w_prime - (rider.w_prime - wbal) * np.exp(
                -((rider.cp - p) / rider.w_prime) * t
            )
        min_wbal = min(min_wbal, wbal)
        seg_time.append(t)
        speeds.append(v)
        wbal_track.append(wbal)

    return PacingResult(
        powers=powers,
        time_s=float(np.sum(seg_time)),
        seg_time_s=np.array(seg_time),
        speed_ms=np.array(speeds),
        wbal_j=np.array(wbal_track),
        min_wbal_j=float(min_wbal),
        course=course,
    )


def constant_power_target(course: Course, rider: Rider, env: Environment) -> float:
    """Puissance constante qui epuise exactement W' a l'arrivee.

    C'est le benchmark "pacing naif" : meme budget energetique que la version
    optimisee, mais reparti uniformement.
    """
    def end_wbal(p):
        return simulate(np.full(course.n, p), course, rider, env).wbal_j[-1]

    return brentq(end_wbal, rider.cp, rider.p_max, xtol=0.1)


def optimize_pacing(course: Course, rider: Rider, env: Environment) -> PacingResult:
    """Repartition de puissance minimisant le temps, a W' identique.

    Probleme d'optimisation sous contrainte : minimiser le temps total, en
    maintenant le bilan de W' positif partout (on ne peut pas depenser plus
    d'energie anaerobie qu'on n'en possede).
    """
    p0 = constant_power_target(course, rider, env)
    x0 = np.full(course.n, p0)
    bounds = [(rider.p_min, rider.p_max)] * course.n
    constraints = [{
        "type": "ineq",
        "fun": lambda p: simulate(p, course, rider, env).min_wbal_j,
    }]

    res = minimize(
        lambda p: simulate(p, course, rider, env).time_s,
        x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-6},
    )
    return simulate(res.x, course, rider, env)


if __name__ == "__main__":
    course = Course.rolling_demo()
    rider = Rider()
    env = Environment()

    p_const = constant_power_target(course, rider, env)
    naive = simulate(np.full(course.n, p_const), course, rider, env)
    optimal = optimize_pacing(course, rider, env)
    gain = naive.time_s - optimal.time_s

    print(f"Parcours : {course.total_km:.1f} km, {course.n} segments")
    print(f"Athlete  : CP={rider.cp:.0f} W, W'={rider.w_prime / 1000:.1f} kJ, "
          f"masse {rider.mass:.0f} kg, CdA {rider.cda}")
    print()
    print(f"Puissance constante : {p_const:.0f} W  ->  "
          f"{naive.time_str}  ({naive.avg_speed_kmh:.1f} km/h)")
    print(f"Pacing optimise     :              ->  "
          f"{optimal.time_str}  ({optimal.avg_speed_kmh:.1f} km/h)")
    print()
    print(f"GAIN : {gain:.1f} s  ({gain / naive.time_s * 100:.2f} %)")
    print()
    print("Seg |  pente |  P opt  |  v opt")
    for i in range(course.n):
        print(f" {i + 1:>2} | {course.grade[i] * 100:+5.1f}% | "
              f"{optimal.powers[i]:5.0f} W | {optimal.speed_ms[i] * 3.6:5.1f} km/h")
