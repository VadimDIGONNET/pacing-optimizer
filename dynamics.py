"""Simulateur dynamique (phase 3) : inertie, virages, freinage.

Au lieu d'une vitesse d'equilibre par segment (pacing_model.simulate), on
integre l'equation du mouvement le long de la distance :

    m_eff * v * dv/ds = P * eta / v - F_res(v, pente, vent)

avec m_eff = masse totale + masse equivalente des roues en rotation
(I / r^2, voir Martin et al. 1998 : I = 0,14 kg m^2 pour deux roues,
r = 0,311 m, soit environ 1,45 kg ; on prend 1,5 kg par defaut).

Vitesse maximale en virage : v_max = sqrt(a_lat * R), ou R est le rayon de la
trajectoire tendue (racing_line_radius : corde interieure, bords exterieurs
en entree et en sortie, largeur de route utilisable 6 m par defaut, route
fermee). Limite d'acceleration laterale a_lat = 6,0 m/s^2 (0,61 g) par
defaut. Justification : les angles d'inclinaison maximaux observes chez les
professionnels sur route seche (35 a 40 degres) correspondent a 0,7 a 0,85 g,
limite d'adherence pneu/asphalte sec ; on retient une valeur en dessous,
car tous les virages ne sont pas pris a la limite (visibilite, devers,
gravillons). La valeur reste un parametre : la phase 4 montre qu'elle n'est
pas identifiable separement de la puissance en descente a partir des seuls
pointages, et reporte le compromis entre les deux.

Freinage : deceleration maximale a_brake = 4,0 m/s^2 (0,41 g), limite par le
basculement vers l'avant sur un velo de course (centre de gravite haut et
empattement court). On construit d'abord le profil de vitesse limite en
remontant le parcours (passe arriere), puis on integre vers l'avant en
plafonnant la vitesse par ce profil : quand le plafond mord, le coureur
cesse de pedaler (puissance appliquee nulle) et freine si necessaire.

Le bilan de W' utilise la puissance reellement appliquee (celle demandee,
reduite ou annulee quand le coureur ne peut pas l'utiliser), ce qui compte
pour l'optimiseur de la phase 6 : demander 400 W dans une epingle ne coute
pas de W'.

API : `simulate_dynamic(powers, seg, rider, env, params)` renvoie un
`DynamicResult` compatible dans l'esprit avec `PacingResult` (temps par
segment, vitesse, bilan de W') plus les traces fines (s, v) pour les
graphiques. `simulate` v1 reste inchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pacing_model import (G, DRIVETRAIN_EFF, Rider, Environment, Course,
                          speed_from_power)


@dataclass
class DynamicsParams:
    a_lat_max: float = 6.0        # acceleration laterale limite en virage (m/s^2), 0,61 g
    a_brake_max: float = 4.0      # deceleration maximale au freinage (m/s^2)
    wheel_inertia_kg: float = 1.5 # masse equivalente des roues (kg)
    road_width_m: float = 6.0     # largeur utilisable pour la trajectoire (m) ; 0 = axe de la route
    v_start: float | None = 1.0   # vitesse initiale (m/s) ; None = vitesse d'equilibre
    v_cap: float = 33.0           # plafond absolu (m/s), ~120 km/h
    substep_m: float = 5.0        # pas d'integration (m)


@dataclass
class DynamicResult:
    powers: np.ndarray            # puissance demandee par segment (W)
    applied_power: np.ndarray     # puissance effectivement appliquee (W), moyenne temporelle
    time_s: float
    seg_time_s: np.ndarray
    speed_ms: np.ndarray          # vitesse moyenne par segment (m/s)
    v_limit_ms: np.ndarray        # plafond de vitesse par segment (virage + freinage)
    braking: np.ndarray           # True si le plafond a mordu dans le segment
    wbal_j: np.ndarray
    min_wbal_j: float
    course: Course
    trace_s: np.ndarray = field(default_factory=lambda: np.zeros(0))   # distance fine (m)
    trace_v: np.ndarray = field(default_factory=lambda: np.zeros(0))   # vitesse fine (m/s)

    @property
    def avg_power(self) -> float:
        return float(np.average(self.applied_power, weights=self.seg_time_s))

    @property
    def avg_speed_kmh(self) -> float:
        return self.course.total_km / (self.time_s / 3600.0)

    @property
    def time_str(self) -> str:
        return f"{int(self.time_s // 60)}:{self.time_s % 60:04.1f}"


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------

def resistive_force(v, grade, rider: Rider, rho: float, wind_ms: float) -> float:
    """Somme gravite + roulement + aero (N), positive quand elle freine.

    L'aero utilise (v + w)|v + w| : avec un fort vent arriere (w < -v) elle
    devient motrice.
    """
    theta = np.arctan(grade)
    f_rg = rider.mass * G * (np.sin(theta) + rider.crr * np.cos(theta))
    rel = v + wind_ms
    return f_rg + 0.5 * rho * rider.cda * rel * abs(rel)


def corner_speed_limit(radius_m, a_lat_max: float) -> np.ndarray:
    r = np.asarray(radius_m, float)
    return np.where(np.isfinite(r), np.sqrt(a_lat_max * np.maximum(r, 0.0)), np.inf)


def racing_line_radius(r_center_m: float, turn_deg: float, width_m: float) -> float:
    """Rayon de la trajectoire tendue dans un virage d'axe de rayon r et d'angle phi.

    Geometrie classique "exterieur - corde - exterieur" : le plus grand cercle
    tangent a l'interieur du virage au point de corde et aux deux bords
    exterieurs des lignes droites d'entree et de sortie. Avec r_i = r - w/2,
    r_o = r + w/2 :
        R = (r_o - r_i cos(phi/2)) / (1 - cos(phi/2)).
    Exemples (w = 6 m) : virage a 90 degres de 25 m -> 42 m ; a 60 degres de
    30 m -> 67 m ; epingle a 180 degres de 12 m -> 15 m (le gain disparait).
    Hypothese : route fermee, largeur entierement utilisable (cas d'un CLM).
    On plafonne R a 10 fois le rayon d'axe pour les virages tres ouverts, ou la
    formule diverge alors que la vitesse n'est plus limitee par le virage.
    """
    if width_m <= 0.0 or turn_deg <= 0.0:
        return r_center_m
    c = np.cos(np.radians(min(turn_deg, 180.0) / 2.0))
    r_i = max(r_center_m - width_m / 2.0, 1.0)
    r_o = r_center_m + width_m / 2.0
    R = (r_o - r_i * c) / max(1.0 - c, 1e-6)
    return float(min(max(R, r_center_m), 10.0 * r_center_m))


def effective_radius(seg, width_m: float) -> np.ndarray:
    """Rayon limitant par segment : trajectoire tendue dans les virages
    detectes, rayon d'axe ailleurs (ou si `seg` n'a pas de table de virages)."""
    r = np.array(seg.radius_m, float)
    corners = getattr(seg, "corners", None)
    if not corners or width_m <= 0.0:
        return r
    r_eff = np.where(np.isfinite(r), np.maximum(r, 150.0), r)   # hors virage : libre
    for c in corners:
        R = racing_line_radius(c["r_min_m"], c["turn_deg"], width_m)
        i0 = max(np.searchsorted(seg.dist_m, c["s_start_m"], side="right") - 1, 0)
        i1 = min(np.searchsorted(seg.dist_m, c["s_end_m"], side="left"), seg.n)
        i1 = max(i1, i0 + 1)
        r_eff[i0:i1] = np.minimum(r_eff[i0:i1], R)
    return r_eff


def speed_limit_profile(seg, params: DynamicsParams, rider: Rider | None = None):
    """Plafond de vitesse aux noeuds (n+1) et par segment (n).

    Passe arriere : v_lim[i] = min(v_virage(i), sqrt(v_lim[i+1]^2 + 2 a L_i)),
    avec a = a_brake + g sin(theta) (la pente aide ou gene le freinage).
    Le plafond de virage d'un segment s'applique aux deux noeuds qui le bordent.
    """
    n = seg.n
    v_corner = np.minimum(corner_speed_limit(effective_radius(seg, params.road_width_m),
                                             params.a_lat_max), params.v_cap)
    v_node = np.full(n + 1, params.v_cap)
    v_node[:-1] = np.minimum(v_node[:-1], v_corner)
    v_node[1:] = np.minimum(v_node[1:], v_corner)
    theta = np.arctan(seg.grade)
    a_eff = np.maximum(params.a_brake_max + G * np.sin(theta), 0.5)
    for i in range(n - 1, -1, -1):
        v_node[i] = min(v_node[i], np.sqrt(v_node[i + 1] ** 2 + 2.0 * a_eff[i] * seg.length_m[i]))
    return v_node, v_corner


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def simulate_dynamic(powers, seg, rider: Rider, env: Environment,
                     params: DynamicsParams | None = None,
                     keep_trace: bool = False) -> DynamicResult:
    """Integre le mouvement segment par segment avec sous-pas de `substep_m`.

    powers : puissance demandee par segment (W), longueur seg.n.
    seg    : SegmentedCourse (course_io) ; un `Course` v1 est accepte, auquel
             cas il n'y a ni virage ni freinage (rayon infini).
    """
    params = params or DynamicsParams()
    powers = np.asarray(powers, float)
    if not hasattr(seg, "radius_m"):
        seg = _course_as_segmented(seg)
    n = seg.n
    course = Course(seg.length_m.copy(), seg.grade.copy())
    m_eff = rider.mass + params.wheel_inertia_kg
    v_node_lim, v_corner = speed_limit_profile(seg, params)
    theta = np.arctan(seg.grade)
    a_eff = np.maximum(params.a_brake_max + G * np.sin(theta), 0.5)

    if params.v_start is None:
        e0 = env.at(0)
        v = min(speed_from_power(powers[0], seg.grade[0], rider, e0), v_node_lim[0])
    else:
        v = min(float(params.v_start), v_node_lim[0])
    v = max(v, 0.5)

    seg_time = np.zeros(n); seg_dist_over_v = np.zeros(n)
    applied = np.zeros(n); braking = np.zeros(n, bool)
    trace_s, trace_v = [0.0], [v]
    wbal = rider.w_prime; min_wbal = wbal; wbal_track = np.zeros(n)
    s_global = 0.0

    for i in range(n):
        e = env.at(i)
        rho, wind = float(e.rho), float(e.wind_ms)
        L = seg.length_m[i]; grade = seg.grade[i]; p_req = powers[i]
        k = max(int(np.ceil(L / params.substep_m)), 1)
        ds = L / k
        t_seg = 0.0; work = 0.0
        v_next_node = v_node_lim[i + 1]
        v_cap_seg = min(v_corner[i], params.v_cap)
        for j in range(k):
            remaining = L - (j + 1) * ds
            v_lim_end = min(v_cap_seg, np.sqrt(v_next_node ** 2 + 2.0 * a_eff[i] * remaining))
            f_res = resistive_force(v, grade, rider, rho, wind)
            # vitesse atteinte en roue libre puis avec la puissance demandee
            v2_coast = v * v - 2.0 * f_res * ds / m_eff
            drive = p_req * DRIVETRAIN_EFF / max(v, 0.5)
            v2_pow = v * v + 2.0 * (drive - f_res) * ds / m_eff
            v_pow = np.sqrt(max(v2_pow, 0.25))
            if v_pow <= v_lim_end:
                v_new, p_app = v_pow, p_req
            elif v2_coast >= v_lim_end ** 2:
                # meme en roue libre on depasse : freinage, puissance nulle
                v_new, p_app = v_lim_end, 0.0
                braking[i] = True
            else:
                # puissance reduite pour arriver exactement au plafond
                v_new = v_lim_end
                p_app = (m_eff * (v_new ** 2 - v * v) / (2.0 * ds) + f_res) * max(v, 0.5) / DRIVETRAIN_EFF
                p_app = float(np.clip(p_app, 0.0, p_req))
                braking[i] = True
            dt = 2.0 * ds / (v + v_new)
            t_seg += dt; work += p_app * dt
            v = v_new
            s_global += ds
            if keep_trace:
                trace_s.append(s_global); trace_v.append(v)

        seg_time[i] = t_seg
        applied[i] = work / t_seg
        # bilan de W' (Skiba) sur la puissance appliquee moyenne du segment
        p_bar = applied[i]
        if p_bar > rider.cp:
            wbal -= (p_bar - rider.cp) * t_seg
        else:
            wbal = rider.w_prime - (rider.w_prime - wbal) * np.exp(-((rider.cp - p_bar) / rider.w_prime) * t_seg)
        min_wbal = min(min_wbal, wbal)
        wbal_track[i] = wbal

    return DynamicResult(
        powers=powers, applied_power=applied, time_s=float(seg_time.sum()),
        seg_time_s=seg_time, speed_ms=seg.length_m / seg_time,
        v_limit_ms=np.minimum(v_corner, params.v_cap), braking=braking,
        wbal_j=wbal_track, min_wbal_j=float(min_wbal), course=course,
        trace_s=np.array(trace_s), trace_v=np.array(trace_v),
    )


class _PlainSegments:
    """Adaptateur : Course v1 -> attributs minimaux d'un SegmentedCourse."""
    def __init__(self, course: Course):
        self.length_m = course.length_m
        self.grade = course.grade
        self.n = course.n
        self.radius_m = np.full(course.n, np.inf)
        self.bearing_deg = np.zeros(course.n)
        self.dist_m = np.concatenate([[0.0], np.cumsum(course.length_m)])
        self.total_km = course.total_km


def _course_as_segmented(course: Course) -> _PlainSegments:
    return _PlainSegments(course)
