"""Optimiseur v2 (phase 6) : puissance par blocs sur le simulateur dynamique.

Variables : une puissance par bloc. Les blocs regroupent les segments de
50 m par tranches lisibles (par defaut 500 m, ou un decoupage par section du
parcours donne par l'utilisateur), afin que le plan tienne sur un compteur.

Objectif : temps total + penalite de lissage lambda * sum((P_k+1 - P_k)^2),
exprimee en secondes par W^2 pour rester homogene. La penalite empeche les
alternances brutales que le modele W' ne sanctionne pas mais qu'un coureur
ne peut pas suivre.

Contraintes : W'bal >= 0 partout (sur la puissance appliquee, donc une
puissance demandee dans un virage freine ne coute rien), bornes
[p_min, p_max] par bloc, blocs eventuellement figes (bounds).

Methode : SLSQP sur les blocs, gradient par differences finies. Pour un CLM
de 30 km a 500 m par bloc : 60 variables, environ 60 simulations par
iteration, moins d'une minute (critere du plan). Point de depart : la
puissance constante qui vide W' a l'arrivee.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize, brentq

from pacing_model import Rider, Environment
from dynamics import simulate_dynamic, DynamicsParams, DynamicResult


@dataclass
class OptimizerResult:
    result: DynamicResult          # simulation du plan optimal
    block_powers: np.ndarray       # puissance par bloc (W)
    block_bounds_km: np.ndarray    # bornes des blocs (km), longueur n_blocs + 1
    constant: DynamicResult        # reference : puissance constante, meme W'
    p_constant: float
    elapsed_s: float
    n_iter: int
    success: bool

    @property
    def gain_s(self) -> float:
        return self.constant.time_s - self.result.time_s


def make_blocks(seg, block_m: float = 500.0, breakpoints_km=None) -> np.ndarray:
    """Indice de bloc pour chaque segment. breakpoints_km : coupures imposees
    (par exemple le pied et le sommet d'une montee) ajoutees aux tranches."""
    edges = list(np.arange(0.0, seg.dist_m[-1], block_m)) + [seg.dist_m[-1]]
    if breakpoints_km:
        edges += [b * 1000.0 for b in breakpoints_km]
    edges = np.unique(np.round(edges, 3))
    edges = edges[(edges >= 0) & (edges <= seg.dist_m[-1] + 1e-6)]
    # supprime les blocs trop courts (< 150 m) crees par des coupures proches
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= 150.0 or e == edges[-1]:
            keep.append(e)
    edges = np.array(keep)
    # chaque bord de bloc est ramene au noeud de segment le plus proche : un
    # bloc contient toujours au moins un segment entier (sinon la variable
    # n'aurait aucun effet, cas d'un parcours v1 a segments de 1 km)
    nodes = np.asarray(seg.dist_m, float)
    edges = np.unique(nodes[np.abs(nodes[None, :] - edges[:, None]).argmin(axis=1)])
    block = np.clip(np.searchsorted(edges, seg.mid_dist_m, side="right") - 1, 0, len(edges) - 2)
    return block, edges


def expand(block_powers, block) -> np.ndarray:
    return np.asarray(block_powers, float)[block]


def constant_power_dynamic(seg, rider: Rider, env: Environment, params: DynamicsParams) -> float:
    """Puissance constante la plus elevee qui garde W'bal >= 0 partout (reference).

    Sur un parcours plat, W'bal est minimal a l'arrivee et on retrouve la
    reference v1 (W' vide sur la ligne). Avec une montee suivie d'une descente,
    le minimum est au sommet : la puissance constante admissible est plus
    basse, ce qui est la vraie contrainte d'un coureur a puissance constante.
    """
    n = seg.n

    def min_wbal(p):
        return simulate_dynamic(np.full(n, p), seg, rider, env, params).min_wbal_j

    if min_wbal(rider.p_max) > 0:
        return rider.p_max
    if min_wbal(rider.cp) < 0:
        return rider.cp
    return brentq(min_wbal, rider.cp, rider.p_max, xtol=0.1)


def auto_sections(seg, min_len_m: float = 1000.0, climb=0.02, descent=-0.02):
    """Coupures automatiques par type de terrain (montee / descente / plat),
    sections d'au moins `min_len_m`. Renvoie des km de coupure."""
    kind = np.where(seg.grade > climb, 1, np.where(seg.grade < descent, -1, 0))
    # lissage majoritaire sur ~500 m pour eviter les micro-sections
    w = max(int(round(500.0 / np.median(seg.length_m))), 1)
    sm = np.convolve(kind, np.ones(w) / w, mode="same")
    kind = np.where(sm > 0.5, 1, np.where(sm < -0.5, -1, 0))
    cuts, start = [], 0
    for i in range(1, seg.n + 1):
        if i == seg.n or kind[i] != kind[start]:
            if i < seg.n and seg.dist_m[i] - seg.dist_m[start] >= min_len_m:
                cuts.append(seg.dist_m[i] / 1000.0)
                start = i
            elif i < seg.n:
                # section trop courte : on la fusionne avec la suivante
                start = i if not cuts else start
    return cuts


def _slsqp(sim, x0, lo, hi, smooth_lambda, maxiter, ftol, eps):
    """SLSQP en watts, contrainte W'bal en kJ (bon conditionnement en pratique :
    une normalisation des variables fait echouer la recherche lineaire)."""

    def objective(x):
        return sim(x).time_s + smooth_lambda * np.sum(np.diff(x) ** 2)

    cons = [{"type": "ineq", "fun": lambda x: sim(x).min_wbal_j / 1000.0}]
    res = minimize(objective, x0, method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=cons, options={"maxiter": maxiter, "ftol": ftol, "eps": eps})
    return np.clip(res.x, lo, hi), res


def _block_time_table(seg, rider, env, block, nb, p_grid):
    """Temps d'equilibre (modele v1) de chaque bloc pour chaque puissance de la
    grille : table (nb, len(p_grid)) utilisee par la methode du prix de l'energie."""
    from pacing_model import speed_from_power
    table = np.zeros((nb, len(p_grid)))
    for i in range(seg.n):
        e = env.at(i)
        for j, p in enumerate(p_grid):
            table[block[i], j] += seg.length_m[i] / speed_from_power(p, seg.grade[i], rider, e)
    return table


def equilibrium_block_plan(seg, rider: Rider, env: Environment, block, nb, lo, hi,
                           p_step: float = 2.0, maxiter: int = 300):
    """Plan initial : SLSQP sur le modele d'equilibre (v1) par bloc.

    Les temps d'equilibre de chaque bloc sont tabules sur une grille de
    puissance (interpolation lineaire ensuite), et le bilan de W' de Skiba est
    evalue bloc par bloc : chaque evaluation coute quelques microsecondes, et
    les gradients sont propres (pas de freinage ni d'inertie). C'est
    l'optimiseur v1, qui trouve la bonne structure (pousser dans les montees,
    relacher dans les descentes) depuis un depart a puissance constante. Le
    polissage dynamique part de la.
    """
    p_grid = np.arange(np.floor(lo.min()), np.ceil(hi.max()) + p_step, p_step)
    table = _block_time_table(seg, rider, env, block, nb, p_grid)
    rows = np.arange(nb)

    def times(x):
        j = np.clip(np.searchsorted(p_grid, x) - 1, 0, len(p_grid) - 2)
        w = (x - p_grid[j]) / (p_grid[j + 1] - p_grid[j])
        return table[rows, j] * (1 - w) + table[rows, j + 1] * w

    def min_wbal(x):
        t = times(x)
        wb = rider.w_prime; m = wb
        for k in range(nb):
            if x[k] > rider.cp:
                wb -= (x[k] - rider.cp) * t[k]
            else:
                wb = rider.w_prime - (rider.w_prime - wb) * np.exp(-((rider.cp - x[k]) / rider.w_prime) * t[k])
            m = min(m, wb)
        return m

    def p_const():
        f = lambda p: min_wbal(np.clip(np.full(nb, p), lo, hi))
        if f(rider.p_max) > 0:
            return rider.p_max
        if f(rider.cp) < 0:
            return rider.cp
        return brentq(f, rider.cp, rider.p_max, xtol=0.1)

    x0 = np.clip(np.full(nb, p_const()), lo, hi)
    res = minimize(lambda x: times(x).sum(), x0, method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=[{"type": "ineq", "fun": lambda x: min_wbal(x) / 1000.0}],
                   options={"maxiter": maxiter, "ftol": 1e-7})
    x = np.clip(res.x, lo, hi)
    return x if min_wbal(x) > -100.0 else x0


def fill_last_cycle(sim, x, block, rider: Rider, lo, hi, tol_j: float = 100.0):
    """Si W' n'est pas vide a l'arrivee, releve la puissance des blocs situes
    apres le dernier passage de W'bal pres de zero (dernier "cycle"), par
    homothetie autour de CP, jusqu'a ce que le minimum de W'bal soit nul.
    Corrige la convergence incomplete du polissage sur la fin de parcours."""
    r = sim(x)
    if r.wbal_j[-1] < tol_j:
        return x, r
    near_zero = np.where(r.wbal_j < tol_j)[0]
    k0 = int(block[near_zero[-1]]) + 1 if len(near_zero) else 0
    if k0 >= len(x):
        return x, r
    sel = np.arange(len(x)) >= k0

    def plan(alpha):
        y = x.copy()
        y[sel] = np.clip(rider.cp + alpha * (x[sel] - rider.cp) + (alpha - 1.0) * 20.0, lo[sel], hi[sel])
        return y

    a_lo, a_hi = 1.0, 1.0
    while sim(plan(a_hi)).min_wbal_j > 0 and a_hi < 8.0:
        a_hi *= 1.5
    if sim(plan(a_hi)).min_wbal_j > 0:
        return plan(a_hi), sim(plan(a_hi))
    for _ in range(30):
        a = 0.5 * (a_lo + a_hi)
        if sim(plan(a)).min_wbal_j < 0:
            a_hi = a
        else:
            a_lo = a
    y = plan(a_lo); ry = sim(y)
    return (y, ry) if ry.time_s < r.time_s else (x, r)


def optimize_dynamic(seg, rider: Rider, env: Environment, params: DynamicsParams | None = None,
                     block_m: float = 500.0, breakpoints_km=None, smooth_lambda: float = 5e-5,
                     bounds=None, maxiter: int = 300, ftol: float = 1e-7, eps: float = 0.5,
                     x0=None, polish: bool = True) -> OptimizerResult:
    """Minimise le temps sous contrainte W'bal >= 0, puissance par bloc.

    1. Plan initial par l'optimiseur d'equilibre v1 sur les blocs
       (equilibrium_block_plan) : rapide, trouve la structure du pacing.
    2. Polissage SLSQP sur le simulateur dynamique (inertie, virages,
       freinage) a partir de ce plan, avec la penalite de lissage ; conserve
       seulement s'il reste faisable et plus rapide (penalite comprise).

    smooth_lambda : poids de la penalite (s/W^2). 5e-5 s/W^2 : un saut de
    100 W entre deux blocs voisins "coute" 0,5 s, assez pour eviter les dents
    de scie entre blocs voisins de pente egale, trop peu pour empecher un
    vrai changement de rythme au pied ou au sommet d'une montee.
    bounds : liste optionnelle de (p_min, p_max) par bloc.
    """
    params = params or DynamicsParams()
    if not hasattr(seg, "radius_m"):
        from dynamics import _course_as_segmented
        seg = _course_as_segmented(seg)
    if breakpoints_km is None:
        breakpoints_km = auto_sections(seg)
    block, edges = make_blocks(seg, block_m, breakpoints_km)
    nb = len(edges) - 1
    t0 = time.perf_counter()

    p_const = constant_power_dynamic(seg, rider, env, params)
    constant = simulate_dynamic(np.full(seg.n, p_const), seg, rider, env, params)

    if bounds is None:
        bounds = [(rider.p_min, rider.p_max)] * nb
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)

    def sim_blocks(xv):
        return simulate_dynamic(expand(xv, block), seg, rider, env, params)

    if x0 is None:
        x0 = equilibrium_block_plan(seg, rider, env, block, nb, lo, hi)
    x0 = np.clip(np.asarray(x0, float), lo, hi)
    r0 = sim_blocks(x0)
    # un plan infaisable (CP trop faible) ou plus lent que la reference est remplace par la reference
    if r0.min_wbal_j < -50.0 or r0.time_s > constant.time_s:
        x0 = np.clip(np.full(nb, p_const), lo, hi); r0 = sim_blocks(x0)
    x, r_x, n_iter, success = x0, r0, 0, True
    if polish:
        xp, res = _slsqp(sim_blocks, x0, lo, hi, smooth_lambda, maxiter, ftol, eps)
        rp = sim_blocks(xp)
        n_iter = int(res.nit)
        pen0 = smooth_lambda * np.sum(np.diff(x0) ** 2); penp = smooth_lambda * np.sum(np.diff(xp) ** 2)
        if rp.min_wbal_j >= -50.0 and rp.time_s + penp < r0.time_s + pen0:
            x, r_x, success = xp, rp, bool(res.success)
    x, r_x = fill_last_cycle(sim_blocks, x, block, rider, lo, hi)
    # la reference constante doit rester comparable : meme contrainte
    return OptimizerResult(result=r_x, block_powers=x, block_bounds_km=edges / 1000.0,
                           constant=constant, p_constant=p_const,
                           elapsed_s=time.perf_counter() - t0, n_iter=n_iter, success=success)


def plan_table(opt: OptimizerResult, seg):
    """Plan lisible : un bloc par ligne (km debut, km fin, pente moyenne,
    puissance cible, vitesse et temps prevus)."""
    import pandas as pd
    rows = []
    r = opt.result
    edges_m = opt.block_bounds_km * 1000.0
    for k in range(len(edges_m) - 1):
        idx = np.where((seg.mid_dist_m >= edges_m[k]) & (seg.mid_dist_m < edges_m[k + 1]))[0]
        if len(idx) == 0:
            continue
        L = seg.length_m[idx].sum(); t = r.seg_time_s[idx].sum()
        rows.append({"km_start": edges_m[k] / 1000, "km_end": edges_m[k + 1] / 1000,
                     "grade_pct": np.average(seg.grade[idx], weights=seg.length_m[idx]) * 100,
                     "power_w": opt.block_powers[k],
                     "applied_w": np.average(r.applied_power[idx], weights=r.seg_time_s[idx]),
                     "speed_kmh": L / t * 3.6, "time_s": t,
                     "wbal_end_kj": r.wbal_j[idx[-1]] / 1000.0,
                     "braking": bool(r.braking[idx].any())})
    return pd.DataFrame(rows)
