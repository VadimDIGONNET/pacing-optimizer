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
    block = np.clip(np.searchsorted(edges, seg.mid_dist_m, side="right") - 1, 0, len(edges) - 2)
    return block, edges


def expand(block_powers, block) -> np.ndarray:
    return np.asarray(block_powers, float)[block]


def constant_power_dynamic(seg, rider: Rider, env: Environment, params: DynamicsParams) -> float:
    """Puissance constante qui vide exactement W' a l'arrivee (reference)."""
    n = seg.n

    def end_wbal(p):
        return simulate_dynamic(np.full(n, p), seg, rider, env, params).wbal_j[-1]

    if end_wbal(rider.p_max) > 0:
        return rider.p_max
    return brentq(end_wbal, rider.cp, rider.p_max, xtol=0.1)


def optimize_dynamic(seg, rider: Rider, env: Environment, params: DynamicsParams | None = None,
                     block_m: float = 500.0, breakpoints_km=None, smooth_lambda: float = 2e-4,
                     bounds=None, maxiter: int = 150, ftol: float = 1e-5,
                     x0=None) -> OptimizerResult:
    """Minimise le temps sous contrainte W'bal >= 0, puissance par bloc.

    smooth_lambda : poids de la penalite de lissage (s/W^2). 2e-4 s/W^2
    signifie qu'un saut de 100 W entre deux blocs voisins "coute" 2 s : assez
    pour eviter les dents de scie, trop peu pour empecher une vraie montee.
    bounds : liste optionnelle de (p_min, p_max) par bloc.
    """
    params = params or DynamicsParams()
    block, edges = make_blocks(seg, block_m, breakpoints_km)
    nb = len(edges) - 1
    t0 = time.perf_counter()

    p_const = constant_power_dynamic(seg, rider, env, params)
    constant = simulate_dynamic(np.full(seg.n, p_const), seg, rider, env, params)

    if bounds is None:
        bounds = [(rider.p_min, rider.p_max)] * nb
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.full(nb, p_const) if x0 is None else np.asarray(x0, float), lo, hi)

    def sim(x):
        return simulate_dynamic(expand(x, block), seg, rider, env, params)

    def objective(x):
        r = sim(x)
        return r.time_s + smooth_lambda * np.sum(np.diff(x) ** 2)

    cons = [{"type": "ineq", "fun": lambda x: sim(x).min_wbal_j / 1000.0}]
    res = minimize(objective, x0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                   options={"maxiter": maxiter, "ftol": ftol, "eps": 1.0})
    x = np.clip(res.x, lo, hi)
    final = sim(x)
    return OptimizerResult(result=final, block_powers=x, block_bounds_km=edges / 1000.0,
                           constant=constant, p_constant=p_const,
                           elapsed_s=time.perf_counter() - t0, n_iter=int(res.nit), success=bool(res.success))


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
