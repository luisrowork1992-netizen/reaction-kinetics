"""Integrated rate laws, half-lives and temperature dependence of rate constants."""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "R_GAS",
    "K_B",
    "H_PLANCK",
    "integrated",
    "half_life",
    "arrhenius",
    "fit_arrhenius",
    "eyring",
    "fit_order",
    "consecutive_first_order",
]

R_GAS = 8.314462618        # J / (mol K)
K_B = 1.380649e-23         # J / K
H_PLANCK = 6.62607015e-34  # J s


def integrated(order: int, c0: float, k: float, t):
    """Concentration vs. time for an nth-order reaction ``A -> products`` (n = 0, 1, 2)."""
    t = np.asarray(t, dtype=float)
    if order == 0:
        return np.maximum(c0 - k * t, 0.0)
    if order == 1:
        return c0 * np.exp(-k * t)
    if order == 2:
        return c0 / (1.0 + k * c0 * t)
    raise ValueError("order must be 0, 1 or 2")


def half_life(order: int, k: float, c0: float | None = None) -> float:
    """Half-life of an nth-order reaction (``c0`` is required for orders 0 and 2)."""
    if order == 1:
        return math.log(2) / k
    if c0 is None:
        raise ValueError("c0 is required for zero- and second-order half-lives")
    if order == 0:
        return c0 / (2 * k)
    if order == 2:
        return 1.0 / (k * c0)
    raise ValueError("order must be 0, 1 or 2")


def arrhenius(A: float, Ea: float, T):
    """``k = A exp(-Ea / (R T))`` with Ea in J/mol."""
    return A * np.exp(-Ea / (R_GAS * np.asarray(T, dtype=float)))


def fit_arrhenius(T, k) -> dict:
    """Least-squares fit of ``ln k`` vs ``1/T``.

    Returns ``{"A": ..., "Ea": ... (J/mol), "r2": ...}``.
    """
    T = np.asarray(T, dtype=float)
    k = np.asarray(k, dtype=float)
    if T.size < 2:
        raise ValueError("need at least two temperatures")
    x, y = 1.0 / T, np.log(k)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"A": float(math.exp(intercept)), "Ea": float(-slope * R_GAS), "r2": r2}


def eyring(dH: float, dS: float, T, kappa: float = 1.0):
    """Transition-state theory rate constant (s^-1) from activation enthalpy (J/mol) and entropy (J/mol/K)."""
    T = np.asarray(T, dtype=float)
    return kappa * K_B * T / H_PLANCK * np.exp(dS / R_GAS) * np.exp(-dH / (R_GAS * T))


def fit_order(t, c) -> dict:
    """Decide whether concentration data look zero-, first- or second-order.

    Fits the linearised form of each integrated rate law (``c``, ``ln c``,
    ``1/c`` against ``t``) and returns the best order with its rate constant
    and the R^2 value of every candidate.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    forms = {0: c, 1: np.log(c), 2: 1.0 / c}
    sign = {0: -1.0, 1: -1.0, 2: 1.0}
    scores, ks = {}, {}
    for n, y in forms.items():
        slope, intercept = np.polyfit(t, y, 1)
        pred = slope * t + intercept
        ss_tot = np.sum((y - y.mean()) ** 2)
        scores[n] = 1.0 - np.sum((y - pred) ** 2) / ss_tot
        ks[n] = sign[n] * slope
    best = max(scores, key=scores.get)
    return {"order": best, "k": float(ks[best]), "r2": {n: float(s) for n, s in scores.items()}}


def consecutive_first_order(a0: float, k1: float, k2: float, t):
    """Analytic solution of ``A -k1-> B -k2-> C`` with only A present at t = 0.

    Returns ``(A, B, C)`` arrays. ``t_max`` for B is ``ln(k1/k2) / (k1 - k2)``.
    """
    t = np.asarray(t, dtype=float)
    A = a0 * np.exp(-k1 * t)
    if math.isclose(k1, k2):
        B = a0 * k1 * t * np.exp(-k1 * t)
    else:
        B = a0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    C = a0 - A - B
    return A, B, C
