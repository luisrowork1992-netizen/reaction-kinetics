"""Mass-action reaction networks written as plain text and integrated with SciPy.

Example
-------
>>> net = ReactionNetwork.from_text('''
...     A + B -> C    ; k = 0.5
...     C -> A + B    ; k = 0.1
... ''')
>>> sol = net.simulate({"A": 1.0, "B": 0.8}, t_end=50)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["Reaction", "ReactionNetwork", "Simulation"]

_TERM = re.compile(r"^\s*(\d*\.?\d*)\s*([A-Za-z][A-Za-z0-9_]*)\s*$")


def _side(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    text = text.strip()
    if text in ("", "0", "∅"):
        return out
    for term in text.split("+"):
        m = _TERM.match(term)
        if not m:
            raise ValueError(f"cannot parse species term {term!r}")
        coef = float(m.group(1)) if m.group(1) else 1.0
        out[m.group(2)] = out.get(m.group(2), 0.0) + coef
    return out


@dataclass(frozen=True)
class Reaction:
    reactants: dict[str, float]
    products: dict[str, float]
    k: float

    def rate(self, conc: dict[str, float]) -> float:
        r = self.k
        for s, nu in self.reactants.items():
            r *= max(conc[s], 0.0) ** nu
        return r


@dataclass
class Simulation:
    t: np.ndarray
    species: list[str]
    y: np.ndarray  # shape (n_species, n_times)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.y[self.species.index(name)]

    def final(self) -> dict[str, float]:
        return {s: float(self.y[i, -1]) for i, s in enumerate(self.species)}


class ReactionNetwork:
    def __init__(self, reactions: list[Reaction]):
        if not reactions:
            raise ValueError("network has no reactions")
        self.reactions = reactions
        names: list[str] = []
        for r in reactions:
            for s in list(r.reactants) + list(r.products):
                if s not in names:
                    names.append(s)
        self.species = names
        n, m = len(names), len(reactions)
        self.stoich = np.zeros((n, m))
        for j, r in enumerate(reactions):
            for s, nu in r.reactants.items():
                self.stoich[names.index(s), j] -= nu
            for s, nu in r.products.items():
                self.stoich[names.index(s), j] += nu

    @classmethod
    def from_text(cls, text: str) -> "ReactionNetwork":
        """One reaction per line: ``A + 2B -> C ; k = 1.5``. ``<->`` needs ``kf`` and ``kb``."""
        reactions = []
        for raw in text.strip().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ";" not in line:
                raise ValueError(f"missing rate constant in {raw!r}")
            eq, params = line.split(";", 1)
            vals = dict(
                (p.split("=")[0].strip(), float(p.split("=")[1])) for p in params.split(",") if "=" in p
            )
            if "<->" in eq:
                lhs, rhs = eq.split("<->")
                reactions.append(Reaction(_side(lhs), _side(rhs), vals["kf"]))
                reactions.append(Reaction(_side(rhs), _side(lhs), vals["kb"]))
            elif "->" in eq:
                lhs, rhs = eq.split("->")
                reactions.append(Reaction(_side(lhs), _side(rhs), vals["k"]))
            else:
                raise ValueError(f"no arrow in {raw!r}")
        return cls(reactions)

    def rates(self, y: np.ndarray) -> np.ndarray:
        conc = dict(zip(self.species, y))
        return np.array([r.rate(conc) for r in self.reactions])

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.stoich @ self.rates(y)

    def simulate(self, initial: dict[str, float], t_end: float, n_points: int = 400,
                 method: str = "LSODA", rtol: float = 1e-9, atol: float = 1e-12) -> Simulation:
        unknown = set(initial) - set(self.species)
        if unknown:
            raise ValueError(f"unknown species: {sorted(unknown)}")
        y0 = np.array([initial.get(s, 0.0) for s in self.species], dtype=float)
        t_eval = np.linspace(0.0, t_end, n_points)
        sol = solve_ivp(self.derivatives, (0.0, t_end), y0, t_eval=t_eval,
                        method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return Simulation(sol.t, list(self.species), sol.y)

    def conserved_quantities(self, tol: float = 1e-10) -> np.ndarray:
        """Basis of the left null space of the stoichiometric matrix (rows = conservation laws)."""
        u, s, vt = np.linalg.svd(self.stoich.T)
        rank = int(np.sum(s > tol))
        return vt[rank:]
