"""kinetix: chemical kinetics, from integrated rate laws to mass-action networks."""

from .network import Reaction, ReactionNetwork, Simulation
from .rate_laws import (
    H_PLANCK,
    K_B,
    R_GAS,
    arrhenius,
    consecutive_first_order,
    eyring,
    fit_arrhenius,
    fit_order,
    half_life,
    integrated,
)

__version__ = "0.1.0"
