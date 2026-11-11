# reaction-kinetics

![tests](https://github.com/luisrowork1992-netizen/reaction-kinetics/actions/workflows/tests.yml/badge.svg)

`kinetix` is a Python toolkit for **chemical kinetics**. It covers closed-form rate laws and Arrhenius / Eyring analysis, and simulates arbitrary mass-action reaction networks written as plain text.

## Features

- Integrated rate laws and half-lives for zeroth-, first- and second-order reactions
- `fit_order`: picks the reaction order from linearised fits of [A], ln[A] and 1/[A]
- Arrhenius rate constants and least-squares Arrhenius fitting (Eₐ, A, R²)
- Eyring (transition-state theory) rate constants
- Analytic consecutive A → B → C solution, including the k₁ = k₂ limit
- `ReactionNetwork.from_text`: reactions written one per line, such as `A + 2B -> C ; k = 1.5`
  - reversible reactions use `<->` with `kf` and `kb`
- Simulation with SciPy `solve_ivp` (LSODA by default, so it handles stiff systems)
- Conservation laws computed from the left null space of the stoichiometric matrix

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/reaction-kinetics.git
cd reaction-kinetics
pip install -e .
```

## Quick start

```python
import kinetix as kx

net = kx.ReactionNetwork.from_text("""
A + B <-> C ; kf = 2.0, kb = 0.5
C -> D      ; k = 0.05
""")
sim = net.simulate({"A": 1.0, "B": 0.6}, t_end=100)
print(sim.final())

fit = kx.fit_arrhenius([300, 310, 320], [1.2e-3, 2.9e-3, 6.6e-3])
print(fit["Ea"] / 1000, "kJ/mol")
```

## Notebook

[`notebooks/kinetics_examples.ipynb`](notebooks/kinetics_examples.ipynb) covers five examples:

- identifying the reaction order
- Arrhenius analysis of synthetic data
- consecutive reactions, comparing the numerical and analytic solutions
- a reversible equilibrium
- a Lotka–Volterra autocatalytic oscillator

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- every half-life is consistent with its integrated law
- the ODE solver matches the first-order, second-order and consecutive-reaction solutions
- t_max of the intermediate
- the equilibrium constant K = k_f/k_b, plus mass balance
- the Arrhenius fit recovers Eₐ and A
- the "double every 10 °C" rule of thumb
- the k_BT/h prefactor
- automatic order detection

## License

MIT
