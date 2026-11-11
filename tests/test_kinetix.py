import math

import numpy as np
import pytest

import kinetix as kx


@pytest.mark.parametrize("order, c0, k", [(0, 1.0, 0.01), (1, 1.0, 0.3), (2, 0.5, 2.0)])
def test_half_life_matches_integrated_law(order, c0, k):
    t_half = kx.half_life(order, k, c0)
    assert float(kx.integrated(order, c0, k, t_half)) == pytest.approx(c0 / 2)


def test_half_life_requires_c0():
    with pytest.raises(ValueError):
        kx.half_life(2, 1.0)


def test_first_order_simulation_matches_analytic():
    net = kx.ReactionNetwork.from_text("A -> B ; k = 0.25")
    sim = net.simulate({"A": 2.0}, t_end=20)
    np.testing.assert_allclose(sim["A"], kx.integrated(1, 2.0, 0.25, sim.t), rtol=1e-6, atol=1e-9)


def test_second_order_simulation_matches_analytic():
    net = kx.ReactionNetwork.from_text("2A -> P ; k = 0.5")
    sim = net.simulate({"A": 1.0}, t_end=10)
    # d[A]/dt = -2 k [A]^2  ->  effective second-order constant 2k
    np.testing.assert_allclose(sim["A"], kx.integrated(2, 1.0, 1.0, sim.t), rtol=1e-6)


def test_consecutive_reactions_t_max_and_simulation():
    k1, k2 = 1.0, 0.2
    t_max = math.log(k1 / k2) / (k1 - k2)
    t = np.linspace(0, 30, 3001)
    A, B, C = kx.consecutive_first_order(1.0, k1, k2, t)
    assert t[np.argmax(B)] == pytest.approx(t_max, abs=0.01)
    np.testing.assert_allclose(A + B + C, 1.0)

    net = kx.ReactionNetwork.from_text("A -> B ; k = 1.0\nB -> C ; k = 0.2")
    sim = net.simulate({"A": 1.0}, t_end=30)
    _, B_an, _ = kx.consecutive_first_order(1.0, k1, k2, sim.t)
    np.testing.assert_allclose(sim["B"], B_an, atol=1e-7)


def test_consecutive_equal_rates_limit():
    t = np.linspace(0, 10, 101)
    _, B_eq, _ = kx.consecutive_first_order(1.0, 0.5, 0.5, t)
    _, B_near, _ = kx.consecutive_first_order(1.0, 0.5, 0.5 + 1e-7, t)
    np.testing.assert_allclose(B_eq, B_near, atol=1e-6)


def test_reversible_reaction_reaches_equilibrium():
    kf, kb = 2.0, 0.5
    net = kx.ReactionNetwork.from_text(f"A + B <-> C ; kf = {kf}, kb = {kb}")
    final = net.simulate({"A": 1.0, "B": 0.6}, t_end=200).final()
    assert final["C"] / (final["A"] * final["B"]) == pytest.approx(kf / kb, rel=1e-5)
    # mass balance on A and B
    assert final["A"] + final["C"] == pytest.approx(1.0, rel=1e-8)
    assert final["B"] + final["C"] == pytest.approx(0.6, rel=1e-8)


def test_conserved_quantities():
    net = kx.ReactionNetwork.from_text("A + B <-> C ; kf = 1, kb = 1")
    cons = net.conserved_quantities()
    assert cons.shape[0] == 2
    assert np.allclose(cons @ net.stoich, 0.0)


def test_arrhenius_fit_recovers_parameters():
    A, Ea = 1.2e11, 75_000.0
    T = np.linspace(280, 360, 9)
    fit = kx.fit_arrhenius(T, kx.arrhenius(A, Ea, T))
    assert fit["Ea"] == pytest.approx(Ea, rel=1e-8)
    assert fit["A"] == pytest.approx(A, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0)


def test_arrhenius_rule_of_thumb():
    # Ea ~ 50 kJ/mol roughly doubles the rate between 25 and 35 C
    ratio = kx.arrhenius(1.0, 52_900.0, 308.15) / kx.arrhenius(1.0, 52_900.0, 298.15)
    assert ratio == pytest.approx(2.0, rel=0.02)


def test_eyring_prefactor():
    # with dH = dS = 0, k = kB T / h ~ 6.2e12 s^-1 at 298 K
    assert float(kx.eyring(0.0, 0.0, 298.15)) == pytest.approx(6.21e12, rel=1e-3)


@pytest.mark.parametrize("order, k", [(0, 0.004), (1, 0.15), (2, 0.8)])
def test_fit_order_identifies_order(order, k):
    t = np.linspace(0, 10, 25)
    c = kx.integrated(order, 1.0, k, t)
    res = kx.fit_order(t, c)
    assert res["order"] == order
    assert res["k"] == pytest.approx(k, rel=1e-6)


def test_network_parse_errors():
    with pytest.raises(ValueError):
        kx.ReactionNetwork.from_text("A -> B")
    with pytest.raises(ValueError):
        kx.ReactionNetwork.from_text("A B ; k = 1")
    with pytest.raises(ValueError):
        kx.ReactionNetwork.from_text("A -> B ; k = 1").simulate({"Z": 1.0}, 1.0)
