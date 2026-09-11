"""The greeks are derivatives, so the test is: do they match a numerical derivative?

Every analytic formula in optlab.bs is checked against a central finite
difference of the price (or of delta / vega for the second-order greeks) across
a grid of in-, at- and out-of-the-money cases. A sign error or a misplaced
discount factor cannot survive this file.
"""

import itertools

import numpy as np
import pytest

from optlab import bs

# spot, strike, maturity, rate, dividend, vol
CASES = list(
    itertools.product(
        [80.0, 100.0, 125.0],  # spot: OTM / ATM / ITM for a 100 strike
        [0.08, 0.75, 2.5],  # maturity in years
        [0.12, 0.35],  # vol
    )
)
K, R, Q = 100.0, 0.03, 0.015


def fd(func, arg_index, args, h):
    """Central difference of `func` in one argument."""
    up, down = list(args), list(args)
    up[arg_index] += h
    down[arg_index] -= h
    return (func(*up) - func(*down)) / (2 * h)


@pytest.mark.parametrize("S,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_matches_put_call_parity(S, T, vol, kind):
    call = bs.price(S, K, T, R, Q, vol, "call")
    put = bs.price(S, K, T, R, Q, vol, "put")
    parity = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert call - put == pytest.approx(parity, abs=1e-9)


@pytest.mark.parametrize("S,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_is_dprice_dspot(S, T, vol, kind):
    f = lambda s, k, t, r, q, v: bs.price(s, k, t, r, q, v, kind)
    numeric = fd(f, 0, (S, K, T, R, Q, vol), h=1e-4)
    assert bs.delta(S, K, T, R, Q, vol, kind) == pytest.approx(numeric, abs=1e-6)


@pytest.mark.parametrize("S,T,vol", CASES)
def test_gamma_is_ddelta_dspot(S, T, vol):
    f = lambda s, k, t, r, q, v: bs.delta(s, k, t, r, q, v, "call")
    numeric = fd(f, 0, (S, K, T, R, Q, vol), h=1e-4)
    assert bs.gamma(S, K, T, R, Q, vol) == pytest.approx(numeric, abs=1e-7)


@pytest.mark.parametrize("S,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_vega_is_dprice_dvol(S, T, vol, kind):
    f = lambda s, k, t, r, q, v: bs.price(s, k, t, r, q, v, kind)
    numeric = fd(f, 5, (S, K, T, R, Q, vol), h=1e-5)
    assert bs.vega(S, K, T, R, Q, vol, kind) == pytest.approx(numeric, abs=1e-4)


@pytest.mark.parametrize("S,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_theta_is_dprice_dcalendar_time(S, T, vol, kind):
    """t advancing by dt means T shrinking by dt, hence the minus sign."""
    f = lambda s, k, t, r, q, v: bs.price(s, k, t, r, q, v, kind)
    numeric = -fd(f, 2, (S, K, T, R, Q, vol), h=1e-5)
    assert bs.theta(S, K, T, R, Q, vol, kind) == pytest.approx(numeric, abs=1e-4)


@pytest.mark.parametrize("S,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_rho_is_dprice_drate(S, T, vol, kind):
    f = lambda s, k, t, r, q, v: bs.price(s, k, t, r, q, v, kind)
    numeric = fd(f, 3, (S, K, T, R, Q, vol), h=1e-6)
    assert bs.rho(S, K, T, R, Q, vol, kind) == pytest.approx(numeric, abs=1e-4)


@pytest.mark.parametrize("S,T,vol", CASES)
def test_vanna_is_ddelta_dvol(S, T, vol):
    f = lambda s, k, t, r, q, v: bs.delta(s, k, t, r, q, v, "call")
    numeric = fd(f, 5, (S, K, T, R, Q, vol), h=1e-5)
    assert bs.vanna(S, K, T, R, Q, vol) == pytest.approx(numeric, abs=1e-5)


@pytest.mark.parametrize("S,T,vol", CASES)
def test_volga_is_dvega_dvol(S, T, vol):
    f = lambda s, k, t, r, q, v: bs.vega(s, k, t, r, q, v, "call")
    numeric = fd(f, 5, (S, K, T, R, Q, vol), h=1e-5)
    assert bs.volga(S, K, T, R, Q, vol) == pytest.approx(numeric, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("S,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_charm_is_ddelta_dcalendar_time(S, T, vol, kind):
    f = lambda s, k, t, r, q, v: bs.delta(s, k, t, r, q, v, kind)
    numeric = -fd(f, 2, (S, K, T, R, Q, vol), h=1e-6)
    assert bs.charm(S, K, T, R, Q, vol, kind) == pytest.approx(numeric, abs=1e-4)


# --- limits and edge cases -------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
def test_expired_option_is_worth_intrinsic(kind):
    for S in (80.0, 100.0, 125.0):
        assert bs.price(S, K, 0.0, R, Q, 0.3, kind) == pytest.approx(
            bs.intrinsic(S, K, kind)
        )


def test_zero_vol_is_discounted_forward_intrinsic():
    S, T = 120.0, 1.0
    fwd = S * np.exp((R - Q) * T)
    expected = np.exp(-R * T) * max(fwd - K, 0.0)
    assert bs.price(S, K, T, R, Q, 0.0, "call") == pytest.approx(expected, abs=1e-8)


def test_deep_otm_is_worthless_not_negative():
    assert bs.price(10.0, K, 0.05, R, Q, 0.1, "call") == pytest.approx(0.0, abs=1e-12)
    assert bs.price(500.0, K, 0.05, R, Q, 0.1, "put") == pytest.approx(0.0, abs=1e-12)


def test_gamma_and_vega_vanish_at_expiry():
    assert bs.gamma(100.0, K, 0.0, R, Q, 0.2) == 0.0
    assert bs.vega(100.0, K, 0.0, R, Q, 0.2) == 0.0


def test_functions_broadcast_over_arrays():
    spots = np.linspace(60, 140, 41)
    mats = np.array([[0.1], [1.0]])
    grid = bs.gamma(spots, K, mats, R, Q, 0.25)
    assert grid.shape == (2, 41)
    # Short-dated gamma is more concentrated: taller at the money, thinner in the wings.
    atm = np.argmin(np.abs(spots - K))
    assert grid[0, atm] > grid[1, atm]
    assert grid[0, 0] < grid[1, 0]


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trips(kind):
    """Inverting a price must reproduce the price to machine precision.

    It need not reproduce the *vol* that precisely. Where vega is tiny (deep
    in- or out-of-the-money, close to expiry) the price barely depends on vol,
    so many vols price identically and the inversion is ill-conditioned. That
    is a property of the option, not a defect of the solver, and the app warns
    about it on the implied-vol page.
    """
    for S, T, vol in CASES:
        target = bs.price(S, K, T, R, Q, vol, kind)
        if target < 1e-8:
            continue  # no information in a worthless price
        recovered = bs.implied_vol(target, S, K, T, R, Q, kind)
        assert bs.price(S, K, T, R, Q, recovered, kind) == pytest.approx(target, abs=1e-8)
        if bs.vega(S, K, T, R, Q, vol) > 1e-4:
            assert recovered == pytest.approx(vol, abs=1e-6)


def test_implied_vol_is_ill_conditioned_when_vega_vanishes():
    """Document the limit above: a price with no vega sensitivity pins no vol."""
    S, T, vol = 125.0, 0.08, 0.12
    assert bs.vega(S, K, T, R, Q, vol) < 1e-6
    target = bs.price(S, K, T, R, Q, vol, "call")
    recovered = bs.implied_vol(target, S, K, T, R, Q, "call")
    assert bs.price(S, K, T, R, Q, recovered, "call") == pytest.approx(target, abs=1e-10)


def test_implied_vol_returns_nan_outside_arbitrage_bounds():
    assert np.isnan(bs.implied_vol(-1.0, 100.0, K, 1.0, R, Q, "call"))
    assert np.isnan(bs.implied_vol(1e9, 100.0, K, 1.0, R, Q, "call"))


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        bs.price(100.0, K, 1.0, R, Q, 0.2, "straddle")
