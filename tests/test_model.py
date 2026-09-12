"""Tests for positions, hedging and the smile.

These check *economic* properties, not just arithmetic: a hedge that works has
to earn gamma and pay theta, a fair-vol hedge has to break even, and hedging
more often has to reduce error in the specific way theory predicts.
"""

import numpy as np
import pytest

from optlab import bs, hedge, smile
from optlab.strategies import (
    PRESETS, Leg, Market, breakevens, cost, first_expiry, max_expiry,
    net_greek_curve, net_greeks, payoff, payoff_extremes, preset_legs, value_at,
)

MKT = Market(spot=100.0, rate=0.04, div_yield=0.0, vol=0.20)


# --- positions ------------------------------------------------------------


@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_builds_and_prices(name):
    legs = preset_legs(name, MKT)
    assert legs
    assert np.isfinite(cost(legs, MKT))
    g = net_greeks(legs, MKT)
    assert all(np.isfinite(v) for v in g.values())


@pytest.mark.parametrize("name", PRESETS)
def test_payoff_is_finite_across_a_wide_spot_range(name):
    legs = preset_legs(name, MKT)
    y = payoff(legs, np.linspace(1.0, 400.0, 500), MKT)
    assert np.all(np.isfinite(y))


def test_a_single_leg_position_agrees_with_the_pricer():
    legs = [Leg(bs.CALL, 1.0, 105.0, 0.75)]
    direct = bs.price(MKT.spot, 105.0, 0.75, MKT.rate, MKT.div_yield, MKT.vol, bs.CALL)
    assert cost(legs, MKT) == pytest.approx(direct)


def test_quantity_scales_the_position_linearly():
    one = [Leg(bs.CALL, 1.0, 100.0, 0.5)]
    three = [Leg(bs.CALL, 3.0, 100.0, 0.5)]
    assert cost(three, MKT) == pytest.approx(3 * cost(one, MKT))
    assert net_greeks(three, MKT)["vega"] == pytest.approx(3 * net_greeks(one, MKT)["vega"])


def test_short_position_is_the_mirror_of_the_long_one():
    long = [Leg(bs.PUT, 1.0, 95.0, 0.4)]
    short = [Leg(bs.PUT, -1.0, 95.0, 0.4)]
    assert cost(short, MKT) == pytest.approx(-cost(long, MKT))
    for name, value in net_greeks(long, MKT).items():
        assert net_greeks(short, MKT)[name] == pytest.approx(-value)


def test_stock_leg_has_unit_delta_and_no_curvature():
    g = net_greeks([Leg("stock", 1.0)], MKT)
    assert g["delta"] == pytest.approx(1.0)
    assert g["gamma"] == pytest.approx(0.0)
    assert g["vega"] == pytest.approx(0.0)
    assert g["theta"] == pytest.approx(0.0)


def test_put_call_parity_holds_for_a_synthetic_forward():
    """Long call + short put at one strike is a forward: delta 1, no gamma, no vega."""
    legs = [Leg(bs.CALL, 1.0, 100.0, 1.0), Leg(bs.PUT, -1.0, 100.0, 1.0)]
    g = net_greeks(legs, MKT)
    assert g["delta"] == pytest.approx(np.exp(-MKT.div_yield * 1.0), abs=1e-9)
    assert g["gamma"] == pytest.approx(0.0, abs=1e-12)
    assert g["vega"] == pytest.approx(0.0, abs=1e-9)


def test_a_bought_spread_costs_money_and_a_sold_one_pays():
    assert cost(preset_legs("Bull call spread", MKT), MKT) > 0
    assert cost(preset_legs("Iron condor", MKT), MKT) < 0  # a net credit


def test_long_straddle_is_long_gamma_and_pays_theta():
    g = net_greeks(preset_legs("Straddle", MKT), MKT)
    assert g["gamma"] > 0
    assert g["theta"] < 0
    assert g["vega"] > 0


def test_iron_condor_is_short_gamma_and_collects_theta():
    g = net_greeks(preset_legs("Iron condor", MKT), MKT)
    assert g["gamma"] < 0
    assert g["theta"] > 0


def test_straddle_breaks_even_on_both_sides():
    be = breakevens(preset_legs("Straddle", MKT), MKT)
    assert len(be) == 2
    assert be[0] < MKT.spot < be[1]


def test_unlimited_upside_reports_as_unbounded_but_downside_never_does():
    best, worst = payoff_extremes(preset_legs("Long call", MKT), MKT)
    assert best is None  # a call's upside does not stop
    assert worst == pytest.approx(-cost(preset_legs("Long call", MKT), MKT))


def test_capped_spread_reports_both_bounds():
    best, worst = payoff_extremes(preset_legs("Bull call spread", MKT), MKT)
    assert best is not None and worst is not None
    assert best > 0 > worst


def test_calendar_payoff_is_drawn_at_the_front_expiry():
    """At the back expiry a same-strike calendar cancels; at the front it is a tent."""
    legs = preset_legs("Calendar spread", MKT)
    assert first_expiry(legs) < max_expiry(legs)
    best, _ = payoff_extremes(legs, MKT)
    assert best is not None and best > 0

    at_back = value_at(legs, np.array([80.0, 100.0, 120.0]), max_expiry(legs), MKT)
    assert np.allclose(at_back, 0.0, atol=1e-9)


def test_value_decays_toward_payoff_as_time_passes():
    legs = preset_legs("Long call", MKT)
    spots = np.linspace(70, 130, 61)
    now = value_at(legs, spots, 0.0, MKT)
    later = value_at(legs, spots, 0.49, MKT)
    expiry = value_at(legs, spots, 0.5, MKT)
    assert np.all(now >= later - 1e-9)
    assert np.all(later >= expiry - 1e-9)


def test_expired_leg_delta_is_a_step_function():
    legs = [Leg(bs.CALL, 1.0, 100.0, 0.5)]
    d = net_greek_curve(legs, np.array([90.0, 110.0]), "delta", 0.5, MKT)
    assert list(d) == [0.0, 1.0]


# --- hedging --------------------------------------------------------------

SHORT_CALL = [Leg(bs.CALL, -1.0, 100.0, 0.25)]


def test_hedging_at_the_vol_the_market_delivers_breaks_even():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, implied_vol=0.2, real_vol=0.2,
                                  n_steps=100, n_paths=600, seed=1))
    assert abs(r.mean_pnl) < 3 * r.std_pnl / np.sqrt(len(r.final_pnl))


def test_selling_vol_above_what_the_market_delivers_makes_money():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, implied_vol=0.35, real_vol=0.20,
                                  n_steps=200, n_paths=600, seed=2))
    assert r.mean_pnl > 0
    assert r.summary()["win_rate"] > 0.95  # the edge is nearly path-independent


def test_selling_vol_below_what_the_market_delivers_loses_money():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, implied_vol=0.15, real_vol=0.30,
                                  n_steps=200, n_paths=600, seed=3))
    assert r.mean_pnl < 0
    assert r.summary()["win_rate"] < 0.05


def test_a_long_option_is_the_exact_mirror():
    kw = dict(implied_vol=0.20, real_vol=0.32, n_steps=200, n_paths=400, seed=4)
    short = hedge.run(hedge.HedgeSpec(SHORT_CALL, **kw))
    long = hedge.run(hedge.HedgeSpec([Leg(bs.CALL, 1.0, 100.0, 0.25)], **kw))
    assert short.mean_pnl == pytest.approx(-long.mean_pnl, rel=1e-9)


def test_theory_predicts_the_simulated_mean():
    for iv, rv in [(0.30, 0.20), (0.20, 0.28), (0.25, 0.15)]:
        spec = hedge.HedgeSpec(SHORT_CALL, implied_vol=iv, real_vol=rv,
                               n_steps=250, n_paths=2000, seed=5)
        r = hedge.run(spec)
        stderr = r.std_pnl / np.sqrt(len(r.final_pnl))
        assert hedge.theoretical_edge(spec) == pytest.approx(r.mean_pnl, abs=6 * stderr + 0.05)


def test_theory_is_zero_when_implied_equals_realised():
    spec = hedge.HedgeSpec(SHORT_CALL, implied_vol=0.2, real_vol=0.2)
    assert hedge.theoretical_edge(spec) == 0.0


def test_hedging_error_falls_like_one_over_root_n():
    """Quadrupling the rebalance count should roughly halve the spread."""
    spreads = {}
    for n in (25, 100, 400):
        r = hedge.run(hedge.HedgeSpec(SHORT_CALL, implied_vol=0.2, real_vol=0.2,
                                      n_steps=n, n_paths=1500, seed=11))
        spreads[n] = r.std_pnl
    assert spreads[25] > spreads[100] > spreads[400]
    # std * sqrt(n) should be roughly constant across the three.
    scaled = [spreads[n] * np.sqrt(n) for n in (25, 100, 400)]
    assert max(scaled) / min(scaled) < 1.35


def test_transaction_costs_only_ever_subtract():
    kw = dict(implied_vol=0.2, real_vol=0.2, n_steps=100, n_paths=500, seed=13)
    free = hedge.run(hedge.HedgeSpec(SHORT_CALL, cost_rate=0.0, **kw))
    costly = hedge.run(hedge.HedgeSpec(SHORT_CALL, cost_rate=0.001, **kw))
    assert costly.mean_pnl < free.mean_pnl


def test_a_wider_no_trade_band_means_fewer_trades():
    kw = dict(implied_vol=0.2, real_vol=0.2, n_steps=150, n_paths=200, seed=17)
    tight = hedge.run(hedge.HedgeSpec(SHORT_CALL, band=0.0, **kw))
    loose = hedge.run(hedge.HedgeSpec(SHORT_CALL, band=0.10, **kw))
    assert (loose.steps["trade"] != 0).sum() < (tight.steps["trade"] != 0).sum()


def test_greek_attribution_reconstructs_the_hedged_pnl():
    """The five attribution terms must add up to the P&L actually booked.

    Gamma, theta, carry, hedge slippage and transaction costs. If these do not
    reconstruct the realised number, the story the app tells about where P&L
    comes from is not the story the simulation is running.
    """
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, implied_vol=0.2, real_vol=0.2,
                                  n_steps=400, n_paths=1, seed=19))
    final = float(r.final_pnl[0])
    attributed = float(r.steps["cum_attributed"].iloc[-1])
    assert attributed == pytest.approx(final, abs=0.05)


def test_a_fully_hedged_book_has_no_delta_slippage():
    """With no band, the hedge cancels the delta term exactly, every step."""
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, band=0.0, n_steps=60, n_paths=1, seed=21))
    assert float(r.steps["cum_slippage"].iloc[-1]) == pytest.approx(0.0, abs=1e-9)


def test_a_no_trade_band_leaves_delta_slippage_behind():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, band=0.15, n_steps=60, n_paths=1, seed=21))
    assert abs(float(r.steps["cum_slippage"].iloc[-1])) > 1e-6


def test_carry_is_the_term_that_rates_drive():
    """Set rates and dividends to zero and the financing term must vanish."""
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, rate=0.0, div_yield=0.0,
                                  n_steps=50, n_paths=1, seed=29))
    assert float(r.steps["cum_carry"].iloc[-1]) == pytest.approx(0.0, abs=1e-12)


def test_detail_for_any_path_matches_the_full_run():
    """Re-walking one path must reproduce its number from the vectorised run.

    This is the property the whole path picker rests on. If a path could
    influence another through the accounting, the detail the app shows would
    not be the detail of the path it claims.
    """
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, implied_vol=0.28, real_vol=0.20,
                                  n_steps=40, n_paths=120, cost_rate=0.0004,
                                  band=0.05, seed=31))
    for i in (0, 1, 17, 60, 119):
        detail = r.detail_for(i)
        assert float(detail["pnl"].iloc[-1]) == pytest.approx(float(r.final_pnl[i]), abs=1e-9)
        assert float(detail["spot"].iloc[-1]) == pytest.approx(float(r.paths[i, -1]))
        assert len(detail) == r.spec.n_steps + 1


def test_named_paths_pick_the_right_ones():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, n_steps=30, n_paths=200, seed=37))
    pnl = r.final_pnl
    assert pnl[r.path_index("Worst")] == pytest.approx(pnl.min())
    assert pnl[r.path_index("Best")] == pytest.approx(pnl.max())
    assert r.path_index("First") == 0
    median_gap = abs(pnl[r.path_index("Median")] - np.median(pnl))
    assert median_gap <= np.min(np.abs(pnl - np.median(pnl))) + 1e-12


def test_every_named_path_has_an_explanation():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, n_steps=20, n_paths=50, seed=39))
    for name in hedge.NOTABLE:
        assert hedge.NOTABLE[name]
        assert 0 <= r.path_index(name) < len(r.final_pnl)


def test_rank_orders_paths_from_worst_to_best():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, n_steps=20, n_paths=80, seed=41))
    assert r.rank_of(r.path_index("Worst")) == 1
    assert r.rank_of(r.path_index("Best")) == len(r.final_pnl)


def test_an_out_of_range_path_is_clamped_not_crashed():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, n_steps=15, n_paths=10, seed=43))
    assert len(r.detail_for(9999)) == r.spec.n_steps + 1
    assert len(r.detail_for(-5)) == r.spec.n_steps + 1


def test_default_detail_is_the_first_path():
    r = hedge.run(hedge.HedgeSpec(SHORT_CALL, n_steps=20, n_paths=30, seed=47))
    assert r.steps["pnl"].iloc[-1] == pytest.approx(float(r.final_pnl[0]), abs=1e-9)


def test_paths_start_at_spot_and_stay_positive():
    paths = hedge.simulate_paths(100.0, 0.05, 0.2, 1.0, 50, 20, seed=23)
    assert paths.shape == (20, 51)
    assert np.allclose(paths[:, 0], 100.0)
    assert np.all(paths > 0)


def test_simulation_is_reproducible_from_its_seed():
    kw = dict(spot=100.0, drift=0.05, vol=0.2, horizon=1.0, n_steps=20, n_paths=5)
    assert np.array_equal(hedge.simulate_paths(seed=3, **kw), hedge.simulate_paths(seed=3, **kw))
    assert not np.array_equal(hedge.simulate_paths(seed=3, **kw), hedge.simulate_paths(seed=4, **kw))


def test_a_zero_horizon_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        hedge.run(hedge.HedgeSpec([Leg(bs.CALL, -1.0, 100.0, 0.0)]))


# --- the 3-D surface ------------------------------------------------------


def test_surface_builds_for_every_greek():
    from optlab import ui

    spots = np.linspace(70.0, 130.0, 30)
    days = np.linspace(1.0, 180.0, 25)
    S, D = np.meshgrid(spots, days)
    for greek, fn in bs.GREEK_FUNCS.items():
        z = bs.scale_for_display(greek, fn(S, 100.0, D / 365.0, 0.04, 0.0, 0.2))
        fig = ui.surface(spots, days, z, x_title="Spot", y_title="Days", z_title=greek)
        assert fig.data[0].type == "surface"
        assert fig.data[0].z.shape == (len(days), len(spots))


def test_capping_lowers_the_drawn_height_but_not_the_reported_value():
    """A capped peak must still tell the truth when you hover it."""
    from optlab import ui

    spots = np.linspace(80.0, 120.0, 40)
    days = np.linspace(1.0, 90.0, 30)
    S, D = np.meshgrid(spots, days)
    z = bs.scale_for_display("gamma", bs.gamma(S, 100.0, D / 365.0, 0.04, 0.0, 0.2))
    cap = float(np.percentile(z, 90))
    fig = ui.surface(spots, days, z, x_title="Spot", y_title="Days", z_title="gamma",
                     cap=cap)
    assert np.nanmax(fig.data[0].z) == pytest.approx(cap)
    assert np.nanmax(fig.data[0].customdata) == pytest.approx(float(np.nanmax(z)))
    assert np.nanmax(z) > cap  # the test would be vacuous otherwise


def test_a_signed_surface_gets_a_symmetric_diverging_scale():
    from optlab import ui

    spots = np.linspace(80.0, 120.0, 20)
    days = np.linspace(1.0, 90.0, 15)
    S, D = np.meshgrid(spots, days)
    z = bs.scale_for_display("theta", bs.theta(S, 100.0, D / 365.0, 0.04, 0.0, 0.2))
    fig = ui.surface(spots, days, z, x_title="Spot", y_title="Days", z_title="theta",
                     diverging=True)
    assert fig.data[0].cmin == pytest.approx(-fig.data[0].cmax)


def test_an_all_negative_field_darkens_toward_its_extreme():
    """Theta is negative everywhere, and its deepest point must be the darkest.

    Mapping the raw range light-to-dark would paint the most negative theta
    palest and the near-zero corner darkest, which reads exactly backwards.
    """
    from optlab import ui

    up_steps, up_domain = ui.sequential_scale(0.0, 0.25)
    down_steps, down_domain = ui.sequential_scale(-0.14, 0.0)
    assert down_steps == up_steps[::-1]
    assert down_domain[1] == 0.0 and down_domain[0] < 0
    assert up_domain[0] == 0.0 and up_domain[1] > 0


def test_the_theta_surface_is_coloured_from_zero_downward():
    from optlab import ui

    spots = np.linspace(80.0, 120.0, 20)
    days = np.linspace(1.0, 120.0, 15)
    S, D = np.meshgrid(spots, days)
    z = bs.scale_for_display("theta", bs.theta(S, 100.0, D / 365.0, 0.04, 0.0, 0.2))
    assert np.nanmax(z) <= 0  # the premise: this really is an all-negative field
    fig = ui.surface(spots, days, z, x_title="S", y_title="D", z_title="theta")
    assert fig.data[0].cmax == pytest.approx(0.0)
    assert fig.data[0].cmin < 0
    darkest = fig.data[0].colorscale[0][1]  # the colour at cmin, the deepest theta
    lightest = fig.data[0].colorscale[-1][1]
    assert darkest == ui.colors()["seq"][-1]
    assert lightest == ui.colors()["seq"][0]


def test_signed_extreme_keeps_the_sign():
    from optlab import ui

    assert ui.signed_extreme([0.1, -0.9, 0.4]) == pytest.approx(-0.9)
    assert ui.signed_extreme([0.1, 0.9, -0.4]) == pytest.approx(0.9)
    assert ui.signed_extreme([np.nan, -2.0, np.nan]) == pytest.approx(-2.0)
    assert ui.signed_extreme([]) == 0.0

    # The case it exists for: theta's extreme is negative, not positive.
    S, D = np.meshgrid(np.linspace(80.0, 120.0, 20), np.linspace(1.0, 90.0, 15))
    z = bs.scale_for_display("theta", bs.theta(S, 100.0, D / 365.0, 0.04, 0.0, 0.2))
    assert ui.signed_extreme(z) < 0


def test_colorscales_run_light_to_dark_and_pole_to_pole():
    from optlab import ui

    seq = ui.plotly_colorscale(diverging=False)
    assert seq[0][0] == 0.0 and seq[-1][0] == 1.0
    div = ui.plotly_colorscale(diverging=True)
    assert [stop for stop, _ in div] == [0.0, 0.5, 1.0]


# --- smile ----------------------------------------------------------------


def test_flat_smile_reproduces_black_scholes():
    strikes = np.array([80.0, 100.0, 120.0])
    vols = smile.smile_vol(strikes, 100.0, 0.5, 0.04, 0.0, atm_vol=0.22)
    assert np.allclose(vols, 0.22)


def test_negative_skew_makes_downside_strikes_dearer_in_vol():
    strikes = np.array([80.0, 100.0, 120.0])
    vols = smile.smile_vol(strikes, 100.0, 0.5, 0.04, 0.0, atm_vol=0.2, skew=-0.4)
    assert vols[0] > vols[1] > vols[2]


def test_curvature_lifts_both_wings():
    strikes = np.array([80.0, 100.0, 120.0])
    vols = smile.smile_vol(strikes, 100.0, 0.5, 0.04, 0.0, atm_vol=0.2, curvature=1.5)
    assert vols[0] > vols[1] and vols[2] > vols[1]


def test_vol_never_goes_negative_however_extreme_the_parameters():
    strikes = np.linspace(1.0, 1000.0, 400)
    vols = smile.smile_vol(strikes, 100.0, 1.0, 0.03, 0.0, atm_vol=0.2, skew=-3.0)
    assert np.all(vols >= smile.MIN_VOL)


def test_at_the_money_forward_vol_is_the_atm_parameter():
    fwd = smile.forward(100.0, 0.5, 0.04, 0.01)
    assert smile.smile_vol(fwd, 100.0, 0.5, 0.04, 0.01, atm_vol=0.27) == pytest.approx(0.27)


def test_the_two_stickiness_rules_agree_only_when_the_smile_is_flat():
    args = (np.array([90.0, 100.0, 110.0]), 100.0, 0.5, 0.04, 0.0)
    flat_a = smile.smile_delta(*args, regime=smile.STICKY_STRIKE, atm_vol=0.2)
    flat_b = smile.smile_delta(*args, regime=smile.STICKY_MONEYNESS, atm_vol=0.2)
    assert np.allclose(flat_a, flat_b)

    skewed_a = smile.smile_delta(*args, regime=smile.STICKY_STRIKE, atm_vol=0.2, skew=-0.4)
    skewed_b = smile.smile_delta(*args, regime=smile.STICKY_MONEYNESS, atm_vol=0.2, skew=-0.4)
    assert not np.allclose(skewed_a, skewed_b)


def test_equity_skew_raises_call_delta_under_sticky_moneyness():
    """Spot up means a fixed strike drops to lower moneyness, where vol is richer.

    Moneyness is log(K / forward). A rising spot pushes a fixed strike to a
    lower k, and under negative skew lower k carries higher vol. Vega is
    positive, so the extra channel adds to delta rather than subtracting.
    """
    args = (np.array([100.0]), 100.0, 0.5, 0.04, 0.0)
    kw = dict(atm_vol=0.2, skew=-0.4)
    strike_rule = smile.smile_delta(*args, regime=smile.STICKY_STRIKE, **kw)
    money_rule = smile.smile_delta(*args, regime=smile.STICKY_MONEYNESS, **kw)
    assert money_rule[0] > strike_rule[0]


def test_upside_skew_flips_the_adjustment_the_other_way():
    args = (np.array([100.0]), 100.0, 0.5, 0.04, 0.0)
    kw = dict(atm_vol=0.2, skew=0.4)
    strike_rule = smile.smile_delta(*args, regime=smile.STICKY_STRIKE, **kw)
    money_rule = smile.smile_delta(*args, regime=smile.STICKY_MONEYNESS, **kw)
    assert money_rule[0] < strike_rule[0]


def test_sticky_strike_reproduces_the_leverage_effect():
    """Spot up, at-the-money vol down - the effect sticky strike gives for free."""
    kw = dict(atm_vol=0.2, skew=-0.4)
    T, r, q = 0.5, 0.04, 0.0
    low_spot, high_spot = 95.0, 105.0
    # Under sticky strike the smile is pinned to the ORIGINAL forward, so read
    # each spot's own at-the-money strike off that one fixed curve.
    atm_vol_low = smile.smile_vol(low_spot, 100.0, T, r, q, **kw)
    atm_vol_high = smile.smile_vol(high_spot, 100.0, T, r, q, **kw)
    assert atm_vol_high < atm_vol_low


def test_smile_prices_fall_as_strike_rises_for_calls():
    strikes = np.linspace(70, 140, 30)
    prices = smile.smile_price(strikes, 100.0, 0.5, 0.04, 0.0, bs.CALL, atm_vol=0.2, skew=-0.3)
    assert np.all(np.diff(prices) < 0)


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError):
        smile.smile_delta(np.array([100.0]), 100.0, 0.5, 0.04, 0.0, regime="sticky vibes")


@pytest.mark.parametrize("shape", list(smile.SHAPES))
def test_every_named_shape_produces_sane_vols(shape):
    vols = smile.smile_vol(np.linspace(60, 160, 50), 100.0, 0.5, 0.04, 0.0,
                           atm_vol=0.2, **smile.SHAPES[shape])
    assert np.all(np.isfinite(vols)) and np.all(vols > 0)
    assert shape in smile.SHAPE_NOTES
