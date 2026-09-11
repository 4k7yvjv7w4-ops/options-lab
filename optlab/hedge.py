"""Delta-hedging simulation: where the greeks stop being abstract.

The lesson this module exists to deliver, in one line:

    a delta-hedged option position earns gamma and pays theta, and nothing else.

Everything here is arranged to make that visible. Each rebalancing period the
hedged P&L is

    1/2 * Gamma * dS^2   +   Theta * dt

The first term is what the spot actually did. The second is what you paid for
the privilege, priced off implied vol. Their race is the entire game: if
realised movement beats the implied movement you were charged for, a long
option position wins, and a short one loses.

Two frictions spoil the clean story, and both are switchable here so you can
see what each one costs:

  * You cannot hedge continuously. Hedging N times a period leaves a random
    error whose size falls roughly like 1/sqrt(N) - so the edge is a mean and
    the discreteness is the variance around it.
  * Every trade pays the spread. Hedging more often shrinks the error and
    raises the bill, which is why the best hedge frequency is finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from . import bs
from .strategies import Leg, Market, net_greek_curve, value_at


@dataclass
class HedgeSpec:
    """Everything the simulator needs. Defaults describe the canonical lesson.

    The default is a short at-the-money call sold at 20 vol into a market that
    actually moves at 20 vol - a fair fight, where the expected P&L is zero and
    all you see is hedging noise. Move `real_vol` away from `implied_vol` and
    the edge appears.
    """

    legs: Sequence[Leg]
    implied_vol: float = 0.20  # the vol the position was priced and is hedged at
    real_vol: float = 0.20  # the vol the spot path actually moves at
    drift: float = 0.05  # real-world drift of the spot
    rate: float = 0.04
    div_yield: float = 0.0
    spot: float = 100.0
    horizon: float | None = None  # defaults to the position's first expiry
    n_steps: int = 50  # rebalances over the horizon
    cost_rate: float = 0.0  # transaction cost as a fraction of traded notional
    band: float = 0.0  # only rebalance once delta drifts this far (0 = always)
    n_paths: int = 400
    seed: int = 7

    def market(self, spot: float | None = None) -> Market:
        return Market(
            spot=self.spot if spot is None else spot,
            rate=self.rate,
            div_yield=self.div_yield,
            vol=self.implied_vol,
        )


@dataclass
class HedgeResult:
    """Output of a run: one path in detail, plus the distribution over many."""

    steps: pd.DataFrame  # per-rebalance detail for the first path
    final_pnl: np.ndarray  # final P&L of every path
    paths: np.ndarray  # the simulated spot paths themselves
    times: np.ndarray
    premium: float  # what the position cost to put on
    spec: HedgeSpec = field(repr=False)

    @property
    def mean_pnl(self) -> float:
        return float(np.mean(self.final_pnl))

    @property
    def std_pnl(self) -> float:
        return float(np.std(self.final_pnl, ddof=1)) if len(self.final_pnl) > 1 else 0.0

    def summary(self) -> dict:
        p = self.final_pnl
        return {
            "premium": self.premium,
            "mean": self.mean_pnl,
            "std": self.std_pnl,
            "p05": float(np.percentile(p, 5)),
            "p50": float(np.percentile(p, 50)),
            "p95": float(np.percentile(p, 95)),
            "worst": float(np.min(p)),
            "best": float(np.max(p)),
            "win_rate": float(np.mean(p > 0)),
            "total_cost": float(np.mean(self.steps["tx_cost"].sum())),
        }


def simulate_paths(
    spot: float, drift: float, vol: float, horizon: float,
    n_steps: int, n_paths: int, seed: int = 7,
) -> np.ndarray:
    """Geometric Brownian motion, exact in the log. Shape (n_paths, n_steps+1).

    Exact rather than Euler-stepped: the log-price increment of GBM is normal
    with known mean and variance, so there is no discretisation error in the
    path itself. Any error you see in the results is hedging error, which is
    the thing we are trying to measure.
    """
    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    log_increments = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
    log_path = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return spot * np.exp(log_path)


def run(spec: HedgeSpec) -> HedgeResult:
    """Hedge the position along every simulated path and account for the P&L.

    The book is kept in three pieces - the option position, the stock hedge and
    a cash account - and it starts at exactly zero value, so whatever it is
    worth at the end IS the profit and loss.
    """
    horizon = spec.horizon
    if horizon is None:
        from .strategies import first_expiry

        horizon = first_expiry(spec.legs)
    if horizon <= 0:
        raise ValueError("hedging horizon must be positive")

    n, dt = spec.n_steps, horizon / spec.n_steps
    paths = simulate_paths(
        spec.spot, spec.drift, spec.real_vol, horizon, n, spec.n_paths, spec.seed
    )
    times = np.linspace(0.0, horizon, n + 1)
    mkt = spec.market()
    growth = np.exp(spec.rate * dt)
    div_accrual = np.exp(spec.div_yield * dt) - 1.0

    def position_value(spots, t):
        return np.asarray(value_at(spec.legs, spots, t, mkt), dtype=float)

    def position_greek(spots, name, t):
        return np.asarray(net_greek_curve(spec.legs, spots, name, t, mkt), dtype=float)

    # --- open the position ------------------------------------------------
    S0 = paths[:, 0]
    value = position_value(S0, 0.0)
    delta = position_greek(S0, "delta", 0.0)
    shares = -delta  # hold the offsetting stock: position delta + hedge = 0
    tx = np.abs(shares) * S0 * spec.cost_rate
    cash = -(value + shares * S0) - tx  # makes the book worth zero at inception
    premium = float(value[0])

    rows = [{
        "step": 0, "t": 0.0, "spot": float(S0[0]),
        "position_value": float(value[0]), "delta": float(delta[0]),
        "gamma": float(position_greek(S0, "gamma", 0.0)[0]),
        "shares": float(shares[0]), "trade": float(shares[0]), "tx_cost": float(tx[0]),
        "cash": float(cash[0]), "pnl": float(value[0] + shares[0] * S0[0] + cash[0]),
        "delta_term": 0.0, "gamma_term": 0.0, "theta_term": 0.0,
        "hedge_term": 0.0, "carry_term": 0.0,
    }]

    # --- walk the path ----------------------------------------------------
    for i in range(1, n + 1):
        t = times[i]
        S_prev, S_now = paths[:, i - 1], paths[:, i]
        dS = S_now - S_prev
        shares_prev = shares

        # Attribution of the period's P&L, every greek read at the START of the
        # period. The option's own value change is the Taylor expansion
        #     dV ~= delta*dS + 1/2*gamma*dS^2 + theta*dt
        # and the book adds two more pieces: what the stock hedge made on the
        # same move, and the financing on the cash and stock the book carries.
        gamma_prev = position_greek(S_prev, "gamma", times[i - 1])
        theta_prev = position_greek(S_prev, "theta", times[i - 1])
        delta_prev = position_greek(S_prev, "delta", times[i - 1])
        delta_term = delta_prev * dS
        gamma_term = 0.5 * gamma_prev * dS**2
        theta_term = theta_prev * dt
        hedge_term = shares_prev * dS  # cancels delta_term exactly when fully hedged
        carry_term = cash * (growth - 1.0) + shares_prev * S_prev * div_accrual

        cash = cash * growth + shares_prev * S_prev * div_accrual
        value = position_value(S_now, t)
        target = -position_greek(S_now, "delta", t)

        # A band means: leave the hedge alone until it has drifted far enough
        # to be worth the spread. Zero band rebalances every step.
        if spec.band > 0:
            move = np.abs(target - shares) > spec.band
            new_shares = np.where(move, target, shares)
        else:
            new_shares = target
        trade = new_shares - shares
        step_tx = np.abs(trade) * S_now * spec.cost_rate
        cash = cash - trade * S_now - step_tx
        shares = new_shares

        rows.append({
            "step": i, "t": float(t), "spot": float(S_now[0]),
            "position_value": float(value[0]), "delta": float(-target[0]),
            "gamma": float(gamma_prev[0]), "shares": float(shares[0]),
            "trade": float(trade[0]), "tx_cost": float(step_tx[0]), "cash": float(cash[0]),
            "pnl": float(value[0] + shares[0] * S_now[0] + cash[0]),
            "delta_term": float(delta_term[0]), "gamma_term": float(gamma_term[0]),
            "theta_term": float(theta_term[0]), "hedge_term": float(hedge_term[0]),
            "carry_term": float(carry_term[0]),
        })

    # --- close out --------------------------------------------------------
    S_end = paths[:, -1]
    unwind_tx = np.abs(shares) * S_end * spec.cost_rate
    final_pnl = position_value(S_end, horizon) + shares * S_end + cash - unwind_tx

    steps = pd.DataFrame(rows)
    for term in ("gamma", "theta", "carry", "delta", "hedge"):
        steps[f"cum_{term}"] = steps[f"{term}_term"].cumsum()
    steps["cum_tx_cost"] = steps["tx_cost"].cumsum()
    # What the hedge leaves behind when delta is fully neutralised: the delta
    # and hedge terms cancel, so only gamma, theta and carry survive.
    steps["cum_slippage"] = steps["cum_delta"] + steps["cum_hedge"]
    steps["cum_attributed"] = (
        steps["cum_gamma"] + steps["cum_theta"] + steps["cum_carry"]
        + steps["cum_slippage"] - steps["cum_tx_cost"]
    )
    steps["cum_gamma_theta"] = steps["cum_gamma"] + steps["cum_theta"]
    steps["moneyness"] = steps["spot"] / _reference_strike(spec.legs)

    return HedgeResult(
        steps=steps, final_pnl=np.asarray(final_pnl, dtype=float),
        paths=paths, times=times, premium=premium, spec=spec,
    )


def _reference_strike(legs: Sequence[Leg]) -> float:
    strikes = [leg.strike for leg in legs if leg.kind != "stock"]
    return float(np.mean(strikes)) if strikes else 1.0


def theoretical_edge(spec: HedgeSpec, horizon: float | None = None, n_time: int = 48,
                     n_nodes: int = 161) -> float:
    """Expected P&L of the hedge, before discreteness and transaction costs.

    A delta-hedged position collects, over each instant,

        1/2 * Gamma * S^2 * (realised variance - implied variance) * dt

    so the expected total is that quantity averaged over where the spot is
    likely to be. Averaging matters: gamma is a sharp peak at the strike, so
    evaluating it at the *average* spot badly overstates it. Jensen's
    inequality is not a rounding error here - it is the difference between a
    number that matches the simulation and one that is half again too big. So
    this integrates over the real-world lognormal distribution of the spot at
    each time, which is what makes it line up with the Monte Carlo mean.
    """
    if horizon is None:
        from .strategies import first_expiry

        horizon = first_expiry(spec.legs)
    if horizon <= 0:
        return 0.0

    mkt = spec.market()
    var_gap = spec.real_vol**2 - spec.implied_vol**2
    if var_gap == 0.0:
        return 0.0

    # Gauss-Hermite nodes over the log-spot distribution at each time slice.
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()

    dt = horizon / n_time
    ts = (np.arange(n_time) + 0.5) * dt  # midpoint rule in time
    total = 0.0
    for t in ts:
        sd = spec.real_vol * np.sqrt(t)
        mean_log = np.log(spec.spot) + (spec.drift - 0.5 * spec.real_vol**2) * t
        spots = np.exp(mean_log + sd * z)
        g = np.asarray(net_greek_curve(spec.legs, spots, "gamma", t, mkt), dtype=float)
        total += 0.5 * float(np.sum(w * g * spots**2)) * var_gap * dt
    return float(total)
