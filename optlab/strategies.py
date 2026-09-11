"""Multi-leg positions: value them, find their payoff, add up their greeks.

The whole page of payoff diagrams rests on one idea: a position is just a list
of legs, and *every* curve the app draws is the same function evaluated at a
different point in time.

    value_at(legs, spots, t=0)          -> what it is worth today
    value_at(legs, spots, t=expiry)     -> the hockey-stick payoff diagram

Because legs carry their own expiry, that single function also handles calendar
spreads correctly: at the horizon the front leg is intrinsic while the back leg
is still priced with time value remaining.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from . import bs

STOCK = "stock"
CASH = "cash"


@dataclass(frozen=True)
class Market:
    """The environment every leg is priced in."""

    spot: float = 100.0
    rate: float = 0.04
    div_yield: float = 0.0
    vol: float = 0.20


@dataclass(frozen=True)
class Leg:
    """One instrument in a position.

    qty is signed: +1 is long one contract, -1 is short one. `vol=None` means
    "use the market vol", which is what you want until you start pricing each
    strike on its own point of the smile.
    """

    kind: str  # "call" | "put" | "stock"
    qty: float = 1.0
    strike: float = 100.0
    expiry: float = 1.0  # years from today
    vol: float | None = None

    def __post_init__(self):
        if self.kind not in (bs.CALL, bs.PUT, STOCK):
            raise ValueError(f"unknown leg kind {self.kind!r}")

    @property
    def label(self) -> str:
        side = "long" if self.qty > 0 else "short"
        size = abs(self.qty)
        size_txt = f"{size:g}x " if size != 1 else ""
        if self.kind == STOCK:
            return f"{side} {size_txt}stock"
        return f"{side} {size_txt}{self.strike:g} {self.kind}"


def leg_vol(leg: Leg, market: Market) -> float:
    return market.vol if leg.vol is None else leg.vol


def value_at(legs: Sequence[Leg], spots, t: float = 0.0, market: Market | None = None):
    """Mark-to-market value of the position at calendar time `t`, per spot level.

    `t` is measured forward from today, so t=0 is now and t=leg.expiry is that
    leg's expiry. A leg past its expiry is worth its intrinsic value.
    """
    market = market or Market()
    spots = np.asarray(spots, dtype=float)
    total = np.zeros_like(spots)
    for leg in legs:
        total = total + leg.qty * _leg_value(leg, spots, t, market)
    return total


def _leg_value(leg: Leg, spots: np.ndarray, t: float, market: Market) -> np.ndarray:
    if leg.kind == STOCK:
        return spots
    remaining = max(leg.expiry - t, 0.0)
    if remaining <= 0:
        return np.asarray(bs.intrinsic(spots, leg.strike, leg.kind), dtype=float)
    return np.asarray(
        bs.price(
            spots, leg.strike, remaining, market.rate, market.div_yield,
            leg_vol(leg, market), leg.kind,
        ),
        dtype=float,
    )


def cost(legs: Sequence[Leg], market: Market | None = None) -> float:
    """What the position costs to put on today. Negative means a net credit."""
    market = market or Market()
    return float(value_at(legs, market.spot, 0.0, market))


def payoff(legs: Sequence[Leg], spots, market: Market | None = None):
    """The classic payoff diagram: P&L at the FIRST expiry in the position.

    First, not last, because that is the moment the position's character
    changes. For a single-expiry position the two are the same. For a calendar
    spread they are emphatically not: at the back leg's expiry both legs are
    intrinsic and a same-strike calendar cancels itself exactly, drawing a flat
    line at minus the premium. At the front expiry, where the short leg dies
    and the long leg still holds time value, you see the tent the trade is
    actually built for.
    """
    market = market or Market()
    return value_at(legs, spots, first_expiry(legs), market) - cost(legs, market)


def pnl_at(legs: Sequence[Leg], spots, t: float, market: Market | None = None):
    """Profit and loss at time `t`, net of the opening cost."""
    market = market or Market()
    return value_at(legs, spots, t, market) - cost(legs, market)


def max_expiry(legs: Sequence[Leg]) -> float:
    """The last expiry in the position: how long it can stay open."""
    opts = [leg.expiry for leg in legs if leg.kind != STOCK]
    return max(opts) if opts else 0.0


def first_expiry(legs: Sequence[Leg]) -> float:
    """The first expiry in the position: when its shape changes."""
    opts = [leg.expiry for leg in legs if leg.kind != STOCK]
    return min(opts) if opts else 0.0


def net_greek_curve(
    legs: Sequence[Leg], spots, name: str, t: float = 0.0, market: Market | None = None
):
    """One position greek across an array of spot levels.

    This is the primitive the whole app plots: net delta as the spot moves, the
    gamma hump that a hedger has to keep up with, the theta the position pays.
    It broadcasts, so the hedging Monte Carlo can price thousands of paths at
    once without a Python loop.
    """
    market = market or Market()
    spots = np.asarray(spots, dtype=float)
    total = np.zeros_like(spots)
    for leg in legs:
        total = total + leg.qty * _leg_greek(leg, name, spots, t, market)
    return total


def net_greeks(
    legs: Sequence[Leg], market: Market | None = None, t: float = 0.0, which=None
) -> dict:
    """Position greeks at the current spot, as a plain dict.

    Stock has delta 1 and no other greek, which is exactly why it is the
    hedging instrument.
    """
    market = market or Market()
    names = which or ["price", "delta", "gamma", "vega", "theta", "rho"]
    return {
        n: float(net_greek_curve(legs, market.spot, n, t, market)) for n in names
    }


def _leg_greek(leg: Leg, name: str, spots: np.ndarray, t: float, market: Market):
    """One leg's greek, per unit, across spot levels."""
    spots = np.asarray(spots, dtype=float)
    if leg.kind == STOCK:
        if name == "delta":
            return np.ones_like(spots)
        if name == "price":
            return spots
        return np.zeros_like(spots)

    remaining = max(leg.expiry - t, 0.0)
    if remaining <= 0:
        # An expired leg has intrinsic value, a step-function delta, and no
        # other sensitivity left at all.
        if name == "price":
            return np.asarray(bs.intrinsic(spots, leg.strike, leg.kind), dtype=float)
        if name == "delta":
            if leg.kind == bs.CALL:
                return np.where(spots > leg.strike, 1.0, 0.0)
            return np.where(spots < leg.strike, -1.0, 0.0)
        return np.zeros_like(spots)

    return np.asarray(
        bs.GREEK_FUNCS[name](
            spots, leg.strike, remaining, market.rate, market.div_yield,
            leg_vol(leg, market), leg.kind,
        ),
        dtype=float,
    )


def breakevens(legs: Sequence[Leg], market: Market | None = None, span=(0.0, 3.0), n=4001):
    """Spot levels where the expiry P&L crosses zero, found by sign change."""
    market = market or Market()
    lo, hi = span[0] * market.spot, span[1] * market.spot
    spots = np.linspace(lo, hi, n)
    y = np.asarray(payoff(legs, spots, market), dtype=float)
    sign_change = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    roots = []
    for i in sign_change:
        x0, x1, y0, y1 = spots[i], spots[i + 1], y[i], y[i + 1]
        roots.append(x0 - y0 * (x1 - x0) / (y1 - y0))  # linear interpolation
    return roots


def payoff_extremes(legs: Sequence[Leg], market: Market | None = None, span=(0.0, 3.0), n=4001):
    """Best and worst expiry P&L, with None meaning "unbounded".

    Only the upside can be unbounded. Spot cannot fall below zero, so the
    bottom of the scan is a real, attainable state and whatever the P&L is
    there is a genuine worst case. The top of the scan is not: if the curve is
    still climbing or still falling at the highest spot we scan, it keeps going.
    """
    market = market or Market()
    spots = np.linspace(span[0] * market.spot, span[1] * market.spot, n)
    y = np.asarray(payoff(legs, spots, market), dtype=float)
    edge = max(3, n // 200)
    still_rising = y[-1] > y[-1 - edge] + 1e-9
    still_falling = y[-1] < y[-1 - edge] - 1e-9
    best = None if still_rising else float(y.max())
    worst = None if still_falling else float(y.min())
    return best, worst


# --- ready-made positions -------------------------------------------------
#
# Each builder takes the market and returns legs, so the presets re-strike
# themselves around whatever spot the user has dialled in.


def _round_strike(x: float) -> float:
    return float(np.round(x, 2))


def preset_legs(name: str, market: Market, expiry: float = 0.5) -> list[Leg]:
    S = market.spot
    e = expiry
    k = _round_strike
    builders = {
        "Long call": lambda: [Leg(bs.CALL, 1, k(S), e)],
        "Long put": lambda: [Leg(bs.PUT, 1, k(S), e)],
        "Covered call": lambda: [Leg(STOCK, 1), Leg(bs.CALL, -1, k(S * 1.05), e)],
        "Protective put": lambda: [Leg(STOCK, 1), Leg(bs.PUT, 1, k(S * 0.95), e)],
        "Straddle": lambda: [Leg(bs.CALL, 1, k(S), e), Leg(bs.PUT, 1, k(S), e)],
        "Strangle": lambda: [Leg(bs.CALL, 1, k(S * 1.1), e), Leg(bs.PUT, 1, k(S * 0.9), e)],
        "Bull call spread": lambda: [Leg(bs.CALL, 1, k(S), e), Leg(bs.CALL, -1, k(S * 1.1), e)],
        "Bear put spread": lambda: [Leg(bs.PUT, 1, k(S), e), Leg(bs.PUT, -1, k(S * 0.9), e)],
        "Risk reversal": lambda: [Leg(bs.CALL, 1, k(S * 1.05), e), Leg(bs.PUT, -1, k(S * 0.95), e)],
        "Butterfly": lambda: [
            Leg(bs.CALL, 1, k(S * 0.9), e),
            Leg(bs.CALL, -2, k(S), e),
            Leg(bs.CALL, 1, k(S * 1.1), e),
        ],
        "Iron condor": lambda: [
            Leg(bs.PUT, 1, k(S * 0.85), e),
            Leg(bs.PUT, -1, k(S * 0.95), e),
            Leg(bs.CALL, -1, k(S * 1.05), e),
            Leg(bs.CALL, 1, k(S * 1.15), e),
        ],
        "Calendar spread": lambda: [
            Leg(bs.CALL, -1, k(S), e / 2),
            Leg(bs.CALL, 1, k(S), e),
        ],
    }
    if name not in builders:
        raise KeyError(f"unknown preset {name!r}")
    return builders[name]()


PRESETS = [
    "Long call", "Long put", "Covered call", "Protective put",
    "Straddle", "Strangle", "Bull call spread", "Bear put spread",
    "Risk reversal", "Butterfly", "Iron condor", "Calendar spread",
]

#: One line on what each preset is FOR - shown next to the payoff chart.
PRESET_NOTES = {
    "Long call": "Pay a premium for unlimited upside. Long gamma, long vega, pays theta every day.",
    "Long put": "Insurance against a fall. Same bill: theta bleeds while you wait.",
    "Covered call": "Own the stock, sell away the upside above the strike. Collects theta, caps the gain.",
    "Protective put": "Own the stock, buy a floor under it. A stop-loss that cannot gap through you.",
    "Straddle": "Buy both at the same strike. A bet on movement in either direction, not on direction.",
    "Strangle": "Cheaper straddle with wider strikes. Needs a bigger move to pay off.",
    "Bull call spread": "Buy a call, sell a higher one. Cuts the premium and caps the payoff.",
    "Bear put spread": "The mirror image, for a fall.",
    "Risk reversal": "Sell a put to fund a call. Stock-like exposure for almost no premium, and the downside comes with it.",
    "Butterfly": "Pin the spot near the middle strike. Cheap, high payoff, narrow window.",
    "Iron condor": "Sell a range, buy wings for protection. Collects theta while the spot stays inside.",
    "Calendar spread": "Sell the near expiry, buy the far one. Long vega, and it wants the spot to sit still.",
}
