"""A volatility smile, and what it does to your hedge.

Flat vol is a fiction. Real markets charge more for downside strikes than
upside ones, and this module carries the smallest parametrisation that shows
why that matters: vol quadratic in log-moneyness.

    sigma(k) = atm + skew * k + curvature * k^2,    k = log(K / forward)

Two parameters, two distinct effects:

  * **skew** tilts the smile. Negative skew - the equity-index shape - makes
    downside puts dearer than upside calls.
  * **curvature** lifts both wings, pricing in the chance of a large move in
    either direction.

The reason this gets its own page is the second-order consequence. Once vol
depends on strike, *delta itself changes*, and by how much depends on a
modelling choice nobody can settle from the quotes alone: when the spot moves,
does each strike keep its own vol, or does the smile slide along with the spot?
Those two assumptions give different hedges for the same market, and
`smile_delta` computes both.
"""

from __future__ import annotations

import numpy as np

from . import bs

STICKY_STRIKE = "sticky strike"
STICKY_MONEYNESS = "sticky moneyness"
REGIMES = (STICKY_STRIKE, STICKY_MONEYNESS)

MIN_VOL = 0.01  # keep the parametrisation from wandering into negative vol


def forward(spot, T, r, q):
    return np.asarray(spot, float) * np.exp((np.asarray(r, float) - np.asarray(q, float)) * np.asarray(T, float))


def log_moneyness(strikes, spot, T, r, q):
    """k = log(strike / forward). Zero is at-the-money-forward."""
    return np.log(np.asarray(strikes, float) / forward(spot, T, r, q))


def smile_vol(strikes, spot, T, r, q, atm_vol=0.2, skew=0.0, curvature=0.0):
    """The vol this smile quotes for each strike."""
    k = log_moneyness(strikes, spot, T, r, q)
    return np.maximum(atm_vol + skew * k + curvature * k**2, MIN_VOL)


def smile_slope(strikes, spot, T, r, q, atm_vol=0.2, skew=0.0, curvature=0.0):
    """d(sigma)/dk, the slope of the smile in log-moneyness."""
    k = log_moneyness(strikes, spot, T, r, q)
    raw = atm_vol + skew * k + curvature * k**2
    return np.where(raw > MIN_VOL, skew + 2.0 * curvature * k, 0.0)


def smile_price(strikes, spot, T, r, q, kind=bs.CALL, **smile_kwargs):
    """Price each strike on its own point of the smile."""
    vols = smile_vol(strikes, spot, T, r, q, **smile_kwargs)
    return bs.price(spot, strikes, T, r, q, vols, kind)


def smile_delta(strikes, spot, T, r, q, kind=bs.CALL, regime=STICKY_STRIKE, **smile_kwargs):
    """Delta in the presence of a smile, under one of the two stickiness rules.

    Under **sticky strike** each strike keeps its own vol as the spot moves, so
    vol is not a function of spot and the Black-Scholes delta (read off that
    strike's vol) is the whole answer.

    Under **sticky moneyness** the smile travels with the spot, so a strike's
    vol changes whenever the spot does. That adds a second channel to delta:

        total delta = BS delta + vega * d(sigma)/d(spot)

    Follow the sign carefully, because the intuition trips people up. Moneyness
    is k = log(K / forward), so when the spot rises a *fixed* strike drops to a
    lower k. With the equity-index shape - downside strikes dearer - lower k
    means richer vol. So a rising spot makes that strike's vol go UP, vega is
    positive, and the total delta comes out ABOVE the Black-Scholes number.

    The other rule contains the effect everyone associates with equity skew.
    Under sticky strike, a rising spot leaves every strike's vol alone, which
    means the at-the-money vol falls, because at-the-money now points at a
    higher strike that was always quoted cheaper. That is the leverage effect -
    spot up, vol down - and it is sticky *strike*, not sticky moneyness, that
    reproduces it.

    Neither rule is the truth; they bracket it. The gap between the two deltas
    is the size of the modelling risk you are carrying, and it is zero only
    when the smile is flat.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    strikes = np.asarray(strikes, float)
    vols = smile_vol(strikes, spot, T, r, q, **smile_kwargs)
    bs_delta = np.asarray(bs.delta(spot, strikes, T, r, q, vols, kind), float)
    if regime == STICKY_STRIKE:
        return bs_delta

    # k = log(K) - log(S) - (r-q)T, so dk/dS = -1/S.
    slope = np.asarray(smile_slope(strikes, spot, T, r, q, **smile_kwargs), float)
    dvol_dspot = -slope / float(spot)
    vega = np.asarray(bs.vega(spot, strikes, T, r, q, vols, kind), float)
    return bs_delta + vega * dvol_dspot


#: Ready-made smiles, so the shapes have names before they have numbers.
SHAPES = {
    "Flat (textbook)": dict(skew=0.0, curvature=0.0),
    "Equity index skew": dict(skew=-0.35, curvature=0.6),
    "Gentle smile": dict(skew=0.0, curvature=1.2),
    "Upside call skew": dict(skew=0.30, curvature=0.5),
}

SHAPE_NOTES = {
    "Flat (textbook)": "One vol for every strike. The Black-Scholes world, and the one real markets never quite live in.",
    "Equity index skew": "Downside puts cost more than upside calls. Crash protection is in demand, and the market charges for it.",
    "Gentle smile": "Both wings bid, no directional tilt. The typical currency shape: a big move either way is what is feared.",
    "Upside call skew": "Upside strikes bid. Seen where the squeeze, not the crash, is the risk people hedge.",
}
