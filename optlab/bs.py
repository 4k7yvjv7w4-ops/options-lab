"""Black-Scholes-Merton pricing and greeks.

Every function is vectorised: pass scalars or numpy arrays for any argument and
they broadcast together. That is what lets the app draw a greek across 400 spot
levels and 5 maturities without a Python loop.

Conventions in this module are the *mathematical* ones, not the trading-desk
display ones:

    vega   = dV/dsigma     per 1.00 of vol   (i.e. per 100 vol points)
    theta  = dV/dt         per 1 year of calendar time (negative for long options)
    rho    = dV/dr         per 1.00 of rate  (i.e. per 100 bp)
    charm  = d(delta)/dt   per 1 year
    vanna  = d(delta)/dsigma
    volga  = d(vega)/dsigma

`scale_for_display` converts them to the units traders actually quote. Keeping
the two apart is deliberate: the unit confusion ("is vega 0.12 or 12?") is one
of the things this app exists to make obvious.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

CALL = "call"
PUT = "put"
KINDS = (CALL, PUT)

# Below these, the closed form divides by ~0. We fall back to the limiting
# (zero-vol / expired) payoff instead, which is the correct limit anyway.
_TINY_T = 1e-12
_TINY_VOL = 1e-12


def _check_kind(kind: str) -> str:
    k = str(kind).lower()
    if k not in KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return k


def d1_d2(S, K, T, r, q, sigma):
    """The two Black-Scholes moneyness terms, broadcast over any array inputs.

    Where T or sigma is ~0 the terms are undefined; we return +/-inf with the
    sign of log-moneyness so that N(d1), N(d2) collapse to the right 0/1 limits
    and the intrinsic-value payoff falls out of the same formula.
    """
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    vol_t = sigma * np.sqrt(T)
    degenerate = (T <= _TINY_T) | (sigma <= _TINY_VOL) | (S <= 0) | (K <= 0)

    safe_S = np.where(S > 0, S, 1.0)
    safe_K = np.where(K > 0, K, 1.0)
    safe_vol_t = np.where(degenerate, 1.0, vol_t)

    d1 = (np.log(safe_S / safe_K) + (r - q + 0.5 * sigma**2) * T) / safe_vol_t
    d2 = d1 - safe_vol_t

    # Limit: the option is in or out of the money with certainty.
    fwd_log = np.log(np.where(S > 0, S, 0.0) / safe_K, where=(S > 0), out=np.full_like(safe_S, -np.inf))
    limit = np.where(fwd_log > 0, np.inf, np.where(fwd_log < 0, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, limit, d2)
    return d1, d2


def price(S, K, T, r, q, sigma, kind=CALL):
    """Black-Scholes-Merton value of a European option on a dividend-paying asset."""
    kind = _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    if kind == CALL:
        out = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        out = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return _as_scalar(np.maximum(out, 0.0))


def intrinsic(S, K, kind=CALL):
    """Payoff if the option expired right now."""
    kind = _check_kind(kind)
    S, K = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float))
    out = np.maximum(S - K, 0.0) if kind == CALL else np.maximum(K - S, 0.0)
    return _as_scalar(out)


def delta(S, K, T, r, q, sigma, kind=CALL):
    kind = _check_kind(kind)
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    T = np.asarray(T, float)
    disc_q = np.exp(-q * T)
    out = disc_q * norm.cdf(d1) if kind == CALL else disc_q * (norm.cdf(d1) - 1.0)
    return _as_scalar(out)


def gamma(S, K, T, r, q, sigma, kind=CALL):
    """Identical for calls and puts; `kind` is accepted so callers can stay uniform."""
    _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    denom = S * sigma * np.sqrt(T)
    out = np.where(denom > 0, np.exp(-q * T) * norm.pdf(d1) / np.where(denom > 0, denom, 1.0), 0.0)
    return _as_scalar(out)


def vega(S, K, T, r, q, sigma, kind=CALL):
    """dV/dsigma per 1.00 of vol. Same for calls and puts."""
    _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    out = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return _as_scalar(out)


def theta(S, K, T, r, q, sigma, kind=CALL):
    """dV/dt per YEAR. Negative for most long options: value bleeds as t advances."""
    kind = _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    sqrt_T = np.sqrt(T)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    bleed = np.where(
        sqrt_T > 0,
        -S * disc_q * norm.pdf(d1) * sigma / (2.0 * np.where(sqrt_T > 0, sqrt_T, 1.0)),
        0.0,
    )
    if kind == CALL:
        out = bleed + q * S * disc_q * norm.cdf(d1) - r * K * disc_r * norm.cdf(d2)
    else:
        out = bleed - q * S * disc_q * norm.cdf(-d1) + r * K * disc_r * norm.cdf(-d2)
    return _as_scalar(out)


def rho(S, K, T, r, q, sigma, kind=CALL):
    """dV/dr per 1.00 of rate."""
    kind = _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    out = K * T * disc_r * norm.cdf(d2) if kind == CALL else -K * T * disc_r * norm.cdf(-d2)
    return _as_scalar(out)


def vanna(S, K, T, r, q, sigma, kind=CALL):
    """d(delta)/dsigma. Same for calls and puts. The smile's fingerprint on delta."""
    _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    ok = sigma > _TINY_VOL
    out = np.where(ok, -np.exp(-q * T) * norm.pdf(d1) * d2 / np.where(ok, sigma, 1.0), 0.0)
    return _as_scalar(out)


def volga(S, K, T, r, q, sigma, kind=CALL):
    """d(vega)/dsigma, a.k.a. vomma. Why far-from-the-money options are vol-convex."""
    _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    ok = sigma > _TINY_VOL
    v = vega(S, K, T, r, q, sigma)
    out = np.where(ok, v * d1 * d2 / np.where(ok, sigma, 1.0), 0.0)
    return _as_scalar(out)


def charm(S, K, T, r, q, sigma, kind=CALL):
    """d(delta)/dt per YEAR: how fast a hedge goes stale just from time passing."""
    kind = _check_kind(kind)
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    )
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    sqrt_T = np.sqrt(T)
    disc_q = np.exp(-q * T)
    ok = (T > _TINY_T) & (sigma > _TINY_VOL)
    denom = np.where(ok, 2.0 * T * sigma * sqrt_T, 1.0)
    common = np.where(ok, disc_q * norm.pdf(d1) * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / denom, 0.0)
    if kind == CALL:
        out = q * disc_q * norm.cdf(d1) - common
    else:
        out = -q * disc_q * norm.cdf(-d1) - common
    return _as_scalar(out)


#: Everything the UI can plot, in the order it should be shown.
GREEK_FUNCS = {
    "price": price,
    "delta": delta,
    "gamma": gamma,
    "vega": vega,
    "theta": theta,
    "rho": rho,
    "vanna": vanna,
    "volga": volga,
    "charm": charm,
}


def greeks(S, K, T, r, q, sigma, kind=CALL, which=None):
    """All greeks at once, as a plain dict. `which` restricts the set."""
    names = which or list(GREEK_FUNCS)
    return {n: GREEK_FUNCS[n](S, K, T, r, q, sigma, kind) for n in names}


#: Multiply a raw greek by this to get the desk-quoted number, plus the label
#: that explains what the quoted number means.
DISPLAY_SCALE = {
    "price": (1.0, "currency"),
    "delta": (1.0, "per $1 move in spot"),
    "gamma": (1.0, "delta gained per $1 move in spot"),
    "vega": (0.01, "per 1 vol point (1%)"),
    "theta": (1.0 / 365.0, "per calendar day"),
    "rho": (0.01, "per 1% (100bp) change in rate"),
    "vanna": (0.01, "delta gained per 1 vol point"),
    "volga": (0.0001, "vega gained per 1 vol point"),
    "charm": (1.0 / 365.0, "delta lost per calendar day"),
}


def scale_for_display(name: str, value):
    """Convert a raw greek into the units a trading desk quotes."""
    factor, _ = DISPLAY_SCALE.get(name, (1.0, ""))
    return np.asarray(value, float) * factor


def display_units(name: str) -> str:
    return DISPLAY_SCALE.get(name, (1.0, ""))[1]


def implied_vol(target_price, S, K, T, r, q, kind=CALL, lo=1e-6, hi=5.0):
    """Recover the vol that reproduces an observed price, or NaN if none exists.

    Returns NaN rather than raising when the price is outside the no-arbitrage
    band, because the app feeds user-typed prices straight into this.
    """
    kind = _check_kind(kind)
    target_price, S, K, T, r, q = (float(x) for x in (target_price, S, K, T, r, q))

    if T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    lower = float(price(S, K, T, r, q, lo, kind))
    upper = float(price(S, K, T, r, q, hi, kind))
    if not (lower - 1e-12 <= target_price <= upper + 1e-12):
        return float("nan")

    def objective(vol):
        return float(price(S, K, T, r, q, vol, kind)) - target_price

    try:
        return float(brentq(objective, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def _as_scalar(arr):
    """Give back a float when every input was scalar, an array otherwise."""
    arr = np.asarray(arr)
    return float(arr) if arr.ndim == 0 else arr
