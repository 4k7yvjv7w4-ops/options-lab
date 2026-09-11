"""Page 2 - the greek as a landscape over spot and time.

A curve shows a greek at one maturity. A map shows what happens to that curve
as expiry approaches, which is where most of the surprises live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optlab import bs, ui

MAP_LESSONS = {
    "delta": "Watch the band where delta transitions from 0 to 1 narrow as expiry "
             "approaches. Far from expiry, delta is a gentle slope across a wide "
             "range of spots. Near expiry it is a cliff at the strike.",
    "gamma": "This is the map worth staring at. Gamma is modest and spread out when "
             "expiry is far away, then concentrates into a spike at the strike in "
             "the final days. A position that looked harmless a month ago becomes "
             "unhedgeable in the last week without the strike ever moving.",
    "vega": "Vega fades toward expiry everywhere. A view on volatility expressed in "
            "short-dated options barely has a position at all; buy maturity instead.",
    "theta": "Theta deepens sharply at the money as expiry approaches, mirroring the "
             "gamma spike exactly. Owning the gamma means paying that bill daily.",
    "rho": "Rho shrinks steadily toward expiry and is trivial for short maturities. "
           "It matters for long-dated options and almost nowhere else.",
    "vanna": "Vanna flips sign across the strike and intensifies near expiry, so a "
             "skew-driven delta adjustment is sharpest exactly when you can least "
             "afford to get the hedge wrong.",
    "volga": "Volga is a pair of ridges in the wings with a valley at the money. "
             "Wing options are the volatility-convexity instruments.",
    "charm": "Charm is strongest just off the strike close to expiry. It is the "
             "reason a hedge set on Friday is wrong on Monday even if nothing moved.",
    "price": "Price contours fan out from the payoff kink. The distance between the "
             "contours and the kink is time value, draining to zero at expiry.",
}


def render() -> None:
    ui.keep("sf_greek", "sf_strike", "sf_max_days", "sf_kind")
    market = ui.current_market()

    ui.page_header(
        "Greek maps",
        "The same greeks, drawn across spot and time at once. The shapes here explain "
        "why options get dangerous near expiry even when nothing about the market changed.",
    )

    c1, c2, c3 = st.columns([1.2, 1, 1], gap="medium")
    with c1:
        greek = st.selectbox("Greek", list(bs.GREEK_FUNCS), index=2, key="sf_greek")
    with c2:
        strike = st.slider("Strike", float(market.spot * 0.6), float(market.spot * 1.4),
                           float(market.spot), 1.0, key="sf_strike")
    with c3:
        max_days = st.select_slider("Longest maturity shown",
                                    [30, 60, 90, 180, 365, 730], value=180, key="sf_max_days")
    kind = st.segmented_control("Option type", [bs.CALL, bs.PUT], default=bs.CALL,
                                key="sf_kind", selection_mode="single") or bs.CALL

    spots = np.linspace(market.spot * 0.6, market.spot * 1.4, 90)
    days = np.linspace(1, max_days, 70)
    S, D = np.meshgrid(spots, days)
    z = bs.GREEK_FUNCS[greek](S, strike, D / 365.0, market.rate, market.div_yield,
                              market.vol, kind)
    z = bs.scale_for_display(greek, z)

    signed = bool(np.nanmin(z) < -1e-12 and np.nanmax(z) > 1e-12)
    frame = ui.grid_frame(spots, days, z, "spot", "days", "value")
    limit, clipped = ui.robust_limit(z)
    ui.show(
        ui.heatmap(frame, "spot", "days", "value", height=420, x_title="Spot",
                   y_title="Days to expiry", value_title=greek, diverging=signed)
    )
    ui.takeaway(MAP_LESSONS.get(greek, ""))

    scale_note = (
        " Two colours because the value changes sign, with the pale midpoint at zero."
        if signed else " One hue, light to dark, because the value keeps one sign."
    )
    st.caption(
        "Read the map bottom-up: each horizontal slice is one maturity, and moving "
        "down the chart is the option ageing toward expiry. Values are "
        f"{bs.display_units(greek)}." + scale_note
    )
    if clipped:
        peak = float(np.nanmax(np.abs(z)))
        st.caption(
            f"The colour scale stops at {limit:,.4f}, though {greek} reaches "
            f"{peak:,.4f} in the corner nearest expiry. Without that cut, the single "
            f"spike would wash the entire rest of the map into one pale shade. Hover "
            f"any cell for its true value."
        )

    st.divider()
    _slices(spots, strike, market, greek, kind, max_days)


def _slices(spots, strike, market, greek, kind, max_days) -> None:
    st.markdown("#### The same thing as slices")
    picks = sorted({1, 7, 30, min(90, max_days), max_days})
    rows = []
    for d in picks:
        vals = bs.GREEK_FUNCS[greek](spots, strike, d / 365.0, market.rate,
                                     market.div_yield, market.vol, kind)
        rows.append(pd.DataFrame({"spot": spots,
                                  "value": bs.scale_for_display(greek, vals),
                                  "maturity": f"{d}d"}))
    df = pd.concat(rows, ignore_index=True)
    ui.show(
        ui.line_chart(df, "spot", "value", "maturity", height=320, x_title="Spot",
                      y_title=f"{greek} ({bs.display_units(greek)})",
                      color_title="Days to expiry", zero_line=True,
                      order=[f"{d}d" for d in picks])
        + ui.vertical_rule(strike, "strike")
    )
    st.caption(
        "A map is good for seeing the whole shape; slices are better for reading off "
        "actual numbers. Use both: find the region on the map, then read it here."
    )
