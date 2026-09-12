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
    c4, c5 = st.columns([1, 1], gap="medium")
    with c4:
        kind = st.segmented_control("Option type", [bs.CALL, bs.PUT], default=bs.CALL,
                                    key="sf_kind", selection_mode="single") or bs.CALL
    with c5:
        view = st.segmented_control(
            "View", ["Flat map", "3-D surface"], default="Flat map", key="sf_view",
            selection_mode="single",
            help="The same numbers either way. The flat map is easier to read a value "
                 "off; the surface is better for seeing the shape.",
        ) or "Flat map"

    spots = np.linspace(market.spot * 0.6, market.spot * 1.4, 90)
    days = np.linspace(1, max_days, 70)
    S, D = np.meshgrid(spots, days)
    z = bs.GREEK_FUNCS[greek](S, strike, D / 365.0, market.rate, market.div_yield,
                              market.vol, kind)
    z = bs.scale_for_display(greek, z)
    signed = bool(np.nanmin(z) < -1e-12 and np.nanmax(z) > 1e-12)

    if view == "3-D surface":
        _surface_view(spots, days, z, greek, signed)
    else:
        _map_view(spots, days, z, greek, signed)

    st.divider()
    _slices(spots, strike, market, greek, kind, max_days)


def _map_view(spots, days, z, greek: str, signed: bool) -> None:
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
        peak = ui.signed_extreme(z)
        st.caption(
            f"The colour scale stops at {abs(limit):,.4f} in magnitude, though {greek} "
            f"reaches {peak:,.4f} in the corner nearest expiry. Without that cut, the "
            f"single spike would wash the entire rest of the map into one pale shade. "
            f"Hover any cell for its true value."
        )


def _surface_view(spots, days, z, greek: str, signed: bool) -> None:
    """The same grid as a landscape you can turn around."""
    limit, spiky = ui.robust_limit(z)
    peak = ui.signed_extreme(z)

    cap = None
    if spiky:
        if st.toggle(
            "Cap the peak", value=False, key="sf_cap",
            help="The spike is so much taller than everything else that it flattens "
                 "the rest of the surface into the floor. Capping it lets you see "
                 "the shape underneath.",
        ):
            cap = limit

    ui.show_surface(
        ui.surface(spots, days, z, x_title="Spot", y_title="Days to expiry",
                   z_title=greek, diverging=signed, cap=cap, height=560)
    )
    ui.takeaway(MAP_LESSONS.get(greek, ""))
    st.caption(
        "Drag to rotate, scroll to zoom, double-click to reset the view. Height and "
        f"colour both carry {greek}, in {bs.display_units(greek)}, so there is no "
        "second scale to reconcile. Turning the surface until you are looking straight "
        "down gives you the flat map back."
    )
    if spiky and cap is None:
        st.caption(
            f"The extreme reaches {peak:,.4f} against roughly {abs(limit):,.4f} in "
            f"magnitude across the rest of the surface, which is why everything else "
            f"looks flat. That contrast is the point, but switch on **cap the peak** to "
            f"see the structure underneath it."
        )
    elif cap is not None:
        st.caption(
            f"Height is capped at {abs(cap):,.4f} in magnitude; the true extreme is "
            f"{peak:,.4f}. Hovering any point still reports its real value, so nothing "
            f"is hidden, only flattened."
        )


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
