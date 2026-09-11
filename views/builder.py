"""Page 3 - build a position and see its shape.

Every structure people trade is a handful of legs added together. Once you can
see the payoff and the net greeks move as you add a leg, the names stop being
vocabulary and start being consequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optlab import bs, ui
from optlab.strategies import (
    PRESET_NOTES, PRESETS, STOCK, Leg, breakevens, cost, first_expiry,
    max_expiry, net_greek_curve, net_greeks, payoff_extremes, preset_legs, value_at,
)

COLUMNS = ["kind", "qty", "strike", "expiry_days"]


def render() -> None:
    ui.keep("sb_preset", "sb_horizon", "sb_greek")
    market = ui.current_market()

    ui.page_header(
        "Strategy builder",
        "Add legs and watch the payoff bend. The solid line is what the position pays "
        "at expiry, the dashed one what it is worth today. The gap between them is time "
        "value, and closing that gap is what waiting does to you.",
    )

    preset = st.selectbox("Start from", PRESETS, key="sb_preset", on_change=_reset_legs)
    if "sb_legs" not in st.session_state:
        _reset_legs()

    st.caption(PRESET_NOTES[preset])

    edited = st.data_editor(
        st.session_state.sb_legs,
        key="sb_editor",
        width="stretch",
        num_rows="dynamic",
        column_config={
            "kind": st.column_config.SelectboxColumn(
                "Instrument", options=[bs.CALL, bs.PUT, STOCK], required=True,
                help="Stock is the hedging instrument: delta 1, no gamma, no vega."),
            "qty": st.column_config.NumberColumn(
                "Quantity", step=1.0, format="%.2f", required=True,
                help="Positive is long, negative is short."),
            "strike": st.column_config.NumberColumn("Strike", min_value=0.01, step=1.0, format="%.2f"),
            "expiry_days": st.column_config.NumberColumn(
                "Days to expiry", min_value=1, step=1, format="%d"),
        },
    )

    legs = _to_legs(edited)
    if not legs:
        st.info("Add at least one leg to see a payoff.", icon="👆")
        return

    _summary(legs, market)
    st.divider()
    _payoff_chart(legs, market)
    st.divider()
    _greek_curves(legs, market)


def _reset_legs() -> None:
    """Rebuild the leg table from the chosen preset, re-struck around today's spot."""
    market = ui.current_market()
    preset = st.session_state.get("sb_preset", PRESETS[0])
    st.session_state.sb_legs = _to_frame(preset_legs(preset, market))


def _to_frame(legs) -> pd.DataFrame:
    return pd.DataFrame(
        [{"kind": l.kind, "qty": float(l.qty), "strike": float(l.strike),
          "expiry_days": int(round(l.expiry * 365))} for l in legs],
        columns=COLUMNS,
    )


def _to_legs(frame: pd.DataFrame) -> list[Leg]:
    legs: list[Leg] = []
    for _, row in frame.iterrows():
        kind = str(row.get("kind") or "").strip()
        if kind not in (bs.CALL, bs.PUT, STOCK):
            continue
        qty = float(row.get("qty") or 0.0)
        if qty == 0.0:
            continue
        days = row.get("expiry_days")
        days = 30 if pd.isna(days) else max(int(days), 1)
        strike = row.get("strike")
        strike = 100.0 if pd.isna(strike) else max(float(strike), 0.01)
        legs.append(Leg(kind=kind, qty=qty, strike=strike, expiry=days / 365.0))
    return legs


def _summary(legs, market) -> None:
    premium = cost(legs, market)
    best, worst = payoff_extremes(legs, market)
    be = breakevens(legs, market)
    g = net_greeks(legs, market)

    ui.metric_row(
        [
            ("Cost to open" if premium >= 0 else "Credit received",
             ui.money(abs(premium)), None),
            ("Max profit", ui.unbounded(best), None),
            ("Max loss", ui.unbounded(worst), None),
            ("Breakeven" + ("s" if len(be) != 1 else ""),
             " / ".join(f"{b:,.2f}" for b in be) if be else "none in range", None),
        ],
        help_texts=[
            "Negative premium means the position pays you up front, and your risk is what happens later.",
            "Best case at the first expiry. 'Unlimited' means the payoff keeps rising with the spot.",
            "Worst case at the first expiry. The spot cannot go below zero, so this side is always a real number.",
            "Spot levels where the position breaks even at the first expiry.",
        ],
    )

    st.markdown("###### Net greeks")
    ui.metric_row(
        [
            ("Delta", f"{g['delta']:+.3f}", None),
            ("Gamma", f"{g['gamma']:+.4f}", None),
            ("Vega", f"{bs.scale_for_display('vega', g['vega']):+.3f}", None),
            ("Theta", f"{bs.scale_for_display('theta', g['theta']):+.3f}", None),
            ("Rho", f"{bs.scale_for_display('rho', g['rho']):+.3f}", None),
        ],
        help_texts=[
            "Net directional exposure, in shares of the underlying.",
            "How fast that directional exposure changes.",
            "Per 1 vol point.",
            "Per calendar day. Positive means time is on your side.",
            "Per 1% change in rates.",
        ],
    )
    _position_verdict(g)


def _position_verdict(g: dict) -> None:
    gamma_side = "long gamma" if g["gamma"] > 1e-9 else ("short gamma" if g["gamma"] < -1e-9 else "gamma neutral")
    vega_side = "long vega" if g["vega"] > 1e-6 else ("short vega" if g["vega"] < -1e-6 else "vega neutral")
    theta_side = "collects theta" if g["theta"] > 0 else "pays theta"
    daily = bs.scale_for_display("theta", g["theta"])
    ui.takeaway(
        f"This position is **{gamma_side}**, **{vega_side}**, and **{theta_side}** "
        f"at about {daily:+.3f} a day. Those three always travel together: being long "
        f"gamma means paying theta, and being short gamma means collecting it. There is "
        f"no structure anywhere that gives you both."
    )


def _payoff_chart(legs, market) -> None:
    st.markdown("#### Payoff")
    horizon_max = int(round(first_expiry(legs) * 365))
    held = st.slider(
        "Days from now", 0, max(horizon_max, 1), 0, key="sb_horizon",
        help="Slide toward expiry and watch the smooth curve collapse onto the kinked payoff.",
    )

    spots = np.linspace(market.spot * 0.55, market.spot * 1.45, 400)
    premium = cost(legs, market)
    expiry_pnl = np.asarray(value_at(legs, spots, first_expiry(legs), market)) - premium
    now_pnl = np.asarray(value_at(legs, spots, held / 365.0, market)) - premium

    label_now = "Value today" if held == 0 else f"Value in {held}d"
    df = pd.concat([
        pd.DataFrame({"spot": spots, "pnl": expiry_pnl, "line": "At expiry"}),
        pd.DataFrame({"spot": spots, "pnl": now_pnl, "line": label_now}),
    ], ignore_index=True)

    chart = ui.line_chart(
        df, "spot", "pnl", "line", height=380, x_title="Spot at expiry",
        y_title="Profit and loss", color_title="", zero_line=True,
        order=["At expiry", label_now], dashed=[label_now],
    ) + ui.vertical_rule(market.spot, "spot now")
    be = breakevens(legs, market)
    if be:
        chart = chart + ui.vertical_rule(be, "breakeven")
    ui.show(chart)

    if max_expiry(legs) > first_expiry(legs) + 1e-9:
        ui.caveat(
            "This position has legs expiring on different dates, so the payoff is drawn "
            "at the FIRST expiry, while the back leg still holds time value. At the back "
            "leg's expiry a same-strike calendar cancels itself out entirely, which is a "
            "flat line and tells you nothing.",
            icon="📅",
        )


def _greek_curves(legs, market) -> None:
    st.markdown("#### Net greeks across the spot")
    greek = st.selectbox("Greek", ["delta", "gamma", "vega", "theta"], key="sb_greek")
    spots = np.linspace(market.spot * 0.55, market.spot * 1.45, 400)
    horizons = sorted({0.0, first_expiry(legs) * 0.5, first_expiry(legs) * 0.9})
    rows = []
    for t in horizons:
        vals = net_greek_curve(legs, spots, greek, t, market)
        label = "today" if t == 0 else f"{int(round(t * 365))}d from now"
        rows.append(pd.DataFrame({"spot": spots,
                                  "value": bs.scale_for_display(greek, vals),
                                  "when": label}))
    df = pd.concat(rows, ignore_index=True)
    order = list(dict.fromkeys(df["when"]))
    ui.show(
        ui.line_chart(df, "spot", "value", "when", height=330, x_title="Spot",
                      y_title=f"net {greek} ({bs.display_units(greek)})",
                      color_title="", zero_line=True, order=order,
                      dashed=[o for o in order if o != "today"])
        + ui.vertical_rule(market.spot, "spot now")
    )
    st.caption(
        "Net greeks are just the legs added up, but the sum often behaves nothing like "
        "its parts. An iron condor is short gamma between its short strikes and long "
        "gamma outside them, so the same position changes character as the spot travels."
    )
