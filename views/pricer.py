"""Page 1 - one option, taken apart.

The aim is that after ten minutes here you can predict the sign and rough size
of every greek before you move the slider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optlab import bs, ui

GREEK_BLURB = {
    "price": "What the option is worth today.",
    "delta": "How much the option gains when the spot rises by 1. Also, roughly, the chance it finishes in the money.",
    "gamma": "How much delta itself moves. High gamma means your hedge goes stale fast.",
    "vega": "What a 1-point change in implied vol is worth. Always positive for a long option.",
    "theta": "What one day of waiting costs. The rent you pay for owning optionality.",
    "rho": "Sensitivity to interest rates. Usually the one you can ignore, until maturities get long.",
    "vanna": "How delta responds to a change in vol. The reason a smile changes your hedge.",
    "volga": "How vega responds to vol. Why wing options are convex in volatility.",
    "charm": "How delta drifts purely because a day passed, with the spot unchanged.",
}


def render() -> None:
    ui.keep("pr_strike", "pr_days", "pr_kind", "pr_greek", "pr_compare")
    market = ui.current_market()

    ui.page_header(
        "Price and greeks",
        "One option, every sensitivity. Move a slider and watch which curve reacts: "
        "that reaction, not the formula, is what the greeks are.",
    )

    left, right = st.columns([1, 1], gap="medium")
    with left:
        strike = st.slider("Strike", float(market.spot * 0.5), float(market.spot * 1.5),
                           float(market.spot), 1.0, key="pr_strike")
    with right:
        days = st.slider("Days to expiry", 1, 730, 90, key="pr_days",
                         help="Time value is roughly proportional to the square root of "
                              "this, so halving it does not halve the premium.")
    kind = st.segmented_control("Option type", [bs.CALL, bs.PUT], default=bs.CALL,
                                key="pr_kind", selection_mode="single") or bs.CALL

    T = days / 365.0
    args = (market.spot, strike, T, market.rate, market.div_yield, market.vol)
    raw = bs.greeks(*args, kind)

    # --- the headline numbers, in the units a desk actually quotes ---------
    st.markdown("#### Where this option stands")
    quoted = {k: float(bs.scale_for_display(k, v)) for k, v in raw.items()}
    intr = float(bs.intrinsic(market.spot, strike, kind))
    time_value = quoted["price"] - intr

    ui.metric_row(
        [
            ("Price", ui.money(quoted["price"]), None),
            ("Delta", f"{quoted['delta']:+.3f}", None),
            ("Gamma", f"{quoted['gamma']:+.4f}", None),
            ("Vega", f"{quoted['vega']:+.3f}", None),
            ("Theta", f"{quoted['theta']:+.3f}", None),
        ],
        help_texts=[
            "Intrinsic value plus time value.",
            bs.display_units("delta") + ". " + GREEK_BLURB["delta"],
            bs.display_units("gamma") + ". " + GREEK_BLURB["gamma"],
            bs.display_units("vega") + ". " + GREEK_BLURB["vega"],
            bs.display_units("theta") + ". " + GREEK_BLURB["theta"],
        ],
    )

    moneyness = market.spot / strike
    where = ("at the money" if abs(moneyness - 1) < 0.02
             else ("in the money" if (moneyness > 1) == (kind == bs.CALL) else "out of the money"))
    st.caption(
        f"This {kind} is **{where}**. Of its {ui.money(quoted['price'])} price, "
        f"{ui.money(intr)} is intrinsic (what you would get exercising now) and "
        f"{ui.money(time_value)} is time value (what you are paying for what might still happen). "
        f"Theta is quoted {bs.display_units('theta')}, vega {bs.display_units('vega')}."
    )

    st.divider()

    # --- the curve --------------------------------------------------------
    st.markdown("#### How it changes as the spot moves")
    pick_col, cmp_col = st.columns([1, 2], gap="medium", vertical_alignment="bottom")
    with pick_col:
        greek = st.selectbox("Show", list(bs.GREEK_FUNCS), index=1, key="pr_greek")
    with cmp_col:
        compare = st.multiselect(
            "Compare maturities (days)", [7, 30, 90, 180, 365, 730],
            default=[7, 90, 365], key="pr_compare",
            help="The same greek at several maturities. Where the curves separate "
                 "is where time matters most.",
        )

    spots = np.linspace(market.spot * 0.4, market.spot * 1.6, 320)
    maturities = sorted(set(compare) | {days})
    rows = []
    for d in maturities:
        vals = bs.GREEK_FUNCS[greek](spots, strike, d / 365.0, market.rate,
                                     market.div_yield, market.vol, kind)
        rows.append(pd.DataFrame({
            "spot": spots,
            "value": bs.scale_for_display(greek, vals),
            "maturity": f"{d}d" + (" (current)" if d == days else ""),
        }))
    curve = pd.concat(rows, ignore_index=True)
    order = [f"{d}d" + (" (current)" if d == days else "") for d in maturities]

    chart = ui.line_chart(
        curve, "spot", "value", "maturity", height=360,
        x_title="Spot", y_title=f"{greek} ({bs.display_units(greek)})",
        color_title="Time to expiry", zero_line=True, order=order,
    ) + ui.vertical_rule(strike, "strike") + ui.vertical_rule(market.spot, "spot")
    ui.show(chart)

    st.caption(f"**{greek.title()}** — {GREEK_BLURB[greek]}")
    _greek_lesson(greek, kind)

    st.divider()
    _value_decomposition(market, strike, T, kind, spots)


def _greek_lesson(greek: str, kind: str) -> None:
    lessons = {
        "price": "Notice the curve never touches the straight payoff line until expiry. "
                 "That gap is time value, and it is widest at the money.",
        "delta": "Delta runs from 0 to 1 for a call and 0 to -1 for a put, and the "
                 "switch happens around the strike. The shorter the maturity, the "
                 "more abruptly it flips: a one-week option is nearly a binary bet on "
                 "which side of the strike you land.",
        "gamma": "Gamma is a hill centred on the strike, and shortening maturity makes "
                 "it taller and narrower, not flatter. That is the single most "
                 "important shape in this app: near expiry, at the money, your hedge "
                 "goes stale almost instantly.",
        "vega": "Vega peaks near the strike and grows with maturity, because a long-dated "
                "option has more time for a change in vol to matter. If you want to "
                "express a view on volatility, buy time.",
        "theta": "Theta is most negative exactly where gamma is highest. That is not a "
                 "coincidence: it is the price of gamma. You cannot buy one without "
                 "paying the other.",
        "rho": "Rho grows with maturity and is nearly invisible on short-dated options. "
               "This is the greek you are allowed to forget first.",
        "vanna": "Vanna changes sign across the strike. It is why a skewed volatility "
                 "surface tilts your delta in opposite directions on either side.",
        "volga": "Volga is near zero at the money and rises in both wings, which is why "
                 "far-out-of-the-money options behave like bets on volatility itself.",
        "charm": "Charm is delta decay. Even with the spot pinned, an option's delta "
                 "drifts toward 0 or 1 as expiry nears, so a hedge left alone over a "
                 "quiet weekend is still wrong on Monday.",
    }
    ui.takeaway(lessons[greek])


def _value_decomposition(market, strike, T, kind, spots) -> None:
    st.markdown("#### Value now against payoff at expiry")
    now = bs.price(spots, strike, T, market.rate, market.div_yield, market.vol, kind)
    at_expiry = bs.intrinsic(spots, strike, kind)
    df = pd.concat([
        pd.DataFrame({"spot": spots, "value": now, "line": "Value today"}),
        pd.DataFrame({"spot": spots, "value": at_expiry, "line": "Payoff at expiry"}),
    ], ignore_index=True)
    ui.show(
        ui.line_chart(df, "spot", "value", "line", height=300, x_title="Spot",
                      y_title="Option value", color_title="",
                      order=["Value today", "Payoff at expiry"],
                      dashed=["Payoff at expiry"])
        + ui.vertical_rule(strike, "strike")
    )
    st.caption(
        "The vertical distance between the two lines is time value, and it is what "
        "decays to nothing by expiry. Time value is largest at the strike, which is "
        "also where you have the most to lose by holding to the end."
    )
