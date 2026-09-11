"""Page 5 - implied volatility, and the smile that flat Black-Scholes denies.

Two ideas, in order. First: implied vol is not an input, it is the answer to
"what vol makes this price?", and sometimes that question has no useful answer.
Second: once vol varies by strike, delta stops being a single number, because
it now depends on what you assume the smile does when the spot moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optlab import bs, smile, ui


def render() -> None:
    ui.keep("vs_shape", "vs_atm", "vs_skew", "vs_curv", "vs_days",
            "vs_iv_price", "vs_iv_strike", "vs_iv_days", "vs_iv_kind")
    market = ui.current_market()

    ui.page_header(
        "Implied volatility and the smile",
        "Implied vol is the price, restated. Turn one into the other, then watch what "
        "happens to your hedge once different strikes trade at different vols.",
    )

    tabs = st.tabs(["The smile", "Price to implied vol"])
    with tabs[0]:
        _smile_section(market)
    with tabs[1]:
        _inversion_section(market)


def _smile_section(market) -> None:
    # Opens on a skewed shape, not the flat one: the flat case is the special
    # case where nothing on this page has anything to show.
    shapes = list(smile.SHAPES)
    shape = st.selectbox("Shape", shapes, index=shapes.index("Equity index skew"),
                         key="vs_shape", on_change=_apply_shape)
    st.caption(smile.SHAPE_NOTES[shape])

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        atm = st.slider("At-the-money vol", 0.05, 0.80, 0.20, 0.01, key="vs_atm", format="%.2f")
    with c2:
        skew = st.slider("Skew", -1.0, 1.0, smile.SHAPES[shape]["skew"], 0.05, key="vs_skew",
                         help="Tilt. Negative makes downside strikes dearer.")
    with c3:
        curv = st.slider("Curvature", 0.0, 3.0, smile.SHAPES[shape]["curvature"], 0.1,
                         key="vs_curv", help="Lifts both wings equally.")
    with c4:
        days = st.slider("Days to expiry", 7, 730, 90, key="vs_days")

    T = days / 365.0
    kw = dict(atm_vol=atm, skew=skew, curvature=curv)
    strikes = np.linspace(market.spot * 0.6, market.spot * 1.4, 200)
    vols = smile.smile_vol(strikes, market.spot, T, market.rate, market.div_yield, **kw)
    fwd = float(smile.forward(market.spot, T, market.rate, market.div_yield))

    st.markdown("#### The vol each strike trades at")
    ui.show(
        ui.line_chart(pd.DataFrame({"strike": strikes, "vol": vols}), "strike", "vol",
                      height=300, x_title="Strike", y_title="Implied volatility",
                      zero_y=False)
        + ui.vertical_rule(fwd, "forward")
    )
    st.caption(
        f"The forward sits at {fwd:,.2f}, and the at-the-money-forward vol is the "
        f"{atm:.0%} you dialled in. Everything else is the skew and curvature working "
        f"on log-moneyness."
    )

    st.divider()
    _price_comparison(market, strikes, T, kw, atm)
    st.divider()
    _delta_regimes(market, strikes, T, kw, skew, curv)


def _apply_shape() -> None:
    """Load a named shape into the sliders. Written before the widgets run."""
    shape = smile.SHAPES[st.session_state.vs_shape]
    st.session_state.vs_skew = shape["skew"]
    st.session_state.vs_curv = shape["curvature"]


def _price_comparison(market, strikes, T, kw, atm) -> None:
    st.markdown("#### What the smile does to prices")
    smiled = smile.smile_price(strikes, market.spot, T, market.rate, market.div_yield,
                               bs.PUT, **kw)
    flat = bs.price(market.spot, strikes, T, market.rate, market.div_yield, atm, bs.PUT)
    df = pd.concat([
        pd.DataFrame({"strike": strikes, "price": smiled, "line": "With the smile"}),
        pd.DataFrame({"strike": strikes, "price": np.asarray(flat), "line": "Flat vol"}),
    ], ignore_index=True)
    ui.show(ui.line_chart(df, "strike", "price", "line", height=300, x_title="Strike",
                          y_title="Put price", color_title="",
                          order=["With the smile", "Flat vol"], dashed=["Flat vol"]))
    diff = np.asarray(smiled) - np.asarray(flat)
    worst = int(np.argmax(np.abs(diff)))
    if abs(diff[worst]) < 1e-4:
        st.caption(
            "With a flat smile the two lines are the same line, by construction. "
            "Add skew or curvature above and they separate."
        )
    else:
        st.caption(
            f"Put prices, with and without the smile. The biggest gap is at strike "
            f"{strikes[worst]:,.1f}, where the smile adds {diff[worst]:+,.3f} to the "
            f"flat-vol price. On an equity index shape that gap is what downside "
            f"protection costs over what Black-Scholes would charge for it."
        )


def _delta_regimes(market, strikes, T, kw, skew, curv) -> None:
    st.markdown("#### The same market, two defensible deltas")
    sticky_k = smile.smile_delta(strikes, market.spot, T, market.rate, market.div_yield,
                                 bs.CALL, regime=smile.STICKY_STRIKE, **kw)
    sticky_m = smile.smile_delta(strikes, market.spot, T, market.rate, market.div_yield,
                                 bs.CALL, regime=smile.STICKY_MONEYNESS, **kw)
    df = pd.concat([
        pd.DataFrame({"strike": strikes, "delta": sticky_k, "rule": "Sticky strike"}),
        pd.DataFrame({"strike": strikes, "delta": sticky_m, "rule": "Sticky moneyness"}),
    ], ignore_index=True)
    ui.show(ui.line_chart(df, "strike", "delta", "rule", height=310, x_title="Strike",
                          y_title="Call delta", color_title="",
                          order=["Sticky strike", "Sticky moneyness"],
                          dashed=["Sticky moneyness"])
            + ui.vertical_rule(market.spot, "spot"))

    spread = float(np.max(np.abs(np.asarray(sticky_m) - np.asarray(sticky_k))))
    if abs(skew) < 1e-9 and curv < 1e-9:
        ui.takeaway(
            "With a flat smile the two rules agree exactly, which is the only case where "
            "delta is unambiguous. Add some skew and watch them separate."
        )
    else:
        ui.takeaway(
            f"The two rules disagree by up to {spread:.3f} of delta on the same option in "
            f"the same market. Neither is wrong. **Sticky strike** says each strike keeps "
            f"its own vol when the spot moves, which makes at-the-money vol fall as the "
            f"spot rises, reproducing the leverage effect real equity markets show. "
            f"**Sticky moneyness** says the smile slides along with the spot, so a fixed "
            f"strike drops to lower moneyness and, under negative skew, richer vol. "
            f"That gap is model risk you carry whether or not you think about it."
        )


def _inversion_section(market) -> None:
    st.markdown("#### Turn a price back into a volatility")
    st.caption(
        "Quotes arrive as prices, but nobody compares prices across strikes: a 90 put "
        "and a 110 call are not comparable in dollars. Implied vol is the common "
        "language, obtained by asking which vol reproduces the observed price."
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        strike = st.number_input("Strike", 1.0, 10_000.0, float(market.spot), 1.0,
                                 key="vs_iv_strike")
    with c2:
        days = st.number_input("Days to expiry", 1, 3650, 90, key="vs_iv_days")
    with c3:
        kind = st.selectbox("Type", [bs.CALL, bs.PUT], key="vs_iv_kind")

    T = float(days) / 365.0
    fair = float(bs.price(market.spot, strike, T, market.rate, market.div_yield,
                          market.vol, kind))
    observed = st.number_input(
        "Observed market price", 0.0, 10_000.0, round(fair, 4), 0.01, key="vs_iv_price",
        help=f"Pre-filled with the price at the sidebar vol of {market.vol:.0%}.",
    )

    iv = bs.implied_vol(observed, market.spot, strike, T, market.rate,
                        market.div_yield, kind)
    intrinsic_now = float(bs.intrinsic(market.spot, strike, kind))

    if np.isnan(iv):
        ui.caveat(
            f"No volatility reproduces that price. The option is worth at least its "
            f"discounted intrinsic value (around {intrinsic_now:,.2f} undiscounted) and "
            f"no more than the underlying itself. A quote outside those bounds is an "
            f"arbitrage, not a volatility."
        )
        return

    vega_at_iv = float(bs.scale_for_display("vega", bs.vega(
        market.spot, strike, T, market.rate, market.div_yield, iv, kind)))
    ui.metric_row(
        [
            ("Implied volatility", f"{iv:.2%}", None),
            ("Vega at that vol", f"{vega_at_iv:+.4f}", None),
            ("Price check", ui.money(float(bs.price(market.spot, strike, T, market.rate,
                                                    market.div_yield, iv, kind)), 4), None),
        ],
        help_texts=[
            "The volatility that reproduces the observed price.",
            f"Quoted {bs.display_units('vega')}. The larger it is, the more the price "
            f"actually depends on volatility, and the more you can trust the number "
            f"on the left.",
            "The observed price, re-derived from the implied vol. It should match "
            "what you typed.",
        ],
    )

    if vega_at_iv < 0.005:
        ui.caveat(
            f"Vega here is only {vega_at_iv:.5f} per vol point, so the price barely "
            f"depends on volatility at all. A whole range of vols prices this option "
            f"identically to the nearest cent, and the number above is not a reliable "
            f"reading. This is why implied vols from deep in- or out-of-the-money "
            f"options near expiry should be treated with suspicion rather than plotted."
        )
    else:
        st.caption(
            f"Vega of {vega_at_iv:.4f} per vol point means a one-cent error in the quoted "
            f"price moves the implied vol by roughly {0.01 / vega_at_iv:.3f} of a vol "
            f"point. The bigger the vega, the more trustworthy the implied vol."
        )

    _inversion_chart(market, strike, T, kind, observed)


def _inversion_chart(market, strike, T, kind, observed) -> None:
    vols = np.linspace(0.01, 1.2, 260)
    prices = bs.price(market.spot, strike, T, market.rate, market.div_yield, vols, kind)
    df = pd.DataFrame({"vol": vols, "price": np.asarray(prices)})
    ui.show(
        ui.line_chart(df, "vol", "price", height=270, x_title="Volatility",
                      y_title=f"{kind} price", zero_y=False)
        + ui.zero_rule(value=float(observed))
    )
    st.caption(
        "Price against volatility, with the observed price as the dashed line. Implied "
        "vol is where they cross. The curve rises everywhere, so there is at most one "
        "crossing, but where the curve is nearly flat the crossing point is poorly "
        "determined, which is exactly the vega warning above."
    )
