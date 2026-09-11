"""Page 0 - orientation, and an honest statement of what the model assumes."""

from __future__ import annotations

import streamlit as st

from optlab import ui

ROUTE = [
    ("Price and greeks", "Start here. One option, every sensitivity, and what each one "
                         "means when you move it."),
    ("Greek maps", "The same greeks drawn over spot and time together. This is where "
                   "the shape of gamma near expiry becomes impossible to unsee."),
    ("Strategy builder", "Combine legs into the structures people actually trade, and "
                         "watch the payoff and the net greeks change as you do."),
    ("Hedging lab", "Run the clock. Sell an option, hedge it, and see where the profit "
                    "and loss really comes from."),
    ("Implied vol and the smile", "Prices restated as volatility, and what happens to "
                                  "your hedge when different strikes trade at different vols."),
    ("Lessons", "Seven questions with answers most people guess wrong. Worth doing after "
                "the pages above, or instead of them if you prefer to learn by being wrong."),
]


def render() -> None:
    ui.page_header(
        "Options Lab",
        "A place to build intuition about options by moving things and watching what "
        "happens. Every number on every page is computed live from the same pricing "
        "model, so the charts and the claims cannot drift apart.",
    )

    st.markdown("#### Where to go")
    for name, why in ROUTE:
        st.markdown(f"**{name}** — {why}")

    st.divider()
    st.markdown("#### The three ideas worth carrying away")
    c1, c2, c3 = st.columns(3, gap="medium", border=True)
    with c1:
        st.markdown(
            "###### Time enters as a square root\n"
            "Value, vega and hedging error all scale with the square root of time, not "
            "time itself. Four times the maturity buys twice the movement. It is the "
            "single most common place intuition goes wrong."
        )
    with c2:
        st.markdown(
            "###### Gamma and theta are one trade\n"
            "Owning convexity costs rent, every day, and the rent is priced to equal the "
            "convexity. No position anywhere gives you both. What you can have is a view "
            "on whether the rent is set too high."
        )
    with c3:
        st.markdown(
            "###### A hedged option is a bet on movement\n"
            "Delta-hedge an option and the direction of the market drops out almost "
            "entirely. What is left is how much the market moved against how much you "
            "were paid to assume it would."
        )

    st.divider()
    st.markdown("#### What this model assumes, and where it breaks")
    st.markdown(
        "Everything here is Black-Scholes-Merton with a lognormal spot and constant "
        "parameters. That is a teaching model, and it is wrong in specific ways worth "
        "knowing before you carry any of this to a real screen."
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            "- **Returns are not lognormal.** Real markets have fat tails and jumps. "
            "A move the model calls impossible happens every few years.\n"
            "- **Volatility is not constant.** It clusters, spikes, and mean-reverts. "
            "The smile page is the first repair, and it is still only a repair.\n"
            "- **Nothing gaps.** The hedging lab assumes you can always trade at the "
            "simulated price. The losses that end careers happen when you cannot."
        )
    with right:
        st.markdown(
            "- **No bid-ask on the option**, only on the hedge. Real entry and exit cost "
            "more than this suggests.\n"
            "- **European exercise only.** No early exercise, no dividends paid as "
            "discrete cash, no borrow costs.\n"
            "- **The paths are simulated, not historical.** Nothing here is a backtest, "
            "a forecast, or advice about any actual trade."
        )
    ui.takeaway(
        "The right way to use a wrong model is to learn its shapes and then learn where "
        "they fail. Both halves matter, and the second half is why the assumptions above "
        "are on the front page rather than buried in a footnote."
    )
