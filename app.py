"""Options Lab - a Streamlit app for building intuition about options.

Run it with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Options Lab",
    page_icon="📈",
    layout="wide",
    # "auto", not "expanded": on a phone an expanded sidebar covers the whole
    # screen, so the first thing you see is the navigation rather than the app.
    # Auto keeps it open on a desktop and tucks it away on a narrow screen.
    initial_sidebar_state="auto",
)

from optlab import ui  # noqa: E402  (must follow set_page_config)
from views import builder, hedging, home, lessons, pricer, surfaces, volsmile  # noqa: E402

PAGES = {
    "": [st.Page(home.render, title="Overview", icon=":material/home:", default=True)],
    "Single option": [
        st.Page(pricer.render, title="Price and greeks", icon=":material/tune:",
                url_path="pricer"),
        st.Page(surfaces.render, title="Greek maps", icon=":material/grid_on:",
                url_path="maps"),
    ],
    "Positions": [
        st.Page(builder.render, title="Strategy builder", icon=":material/stacked_line_chart:",
                url_path="builder"),
    ],
    "Risk in motion": [
        st.Page(hedging.render, title="Hedging lab", icon=":material/balance:",
                url_path="hedging"),
        st.Page(volsmile.render, title="Implied vol and smile", icon=":material/sentiment_satisfied:",
                url_path="smile"),
    ],
    "Practice": [
        st.Page(lessons.render, title="Lessons", icon=":material/school:", url_path="lessons"),
    ],
}


def main() -> None:
    page = st.navigation(PAGES)
    # Drawn once per run, below the navigation, so every page shares one market.
    st.session_state.market = ui.market_sidebar()
    with st.sidebar:
        st.divider()
        st.caption(
            "Educational model only. Black-Scholes with constant volatility. "
            "Not investment advice."
        )
    page.run()


main()
