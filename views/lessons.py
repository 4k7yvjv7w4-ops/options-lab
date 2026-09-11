"""Page 6 - predict first, then find out.

Reading that gamma rises near expiry teaches almost nothing. Committing to a
guess and being wrong teaches a great deal. So every exercise here asks for an
answer before it shows one, and every revealed answer is computed from the same
model the rest of the app runs on, live, at whatever market you have dialled in.
Nothing below is a hardcoded number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optlab import bs, hedge, ui
from optlab.strategies import Leg, Market, net_greeks, preset_legs


def render() -> None:
    market = ui.current_market()
    ui.page_header(
        "Lessons",
        "Seven questions with counter-intuitive answers. Commit to a guess before you "
        "reveal each one. Every answer is recomputed from your current market settings, "
        "so the numbers are real rather than remembered.",
    )

    st.session_state.setdefault("lesson_state", {})
    state = st.session_state.lesson_state

    answered = [l["id"] for l in LESSONS if state.get(l["id"], {}).get("revealed")]
    correct = [i for i in answered if state[i].get("choice") == _lesson(i)["answer"]]
    if answered:
        st.progress(len(answered) / len(LESSONS),
                    text=f"{len(correct)} right out of {len(answered)} revealed, "
                         f"{len(LESSONS)} in total")

    for lesson in LESSONS:
        _render_lesson(lesson, market, state)

    st.divider()
    if st.button("Reset all answers"):
        st.session_state.lesson_state = {}
        st.rerun()


def _lesson(lesson_id: str) -> dict:
    return next(l for l in LESSONS if l["id"] == lesson_id)


def _render_lesson(lesson: dict, market: Market, state: dict) -> None:
    entry = state.setdefault(lesson["id"], {"choice": None, "revealed": False})
    done = entry["revealed"]
    mark = ""
    if done:
        mark = " ✅" if entry.get("choice") == lesson["answer"] else " ❌"

    with st.expander(f"**{lesson['title']}**{mark}", expanded=not done):
        st.markdown(lesson["question"])
        choice = st.radio(
            "Your answer", lesson["options"], index=entry["choice"],
            key=f"radio_{lesson['id']}", label_visibility="collapsed",
            disabled=done,
        )
        picked = lesson["options"].index(choice) if choice is not None else None

        if not done:
            # Reveal stays locked until a guess is on the record. Predicting is
            # the part that does the teaching, so skipping it defeats the page.
            if st.button("Reveal", key=f"reveal_{lesson['id']}", type="primary",
                         disabled=picked is None,
                         help=None if picked is not None else "Pick an answer first."):
                entry["choice"] = picked
                entry["revealed"] = True
                st.rerun()
            return

        answer_text = lesson["options"][lesson["answer"]]
        if entry["choice"] == lesson["answer"]:
            st.success(f"Correct — {answer_text}", icon="✅")
        elif entry["choice"] is None:
            st.info(f"The answer is **{answer_text}**.", icon="💡")
        else:
            st.error(
                f"You said *{lesson['options'][entry['choice']]}*. "
                f"The answer is **{answer_text}**.",
                icon="❌",
            )
        lesson["explain"](market)
        if st.button("Try again", key=f"retry_{lesson['id']}"):
            state[lesson["id"]] = {"choice": None, "revealed": False}
            st.rerun()


# --- the explanations, each computed live ---------------------------------


def _explain_sqrt_time(market: Market) -> None:
    K, r, q, v = market.spot, market.rate, market.div_yield, market.vol
    long_p = float(bs.price(market.spot, K, 90 / 365, r, q, v))
    short_p = float(bs.price(market.spot, K, 45 / 365, r, q, v))
    st.markdown(
        f"At your current market, the 90-day at-the-money call is worth "
        f"**{long_p:,.3f}** and the 45-day is worth **{short_p:,.3f}**. Halving the time "
        f"removed only **{(1 - short_p / long_p):.1%}** of the value, not half of it.\n\n"
        f"Time value grows with the *square root* of time, because that is how far a "
        f"random walk travels. Four times the time buys twice the expected movement, not "
        f"four times. This is why the last few days of an option's life are cheap to buy "
        f"and why selling short-dated options collects less premium than the calendar "
        f"suggests."
    )
    days = np.arange(1, 366)
    prices = bs.price(market.spot, K, days / 365, r, q, v)
    ui.show(ui.line_chart(pd.DataFrame({"days": days, "price": np.asarray(prices)}),
                          "days", "price", height=260, x_title="Days to expiry",
                          y_title="At-the-money call price"))
    st.caption("The curve bends because of the square root. Doubling the maturity never doubles the price.")


def _explain_gamma_spike(market: Market) -> None:
    K, r, q, v = market.spot, market.rate, market.div_yield, market.vol
    g30 = float(bs.gamma(market.spot, K, 30 / 365, r, q, v))
    g3 = float(bs.gamma(market.spot, K, 3 / 365, r, q, v))
    st.markdown(
        f"At the money, gamma goes from **{g30:.4f}** at 30 days to **{g3:.4f}** at 3 days "
        f"— about **{g3 / g30:.1f} times larger**. Gamma scales with one over the square "
        f"root of time, so shrinking the maturity by a factor of ten raises it by a bit "
        f"over three.\n\n"
        f"And it does not merely rise; it *concentrates*. The hill narrows as it grows. "
        f"A few days from expiry, an option pinned near its strike has enormous gamma "
        f"over a tiny range of spot and almost none outside it. That is why traders talk "
        f"about pin risk, and why an expiring at-the-money position is genuinely hard to "
        f"hedge no matter how disciplined you are."
    )
    spots = np.linspace(market.spot * 0.8, market.spot * 1.2, 260)
    rows = [pd.DataFrame({"spot": spots,
                          "gamma": np.asarray(bs.gamma(spots, K, d / 365, r, q, v)),
                          "maturity": f"{d}d"}) for d in (3, 10, 30, 90)]
    ui.show(ui.line_chart(pd.concat(rows, ignore_index=True), "spot", "gamma", "maturity",
                          height=280, x_title="Spot", y_title="Gamma",
                          color_title="Days to expiry",
                          order=["3d", "10d", "30d", "90d"]))


def _explain_delta_probability(market: Market) -> None:
    K, r, q, v = market.spot * 1.15, market.rate, market.div_yield, market.vol
    d_long = float(bs.delta(market.spot, K, 365 / 365, r, q, v))
    d_short = float(bs.delta(market.spot, K, 5 / 365, r, q, v))
    st.markdown(
        f"Take a call struck 15% above the spot. With a year to run its delta is "
        f"**{d_long:.3f}**; with five days left it is **{d_short:.4f}**. Same strike, same "
        f"market, same vol — the only thing that changed is how much time remains for the "
        f"spot to get there.\n\n"
        f"Delta is close to the risk-neutral probability of finishing in the money, so it "
        f"has to collapse toward zero as the time to travel runs out. The useful habit: "
        f"read a delta as a rough chance, not as a hedge ratio you were handed. A 5-delta "
        f"option is a lottery ticket and will behave like one."
    )
    days = np.arange(1, 366)
    ui.show(ui.line_chart(
        pd.DataFrame({"days": days,
                      "delta": np.asarray(bs.delta(market.spot, K, days / 365, r, q, v))}),
        "days", "delta", height=250, x_title="Days to expiry",
        y_title="Delta of a 15% out-of-the-money call"))


def _explain_vega_maturity(market: Market) -> None:
    K, r, q, v = market.spot, market.rate, market.div_yield, market.vol
    v1m = float(bs.scale_for_display("vega", bs.vega(market.spot, K, 30 / 365, r, q, v)))
    v1y = float(bs.scale_for_display("vega", bs.vega(market.spot, K, 365 / 365, r, q, v)))
    st.markdown(
        f"Per vol point, the one-month at-the-money option is worth **{v1m:.4f}** and the "
        f"one-year is worth **{v1y:.4f}** — roughly **{v1y / v1m:.1f} times** as much, "
        f"close to the square root of twelve.\n\n"
        f"Vega grows with the square root of maturity for the same reason time value does. "
        f"If your view is that volatility is mispriced, expressing it in weekly options "
        f"gives you very little position for the premium and the theta bill you take on. "
        f"Views on volatility want maturity."
    )
    days = np.arange(7, 731)
    ui.show(ui.line_chart(
        pd.DataFrame({"days": days,
                      "vega": bs.scale_for_display("vega", bs.vega(market.spot, K, days / 365, r, q, v))}),
        "days", "vega", height=250, x_title="Days to expiry",
        y_title="Vega per vol point"))


@st.cache_data(show_spinner=False)
def _sell_rich_vol(spot, rate, div):
    spec = hedge.HedgeSpec(
        legs=[Leg(bs.CALL, -1.0, spot, 0.25)], implied_vol=0.30, real_vol=0.20,
        rate=rate, div_yield=div, spot=spot, n_steps=100, n_paths=1200, seed=41,
    )
    return hedge.run(spec)


def _explain_selling_vol(market: Market) -> None:
    res = _sell_rich_vol(market.spot, market.rate, market.div_yield)
    s = res.summary()
    st.markdown(
        f"Selling a three-month at-the-money call at 30 vol into a market that moves at "
        f"20, and rebalancing 100 times, **{s['win_rate']:.0%}** of paths finish in profit. "
        f"The average is **{s['mean']:+,.3f}** and the worst path of "
        f"{len(res.final_pnl):,} was **{s['worst']:+,.3f}**.\n\n"
        f"This is the part that surprises people: the profit barely depends on which way "
        f"the market went. A delta hedge removes the direction, leaving a bet purely on "
        f"how *much* the market moved against how much you were paid to assume it would. "
        f"Sell volatility above what gets delivered and you win on nearly every path, "
        f"whether the market rallies, falls, or goes nowhere.\n\n"
        f"The catch lives in the tail. Real markets deliver variance in bursts, and the "
        f"vol you sold is not guaranteed to be the vol you get."
    )
    ui.show(ui.histogram(res.final_pnl, bins=45, height=260,
                         x_title="P&L per path, short a 30-vol call into a 20-vol market"))


@st.cache_data(show_spinner=False)
def _frequency_effect(spot, rate, div):
    out = {}
    for n in (25, 100, 400):
        spec = hedge.HedgeSpec(
            legs=[Leg(bs.CALL, -1.0, spot, 0.25)], implied_vol=0.20, real_vol=0.20,
            rate=rate, div_yield=div, spot=spot, n_steps=n, n_paths=1200, seed=43,
        )
        out[n] = hedge.run(spec).std_pnl
    return out


def _explain_hedge_frequency(market: Market) -> None:
    sd = _frequency_effect(market.spot, market.rate, market.div_yield)
    st.markdown(
        f"Hedging a fairly-priced option 25 times gives a P&L spread of **{sd[25]:.3f}**. "
        f"At 100 rebalances it is **{sd[100]:.3f}**, and at 400 it is **{sd[400]:.3f}**. "
        f"Each fourfold increase in effort roughly halves the error, because the error "
        f"falls with one over the square root of the rebalance count.\n\n"
        f"That square root is expensive. Cutting your hedging error by a factor of ten "
        f"means trading a hundred times as often, and every one of those trades pays the "
        f"spread. Which is why nobody hedges continuously: past some frequency the "
        f"transaction costs you add exceed the error you remove, and the optimal hedge "
        f"is deliberately imperfect."
    )
    df = pd.DataFrame({"rebalances": list(sd), "spread": list(sd.values())})
    df["spread x sqrt(n)"] = df["spread"] * np.sqrt(df["rebalances"])
    st.dataframe(df.round(4), width="stretch", hide_index=True)
    st.caption("The last column stays roughly constant, which is the square-root law showing up directly.")


def _explain_gamma_theta(market: Market) -> None:
    straddle = preset_legs("Straddle", market)
    condor = preset_legs("Iron condor", market)
    gs, gc = net_greeks(straddle, market), net_greeks(condor, market)
    st.markdown(
        f"No. They are two sides of one coin, and the model makes it impossible.\n\n"
        f"The straddle is long gamma at **{gs['gamma']:+.4f}** and pays theta at "
        f"**{bs.scale_for_display('theta', gs['theta']):+.3f}** a day. The iron condor is "
        f"short gamma at **{gc['gamma']:+.4f}** and collects "
        f"**{bs.scale_for_display('theta', gc['theta']):+.3f}** a day. Every position in "
        f"this app obeys the same trade.\n\n"
        f"The reason is the pricing equation itself. For a delta-hedged position, theta "
        f"is very nearly minus one half gamma times spot squared times variance. The "
        f"premium you pay each day is precisely the value of the convexity you own. "
        f"A structure that promised both would be an arbitrage, and the price of every "
        f"option is set so that it is not available.\n\n"
        f"What you *can* have is a view: own gamma when you think the market will move "
        f"more than the premium implies, sell it when you think it will move less. That "
        f"is a forecast, not a free lunch."
    )
    spots = np.linspace(market.spot * 0.7, market.spot * 1.3, 240)
    from optlab.strategies import net_greek_curve
    rows = []
    for name, legs in (("Straddle", straddle), ("Iron condor", condor)):
        rows.append(pd.DataFrame({
            "spot": spots,
            "gamma": net_greek_curve(legs, spots, "gamma", 0.0, market),
            "position": name,
        }))
    ui.show(ui.line_chart(pd.concat(rows, ignore_index=True), "spot", "gamma", "position",
                          height=270, x_title="Spot", y_title="Net gamma",
                          color_title="", zero_line=True,
                          order=["Straddle", "Iron condor"]))


LESSONS = [
    {
        "id": "sqrt_time",
        "title": "1. Halving the time to expiry",
        "question": "An at-the-money call has 90 days to run. Cut that to 45 days and "
                    "leave everything else alone. What happens to its price?",
        "options": ["It roughly halves", "It falls by about 30%",
                    "It falls by about 10%", "It barely changes"],
        "answer": 1,
        "explain": _explain_sqrt_time,
    },
    {
        "id": "gamma_spike",
        "title": "2. Gamma as expiry approaches",
        "question": "An at-the-money option's gamma at 30 days versus the same option "
                    "at 3 days. How do they compare?",
        "options": ["Roughly the same", "About 3 times larger at 3 days",
                    "About 10 times larger at 3 days", "Smaller at 3 days"],
        "answer": 1,
        "explain": _explain_gamma_spike,
    },
    {
        "id": "delta_prob",
        "title": "3. What delta is really telling you",
        "question": "A call struck 15% above the spot. Its delta with one year to run, "
                    "against its delta with five days to run?",
        "options": ["Almost unchanged", "Somewhat lower at five days",
                    "Close to zero at five days", "Higher at five days"],
        "answer": 2,
        "explain": _explain_delta_probability,
    },
    {
        "id": "vega_maturity",
        "title": "4. Where to express a view on volatility",
        "question": "You think implied volatility is too low. One-month or one-year "
                    "at-the-money options — which gives you more vega per contract?",
        "options": ["One-month, by a little", "About the same",
                    "One-year, by roughly 3 to 4 times", "One-year, by about 12 times"],
        "answer": 2,
        "explain": _explain_vega_maturity,
    },
    {
        "id": "sell_vol",
        "title": "5. Selling volatility that is too expensive",
        "question": "You sell a three-month call at 30 vol. The market then moves at 20 "
                    "vol, and you delta-hedge throughout. On what share of paths do you "
                    "make money?",
        "options": ["About half — it depends which way the market goes",
                    "About 70%", "Very nearly all of them",
                    "It depends entirely on the drift"],
        "answer": 2,
        "explain": _explain_selling_vol,
    },
    {
        "id": "hedge_freq",
        "title": "6. The price of hedging more often",
        "question": "You quadruple how often you rebalance a delta hedge. What happens "
                    "to the spread of your final P&L?",
        "options": ["It falls to about a quarter", "It falls to about half",
                    "It is unchanged", "It falls almost to zero"],
        "answer": 1,
        "explain": _explain_hedge_frequency,
    },
    {
        "id": "gamma_theta",
        "title": "7. The one trade that does not exist",
        "question": "Is there any position, anywhere in this app, that is long gamma "
                    "and also collects theta?",
        "options": ["Yes, a calendar spread", "Yes, an iron condor",
                    "No — the two always have opposite signs",
                    "Yes, if rates are high enough"],
        "answer": 2,
        "explain": _explain_gamma_theta,
    },
]
