"""Page 4 - the hedging lab, where the greeks pay out.

Everything else in the app is static: a number, a curve, a surface. This page
runs time forward. You sell an option, hedge it, and find out what you actually
made - which is the only way the relationship between gamma, theta and realised
volatility ever really lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optlab import bs, hedge, ui
from optlab.strategies import Leg


@st.cache_data(show_spinner=False)
def _run(kind, qty, strike, days, implied_vol, real_vol, drift, rate, div,
         spot, n_steps, cost_rate, band, n_paths, seed):
    """Cached so moving an unrelated slider does not re-simulate everything."""
    spec = hedge.HedgeSpec(
        legs=[Leg(kind, qty, strike, days / 365.0)],
        implied_vol=implied_vol, real_vol=real_vol, drift=drift, rate=rate,
        div_yield=div, spot=spot, n_steps=n_steps, cost_rate=cost_rate,
        band=band, n_paths=n_paths, seed=seed,
    )
    result = hedge.run(spec)
    return result, hedge.theoretical_edge(spec)


def render() -> None:
    ui.keep("hg_kind", "hg_qty", "hg_strike", "hg_days", "hg_impl", "hg_real",
            "hg_steps", "hg_cost", "hg_band", "hg_paths", "hg_seed", "hg_drift")
    market = ui.current_market()

    ui.page_header(
        "Hedging lab",
        "Sell an option, hedge the delta, run the clock. A delta-hedged position earns "
        "gamma and pays theta, and nothing else. This page is that sentence, simulated.",
    )

    spec_args = _controls(market)
    result, edge = _run(*spec_args)

    _verdict(result, edge)
    st.divider()
    chosen = _path_picker(result)
    _distribution(result, chosen)
    st.divider()
    _single_path(result, chosen)
    st.divider()
    _frequency_study(spec_args)


def _controls(market) -> tuple:
    with st.sidebar:
        st.markdown("### The trade")
        kind = st.segmented_control("Option", [bs.CALL, bs.PUT], default=bs.CALL,
                                    key="hg_kind", selection_mode="single") or bs.CALL
        qty = st.selectbox("Position", [-1.0, 1.0], index=0, key="hg_qty",
                           format_func=lambda q: "sell 1 (short)" if q < 0 else "buy 1 (long)")
        strike = st.slider("Strike", float(market.spot * 0.7), float(market.spot * 1.3),
                           float(market.spot), 1.0, key="hg_strike")
        days = st.slider("Days to expiry", 5, 365, 90, key="hg_days")

        st.markdown("### The two volatilities")
        implied = st.slider("Sold at implied vol", 0.05, 0.80, 0.25, 0.01, key="hg_impl",
                            format="%.2f",
                            help="The vol the option is priced at, and the vol you compute "
                                 "your hedge ratio with.")
        real = st.slider("Market actually moves at", 0.05, 0.80, 0.20, 0.01, key="hg_real",
                         format="%.2f",
                         help="The volatility of the simulated path. The gap between this "
                              "and the implied vol is the entire edge.")

        st.markdown("### Hedging rules")
        n_steps = st.select_slider("Rebalances over the life",
                                   [5, 10, 25, 50, 100, 250, 500], value=50, key="hg_steps")
        cost_bps = st.slider("Transaction cost (bp of traded notional)", 0.0, 25.0, 0.0, 0.5,
                             key="hg_cost",
                             help="Charged on every share traded, including the final unwind.")
        band = st.slider("No-trade band (delta)", 0.0, 0.30, 0.0, 0.01, key="hg_band",
                         help="Leave the hedge alone until delta has drifted this far. "
                              "Fewer trades, more slippage.")

        with st.expander("Simulation"):
            n_paths = st.select_slider("Paths", [100, 250, 500, 1000, 2000],
                                       value=500, key="hg_paths")
            drift = st.slider("Real-world drift", -0.20, 0.30, 0.05, 0.01, key="hg_drift")
            seed = st.number_input("Random seed", 0, 10_000, 7, key="hg_seed")

    return (kind, qty, strike, days, implied, real, drift, market.rate,
            market.div_yield, market.spot, n_steps, cost_bps / 10_000.0, band,
            n_paths, int(seed))


def _verdict(result: hedge.HedgeResult, edge: float) -> None:
    s = result.summary()
    spec = result.spec
    stderr = result.std_pnl / np.sqrt(max(len(result.final_pnl), 1))

    ui.metric_row(
        [
            ("Premium received" if s["premium"] < 0 else "Premium paid",
             ui.money(abs(s["premium"])), None),
            ("Average P&L", f"{s['mean']:+,.3f}", None),
            ("Spread of outcomes", f"{s['std']:,.3f}", None),
            ("Profitable paths", ui.pct(s["win_rate"], 0), None),
        ],
        help_texts=[
            "What the option was sold or bought for at the implied vol.",
            f"Mean across every simulated path, after costs. Standard error ±{stderr:.3f}.",
            "One standard deviation of the outcome across paths. This is hedging "
            "error, not edge.",
            "Share of paths that finished in profit.",
        ],
    )

    gap = spec.implied_vol - spec.real_vol
    short = spec.legs[0].qty < 0
    if abs(gap) < 0.005:
        ui.takeaway(
            "You sold volatility at exactly the vol the market delivered. The average "
            "P&L is around zero and every path still lands somewhere different. That "
            "scatter is not edge and not luck about direction. It is the price of "
            "hedging at discrete moments instead of continuously."
        )
    else:
        rich = gap > 0
        winner = (rich and short) or (not rich and not short)
        ui.takeaway(
            f"Implied vol is {abs(gap) * 100:.1f} points "
            f"{'above' if rich else 'below'} what the market actually delivers, and you "
            f"are {'short' if short else 'long'} the option. "
            f"So you {'collect' if winner else 'pay'} that difference, "
            f"{'and the win rate is close to certain' if abs(gap) > 0.05 else 'though noise still decides individual paths'}. "
            f"Theory says the edge is about {edge:+,.3f}; the simulation came out at "
            f"{result.mean_pnl:+,.3f}. Direction of the market barely enters into it."
        )
    if s["total_cost"] > 0:
        st.caption(
            f"Transaction costs consumed about {s['total_cost']:,.3f} per path on the "
            f"first path's trading schedule. Costs only ever subtract, and they grow "
            f"with rebalancing frequency."
        )


def _path_picker(result: hedge.HedgeResult) -> int:
    """Choose which of the simulated paths to open up below."""
    st.markdown("#### Pick a path to inspect")
    total = len(result.final_pnl)
    options = list(hedge.NOTABLE) + ["Pick by number"]
    choice = st.segmented_control(
        "Which path", options, default="Worst", key="hg_path_pick",
        selection_mode="single", label_visibility="collapsed",
    ) or "Worst"

    if choice == "Pick by number":
        index = int(st.number_input("Path number", 0, total - 1, 0, key="hg_path_index",
                                    help=f"Any path from 0 to {total - 1}."))
        note = "Whichever path you asked for."
    else:
        index = result.path_index(choice)
        note = hedge.NOTABLE[choice]

    pnl = float(result.final_pnl[index])
    rank = result.rank_of(index)
    st.caption(
        f"**Path {index}** finished at **{pnl:+,.3f}**, ranked {rank:,} out of "
        f"{total:,} from worst to best. {note}"
    )
    return index


def _distribution(result: hedge.HedgeResult, chosen: int) -> None:
    st.markdown("#### Every path's final P&L")
    ui.show(ui.histogram(result.final_pnl, bins=45, height=280,
                         x_title="Profit and loss per path",
                         highlight=float(result.final_pnl[chosen]),
                         highlight_label=f"path {chosen}"))
    s = result.summary()
    st.caption(
        f"Median {s['p50']:+,.3f}. The middle 90% of paths land between "
        f"{s['p05']:+,.3f} and {s['p95']:+,.3f}, worst {s['worst']:+,.3f}, "
        f"best {s['best']:+,.3f}. The orange line is the mean, the green line is the "
        f"path you selected, and the dashed line is zero. When implied and realised vol "
        f"differ enough, the whole distribution shifts to one side of zero and hedging "
        f"error can no longer change the sign of the answer."
    )


def _single_path(result: hedge.HedgeResult, chosen: int) -> None:
    st.markdown(f"#### Path {chosen}, step by step")
    steps = result.detail_for(chosen)
    strike = result.spec.legs[0].strike

    ui.show(
        ui.line_chart(steps[["days", "spot"]], "days", "spot", height=220,
                      x_title="Days elapsed", y_title="Spot", zero_y=False)
        + ui.zero_rule(value=strike)
    )
    st.caption(f"The simulated spot path. The dashed line is the strike at {strike:,.2f}.")

    hedge_df = pd.concat([
        pd.DataFrame({"days": steps["days"], "value": steps["delta"], "line": "Option delta"}),
        pd.DataFrame({"days": steps["days"], "value": steps["shares"], "line": "Shares held"}),
    ], ignore_index=True)
    ui.show(ui.line_chart(hedge_df, "days", "value", "line", height=230,
                          x_title="Days elapsed", y_title="Delta / shares",
                          color_title="", zero_line=True,
                          order=["Option delta", "Shares held"],
                          dashed=["Shares held"]))
    st.caption(
        "The hedge is the mirror of the delta: whatever the option's delta is, you hold "
        "the opposite in stock. Gamma is why that mirror has to keep being redrawn."
    )

    attribution = pd.concat([
        pd.DataFrame({"days": steps["days"], "value": steps["cum_gamma"], "line": "Gamma earned"}),
        pd.DataFrame({"days": steps["days"], "value": steps["cum_theta"], "line": "Theta paid"}),
        pd.DataFrame({"days": steps["days"], "value": steps["pnl"], "line": "Actual P&L"}),
    ], ignore_index=True)
    ui.show(ui.line_chart(attribution, "days", "value", "line", height=300,
                          x_title="Days elapsed", y_title="Cumulative P&L",
                          color_title="", zero_line=True,
                          order=["Gamma earned", "Theta paid", "Actual P&L"],
                          dashed=["Actual P&L"]))
    _attribution_takeaway(steps, result, chosen)
    _reconciliation(steps)

    with st.expander("The numbers behind this path"):
        show_cols = ["step", "days", "spot", "delta", "gamma", "shares", "trade",
                     "tx_cost", "gamma_term", "theta_term", "carry_term", "pnl"]
        st.dataframe(
            steps[show_cols].round(5), width="stretch", height=320, hide_index=True,
            column_config={
                "days": st.column_config.NumberColumn("days", format="%.2f"),
                "gamma_term": st.column_config.NumberColumn("gamma P&L", format="%.5f"),
                "theta_term": st.column_config.NumberColumn("theta P&L", format="%.5f"),
                "carry_term": st.column_config.NumberColumn("financing", format="%.5f"),
                "tx_cost": st.column_config.NumberColumn("cost", format="%.5f"),
            },
        )


def _attribution_takeaway(steps: pd.DataFrame, result: hedge.HedgeResult,
                          chosen: int) -> None:
    """Say what THIS path did, not what paths do in general."""
    last = steps.iloc[-1]
    gamma, theta = float(last["cum_gamma"]), float(last["cum_theta"])
    travelled = float(np.abs(np.diff(steps["spot"].to_numpy())).sum())
    net_move = float(steps["spot"].iloc[-1] - steps["spot"].iloc[0])
    base = (
        "This is the whole argument in one picture. Gamma and theta run in opposite "
        "directions, and where they end up relative to each other is the P&L. The "
        "direction the spot went does not appear anywhere."
    )
    detail = (
        f" On this path the spot travelled {travelled:,.1f} in total to end up "
        f"{net_move:+,.1f} from where it started, and that total distance, not the "
        f"{net_move:+,.1f}, is what set the gamma line at {gamma:+,.3f} against "
        f"{theta:+,.3f} of theta."
    )
    ui.takeaway(base + detail)


def _reconciliation(steps: pd.DataFrame) -> None:
    """Show the books balancing rather than claiming that they do."""
    last = steps.iloc[-1]
    gamma, theta = float(last["cum_gamma"]), float(last["cum_theta"])
    carry, slip = float(last["cum_carry"]), float(last["cum_slippage"])
    costs, realised = float(last["cum_tx_cost"]), float(last["pnl"])
    total = gamma + theta + carry + slip - costs

    rows = [
        ("Gamma earned", gamma, "Half of gamma times every squared move the spot made."),
        ("Theta paid", theta, "The daily bill for holding the position."),
        ("Financing", carry, "Interest on the cash balance and dividends on the stock hedge."),
        ("Hedge slippage", slip, "Delta left unhedged inside the no-trade band. Zero when "
                                 "the band is zero."),
        ("Transaction costs", -costs, "The spread paid on every share traded."),
        ("Attributed total", total, "The five lines above, added up."),
        ("Realised P&L", realised, "What the book was actually worth at the end."),
    ]
    df = pd.DataFrame(
        [{"Component": n, "Amount": v, "What it is": d} for n, v, d in rows]
    )
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={"Amount": st.column_config.NumberColumn("Amount", format="%.4f")},
    )
    residual = abs(total - realised)
    scale = max(abs(realised), 1e-9)
    note = (
        f"The attribution lands within {residual:.4f} of the realised number. That "
        f"residual is everything the greeks do not name: the third-order part of the "
        f"Taylor expansion, which delta, gamma and theta truncate."
    )
    if residual > 0.1 * scale:
        note += (
            " It is sizeable on this path, which is itself informative. The expansion "
            "is only accurate over small moves, so coarse rebalancing, a wide no-trade "
            "band, or a path that jumps around lets the untracked term grow. Raise the "
            "rebalance count and watch it shrink."
        )
    else:
        note += " Everything else in this trade has a greek attached to it."
    st.caption(note)


def _frequency_study(spec_args: tuple) -> None:
    st.markdown("#### How often should you rebalance?")
    st.caption(
        "Hedging more often shrinks the random error but pays the spread more times. "
        "With zero costs, more is always better and the error falls like one over the "
        "square root of the number of rebalances. Add a cost and the curve turns: "
        "the best hedging frequency becomes finite, and it depends on what you pay to trade."
    )
    if not st.toggle("Run the sweep", key="hg_sweep",
                     help="Simulates the same trade at several rebalancing frequencies."):
        return

    args = list(spec_args)
    frequencies = [5, 10, 25, 50, 100, 250]
    rows = []
    progress = st.progress(0.0, text="Simulating...")
    for i, n in enumerate(frequencies, start=1):
        args[10] = n  # n_steps
        args[13] = min(args[13], 500)  # keep the sweep quick
        res, _ = _run(*args)
        rows.append({"rebalances": n, "mean": res.mean_pnl, "spread": res.std_pnl})
        progress.progress(i / len(frequencies), text=f"Simulating {n} rebalances...")
    progress.empty()

    df = pd.DataFrame(rows)
    ui.show(ui.line_chart(df, "rebalances", "spread", height=260,
                          x_title="Rebalances over the option's life",
                          y_title="Spread of P&L (1 s.d.)"))
    ui.show(ui.line_chart(df, "rebalances", "mean", height=240,
                          x_title="Rebalances over the option's life",
                          y_title="Average P&L", zero_line=True))

    df["spread x sqrt(n)"] = df["spread"] * np.sqrt(df["rebalances"])
    st.dataframe(df.round(4), width="stretch", hide_index=True)
    ui.takeaway(
        "The last column is the test: if hedging error really falls like one over the "
        "square root of the rebalance count, then spread times the square root of n is "
        "roughly constant down the column. It is. That is why halving your hedging "
        "error costs four times as many trades."
    )
