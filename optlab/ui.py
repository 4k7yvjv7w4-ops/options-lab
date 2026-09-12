"""Shared chrome: theme, charts, controls, and the little teaching panels.

Every page in the app draws from here, so a curve means the same thing and
wears the same colour wherever you meet it. Delta is always slot 1, gamma
always slot 2, and so on, which is what lets you carry an intuition from the
pricer page to the hedging page without re-learning the picture.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .strategies import Market

# --- palette --------------------------------------------------------------
# A validated categorical palette: the hue ORDER is the colour-blind-safety
# mechanism, so take the slots in order and do not reshuffle them. Dark is a
# selected set of steps for the dark surface, not an automatic inversion.

_LIGHT = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "good": "#0ca30c",
    "bad": "#d03b3b",
    "warn": "#fab219",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "div_low": "#d03b3b",
    "div_mid": "#f0efec",
    "div_high": "#2a78d6",
}

_DARK = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink_soft": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "good": "#0ca30c",
    "bad": "#d03b3b",
    "warn": "#fab219",
    "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"][::-1],
    "div_low": "#e66767",
    "div_mid": "#383835",
    "div_high": "#3987e5",
}


def theme_name() -> str:
    """'light' or 'dark', following the viewer's Streamlit theme."""
    try:
        kind = st.context.theme.type  # Streamlit resolves 'system' for us
        if kind in ("light", "dark"):
            return kind
    except Exception:
        pass
    return "light"


def colors() -> dict:
    return _DARK if theme_name() == "dark" else _LIGHT


#: Fixed colour per greek, so a curve keeps its identity across pages.
GREEK_SLOT = {
    "price": 0, "delta": 0, "gamma": 1, "vega": 2,
    "theta": 3, "rho": 4, "vanna": 5, "volga": 6, "charm": 7,
}


def greek_color(name: str) -> str:
    return colors()["series"][GREEK_SLOT.get(name, 0) % 8]


def series_scale(names) -> alt.Scale:
    """Pin each series to its slot, so filtering never repaints the survivors."""
    pal = colors()["series"]
    return alt.Scale(domain=list(names), range=[pal[i % 8] for i in range(len(names))])


# --- chart building blocks ------------------------------------------------

_AXIS = dict(grid=True, gridOpacity=0.6, tickSize=4, labelPadding=4, titlePadding=8)


def _base(df: pd.DataFrame, height: int) -> alt.Chart:
    return alt.Chart(df).properties(height=height)


def line_chart(
    df: pd.DataFrame, x: str, y: str, color: str | None = None, *,
    height: int = 320, x_title: str | None = None, y_title: str | None = None,
    color_title: str | None = None, tooltip=None, zero_line: bool = False,
    domain=None, order=None, zero_y: bool | None = None,
    dashed: list[str] | None = None,
) -> alt.LayerChart:
    """A line chart with a crosshair tooltip, because a static curve wastes the medium.

    `dashed` names series to draw with a dashed stroke. That is a second channel
    carrying the same distinction as colour, so the chart still reads when
    colour does not survive - a colour-blind viewer, a greyscale print, a
    projector. Use it whenever a chart has exactly two series.
    """
    c = colors()
    x_enc = alt.X(f"{x}:Q", title=x_title or x, axis=alt.Axis(**_AXIS))
    y_kwargs = dict(title=y_title or y, axis=alt.Axis(**_AXIS))
    if domain is not None:
        y_kwargs["scale"] = alt.Scale(domain=list(domain), nice=False)
    elif zero_y is False:
        # Anchoring at zero flattens a curve whose whole story is its shape,
        # like an implied-vol smile sitting between 0.18 and 0.24.
        y_kwargs["scale"] = alt.Scale(zero=False)
    y_enc = alt.Y(f"{y}:Q", **y_kwargs)
    tips = tooltip or [
        alt.Tooltip(f"{x}:Q", title=x_title or x, format=",.2f"),
        alt.Tooltip(f"{y}:Q", title=y_title or y, format=",.4f"),
    ]

    if color:
        names = order or list(pd.unique(df[color]))
        # An explicit "" means the series names speak for themselves; None means
        # fall back to the column name.
        legend_title = color if color_title is None else (color_title or None)
        col_enc = alt.Color(
            f"{color}:N", scale=series_scale(names), sort=names, title=legend_title,
            legend=alt.Legend(orient="top", direction="horizontal", offset=6,
                              titleLimit=0, symbolStrokeWidth=3),
        )
        if color not in [t.shorthand.split(":")[0] for t in tips if hasattr(t, "shorthand")]:
            tips = tips + [alt.Tooltip(f"{color}:N", title=legend_title or color)]
        enc = dict(x=x_enc, y=y_enc, color=col_enc)
        if dashed:
            enc["strokeDash"] = alt.StrokeDash(
                f"{color}:N", legend=None,
                scale=alt.Scale(domain=names,
                                range=[[6, 4] if n in dashed else [1, 0] for n in names]),
            )
        line = _base(df, height).mark_line(strokeWidth=2).encode(**enc)
        points = _base(df, height).mark_circle(size=70, opacity=0).encode(
            x=x_enc, y=y_enc, color=col_enc, tooltip=tips
        )
    else:
        line = _base(df, height).mark_line(strokeWidth=2, color=c["series"][0]).encode(x=x_enc, y=y_enc)
        points = _base(df, height).mark_circle(size=70, opacity=0).encode(x=x_enc, y=y_enc, tooltip=tips)

    layers = [line, points.add_params(alt.selection_point(on="pointerover", nearest=True, empty=False))]
    if zero_line:
        layers.insert(0, zero_rule(c))
    return alt.layer(*layers).properties(height=height)


def zero_rule(c=None, value: float = 0.0) -> alt.Chart:
    c = c or colors()
    return alt.Chart(pd.DataFrame({"_y": [value]})).mark_rule(
        color=c["axis"], strokeWidth=1, strokeDash=[4, 3]
    ).encode(y="_y:Q")


def vertical_rule(x_values, label: str | None = None, dash=True) -> alt.Chart:
    """A marker for a meaningful spot level: a strike, the current spot, a breakeven."""
    c = colors()
    vals = np.atleast_1d(np.asarray(x_values, dtype=float))
    df = pd.DataFrame({"_x": vals, "_label": [label or ""] * len(vals)})
    mark = df_rule = alt.Chart(df).mark_rule(
        color=c["muted"], strokeWidth=1, strokeDash=[4, 3] if dash else [],
    ).encode(x="_x:Q", tooltip=[alt.Tooltip("_x:Q", title=label or "level", format=",.2f")])
    return mark


def area_band(df: pd.DataFrame, x: str, lo: str, hi: str, color: str, opacity=0.18) -> alt.Chart:
    return alt.Chart(df).mark_area(opacity=opacity, color=color).encode(
        x=alt.X(f"{x}:Q", axis=alt.Axis(**_AXIS)), y=f"{lo}:Q", y2=f"{hi}:Q"
    )


def robust_limit(values, percentile: float = 98.0) -> tuple[float, bool]:
    """A colour-scale ceiling that ignores a lone extreme cell.

    Gamma at the money one day from expiry is tens of times anything else on
    the map. On a plain linear scale that single spike pushes every other cell
    into the palest step and the picture goes blank. Cutting the scale at a
    high percentile keeps the structure readable; the spike simply saturates,
    and the tooltip still reports its true value.
    """
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0, False
    top = float(np.nanmax(arr))
    cut = float(np.nanpercentile(arr, percentile))
    if cut <= 0:
        return (top or 1.0), False
    return cut, top > cut * 1.5


def signed_extreme(values) -> float:
    """The value furthest from zero, sign intact.

    Reporting max(abs(...)) would tell the reader theta "reaches 0.1974" when
    the number on the chart is -0.1974.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr[np.argmax(np.abs(arr))])


def heatmap(
    df: pd.DataFrame, x: str, y: str, value: str, *, height: int = 340,
    x_title=None, y_title=None, value_title=None, diverging: bool = False,
    value_format: str = ",.4f", percentile: float = 98.0,
) -> alt.Chart:
    """A magnitude grid: one hue light-to-dark, or two hues either side of zero."""
    c = colors()
    values = df[value].to_numpy()
    lim, _ = robust_limit(values, percentile)
    if diverging:
        scale = alt.Scale(range=[c["div_low"], c["div_mid"], c["div_high"]],
                          domain=[-lim, 0.0, lim], type="linear", clamp=True)
    else:
        # Clip toward zero from whichever side the field actually lives on.
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        steps, _ = sequential_scale(vmin, vmax)
        domain = [-lim, 0.0] if vmax <= 0 else [min(0.0, vmin), lim]
        scale = alt.Scale(range=steps, type="linear", domain=domain, clamp=True)
    return alt.Chart(df).mark_rect().encode(
        x=alt.X(f"{x}:Q", bin="binned", title=x_title or x, axis=alt.Axis(**_AXIS)),
        x2=f"{x}2:Q",
        y=alt.Y(f"{y}:Q", bin="binned", title=y_title or y, axis=alt.Axis(**_AXIS)),
        y2=f"{y}2:Q",
        color=alt.Color(f"{value}:Q", scale=scale, title=value_title or value,
                        legend=alt.Legend(orient="right", gradientLength=180)),
        tooltip=[
            alt.Tooltip(f"{x}:Q", title=x_title or x, format=",.2f"),
            alt.Tooltip(f"{y}:Q", title=y_title or y, format=",.3f"),
            alt.Tooltip(f"{value}:Q", title=value_title or value, format=value_format),
        ],
    ).properties(height=height)


def sequential_scale(vmin: float, vmax: float) -> tuple[list, list]:
    """Colours and domain for a field that keeps one sign.

    A sequential ramp has to run from "nearest zero" to "largest magnitude",
    and for an all-negative field, like theta, largest magnitude is the most
    NEGATIVE number. Mapping the raw range light-to-dark in that case paints
    the deepest theta palest and the near-zero corner darkest, which reads
    exactly backwards. So the ramp is reversed when the field sits below zero.
    """
    steps = list(colors()["seq"])
    if vmax <= 0:
        return steps[::-1], [float(vmin), 0.0]  # darkest at the most negative
    return steps, [min(0.0, float(vmin)), float(vmax)]


def plotly_colorscale(diverging: bool = False, steps: list | None = None) -> list:
    """A palette in the [[position, colour], ...] form Plotly wants."""
    c = colors()
    if diverging:
        return [[0.0, c["div_low"]], [0.5, c["div_mid"]], [1.0, c["div_high"]]]
    steps = steps or c["seq"]
    return [[i / (len(steps) - 1), col] for i, col in enumerate(steps)]


def surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, x_title: str, y_title: str,
    z_title: str, height: int = 560, diverging: bool = False,
    hover_format: str = ".4f", cap: float | None = None,
):
    """A 3-D surface of the same grid the flat map draws.

    Plotly rather than Altair, because Vega-Lite has no third dimension at all.
    This is the one escape hatch the rest of the app deliberately avoids.

    `cap` clamps the height so a single spike cannot flatten everything else
    into the floor. The colour scale follows whatever the height does, so the
    two never tell different stories.
    """
    import plotly.graph_objects as go  # imported lazily: only this view needs it

    c = colors()
    z = np.asarray(z, dtype=float)
    plotted = np.clip(z, -cap, cap) if cap is not None else z

    if diverging:
        lim = float(np.nanmax(np.abs(plotted))) or 1.0
        scale, cmin, cmax = plotly_colorscale(True), -lim, lim
    else:
        steps, (cmin, cmax) = sequential_scale(
            float(np.nanmin(plotted)), float(np.nanmax(plotted))
        )
        scale = plotly_colorscale(False, steps)

    fig = go.Figure(
        go.Surface(
            x=np.asarray(x, float), y=np.asarray(y, float), z=plotted,
            customdata=z,  # the TRUE value, so a capped peak still reports itself
            colorscale=scale, cmin=cmin, cmax=cmax,
            colorbar=dict(
                title=dict(text=z_title, side="right",
                           font=dict(color=c["ink_soft"], size=12)),
                thickness=14, outlinewidth=0,
                tickfont=dict(color=c["ink_soft"], size=11),
            ),
            hovertemplate=(f"{x_title}: %{{x:,.2f}}<br>{y_title}: %{{y:,.2f}}"
                           f"<br>{z_title}: %{{customdata:{hover_format}}}<extra></extra>"),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=False,
                                 width=1, highlightcolor=c["muted"])),
        )
    )

    axis = dict(
        gridcolor=c["grid"], zerolinecolor=c["axis"], showbackground=False,
        color=c["ink_soft"], tickfont=dict(size=10),
    )

    def titled(text):
        return dict(title=dict(text=text, font=dict(size=12)), **axis)

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["ink_soft"], size=12),
        scene=dict(
            xaxis=titled(x_title),
            yaxis=titled(y_title),
            zaxis=titled(z_title),
            # Pulled in close: the default camera distance leaves the surface
            # small in a lot of empty box.
            camera=dict(eye=dict(x=1.22, y=-1.32, z=0.72)),
            aspectratio=dict(x=1.25, y=1.25, z=0.85),
        ),
    )
    return fig


def show_surface(fig) -> None:
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False,
                            "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"]})


def grid_frame(x_edges: np.ndarray, y_edges: np.ndarray, z: np.ndarray,
               x_name: str, y_name: str, value_name: str) -> pd.DataFrame:
    """Turn a 2-D array into the binned-rect long frame Altair wants."""
    xs, ys = np.asarray(x_edges, float), np.asarray(y_edges, float)
    dx = np.gradient(xs) if len(xs) > 1 else np.array([1.0])
    dy = np.gradient(ys) if len(ys) > 1 else np.array([1.0])
    X, Y = np.meshgrid(xs, ys)
    DX, DY = np.meshgrid(dx, dy)
    return pd.DataFrame({
        x_name: (X - DX / 2).ravel(), f"{x_name}2": (X + DX / 2).ravel(),
        y_name: (Y - DY / 2).ravel(), f"{y_name}2": (Y + DY / 2).ravel(),
        value_name: np.asarray(z, float).ravel(),
    })


def histogram(values, *, bins: int = 40, height: int = 260, x_title: str = "value",
              highlight_zero: bool = True, mean_line: bool = True,
              highlight: float | None = None, highlight_label: str = "selected") -> alt.LayerChart:
    """Distribution of an outcome, with the zero line and the mean called out.

    `highlight` marks one observation inside the distribution, which is how a
    single inspected path is placed among all the others.
    """
    c = colors()
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    df = pd.DataFrame({
        "lo": edges[:-1], "hi": edges[1:], "count": counts,
        "mid": (edges[:-1] + edges[1:]) / 2,
        "base": np.zeros(len(counts)),  # explicit baseline, see below
    })
    # Ranged bars via x/x2 WITHOUT bin="binned": a binned encoding sharing a
    # layer with the plain-quantitative rule marks below silently renders
    # nothing at all, which is how this chart first shipped as an empty grid.
    bars = alt.Chart(df).mark_bar(color=c["series"][0], cornerRadius=2, opacity=0.9).encode(
        x=alt.X("lo:Q", title=x_title, axis=alt.Axis(**_AXIS),
                scale=alt.Scale(nice=False, domain=[float(edges[0]), float(edges[-1])])),
        x2="hi:Q",
        # y2 must be given explicitly. A bar that already carries x2 stops
        # extending to the zero baseline on its own and renders as a thin cap
        # floating at the count.
        y=alt.Y("count:Q", title="paths", axis=alt.Axis(**_AXIS)),
        y2="base:Q",
        tooltip=[alt.Tooltip("mid:Q", title=x_title, format=",.2f"),
                 alt.Tooltip("count:Q", title="paths")],
    )
    layers = [bars]
    if highlight_zero:
        layers.append(alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
            color=c["muted"], strokeWidth=1.5, strokeDash=[4, 3]).encode(x="x:Q"))
    if mean_line:
        layers.append(alt.Chart(pd.DataFrame({"x": [float(values.mean())]})).mark_rule(
            color=c["series"][1], strokeWidth=2).encode(
            x="x:Q", tooltip=[alt.Tooltip("x:Q", title="mean", format=",.3f")]))
    if highlight is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [float(highlight)]})).mark_rule(
            color=c["series"][2], strokeWidth=2.5).encode(
            x="x:Q", tooltip=[alt.Tooltip("x:Q", title=highlight_label, format=",.3f")]))
    return alt.layer(*layers).properties(height=height)


def show(chart) -> None:
    st.altair_chart(chart, width="stretch")


# --- page furniture -------------------------------------------------------


def keep(*keys: str) -> None:
    """Stop widget state evaporating when you navigate away and come back.

    Streamlit drops the state of any widget that did not render on a run, and a
    page you are not looking at does not render. Re-assigning a key to itself
    marks it as set through the state API, which exempts it. Call this at the
    TOP of a page, before the widgets are created, or Streamlit raises.
    """
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def page_header(title: str, lede: str) -> None:
    st.title(title)
    st.markdown(
        f"<p style='color:{colors()['ink_soft']};font-size:1.02rem;"
        f"margin-top:-0.5rem;max-width:60rem'>{lede}</p>",
        unsafe_allow_html=True,
    )


def takeaway(text: str, icon: str = "💡") -> None:
    """The one sentence the page exists to deliver."""
    st.info(text, icon=icon)


def caveat(text: str, icon: str = "⚠️") -> None:
    st.warning(text, icon=icon)


def metric_row(items: list[tuple[str, str, str | None]], help_texts=None,
               delta_colors=None) -> None:
    """A row of bordered metric cards: (label, value, delta).

    The delta slot here usually carries a unit or a note rather than a change,
    so it defaults to "off" - no green arrow implying something went up.
    """
    cols = st.columns(len(items), border=True)
    helps = help_texts or [None] * len(items)
    tints = delta_colors or ["off"] * len(items)
    for col, (label, value, delta), helptext, tint in zip(cols, items, helps, tints):
        col.metric(label, value, delta=delta, help=helptext, delta_color=tint)


def market_sidebar() -> Market:
    """The market every page shares. Drawn once per run, above the page's own controls."""
    with st.sidebar:
        st.markdown("### Market")
        spot = st.number_input("Spot", min_value=1.0, max_value=10_000.0, value=100.0,
                               step=1.0, key="mkt_spot",
                               help="Price of the underlying right now.")
        vol = st.slider("Implied volatility", 0.01, 1.50, 0.20, 0.01, key="mkt_vol",
                        format="%.2f",
                        help="Annualised. 0.20 means the market is pricing roughly "
                             "a 20% move over a year, or about 1.25% on a typical day.")
        with st.expander("Rates and dividends"):
            rate = st.slider("Risk-free rate", -0.02, 0.15, 0.04, 0.005, key="mkt_rate",
                             format="%.3f")
            div = st.slider("Dividend yield", 0.0, 0.10, 0.0, 0.005, key="mkt_div",
                            format="%.3f")
        st.caption(
            f"A {vol:.0%} vol implies a daily move of about "
            f"{vol / np.sqrt(252) * 100:.2f}%."
        )
    return Market(spot=spot, rate=rate, div_yield=div, vol=vol)


def current_market() -> Market:
    return st.session_state.get("market") or Market()


# --- formatting -----------------------------------------------------------


def money(x: float, dp: int = 2) -> str:
    return f"{x:,.{dp}f}"


def signed(x: float, dp: int = 4) -> str:
    return f"{x:+,.{dp}f}"


def pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


def unbounded(x, dp: int = 2) -> str:
    return "unlimited" if x is None else f"{x:,.{dp}f}"
