# Options Lab

A Streamlit app for building intuition about options: what the greeks actually
do, how a position's shape follows from its legs, and where the profit in a
delta-hedged book really comes from.

It is a teaching instrument, not a trading tool. Everything is computed live
from one Black-Scholes-Merton implementation, so the charts and the claims in
the text cannot drift apart.

## Running it

Needs Python 3.10 or newer. Use a virtual environment: it keeps these packages
out of your system Python, and it puts `pip` and `streamlit` on your PATH,
which is why the bare `pip` command does not exist on a fresh macOS.

```bash
cd options-lab
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. Press Ctrl-C to stop it, and `deactivate`
to leave the virtual environment. Next time, only the `source` line and
`streamlit run app.py` are needed.

**`command not found: python3`** means the macOS command line tools are not
installed. Run `xcode-select --install` and accept the dialog, or install
Python from python.org or with `brew install python`.

## The pages

| Page | What it is for |
|---|---|
| **Overview** | Where to start, the three ideas worth carrying away, and an honest list of what the model assumes. |
| **Price and greeks** | One option, every sensitivity, at several maturities at once. Each greek comes with the one sentence that explains its shape. |
| **Greek maps** | The same greeks drawn over spot and time together. The gamma cone narrowing into expiry is the picture the rest of the app keeps referring back to. |
| **Strategy builder** | Twelve presets and a free-form leg editor. Payoff at expiry against value today, breakevens, bounded and unbounded outcomes, and net greeks across the spot. |
| **Hedging lab** | Sell an option, hedge the delta, run the clock over hundreds of paths. Full P&L attribution and a rebalancing-frequency sweep. |
| **Implied vol and smile** | Prices inverted back into volatility, a parametric smile, and the two defensible deltas a smile gives you for the same option. |
| **Lessons** | Seven predict-then-reveal questions. Each answer is recomputed from your current market, not hardcoded. |

## The three things it is built to teach

**Time enters as a square root.** Value, vega and hedging error all scale with
the square root of time. Four times the maturity buys twice the expected
movement. Halving the time to expiry takes about 30% off an at-the-money
option, not half.

**Gamma and theta are a single trade.** Owning convexity costs rent every day,
and the rent is priced to equal the convexity. No position in the app is long
gamma and collecting theta, because the pricing equation forbids it. What you
can have is a view on whether the rent is set too high.

**A delta-hedged option is a bet on movement, not direction.** Hedge the delta
and the market's direction drops out. Sell a 30-vol option into a market that
moves at 20 and you profit on nearly every path, up or down. The hedging lab
demonstrates this and then shows the two things that spoil it: you cannot hedge
continuously, and every trade pays the spread.

## Layout

```
app.py                  navigation and the shared market sidebar
optlab/
  bs.py                 Black-Scholes-Merton pricing, greeks, implied vol
  strategies.py         legs, positions, payoffs, net greeks, presets
  hedge.py              path simulation, the hedging engine, P&L attribution
  smile.py              parametric vol smile and smile-adjusted delta
  ui.py                 theme, validated palette, chart builders, page furniture
views/                  one module per page
tests/                  the maths tests and the headless app tests
```

Every function in `optlab` is vectorised over numpy arrays, which is what lets
a page draw a greek across hundreds of spot levels and several maturities, or
run a thousand hedged paths, without a Python loop.

## Tests

```bash
pip install pytest && python -m pytest tests/ -q
```

Roughly 380 tests in two groups.

**The maths is checked against numerical derivatives.** Every analytic greek,
including vanna, volga and charm, is compared with a central finite difference
of the price across in-, at- and out-of-the-money cases at several maturities
and vols. A sign error or a misplaced discount factor cannot survive that.
Put-call parity, the expiry and zero-vol limits, and the implied-vol round trip
are checked separately.

The hedging engine is held to economic properties rather than to fixed numbers:
hedging at the vol the market delivers must break even, selling vol above it
must win on nearly every path, the P&L attribution must reconstruct the realised
number, and hedging error must fall like one over the square root of the
rebalance count.

**The app is tested headlessly** with `streamlit.testing.v1.AppTest`, which runs
each page with no browser and collects every exception. Each page is loaded,
then driven: every greek on the maps page, every preset in the builder, every
smile shape, the hedging lab with costs and a no-trade band, and all seven
lesson explanations revealed at once.

## What the model assumes

Black-Scholes-Merton with a lognormal spot and constant parameters, European
exercise, a continuous dividend yield, and no bid-ask on the option itself.
Real returns have fat tails and jumps; real volatility clusters and spikes; and
the hedging lab assumes you can always trade at the simulated price, which is
exactly the assumption that fails when it matters most. The Overview page says
all of this before you touch anything.

Nothing here is a backtest, a forecast, or advice about any actual trade.
