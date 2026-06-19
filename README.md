# Collateral-Aware SOFR Curve & Risk Engine

A Python engine that bootstraps a dual-curve SOFR/OIS framework from futures and
swaps, prices collateral-aware fixed-vs-SOFR swaps, and produces curve, risk, and
scenario diagnostics — **independently validated against QuantLib**.

It strips a SOFR projection curve from 1m/3m SOFR futures (with a Hull-White
convexity adjustment) and SOFR swaps, discounts on a separate OIS collateral
curve, and exports a market-snapshot JSON that downstream valuation or xVA code
can consume without importing this package. Synthetic data ships by default, so
the full workflow runs out of the box.

## Results at a glance

Valuation date 2025-04-15, bundled synthetic snapshot (`a = 0.03`, σ auto-calibrated):

| What | Result |
|---|---|
| **SOFR futures repricing** (12 contracts, 1m + 3m) | max error **6e-12 bp** — exact to machine precision |
| **Swap repricing** (2Y–10Y) | avg **0.08 bp**, max **0.73 bp** |
| **Independent QuantLib check** (par rates, 2Y–10Y) | agree to within **0.28 bp** |
| **σ calibration** (no OIS-option data) | SOFR-curve-smoothness objective → σ ≈ **0.0010** |
| **Curve sanity** | discount & projection DFs positive and monotone |

### Validated against QuantLib

The same exported snapshot is repriced with QuantLib's own OIS machinery
(`ql.Sofr` + `ql.MakeOIS`, OIS-discounted), so agreement is a true third-party
check rather than the engine grading itself. Par-rate gaps across the curve:

| Tenor | Engine market rate | QuantLib fair rate | Diff (bp) |
|------|------:|------:|------:|
| 2Y | 4.9013% | 4.8985% | -0.28 |
| 5Y | 5.2591% | 5.2607% | +0.16 |
| 10Y | 5.6047% | 5.6057% | +0.10 |

Full table: [`outputs/tables/quantlib_comparison.csv`](outputs/tables/quantlib_comparison.csv).
The sub-bp residual is expected and documented: the engine's floating leg uses a
projection-curve ratio approximation, while QuantLib compounds SOFR daily. The
comparison also reports a DV01 ladder from ~$18.9k (2Y) to ~$77.0k (10Y) per bp
on $100mm notional.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                       # 38 tests
python -m src.cli export-snapshot --output outputs/curves/sofr_market_snapshot.json

# optional QuantLib cross-check
pip install -r requirements-optional.txt
python -m src.quantlib_bridge             # writes outputs/tables/quantlib_comparison.csv + figure
```

Programmatic use:

```python
from src.bootstrap import build_full_curves
result = build_full_curves(project_root=".")
print(result.swap_repricing)
print(result.diagnostics)
```

Start with `notebooks/00_project_demo.ipynb` for the end-to-end showcase.

## Methods

- **Dual-curve construction.** OIS discount curve from collateral zero pillars; a
  separate SOFR projection curve stripped from 1m/3m SOFR futures and swaps, with
  swaps solved recursively under OIS discounting. Interpolation is log-linear on
  discount factors to keep them positive and monotone.
- **Hull-White convexity.** One-factor, constant mean reversion `a`; futures use
  the closed-form constant-σ convexity adjustment
  (`B(t,T) = (1 - e^{-a(T-t)})/a`).
- **Smoothness σ-calibration.** With no OIS-option data, σ is chosen to maximize
  SOFR-projection-curve smoothness (the Mercurio no-option-data alternative) —
  here it pins σ to its lower bound, i.e. minimal convexity is preferred by the
  data.
- **Joint-model layer.** Shifted-lognormal LMM caplet-vol calibration for forward
  LIBOR, Hull-White for OIS/SOFR, and Mercurio's multiplicative LIBOR-OIS basis;
  the basis-implied σ cross-checks the curve σ.
- **Risk.** PV01, key-rate DV01, and a scenario engine: parallel shifts,
  2s10s steepener/flattener, front-end stress, basis widening, funding-spread
  overlays, and a convexity-off comparison.

Conventions (`data/metadata/conventions.yaml`): WEEKEND calendar, Modified
Following, ACT/360, annual SOFR swaps — matched to QuantLib for the cross-check.

## Repo layout

```
src/            curve build, HW model, pricers, risk, export, quantlib_bridge
tests/          38 tests incl. QuantLib alignment (skips if QuantLib absent)
data/           synthetic market snapshot (futures, swaps, OIS, fixings)
notebooks/      00_demo + focused walkthroughs
outputs/        curves, tables, figures, reports
docs/           Mercurio joint-model alignment note
```

## Validation

The suite covers day-count/schedule logic, IMM detection, monotone positive DFs,
zero-convexity identities at σ = 0, futures and swap repricing, shifted-LMM
caplet-vol identities, basis-vol minimization, and — when QuantLib is installed —
par-rate agreement to within 0.5 bp (`tests/test_quantlib_alignment.py`).

## Limitations & extensions

Synthetic, internally consistent data; weekend-only holiday calendar; constant-σ
specialization of Hull-White (time-dependent σ(t) is the natural extension); a
deterministic SOFR-OIS basis; and a projection-ratio float leg rather than full
daily SOFR compounding (the source of the sub-bp QuantLib gap). Natural next
steps: real market snapshots, exact-schedule swap construction to tighten the
QuantLib match, full LMM Monte Carlo, and extending risk to caps/floors and
swaptions.
