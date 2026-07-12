# Collateral-Aware SOFR Curve & Risk Engine

A typed Python package for building separate OIS discount and SOFR projection
curves, pricing fixed-vs-SOFR swaps, and producing curve and scenario risk. The
repository ships a synthetic snapshot so the workflow is reproducible without
a market-data license.

This is a research and portfolio implementation, not a production trading or
valuation system. In particular, the floating leg uses projection-curve ratios
rather than full daily SOFR compounding, and the holiday calendar is
weekend-only.

## Results at a glance

For the bundled 2025-04-15 synthetic snapshot:

| Check | Fresh result |
|---|---:|
| SOFR futures repricing (12 contracts) | max error about **6e-12 bp** |
| Swap repricing (2Y-10Y) | average about **0.08 bp**, max about **0.73 bp** |
| QuantLib swap-pricing cross-check | max par-rate gap about **0.28 bp** |
| Discount/projection curve sanity | positive and monotone DFs |
| Smoothness regularization proxy | `sigma` approximately **0.001**, at the lower bound |

Exact in-sample repricing is a bootstrap identity, not independent model
validation. The QuantLib check reuses the exported curves and therefore tests
swap-pricing and convention alignment; it does not independently rebuild the
curves from raw instruments. A 10Y holdout example is included under
`outputs/tables/oos_swap_repricing.csv` and currently misses by roughly 5.67 bp.

## Install and run

Python 3.10-3.13 is supported.

```bash
python -m pip install -e .
python -m pytest -q                       # 55 tests; QuantLib test skips if absent
python -m sofr_curve_engine.cli export-snapshot \
  --output outputs/curves/sofr_market_snapshot.json
```

For development, use the shared compatibility constraints:

```bash
python -m pip install -c constraints.txt -e ".[dev]"
ruff check .
mypy
pytest -q -m "not quantlib"
```

The optional swap-pricing comparison is isolated from the core install:

```bash
python -m pip install -c constraints.txt -e ".[quantlib]"
pytest -q -m quantlib
python -m sofr_curve_engine.quantlib_bridge
```

Programmatic use:

```python
from sofr_curve_engine import build_full_curves

result = build_full_curves(project_root=".")
print(result.swap_repricing)
print(result.calibration_diagnostics.to_dict())
```

Start with `notebooks/00_project_demo.ipynb` for the end-to-end walkthrough.

## Model scope

- **Dual curves.** OIS zero pillars define the collateral discount curve. A
  separate SOFR pseudo-discount curve is stripped from 1m/3m futures and annual
  SOFR swaps under OIS discounting.
- **Interpolation.** Discount factors are interpolated log-linearly and the
  exported schema treats discount factors as the canonical representation.
- **Hull-White overlay.** One-factor constant-parameter formulas supply futures
  convexity adjustments and an optional swaption-surface calibration.
- **Smoothness regularization proxy.** When option data are not used, fixed
  mean reversion and curve smoothness select `sigma`. The bundled solution is
  pinned to the lower bound, so it is explicitly reported as a regularization
  proxy—not as market-implied volatility.
- **Joint-model extension.** A shifted-lognormal forward-LIBOR layer reports
  caplet-vol transforms, multiplicative basis diagnostics, and a basis-implied
  Hull-White sigma cross-check.
- **Risk.** Signed payer PV01, partitioned key-rate DV01, and curve scenarios are
  computed by central bump-and-revalue. PV01 is normalized to one basis point.

## Snapshot contract: schema 1.0

`sofr-export-snapshot` emits a validated JSON contract intended for downstream
valuation and xVA consumers. Version `1.0` preserves the existing curve-node
shape and adds explicit metadata:

- `schema_version: "1.0"`;
- `source` and `source_revision`;
- canonical units;
- `conventions.time_basis: "ACT/365F"`;
- `conventions.zero_rate_compounding: "continuous"`;
- `conventions.canonical_representation: "discount_factor"`;
- structured optimizer convergence, objective, parameter-bound, and boundary
  diagnostics under `model.calibration`.

Validation rejects non-finite values, unsupported schema/conventions, invalid
dates or times, non-positive discount factors, and zero rates that do not
reconcile to `-log(df) / time`.

## Repository layout

```text
src/sofr_curve_engine/   installable package: curves, models, pricing, risk, export
tests/                   unit, contract, risk, and optional QuantLib checks
data/                    synthetic market inputs and explicit conventions
notebooks/               demo plus focused walkthroughs
docs/                    learning guide
outputs/                 versioned example snapshots, tables, reports, and figures
.github/workflows/       Python matrix CI plus optional QuantLib job
```

The core architecture is:

```text
CSV/source -> validated MarketData -> OIS + SOFR curves -> pricing/risk -> schema 1.0 export
```

Regenerate every tracked example table, report, and figure with:

```bash
python scripts/regenerate_outputs.py
```

## Validation and controls

The suite covers date/day-count logic, input normalization, curve positivity,
futures and swap repricing, smoothness-proxy boundary reporting, optimizer
failure handling, Hull-White identities and round trips, LIBOR extension
identities, snapshot contract enforcement, PV01 sign/bump normalization,
key-rate reconciliation, and optional QuantLib alignment.

GitHub Actions runs core tests across Python 3.10-3.13. Ruff and mypy run in a
dedicated Python 3.10 job matching the minimum supported language target. A
separate job installs QuantLib so the external-library check cannot silently
skip in CI.

## Known limitations and next steps

- Synthetic data only; the repository does not yet include a versioned market
  data acquisition/generation pipeline.
- Weekend-only calendar and simplified schedules; no payment lag, observation
  shift, lockout, or holiday calendar.
- No realized-fixing split for prompt futures and no full daily compounded SOFR
  coupon implementation.
- OIS discount factors start from supplied zero pillars rather than an
  independent OIS instrument bootstrap.
- Constant-parameter Hull-White and synthetic swaption surface; no independent
  option-pricing benchmark is part of the core suite.
- Deterministic basis and diagnostic shifted-LMM layer; no full LMM Monte Carlo.
- The QuantLib bridge validates swap-pricing alignment using the engine's
  curves, not independent curve construction.

The natural next validation milestone is an independently sourced market
snapshot with exact schedules, daily SOFR compounding, known-fixing treatment,
and trusted price/Greek benchmarks.

## License

MIT. See `LICENSE`.
