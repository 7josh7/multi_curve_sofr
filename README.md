# Collateral-Aware SOFR Curve and Risk Engine

This project builds a Python pricing and risk engine for collateral-aware SOFR products. It strips discount factors from SOFR futures, applies Hull-White convexity adjustments, prices fixed-vs-SOFR swaps, and produces scenario/risk diagnostics for rates and basis moves. It now also includes a synthetic Mercurio joint-model calibration layer for OIS/SOFR/LIBOR assumptions.

This project is the rates market-construction layer. It builds OIS discount and SOFR projection curves, validates repricing errors, and exports a market snapshot JSON that can be consumed by downstream valuation or xVA engines without importing this codebase.

Synthetic data is the default market snapshot so the full workflow runs out of the box; the CSV files under `data/market/` can be replaced with real snapshots later.

## Problem Statement

The repo targets a practical MVP for SOFR curve construction and collateral-aware valuation. The engine bootstraps a SOFR projection curve from 1m and 3m SOFR futures plus SOFR swaps, discounts cashflows on a separate OIS collateral curve, and reports repricing quality, PV01, key-rate DV01, and scenario outcomes.

The Mercurio extension adds the paper's three modeling pillars as a calibration layer:

- forward LIBORs use shifted-lognormal LMM volatility calibration from synthetic ATM caplet vols;
- Fed-fund/OIS and SOFR use the Hull-White one-factor convexity layer;
- SOFR-OIS basis remains deterministic, while LIBOR-OIS basis diagnostics follow Mercurio's multiplicative basis formula.

Core workflow:

```python
from src.bootstrap import build_full_curves
from src.libor_model import build_joint_model_calibration

result = build_full_curves(project_root=".")
print(result.swap_repricing)
print(result.diagnostics)

joint = build_joint_model_calibration(project_root=".")
print(joint.diagnostics)
```

To export the decoupled market snapshot:

```bash
python -m src.cli export-snapshot --output outputs/curves/sofr_market_snapshot.json
```

After installing the package, the equivalent script entry point is:

```bash
sofr-export-snapshot --output outputs/curves/sofr_market_snapshot.json
```

The exported file contains valuation date, OIS discount curve nodes, SOFR projection curve nodes, Hull-White model parameters, and curve-build diagnostics.

## Modeling Assumptions

- Discounting uses a synthetic OIS collateral curve loaded from `data/market/ois_curve.csv`.
- SOFR projection uses deterministic pseudo-discount factors stripped from SOFR futures and swaps.
- Hull-White convexity uses one factor with constant mean reversion `a` and constant volatility `sigma`.
- Business-day logic uses a weekend calendar plus `Following` and `Modified Following`.
- Day-count support includes `ACT/360`, `ACT/365F`, and `30/360`.
- Synthetic data is parameterized to be internally consistent, which keeps repricing and diagnostics transparent.

## Instrument Universe

- Historical SOFR fixings in `data/fixings/sofr_fixings.csv`
- 1m SOFR futures in `data/market/sofr_1m_futures.csv`
- 3m SOFR futures in `data/market/sofr_3m_futures.csv`
- Fixed-vs-SOFR swaps from `2Y` to `10Y` in `data/market/sofr_swaps.csv`
- OIS collateral zero-rate pillars in `data/market/ois_curve.csv`
- Synthetic LIBOR/caplet/basis calibration points in `data/market/libor_lmm_calibration.csv`

## Mathematical Formulas

The implementation follows a deterministic-basis plus one-factor Hull-White overlay:

- `B(t, T) = (1 - exp(-a (T - t))) / a`
- `A(t, T) = exp(0.5 * integral_t^T sigma^2 B(u, T)^2 du)`
- `P_s(0, T_j) = P_s(0, T_{j-1}) / ((1 + tau_j F_j^future) * exp(U_j))` for 3m futures stripping
- `DF_end = DF_start * exp(-delta * (future_rate - convexity_1m))` for 1m futures stripping
- `S = sum_j tau_j D_d(T_j) F_j^s / sum_j tau_j D_d(T_j)` for the collateral-aware par swap rate
- `dL_j = rho sigma_j sigma B(t,T_j)(L_j + alpha_j)dt + sigma_j(L_j + alpha_j)dW_j` for forward LIBOR under the OIS risk-neutral measure
- `B_j = (L_j - F_j) / (1 + tau_j F_j)` for Mercurio's multiplicative LIBOR-OIS basis

Interpolation is done on log discount factors to keep discount factors positive and stable between pillars.

## Bootstrap Sequence

1. Load conventions and the synthetic market snapshot.
2. Build the OIS discount curve from collateral zero-rate pillars.
3. Seed the SOFR projection curve with `DF(0) = 1`.
4. Strip monthly SOFR discount factors from 1m futures with Hull-White convexity adjustments.
5. Extend the curve with 3m SOFR futures using the closed-form `U_j` adjustment.
6. Solve remaining annual projection pillars recursively from SOFR swaps using OIS discounting.
7. Reprice futures and swaps, then generate diagnostics and risk tables.

## Calibration Approach

The default snapshot fixes:

- `a = 0.03`
- `sigma = 0.01`

Optional calibration is implemented through `calibrate_sigma_value(...)`, which minimizes zero-curve roughness after the futures/swap splice while keeping `a` fixed. This keeps the MVP deterministic and interview-friendly while still exposing a clean extension point for calibration experiments.

The joint-model calibration layer uses synthetic LIBOR inputs:

- `shifted_lmm_vol_from_atm(...)` implements Mercurio's shifted-lognormal caplet-vol calibration formula;
- `hw_sigma_from_basis_minimization(...)` implements the constant-sigma basis-volatility minimizer from Mercurio section 3.6;
- `build_joint_model_calibration(...)` combines the SOFR/OIS curve build with synthetic LIBOR forwards, caplet vols, shifts, and correlations.
- `pv_libor_swap_with_sofr_fallback(...)` and `pv_libor_sofr_basis_swap_to_libor_payer(...)` implement compact valuation helpers for Mercurio sections 9 and 10.

On the bundled synthetic data, the basis-implied Hull-White sigma is close to the curve sigma:

```text
curve sigma              = 0.0100
basis-implied HW sigma   ≈ 0.0111
synthetic LIBOR rows     = 20
```

## Validation Checks

The test suite covers:

- day-count fractions and schedule generation
- IMM date detection
- monotone positive discount factors
- zero-convexity identities when `sigma = 0`
- futures repricing after stripping
- near-zero PV for a par market swap
- shifted-LMM caplet-vol calibration identities
- Mercurio basis-volatility minimization diagnostics

If `pytest` is available, run:

```bash
python -m pytest -q
```

## Example Results

On the bundled synthetic snapshot:

- 1m/3m SOFR futures reprice essentially exactly after stripping
- swap repricing is exact for newly bootstrapped maturities and within `0.01 bp` for the `2Y` check maturity
- both discount and projection discount factors stay positive and monotone
- scenario analytics include parallel shifts, steepeners/flatteners, front-end stress, basis widening, funding-spread overlays, and a convexity-off comparison

The notebooks are organized as:

- `notebooks/00_project_demo.ipynb`
- `notebooks/01_data_check.ipynb`
- `notebooks/02_curve_bootstrap.ipynb`
- `notebooks/03_convexity_adjustment.ipynb`
- `notebooks/04_swap_pricing_risk.ipynb`

Start with `notebooks/00_project_demo.ipynb` for the single end-to-end showcase. The others are smaller focused walkthroughs. They are designed to save figures and tables into `outputs/`.

Bundled example artifacts:

- `outputs/curves/sofr_market_snapshot.json`
- `outputs/figures/discount_curves.png`
- `outputs/figures/convexity_adjustments.png`
- `outputs/figures/swap_pv_scenarios.png`
- `outputs/tables/futures_repricing.csv`
- `outputs/tables/swap_repricing.csv`
- `outputs/tables/scenario_table.csv`
- `outputs/tables/joint_model_calibration.csv`
- `outputs/reports/joint_model_diagnostics.json`
- `docs/mercurio_joint_model_alignment.tex`

## Limitations

- Holidays are approximated with a weekend-only calendar.
- The synthetic dataset avoids prompt-contract fixing complications.
- Hull-White volatility is constant; no time-dependent `sigma(t)` calibration is included.
- The SOFR-OIS basis is deterministic.
- The project now calibrates shifted-lognormal forward LIBOR dynamics, but does not yet simulate full LMM paths or price caplets/swaptions.
- LIBOR fallback and LIBOR-SOFR basis swap valuation are implemented as compact formula helpers, not as full trade objects with production schedule conventions.
- Swap floating coupons use a projection-curve ratio approximation rather than daily compounded realized SOFR.

## Future Extensions

- Replace the synthetic CSV snapshot with Bloomberg or desk-exported market data.
- Add prompt-contract handling that blends realized SOFR fixings with projected accruals.
- Add separate OIS and SOFR deterministic-basis stripping across all maturities.
- Add full Monte Carlo simulation of the shifted-lognormal LMM under the OIS measure.
- Promote LIBOR fallback and LIBOR-SOFR basis swap helpers into full trade objects with richer conventions.
- Extend the risk layer to basis swaps, caps/floors, and swaptions.
- Add a Streamlit or CLI front end for scenario runs and report export.
