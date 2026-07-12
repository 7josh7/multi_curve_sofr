# Hiring Manager Review: Collateral-Aware SOFR Curve & Risk Engine

Review date: 2026-07-12
Reviewed revision: `3f84cee` on `master`
Review basis: source, data, outputs, tests, notebooks, packaging, public presentation, and repository history. This is a hiring review, not formal model validation.

## Implementation update (2026-07-12)

The requested quality pass has resolved the principal engineering findings:

- The package now uses the standard `src/sofr_curve_engine` layout and builds as
  version 0.2.0; imports, CLI, notebooks, and installed-package use were updated.
- Smoothness-selected sigma is explicitly a regularization proxy, and both
  smoothness and Hull-White optimizers expose validated convergence, objective,
  parameter-bound, and boundary-hit diagnostics.
- The producer emits and validates snapshot schema 1.0 with canonical discount
  factors, explicit units/conventions, and DF/zero-rate reconciliation; the CVA
  sibling consumes it in strict mode.
- PV01/key-rate definitions were corrected and regression-tested; the QuantLib
  sign label and a 3m convexity-sign inconsistency were fixed.
- All tracked `desktop.ini` files were removed; CI, Ruff, mypy, build checks,
  constraints, MIT license, and deterministic output regeneration were added.
- The expanded suite passes **55 tests**; wheel/sdist, clean install, CLI export,
  Ruff, and mypy checks pass.

The remaining limitations are model-scope items rather than hidden defects:
synthetic data, supplied OIS pillars, simplified calendars/schedules, no daily
SOFR compounding, constant-parameter Hull-White, a diagnostic LMM extension,
and a QuantLib comparison that uses engine-produced curves.

The remainder of this file preserves the original review and remediation
roadmap as a historical baseline. Findings identified as resolved above should
not be read as current defects.

## Original executive verdict (pre-remediation baseline)

**Advance to a technical interview for a junior-to-associate rates quant developer or quantitative software engineer.** This is the strongest of the three reviewed pinned projects. It has a coherent problem statement, a results-first README, a working dual-curve workflow, explicit market-data schemas, useful diagnostics, out-of-sample repricing checks, and an executable comparison against QuantLib. The snapshot interface also works as the upstream market layer for the companion CVA repository.

The repository does not yet demonstrate senior production model ownership. Its market data are synthetic, conventions are intentionally simplified, the default smoothness “calibration” selects the volatility lower bound, and the numerical/engineering controls are not release-grade. The candidate should receive credit for presenting these limitations honestly, but should be able to distinguish calibration, regularization, internal consistency, and independent validation in interview.

| Dimension | Assessment |
|---|---|
| Likely level | Strong junior / early associate |
| Best fit | Rates quant development, curve/risk analytics, model implementation under review |
| Strongest evidence | QuantLib par-rate alignment, 39 passing tests, coherent snapshot boundary |
| Main reservation | Synthetic/self-consistent evidence and a boundary-selected model parameter |
| Overall portfolio score | **7.7 / 10** |

## Repository map

The principal workflow is:

**CSV/YAML inputs -> OIS discount curve -> SOFR projection bootstrap -> Hull-White convexity adjustment -> swap repricing and diagnostics -> PV/risk/scenarios -> portable JSON snapshot -> QuantLib and xVA consumers**

| Path | Hiring-manager interpretation |
|---|---|
| `README.md` | Excellent first-page structure and concise numerical claims; a few claims/counts are stale or need sharper terminology. |
| `data/market/` | Synthetic OIS, 1m/3m futures, SOFR swaps, swaption vols, and LIBOR/LMM calibration inputs. Good reproducibility, limited external validity. |
| `data/metadata/conventions.yaml` | Central conventions/model/risk configuration. Defaults to weekend-only calendar and smoothness-selected sigma. |
| `src/data_input.py` | Typed data-source protocols, CSV normalization, schema errors, and injection points. A positive design signal. |
| `src/bootstrap.py` | Discount/projection builds, convexity calibration modes, repricing, holdout diagnostics, and orchestration. At 500+ lines it is the main concentration of model risk. |
| `src/curves.py`, `src/interpolation.py` | Curve abstractions and log-discount-factor interpolation. |
| `src/hw_model.py` | Hull-White convexity, bond options, Jamshidian-style swaption pricing, implied normal vol, and calibration. |
| `src/libor_model.py`, `src/libor_pricers.py` | Shifted-LMM/basis research layer and fallback/basis-swap examples. Interesting breadth, less independently validated than the SOFR core. |
| `src/pricers.py`, `src/risk.py` | Swap pricing, PV01/key-rate DV01, and named scenarios. |
| `src/export.py`, `src/cli.py` | Versionless but useful portable snapshot and CLI. The sibling CVA engine successfully consumed the committed snapshot. |
| `src/quantlib_bridge.py` | Independent OIS repricing comparison and result chart. This materially improves the hiring signal. |
| `tests/` | Eleven test modules; 39 tests passed in review, including QuantLib alignment. |
| `notebooks/`, `docs/` | Six explanatory notebooks, but all code cells are unexecuted and contain no stored outputs. |
| `outputs/` | Curated tables, JSON, and figures support the README claims, though provenance/version metadata could be stronger. |

## Scorecard

| Category | Score | Assessment |
|---|---:|---|
| Rates-domain modeling | 8.0 / 10 | Strong portfolio breadth across dual curves, convexity, swaps, risk, basis, and optional swaption calibration. Simplified schedules/basis and synthetic inputs constrain the claim. |
| Correctness and model risk | 7.0 / 10 | Good identities, repricing checks, positive/monotone diagnostics, holdout metrics, and QuantLib comparison. Boundary calibration and incomplete optimizer/error diagnostics remain. |
| Software design | 7.2 / 10 | Clear layers, dataclasses/protocols, configuration, CLI, and export boundary. The import package named `src` and a large bootstrap module reduce polish. |
| Tests and validation | 8.0 / 10 | 39 tests passed, including a genuine third-party check. More adverse-input, sensitivity, benchmark, and full workflow tests are needed. |
| Documentation and reproducibility | 8.2 / 10 | Results-first README and shipped synthetic data make the workflow easy to understand and run. No lockfile; notebooks are not rendered. |
| Hygiene and presentation | 5.8 / 10 | No CI, license, release, About metadata, or benchmark provenance; 13 `desktop.ini` files remain tracked. |

## Strongest hiring signals

### 1. Independent comparison is executable

`tests/test_quantlib_alignment.py:18-22` exports the engine snapshot, reprices the swap set through QuantLib, and enforces a 0.5 bp maximum gap. In this environment QuantLib was installed, so the test ran rather than skipped. The full suite passed 39/39, and the README's reported worst gap of about 0.28 bp is consistent with the committed table.

This is stronger evidence than tests that only compare two functions written in the same repository. The README also identifies the residual's likely source: projection-ratio floating-leg valuation versus exact compounded overnight coupons.

### 2. The cross-repository interface is well chosen

`src/export.py` emits curves, model values, diagnostics, and metadata without requiring downstream code to import this package. The companion CVA engine consumed `outputs/curves/sofr_market_snapshot.json` successfully in a 1,000-path run. That demonstrates architectural judgment and a meaningful portfolio narrative.

### 3. Validation goes beyond in-sample exact fit

`swap_holdout_repricing_table` and `diagnostics_summary` in `src/bootstrap.py:397-490` expose holdout error and quote-shock stability. Curve positivity/monotonicity and futures/swap repricing are tested. This is the right direction for quantitative work: show failure-sensitive diagnostics rather than one headline price.

### 4. Data ingestion is more mature than a notebook demo

`src/data_input.py` defines protocols, typed errors, validation, and injectable sources. Tests verify duck-typed sources and load counts. A hiring manager can see how the candidate would adapt the prototype to another data source without rewriting the model.

### 5. Limitations are disclosed

The README states that the data are synthetic, the calendar is weekend-only, sigma is constant, basis is deterministic, and the floating leg is approximate. The default sigma landing on its lower bound is disclosed rather than hidden. This improves trust.

## Detailed room for improvement

## Critical credibility issue

### C1. Treat lower-bound smoothness selection as regularization, not market calibration

**Evidence:** `data/metadata/conventions.yaml:18-21` sets sigma bounds `[0.001, 0.05]` and default method `sofr_curve_smoothness`. `calibrate_sigma_from_sofr_curve_smoothness` in `src/bootstrap.py:212-231` minimizes curve roughness, and the README reports sigma approximately 0.0010—the lower bound.

**Why a hiring manager cares:** an optimizer pinned to a bound is telling the reviewer that these inputs do not identify a positive volatility through the chosen objective. Calling it an automatic calibration can overstate what the market evidence establishes. Smoothness is a regularization preference, not an option-market observation.

**Remediation:** label the value `smoothness_selected_sigma` or `convexity_proxy`; record optimizer success, objective, gradient/tolerance, bound status, and sensitivity to bounds. Show results for alternative bounds and quote perturbations. Prefer swaption/cap data when claiming Hull-White calibration.

**Acceptance criteria:** the report explicitly says whether the optimum is interior; bound changes and 1 bp quote perturbations produce a documented stability table; a synthetic option surface recovers known parameters; no output calls a boundary proxy independently calibrated market volatility.

## High priority

### H1. Package name and install surface are unprofessional

`pyproject.toml` packages the top-level module as `src`, and examples import `from src.bootstrap ...`. `src` should be a layout directory, not the public domain namespace.

**Fix:** move code to `src/sofr_curve_engine/` (or another unique name), update imports/entry points/tests, expose a small public API, and verify wheel/sdist installation in a clean environment.

**Acceptance:** `pip install .` followed by an import and CLI smoke test succeeds from outside the repository; no consumer imports a module named `src`.

### H2. Optimizer outcomes can be silently accepted

`src/hw_model.py:231-238` returns `result.x` without checking `result.success`, objective finiteness, termination reason, or boundary status. The smoothness objective and swaption calibration also convert broad exceptions into large penalties (`src/bootstrap.py:184-187`, `src/hw_model.py:220-228`), which can obscure the underlying failure.

**Fix:** return a typed calibration result with parameters, success, status/message, objective, iterations, bounds, and residuals. Raise a domain-specific calibration error on failure and preserve the causal exception in diagnostics.

**Acceptance:** tests force non-convergence, empty quotes, invalid bounds, and pricing failures; none returns plausible parameters without a failure flag.

### H3. Core validation is still narrow and synthetic

QuantLib alignment is valuable, but it uses the engine's exported curves and the same synthetic swap set. Many tests establish identities or broad bounds rather than tolerance against independent market fixtures. The LIBOR/LMM layer has no comparable third-party benchmark.

**Fix:** add date-exact QuantLib instrument builds across more maturities and irregular schedules; compare PV, par rates, DV01, forwards, discount factors, and curve-node sensitivities. Add timestamped public/example market data with units and provenance, plus adverse/negative-rate and non-monotone input cases.

**Acceptance:** a versioned benchmark file records source, as-of date, conventions, expected values, tolerances, and environment; CI fails on drift.

### H4. Risk results need reconciliation tests

`src/risk.py` offers PV01, key-rate DV01, and scenarios, but the suite has no dedicated `test_risk.py`. Hiring evidence should demonstrate sign conventions, bump size, central versus one-sided differences, curve rebuild behavior, and reconciliation of key-rate buckets to parallel risk.

**Fix/acceptance:** test payer/receiver signs, bump linearity, central differences, key-rate sum versus parallel PV01, scenario symmetry for small shocks, and quote-bump/rebootstrap risk versus direct zero-node bumps. Document units (`currency per bp`) in every table.

### H5. Snapshot contract needs versioning and round-trip validation

The snapshot is useful but contains no explicit schema version. Downstream interpretation of curve fields, compounding, day count, index, currency, and model-parameter precedence is only implicit.

**Fix:** publish a JSON schema with `schema_version`, units, currency/index, compounding, day count, calendar, source revision, as-of timestamp, calibration method/status, and canonical curve representation. Add producer/consumer contract tests with the CVA repository.

**Acceptance:** an incompatible schema fails early; a round trip preserves node DFs and benchmark prices exactly within tolerance.

## Medium priority

### M1. Exact overnight cashflow conventions remain the largest modeling extension

The core floating leg uses projection-curve ratios and a weekend-only calendar. This explains the sub-bp QuantLib gap, but it also limits the library's use as a convention-accurate SOFR engine.

Implement date-exact observation periods, daily compounding, lookback/lockout/payment lag, holiday calendars, stubs, fixing history, and in-period accrued coupons. Reconcile every supported convention to QuantLib or hand-calculated fixtures.

### M2. The QuantLib chart has a sign-label defect

`src/quantlib_bridge.py:69-74` computes `(QuantLib fair rate - engine market rate) * 1e4`, while line 99 labels the chart “Engine - QuantLib.” Correct the label or reverse the calculation and make the CSV column name explicit.

### M3. Repository hygiene weakens an otherwise polished first impression

Thirteen `desktop.ini` files are tracked, including at repository root and throughout `data/`, `src/`, `tests/`, `notebooks/`, and `outputs/`. The `.gitignore` now excludes them, but already tracked files remain visible on GitHub.

Remove them from Git history's current tree, keep the ignore rule, and add a hygiene check. Also add CI, a license, Python-version matrix, lint/type/coverage configuration, build verification, dependency constraints/lock, and a tagged release.

### M4. README and artifacts have small provenance gaps

The README claims 38 tests, while 39 passed in this review. Generated tables/figures do not clearly embed commit/config/dependency provenance. The public GitHub About panel has no description, topics, website, or release.

Update counts automatically or avoid hard-coded counts. Add a manifest beside golden outputs containing git SHA, input hashes, config, Python/dependency versions, and generation command. Populate repository metadata and publish a cleaned `v0.1.0`.

### M5. The notebooks do not demonstrate successful execution on GitHub

All six notebooks contain zero executed code cells and zero stored outputs. That keeps diffs small, but a hiring manager clicking the recommended demo cannot see a result without setting up the environment.

Commit one sanitized executed showcase or a rendered HTML/PDF, keep focused development notebooks output-free, and smoke-test notebook execution in CI.

### M6. `bootstrap.py` and the research layers need clearer boundaries

`src/bootstrap.py` combines orchestration, calibration choice, curve building, repricing, holdout studies, and diagnostics. The joint LIBOR/LMM layer expands the claim beyond the most validated SOFR core.

Split calibration, builders, validation, and reporting into cohesive modules. Mark research/experimental APIs explicitly and avoid presenting them at the same maturity level as the QuantLib-checked SOFR path.

## Quantitative model-risk assessment

| Area | Positive evidence | Remaining risk | Best next validation |
|---|---|---|---|
| OIS/SOFR bootstrap | In-sample repricing, positive/monotone DFs, holdout metrics | Synthetic quotes; simplified schedules/interpolation | Independent node/PV/risk benchmarks on dated inputs |
| Futures convexity | Zero-sigma identities and HW functions tested | Default sigma is bound-selected; parameter identification weak | Option-market calibration and perturbation study |
| Swap pricing | Par/PV tests and QuantLib sub-0.5 bp check | Projection-ratio coupon rather than exact daily compounding | Date-exact SOFR cashflow reconciliation |
| Interpolation | Log-DF approach supports positive DFs | Monotonicity alone does not guarantee sensible forwards under every input | Forward/no-arbitrage diagnostics and adverse fixtures |
| Risk | PV01/key-rate/scenarios implemented | Sparse direct testing and unclear rebuild/reconciliation semantics | Bucket-to-parallel reconciliation and QuantLib DV01 |
| Joint LIBOR/LMM basis | Interesting Mercurio-aligned research layer | Synthetic calibration and limited independent reference evidence | Published formula cases or second implementation |
| Export contract | Successful CVA consumer run | No schema/version/canonical representation | Producer-consumer contract CI |

## Portfolio-presentation critique

### What works

- The title explains the problem rather than using a generic class-project name.
- The first screen gives specific errors, tenors, calibration outcome, and validation.
- Quickstart and programmatic usage are short.
- Architecture, methods, conventions, limitations, and extensions are visible without reading code.
- Committed output tables make the claims inspectable.

### What to improve

- Add a concise GitHub About description and topics such as `sofr`, `yield-curves`, `quantitative-finance`, `hull-white`, and `fixed-income`.
- Embed one curve/validation figure in the README and link to a rendered demo.
- Replace the generic `src` import in every example.
- Correct the test count and QuantLib sign label.
- Remove machine debris and add CI/license/release badges only after those controls exist.
- Add a short “What I personally designed and validated” section, especially because the latest PR is titled `[codex]`.

## Recommended interview questions

1. Derive the one- and three-month futures convexity adjustment and identify the measure/assumptions.
2. Why does the smoothness objective choose the lower sigma bound? What information would make sigma identifiable?
3. Is curve smoothness calibration, regularization, or model calibration? Defend the terminology.
4. Explain dual-curve swap valuation and why projection and discount curves enter different legs.
5. Reconcile the projection-ratio float leg with daily compounded SOFR and explain the 0.28 bp QuantLib gap.
6. What does monotone discount factor guarantee, and what can still go wrong with forward rates?
7. How should key-rate DV01 be generated—zero-node bump, quote bump/rebootstrap, or both—and how should buckets reconcile?
8. Ask the candidate to add optimizer failure handling and a regression test live.
9. Ask the candidate to version the snapshot schema without breaking the CVA consumer.
10. Which components were personally authored or materially reviewed after AI assistance, and what generated suggestion was rejected or corrected?

## Prioritized roadmap

### One day

1. Remove all tracked `desktop.ini` files.
2. Correct the 39-test count and QuantLib difference label.
3. Add optimizer success/bound diagnostics to output, with failing tests.
4. Add license text, GitHub About metadata, and an authorship/AI-assistance note.
5. Render or execute one small demo artifact.

### One week

1. Rename the public package from `src` and verify wheel/sdist installation.
2. Add GitHub Actions for install, 39 tests with optional QuantLib, lint, types, coverage, and package build.
3. Version the snapshot schema and add cross-repository contract tests.
4. Add dedicated risk reconciliation and adverse-input tests.
5. Split bootstrap calibration/build/validation/reporting responsibilities.

### One month

1. Implement convention-accurate daily compounded SOFR cashflows and calendars.
2. Add dated input provenance and independent benchmark suites for PV, curves, and risk.
3. Calibrate Hull-White to a documented option surface with uncertainty/sensitivity reporting.
4. Establish a model-validation document distinguishing validated, benchmarked, experimental, and illustrative features.
5. Publish a reproducible `v0.1.0` release with a rendered case study connecting curve build to the CVA consumer.

## Checks performed

| Check | Result |
|---|---|
| Clone and revision | Clean clone of `master` at `3f84cee` |
| Repository inventory | 74 tracked files; 30 Python files; 11 test files; about 2,627 Python lines |
| `python -m pytest -q` | **39 passed in 1.90s**; no skip reported, so QuantLib alignment executed |
| Snapshot CLI | Succeeded and wrote a 5,379-byte JSON snapshot to a temporary path |
| Companion integration | CVA repository consumed the committed snapshot successfully in its 1,000-path case |
| Notebook inspection | Six notebooks; all code cells unexecuted; no stored outputs |
| History | Nine commits; latest merge is PR #1, `[codex] Add SOFR smoothness calibration` |
| Hygiene inventory | 13 tracked `desktop.ini` files; no CI workflow, standalone license, lockfile, or release |
| Public metadata | No GitHub description, website, topics, or release at review time |

## Review limitations

- No dependency installation or source modification was performed.
- Passing tests validate the current code and fixtures; they are not formal independent model validation.
- Market inputs are synthetic, so fit quality does not establish performance on live noisy markets.
- The review did not run a historical production dataset, performance/load test, or formal coverage/type/lint suite.
- Public repository/history evidence cannot by itself establish personal authorship; interview discussion is required.

## Bottom line

This repository should help the candidate get an interview. It is focused, runnable, numerically literate, candid about limitations, and independently checked in a meaningful way. The best next move is not to add another model layer. It is to turn the current prototype into a trustworthy release: honest calibration language, optimizer diagnostics, a real package namespace, clean repository hygiene, CI, versioned data contracts, convention-accurate SOFR cashflows, and deeper independent risk benchmarks.
