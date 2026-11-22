# Options Pricing Lab v1.0

A standalone Python research application for European vanilla options. Compare Black–Scholes–Merton with two neural pricers and a constrained mixture, measure their errors on held-out contracts, and simulate delta hedging under stress.

## Run locally

Python 3.10 or later is required. From the extracted `options_pricing_lab` directory in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m options_lab.server
```

The app opens at [http://127.0.0.1:8515](http://127.0.0.1:8515). Stop it with Ctrl+C. Pass `--port 8516` for a different port. The first pricing run fits all three models locally. On macOS and Linux, use `.venv/bin/python` for the same commands. The built-in examples and CSV uploads run offline after installation.

## v1.0 research report

Open **Research report → Generate report** for a reproducible benchmark. The first run can take several seconds while the models fit. The report combines pricing and seven Greeks for the reference 90-day call, model robustness across three training seeds, and six paired hedge stress scenario summaries.

The dashboard displays worst-seed call price MAE across five contract regimes and baseline and combined-stress hedge P&L. The JSON includes the contract inputs, training and held-out seeds, per-run call and put errors, dependency versions, and scenario settings. The reference pricing and hedge simulations use the first training seed. The path-level CSV remains available from the **Hedge stress** tab.

The benchmark uses synthetic Black–Scholes targets and simulated hedge paths. The source package includes the implementation and automated tests. Use `python -m options_lab.server --version` from the installed environment to confirm the release.

## Experiments

| View | What you can do | Download |
|---|---|---|
| Pricing | Change the contract, compare prices and seven Greeks across spot levels, and back out implied volatility from a supplied premium | Price and Greek spot sweep CSV |
| Model evaluation | Compare held-out price and Greek errors using MAE, RMSE, or the 95th percentile. Review strike and maturity checks, bound violations, mixture training loss, and timing | Evaluation JSON |
| Robustness | Compare fitted pricers across training seeds and five held-out contract regimes | Robustness JSON |
| Hedge simulation | Compare all four delta hedges across identical price paths. Vary rebalancing frequency and trading costs | Paired P&L CSV |
| Hedge stress | Stress realized volatility, spot jumps, trading costs, and hedge frequency individually and together | Scenario and paired path P&L CSV |
| Numerical methods | Compare a Cox–Ross–Rubinstein tree and two Monte Carlo samplers against Black–Scholes on a single European contract. Sweep tree steps and simulated payoffs to measure approximation error and runtime | Benchmark JSON |
| Volatility surface | Generate a synthetic chain, recover annualized implied volatility, and inspect strike slices across seven expiries | Surface CSV |
| Market chains | Analyze bid and ask quotes, visualize midpoint IV by strike, and compare four model premiums at a common reference volatility | Chain analysis CSV |
| Research report | Reproduce the reference pricing, robustness, and hedge stress benchmarks in one run | Research report JSON |

## v0.9 model robustness

Open **Robustness → Run robustness study**. The default compares training seeds 42, 43, and 44 using 512 synthetic contracts per regime, each priced as a call and put. All models and training seeds share the same test draw. The held-out seed controls a separate random stream from model training.

| Regime | What varies | Training range |
|---|---|---|
| In-domain | Broad forward moneyness and total volatility | Inside |
| Near expiry, near ATM | One to seven days to expiry and near the forward | Inside |
| Deep moneyness | Low and high forward moneyness | Outside |
| High total volatility | Total volatility above the training cap | Outside |
| Low total volatility | Total volatility below the training floor, near the forward | Outside |

The plot and table measure approximation error against Black–Scholes for prices and seven Greeks on synthetic contracts. The summary shows mean and worst error across the selected seeds, and the highest observed count of out-of-bounds call prices, negative gamma, or negative vega. The JSON report includes every seed and regime, training settings, and library versions. Seed variation reflects both changes in the training sample and changes in model initialization.

## v0.8 hedge stress testing

Open **Hedge stress → Run stress test** and choose a hedge model and scenario. The histogram compares the selected stress with the baseline using common bins. The table reports all six scenarios for the selected model. The CSV contains every model's terminal P&L and the terminal spot for each path and scenario. Match rows on the path number to compare outcomes from the same random draws.

| Scenario | Change from baseline |
|---|---|
| Baseline | Simulated spot volatility equals hedge volatility. Hedge at every path interval. |
| Volatility mismatch | Use the stressed volatility for the spot path. |
| Midpoint jump | Multiply spot by the specified jump factor from the midpoint onward. |
| Higher trading costs | Apply the stressed fee to entry, rebalancing, and liquidation. |
| Coarser rebalancing | Skip hedge dates according to the selected spacing. |
| Combined stress | Apply the stressed volatility, spot jump, costs, and hedge frequency together. |

Every scenario starts with the same Black–Scholes premium and the same fixed hedge volatility. The jump is a one-time proportional shock at the first grid point at or after halfway to expiry. The hedge sees the new spot when it next rebalances. Cash accrues across each elapsed interval, including any shorter final interval. P&L is per underlying share for one short European option with zero dividends. The tail mean averages the worst 5% of simulated paths, rounding the count up. These are synthetic scenarios with user-specified shocks. [Bosserhoff and Stadje on hedge model misspecification](https://epubs.siam.org/doi/abs/10.1137/22M149435X)

## v0.7 market option chains

Open **Market chains → Analyze chain** to load a labeled synthetic example with 54 quotes across three expiries. Select an expiry and call or put to compare midpoint IV and model premiums. The CSV export includes all expiries, both option types, quote timestamps, quote and IV status, the model inputs, and data provenance.

**CSV upload.** Upload UTF-8 CSV for a single underlying and provide its spot price and the valuation date in UTC. Use `expiry,strike,kind,bid,ask` as the column names. Expiry uses `YYYY-MM-DD` and kind is `call` or `put`. An optional `quote_time` column accepts ISO timestamps with an explicit time zone. Bid or ask may be blank. The form accepts up to 1,000 quotes and includes a downloadable example for a $100 underlying on January 15, 2026.

**Alpaca snapshot.** Launch the server from a shell with `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` set, then choose **Alpaca snapshot** in the app. The server requests read-only market data. Spot comes from the latest IEX trade. The chain covers standard-root contracts within 20% of that spot. Leave expiry blank to search the next 45 calendar days, or select a single expiry. The app fetches up to 1,000 quotes per run and labels partial retrievals. The Indicative feed uses modified option quotes, while OPRA access depends on your subscription. [Alpaca option-chain API](https://docs.alpaca.markets/us/reference/optionchain) · [Latest stock trade API](https://docs.alpaca.markets/us/reference/stocklatesttradesingle-1)

For an interactive PowerShell session, start in the project folder with the virtual environment already installed.

```powershell
$env:APCA_API_KEY_ID = Read-Host 'Alpaca key ID'
$secret = Read-Host 'Alpaca secret key' -AsSecureString
$env:APCA_API_SECRET_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
Remove-Variable secret
.\.venv\Scripts\python.exe -m options_lab.server
```

**Comparison method.** Midpoint IV inverts Black–Scholes from a positive, uncrossed bid and ask. Quotes after the valuation date, unsupported expiries, and premiums outside the European price bounds receive explicit IV statuses. Older quote dates remain visible alongside elapsed calendar days. Model premiums use the chosen reference volatility for every quote, with maturity calculated using ACT/365. Standardized U.S. equity options use American exercise, so the lab's European benchmark and implied volatility are model-based comparisons. [Options Industry Council exercise guide](https://www.optionseducation.org/referencelibrary/faq/options-exercise)

## v0.6 expanded Greeks

In **Pricing**, select any Greek to plot it across the same 101 spot prices for all four pricers. The spot sweep CSV contains each model's premium and seven Greeks. In **Model evaluation**, choose **Theta, rho, vanna, vomma** to compare errors against Black–Scholes on the same independent holdout used for price and the original Greeks.

| Greek | What it measures | Displayed units |
|---|---|---|
| Theta | Local premium change as calendar time passes with spot and other market inputs held fixed | Dollars per calendar day |
| Rho | Premium sensitivity to the annual risk-free rate | Dollars per percentage point of the rate |
| Vanna | Delta sensitivity to annual volatility | Delta per volatility point |
| Vomma | Vega sensitivity to annual volatility | Dollars per volatility point squared |

All dollar figures are per underlying share. For the default 90-day call, Black–Scholes theta is about `−$0.03274` per calendar day and rho is about `$0.12377` per rate point. These are local sensitivities at the selected contract inputs.

The built-in pricers compute these Greeks analytically. With `P` the premium, `T` years to expiry, and `vega` already scaled per volatility point, calendar-day theta is `[rP − (r − q)SΔ − 100σ × vega/(2T)]/365`. Rho is `T(SΔ − P)/100`. The neural models propagate the mixed and second volatility derivatives through the tanh layers. The constrained mixture differentiates each volatility clock twice and aggregates the weighted component derivatives. Tests independently bump time, rate, and volatility for both option types and check put–call parity.

## v0.5 implied-volatility surface

Open **Volatility surface** and click **Generate surface**. Select an expiry in the heatmap to inspect that curve and the corresponding strikes and premiums. The default surface recovers IV at all 77 sampled quotes. Recovered annualized volatility ranges from 19.41% to 38.53%, and the largest reconstruction error is below `1e-8` volatility points.

The grid uses 14 to 730 days to expiry and 11 strikes from 75% to 125% of each expiry's forward. The forward is `F = S × exp((r − q)T)`. Strike is shown both in dollars and as a percentage of forward.

The generator uses `x = (K/F − 1) / 0.25` and sets the annualized input IV in percent to:

```text
90-day ATM vol + skew × x + smile × x² + term slope × log₂(days / 90)
```

Use skew to tilt the curve and smile to change its curvature. Term slope controls the change in IV for each doubling of expiry days. All sampled volatility values must stay between 1% and 250%. This pointwise parameterization is designed for exploring the shape and testing premium inversion.

Each synthetic quote uses a Black–Scholes put below the forward or a call at or above it. Brent's method recovers IV using the premium and contract terms. A quote whose time value is at or below 32 times machine epsilon times the discounted forward is marked `low_time_value`. The heatmap and CSV leave its recovered IV blank. The export includes the full input and recovered grid, quote status, and the settings needed to reproduce it.

## v0.4 numerical benchmark

Open **Numerical methods** and click **Run benchmark**. The default contract is a 90-day at-the-money call with spot and strike at $100, 25% annual volatility, a 4% risk-free rate, and zero dividends. These are the results for seed 42, in USD per underlying share.

| Method | Resolution | Price | Absolute error | Standard error | Approximate 95% interval |
|---|---:|---:|---:|---:|---:|
| Black–Scholes | Analytic | 5.431281 | 0 | — | — |
| CRR tree | 8 steps | 5.279664 | 0.151618 | — | — |
| CRR tree | 1,024 steps | 5.430075 | 0.001206 | — | — |
| Plain Monte Carlo | 131,072 payoffs | 5.425252 | 0.006030 | 0.022403 | [5.381342, 5.469161] |
| Antithetic Monte Carlo | 131,072 payoffs | 5.420532 | 0.010749 | 0.016748 | [5.387707, 5.453357] |

The convergence plots use logarithmic resolution axes. Tree error can oscillate as the payoff kink moves relative to lattice nodes. Monte Carlo estimates for successive budgets share earlier draws, and the two samplers share one random stream at each matched budget. The realized error of either sampler can rise when the budget grows. Standard error estimates sampling uncertainty, while absolute error measures the distance to the known analytic price for this run.

- **Tree.** Each step uses `u = exp(σ√Δt)`, `d = 1/u`, and risk-neutral up probability `p = (exp((r − q)Δt) − d)/(u − d)`. Backward induction discounts the European terminal payoff. The report flags coarse resolutions where the risk-neutral probability falls outside the open interval `(0, 1)`.
- **Monte Carlo.** Draw terminal prices directly from the risk-neutral lognormal distribution and average discounted payoffs. Antithetic sampling pairs shocks `Z` and `−Z` and estimates standard error from independent pair averages. The approximate normal interval uses `estimate ± 1.96 × standard error` and is shown only when at least 20 independent observations have positive payoffs and the estimated variance is positive. Rare payoffs can be missed at finite budgets.
- **Timing.** The benchmark reports median elapsed time for five warmed single-contract calls per method and resolution. Monte Carlo timing includes random draws. Numerical benchmarks run independently of neural fitting.

## v0.3 results

The default run uses seed 42, 4,096 held-out contracts per option type, and 1,000 identical simulated paths per hedge. The hedge sells a 90-day at-the-money call with 25% volatility, a 4% rate, 30 hedge intervals, and 5 basis points of trading cost.

| Metric | Price-only | Price + delta | Constrained mixture |
|---|---:|---:|---:|
| Price MAE per share | $0.292 | $0.238 | $0.000072 |
| Delta MAE | 0.0437 | 0.0235 | 0.00000230 |
| Gamma MAE | 0.00733 | 0.00586 | 0.000000337 |
| Vega MAE per vol point | $0.0419 | $0.0384 | $0.00000295 |
| Price-bound violations | 627 | 312 | 0 |
| Largest price-bound violation per share | $1.99 | $0.79 | $0.00 within tolerance |
| Negative gamma rows | 481 | 288 | 0 |
| Negative vega rows | 326 | 397 | 0 |
| Terminal hedge P&L RMSE per share | $1.235 | $0.941 | $0.822 |
| Paired P&L difference RMSE vs. Black–Scholes | $0.888 | $0.446 | $0.000029 |

The Black–Scholes hedge had $0.822 terminal P&L RMSE. The constrained mixture passed all 7,200 strike-slope checks, 7,160 strike-convexity checks, and 7,059 fixed-forward maturity checks. Seeds 77 and 101 also had zero held-out price-bound, negative-gamma, and negative-vega violations for the mixture. The synthetic target and simulated asset dynamics are Black–Scholes. The mixture contains that model as a special case, which explains the near-exact fit. These results test recovery of the synthetic benchmark with structural constraints. Market validation remains open.

To reproduce the table, run **Model evaluation** at seed 42, then run **Hedge simulation** with its defaults. The JSON and CSV exports preserve the exact evaluation and path results.

## Methodology

- **Benchmark.** Vectorized Black–Scholes–Merton prices and analytic delta, gamma, and vega use annualized rates and volatility with ACT/365 year fractions. The contract is European with continuous dividend yield. Premiums and P&L are USD per underlying share. Vega is quoted per one percentage point of volatility.
- **Neural model.** An Adam-trained, two-hidden-layer tanh MLP learns the call price divided by the discounted strike. Inputs are log-forward moneyness `log(F/K)` and total volatility `sigma * sqrt(T)`. Put prices use parity. Neural Greeks differentiate the trained network by the chain rule. Exact deterministic prices handle zero time or volatility in the Python API.
- **Fine-tuning.** Initialize the second network with a copy of the price-only weights, then run 120 more epochs on the same training sample. The objective averages squared normalized-price error, plus 0.30 times squared error in delta divided by the dividend discount factor, plus 20 times squared price-bound excess in normalized price units. Delta labels are computed analytically. The fitted function and its Greeks come from the same output. The price-bound penalty is soft, so violations remain possible.
- **Constrained mixture.** Fit three lognormal components to the same normalized training prices and deltas. Each component starts at the same forward and uses a positive, ascending total-volatility clock with a linear term and nonnegative weighted tanh activations. Model-wide mixture probabilities are positive, sum to one, and are independent of spot, strike, maturity, and volatility. The deterministic optimizer is L-BFGS-B, with analytic parameter gradients. See the derivation below.
- **Held-out evaluation.** Training and evaluation use separate seeded draws. The holdout deliberately includes extra near-the-money, low-total-volatility cases. MAE values are in each metric's native units. The diagnostics count sampled no-arbitrage price-bound violations and negative gamma or vega. All three fitted models are evaluated against the same contracts. Gamma and vega signs are diagnostics. The grid report also checks strike slopes, strike convexity, and normalized maturity ordering.
- **Timing.** Each engine prices a warmed batch of one or 2,048 contracts 21 times with numerical libraries limited to one thread. The report gives median wall time for each entire pricing call. It excludes model fitting.
- **Volatility inversion.** The Pricing tab solves for the Black–Scholes volatility that matches a supplied premium. The Volatility surface tab applies the solver to every recoverable premium in a generated chain.
- **Hedging.** Each simulated path shorts one option at the same Black–Scholes premium. A self-financing stock and cash portfolio targets each engine's delta at fixed intervals. Geometric Brownian motion has risk-neutral drift, constant volatility, and zero dividends. Cash accrues at the risk-free rate. Proportional trading costs apply to entry, rebalancing, and liquidation. P&L subtracts the terminal option payoff.
- **Scope.** Benchmarks use synthetic data from the same pricing model the neural network learns. The evaluation measures approximation quality. Results outside the fitted input range are extrapolations, and simulated performance depends on the chosen model and settings.

## Why the mixture satisfies the constraints

Let `w = σ√T` and `x = log(F/K)`. The model computes `Σ pⱼ BS(x, hⱼ(w))` in units of discounted strike. The weights `pⱼ` sum to one. Each clock `hⱼ(w) = aⱼ w + Σ bᵢⱼ sᵢ tanh(w/sᵢ)` has `aⱼ > 0`, `bᵢⱼ ≥ 0`, and hence a positive derivative for nonnegative `w`. The component forwards are identical.

- Each component is an expected discounted payoff of a positive terminal price with mean equal to the forward. Averaging preserves European price bounds, put–call parity, strike monotonicity, and strike convexity.
- At fixed contract inputs, gamma is the probability-weighted sum of nonnegative component gammas. Vega applies the chain rule and multiplies each nonnegative component vega by the increasing clock's slope.
- At fixed forward and strike, the normalized option value is nondecreasing as `w` grows. Equivalently, the component terminal prices can be coupled as time-changed exponential Brownian martingales. For calendar comparisons in the app, the grid fixes zero carry, uses a fixed forward, and increases `w`. These guarantees apply to the model-defined family. Independently changing volatility or calibrating new parameters strike by strike or maturity by maturity changes that family.

## Code layout

- `pricing.py` implements the formula, Greeks, and implied-volatility inversion.
- `neural.py` fits the MLP and evaluates its price and sensitivities.
- `differential.py` fine-tunes the baseline with the joint objective.
- `mixture.py` fits and evaluates the constrained mixture and its Greeks.
- `numerical.py` implements the CRR tree, Monte Carlo estimators, and convergence sweeps.
- `surface.py` generates the synthetic option chain and recovers implied volatility from its premiums.
- `sensitivities.py` combines model derivatives for theta, rho, vanna, and vomma.
- `market.py` validates option quotes and compares their midpoints with the local pricers.
- `alpaca_data.py` reads and normalizes Alpaca option and underlying snapshots.
- `experiments.py` runs independent evaluation, timing, and hedging.
- `robustness.py` builds matched test regimes and compares fitted models across training seeds.
- `server.py` and `web/` provide the local dashboard.
- `tests/` covers reference prices, derivative consistency, numerical estimators, sampling diagnostics, and paired hedge accounting.

## References

Martin Haugh, [The Black–Scholes Model](https://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf), Foundations of Financial Engineering, Columbia University.

Brian Huge and Antoine Savine, [Differential Machine Learning](https://arxiv.org/abs/2005.02347), 2020.

Damiano Brigo, Fabio Mercurio, and Francesco Rapisarda, [Lognormal-mixture dynamics and calibration to market volatility smiles](https://wwwf.imperial.ac.uk/~dbrigo/lognsmile.pdf).

John C. Cox, Stephen A. Ross, and Mark Rubinstein, [Option Pricing: A Simplified Approach](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/36891/files/2017/07/CRR79-1yy8av8.pdf), 1979.

Phelim Boyle, Mark Broadie, and Paul Glasserman, [Monte Carlo Methods for Security Pricing](https://business.columbia.edu/sites/default/files-efs/pubfiles/4336/monte_carlo_methods_security_pricing.pdf), 2001.

Cboe, [Cboe S&P 500 Smile Index fact sheet](https://cdn.cboe.com/resources/indices/documents/cboe-smile-index-fact-sheet.pdf).

CME Group, [Volatility Term Structure Tool](https://www.cmegroup.com/tools-information/quikstrike/volatility-term-structure.html?pid=40).

ETH Zürich, [Mathematical Foundations for Finance, section 7.3](https://metaphor.ethz.ch/x/2021/hs/401-3913-01L/auth/lethz/notes/mff_script/lecture_notes.pdf).
