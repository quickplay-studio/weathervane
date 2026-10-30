# Weathervane lab: every researched repo, integrated

Companion to DESIGN.md. The lab is an isolated, optional layer: its own requirements files, its own weekly
workflow, and adapters that catch every import or runtime failure and record it. Nothing in the lab can
break `signals` or `research`. Lab outputs are research context only: they never become trading rules
(rules must stay Pine-portable, DESIGN.md section 5).

## 1. Repo map (every item the user supplied)

| Repo | Where it lives | What it does in Weathervane |
|---|---|---|
| karpathy/autoresearch | core `researcher.py` + `program.md` | The nightly keep-or-revert experiment loop (pattern, not a dependency) |
| TauricResearch/TradingAgents | core `companies.py` | Daily company research reports |
| OpenBB-finance/OpenBB (ODP) | lab `adapters/openbb_odp.py` + core data fallback | CFTC COT positioning, macro series; fallback OHLCV source if yfinance fails |
| hmmlearn | core `regime.py` | HMM regime labels (gate + chart shading) |
| pmorissette/ffn | core `reports.py` + lab | Performance stats; cross-check of core metrics |
| enthought/pyql | lab legacy | QuantLib pricing of American ETF options (built against apt `libquantlib0-dev`); fallback `QuantLib` wheel if the build fails |
| vollib/vollib | lab modern | Implied vol and Greeks for every option in the fetched chains |
| jsmidt/QuantPy | lab legacy | Mean-variance and inverse-vol weights for sleeves (research suggestion only) |
| alpha-miner/Finance-Python | lab legacy | Independent rolling operators to cross-check core indicators (SMA, stdev, rolling max/min) |
| GriffinAustin/pynance | lab legacy | Independent data analytics cross-check (returns, volatility) |
| ynouri/pysabr | lab modern | SABR smile fit per expiry: skew (rho) and vol-of-vol (nu) |
| domokane/FinancePy | lab modern | Equity and FX option valuation and Greeks cross-check |
| goldmansachs/gs-quant | lab modern | Offline `gs_quant.timeseries` functions (volatility, z-scores, returns) as features and cross-checks; Marquee data only if GS_CLIENT_ID and GS_CLIENT_SECRET exist |
| federicomariamassari/willowtree | lab modern | Willow-tree lattice pricing of American options, cross-checked against QuantLib |
| federicomariamassari/financial-engineering | lab modern | Its Monte Carlo methods (antithetic / control variates) used for the MC pricing cross-check; pinned clone, import only modules that are importable without side effects, otherwise reimplement the method cited from its README |
| dbrojas/optlib | lab legacy | `optlib.gbs` pricing: Black-76 (futures options), Garman-Kohlhagen (FX), Bjerksund-Stensland (American). `optlib.api` is never used (TD Ameritrade API is dead) |
| google/tf-quant-finance | lab legacy | Vectorised Black-Scholes and Heston pricing of the test set (archived; best effort) |
| RomanMichaelPaolucci/Q-Fin (`qfin`) | lab modern | Stochastic process simulation and MC option pricing cross-check |
| quantsbin/Quantsbin | lab modern (needs Python 3.12+) | Vanilla pricing, Greeks and payoff plots for the report |
| bbcho/finoptions-dev | lab modern | fOptions pricing incl. barrier and Asian exotics (illustrative hedging what-ifs) |
| jo-inc/camofox-browser | lab `fetch/camofox.py` | Headless renderer for public news pages that have no API. Honours robots.txt, user-controlled domain allowlist, 1 request per 5 s per domain, hard blocklist of broker and prop-firm domains. Anti-detection features are not relied on |
| heygen-com/hyperframes | lab `video/` | Weekly 45 s performance recap video (equity curves, open signals, exam pass probability) rendered from an HTML template to MP4 |

Not repos (from the screenshots): the "Kurzgesagt explainer" post (a prompt, not code) and the Matt Dancho
stack (Python, pandas, scikit-learn, LangChain, LangGraph are already used via TradingAgents; the paid
OpenAI API is replaced by the free provider chain).

## 2. Layout

```
weathervane/lab/
  __main__.py            python -m weathervane.lab run --group modern|legacy|video|all
  registry.py            ADAPTERS list; each adapter: name, repo, group, install hint, run(ctx) -> dict
  context.py             shared inputs: option chains, test option set, price data, output dir
  adapters/<one file per library>.py
  fetch/camofox.py       robots.txt check, allowlist/blocklist, rate limit, REST client for the camofox server
  video/template.html    HyperFrames composition; video/render.py fills data and calls the CLI
  report.py              merges group JSON outputs into reports/lab/<date>.md and results/context.json
requirements-lab-modern.txt   (Python 3.12)
requirements-lab-legacy.txt   (Python 3.10)
config/lab.yaml          option tickers, test option set, COT markets, news allowlist/blocklist, video on/off
```

## 3. Inputs

- Option chains via yfinance `Ticker.options` / `option_chain(expiry)` for: SPY, QQQ, IWM, GLD, USO, TLT, FXE (config). Nearest 4 expiries 7 to 120 days out. Keep strikes with open interest > 0 and bid > 0; mid = (bid+ask)/2. Save `out/lab/chains.parquet` (artifact shared between jobs).
- Test option set: 12 deterministic contracts (calls and puts, 3 moneyness levels, 2 expiries, ATM-ish vol 20%, r 4%, q 1%) priced by every pricing library for the agreement table, so libraries are compared on identical inputs even when chains are unavailable.
- Rates: FRED DGS3MO if FRED_API_KEY, else 4% constant.

## 4. Outputs

- `out/lab/<group>.json` per job: per adapter {status: ok|failed|skipped, reason, seconds, results}.
- `reports/lab/<date>.md` (committed):
  1. Library status table (all 22 rows above, including core ones marked "core").
  2. Pricing agreement: per test contract, each library's price and max abs deviation vs QuantLib; flag > 1%.
  3. Volatility dashboard per ticker: ATM IV (30d interpolated), IV rank and percentile over the stored history, 25-delta risk reversal, term slope (90d minus 30d ATM IV), SABR rho and nu, implied 1-week move.
  4. CFTC COT: managed money net position and its 3-year percentile for gold, crude, S&P 500 e-mini, EUR, JPY (via OpenBB `cftc` provider; needs `cftc_app_token` optional).
  5. Indicator cross-check: max abs difference between core indicators and Finance-Python / gs-quant / ffn equivalents on 3 instruments (flag anything above 1e-6 relative where definitions match).
  6. Sleeve weight suggestions (QuantPy MVO and inverse vol) vs current equal risk.
  7. News headlines fetched via camofox for watchlist tickers (title + URL only, allowlisted domains).
- `results/context.json`: compact volatility, positioning and regime context for the LLM researcher prompt and the signals table "context" column (informational).
- `results/lab_history.parquet` is NOT committed; IV history lives in actions/cache under `data/lab/`.
- Video: `out/video/weekly.mp4` force-pushed to the `charts` branch; link in the weekly report.

## 5. Workflow lab.yml

- cron `29 7 * * 6` (Saturday), workflow_dispatch. Jobs:
  1. `chains` (3.12): fetch chains and test set -> upload artifact.
  2. `modern` (3.12, needs chains): pip install -r requirements-lab-modern.txt (each package installed separately with `|| echo failed` so one failure does not stop the rest), run group modern.
  3. `legacy` (3.10, needs chains): apt-get install libquantlib0-dev swig build-essential; install legacy packages one by one, best effort; run group legacy.
  4. `video` (Node 22 + FFmpeg, needs modern): install HyperFrames per its README, render; skip cleanly on any error.
  5. `report` (3.12, needs all, `if: always()`): merge, commit reports/lab and results/context.json, push video to charts branch.
- timeout 25 per job. Estimated ~100 minutes a month.
- camofox runs only inside `modern` when `lab.yaml: news.enabled: true` (default false), started with `npx @askjo/camofox-browser` and `CAMOFOX_CRASH_REPORT_ENABLED=false`.

## 6. Rules for adapters

- Every adapter returns within 120 s (enforced with a timeout wrapper) and never raises.
- Install-time failures are detected by import failure and reported with the exception text (first 200 chars).
- No adapter writes outside `out/lab/`. No adapter uses credentials other than those named above.
- Tests: registry lists every repo in section 1; adapters for numpy-only libraries (vollib, pysabr, qfin, finoptions, quantsbin, optlib, willowtree) have a unit test on one contract with tolerance 1e-2 vs a reference Black-Scholes; tests skip if the library is not installed.
