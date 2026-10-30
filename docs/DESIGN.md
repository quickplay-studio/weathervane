# Weathervane v2: design spec

Self-improving swing-trading analysis engine that runs entirely on GitHub Actions.
It outputs long / short / flat signals with entry, stop, target and exit rules, draws them on charts,
forward-tests them on paper accounts, and searches for better strategies every night.
It never places real-money orders. It never calls Anthropic/Claude APIs at runtime.

## 1. Hard constraints

- Runtime LLMs: free keys only (Gemini, Groq, OpenRouter) via plain HTTPS `requests`. No Anthropic SDK. GitHub Models is retired (30 Jul 2026), do not use it.
- Paper trading only: hardcode `https://api-fxpractice.oanda.com` and `https://paper-api.alpaca.markets`. Any other base URL raises. No FTMO credentials anywhere.
- Everything must degrade gracefully: missing secret -> that feature is skipped with a log line, never a crash.
- Private repo on the free plan: about 2,000 Linux minutes a month in total. Every workflow has a hard `timeout-minutes` and a time budget inside the code.
- Python 3.12 on Actions (code must also run on 3.13 locally). Bar timeframe: daily (`1d`) for v2.0.
- Prose in README / docs / reports: plain, terse, no em dashes.

## 2. Repo layout

```
weathervane/
  README.md                 setup, how it works, where outputs live
  requirements.txt
  program.md                instructions read by the LLM researcher (AutoResearch pattern)
  config/
    engine.yaml             budgets, walk-forward windows, gates, LLM provider chain, notify mention
    universe.yaml           instruments and symbol maps
    rules/ftmo_2step_swing.yaml
    portfolios.yaml         portfolios: id, rules file, allowed instruments
  champion/<portfolio>.json current champion per portfolio (sleeves + live params)
  champion/history/         previous champions, timestamped
  weathervane/              python package
    __main__.py             CLI: fetch | research | promote | signals | paper | charts | pine | report | companies | selftest
    config.py  data.py  indicators.py  spec.py  signals_core.py  simulator.py
    metrics.py  exam.py  regime.py  walkforward.py  search.py  scoring.py
    researcher.py  llm.py  live.py  ledger.py  charts.py  pine.py  reports.py
    notify.py  companies.py  synthetic.py
    paper/oanda.py  paper/alpaca.py  paper/sync.py
  results/experiments.csv   every experiment (append-only)
  results/leaderboard.json  best challengers that passed OOS gates, awaiting weekly promotion
  results/paper/            ledger.csv, state.json (internal paper ledger)
  signals/latest.json  signals/latest.md  signals/history.csv
  pine/<portfolio>_<sleeve>.pine
  reports/weekly/<date>.md  reports/companies/<date>_<ticker>.md
  tests/
  .github/workflows/research.yml signals.yml weekly.yml companies.yml
```

Charts (PNG) are NOT committed to main (git history bloat). They are force-pushed each run to an orphan branch `charts` holding a single commit. README links to `../../blob/charts/<file>.png`.
Market data is NOT committed. It lives in `data/` restored with `actions/cache` (key by date, restore-keys prefix) and is fetched incrementally.

## 3. Universe (config/universe.yaml)

Each instrument: `id`, `name`, `asset_class` (fx, index, commodity, crypto, etf), `yf` (yfinance ticker), `oanda` (OANDA v20 instrument or null), `tv` (TradingView symbol), `alpaca_proxy` (ETF or null), `ftmo` (bool, tradable at FTMO), `cost_bps` (round-trip cost estimate), `contract_note` (free text, "verify against FTMO spec").

Seed list:
- FX: EURUSD (EURUSD=X, EUR_USD, OANDA:EURUSD, FXE), GBPUSD (GBPUSD=X, GBP_USD, OANDA:GBPUSD, FXB), USDJPY (JPY=X, USD_JPY, OANDA:USDJPY, null), AUDUSD (AUDUSD=X, AUD_USD, OANDA:AUDUSD, FXA), USDCAD (CAD=X, USD_CAD, OANDA:USDCAD, null), USDCHF (CHF=X, USD_CHF, OANDA:USDCHF, null), NZDUSD (NZDUSD=X, NZD_USD, OANDA:NZDUSD, null). cost 1.0 bps.
- Indices: US500 (ES=F, SPX500_USD, OANDA:SPX500USD, SPY), US100 (NQ=F, NAS100_USD, OANDA:NAS100USD, QQQ), US30 (YM=F, US30_USD, OANDA:US30USD, DIA), GER40 (^GDAXI, DE30_EUR, OANDA:DE30EUR, EWG), UK100 (^FTSE, UK100_GBP, OANDA:UK100GBP, EWU), JP225 (^N225, JP225_USD, OANDA:JP225USD, EWJ). cost 1.5 bps.
- Commodities: XAUUSD (GC=F, XAU_USD, OANDA:XAUUSD, GLD) 2 bps, XAGUSD (SI=F, XAG_USD, OANDA:XAGUSD, SLV) 3 bps, USOIL (CL=F, WTICO_USD, OANDA:WTICOUSD, USO) 4 bps, NATGAS (NG=F, NATGAS_USD, OANDA:NATGASUSD, UNG) 6 bps.
- Crypto: BTCUSD (BTC-USD, null, BITSTAMP:BTCUSD, null) 10 bps.
- Legacy Weathervane ETFs (ftmo false, alpaca_proxy = itself): SPY, EFA, EEM, IEF, TLT, LQD, GLD, DBC, VNQ, UUP. cost 2 bps.

## 4. Data (data.py)

- `fetch(instrument_ids)`: if `OANDA_TOKEN` and instrument has `oanda` -> OANDA practice candles (`/v3/instruments/{i}/candles?granularity=D&price=M&count=5000`, daily alignment `dailyAlignment=17`, `alignmentTimezone=America/New_York`), else yfinance `history(period="max", interval="1d", auto_adjust=True)`.
- Normalise to columns open, high, low, close, volume; tz-naive date index; drop rows with NaN OHLC; drop the current incomplete bar (bar date >= today UTC for yfinance unless market closed; for OANDA use `complete` flag).
- Cache `data/<id>.parquet`; incremental update (fetch last 30 days, merge, dedupe). If fetch fails, keep cache and log. Retry with backoff; sleep 1s between yfinance tickers.
- `load(ids, start=None)` returns dict id -> DataFrame. Minimum history 1,500 bars or instrument is skipped.
- `synthetic.py`: deterministic generator (seeded) of OHLC series with regime switching (trend up, trend down, chop, high-vol), realistic gaps and intrabar ranges. `--synthetic` flag on every CLI command swaps the loader. All tests and `selftest` use it (the build sandbox cannot reach Yahoo).

## 5. Strategy grammar (spec.py): Pine-portable only

Every rule must be exactly reproducible in Pine Script v6. Indicators must follow Pine semantics:
- `sma`, `ema` (alpha 2/(n+1), seeded with SMA of first n values like Pine `ta.ema`), `rma` (Wilder, alpha 1/n, seeded with SMA), `rsi` (rma of gains/losses), `atr` (rma of true range; first TR = high-low), `highest(src,n)`, `lowest(src,n)`, `roc(src,n)=100*(src-src[n])/src[n]`, `stdev` (population, like `ta.stdev` biased), Bollinger (sma ± mult*stdev), Keltner (ema ± mult*atr), DMI/ADX exactly as `ta.dmi(len, len)`, `percentrank(src,n)` = Pine `ta.percentrank` (percent of the previous n values that are <= current), `crossover(a,b)` = a>b and a[1]<=b[1], `crossunder` mirror.

A **sleeve** is a JSON object:
```json
{
  "id": "trend_a",
  "instruments": ["EURUSD","US500","XAUUSD"],
  "direction": "both",
  "entry": {"type": "donchian_breakout", "len": 55},
  "filters": [{"type": "trend_ma", "kind": "ema", "len": 200}],
  "exit": {"stop_atr": 2.5, "target_atr": null, "trail_atr": 3.0, "time_stop_bars": null,
           "exit_signal": {"type": "donchian_exit", "len": 20}, "reverse_on_opposite": true},
  "atr_len": 20,
  "param_space": {"entry.len": [20, 100, 5], "exit.stop_atr": [1.5, 4.0, 0.25], "exit.trail_atr": [2.0, 5.0, 0.25]}
}
```
`param_space` maps dotted paths to `[low, high, step]` (ints if all ints). Shared params across all instruments in the sleeve (no per-instrument fitting).

Entry types (signal evaluated on bar close t):
- `donchian_breakout(len)`: long if close > highest(high,len)[1]; short if close < lowest(low,len)[1].
- `ma_cross(kind, fast, slow)`: long on crossover(ma_fast, ma_slow); short on crossunder.
- `keltner_breakout(len, mult)`: long on crossover(close, ema+mult*atr(len)); short on crossunder(close, ema-mult*atr).
- `momentum_roc(len, thr)`: long on crossover(roc, thr); short on crossunder(roc, -thr).
- `rsi_reversion(len, lower, upper)`: long on crossover(rsi, lower); short on crossunder(rsi, upper).
- `bollinger_reversion(len, mult)`: long on crossover(close, lower band); short on crossunder(close, upper band).

Filters (all must pass; a filter gates entries only):
- `trend_ma(kind, len)`: longs only if close > ma, shorts only if close < ma.
- `adx_min(len, min)` / `adx_max(len, max)`.
- `atr_rank(len, lookback, min, max)`: percentrank(atr/close, lookback) within [min, max].
- `roc_dir(len)`: longs only if roc(len) > 0, shorts only if < 0.

Exits:
- `stop_atr`: stop = signal_close ∓ stop_atr * atr[t] (computed on the signal bar, active from the fill bar).
- `target_atr` (nullable): limit = signal_close ± target_atr * atr[t].
- `trail_atr` (nullable): chandelier. Each bar after fill: long stop = max(stop, highest(high since entry) - trail_atr*atr); short mirror. Updated on bar close, effective next bar.
- `time_stop_bars` (nullable): market exit at next open once bars_in_trade >= N.
- `exit_signal` (nullable): `donchian_exit(len)` (long exits if close < lowest(low,len)[1]) or `ma_exit(kind,len)` (long exits if close < ma). Market exit next open.
- `reverse_on_opposite`: opposite entry signal while in a position -> close and open opposite at next open (Pine `strategy.entry` reversal). If false, opposite signal is ignored.

`direction`: both | long | short.
Validation: `spec.validate(sleeve)` checks types, ranges, instrument ids exist, max 4 filters, numeric bounds (len 2..400, atr multiples 0.5..10). Unknown keys rejected. Researcher output that fails validation is discarded.

## 6. Simulator (simulator.py): must match Pine's broker emulator

Defaults identical to Pine `strategy()` with `process_orders_on_close=false`:
- Signals on bar close t; market orders fill at open of t+1, plus half the round-trip cost.
- Stop and limit orders become active on the fill bar (t+1) and are checked every bar using Pine's intrabar path assumption: if |open-high| < |open-low| path is open -> high -> low -> close, else open -> low -> high -> close. First level touched on the path wins.
- Gaps: if the open is already beyond the stop (or target), fill at the open.
- Exits by market (time stop, exit_signal, reversal) fill at next open.
- One position per instrument per sleeve. No pyramiding.
- Implement the bar loop with `numba.njit` (fallback to pure Python if numba import fails). Indicators and raw signals are vectorised beforehand.

Output per instrument: trades list (entry/exit date, side, entry/exit price, stop at entry, R multiple = pnl / initial risk, bars held, MAE in R, exit reason) and a daily series with:
- `r_close[t]`: P&L in R units realised + change in open MTM at close.
- `r_worst[t]`: worst intraday P&L in R relative to previous close (uses low for longs / high for shorts on open positions, including the fill bar from the fill price).
- `active[t]`: 1 if any position was open or traded that day.
R units: each trade risks 1R = |entry - initial stop|; MTM is (price - entry)/initial_risk * side.

Portfolio daily series = sum across instruments and sleeves of r_close / r_worst (aligned on union of dates, missing = 0). This is "R at 1 unit risk per trade"; scaling by risk fraction r gives % of account.

## 7. Exam Monte Carlo (exam.py)

Input: portfolio daily DataFrame (r_close, r_worst, active), rules dict, risk fraction grid, n_sims (default 5,000), block length (default 10), seed.
Stationary bootstrap of whole days (keeps cross-asset correlation within a day). Per simulated path, per phase:
- equity starts at 1.0 (account fraction). Day loop:
  - day_start = max(balance, equity) at day start (FTMO basis; equity==balance at close in this model).
  - worst = equity + r * r_worst. Breach if worst <= day_start - daily_loss_limit (fraction of initial) -> fail "daily".
  - floor: static -> 1 - max_loss; eod_trailing -> max(eod high-water) - max_loss, optionally locked at 1.0 once reached (rule flag `trailing_lock_at_start`); intraday_trailing -> same using max(intraday best) approximated by equity + r*max(0, r_close) (document approximation). Breach if worst <= floor -> fail "max_loss".
  - equity += r * r_close; trading_days += active.
  - Pass when equity - 1 >= target and trading_days >= min_days and (consistency is null or best_day_profit <= consistency_pct * total_profit). If consistency fails at target, keep going.
  - Time limit: if set and days exceeded -> fail "time". If no time limit, cap at 500 days -> "timeout".
- Phase 2 starts fresh from 1.0 with its own target, continuing the bootstrap.
Outputs: pass_p1, pass_p2, pass_combined, median and p90 days to pass, fail reason shares.
`optimise_risk(daily, rules, grid=[0.0025..0.02 step 0.0025])` picks r maximising pass_combined (ties -> lower r). Use numpy vectorised over sims (no Python loop over sims).

## 8. Regime (regime.py)

Gaussian HMM (hmmlearn, 3 states, diag cov) on [daily log return, 20d realised vol] of a reference series (US500 for indices/all, per asset class optional). Fit on training data only; label states by mean return (bull, bear, chop). Fallback if hmmlearn missing: rule-based labels (200d trend sign x vol tercile). Used for: (a) gate: trade expectancy per regime, (b) chart background shading. NOT used inside trading rules (not Pine-portable).

## 9. Walk-forward, search, scoring

- Vault: last `vault_days` (default 252) of data is excluded from everything except the weekly promotion check.
- Walk-forward (walkforward.py): rolling train `train_years` (6), test `test_years` (1), step 1 year, first test year 2010 (skip folds lacking data). For each fold: optuna TPE (`n_trials` default 25, seed fixed, objective = train-period score below) over the sleeve's `param_space`, then simulate test year with best params. Stitch test years -> OOS daily series. Cache simulations by (sleeve hash, params, fold) in memory.
- Train objective: exam pass_combined at the best risk on the train daily series (n_sims 1,000 for speed); if trades < 30 multiply by trades/30.
- Live params: after the walk-forward, run optuna once on the last `train_years` before the vault; store as `live_params` in the candidate.
- Scoring (scoring.py) on the OOS series, with rules from the portfolio's rules file:
  - `oos_pass` = pass_combined at the optimised risk (n_sims 5,000).
  - `oos_trades` >= `min_oos_trades` (default 60).
  - Deflated Sharpe ratio probability (Bailey and Lopez de Prado, using daily OOS returns at the chosen risk, number of trials = experiments run so far for this portfolio, variance of their OOS Sharpes) >= `min_dsr` (0.90).
  - Regime check: no regime with >= 20 trades has expectancy below -0.15R.
  - OOS max drawdown at chosen risk < 0.8 * max_loss.
  - Candidate beats champion if all gates pass and oos_pass >= champion_oos_pass + `min_improvement` (0.02). Champion is re-scored on the same folds when data changes (cache champion score per data date).
- Weekly promotion (promote command): take best leaderboard entry, run vault check: simulate vault period with live params; require vault exam pass_combined (bootstrapped from vault days) >= 0.5 * oos_pass and vault total R > 0. Pass -> becomes champion (old one archived), Pine regenerated, notify. Fail -> entry marked `vault_failed` and dropped. Max one vault check per week per portfolio (log count in leaderboard).

## 10. Research loop (researcher.py) = AutoResearch pattern

`research --minutes N --portfolio ftmo`:
1. Load champion (or seed if none). Ensure champion scored for today's data.
2. Repeat until time budget is used:
   a. Propose a candidate portfolio = champion with exactly ONE change. Sources in order: LLM (if any provider key and daily LLM budget left, max `llm_proposals_per_run` default 8), else random mutation. Change types: tweak param_space ranges, swap entry type, add/remove/replace a filter, change exits, change direction, add/remove instruments in a sleeve, add a new sleeve (max 3 sleeves), remove a sleeve.
   b. LLM prompt = program.md + JSON schema of the grammar + current champion JSON + champion scores + last 25 experiment rows (compact) + per-instrument OOS contribution table. Ask for JSON only: {"change_description": str, "rationale": str, "portfolio": {...}}. Parse leniently (strip code fences), validate; on failure one retry with the error; then fall back to random.
   c. Evaluate (walk-forward + scoring). Append a row to results/experiments.csv: ts, portfolio, exp_id, parent_hash, source (llm:provider/model | random), change_description, oos_pass, oos_trades, dsr, regime_ok, maxdd, risk, decision (rejected | leaderboard), runtime_s, spec_hash.
   d. If it beats the champion on all gates -> add to leaderboard.json (keep top 5 by oos_pass). Champion is only replaced by weekly `promote` (limits vault peeking).
3. Also per run: asset scan. Champion sleeves evaluated per instrument OOS; table of each instrument's contribution saved to results/asset_scan.json (feeds the LLM prompt).
Dedup: skip candidates whose spec_hash already appears in experiments.csv.

program.md tells the LLM: goal is the highest probability of passing the exam rules, robustness over performance, prefer fewer parameters, one change at a time, only grammar features, explain the hypothesis in one line.

## 11. LLM router (llm.py)

Provider chain from engine.yaml, each `{provider, model, env_key}`; defaults:
1. groq / `openai/gpt-oss-120b` / GROQ_API_KEY (OpenAI-compatible, `https://api.groq.com/openai/v1/chat/completions`)
2. gemini / `gemini-flash-lite-latest` / GEMINI_API_KEY (`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=`; response JSON mime type)
3. openrouter / `openai/gpt-oss-20b:free` / OPENROUTER_API_KEY (`https://openrouter.ai/api/v1/chat/completions`)
Model names overridable by env (`WV_GROQ_MODEL` etc). On 429/5xx/timeouts move to next provider; record per-provider daily usage in results/llm_usage.json; respect daily caps from config. Timeout 90s. Temperature 0.7. Never log keys.

## 12. Live signals (live.py), ledger, paper, charts, Pine

`signals` command (daily after close):
- For each portfolio champion, each sleeve, each instrument: run the simulator with `live_params` over full history (vault included) and read the state at the last completed bar: position side, entry date/price, initial stop, current stop (after trail), target, bars in trade, and the action for the next open: ENTER_LONG, ENTER_SHORT, EXIT, REVERSE_TO_LONG, REVERSE_TO_SHORT, HOLD, MOVE_STOP, FLAT.
- Sizing: risk_fraction = champion's optimised risk (capped by rules `max_risk_per_trade_pct` if set). Risk cash = account_size * risk_fraction. Units = risk cash / |entry_est - stop| where entry_est = last close. Report risk cash, stop distance, units of the underlying, and `contract_note`. Risk guard: total open risk (sum of open positions' distance-to-stop in cash) must stay <= `risk_guard_pct_of_daily` * daily limit; if exceeded, new entries are scaled down and flagged.
- Write signals/latest.json (machine), signals/latest.md (human table: instrument, TradingView symbol, action, side, entry, stop, target, units, risk, reason), append changes to signals/history.csv.
- `ledger`: internal paper ledger starting at `paper_start_date` (first run date if unset). Each run replays champion live signals forward from the start date using actual next-open fills from data, applies rules day by day (exam state: equity, phase, days, breaches) and writes results/paper/ledger.csv and state.json. When the champion changes, the ledger continues (records champion id per trade).
- `paper`: if OANDA_TOKEN+OANDA_ACCOUNT_ID -> sync OANDA practice for instruments with `oanda` symbol: desired position (side, units scaled to the practice account NAV and risk fraction, stop, target) vs open trades tagged with clientExtensions id `wv-<sleeve>-<instrument>`; open with market order + stopLossOnFill + takeProfitOnFill, close on EXIT, replace stop on MOVE_STOP. If ALPACA keys -> same via Alpaca paper for `alpaca_proxy` symbols (bracket orders; shorts allowed when shortable; whole shares; skip if size < 1). Write results/paper/broker_sync.json with what was done. Dry-run flag prints intended orders.
- `charts`: one PNG per instrument with a live or recent position (plus all champion instruments, max 30): last 300 daily bars, mplfinance candles, regime shading (faint), entry markers (long: green up triangle below bar, short: red down triangle above bar), exits (black x), stop/target as dashed lines for the open position, title with action. Also equity charts: portfolio OOS curve and paper ledger curve. Output to `out/charts/`, the workflow force-pushes them to the `charts` branch.
- `pine`: for each champion sleeve generate `pine/<portfolio>_<sleeve>.pine` from templates: Pine v6 `strategy()` with `overlay=true, process_orders_on_close=false, calc_on_every_tick=false, pyramiding=0, initial_capital=100000, commission_type=strategy.commission.percent, commission_value=<cost_bps/2/100>`; inputs for every param defaulting to live_params; risk % input defaulting to champion risk; qty = (strategy.equity * risk%) / (stop distance * syminfo.pointvalue); entries with `strategy.entry`, stops/targets with `strategy.exit(stop=, limit=)` issued on the signal bar and re-issued each bar for trailing; time stop and exit_signal via `strategy.close`; plot current stop/target with `plot(..., style=plot.style_linebr)`; header comment listing instruments and TradingView symbols, champion id, date, and "generated, do not edit". Same indicator semantics as section 5 (use ta.* builtins).
- `notify`: if signals changed (any action other than HOLD/FLAT) or a champion was promoted, post a comment on a single issue titled "Weathervane signals" (create if missing, label `signals`) via GitHub REST with GITHUB_TOKEN, mentioning `notify_mention` from engine.yaml so the user gets a push notification. Include the latest.md table and links.

## 13. Reports

- `report` (weekly): reports/weekly/<date>.md: champion summary, OOS metrics (Sharpe, Sortino, Calmar, max DD, CVaR 95, expectancy, win rate, profit factor, avg MAE), exam pass probabilities and days, regime breakdown, paper ledger vs backtest, top 5 experiments of the week, LLM usage, Actions minutes estimate. quantstats HTML tearsheet optional into `out/` (pushed to charts branch).
- `companies`: TradingAgents on `companies_per_run` (default 2) tickers from the watchlist rotation (default S&P 100 list in config, rotate by date). Install from git in its own workflow (pinned ref in config). Use its Python API with llm_provider google (quick) and groq or google (deep), data vendors yfinance + SEC EDGAR (`SEC_EDGAR_USER_AGENT` from secret SEC_USER_AGENT) + FRED. Defensive: wrap everything; on quota errors write a short "skipped" note. Save markdown report to reports/companies/. Research only, never feeds signals.

## 14. Workflows (all have workflow_dispatch, Python 3.12, pip cache, `concurrency: {group: weathervane-commit, cancel-in-progress: false}` for those that commit, `permissions: contents: write, issues: write`)

- research.yml: cron `17 2 * * *`, timeout 45. cache restore data -> fetch -> `research --minutes 30` for each portfolio round-robin by weekday -> commit results/ champion/ (git pull --rebase then push, retry 3x).
- signals.yml: cron `23 22 * * 1-5`, timeout 20. fetch -> signals -> ledger -> paper -> charts (push charts branch) -> notify -> commit.
- weekly.yml: cron `41 9 * * 6`, timeout 30. fetch -> promote -> pine -> report -> commit + charts branch.
- companies.yml: cron `13 6 * * 1-5`, timeout 25.
Secrets mapped to env: GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, FRED_API_KEY, FINNHUB_API_KEY, SEC_USER_AGENT, OANDA_TOKEN, OANDA_ACCOUNT_ID, ALPACA_API_KEY, ALPACA_SECRET_KEY.
Estimated monthly minutes: research ~1,050, signals ~130, companies ~330, weekly ~60.

## 15. Rules file (config/rules/ftmo_2step_swing.yaml)

```yaml
firm: FTMO
program: 2-Step Swing
source: https://ftmo.com/en/trading-objectives/
verified_on: null   # user confirms before each exam
account_size: 100000
phases:
  - {name: Challenge, profit_target_pct: 10, min_trading_days: 4}
  - {name: Verification, profit_target_pct: 5, min_trading_days: 4}
daily_loss_limit_pct: 5
max_loss_pct: 10
max_loss_type: static          # static | eod_trailing | intraday_trailing
trailing_lock_at_start: false
consistency_best_day_pct: null
time_limit_days: null
max_risk_per_trade_pct: null
risk_guard_pct_of_daily: 60
weekend_holding: true
news_restriction: false
```

## 16. Seed champions

- Portfolio `ftmo` (rules ftmo_2step_swing, instruments where ftmo true): sleeve `trend_a` = donchian_breakout 55, trend_ma ema 200 filter, stop 2.5 ATR, trail 3 ATR, donchian_exit 20, reverse_on_opposite true, direction both, atr_len 20, instruments: EURUSD, GBPUSD, USDJPY, AUDUSD, US500, US100, GER40, XAUUSD, USOIL.
- Portfolio `etf` (legacy Weathervane, rules ftmo_2step_swing as a risk discipline, Alpaca paper): sleeve `etf_trend` = ma_cross ema 50/200, stop 3 ATR, trail 4 ATR, direction both, the 10 legacy ETFs.

## 17. Tests and selftest

- tests: indicator semantics (ema/rma seeding, rsi, atr, percentrank, dmi on small hand-checked arrays), simulator (stop hit, target hit, gap through stop fills at open, intrabar path rule, reversal, time stop, trailing), exam (constant +1%/day passes p1 in 10 days with min days; constant -6% day fails daily; static vs trailing floors), spec validation, pine generation contains every input and compiles syntactically by simple checks (balanced parens, `//@version=6`).
- `selftest`: synthetic data end-to-end: research 2 minutes (random mutations only), promote (force), signals, ledger, charts, pine, report. Must finish < 5 minutes on a laptop.
