# Weathervane

Weathervane is a swing-trading research engine that runs on free GitHub Actions minutes. It searches for daily-bar trading rules, keeps one champion portfolio per account, turns the champion into end-of-day signals, and paper trades them. It never places a real order. Nothing here is financial advice.

The goal it optimises is the probability of passing a prop-firm exam (FTMO 2-Step Swing by default) under the exam's own rules, measured on data the optimiser never saw.

## What it does

- Fetches daily bars (Yahoo Finance, OANDA practice for FX when a token is set) into a local cache.
- Runs a nightly experiment loop: one change to the champion, walk-forward test, keep or throw away.
- Scores every candidate with a Monte Carlo of the exam rules (stationary bootstrap of out-of-sample days).
- Promotes the best challenger once a week, after a check on a held-back vault of the latest year.
- Writes daily signals for the champion: which instruments to enter, exit, or move a stop on at the next open.
- Replays the champion forward in an internal paper ledger with the exam rules applied day by day.
- Optionally mirrors the signals to an OANDA practice account and an Alpaca paper account.
- Draws PNG charts, generates Pine Script v6 strategies for TradingView, and posts a GitHub issue comment when something needs doing.
- Writes a weekly report and, on weekdays, TradingAgents company research notes (research only).

## How the nightly loop works

1. `fetch` updates the cache. A cache is also kept between runs by `actions/cache`.
2. `research` picks a portfolio (alternating by date) and runs for a fixed time budget (30 minutes in the workflow).
3. Each experiment is one change to the champion: a parameter range, an entry type, a filter, an exit, the direction, the instruments, or a sleeve. Changes come from an LLM when a key is set and the daily request cap allows, otherwise from random mutation. `program.md` is the brief the LLM reads.
4. Each candidate is checked against the grammar, then evaluated with walk-forward optimisation (6 years train, 1 year test, rolling) and scored on the stitched out-of-sample series.
5. A candidate must pass every gate: enough trades, deflated Sharpe probability of at least 0.90, no losing regime, drawdown well inside the exam limit, and an exam pass probability at least 0.02 above the champion. Winners go on a leaderboard of five. Every experiment is logged to `results/experiments.csv`.
6. The champion is replaced only by `promote`, run weekly. It simulates the vault year (never used in search) and requires an exam pass rate of at least half the out-of-sample figure and a positive total R. A failed entry is dropped. The old champion is archived in `champion/history/`.

Everything the engine trades is expressed in a small grammar (donchian, moving average cross, keltner, roc, rsi and bollinger entries; trend, adx, atr rank and roc direction filters; ATR stops, targets, chandelier trail, time stop). That keeps every champion portable to Pine.

## Outputs and where they live

| Path | What |
|---|---|
| `champion/<portfolio>.json` | Current champion spec. `champion/history/` has the archive. |
| `results/experiments.csv` | One row per experiment, accepted or not. |
| `results/leaderboard.json` | Top five challengers per portfolio. |
| `results/paper/ledger.csv`, `equity.csv`, `state.json` | Internal paper ledger and exam state. |
| `results/paper/broker_sync.json` | What the last `paper` run did (or would do in a dry run). |
| `signals/latest.md`, `latest.json`, `history.csv` | Today's signals, machine copy, and every change. |
| `pine/<portfolio>_<sleeve>.pine` | Generated TradingView strategies. Do not edit. |
| `reports/weekly/<date>.md` | Weekly report. |
| `reports/companies/<date>_<ticker>.md` | TradingAgents notes. |
| `charts` branch | Candle charts and equity curves (`out/charts/`, force-pushed, never on `main`). |
| Issue "Weathervane signals" | One comment per day that has something to do. |

## Setup

1. Create a private GitHub repo, push this folder to it, and enable Actions. Python 3.12 is used in the workflows. Locally 3.12 or 3.13 both work (`pip install -r requirements.txt`).
2. Add these repository secrets (Settings, Secrets and variables, Actions). All are optional. Without a key that feature is skipped.

| Secret | Used for |
|---|---|
| `GEMINI_API_KEY` | LLM proposals, and the quick model in company reports |
| `GROQ_API_KEY` | LLM proposals, and the deep model in company reports |
| `OPENROUTER_API_KEY` | Last fallback for LLM proposals |
| `FRED_API_KEY` | Macro data for company reports |
| `FINNHUB_API_KEY` | Reserved for company data |
| `SEC_USER_AGENT` | Contact string SEC EDGAR requires, for example `Your Name you@example.com` |
| `OANDA_TOKEN`, `OANDA_ACCOUNT_ID` | OANDA practice account (FX data and paper orders) |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca paper account |

3. Set `notify.notify_mention` in `config/engine.yaml` to your handle (for example `@yourname`) to get a push notification from the signals issue.
4. Optionally set `signals.paper_start_date` in the same file. Left empty, the ledger starts on the first signals run.
5. Run the workflows once by hand from the Actions tab: research, then signals.

Broker URLs are hardcoded to OANDA practice and Alpaca paper. Any other host raises an error. Run `python -m weathervane paper --dry-run` to see the intended orders without sending any.

Use the tool locally:

```
python -m weathervane selftest --synthetic     # end to end on generated data, about 2.5 minutes
python -m weathervane research --minutes 5 --synthetic
python -m weathervane signals --synthetic
python -m pytest -q
```

The commands are `fetch`, `research`, `promote`, `signals`, `ledger`, `paper`, `charts`, `pine`, `report`, `companies`, `notify` and `selftest`. Commands that read market data take `--synthetic`.

## TradingView: using the Pine strategies

1. Open `pine/<portfolio>_<sleeve>.pine` on GitHub (or locally) and copy the whole file.
2. In TradingView open the Pine Editor, paste it over the editor contents, and click Save, then Add to chart.
3. Open a daily chart on one of the instruments listed in the file header, using the TradingView symbol shown there (for example `OANDA:EURUSD`).
4. In the strategy Settings, check the Inputs tab. Every parameter defaults to the champion's live value. Leave them alone.
5. Read the Strategy Tester. Trades fill at the next open, stops and targets are checked inside the bar, and costs use the commission percent in the header. Small differences from Python are expected because TradingView's data feed and gap handling differ.
6. Files are regenerated on every promotion. Paste the new file when the champion changes.

The Pine files are generated and checked for syntax and for trade-for-trade agreement with a Python transcription of the same logic. They have not been run inside TradingView by the author of this repo, so treat the first import as a test.

## Exam rules file

`config/rules/ftmo_2step_swing.yaml` holds the exam: phases and profit targets, minimum trading days, daily and maximum loss, whether the maximum loss is static or trailing, and the open-risk guard. The engine reads every exam number from this file.

Rules change. Before each real exam, open the firm's page (the `source` line), compare, edit the file, and set `verified_on` to the date you checked. Changing the file changes the pass probabilities. The champion is rescored once per data date, so the new figures appear after the next fetch brings in a new day.

## Actions minutes budget

The free plan gives about 2,000 Linux minutes a month for private repos. Every job has a hard timeout and a time budget inside the code.

| Workflow | Schedule (UTC) | Timeout | Estimated minutes a month |
|---|---|---|---|
| research | daily 02:17 | 45 | 1,050 |
| signals | weekdays 22:23 | 20 | 130 |
| companies | weekdays 06:13 | 25 | 330 |
| weekly | Saturday 09:41 | 30 | 60 |

That is about 1,570 of 2,000. The weekly report prints the estimate against `actions.minutes_per_month_limit`. To save minutes, lower `--minutes` in `research.yml` or run companies less often.

Workflows that commit use one concurrency group and `git pull --rebase` with three retries, so runs do not overwrite each other.

## Lab

An optional, isolated layer integrates a list of other open source finance libraries as research context. It has its own requirements files and workflow, cannot break `signals` or `research`, and never feeds trading rules. See `docs/LAB.md`.

## More detail

`docs/DESIGN.md` is the full specification: grammar, simulator semantics, exam Monte Carlo, walk-forward, scoring gates, live signals, and Pine generation.
