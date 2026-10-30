"""Company research reports from TauricResearch/TradingAgents. Research only: nothing here ever feeds a signal.

TradingAgents is installed from git by the companies workflow (pinned ref in config/engine.yaml) and is imported lazily.
Everything is wrapped: a missing package, missing key, quota error or crash writes a short "skipped" note and exits 0.
The API used (read from the pinned source, not run in the build sandbox):
    from tradingagents.default_config import build_default_config
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    ta = TradingAgentsGraph(debug=False, config=cfg)
    final_state, rating = ta.propagate(ticker, "YYYY-MM-DD")
    ta.save_reports(final_state, ticker, save_path=dir, html=False)   # returns the path of complete_report.md
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger("weathervane.companies")

BUSDAY_ANCHOR = "2024-01-01"
QUOTA_HINTS = ("429", "quota", "rate limit", "rate_limit", "resource_exhausted", "resourceexhausted", "too many requests", "tokens per")
SECRET_ENV = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "FRED_API_KEY", "FINNHUB_API_KEY", "GOOGLE_API_KEY",
              "OANDA_TOKEN", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GITHUB_TOKEN")
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def reports_dir() -> Path:
    return config.path("reports", "companies")


def last_trading_day(today: date) -> date:
    """The most recent weekday on or before `today`."""
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today


def pick_tickers(watchlist: list[str], per_run: int, day: date) -> list[str]:
    """Rotate through the watchlist by business day so every ticker comes round in turn."""
    n = int(np.busday_count(BUSDAY_ANCHOR, day.isoformat()))
    return [watchlist[(n * per_run + k) % len(watchlist)] for k in range(per_run)]


def redact(text: str) -> str:
    for name in SECRET_ENV:
        val = os.environ.get(name)
        if val and len(val) > 6:
            text = text.replace(val, "***")
    return text


def is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(h in text for h in QUOTA_HINTS)


def prepare_env() -> None:
    """Map the repo's secret names to the names TradingAgents reads. Never overwrites a value that is already set."""
    for src, dst in (("GEMINI_API_KEY", "GOOGLE_API_KEY"), ("SEC_USER_AGENT", "SEC_EDGAR_USER_AGENT")):
        if os.environ.get(src) and not os.environ.get(dst):
            os.environ[dst] = os.environ[src]


def ta_config(cfg: dict, work: Path, deep: tuple[str, str] | None = None) -> dict:
    """TradingAgents config: google quick tier, groq (or google) deep tier, yfinance + SEC EDGAR + FRED data."""
    from tradingagents.default_config import build_default_config
    c = cfg["companies"]
    ta = build_default_config()
    deep_provider, deep_model = deep or (c["deep_provider"], c["deep_model"])
    ta.update({
        "llm_provider": c["quick_provider"], "quick_think_llm": c["quick_model"],
        "deep_think_provider": deep_provider, "deep_think_llm": deep_model,
        "results_dir": str(work / "logs"), "data_cache_dir": str(work / "cache"),
        "memory_log_path": str(work / "memory" / "trading_memory.md"),
        "max_debate_rounds": 1, "max_risk_discuss_rounds": 1, "llm_max_retries": 2,
    })
    vendors = dict(ta.get("data_vendors") or {})
    vendors.update({"core_stock_apis": "yfinance", "technical_indicators": "yfinance", "news_data": "yfinance",
                    "fundamental_data": "sec_edgar,yfinance", "macro_data": "fred"})
    ta["data_vendors"] = vendors
    return ta


KEY_ENV = {"groq": ("GROQ_API_KEY",), "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def has_key(provider: str) -> bool:
    return any(os.environ.get(name) for name in KEY_ENV.get(provider, ()))


def deep_choices(cfg: dict) -> list[tuple[str, str]]:
    """Deep-tier (provider, model) attempts in order, skipping providers without a key."""
    c = cfg["companies"]
    out = [(c["deep_provider"], c["deep_model"]), (c["deep_fallback_provider"], c["deep_fallback_model"])]
    return [(p, m) for p, m in out if has_key(p)]


def note(ticker: str, day: str, status: str, detail: str) -> str:
    return (f"# TradingAgents report: {ticker} ({day})\n\nStatus: {status}.\n\n{detail}\n\n"
            "Research only. This never feeds the trading signals.\n")


def analyse(ticker: str, day: str, cfg: dict, work: Path) -> tuple[str, str]:
    """(status, markdown) for one ticker. Never raises."""
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        return "skipped", note(ticker, day, "skipped", "No GEMINI_API_KEY is set, so the quick model tier is unavailable.")
    choices = deep_choices(cfg)
    if not choices:
        return "skipped", note(ticker, day, "skipped", "No key is set for any deep-tier provider (GROQ_API_KEY or GEMINI_API_KEY).")
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except Exception as exc:
        return "skipped", note(ticker, day, "skipped", f"TradingAgents is not installed ({type(exc).__name__}). "
                               "The companies workflow installs it from git.")
    last_error = ""
    for deep in choices:
        try:
            ta = TradingAgentsGraph(debug=False, config=ta_config(cfg, work, deep))
            final_state, rating = ta.propagate(ticker, day)
            out_dir = work / "reports" / ticker
            path = Path(ta.save_reports(final_state, ticker, save_path=out_dir, html=False))
            body = path.read_text(encoding="utf-8").replace("\u2014", "-")
            header = (f"<!-- generated by weathervane.companies with TradingAgents; research only -->\n"
                      f"Provider tiers: quick {cfg['companies']['quick_provider']}/{cfg['companies']['quick_model']}, "
                      f"deep {deep[0]}/{deep[1]}. Rating: {rating}.\n\nResearch only. This never feeds the trading signals.\n\n")
            return "ok", header + body
        except Exception as exc:  # TradingAgents and its LLM stack raise many kinds of error
            last_error = redact(f"{type(exc).__name__}: {str(exc)[:240]}")
            log.warning("%s with deep %s/%s failed: %s", ticker, deep[0], deep[1], last_error)
            if not is_quota_error(exc):
                break
    status = "skipped (quota)" if any(h in last_error.lower() for h in QUOTA_HINTS) else "failed"
    return status, note(ticker, day, status, f"The run did not finish. Last error: {last_error}")


def run(cfg: dict, tickers: list[str] | None = None, minutes: float = 18.0, today: date | None = None) -> list[dict]:
    """Analyse the day's tickers, save reports/companies/<date>_<ticker>.md, and return a status row per ticker."""
    prepare_env()
    day = last_trading_day(today or datetime.now(timezone.utc).date())
    picks = tickers or pick_tickers(cfg["companies"]["watchlist"], int(cfg["companies"]["per_run"]), day)
    deadline = time.monotonic() + minutes * 60
    results = []
    reports_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wv_ta_") as tmp:
        for ticker in picks:
            if not TICKER_RE.match(ticker):
                log.warning("skipping invalid ticker %r", ticker)
                continue
            if time.monotonic() > deadline:
                status, text = "skipped", note(ticker, day.isoformat(), "skipped", "The time budget for this run was used up.")
            else:
                started = time.monotonic()
                status, text = analyse(ticker, day.isoformat(), cfg, Path(tmp))
                log.info("%s: %s in %.0fs", ticker, status, time.monotonic() - started)
            path = reports_dir() / f"{day.isoformat()}_{ticker}.md"
            path.write_text(text, encoding="utf-8")
            results.append({"ticker": ticker, "status": status, "path": str(path)})
    return results
