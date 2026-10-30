"""TradingAgents wrapper: rotation, key handling, defensive skipping. TradingAgents itself is never run."""
import sys
import types
from datetime import date

import pytest

from weathervane import companies, config


def test_rotation_covers_the_watchlist_without_repeats(cfg):
    import numpy as np
    wl, per = cfg["companies"]["watchlist"], cfg["companies"]["per_run"]
    days = np.busday_offset("2026-01-05", np.arange(len(wl) // per), roll="forward")
    seen = [t for d in days for t in companies.pick_tickers(wl, per, date.fromisoformat(str(d)))]
    assert len(seen) == len(set(seen)) == (len(wl) // per) * per


def test_rotation_is_deterministic_and_advances_by_business_day():
    wl = list("ABCDEFGH")
    mon = companies.pick_tickers(wl, 2, date(2026, 10, 26))
    assert companies.pick_tickers(wl, 2, date(2026, 10, 26)) == mon
    tue = companies.pick_tickers(wl, 2, date(2026, 10, 27))
    assert tue != mon and set(tue).isdisjoint(mon)


def test_last_trading_day_skips_weekends():
    assert companies.last_trading_day(date(2026, 10, 31)) == date(2026, 10, 30)      # Saturday
    assert companies.last_trading_day(date(2026, 11, 1)) == date(2026, 10, 30)       # Sunday
    assert companies.last_trading_day(date(2026, 10, 29)) == date(2026, 10, 29)


def test_env_mapping_never_overwrites(monkeypatch):
    for k in ("GOOGLE_API_KEY", "SEC_EDGAR_USER_AGENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123456")
    monkeypatch.setenv("SEC_USER_AGENT", "me me@example.com")
    companies.prepare_env()
    import os
    assert os.environ["GOOGLE_API_KEY"] == "gem-key-123456" and os.environ["SEC_EDGAR_USER_AGENT"] == "me me@example.com"
    monkeypatch.setenv("GOOGLE_API_KEY", "other")
    companies.prepare_env()
    assert os.environ["GOOGLE_API_KEY"] == "other"


@pytest.mark.parametrize("text,quota", [("HTTP 429 Too Many Requests", True), ("RESOURCE_EXHAUSTED: quota exceeded", True),
                                         ("Rate limit reached for tokens per minute", True), ("KeyError: 'foo'", False)])
def test_quota_detection(text, quota):
    assert companies.is_quota_error(RuntimeError(text)) is quota


def test_redact_hides_secret_values(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretvalue")
    assert "gsk_supersecretvalue" not in companies.redact("failed with key gsk_supersecretvalue in header")


def test_skips_without_keys_and_still_writes_a_note(workspace, cfg):
    rows = companies.run(cfg, ["AAPL", "MSFT"], minutes=1, today=date(2026, 10, 28))
    assert [r["status"] for r in rows] == ["skipped", "skipped"]
    text = (workspace / "reports" / "companies" / "2026-10-28_AAPL.md").read_text()
    assert "skipped" in text and "Research only" in text and "—" not in text


def test_invalid_tickers_are_dropped(workspace, cfg):
    rows = companies.run(cfg, ["AAPL", "../../etc/passwd", "bad ticker"], minutes=1, today=date(2026, 10, 28))
    assert [r["ticker"] for r in rows] == ["AAPL"]
    assert not any(p.name.startswith("..") for p in (workspace / "reports" / "companies").iterdir())


def test_default_run_uses_the_rotation(workspace, cfg):
    rows = companies.run(cfg, None, minutes=1, today=date(2026, 10, 28))
    assert len(rows) == cfg["companies"]["per_run"]
    assert [r["ticker"] for r in rows] == companies.pick_tickers(cfg["companies"]["watchlist"], cfg["companies"]["per_run"], date(2026, 10, 28))


def test_missing_package_is_a_skip_not_a_crash(workspace, cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123456")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-123456")
    for name in [m for m in sys.modules if m.startswith("tradingagents")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "tradingagents", None)             # makes `import tradingagents.x` raise ImportError
    rows = companies.run(cfg, ["AAPL"], minutes=1, today=date(2026, 10, 28))
    assert rows[0]["status"] == "skipped"
    assert "not installed" in (workspace / "reports" / "companies" / "2026-10-28_AAPL.md").read_text()


def install_fake_tradingagents(monkeypatch, propagate):
    """A stand-in with the same call shape as the pinned TradingAgents."""
    pkg = types.ModuleType("tradingagents")
    dc = types.ModuleType("tradingagents.default_config")
    dc.build_default_config = lambda: {"data_vendors": {"core_stock_apis": "x"}}
    graph = types.ModuleType("tradingagents.graph")
    tg = types.ModuleType("tradingagents.graph.trading_graph")
    seen = {}

    class Graph:
        def __init__(self, debug=False, config=None):
            seen["config"] = config

        def propagate(self, ticker, day):
            return propagate(ticker, day)

        def save_reports(self, state, ticker, save_path=None, html=False):
            save_path.mkdir(parents=True, exist_ok=True)
            p = save_path / "complete_report.md"
            p.write_text(f"# {ticker}\n\nBody — with a dash\n", encoding="utf-8")
            return p

    tg.TradingAgentsGraph = Graph
    for name, mod in (("tradingagents", pkg), ("tradingagents.default_config", dc), ("tradingagents.graph", graph),
                      ("tradingagents.graph.trading_graph", tg)):
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


def test_successful_run_writes_the_report_with_a_header(workspace, cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123456")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-123456")
    seen = install_fake_tradingagents(monkeypatch, lambda t, d: ({"x": 1}, "Overweight"))
    rows = companies.run(cfg, ["AAPL"], minutes=1, today=date(2026, 10, 28))
    assert rows[0]["status"] == "ok"
    text = (workspace / "reports" / "companies" / "2026-10-28_AAPL.md").read_text()
    assert "Rating: Overweight" in text and "Research only" in text and "—" not in text
    c = seen["config"]
    assert c["llm_provider"] == "google" and c["deep_think_provider"] == "groq" and c["max_debate_rounds"] == 1
    assert c["data_vendors"]["fundamental_data"] == "sec_edgar,yfinance"


def test_quota_error_falls_back_to_the_second_deep_provider(workspace, cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123456")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-123456")
    calls = []

    def propagate(ticker, day):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 Too Many Requests")
        return {}, "Hold"

    install_fake_tradingagents(monkeypatch, propagate)
    rows = companies.run(cfg, ["AAPL"], minutes=1, today=date(2026, 10, 28))
    assert rows[0]["status"] == "ok" and len(calls) == 2


def test_other_errors_do_not_retry_and_never_leak_keys(workspace, cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123456")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-123456")
    calls = []

    def propagate(ticker, day):
        calls.append(1)
        raise ValueError("bad thing with groq-key-123456 inside")

    install_fake_tradingagents(monkeypatch, propagate)
    rows = companies.run(cfg, ["AAPL"], minutes=1, today=date(2026, 10, 28))
    assert rows[0]["status"] == "failed" and len(calls) == 1
    assert "groq-key-123456" not in (workspace / "reports" / "companies" / "2026-10-28_AAPL.md").read_text()


def test_spent_time_budget_skips_remaining_tickers(workspace, cfg):
    rows = companies.run(cfg, ["AAPL", "MSFT"], minutes=0, today=date(2026, 10, 28))
    assert [r["status"] for r in rows] == ["skipped", "skipped"]
