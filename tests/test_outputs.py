"""Charts, weekly report and signals files on synthetic data (slowest unit tests: a few seconds each)."""
import copy
import struct
from datetime import datetime, timezone

import pytest

from weathervane import charts, config, ledger, live, reports

FAST = {"walkforward": {"n_trials": 4, "train_sims": 200, "oos_sims": 600}}


@pytest.fixture
def fast_cfg(workspace):
    return config.load_engine(FAST)


def png_size(path):
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_charts_write_pngs_and_an_index(workspace, fast_cfg):
    ledger.update(["ftmo"], fast_cfg, synthetic=True, start_override="2026-06-01")
    names = charts.generate(["ftmo"], fast_cfg, synthetic=True)
    out = workspace / "out" / "charts"
    assert names and all((out / n).exists() for n in names)
    assert "ftmo_paper_equity.png" in names
    instrument_charts = [n for n in names if n.startswith("ftmo_") and "equity" not in n]
    assert 1 <= len(instrument_charts) <= fast_cfg["charts"]["max_instruments"]
    w, h = png_size(out / instrument_charts[0])
    assert w >= 800 and h >= 400
    index = (out / "index.md").read_text()
    assert "## ftmo" in index and instrument_charts[0] in index and "—" not in index


def test_charts_are_replaced_not_accumulated(workspace, fast_cfg):
    out = workspace / "out" / "charts"
    out.mkdir(parents=True)
    (out / "stale_chart.png").write_bytes(b"x")
    charts.generate(["ftmo"], fast_cfg, synthetic=True)
    assert not (out / "stale_chart.png").exists()


def test_charts_without_data_do_not_crash(workspace, fast_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("weathervane.data.load", lambda ids, cfg, synthetic=False: {})
    names = charts.generate(["ftmo"], fast_cfg, synthetic=False)
    assert not any(n.endswith("_XAUUSD.png") for n in names)


def test_weekly_report_has_every_section_and_no_em_dash(workspace, fast_cfg):
    path = reports.write(["ftmo"], fast_cfg, synthetic=True, now=datetime(2026, 10, 31, 9, 41, tzinfo=timezone.utc))
    assert path.name == "2026-10-31.md" and path.parent == workspace / "reports" / "weekly"
    text = path.read_text()
    assert "—" not in text
    for needle in ("# Weathervane weekly report 2026-10-31", "## ftmo", "seed-ftmo", "pass", "Actions minutes"):
        assert needle.lower() in text.lower(), needle


def test_report_survives_a_portfolio_without_data(workspace, fast_cfg, monkeypatch):
    def boom(pid, cfg, synthetic, champion):
        raise SystemExit("no data for etf")
    monkeypatch.setattr(reports, "ensure_scores", boom)
    text = reports.build(["etf"], fast_cfg, synthetic=True)
    assert "Skipped: no data for etf" in text


def test_signals_markdown_sample_is_stable(workspace, fast_cfg):
    payload = live.generate(["ftmo", "etf"], fast_cfg, synthetic=True)
    md = (workspace / "signals" / "latest.md").read_text()
    assert md.startswith("#") and "ftmo" in md.lower() and "—" not in md
    assert payload["data_date"] == "2026-10-28"
