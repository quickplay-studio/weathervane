"""Paper broker connectors: hardcoded hosts, order planning, execution, dry-run, and the sync wrapper."""
import json

import pytest
import requests

from weathervane import config
from weathervane.paper import Order, alpaca, oanda, sync

UNIVERSE = config.load_universe()


def row(instrument="EURUSD", action="ENTER_LONG", side="long", close=1.1, stop=1.08, target=None, sleeve="trend_a", **kw):
    r = {"portfolio": "ftmo", "sleeve": sleeve, "instrument": instrument, "tv": "X", "action": action, "side": side, "close": close,
         "entry_price": close, "stop": stop, "target": target, "stop_distance": round(abs(close - stop), 10), "risk_fraction": 0.005, "guard_scale": 1.0,
         "bars_in_trade": 0, "units": 1.0, "risk_cash": 500.0, "reason": "t"}
    r.update(kw)
    return r


def test_hosts_are_hardcoded():
    assert oanda.BASE_URL == "https://api-fxpractice.oanda.com" and alpaca.BASE_URL == "https://paper-api.alpaca.markets"
    for bad in ("https://api-fxtrade.oanda.com", "https://api.oanda.com", "http://api-fxpractice.oanda.com", "https://evil.example"):
        with pytest.raises(ValueError):
            oanda.OandaClient("t", "a", base_url=bad)
    for bad in ("https://api.alpaca.markets", "https://paper-api.alpaca.markets.evil.com", "http://paper-api.alpaca.markets"):
        with pytest.raises(ValueError):
            alpaca.AlpacaClient("k", "s", base_url=bad)
    with pytest.raises(ValueError):
        alpaca.AlpacaClient("k", "s", data_url="https://evil.example")
    oanda.OandaClient("t", "a")
    alpaca.AlpacaClient("k", "s")
    oanda.OandaClient("t", "a", base_url="https://api-fxpractice.oanda.com/")


# ---- OANDA ------------------------------------------------------------------------------------------------------------

def test_oanda_open_order_sizing_and_levels():
    o = oanda.plan([row(target=1.16)], UNIVERSE, [], 100000.0, {"EUR_USD": {"displayPrecision": 5, "tradeUnitsPrecision": 0}},
                   {"EUR_USD": 1.0})[0]
    assert (o.kind, o.side, o.symbol, o.tag) == ("open", "long", "EUR_USD", "wv-trend_a-EURUSD")
    assert o.units == int(100000 * 0.005 / 0.02) == 25000 and o.stop == 1.08 and o.target == 1.16


def test_oanda_short_units_are_negative_and_conversion_applies():
    o = oanda.plan([row("USDJPY", "ENTER_SHORT", "short", 150.0, 152.0)], UNIVERSE, [], 100000.0, {}, {"USD_JPY": 0.0067})[0]
    assert o.units < 0 and o.side == "short"
    assert abs(o.units) == int(100000 * 0.005 / (2.0 * 0.0067))


def test_oanda_guard_scale_reduces_units():
    full = oanda.plan([row()], UNIVERSE, [], 100000.0, {}, {})[0].units
    half = oanda.plan([row(guard_scale=0.5)], UNIVERSE, [], 100000.0, {}, {})[0].units
    assert half == pytest.approx(full / 2, abs=1)


def test_oanda_close_modify_and_skip():
    trade = {"id": "55", "clientExtensions": {"id": "wv-trend_a-EURUSD"}, "stopLossOrder": {"price": "1.07000"}}
    exit_ = oanda.plan([row(action="EXIT")], UNIVERSE, [trade], 100000.0, {}, {})
    assert [(o.kind, o.ref) for o in exit_] == [("close", "55")]
    move = oanda.plan([row(action="MOVE_STOP", stop=1.09, stop_distance=0.01)], UNIVERSE, [trade], 100000.0, {}, {})
    assert [(o.kind, o.ref, o.stop) for o in move] == [("modify", "55", 1.09)]
    same = oanda.plan([row(action="HOLD", stop=1.07)], UNIVERSE, [trade], 100000.0, {}, {})
    assert same == []
    rev = oanda.plan([row(action="REVERSE_TO_SHORT", side="short", stop=1.12)], UNIVERSE, [trade], 100000.0, {}, {})
    assert [o.kind for o in rev] == ["close", "open"] and rev[1].units < 0
    tiny = oanda.plan([row(stop=0.0, stop_distance=1e12)], UNIVERSE, [], 100000.0, {}, {})
    assert tiny[0].kind == "skip"


def test_oanda_catch_up_opens_a_young_position_that_is_missing():
    young = row(action="HOLD", bars_in_trade=0)
    assert [o.kind for o in oanda.plan([young], UNIVERSE, [], 100000.0, {}, {})] == ["open"]
    old = row(action="HOLD", bars_in_trade=12)
    assert oanda.plan([old], UNIVERSE, [], 100000.0, {}, {}) == []


def test_oanda_instruments_without_a_symbol_are_ignored():
    assert oanda.plan([row("BTCUSD")], UNIVERSE, [], 100000.0, {}, {}) == []


class FakeOanda:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def market_order(self, *a):
        self.calls.append(("market", a))
        if self.fail:
            raise requests.HTTPError("boom")
        return {"orderFillTransaction": {}}

    def close_trade(self, tid):
        self.calls.append(("close", tid))
        return {"orderFillTransaction": {}}

    def set_exits(self, *a):
        self.calls.append(("exits", a))
        return {"stopLossOrderTransaction": {}}


def test_oanda_execute_dry_run_sends_nothing_and_errors_do_not_stop_the_rest():
    orders = [Order("oanda", "open", "EUR_USD", "EURUSD", "s", side="long", units=1000, stop=1.0, tag="t"),
              Order("oanda", "close", "GBP_USD", "GBPUSD", "s", ref="9")]
    client = FakeOanda()
    oanda.execute(orders, client, True, {})
    assert client.calls == [] and [o.status for o in orders] == ["dry_run", "dry_run"]
    bad = FakeOanda(fail=True)
    orders = [Order("oanda", "open", "EUR_USD", "EURUSD", "s", side="long", units=1000, stop=1.0, tag="t"),
              Order("oanda", "close", "GBP_USD", "GBPUSD", "s", ref="9")]
    oanda.execute(orders, bad, False, {})
    assert [o.status for o in orders] == ["error", "sent"] and "HTTPError" in orders[0].detail


# ---- Alpaca -----------------------------------------------------------------------------------------------------------

def etf_row(**kw):
    kw.setdefault("stop", 96.0)
    return row("SPY", close=100.0, **kw)


def test_alpaca_open_order_is_whole_shares_with_bracket_levels():
    o = alpaca.plan([etf_row(target=108.0)], UNIVERSE, [], [], 100000.0, 200000.0, {"SPY": {"tradable": True, "shortable": True}}, {})[0]
    assert (o.kind, o.symbol, o.side, o.units, o.stop, o.target) == ("open", "SPY", "long", 125.0, 96.0, 108.0)
    body = alpaca.order_body(o, "20261028")
    assert body["order_class"] == "bracket" and body["take_profit"] == {"limit_price": "108.00"} and body["stop_loss"] == {"stop_price": "96.00"}
    assert body["qty"] == "125" and body["side"] == "buy" and body["type"] == "market" and body["client_order_id"].endswith("-20261028")
    o2 = alpaca.plan([etf_row()], UNIVERSE, [], [], 100000.0, None, {}, {})[0]
    assert alpaca.order_body(o2, "x")["order_class"] == "oto" and "take_profit" not in alpaca.order_body(o2, "x")


def test_alpaca_size_below_one_share_and_buying_power_cap():
    expensive = etf_row(stop_distance=4.0)
    expensive["risk_fraction"] = 0.00001
    assert alpaca.plan([expensive], UNIVERSE, [], [], 100000.0, None, {}, {})[0].kind == "skip"
    capped = alpaca.plan([etf_row()], UNIVERSE, [], [], 100000.0, 5000.0, {}, {})[0]
    assert capped.units == int(5000 * 0.95 / 100.0)


def test_alpaca_short_needs_shortable_asset_and_account():
    short = etf_row(action="ENTER_SHORT", side="short", stop=104.0)
    assert alpaca.plan([short], UNIVERSE, [], [], 100000.0, None, {"SPY": {"shortable": False}}, {})[0].kind == "skip"
    assert alpaca.plan([short], UNIVERSE, [], [], 100000.0, None, {}, {}, shorting_enabled=False)[0].kind == "skip"
    ok = alpaca.plan([short], UNIVERSE, [], [], 100000.0, None, {"SPY": {"shortable": True}}, {})[0]
    assert ok.kind == "open" and ok.side == "short" and alpaca.order_body(ok, "x")["side"] == "sell"


def test_alpaca_close_modify_protect():
    pos = {"symbol": "SPY", "qty": "50", "side": "long", "current_price": "101"}
    leg = {"id": "o1", "symbol": "SPY", "type": "stop", "stop_price": "95.00", "status": "new"}
    assert [(o.kind, o.symbol) for o in alpaca.plan([etf_row(action="EXIT")], UNIVERSE, [pos], [leg], 1e5, None, {}, {})] == [("close", "SPY")]
    move = alpaca.plan([etf_row(action="MOVE_STOP", stop=97.5)], UNIVERSE, [pos], [leg], 1e5, None, {}, {})
    assert [(o.kind, o.ref, o.stop) for o in move] == [("modify", "o1", 97.5)]
    protect = alpaca.plan([etf_row(action="HOLD", stop=97.5)], UNIVERSE, [pos], [], 1e5, None, {}, {})
    assert [(o.kind, o.units) for o in protect] == [("protect", 50.0)]
    assert alpaca.plan([etf_row(action="HOLD", stop=95.0)], UNIVERSE, [pos], [leg], 1e5, None, {}, {}) == []
    short_pos = {"symbol": "SPY", "qty": "-50"}
    assert alpaca.position_side(short_pos) == "short"


def test_alpaca_proxy_levels_are_scaled_to_the_etf_price():
    r = row("EURUSD", close=1.1, stop=1.078, stop_distance=0.022, target=1.144)
    o = alpaca.plan([r], UNIVERSE, [], [], 100000.0, None, {}, {"FXE": 110.0})[0]          # proxy trades 100x the pair
    assert o.symbol == "FXE" and "proxy for EURUSD" in o.note
    assert o.stop == pytest.approx(110.0 - 2.2, abs=0.01) and o.target == pytest.approx(110.0 + 4.4, abs=0.01)
    assert o.units == int(100000 * 0.005 / 2.2)
    assert alpaca.plan([r], UNIVERSE, [], [], 100000.0, None, {}, {})[0].kind == "skip"       # no proxy price known


def test_alpaca_one_row_per_symbol():
    a, b = etf_row(action="ENTER_LONG"), etf_row(action="ENTER_LONG", sleeve="other")
    kinds = [o.kind for o in alpaca.plan([a, b], UNIVERSE, [], [], 1e5, None, {}, {})]
    assert kinds == ["open", "skip"]


class FakeAlpaca:
    def __init__(self):
        self.calls = []

    def submit(self, body):
        self.calls.append(("submit", body))
        return {"id": "abc"}

    def cancel(self, oid):
        self.calls.append(("cancel", oid))

    def close_position(self, sym):
        self.calls.append(("close", sym))
        return {"id": "c1"}

    def replace(self, oid, body):
        self.calls.append(("replace", oid, body))
        return {"id": oid}


def test_alpaca_execute_cancels_legs_before_closing_and_dry_run_is_silent():
    orders = [Order("alpaca", "close", "SPY", "SPY", "s", ref="SPY"), Order("alpaca", "modify", "GLD", "GLD", "s", ref="o2", stop=150.0)]
    legs = [{"id": "o1", "symbol": "SPY", "status": "new"}, {"id": "o9", "symbol": "GLD", "status": "new"}]
    client = FakeAlpaca()
    alpaca.execute(orders, client, False, "20261028", legs)
    assert client.calls[:2] == [("cancel", "o1"), ("close", "SPY")] and client.calls[2][0] == "replace"
    silent = FakeAlpaca()
    alpaca.execute([Order("alpaca", "open", "SPY", "SPY", "s", side="long", units=5, stop=1.0)], silent, True, "x", [])
    assert silent.calls == []


# ---- sync -------------------------------------------------------------------------------------------------------------

def write_signals(rows_by_pf):
    payload = {"data_date": "2026-10-28", "portfolios": {pid: {"account_size": 100000.0, "rows": rows} for pid, rows in rows_by_pf.items()}}
    config.write_json(config.path("signals", "latest.json"), payload)


def test_sync_dry_run_without_credentials_never_touches_the_network(workspace, cfg, monkeypatch, capsys):
    def no_network(*a, **k):
        raise AssertionError("network used in dry-run without credentials")
    monkeypatch.setattr(requests.Session, "request", no_network)
    write_signals({"ftmo": [row()], "etf": [etf_row()]})
    summary = sync.run(cfg, dry_run=True, env={})
    out = capsys.readouterr().out
    assert summary["dry_run"] and "oanda" in out and "alpaca" in out and "EUR_USD" in out and "SPY" in out
    saved = json.loads((workspace / "results" / "paper" / "broker_sync.json").read_text())
    assert saved["brokers"]["oanda"]["orders"][0]["status"] == "dry_run" and saved["brokers"]["alpaca"]["orders"][0]["kind"] == "open"


def test_sync_skips_brokers_without_credentials(workspace, cfg):
    write_signals({"ftmo": [row()], "etf": [etf_row()]})
    summary = sync.run(cfg, dry_run=False, env={})
    assert all(b["status"] == "skipped: no credentials" and b["orders"] == [] for b in summary["brokers"].values())


def test_sync_routes_portfolios_by_their_brokers(workspace, cfg):
    write_signals({"ftmo": [row()], "etf": [etf_row()]})
    summary = sync.run(cfg, dry_run=True, env={})
    assert {o["broker"] for o in summary["brokers"]["oanda"]["orders"]} == {"oanda"}
    assert {o["broker"] for o in summary["brokers"]["alpaca"]["orders"]} == {"alpaca"}
    assert all(o["instrument"] == "EURUSD" for o in summary["brokers"]["oanda"]["orders"])


def test_sync_synthetic_never_sends_orders_even_with_credentials(workspace, cfg, monkeypatch):
    class FakeClient(FakeOanda):
        def __init__(self, token, account):
            super().__init__()
            FakeClient.last = self

    monkeypatch.setattr(oanda, "OandaClient", FakeClient)
    monkeypatch.setattr(oanda, "gather_state", lambda client, symbols: (50000.0, [], {}, {"EUR_USD": 1.0}))
    write_signals({"ftmo": [row()]})
    summary = sync.run(cfg, dry_run=False, synthetic=True, env={"OANDA_TOKEN": "t", "OANDA_ACCOUNT_ID": "a"})
    assert summary["dry_run"] is True
    assert FakeClient.last.calls == [] and summary["brokers"]["oanda"]["orders"][0]["status"] == "dry_run"


def test_sync_with_credentials_uses_the_clients(workspace, cfg, monkeypatch):
    class FakeClient(FakeOanda):
        def __init__(self, token, account):
            super().__init__()
            FakeClient.last = self

    monkeypatch.setattr(oanda, "OandaClient", FakeClient)
    monkeypatch.setattr(oanda, "gather_state", lambda client, symbols: (50000.0, [], {"EUR_USD": {"displayPrecision": 5, "tradeUnitsPrecision": 0}},
                                                                    {"EUR_USD": 1.0}))
    write_signals({"ftmo": [row()]})
    summary = sync.run(cfg, dry_run=False, env={"OANDA_TOKEN": "t", "OANDA_ACCOUNT_ID": "a"})
    res = summary["brokers"]["oanda"]
    assert res["status"] == "ok" and res["nav"] == 50000.0 and res["orders"][0]["status"] == "sent"
    assert FakeClient.last.calls[0][0] == "market" and FakeClient.last.calls[0][1][1] == 12500
    assert summary["brokers"]["alpaca"]["status"].startswith("skipped")


def test_sync_account_read_failure_is_recorded_not_raised(workspace, cfg, monkeypatch):
    monkeypatch.setattr(oanda, "OandaClient", lambda *a, **k: object())
    def broken(client, symbols):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(oanda, "gather_state", broken)
    write_signals({"ftmo": [row()]})
    summary = sync.run(cfg, dry_run=False, env={"OANDA_TOKEN": "t", "OANDA_ACCOUNT_ID": "a"})
    assert summary["brokers"]["oanda"]["status"].startswith("error") and summary["brokers"]["oanda"]["orders"] == []


def test_sync_needs_signals(workspace, cfg):
    with pytest.raises(SystemExit):
        sync.run(cfg, dry_run=True, env={})
