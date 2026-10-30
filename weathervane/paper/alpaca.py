"""Alpaca paper-trading connector: plan bracket orders from signal rows, then send them (or only print them in dry-run).

Only the paper trading host is accepted for orders and account calls. Prices for ETF proxies come from Alpaca's market
data host, which is also hardcoded and is only ever used for GET requests.
"""
from __future__ import annotations

import logging
import math

import requests

from . import Order, entry_side, wants_open

log = logging.getLogger("weathervane.alpaca")

BASE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
ALLOWED_BASES = frozenset({BASE_URL})
ALLOWED_DATA_BASES = frozenset({DATA_URL})
TAG_PREFIX = "wv"
BUYING_POWER_USE = 0.95


def trade_tag(sleeve: str, instrument: str) -> str:
    return f"{TAG_PREFIX}-{sleeve}-{instrument}"


class AlpacaClient:
    """Thin REST client. Refuses any trading host other than the paper one."""

    def __init__(self, key: str, secret: str, base_url: str = BASE_URL, data_url: str = DATA_URL,
                 session: requests.Session | None = None):
        if base_url.rstrip("/") not in ALLOWED_BASES:
            raise ValueError(f"refusing Alpaca base url {base_url!r}: only the paper host is allowed")
        if data_url.rstrip("/") not in ALLOWED_DATA_BASES:
            raise ValueError(f"refusing Alpaca data url {data_url!r}")
        self.base, self.data = base_url.rstrip("/"), data_url.rstrip("/")
        self.http = session or requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _call(self, method: str, path: str, **kw):
        resp = self.http.request(method, f"{self.base}{path}", timeout=30, **kw)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def account(self) -> dict:
        return self._call("GET", "/v2/account")

    def positions(self) -> list[dict]:
        return self._call("GET", "/v2/positions")

    def open_orders(self) -> list[dict]:
        """Open orders with bracket legs rolled up, then flattened so every leg is a row of its own."""
        rows = self._call("GET", "/v2/orders", params={"status": "open", "limit": 500, "nested": "true"})
        return flatten_orders(rows)

    def asset(self, symbol: str) -> dict:
        return self._call("GET", f"/v2/assets/{symbol}")

    def last_price(self, symbol: str) -> float | None:
        resp = self.http.get(f"{self.data}/v2/stocks/{symbol}/trades/latest", params={"feed": "iex"}, timeout=30)
        resp.raise_for_status()
        price = resp.json().get("trade", {}).get("p")
        return float(price) if price else None

    def submit(self, body: dict) -> dict:
        return self._call("POST", "/v2/orders", json=body)

    def cancel(self, order_id: str) -> None:
        self._call("DELETE", f"/v2/orders/{order_id}")

    def close_position(self, symbol: str) -> dict:
        return self._call("DELETE", f"/v2/positions/{symbol}")

    def replace(self, order_id: str, body: dict) -> dict:
        return self._call("PATCH", f"/v2/orders/{order_id}", json=body)


def flatten_orders(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows or []:
        out.append(row)
        out.extend(flatten_orders(row.get("legs") or []))
    return out


def position_side(pos: dict) -> str:
    qty = float(pos.get("qty", 0))
    side = pos.get("side") or ("short" if qty < 0 else "long")
    return "short" if side == "short" else "long"


def size_shares(row: dict, nav: float, ratio: float, buying_power: float | None, price: float) -> int:
    """Whole shares so that a stop-out loses about nav * risk_fraction * guard_scale. ratio converts underlying to proxy price."""
    dist = row["stop_distance"] * ratio
    if dist <= 0 or price <= 0:
        return 0
    qty = math.floor(nav * row["risk_fraction"] * row.get("guard_scale", 1.0) / dist)
    if buying_power is not None and qty * price > buying_power * BUYING_POWER_USE:
        qty = math.floor(buying_power * BUYING_POWER_USE / price)
    return max(0, qty)


def rescale(level: float | None, row: dict, price: float, ratio: float) -> float | None:
    """A price level of the underlying expressed in the proxy: same offset from the close, scaled by `ratio`."""
    return None if level is None else round(price + (level - row["close"]) * ratio, 2)


def plan(rows: list[dict], universe: dict, positions: list[dict], orders: list[dict], nav: float, buying_power: float | None,
         assets: dict[str, dict], prices: dict[str, float], shorting_enabled: bool = True) -> list[Order]:
    """Compare each signal row with the positions and open orders of its proxy symbol and list the orders that close the gap."""
    by_symbol = {p["symbol"]: p for p in positions}
    out: list[Order] = []
    seen: set[str] = set()
    ranked = sorted(rows, key=lambda r: r["action"] == "FLAT")          # one row per symbol: actionable rows first
    for row in ranked:
        inst = universe.get(row["instrument"])
        if inst is None or not inst.alpaca_proxy:
            continue
        sym = inst.alpaca_proxy
        base = dict(broker="alpaca", symbol=sym, instrument=row["instrument"], sleeve=row["sleeve"], tag=trade_tag(row["sleeve"], row["instrument"]))
        if sym in seen:
            out.append(Order(kind="skip", note=f"{sym} already handled by another sleeve or instrument", **base))
            continue
        seen.add(sym)
        existing = by_symbol.get(sym)
        act = row["action"]
        if existing and (act in ("EXIT", "FLAT") or act.startswith("REVERSE")):
            out.append(Order(kind="close", ref=sym, note=act, **base))
            existing = None
        if wants_open(row, existing is not None):
            side = entry_side(row)
            info = assets.get(sym, {})
            price = prices.get(sym) or (row["close"] if inst.id == sym else None)
            if price is None:
                out.append(Order(kind="skip", note="no price for the proxy symbol", **base))
                continue
            if info and not info.get("tradable", True):
                out.append(Order(kind="skip", note="asset not tradable", **base))
                continue
            if side == "short" and (not shorting_enabled or (info and not info.get("shortable", True))):
                out.append(Order(kind="skip", note="not shortable on the paper account", **base))
                continue
            ratio = 1.0 if inst.id == sym else price / row["close"]
            qty = size_shares(row, nav, ratio, buying_power, price)
            if qty < 1:
                out.append(Order(kind="skip", note="size below one share", **base))
                continue
            if inst.id == sym:
                stop, target = round(row["stop"], 2), None if row.get("target") is None else round(row["target"], 2)
            else:
                stop, target = rescale(row["stop"], row, price, ratio), rescale(row.get("target"), row, price, ratio)
            out.append(Order(kind="open", side=side, units=float(qty), stop=stop, target=target,
                             note=act + ("" if inst.id == sym else f" (proxy for {inst.id})"), **base))
        elif existing and act in ("MOVE_STOP", "HOLD") and row.get("stop") is not None:
            if inst.id == sym:
                stop = round(row["stop"], 2)
            else:
                last = float(existing.get("current_price") or existing.get("avg_entry_price") or row["close"])
                stop = rescale(row["stop"], row, last, last / row["close"])
            leg = next((o for o in orders if o.get("symbol") == sym and o.get("type") in ("stop", "stop_limit")), None)
            if leg is None:
                out.append(Order(kind="protect", side=position_side(existing), units=abs(float(existing["qty"])), stop=stop,
                                 note="no stop order found, placing one", **base))
            elif abs(float(leg.get("stop_price") or 0) - stop) > 0.005:
                out.append(Order(kind="modify", ref=leg["id"], stop=stop, note=act, **base))
    return out


def order_body(o: Order, stamp: str) -> dict:
    """POST /v2/orders body for an opening order: market entry with a stop loss and optional take profit."""
    body: dict = {"symbol": o.symbol, "qty": str(int(o.units)), "side": "buy" if o.side == "long" else "sell", "type": "market",
                  "time_in_force": "day", "client_order_id": f"{o.tag}-{stamp}"[:128],
                  "stop_loss": {"stop_price": f"{o.stop:.2f}"}}
    if o.target is not None:
        body["order_class"] = "bracket"
        body["take_profit"] = {"limit_price": f"{o.target:.2f}"}
    else:
        body["order_class"] = "oto"
    return body


def execute(orders: list[Order], client: AlpacaClient | None, dry_run: bool, stamp: str, open_orders: list[dict]) -> list[Order]:
    """Send the planned orders. Each failure is recorded on the order and the rest still run."""
    for o in orders:
        if o.kind == "skip":
            o.status = "skipped"
            continue
        if dry_run or client is None:
            o.status = "dry_run"
            log.info("dry-run: %s %s %s qty=%s stop=%s target=%s", o.kind, o.symbol, o.side, o.units, o.stop, o.target)
            continue
        try:
            if o.kind == "open":
                resp = client.submit(order_body(o, stamp))
            elif o.kind == "close":
                for leg in open_orders:
                    if leg.get("symbol") == o.symbol and leg.get("status") not in ("filled", "canceled"):
                        try:
                            client.cancel(leg["id"])
                        except requests.RequestException:
                            pass                     # legs of a cancelled parent may already be gone
                resp = client.close_position(o.symbol)
            elif o.kind == "protect":
                resp = client.submit({"symbol": o.symbol, "qty": str(int(o.units)), "side": "sell" if o.side == "long" else "buy",
                                      "type": "stop", "stop_price": f"{o.stop:.2f}", "time_in_force": "gtc",
                                      "client_order_id": f"{o.tag}-stop-{stamp}"[:128]})
            else:
                resp = client.replace(o.ref, {"stop_price": f"{o.stop:.2f}"})
            o.status, o.detail = "sent", str(resp.get("id", ""))[:60]
        except requests.RequestException as exc:
            o.status, o.detail = "error", _error_text(exc)
            log.warning("alpaca %s %s failed: %s", o.kind, o.symbol, o.detail)
    return orders


def _error_text(exc: requests.RequestException) -> str:
    resp = getattr(exc, "response", None)
    body = resp.text[:200] if resp is not None and resp.text else ""
    return f"{type(exc).__name__} {getattr(resp, 'status_code', '')} {body}".strip()


def gather_state(client: AlpacaClient, symbols: list[str]) -> dict:
    """Account, positions, open orders, asset flags and last prices for the symbols in play."""
    acct = client.account()
    assets, prices = {}, {}
    for sym in symbols:
        try:
            assets[sym] = client.asset(sym)
        except requests.RequestException as exc:
            log.warning("alpaca asset %s: %s", sym, _error_text(exc))
        try:
            price = client.last_price(sym)
            if price:
                prices[sym] = price
        except requests.RequestException as exc:
            log.warning("alpaca price %s: %s", sym, _error_text(exc))
    return {"nav": float(acct["equity"]), "buying_power": float(acct.get("buying_power", acct["equity"])),
            "shorting_enabled": bool(acct.get("shorting_enabled", True)) and not acct.get("trading_blocked", False),
            "positions": client.positions(), "orders": client.open_orders(), "assets": assets, "prices": prices}
