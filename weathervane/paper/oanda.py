"""OANDA v20 practice connector: plan orders from signal rows, then send them (or only print them in dry-run)."""
from __future__ import annotations

import logging

import requests

from . import Order, entry_side, wants_open

log = logging.getLogger("weathervane.oanda")

BASE_URL = "https://api-fxpractice.oanda.com"
ALLOWED_BASES = frozenset({BASE_URL})
TAG_PREFIX = "wv"
DEFAULT_PRICE_PRECISION = 5
JPY_PRECISION = 3


def trade_tag(sleeve: str, instrument: str) -> str:
    return f"{TAG_PREFIX}-{sleeve}-{instrument}"


class OandaClient:
    """Thin v20 REST client. Only the practice host is accepted."""

    def __init__(self, token: str, account_id: str, base_url: str = BASE_URL, session: requests.Session | None = None):
        if base_url.rstrip("/") not in ALLOWED_BASES:
            raise ValueError(f"refusing OANDA base url {base_url!r}: only the practice host is allowed")
        self.base = base_url.rstrip("/")
        self.account = account_id
        self.http = session or requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

    def _call(self, method: str, path: str, **kw) -> dict:
        resp = self.http.request(method, f"{self.base}/v3/accounts/{self.account}{path}", timeout=30, **kw)
        resp.raise_for_status()
        return resp.json()

    def nav(self) -> float:
        return float(self._call("GET", "/summary")["account"]["NAV"])

    def open_trades(self) -> list[dict]:
        return self._call("GET", "/openTrades").get("trades", [])

    def instrument_info(self, instrument: str) -> dict:
        rows = self._call("GET", "/instruments", params={"instruments": instrument}).get("instruments", [])
        return rows[0] if rows else {}

    def quote_home_factor(self, instrument: str) -> float:
        prices = self._call("GET", "/pricing", params={"instruments": instrument}).get("prices", [])
        factors = prices[0].get("quoteHomeConversionFactors", {}) if prices else {}
        return float(factors.get("positiveUnits", 1.0))

    def market_order(self, instrument: str, units: float, stop: float | None, target: float | None, tag: str,
                     price_prec: int) -> dict:
        order: dict = {"type": "MARKET", "instrument": instrument, "units": str(units), "timeInForce": "FOK",
                       "positionFill": "DEFAULT", "tradeClientExtensions": {"id": tag, "tag": TAG_PREFIX}}
        if stop is not None:
            order["stopLossOnFill"] = {"price": f"{stop:.{price_prec}f}", "timeInForce": "GTC"}
        if target is not None:
            order["takeProfitOnFill"] = {"price": f"{target:.{price_prec}f}", "timeInForce": "GTC"}
        return self._call("POST", "/orders", json={"order": order})

    def close_trade(self, trade_id: str) -> dict:
        return self._call("PUT", f"/trades/{trade_id}/close", json={"units": "ALL"})

    def set_exits(self, trade_id: str, stop: float | None, target: float | None, price_prec: int) -> dict:
        body: dict = {}
        if stop is not None:
            body["stopLoss"] = {"price": f"{stop:.{price_prec}f}", "timeInForce": "GTC"}
        if target is not None:
            body["takeProfit"] = {"price": f"{target:.{price_prec}f}", "timeInForce": "GTC"}
        return self._call("PUT", f"/trades/{trade_id}/orders", json=body)


def price_precision(info: dict, instrument: str) -> int:
    if "displayPrecision" in info:
        return int(info["displayPrecision"])
    return JPY_PRECISION if instrument.endswith("JPY") else DEFAULT_PRICE_PRECISION


def size_units(row: dict, nav: float, conv: float, units_prec: int) -> float:
    """Units so that a stop-out loses nav * risk_fraction * guard_scale in account currency."""
    dist = row["stop_distance"] * conv
    if dist <= 0:
        return 0.0
    raw = nav * row["risk_fraction"] * row.get("guard_scale", 1.0) / dist
    return round(raw, units_prec) if units_prec > 0 else float(int(raw))


def plan(rows: list[dict], universe: dict, trades: list[dict], nav: float, info: dict[str, dict],
         conv: dict[str, float]) -> list[Order]:
    """Compare each signal row with the open trades tagged for it and list the orders that close the gap."""
    by_tag = {t.get("clientExtensions", {}).get("id", ""): t for t in trades}
    orders: list[Order] = []
    for row in rows:
        inst = universe.get(row["instrument"])
        if inst is None or not inst.oanda:
            continue
        tag = trade_tag(row["sleeve"], row["instrument"])
        existing = by_tag.get(tag)
        base = dict(broker="oanda", symbol=inst.oanda, instrument=row["instrument"], sleeve=row["sleeve"], tag=tag)
        prec = price_precision(info.get(inst.oanda, {}), inst.oanda)
        act = row["action"]
        if existing and (act in ("EXIT", "FLAT") or act.startswith("REVERSE")):
            orders.append(Order(kind="close", ref=existing["id"], note=f"{act}", **base))
            existing = None
        if wants_open(row, existing is not None):
            units = size_units(row, nav, conv.get(inst.oanda, 1.0), int(info.get(inst.oanda, {}).get("tradeUnitsPrecision", 0)))
            if units <= 0:
                orders.append(Order(kind="skip", note="size rounds to zero", **base))
                continue
            side = entry_side(row)
            orders.append(Order(kind="open", side=side, units=units if side == "long" else -units,
                                stop=round(row["stop"], prec), target=None if row.get("target") is None else round(row["target"], prec),
                                note=f"{act}", **base))
        elif existing and act in ("MOVE_STOP", "HOLD"):
            cur = existing.get("stopLossOrder", {}).get("price")
            if cur is None or abs(float(cur) - row["stop"]) > 0.5 * 10 ** -prec:
                orders.append(Order(kind="modify", ref=existing["id"], stop=round(row["stop"], prec),
                                    target=None if row.get("target") is None else round(row["target"], prec), note=act, **base))
    return orders


def execute(orders: list[Order], client: OandaClient | None, dry_run: bool, info: dict[str, dict]) -> list[Order]:
    """Send the planned orders. Each failure is recorded on the order and the rest still run."""
    for o in orders:
        if o.kind == "skip":
            o.status = "skipped"
            continue
        if dry_run or client is None:
            o.status = "dry_run"
            log.info("dry-run: %s %s %s units=%s stop=%s target=%s", o.kind, o.symbol, o.side, o.units, o.stop, o.target)
            continue
        prec = price_precision(info.get(o.symbol, {}), o.symbol)
        try:
            if o.kind == "open":
                resp = client.market_order(o.symbol, o.units, o.stop, o.target, o.tag, prec)
            elif o.kind == "close":
                resp = client.close_trade(o.ref)
            else:
                resp = client.set_exits(o.ref, o.stop, o.target, prec)
            o.status, o.detail = "sent", ",".join(k for k in resp if k.endswith("Transaction"))[:200]
        except requests.RequestException as exc:
            o.status, o.detail = "error", _error_text(exc)
            log.warning("oanda %s %s failed: %s", o.kind, o.symbol, o.detail)
    return orders


def _error_text(exc: requests.RequestException) -> str:
    resp = getattr(exc, "response", None)
    body = resp.text[:200] if resp is not None and resp.text else ""
    return f"{type(exc).__name__} {getattr(resp, 'status_code', '')} {body}".strip()


def gather_state(client: OandaClient, symbols: list[str]) -> tuple[float, list[dict], dict[str, dict], dict[str, float]]:
    """NAV, open trades, instrument details and quote-to-account conversion factors for the symbols in play."""
    info, conv = {}, {}
    for sym in symbols:
        info[sym] = client.instrument_info(sym)
        conv[sym] = client.quote_home_factor(sym)
    return client.nav(), client.open_trades(), info, conv
