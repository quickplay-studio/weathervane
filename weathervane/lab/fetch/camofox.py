"""Headline fetching through a local camofox-browser server (jo-inc/camofox-browser), with strict guard rails.

Rules enforced before every request, in this order:
  1. hard blocklist: broker and prop-firm domains (including FTMO, TopStep, OANDA, Alpaca) can never be fetched, whatever
     the configuration says; the user's own blocklist can only add domains;
  2. user allowlist: only hosts under a domain listed in config/lab.yaml `news.allowlist` are fetched; an empty list
     fetches nothing;
  3. robots.txt of the host must allow the URL (a missing robots.txt allows; 401/403, server errors and unreachable
     hosts count as a denial);
  4. one request per `news.delay_s` (5 s) per domain, robots.txt downloads included.
The result keeps only the headline text and the URL of allowlisted links. No cookies, no logins, no anti-bot tricks:
camofox's fingerprint features are not relied on, and a page that denies access is simply skipped.

The server is started by the workflow with `npx @askjo/camofox-browser` (CAMOFOX_CRASH_REPORT_ENABLED=false) only when
news.enabled is true. The REST calls follow the server's openapi.json (/health, /tabs, /tabs/{id}/navigate,
/tabs/{id}/links, DELETE /tabs/{id}, DELETE /sessions/{userId}); they have not been run against a live server here.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote, urlparse

import requests

from ..adapters.base import Skipped
from ..context import LabContext
from ..util import clip

log = logging.getLogger("weathervane.lab")

USER_AGENT = "WeathervaneLab/1.0 (research context; robots.txt honoured)"
USER_ID = "weathervane-lab"
SESSION_KEY = "news"

# Broker, exchange-trading and prop-firm domains: never fetched, not configurable away.
HARD_BLOCKLIST = (
    "ftmo.com", "topstep.com", "topstepx.com", "oanda.com", "alpaca.markets", "alpaca.markets.com",
    "interactivebrokers.com", "ibkr.com", "tradestation.com", "tastytrade.com", "tastyworks.com", "schwab.com", "fidelity.com",
    "robinhood.com", "etrade.com", "webull.com", "ig.com", "fxcm.com", "forex.com", "saxo.com", "saxobank.com", "pepperstone.com",
    "icmarkets.com", "xm.com", "plus500.com", "cmcmarkets.com", "capital.com", "etoro.com", "binance.com", "coinbase.com", "kraken.com",
    "bybit.com", "metatrader4.com", "metatrader5.com", "mql5.com", "myfundedfx.com", "fundednext.com", "the5ers.com", "e8markets.com",
    "e8funding.com", "fundingpips.com", "apextraderfunding.com", "alphacapitalgroup.uk", "tradeify.co", "fxify.com", "trueforexfunds.com",
    "blueberryfunded.com", "goatfundedtrader.com", "brightfunded.com", "fundedelitetraders.com",
)


# ----------------------------------------------------------------------------------------------------------------
# domain rules


def host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def matches(host: str, domains) -> bool:
    """True when host equals a listed domain or is a subdomain of one."""
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in (x.lower().lstrip(".") for x in domains) if d)


def is_blocked(url: str, extra: tuple | list = ()) -> bool:
    host = host_of(url)
    return not host or matches(host, HARD_BLOCKLIST) or matches(host, extra)


def is_allowed(url: str, allowlist, extra_block: tuple | list = ()) -> bool:
    """Allowed means: http(s), not blocked, and under an allowlisted domain. An empty allowlist allows nothing."""
    if urlparse(url).scheme not in ("http", "https") or is_blocked(url, extra_block):
        return False
    return matches(host_of(url), allowlist or ())


class RateLimiter:
    """At most one request per `interval` seconds for each domain."""

    def __init__(self, interval: float = 5.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval, self.clock, self.sleep = float(interval), clock, sleep
        self._last: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def wait(self, host: str) -> float:
        now = self.clock()
        last = self._last.get(host)
        pause = max(0.0, self.interval - (now - last)) if last is not None else 0.0
        if pause > 0:
            self.sleep(pause)
        self._last[host] = self.clock()
        self.count[host] = self.count.get(host, 0) + 1
        return pause


class Robots:
    """robots.txt per host, fetched through the rate limiter and cached for the run."""

    def __init__(self, limiter: RateLimiter, session: requests.Session | None = None, timeout: float = 10.0) -> None:
        self.limiter, self.session, self.timeout = limiter, session or requests.Session(), timeout
        self._cache: dict[str, urllib.robotparser.RobotFileParser | bool] = {}

    def allows(self, url: str) -> bool:
        parts = urlparse(url)
        key = f"{parts.scheme}://{parts.netloc}"
        if key not in self._cache:
            self._cache[key] = self._load(key, parts.hostname or "")
        rules = self._cache[key]
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch("*", url) and rules.can_fetch(USER_AGENT, url)

    def _load(self, origin: str, host: str):
        self.limiter.wait(host)
        try:
            resp = self.session.get(origin + "/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=self.timeout)
        except requests.RequestException:
            return False                       # unreachable: be conservative
        if resp.status_code in (401, 403) or resp.status_code >= 500:
            return False
        if resp.status_code >= 400:
            return True                        # 404 and friends: no robots.txt, everything allowed
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser


@dataclass
class Guard:
    allowlist: list[str]
    blocklist: list[str] = field(default_factory=list)
    robots: Robots | None = None
    limiter: RateLimiter = field(default_factory=RateLimiter)

    def check(self, url: str) -> tuple[bool, str]:
        if is_blocked(url, self.blocklist):
            return False, "blocked domain"
        if not is_allowed(url, self.allowlist, self.blocklist):
            return False, "not on the allowlist"
        if self.robots is not None and not self.robots.allows(url):
            return False, "robots.txt denies"
        return True, ""

    def pace(self, url: str) -> None:
        self.limiter.wait(host_of(url))


# ----------------------------------------------------------------------------------------------------------------
# REST client


class CamofoxClient:
    def __init__(self, base_url: str, guard: Guard, session: requests.Session | None = None, timeout: float = 60.0,
                 user_id: str = USER_ID) -> None:
        self.base, self.guard, self.timeout, self.user_id = base_url.rstrip("/"), guard, timeout, user_id
        self.http = session or requests.Session()
        self.tab: str | None = None

    def _json(self, method: str, path: str, **kwargs) -> dict:
        resp = self.http.request(method, self.base + path, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def health(self) -> bool:
        try:
            return bool(self._json("GET", "/health").get("ok"))
        except (requests.RequestException, ValueError):
            return False

    def open(self, url: str) -> dict:
        """Navigate (creating the tab on first use). The caller has already passed the guard."""
        self.guard.pace(url)
        if self.tab is None:
            data = self._json("POST", "/tabs", json={"userId": self.user_id, "sessionKey": SESSION_KEY, "url": url})
            self.tab = data.get("tabId")
            return data
        return self._json("POST", f"/tabs/{quote(self.tab, safe='')}/navigate", json={"userId": self.user_id, "url": url})

    def links(self) -> list[dict]:
        if self.tab is None:
            return []
        data = self._json("GET", f"/tabs/{quote(self.tab, safe='')}/links", params={"userId": self.user_id})
        return list(data.get("links") or [])

    def close(self) -> None:
        for method, path, params in (("DELETE", f"/tabs/{quote(self.tab, safe='')}" if self.tab else None, {"userId": self.user_id}),
                                     ("DELETE", f"/sessions/{quote(self.user_id, safe='')}", None)):
            if path is None:
                continue
            try:
                self.http.request(method, self.base + path, params=params, timeout=15)
            except requests.RequestException:
                pass
        self.tab = None


# ----------------------------------------------------------------------------------------------------------------
# headlines


MIN_TITLE_CHARS = 25
SKIP_TITLE = re.compile(r"^(sign in|log in|subscribe|privacy|terms|cookie|menu|skip to|read more|more|home)\b", re.I)


def headline_links(raw_links: list[dict], page_url: str, guard: Guard, limit: int, seen: set[str]) -> list[dict]:
    """Headline-like links on an allowlisted host: text long enough to be a title, deduplicated, title and URL only."""
    out = []
    for link in raw_links:
        title = " ".join(str(link.get("text") or "").split())
        href = str(link.get("href") or "")
        if len(title) < MIN_TITLE_CHARS or SKIP_TITLE.match(title) or href in seen:
            continue
        if not is_allowed(href, guard.allowlist, guard.blocklist):
            continue
        seen.add(href)
        out.append({"title": clip(title, 200), "url": href, "domain": host_of(href)})
        if len(out) >= limit:
            break
    return out


def fetch_headlines(client: CamofoxClient, tickers: list[str], sources: list[str], per_ticker: int) -> dict:
    result: dict = {"headlines": [], "skipped": [], "errors": []}
    seen: set[str] = set()
    for ticker in tickers:
        found = 0
        for template in sources:
            url = template.replace("{ticker}", quote(ticker, safe=""))
            ok, why = client.guard.check(url)
            if not ok:
                result["skipped"].append({"url": url, "reason": why})
                continue
            try:
                client.open(url)
                links = client.links()
            except (requests.RequestException, ValueError) as exc:
                result["errors"].append({"url": url, "error": clip(f"{type(exc).__name__}: {exc}", 120)})
                continue
            for item in headline_links(links, url, client.guard, per_ticker - found, seen):
                result["headlines"].append({"ticker": ticker, **item})
                found += 1
            if found >= per_ticker:
                break
    return result


def run(ctx: LabContext) -> dict:
    news = ctx.cfg["news"]
    if not news.get("enabled"):
        raise Skipped("news.enabled is false in config/lab.yaml")
    if not news.get("allowlist"):
        raise Skipped("news.allowlist is empty: no domain is permitted")
    if not news.get("sources"):
        raise Skipped("news.sources is empty: no page template to fetch")
    limiter = RateLimiter(float(news.get("delay_s", 5.0)))
    guard = Guard(list(news["allowlist"]), list(news.get("blocklist") or []), Robots(limiter), limiter)
    client = CamofoxClient(news["base_url"], guard)
    if not client.health():
        raise RuntimeError(f"camofox server not reachable at {news['base_url']}")
    try:
        result = fetch_headlines(client, list(news["tickers"]), list(news["sources"]), int(news["max_per_ticker"]))
    finally:
        client.close()
    result["requests_by_domain"] = dict(limiter.count)
    return result
