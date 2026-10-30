"""GitHub notifications: one issue titled "Weathervane signals", one comment per event, mentioning the user.

Uses the GitHub REST API with GITHUB_TOKEN and GITHUB_REPOSITORY from the Actions environment. Locally (either variable
missing) it does nothing and says so. A failure to post is logged and never raised.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re

import requests

from . import config
from .live import QUIET_ACTIONS, fmt_price, fmt_units

log = logging.getLogger("weathervane.notify")

API = "https://api.github.com"
MAX_COMMENT = 60000
MENTION_RE = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})(?:/[A-Za-z0-9_.-]+)?$")
LABEL_COLOUR = "1f5fbf"


def state_path():
    return config.path("results", "notify_state.json")


def valid_mention(text: str) -> str:
    """The mention to put in the comment, or an empty string when none is configured or it is malformed."""
    text = (text or "").strip()
    if not text:
        return ""
    if not MENTION_RE.match(text):
        log.warning("notify_mention %r is not a valid @handle, no mention is added", text)
        return ""
    return text


def actionable_rows(payload: dict) -> list[dict]:
    """Rows whose action asks the user to do something (everything except HOLD and FLAT)."""
    return [r for p in payload.get("portfolios", {}).values() for r in p["rows"] if r["action"] not in QUIET_ACTIONS]


def change_key(payload: dict) -> str:
    """Stable id of this set of actions, so the same signals are not posted twice."""
    items = sorted(f"{r['portfolio']}|{r['sleeve']}|{r['instrument']}|{r['action']}|{fmt_price(r['stop'])}" for r in actionable_rows(payload))
    return hashlib.sha1(("\n".join([str(payload.get("data_date"))] + items)).encode()).hexdigest()[:16]


def links(env: dict) -> dict[str, str]:
    server = env.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repo, ref = env.get("GITHUB_REPOSITORY", ""), env.get("GITHUB_REF_NAME") or "main"
    out = {"signals": f"{server}/{repo}/blob/{ref}/signals/latest.md", "charts": f"{server}/{repo}/blob/charts"}
    if env.get("GITHUB_RUN_ID"):
        out["run"] = f"{server}/{repo}/actions/runs/{env['GITHUB_RUN_ID']}"
    return out


def signals_body(payload: dict, mention: str, env: dict) -> str:
    rows = actionable_rows(payload)
    ln = links(env)
    head = f"{mention} " if mention else ""
    lines = [f"{head}**Weathervane signals for {payload.get('data_date')}**", "",
             "| Portfolio | Instrument | TradingView | Action | Side | Entry | Stop | Target | Units | Risk | Reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['portfolio']} | {r['instrument']} | {r['tv']} | {r['action']} | {r['side']} | {fmt_price(r['entry_price'])} | "
                     f"{fmt_price(r['stop'])} | {fmt_price(r['target'])} | {fmt_units(r['units']) if r['units'] else ''} | "
                     f"{r['risk_cash']:,.0f} | {r['reason']} |")
    lines += ["", "Orders are decided on the last close and fill at the next open. Paper trading only."]
    chart_links = sorted({f"[{r['instrument']}]({ln['charts']}/{r['portfolio']}_{r['instrument']}.png)" for r in rows})
    if env.get("GITHUB_REPOSITORY"):
        lines += ["", f"Full table: [signals/latest.md]({ln['signals']}). Charts: " + ", ".join(chart_links[:20])
                  + (f". [Run]({ln['run']})" if "run" in ln else ".")]
    return "\n".join(lines)[:MAX_COMMENT]


def promotion_body(results: list[dict], mention: str, env: dict) -> str:
    head = f"{mention} " if mention else ""
    lines = [f"{head}**Weathervane champion promoted**", ""]
    for r in results:
        lines.append(f"- {r['portfolio']}: {r.get('previous')} replaced by {r.get('champion')}. Vault exam pass {r.get('vault_pass', 0):.1%} "
                     f"(needed {r.get('needed_pass', 0):.1%}), vault total {r.get('vault_total_r', 0):+.1f} R.")
    ln = links(env)
    lines += ["", "Pine scripts were regenerated in `pine/`. Re-paste them into TradingView.",
              f"[Run]({ln['run']})" if "run" in ln else ""]
    return "\n".join(lines)[:MAX_COMMENT]


class GitHub:
    """The few REST calls needed. The token is only ever sent to api.github.com."""

    def __init__(self, token: str, repo: str, session: requests.Session | None = None):
        self.repo, self.token = repo, token
        self.http = session or requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                  "X-GitHub-Api-Version": "2022-11-28"})

    def call(self, method: str, path: str, **kw) -> requests.Response:
        return self.http.request(method, f"{API}/repos/{self.repo}{path}", timeout=30, **kw)

    def find_issue(self, title: str, label: str) -> int | None:
        resp = self.call("GET", "/issues", params={"labels": label, "state": "all", "per_page": 100})
        resp.raise_for_status()
        for issue in resp.json():
            if issue.get("title") == title and "pull_request" not in issue:
                return int(issue["number"])
        return None

    def ensure_label(self, label: str) -> None:
        resp = self.call("POST", "/labels", json={"name": label, "color": LABEL_COLOUR, "description": "Weathervane signal alerts"})
        if resp.status_code not in (201, 422):          # 422 = already exists
            resp.raise_for_status()

    def create_issue(self, title: str, label: str, body: str) -> int:
        self.ensure_label(label)
        resp = self.call("POST", "/issues", json={"title": title, "body": body, "labels": [label]})
        resp.raise_for_status()
        return int(resp.json()["number"])

    def comment(self, number: int, body: str) -> None:
        resp = self.call("POST", f"/issues/{number}/comments", json={"body": body})
        resp.raise_for_status()


def post(body: str, cfg: dict, env: dict | None = None, session: requests.Session | None = None, dry_run: bool = False) -> dict:
    """Post a comment on the single signals issue, creating the issue first when it does not exist."""
    env = dict(os.environ) if env is None else env
    n_cfg = cfg["notify"]
    if dry_run:
        print(body)
        return {"posted": False, "reason": "dry run"}
    token, repo = env.get("GITHUB_TOKEN"), env.get("GITHUB_REPOSITORY")
    if not token or not repo:
        log.info("notify: GITHUB_TOKEN or GITHUB_REPOSITORY not set, nothing posted")
        return {"posted": False, "reason": "not running in GitHub Actions"}
    gh = GitHub(token, repo, session)
    try:
        number = gh.find_issue(n_cfg["issue_title"], n_cfg["label"])
        if number is None:
            number = gh.create_issue(n_cfg["issue_title"], n_cfg["label"],
                                     "Weathervane posts one comment here whenever a signal needs action or a champion changes. "
                                     "Paper trading only. Mute by unsubscribing from this issue.")
        gh.comment(number, body)
    except requests.RequestException as exc:
        text = str(exc).replace(token, "***")
        log.warning("notify failed: %s: %s", type(exc).__name__, text[:200])
        return {"posted": False, "reason": f"error: {type(exc).__name__}"}
    log.info("notify: commented on issue #%d", number)
    return {"posted": True, "issue": number}


def notify_signals(cfg: dict, payload: dict | None = None, env: dict | None = None, session: requests.Session | None = None,
                   dry_run: bool = False, force: bool = False) -> dict:
    """Comment with the actionable signal rows when they changed since the last notification."""
    payload = payload or config.read_json(config.path("signals", "latest.json"))
    if not payload:
        return {"posted": False, "reason": "no signals/latest.json"}
    if not actionable_rows(payload):
        log.info("notify: no actionable signals")
        return {"posted": False, "reason": "no actionable signals"}
    key = change_key(payload)
    if not force and not dry_run and config.read_json(state_path(), {}).get("last_key") == key:
        log.info("notify: these signals were already posted")
        return {"posted": False, "reason": "already posted"}
    env = dict(os.environ) if env is None else env
    result = post(signals_body(payload, valid_mention(cfg["notify"].get("notify_mention", "")), env), cfg, env, session, dry_run)
    if result["posted"]:
        config.write_json(state_path(), {"last_key": key, "data_date": payload.get("data_date")})
    return result


def notify_promotion(results: list[dict], cfg: dict, env: dict | None = None, session: requests.Session | None = None,
                     dry_run: bool = False) -> dict:
    """Comment that one or more champions were promoted."""
    promoted = [r for r in results if r.get("result") == "promoted"]
    if not promoted:
        return {"posted": False, "reason": "no promotion"}
    env = dict(os.environ) if env is None else env
    return post(promotion_body(promoted, valid_mention(cfg["notify"].get("notify_mention", "")), env), cfg, env, session, dry_run)
