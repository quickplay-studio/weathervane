"""GitHub notification: no-op locally, one issue, one comment with the mention, no duplicates, no secrets in output."""
import json

import pytest

from weathervane import config, notify

TOKEN = "ghs_secret_token_value_123456"
ENV = {"GITHUB_TOKEN": TOKEN, "GITHUB_REPOSITORY": "me/weathervane", "GITHUB_RUN_ID": "99", "GITHUB_REF_NAME": "main"}


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body if body is not None else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, issues=None, fail_on=None):
        self.headers, self.calls, self.issues, self.fail_on = {}, [], issues or [], fail_on

    def request(self, method, url, **kw):
        self.calls.append((method, url.replace("https://api.github.com/repos/me/weathervane", ""), kw))
        path = url.split("weathervane", 1)[1]
        if self.fail_on and self.fail_on in path:
            return FakeResp(500)
        if method == "GET" and path == "/issues":
            return FakeResp(200, self.issues)
        if method == "POST" and path == "/labels":
            return FakeResp(201)
        if method == "POST" and path == "/issues":
            self.issues.append({"number": 7, "title": kw["json"]["title"]})
            return FakeResp(201, {"number": 7})
        if method == "POST" and path.endswith("/comments"):
            return FakeResp(201, {})
        return FakeResp(404)


def payload(actions=("ENTER_LONG", "HOLD", "FLAT")):
    rows = []
    for i, a in enumerate(actions):
        rows.append({"portfolio": "ftmo", "sleeve": "trend_a", "instrument": f"INS{i}", "tv": f"OANDA:INS{i}", "action": a, "side": "long",
                     "entry_price": 1.2345, "stop": 1.2, "target": None, "units": 1000.0, "risk_cash": 500.0, "reason": "entry signal"})
    return {"data_date": "2026-10-28", "portfolios": {"ftmo": {"rows": rows}}}


def cfg_with(mention="@charlie"):
    c = config.load_engine()
    c["notify"]["notify_mention"] = mention
    return c


def test_noop_without_github_env(workspace):
    sess = FakeSession()
    res = notify.notify_signals(cfg_with(), payload(), env={}, session=sess)
    assert res["posted"] is False and "not running" in res["reason"] and sess.calls == []


def test_only_actionable_rows_are_posted(workspace):
    sess = FakeSession()
    res = notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess)
    assert res == {"posted": True, "issue": 7}
    comment = next(kw for m, p, kw in sess.calls if p.endswith("/comments"))["json"]["body"]
    assert comment.startswith("@charlie ") and "INS0" in comment and "INS1" not in comment and "INS2" not in comment
    assert "signals/latest.md" in comment and "blob/charts/ftmo_INS0.png" in comment and "actions/runs/99" in comment


def test_creates_issue_with_label_when_missing_and_reuses_it_later(workspace):
    sess = FakeSession()
    notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess)
    methods = [(m, p) for m, p, _ in sess.calls]
    assert ("POST", "/labels") in methods and ("POST", "/issues") in methods
    created = next(kw for m, p, kw in sess.calls if m == "POST" and p == "/issues")["json"]
    assert created["title"] == "Weathervane signals" and created["labels"] == ["signals"]
    sess2 = FakeSession(issues=[{"number": 7, "title": "Weathervane signals"}])
    notify.notify_signals(cfg_with(), payload(("EXIT",)), env=ENV, session=sess2)
    assert not any(m == "POST" and p == "/issues" for m, p, _ in sess2.calls)
    assert any(p == "/issues/7/comments" for _, p, _ in sess2.calls)


def test_pull_requests_with_the_same_title_are_ignored(workspace):
    sess = FakeSession(issues=[{"number": 3, "title": "Weathervane signals", "pull_request": {}}])
    notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess)
    assert any(m == "POST" and p == "/issues" for m, p, _ in sess.calls)


def test_same_signals_are_not_posted_twice(workspace):
    sess = FakeSession()
    assert notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess)["posted"]
    n = len(sess.calls)
    second = notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess)
    assert second == {"posted": False, "reason": "already posted"} and len(sess.calls) == n
    assert notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess, force=True)["posted"]
    assert notify.notify_signals(cfg_with(), payload(("EXIT",)), env=ENV, session=sess)["posted"]


def test_nothing_actionable_means_no_post(workspace):
    sess = FakeSession()
    res = notify.notify_signals(cfg_with(), payload(("HOLD", "FLAT")), env=ENV, session=sess)
    assert res["posted"] is False and sess.calls == []


def test_reads_latest_json_by_default(workspace):
    config.write_json(config.path("signals", "latest.json"), payload())
    sess = FakeSession()
    assert notify.notify_signals(cfg_with(), env=ENV, session=sess)["posted"]


@pytest.mark.parametrize("text,ok", [("@charlie", True), ("@my-org/team_1", True), ("charlie", False), ("@", False),
                                      ("@a b", False), ("@x\n@everyone", False), ("", False)])
def test_mention_validation(text, ok):
    assert (notify.valid_mention(text) == text) is ok or (not ok and notify.valid_mention(text) == "")


def test_no_mention_configured_posts_without_one(workspace):
    sess = FakeSession()
    notify.notify_signals(cfg_with(""), payload(), env=ENV, session=sess)
    body = next(kw for m, p, kw in sess.calls if p.endswith("/comments"))["json"]["body"]
    assert body.startswith("**Weathervane signals")


def test_http_failure_is_reported_not_raised(workspace):
    sess = FakeSession(fail_on="/comments")
    res = notify.notify_signals(cfg_with(), payload(), env=ENV, session=sess)
    assert res["posted"] is False and res["reason"].startswith("error")
    assert not (config.path("results", "notify_state.json")).exists(), "a failed post is retried next time"


def test_dry_run_prints_the_comment(workspace, capsys):
    res = notify.notify_signals(cfg_with(), payload(), env=ENV, session=FakeSession(), dry_run=True)
    assert res["posted"] is False and "@charlie" in capsys.readouterr().out


def test_promotion_comment(workspace):
    sess = FakeSession()
    results = [{"portfolio": "ftmo", "result": "promoted", "previous": "seed-ftmo", "champion": "ftmo-00042", "vault_pass": 0.61,
                "needed_pass": 0.3, "vault_total_r": 4.2}, {"portfolio": "etf", "result": "vault_failed"}]
    assert notify.notify_promotion(results, cfg_with(), env=ENV, session=sess)["posted"]
    body = next(kw for m, p, kw in sess.calls if p.endswith("/comments"))["json"]["body"]
    assert "ftmo-00042" in body and "seed-ftmo" in body and "etf" not in body and "—" not in body
    assert not notify.notify_promotion([{"portfolio": "etf", "result": "vault_failed"}], cfg_with(), env=ENV, session=sess)["posted"]


def test_token_is_only_sent_to_github(workspace):
    sess = FakeSession()
    notify.post("hello", cfg_with(), env=ENV, session=sess)
    assert all(not isinstance(kw.get("json"), dict) or TOKEN not in json.dumps(kw["json"]) for _, _, kw in sess.calls)
