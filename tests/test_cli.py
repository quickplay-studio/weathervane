"""CLI wiring and repository plumbing: parser, command table, workflows, scripts, secrets hygiene."""
import re
from pathlib import Path

import pytest
import yaml

from weathervane import __main__ as cli

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = ["research", "signals", "weekly", "companies"]
SECRETS = ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "FRED_API_KEY", "FINNHUB_API_KEY", "SEC_USER_AGENT", "OANDA_TOKEN",
           "OANDA_ACCOUNT_ID", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]


def test_every_command_has_a_handler_and_parses():
    ap = cli.build_parser()
    assert set(cli.HANDLERS) == set(cli.COMMANDS)
    expected = {"fetch", "research", "promote", "signals", "ledger", "paper", "charts", "pine", "report", "companies", "selftest"}
    assert expected <= set(cli.COMMANDS)
    for name in cli.COMMANDS:
        assert ap.parse_args([name]).command == name


@pytest.mark.parametrize("argv,attr,value", [
    (["research", "--synthetic", "--minutes", "3"], "minutes", 3.0), (["research"], "portfolio", "auto"),
    (["paper", "--dry-run"], "dry_run", True), (["signals", "--portfolio", "etf"], "portfolio", "etf"),
    (["ledger", "--paper-start", "2026-04-01"], "paper_start", "2026-04-01"), (["companies", "--tickers", "aapl,msft"], "tickers", "aapl,msft"),
    (["selftest", "--synthetic"], "synthetic", True), (["promote", "--force"], "force", True)])
def test_arguments(argv, attr, value):
    assert getattr(cli.build_parser().parse_args(argv), attr) == value


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["trade-for-real"])


def test_signals_command_end_to_end(workspace, capsys):
    assert cli.main(["signals", "--synthetic"]) == 0
    assert "actionable" in capsys.readouterr().out and (workspace / "signals" / "latest.md").exists()


def test_pine_command_writes_files(workspace, capsys):
    assert cli.main(["pine", "--portfolio", "ftmo"]) == 0
    files = sorted((workspace / "pine").glob("ftmo_*.pine"))
    assert files and files[0].read_text().startswith("//@version=6")


def test_companies_command_skips_cleanly(workspace, capsys):
    assert cli.main(["companies", "--tickers", "aapl", "--minutes", "1"]) == 0
    assert "AAPL: skipped" in capsys.readouterr().out


def test_paper_dry_run_command(workspace, capsys):
    assert cli.main(["signals", "--synthetic"]) == 0
    assert cli.main(["paper", "--dry-run", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "oanda: skipped" in out or "oanda:" in out


def workflow(name):
    return yaml.safe_load((REPO / ".github" / "workflows" / f"{name}.yml").read_text())


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_shape(name):
    wf = workflow(name)
    triggers = wf.get(True) or wf.get("on")                          # YAML 1.1 reads a bare `on` as True
    assert "schedule" in triggers and "workflow_dispatch" in triggers
    assert wf["concurrency"]["group"] == "weathervane-commit" and wf["concurrency"]["cancel-in-progress"] is False
    assert wf["permissions"]["contents"] == "write"
    job = next(iter(wf["jobs"].values()))
    assert job["runs-on"] == "ubuntu-latest" and 5 <= job["timeout-minutes"] <= 60
    env = job["env"]
    for secret in SECRETS:
        assert env[secret] == "${{ secrets." + secret + " }}", secret
    steps = job["steps"]
    assert steps[0]["uses"].startswith("actions/checkout")
    assert any("setup-python" in s.get("uses", "") and str(s["with"]["python-version"]) == "3.12" for s in steps)


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflows_do_not_interpolate_inputs_into_shell(name):
    text = (REPO / ".github" / "workflows" / f"{name}.yml").read_text()
    for step in (s for job in yaml.safe_load(text)["jobs"].values() for s in job["steps"]):
        assert "${{ inputs." not in step.get("run", "") and "${{ github.event" not in step.get("run", "")


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_commands_exist(name):
    ap = cli.build_parser()
    text = (REPO / ".github" / "workflows" / f"{name}.yml").read_text()
    for cmd in re.findall(r"python -m weathervane ([a-z]+)([^\n]*)", text):
        ap.parse_args([cmd[0], *[a for a in cmd[1].replace('"', "").split() if not a.startswith("$")]][:1])


def test_workflow_files_for_the_other_agent_are_untouched_by_name():
    names = {p.name for p in (REPO / ".github" / "workflows").glob("*.yml")}
    assert {"research.yml", "signals.yml", "weekly.yml", "companies.yml"} <= names


def test_commit_scripts_retry_with_rebase():
    commit = (REPO / "scripts" / "commit_push.sh").read_text()
    assert "git pull --rebase" in commit and "for" in commit and "git push" in commit
    charts = (REPO / "scripts" / "push_charts.sh").read_text()
    assert "git init" in charts and "-b charts" in charts and "--force" in charts and "x-access-token" in charts


def test_repo_has_no_secret_values_or_em_dashes():
    token_like = re.compile(r"(ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{30,})")
    skip = {"lab", "__pycache__", ".git", "out", "data", "node_modules"}
    for p in REPO.rglob("*"):
        if not p.is_file() or skip & set(p.relative_to(REPO).parts) or p.suffix in {".png", ".parquet", ".pyc"}:
            continue
        if p.name.startswith("test_lab") or p.name.startswith("requirements-lab"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert not token_like.search(text), f"secret-looking value in {p}"
        if p.suffix in {".md", ".yml", ".yaml", ".sh"} and p.name != "DESIGN.md":
            assert "—" not in text, f"em dash in {p.relative_to(REPO)}"
