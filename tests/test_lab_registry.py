"""Lab registry, adapter guard, configuration and the deterministic test set (docs/LAB.md sections 1, 2, 6)."""
import re
import time
from pathlib import Path

import pytest

from weathervane.lab import context, registry
from weathervane.lab.adapters import base

REPO = Path(__file__).resolve().parents[1]


def doc_names() -> list[str]:
    """Lower-cased repo names (last path part) from the first column of the section 1 table of docs/LAB.md."""
    text = (REPO / "docs" / "LAB.md").read_text(encoding="utf-8")
    section = text.split("## 1.", 1)[1].split("\n## ", 1)[0]
    names = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or not cells or cells[0] in ("Repo", "") or set(cells[0]) <= set("-"):
            continue
        names.append(re.sub(r"\s*\(.*\)", "", cells[0]).split("/")[-1].lower())
    return names


def test_registry_has_22_unique_rows():
    assert len(registry.ADAPTERS) == 22
    assert len(set(registry.repos())) == 22
    assert len({e.key for e in registry.ADAPTERS}) == 22


def test_registry_lists_every_repo_in_the_spec():
    spec = doc_names()
    assert len(spec) == 22, spec
    assert sorted(spec) == sorted(r.split("/")[-1].lower() for r in registry.repos())


def test_groups_and_modules():
    groups = {g: registry.group_entries(g) for g in registry.GROUPS}
    assert {e.key for e in groups["video"]} == {"hyperframes"}
    assert "camofox" in {e.key for e in groups["modern"]}
    assert {e.key for e in groups["legacy"]} >= {"pyql", "quantpy", "finance_python", "pynance", "optlib", "tf_quant_finance"}
    for e in registry.ADAPTERS:
        assert e.group in (registry.CORE, *registry.GROUPS)
        if e.group == registry.CORE:
            assert e.core and not e.module
        else:
            assert e.module and e.install


def test_every_adapter_module_defines_run():
    import importlib
    for e in registry.ADAPTERS:
        if e.module:
            assert callable(importlib.import_module(e.module).run), e.key


# --- the guard -----------------------------------------------------------------------------------------------------


def test_guard_ok_and_shape():
    rec = base.run_guarded(lambda ctx: {"x": 1.5}, None, timeout=5)
    assert set(rec) == {"status", "reason", "seconds", "results"}
    assert rec["status"] == "ok" and rec["results"] == {"x": 1.5}


def test_guard_never_raises():
    def boom(ctx):
        raise ValueError("bad input " * 50)

    rec = base.run_guarded(boom, None, timeout=5)
    assert rec["status"] == "failed" and rec["reason"].startswith("ValueError") and len(rec["reason"]) <= 200
    rec = base.run_guarded(lambda ctx: (_ for _ in ()).throw(SystemExit(3)), None, timeout=5)
    assert rec["status"] == "failed"
    rec = base.run_guarded(lambda ctx: "not a dict", None, timeout=5)
    assert rec["status"] == "failed"


def test_guard_import_error_reason():
    def f(ctx):
        import definitely_not_a_module_xyz  # noqa: F401

    rec = base.run_guarded(f, None, timeout=5)
    assert rec["status"] == "failed" and rec["reason"].startswith("import failed")


def test_guard_skipped_and_forced_status():
    def skip(ctx):
        raise base.Skipped("no credentials")

    assert base.run_guarded(skip, None, timeout=5)["status"] == "skipped"
    rec = base.run_guarded(lambda ctx: {"_status": "failed", "_reason": "partial", "kept": 1}, None, timeout=5)
    assert rec["status"] == "failed" and rec["reason"] == "partial" and rec["results"] == {"kept": 1}


def test_guard_timeout_returns_promptly():
    before = len(base.STUCK_THREADS)
    t0 = time.monotonic()
    rec = base.run_guarded(lambda ctx: time.sleep(3) or {}, None, timeout=0.3, name="sleeper")
    assert time.monotonic() - t0 < 2
    assert rec["status"] == "failed" and "timeout" in rec["reason"]
    assert len(base.STUCK_THREADS) == before + 1
    base.STUCK_THREADS.pop()


# --- context -------------------------------------------------------------------------------------------------------


def test_test_set_is_twelve_deterministic_contracts():
    a, b = context.build_test_set(), context.build_test_set()
    assert a == b and len(a) == 12 and len({c.id for c in a}) == 12
    c = next(c for c in a if c.id == "C100_45d")
    assert (c.S, c.K, c.r, c.q, c.sigma) == (100.0, 100.0, 0.04, 0.01, 0.20)
    assert c.T == pytest.approx(45 / 365)


def test_config_file_matches_defaults_shape():
    cfg = context.load_config()
    for key in context.DEFAULTS:
        assert key in cfg
    assert cfg["news"]["enabled"] is False and cfg["news"]["allowlist"] == []
    assert cfg["tolerances"]["agreement_flag_rel"] == 0.01
    assert cfg["video"]["hyperframes_version"]


def test_synthetic_chains_are_priceable(workspace):
    cfg = context.load_config()
    df, meta = context.synthetic_chains(cfg, context.today_utc(), 0.04)
    assert not df.empty and meta["source"] == "synthetic"
    assert {"ticker", "expiry", "days", "strike", "cp", "mid", "spot", "T", "r", "q"} <= set(df.columns)
