"""Fill the HyperFrames composition with this week's numbers and render it to out/video/weekly.mp4.

Steps: collect data (simulated equity, open signals, exam odds, lab context), write a one-file project to
out/video/project/index.html, then run the pinned HyperFrames CLI (`lint`, `check`, `render`) through npx with telemetry
and update checks off. Any failure (no Node, no network, lint errors, render crash) ends in a failed or skipped record
and never raises; the video is optional context for the weekly report.

This file builds the project and the command lines; whether the CLI accepts them is only confirmed by a real render in
CI (Node 22+ and FFmpeg are required by HyperFrames).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ... import config
from ..adapters.base import Skipped
from ..context import LabContext
from ..util import clip, write_json

log = logging.getLogger("weathervane.lab")
TEMPLATE = Path(__file__).with_name("template.html")
SCENE_SPLIT = (("a", 0.0, 0.12), ("b", 0.12, 0.40), ("c", 0.40, 0.66), ("d", 0.66, 0.88), ("e", 0.88, 1.0))
MAX_POINTS = 160
CLI_ENV = {"HYPERFRAMES_NO_TELEMETRY": "1", "DO_NOT_TRACK": "1", "HYPERFRAMES_NO_UPDATE_CHECK": "1",
           "HYPERFRAMES_NO_AUTO_INSTALL": "1", "HYPERFRAMES_SKIP_SKILLS": "1", "CI": "1"}
QUIET_ACTIONS = ("HOLD", "FLAT")


def video_dir(*parts: str) -> Path:
    p = config.path("out", "video", *parts)
    (p.parent if parts and "." in parts[-1] else p).mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------------------------------------------
# data


def downsample(values: list[float], limit: int = MAX_POINTS) -> list[float]:
    if len(values) <= limit:
        return [float(v) for v in values]
    idx = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(int))
    return [float(values[i]) for i in idx]


def portfolio_block(pid: str) -> dict:
    from .. import sleeves
    cfg = config.load_engine()
    pf = config.load_portfolios()[pid]
    rules = config.load_rules(pf.rules)
    champion = config.load_champion(pid)
    block: dict = {"id": pid, "champion": champion.get("id"), "firm": f"{rules.firm} {rules.program}", "exam": None,
                   "equity": [], "days": 0, "data_source": None}
    try:
        streams, info = sleeves.portfolio_streams(pid)
        risk = float(config.live_risk(champion, cfg, rules))
        equity = 1.0 + np.cumsum(streams.sum(axis=1).to_numpy(float) * risk)
        block.update(equity=downsample(list(equity)), days=int(len(streams)), data_source=info["data_source"])
    except Exception as exc:
        block["note"] = clip(f"{type(exc).__name__}: {exc}", 120)
    exam = (champion.get("scores") or {}).get("exam")
    if exam:
        block["exam"] = {k: exam.get(k) for k in ("pass_combined", "pass_p1", "pass_p2", "days_median")}
    return block


def signals_block() -> tuple[list[dict], str]:
    from .. import live
    payload = config.read_json(config.path("signals", "latest.json"), None)
    if not payload:
        return [], "signals/latest.json not found"
    rows = []
    for pid, p in (payload.get("portfolios") or {}).items():
        for r in p.get("rows", []):
            if r.get("action") in QUIET_ACTIONS:
                continue
            rows.append({"portfolio": pid, "instrument": r.get("instrument"), "action": r.get("action"), "side": r.get("side"),
                         "entry": live.fmt_price(r.get("entry_price")), "stop": live.fmt_price(r.get("stop")),
                         "target": live.fmt_price(r.get("target"))})
    return rows, "" if rows else "No open signals right now."


def lab_block(ctx: LabContext) -> dict:
    """Volatility and positioning lines from the group results already on disk (empty when none)."""
    from .. import report
    block: dict = {"vol": [], "cot": []}
    try:
        groups = report.load_groups(ctx.out)
        vol = report.vol_section(ctx, groups, write_history=False)
        block["vol"] = [{"ticker": t, "atm_iv_30d": r.get("atm_iv_30d"), "iv_rank": r.get("iv_rank")}
                        for t, r in vol.items() if r.get("atm_iv_30d") is not None]
        cot = report.adapter_results(groups, "openbb_odp").get("markets", {})
        block["cot"] = [{"label": m["label"], "net": m.get("net"), "percentile": m.get("percentile")}
                        for m in cot.values() if m.get("status") == "ok"]
    except Exception as exc:
        log.warning("video lab context unavailable: %s", clip(exc))
    return block


def collect(ctx: LabContext) -> dict:
    portfolios = []
    for pid in config.load_portfolios():
        try:
            portfolios.append(portfolio_block(pid))
        except Exception as exc:
            log.warning("video: portfolio %s skipped: %s", pid, clip(exc))
    signals, note = signals_block()
    return {"date": ctx.today.isoformat(), "portfolios": portfolios, "signals": signals, "signals_note": note,
            "lab": lab_block(ctx)}


# ----------------------------------------------------------------------------------------------------------------
# project and CLI


def scene_times(seconds: float) -> dict[str, dict[str, float]]:
    return {name: {"start": round(a * seconds, 3), "duration": round((b - a) * seconds, 3)} for name, a, b in SCENE_SPLIT}


def fill_template(data: dict, cfg_video: dict, template: str | None = None) -> str:
    seconds = float(cfg_video["seconds"])
    scenes = scene_times(seconds)
    html = template if template is not None else TEMPLATE.read_text(encoding="utf-8")
    tokens = {"WIDTH": str(int(cfg_video["width"])), "HEIGHT": str(int(cfg_video["height"])), "DURATION": f"{seconds:g}",
              "SCENES_JSON": json.dumps(scenes)}
    for name, t in scenes.items():
        tokens[f"{name.upper()}_START"] = f"{t['start']:g}"
        tokens[f"{name.upper()}_DUR"] = f"{t['duration']:g}"
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, default=str).replace("</", "<\\/")
    tokens["DATA_JSON"] = payload          # last, so a '{{' inside the data is never expanded
    for key, value in tokens.items():
        html = html.replace("{{" + key + "}}", value)
    return html


def build_project(ctx: LabContext, data: dict) -> Path:
    project = video_dir("project")
    (project / "index.html").write_text(fill_template(data, ctx.cfg["video"]), encoding="utf-8")
    write_json(video_dir("data.json"), data)
    return project


def cli(version: str, *args: str) -> list[str]:
    """Command line for the pinned CLI through npx."""
    return ["npx", "--yes", f"hyperframes@{version}", *args]


def render_args(output: Path, fps: int) -> list[str]:
    return ["render", "--output", str(output), "--fps", str(int(fps)), "--quality", "standard", "--workers", "2", "--strict", "--quiet"]


def run_cli(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    env = {**os.environ, **CLI_ENV}
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        text = out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
        return 124, f"timeout after {timeout}s {text[-300:]}"
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def parse_json_tail(text: str) -> dict:
    """The CLI prints one JSON object with --json; take the last line that parses as one."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    return {}


def run(ctx: LabContext) -> dict:
    v = ctx.cfg["video"]
    if not v.get("enabled", True):
        raise Skipped("video.enabled is false in config/lab.yaml")
    if shutil.which("npx") is None:
        raise Skipped("npx not found (HyperFrames needs Node.js 22+ and FFmpeg)")
    data = collect(ctx)
    project = build_project(ctx, data)
    version = str(v["hyperframes_version"])
    out: dict = {"project": str(project.relative_to(config.root())), "version": version,
                 "portfolios": [p["id"] for p in data["portfolios"]], "signals": len(data["signals"])}

    rc, text = run_cli(cli(version, "lint", str(project), "--json"), project, 240)
    lint = parse_json_tail(text)
    out["lint"] = {"errors": lint.get("errorCount"), "warnings": lint.get("warningCount")}
    if rc != 0 or (lint.get("errorCount") or 0) > 0:
        out.update(_status="failed", _reason=clip(f"hyperframes lint failed (exit {rc}): {text[-160:]}"))
        return out

    rc, text = run_cli(cli(version, "check", str(project), "--json"), project, 420)
    check = parse_json_tail(text)
    out["check"] = {"ok": check.get("ok"), "exit": rc}      # informational: layout findings do not block the render

    target = video_dir("weekly.mp4")
    if target.exists():
        target.unlink()
    rc, text = run_cli(cli(version, *render_args(target, int(v["fps"]))), project, 1500)
    if rc != 0 or not target.exists() or target.stat().st_size < 10_000:
        out.update(_status="failed", _reason=clip(f"hyperframes render failed (exit {rc}): {text[-160:]}"))
        return out
    out["video"] = str(target.relative_to(config.root()))
    out["bytes"] = int(target.stat().st_size)
    return out
