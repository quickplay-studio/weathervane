"""Merge the group results into reports/lab/<date>.md and results/context.json.

Every section reads what the adapters stored in out/lab/<group>.json and degrades to a one-line note when its inputs
are missing, so a failed job never blocks the report. Nothing here changes trading rules: the output is research
context (DESIGN.md section 5).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
from pathlib import Path

import numpy as np

from .. import config
from . import refprice, registry, volsurf
from .context import LabContext
from .util import clip, read_json, write_json

log = logging.getLogger("weathervane.lab")
GROUP_FILES = (registry.MODERN, registry.LEGACY, registry.VIDEO)
CROSSCHECK_KEYS = ("finance_python", "gs_quant", "pynance", "ffn")
STATUS_ORDER = {"ok": 0, "failed": 1, "skipped": 2}


# ----------------------------------------------------------------------------------------------------------------
# reading results


def load_groups(out: Path) -> dict[str, dict]:
    """{group: payload} from out/lab/<group>.json for the groups that were run."""
    groups = {}
    for g in GROUP_FILES:
        data = read_json(out / f"{g}.json")
        if isinstance(data, dict) and isinstance(data.get("adapters"), dict):
            groups[g] = data
    return groups


def adapter_record(groups: dict, key: str) -> dict | None:
    for payload in groups.values():
        rec = payload.get("adapters", {}).get(key)
        if rec is not None:
            return rec
    return None


def adapter_results(groups: dict, key: str) -> dict:
    rec = adapter_record(groups, key)
    return (rec or {}).get("results") or {}


def fmt(x, digits: int = 4, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%" if pct else f"{x:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return lines + [""]


# ----------------------------------------------------------------------------------------------------------------
# 1. library status


def status_rows(groups: dict) -> list[dict]:
    rows = []
    for e in registry.ADAPTERS:
        if e.is_core:
            rows.append({"repo": e.repo, "key": e.key, "group": "core", "status": "core", "reason": e.core, "seconds": None,
                         "version": "", "role": e.role})
            continue
        rec = adapter_record(groups, e.key)
        if rec is None:
            rows.append({"repo": e.repo, "key": e.key, "group": e.group, "status": "not run", "reason": f"{e.group} job did not produce results",
                         "seconds": None, "version": "", "role": e.role})
        else:
            res = rec.get("results") or {}
            rows.append({"repo": e.repo, "key": e.key, "group": e.group, "status": rec.get("status", "failed"),
                         "reason": rec.get("reason", ""), "seconds": rec.get("seconds"),
                         "version": str(res.get("version") or ""), "role": e.role})
    return rows


def status_section(groups: dict) -> list[str]:
    rows = status_rows(groups)
    table = [[r["repo"], r["group"], r["status"], r["version"], fmt(r["seconds"], 1) if r["seconds"] is not None else "",
              clip(r["reason"], 110) or clip(r["role"], 110)] for r in rows]
    return ["## 1. Library status", ""] + md_table(["Repo", "Group", "Status", "Version", "Seconds", "Reason or role"], table)


# ----------------------------------------------------------------------------------------------------------------
# 2. pricing agreement


def reference_prices(groups: dict, contracts) -> tuple[str, dict]:
    """QuantLib prices from the pyql adapter, or the numpy reference when that adapter did not run."""
    res = adapter_results(groups, "pyql")
    prices = res.get("prices")
    if prices and prices.get("european"):
        return f"QuantLib ({res.get('backend', 'pyql')})", prices
    eu = {c.id: refprice.bs_price(c.cp, c.S, c.K, c.T, c.r, c.q, c.sigma) for c in contracts}
    am = {c.id: refprice.crr_american(c.cp, c.S, c.K, c.T, c.r, c.q, c.sigma, 800) for c in contracts}
    return "numpy reference (BSM and CRR 800 steps; QuantLib results missing)", {"european": eu, "american": am}


def pricing_tables(ctx: LabContext, groups: dict) -> dict:
    tol = ctx.cfg["tolerances"]
    ref_name, ref = reference_prices(groups, ctx.contracts)
    libs: dict[str, dict] = {}
    for e in registry.ADAPTERS:
        if not e.pricing:
            continue
        rec = adapter_record(groups, e.key)
        prices = ((rec or {}).get("results") or {}).get("prices")
        if rec and rec.get("status") == "ok" and prices:
            libs[e.key] = {"prices": prices, "known_issue": ((rec.get("results") or {}).get("known_issue") or "")}
    deviations: dict[str, dict] = {}
    for key, lib in libs.items():
        for style, series in lib["prices"].items():
            base = style.split(":")[0]
            ref_series = ref.get(base) or {}
            worst, worst_id, flagged = 0.0, "", []
            for cid, p in series.items():
                r = ref_series.get(cid)
                if p is None or r is None:
                    continue
                gap = abs(p - r)
                if gap / max(abs(r), tol["agreement_abs_floor"]) > tol["agreement_flag_rel"]:
                    flagged.append(cid)
                if gap > worst:
                    worst, worst_id = gap, cid
            deviations.setdefault(key, {})[style] = {"max_abs": worst, "worst": worst_id, "flagged": flagged}
    return {"reference": ref_name, "ref_prices": ref, "libs": libs, "deviations": deviations,
            "contracts": [c.id for c in ctx.contracts], "flag_rel": tol["agreement_flag_rel"]}


def pricing_section(ctx: LabContext, tables: dict) -> list[str]:
    lines = ["## 2. Pricing agreement", "",
             f"{len(tables['contracts'])} test contracts: spot {ctx.cfg['test_set']['spot']:g}, flat vol {ctx.cfg['test_set']['sigma']:.0%}, "
             f"r {ctx.cfg['test_set']['r']:.0%}, q {ctx.cfg['test_set']['q']:.0%}. Reference: {tables['reference']}. "
             f"A price is marked * when it is more than {tables['flag_rel']:.0%} away from the reference.", ""]
    if not tables["libs"]:
        return lines + ["No pricing library produced results.", ""]
    for base, title in (("european", "European"), ("american", "American")):
        cols = []
        for key, lib in tables["libs"].items():
            for style in lib["prices"]:
                if style.split(":")[0] == base:
                    cols.append((key, style))
        if not cols:
            continue
        head = ["Contract", "Reference"] + [k if s == base else f"{k} (MC)" for k, s in cols]
        rows = []
        for cid in tables["contracts"]:
            ref_p = (tables["ref_prices"].get(base) or {}).get(cid)
            row = [cid, fmt(ref_p)]
            for key, style in cols:
                p = tables["libs"][key]["prices"][style].get(cid)
                mark = "*" if cid in tables["deviations"].get(key, {}).get(style, {}).get("flagged", []) else ""
                row.append(fmt(p) + mark if p is not None else "n/a")
            rows.append(row)
        lines += [f"### {title}", ""] + md_table(head, rows)
    summary = []
    for key, styles in tables["deviations"].items():
        for style, d in styles.items():
            summary.append([key, style, fmt(d["max_abs"], 5), d["worst"] or "", ", ".join(d["flagged"]) or "none"])
    lines += ["### Deviation from the reference", ""] + md_table(["Library", "Style", "Max abs deviation", "Worst contract", "Flagged (>1%)"], summary)
    issues = [f"- {k}: {v['known_issue']}" for k, v in tables["libs"].items() if v.get("known_issue")]
    if issues:
        lines += ["Known library issues found by this comparison:", ""] + issues + [""]
    return lines


# ----------------------------------------------------------------------------------------------------------------
# 3. volatility


def vol_section(ctx: LabContext, groups: dict, write_history: bool = True) -> dict:
    chains = ctx.chains()
    if chains.empty:
        return {}
    civ = volsurf.load_chain_iv(ctx.out, chains)
    sabr = adapter_results(groups, "pysabr").get("sabr") or {}
    history = None if write_history else volsurf.read_history()
    return volsurf.dashboard(civ, sabr, ctx.today.isoformat(), history)


def vol_lines(ctx: LabContext, vol: dict) -> list[str]:
    meta = ctx.chains_meta()
    lines = ["## 3. Volatility dashboard", ""]
    if not vol:
        failed = {t: i.get("error") for t, i in (meta.get("tickers") or {}).items() if i.get("status") != "ok"}
        why = meta.get("error") or (clip("; ".join(f"{t}: {e}" for t, e in list(failed.items())[:3]), 200) if failed else "no chains saved")
        return lines + [f"Option chains were not available this run ({why}).", ""]
    lines += [f"Chains from {meta.get('source', 'unknown')} as of {meta.get('asof', '?')}; rate {fmt(ctx.rate, 2, pct=True)} ({ctx.rate_source}). "
              "ATM IV is interpolated in total variance to 30 days; term slope is 90d minus 30d; risk reversal is call IV minus put IV at 25 delta (30d); "
              "implied move is one standard deviation over 7 calendar days.", ""]
    rows = []
    for t, r in vol.items():
        if "atm_iv_30d" not in r:
            rows.append([t, fmt(r.get("spot"), 2), r.get("note", ""), "", "", "", "", "", ""])
            continue
        s = r.get("sabr") or {}
        rank = "n/a (%d days of history)" % r.get("history_days", 0) if r.get("iv_rank") is None else f"{r['iv_rank']:.0f} / {r['iv_percentile']:.0f}"
        rows.append([t, fmt(r["spot"], 2), fmt(r["atm_iv_30d"], 1, pct=True), rank, fmt(r.get("rr25_30d"), 1, pct=True) if r.get("rr25_30d") is not None else "n/a",
                     fmt(r["term_slope"], 1, pct=True), fmt(s.get("rho"), 2), fmt(s.get("nu"), 2), f"{r['implied_move_1w_pct']:.1f}%"])
    return lines + md_table(["Ticker", "Spot", "ATM IV 30d", "IV rank / pct", "25d RR", "Term slope", "SABR rho", "SABR nu", "1w move"], rows)


# ----------------------------------------------------------------------------------------------------------------
# 4. COT


def cot_section(groups: dict) -> tuple[list[str], dict]:
    lines = ["## 4. CFTC positioning", ""]
    rec = adapter_record(groups, "openbb_odp")
    if rec is None:
        return lines + ["The OpenBB adapter did not run.", ""], {}
    markets = (rec.get("results") or {}).get("markets") or {}
    if not markets:
        return lines + [f"No COT data: {rec.get('reason') or rec.get('status')}.", ""], {}
    rows = []
    for key, m in markets.items():
        if m.get("status") == "ok":
            pct = f"{m['percentile']:.0f}" if m.get("percentile") is not None else "n/a"
            rows.append([m["label"], m["group"], m["date"], f"{m['net']:+,.0f}", pct, f"{m['change_1w']:+,.0f}" if m.get("change_1w") is not None else "n/a",
                         f"{m['weeks']}"])
        else:
            rows.append([m["label"], m.get("report", ""), "failed", clip(m.get("error", ""), 60), "", "", ""])
    lines += ["Net = long minus short contracts of the speculative group (managed money in the disaggregated reports, leveraged funds in the TFF reports "
              "that have no managed-money category). Percentile is over the lookback window.", ""]
    return lines + md_table(["Market", "Group", "As of", "Net position", "3y percentile", "1w change", "Weeks"], rows), markets


# ----------------------------------------------------------------------------------------------------------------
# 5. indicator cross-check


def crosscheck_section(groups: dict) -> tuple[list[str], list[str]]:
    lines = ["## 5. Indicator cross-check", ""]
    rows, flagged = [], []
    for key in CROSSCHECK_KEYS:
        rec = adapter_record(groups, key)
        if rec is None or rec.get("status") != "ok":
            rows.append([key, "", "", "", "", (rec.get("status", "not run") + ": " + clip(rec.get("reason", ""), 60)) if rec else "not run"])
            continue
        res = rec["results"]
        merged: dict[str, dict] = {}
        for inst, checks in (res.get("instruments") or {}).items():
            for name, r in checks.items():
                m = merged.setdefault(name, {"max_abs": 0.0, "max_rel": 0.0, "n": 0, "match": r.get("definitions_match", True), "note": r.get("note", ""),
                                            "error": r.get("error"), "flagged": False})
                if r.get("max_abs") is not None:
                    m["max_abs"], m["max_rel"] = max(m["max_abs"], r["max_abs"]), max(m["max_rel"], r["max_rel"])
                    m["n"] += r.get("n", 0)
                m["flagged"] = m["flagged"] or bool(r.get("flagged"))
        for name, m in merged.items():
            if m["error"] and not m["n"]:
                rows.append([key, name, "", "", "", "error: " + clip(m["error"], 60)])
                continue
            state = "FLAG" if m["flagged"] else ("differs by definition" if not m["match"] else "ok")
            rows.append([key, name, f"{m['max_abs']:.3g}", f"{m['max_rel']:.3g}", str(m["n"]), state])
            if m["flagged"]:
                flagged.append(f"{key}:{name}")
    lines += ["Largest absolute and relative difference over the compared instruments (last bars). Core series are converted to each library's convention where "
              "they differ (sample versus population stdev). Flag threshold: relative difference above 1e-6 where definitions match.", ""]
    return lines + md_table(["Library", "Check", "Max abs diff", "Max rel diff", "Points", "Result"], rows), flagged


# ----------------------------------------------------------------------------------------------------------------
# 6. sleeve weights


def weights_section(groups: dict) -> list[str]:
    lines = ["## 6. Sleeve weight suggestions", ""]
    rec = adapter_record(groups, "quantpy")
    if rec is None:
        return lines + ["The QuantPy adapter did not run.", ""]
    portfolios = (rec.get("results") or {}).get("portfolios") or {}
    if not portfolios:
        return lines + [f"No weights: {rec.get('reason') or rec.get('status')}.", ""]
    lines += ["Suggestions only. The engine sizes every sleeve at equal risk. Inverse vol is proportional to 1/stdev of daily R; MVO is QuantPy's tangency "
              "formula on summed returns, clipped to long-only and renormalised; min variance is its 'characteristic' weighting.", ""]
    for pid, p in portfolios.items():
        if p.get("error"):
            lines += [f"**{pid}**: {p['error']}", ""]
            continue
        lines += [f"**{pid}** (champion {p.get('champion')}, {p.get('days')} days, {p.get('unit')} streams, data: {p.get('data_source')})", ""]
        if "inverse_vol" not in p:
            lines += [p.get("note", "no optimisation"), ""]
            continue
        rows = [[s, fmt(p["equal"][s], 3), fmt(p["inverse_vol"][s], 3), fmt(p["mvo_long_only"][s], 3), fmt(p["min_variance"][s], 3),
                 fmt(p["stats"][s]["sharpe"], 2)] for s in p["streams"]]
        lines += md_table(["Stream", "Equal risk", "Inverse vol", "MVO long-only", "Min variance", "Sharpe"], rows)
    return lines


# ----------------------------------------------------------------------------------------------------------------
# 7. news and video


def news_section(groups: dict) -> tuple[list[str], list[dict]]:
    lines = ["## 7. News headlines", ""]
    rec = adapter_record(groups, "camofox")
    if rec is None:
        return lines + ["The news fetch did not run.", ""], []
    if rec.get("status") != "ok":
        return lines + [f"Not fetched: {rec.get('reason') or rec.get('status')}.", ""], []
    items = (rec.get("results") or {}).get("headlines") or []
    if not items:
        return lines + ["No headlines passed the allowlist and robots.txt checks.", ""], []
    lines += ["Title and link only, from allowlisted domains, fetched under robots.txt and a 5 second per-domain delay.", ""]
    lines += [f"- {i['ticker']}: [{i['title']}]({i['url']}) ({i['domain']})" for i in items] + [""]
    return lines, items


def video_section(groups: dict) -> list[str]:
    lines = ["## Weekly video", ""]
    rec = adapter_record(groups, "hyperframes")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if rec and rec.get("status") == "ok" and (rec.get("results") or {}).get("video"):
        link = f"https://raw.githubusercontent.com/{repo}/charts/lab/weekly.mp4" if repo else "out/video/weekly.mp4"
        return lines + [f"Rendered: {link}", ""]
    reason = rec.get("reason") or rec.get("status") if rec else "the video job did not run"
    return lines + [f"No video this week ({reason}).", ""]


# ----------------------------------------------------------------------------------------------------------------
# context.json


def regime_block(ctx: LabContext) -> dict | None:
    """Current HMM regime of the reference series, only when real cached data exists."""
    try:
        from .. import regime
        ref_id = next(iter(config.load_portfolios().values())).regime_reference
        df = ctx.prices([ref_id])[ref_id]
        if ctx.price_source.get(ref_id) != "cache":
            return None
        labels = regime.fit(df).label(df)
        return {"reference": ref_id, "label": str(labels.iloc[-1]), "date": str(df.index[-1].date())}
    except Exception as exc:
        log.warning("regime context unavailable: %s", clip(exc))
        return None


def build_context_json(ctx: LabContext, vol: dict, markets: dict) -> dict:
    volatility = {}
    for t, r in vol.items():
        if "atm_iv_30d" not in r:
            continue
        s = r.get("sabr") or {}
        volatility[t] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in {
            "atm_iv_30d": r["atm_iv_30d"], "iv_rank": r.get("iv_rank"), "iv_percentile": r.get("iv_percentile"), "rr25_30d": r.get("rr25_30d"),
            "term_slope": r["term_slope"], "implied_move_1w_pct": r["implied_move_1w_pct"], "sabr_rho": s.get("rho"), "sabr_nu": s.get("nu")}.items()
            if v is not None}
    positioning = {k: {"label": m["label"], "group": m["group"], "date": m["date"], "net": round(m["net"]), "percentile_3y": None if m.get("percentile") is None
                       else round(m["percentile"], 1)} for k, m in markets.items() if m.get("status") == "ok"}
    out = {"asof": ctx.today.isoformat(), "kind": "research context from the weathervane lab: informational, never a trading rule",
           "volatility": volatility, "positioning": positioning}
    regime = regime_block(ctx)
    if regime:
        out["regime"] = regime
    return out


# ----------------------------------------------------------------------------------------------------------------
# assembly


def summary_line(rows: list[dict], flagged_prices: int, flagged_checks: list[str]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = [f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: STATUS_ORDER.get(kv[0], 9))]
    return (f"Libraries: {', '.join(parts)}. Pricing cells flagged: {flagged_prices}. Indicator checks flagged: {len(flagged_checks)}"
            + (f" ({', '.join(flagged_checks)})" if flagged_checks else "") + ".")


def build(ctx: LabContext, groups: dict | None = None, write_history: bool = True) -> tuple[str, dict]:
    """(markdown, context.json payload)."""
    groups = load_groups(ctx.out) if groups is None else groups
    rows = status_rows(groups)
    tables = pricing_tables(ctx, groups)
    vol = vol_section(ctx, groups, write_history)
    cot_lines, markets = cot_section(groups)
    cross_lines, flagged_checks = crosscheck_section(groups)
    news_lines, _ = news_section(groups)
    flagged_prices = sum(len(d["flagged"]) for styles in tables["deviations"].values() for d in styles.values())
    sources = [f"price data: {', '.join(f'{k} {v}' for k, v in ctx.price_source.items())}" if ctx.price_source else "price data: not needed"]
    head = [f"# Weathervane lab report {ctx.today.isoformat()}", "",
            "Research context only. Nothing here changes trading rules, sizing or promotion.", "", summary_line(rows, flagged_prices, flagged_checks), ""]
    body = (status_section(groups) + pricing_section(ctx, tables) + vol_lines(ctx, vol) + cot_lines + cross_lines + weights_section(groups)
            + news_lines + video_section(groups))
    md = "\n".join(head + body).rstrip() + "\n"
    return md, build_context_json(ctx, vol, markets)


def write(ctx: LabContext, groups: dict | None = None) -> dict:
    md, context = build(ctx, groups)
    report_path = config.path("reports", "lab", f"{ctx.today.isoformat()}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    write_json(config.path("results", "context.json"), context)
    return {"report": str(report_path), "context": str(config.path("results", "context.json")), "bytes": len(md)}
