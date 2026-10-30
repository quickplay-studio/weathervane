"""The repo map of docs/LAB.md section 1 as data: 22 rows, one per researched repository.

Core rows point at the engine module that already uses the idea; lab rows name the adapter module that runs them. The
order inside a group is the run order (vollib runs before pysabr because the SABR fit reads the implied vols it saves).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from .adapters import base
from .context import LabContext

CORE, MODERN, LEGACY, VIDEO = "core", "modern", "legacy", "video"
GROUPS = (MODERN, LEGACY, VIDEO)


@dataclass(frozen=True)
class Entry:
    key: str                  # short id used in out/lab/<group>.json
    repo: str                 # owner/name as in LAB.md
    group: str                # core | modern | legacy | video
    role: str                 # what it does in Weathervane
    module: str = ""          # dotted module with run(ctx) for lab rows
    core: str = ""            # engine module(s) for core rows, or the core counterpart of a lab row
    install: str = ""         # install hint
    pricing: bool = False     # contributes a column to the pricing agreement table
    timeout_s: int = 120      # adapters return within 120 s; the video render is the exception
    python: str = ""          # interpreter requirement, when one matters
    note: str = ""

    @property
    def is_core(self) -> bool:
        return self.group == CORE


A = "weathervane.lab.adapters."

ADAPTERS: tuple[Entry, ...] = (
    Entry("autoresearch", "karpathy/autoresearch", CORE, "Nightly keep-or-revert experiment loop (a pattern, not a dependency)",
          core="weathervane/researcher.py + program.md"),
    Entry("tradingagents", "TauricResearch/TradingAgents", CORE, "Daily company research reports", core="weathervane/companies.py"),
    Entry("openbb_odp", "OpenBB-finance/OpenBB", MODERN, "CFTC COT positioning (managed money net, 3-year percentile)",
          module=A + "openbb_odp", core="data fallback source (not in weathervane/data.py yet)",
          install="pip install openbb-core openbb-cftc", note="provider layer only; optional CFTC_APP_TOKEN"),
    Entry("hmmlearn", "hmmlearn/hmmlearn", CORE, "HMM regime labels for the OOS gate and chart shading", core="weathervane/regime.py"),
    Entry("ffn", "pmorissette/ffn", MODERN, "Performance statistics and return conventions cross-checked against core metrics",
          module=A + "ffn", core="weathervane/reports.py (planned) + weathervane/metrics.py", install="pip install ffn"),
    Entry("pyql", "enthought/pyql", LEGACY, "QuantLib pricing of the test options; the QuantLib wheel is the fallback and the reference",
          module=A + "pyql", install="apt libquantlib0-dev swig; pip install pyql (falls back to pip install QuantLib)", pricing=True,
          python="3.10"),
    Entry("vollib", "vollib/vollib", MODERN, "Implied volatility and Greeks for every option in the fetched chains",
          module=A + "vollib", install="pip install vollib py_vollib", pricing=True),
    Entry("quantpy", "jsmidt/QuantPy", LEGACY, "Mean-variance and inverse-volatility weights for sleeves (suggestion only)",
          module=A + "quantpy", install="pip install git+https://github.com/jsmidt/QuantPy", python="3.10",
          note="imports the removed pandas.io.data; weights use its formulas either way"),
    Entry("finance_python", "alpha-miner/Finance-Python", LEGACY, "Independent rolling operators to cross-check core indicators",
          module=A + "finance_python", install="pip install git+https://github.com/alpha-miner/Finance-Python@3e33d3e7 (cython<3, no build isolation)",
          python="3.10"),
    Entry("pynance", "GriffinAustin/pynance", LEGACY, "Independent returns, moving-average and volatility cross-check",
          module=A + "pynance", install="pip install pynance", python="3.10"),
    Entry("pysabr", "ynouri/pysabr", MODERN, "SABR smile fit per expiry: skew (rho) and vol-of-vol (nu)",
          module=A + "pysabr", install="pip install pysabr", pricing=True),
    Entry("financepy", "domokane/FinancePy", MODERN, "Equity option valuation cross-check (Black-Scholes and CRR tree)",
          module=A + "financepy", install="pip install --no-deps financepy (its pins fight numpy 2)", pricing=True),
    Entry("gs_quant", "goldmansachs/gs-quant", MODERN, "Offline gs_quant.timeseries (volatility, z-scores, returns) as cross-checks",
          module=A + "gs_quant", install="pip install gs-quant", note="Marquee only with GS_CLIENT_ID and GS_CLIENT_SECRET"),
    Entry("willowtree", "federicomariamassari/willowtree", MODERN, "Willow-tree lattice pricing of American options vs QuantLib",
          module=A + "willowtree", install="git clone at the pinned commit into data/lab/src/willowtree", pricing=True,
          note="not the PyPI package of the same name; needs numpy/scipy shims"),
    Entry("financial_engineering", "federicomariamassari/financial-engineering", MODERN,
          "Monte Carlo (antithetic, control variate) cross-check and Merton jump diffusion",
          module=A + "financial_engineering", install="git clone at the pinned commit into data/lab/src/financial-engineering", pricing=True,
          note="repo holds only the Merton jump-diffusion module; the variance reduction is reimplemented"),
    Entry("optlib", "dbrojas/optlib", LEGACY, "optlib.gbs pricing (Merton, Bjerksund-Stensland); optlib.api is never used",
          module=A + "optlib", install="pip install git+https://github.com/dbrojas/optlib (needs scipy<1.12)", pricing=True, python="3.10"),
    Entry("tf_quant_finance", "google/tf-quant-finance", LEGACY, "Vectorised Black-Scholes and CRR tree pricing of the test set (archived)",
          module=A + "tf_quant_finance", install="pip install tf-quant-finance tensorflow", pricing=True, python="3.10"),
    Entry("qfin", "RomanMichaelPaolucci/Q-Fin", MODERN, "Closed-form and Monte Carlo option pricing cross-check",
          module=A + "qfin", install="pip install qfin", pricing=True),
    Entry("quantsbin", "quantsbin/Quantsbin", MODERN, "Vanilla pricing, Greeks and payoff plots", module=A + "quantsbin",
          install="pip install quantsbin", pricing=True, python="3.12+"),
    Entry("finoptions", "bbcho/finoptions-dev", MODERN, "fOptions pricing incl. tree and Monte Carlo, Asian illustration",
          module=A + "finoptions", install="pip install finoptions", pricing=True),
    Entry("camofox", "jo-inc/camofox-browser", MODERN, "Headline fetch (title and URL) from allowlisted pages",
          module="weathervane.lab.fetch.camofox", core="lab/fetch/camofox.py", install="npx @askjo/camofox-browser (news.enabled only)",
          note="off by default"),
    Entry("hyperframes", "heygen-com/hyperframes", VIDEO, "Weekly recap video rendered from an HTML template",
          module="weathervane.lab.video.render", core="lab/video/", install="npx hyperframes@<pinned> (Node 22+, FFmpeg)", timeout_s=1800),
)

BY_KEY = {e.key: e for e in ADAPTERS}


def repos() -> list[str]:
    return [e.repo for e in ADAPTERS]


def group_entries(group: str) -> list[Entry]:
    """Entries a `run --group` executes. Lab rows only; core rows have nothing to run."""
    return [e for e in ADAPTERS if e.group == group and e.module]


def _call(module: str):
    def fn(ctx: LabContext) -> dict:
        return importlib.import_module(module).run(ctx)
    return fn


def run_entry(entry: Entry, ctx: LabContext) -> dict:
    """Run one adapter inside the guard (the import itself is guarded too). Never raises."""
    return base.run_guarded(_call(entry.module), ctx, timeout=entry.timeout_s, name=entry.key)
