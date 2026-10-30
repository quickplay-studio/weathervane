"""Weathervane lab: an isolated, optional layer that runs every researched finance library as an adapter.

The lab never feeds trading rules (they must stay Pine-portable). Each adapter catches its own import and runtime
failures and reports them, so nothing here can break `signals` or `research`. See docs/LAB.md.
"""
from __future__ import annotations

__all__ = ["registry", "context", "report"]
