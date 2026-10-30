"""One module per library. Each exposes run(ctx) -> dict (the `results` payload) and raises Skipped to opt out.

Imports of the third-party library happen inside run() so that a missing or broken package is reported by the guard in
base.run_guarded instead of breaking the lab.
"""
