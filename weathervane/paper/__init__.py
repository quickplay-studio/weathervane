"""Paper-trading connectors. Practice and paper base URLs are hardcoded, anything else raises."""
from __future__ import annotations

from dataclasses import asdict, dataclass

ENTRY_ACTIONS = ("ENTER_LONG", "ENTER_SHORT")
REVERSE_ACTIONS = ("REVERSE_TO_LONG", "REVERSE_TO_SHORT")
CATCH_UP_BARS = 1     # a held position this young with no broker position is opened late (missed or rejected order)


@dataclass
class Order:
    """One intended broker action and, after execution, what happened to it."""

    broker: str
    kind: str                 # open | close | modify | skip
    symbol: str
    instrument: str
    sleeve: str
    side: str = ""            # long | short
    units: float = 0.0
    stop: float | None = None
    target: float | None = None
    ref: str = ""             # broker trade or order id to act on
    tag: str = ""
    note: str = ""
    status: str = "planned"   # planned | dry_run | sent | error | skipped
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def wants_open(row: dict, has_position: bool) -> bool:
    """True when the row asks for a new position, or a young position is missing at the broker."""
    if row["action"] in ENTRY_ACTIONS or row["action"] in REVERSE_ACTIONS:
        return True
    return (not has_position and row["action"] in ("HOLD", "MOVE_STOP") and row.get("bars_in_trade", 99) <= CATCH_UP_BARS)


def entry_side(row: dict) -> str:
    """Side of the position the row wants to open."""
    if row["action"] in ENTRY_ACTIONS or row["action"] in REVERSE_ACTIONS:
        return "long" if row["action"].endswith("LONG") else "short"
    return row["side"]
