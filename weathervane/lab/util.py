"""Small helpers shared across the lab: JSON-safe conversion, output paths, text clipping."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config


def out_dir(*parts: str) -> Path:
    """out/lab[/parts]. Adapters write only below this folder. A last part with a dot is treated as a file name."""
    p = config.path("out", "lab", *parts)
    (p.parent if parts and "." in parts[-1] else p).mkdir(parents=True, exist_ok=True)
    return p


def clip(text: object, n: int = 200) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def jsonable(obj: Any) -> Any:
    """Convert numpy/pandas/Path/NaN values into plain JSON types (NaN and inf become None)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, Path):
        return str(obj)
    return clip(repr(obj), 300)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def today_utc() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()
