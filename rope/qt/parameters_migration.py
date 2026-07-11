"""Load / save helpers for legacy and current saved_parameters.json files.

The legacy GUI (rope/GUI.py) wrote a flat {widget_name: value} JSON. The new
schema in rope/qt/parameters.py keeps the same on-disk format, so
"migration" really means "validate the keys are known, drop the rest".

Public surface:

    load(path) -> dict[str, Any]
        Reads the JSON, keeps only keys present in PARAMETER_BY_NAME, and
        coerces values to the appropriate Python type using the param's
        kind. Returns a dict ready to feed into a ParameterValues store.

    save(values, path) -> None
        Writes {name: value} JSON. Keys not present in PARAMETER_BY_NAME
        are dropped silently (defensive — same as legacy load behavior).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rope.qt.parameters import PARAMETER_BY_NAME, SelectParam


def _coerce(name: str, value: Any) -> Any:
    """Coerce a JSON-loaded value to the kind the param expects.

    JSON gives us str/int/float/bool/list. The widgets, however, expect:
    - slider: float (current GUI stores ints sometimes, floats elsewhere)
    - switch: bool
    - select: str (must be in modes)
    - entry: str
    """
    p = PARAMETER_BY_NAME.get(name)
    if p is None:
        return value
    if p.kind == "switch":
        return bool(value)
    if p.kind == "slider":
        try:
            return float(value)
        except (TypeError, ValueError):
            return p.default
    if p.kind == "select":
        # Must match a known mode, else fall back to default.
        if isinstance(p, SelectParam) and value in p.modes:
            return value
        return p.default
    if p.kind == "entry":
        return "" if value is None else str(value)
    return value


def load(path: str | Path) -> dict[str, Any]:
    """Read a saved_parameters.json and return a sanitized {name: value} dict.

    Unknown keys are dropped. Type-coerced values are returned. If the file
    doesn't exist, returns {}.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for name, value in raw.items():
        if name in PARAMETER_BY_NAME:
            out[name] = _coerce(name, value)
    return out


def save(values: dict[str, Any], path: str | Path) -> None:
    """Write {name: value} JSON. Drops unknown keys defensively."""
    filtered = {k: v for k, v in values.items() if k in PARAMETER_BY_NAME}
    Path(path).write_text(json.dumps(filtered, indent=2), encoding="utf-8")
