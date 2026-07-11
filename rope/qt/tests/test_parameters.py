"""Roundtrip + schema-coverage tests for rope/qt/parameters.py.

Run:  venv\\Scripts\\python.exe -m rope.qt.tests.test_parameters
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from rope.qt import parameters as P
from rope.qt import parameters_migration as M


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_every_param_resolves_schema():
    for p in P.PARAMETERS:
        try:
            if p.kind == "slider":
                _ = p.default; _ = p.min; _ = p.max; _ = p.inc
            elif p.kind == "switch":
                _ = p.default
            elif p.kind == "select":
                d = p.default
                modes = p.modes
                if d not in modes:
                    fail(f"select {p.name}: default {d!r} not in modes {modes!r}")
            elif p.kind == "entry":
                _ = p.default
            elif p.kind == "button":
                _ = p.default
                _ = p.display
                _ = p.text
        except KeyError as exc:
            fail(f"{p.kind} {p.name!r}: missing DEFAULT_DATA key {exc}")
    print(f"  ok  every param ({len(P.PARAMETERS)}) resolves its schema fields")


def test_no_duplicate_names():
    names = [p.name for p in P.PARAMETERS]
    dupes = sorted(set(n for n in names if names.count(n) > 1))
    if dupes:
        fail(f"duplicate param names: {dupes}")
    print(f"  ok  no duplicate names ({len(names)} unique)")


def test_default_values_filter_by_scope():
    parameter_scope = P.default_values(scope="parameter")
    control_scope = P.default_values(scope="control")
    assert "RestorerSwitch" in parameter_scope, "RestorerSwitch missing from parameter scope"
    assert "CaptureFPSSlider" in control_scope, "CaptureFPSSlider missing from control scope"
    assert "RestorerSwitch" not in control_scope, "RestorerSwitch leaked into control scope"
    print(
        f"  ok  default_values: {len(parameter_scope)} parameter, "
        f"{len(control_scope)} control"
    )


def test_legacy_roundtrip():
    # Synthesize a legacy-style file with a mix of known + unknown keys
    # and slightly wrong types (strings where floats expected).
    legacy = {
        "RestorerSwitch": True,
        "RestorerSlider": "42",
        "RestorerTypeTextSel": "CF",
        "BogusKeyFromOldVersion": "ignore_me",
        "DetectTypeTextSel": "SCRDF",
        "CaptureFPSSlider": 25,
        "MergeTextSel": "NotAMode",   # invalid mode -> falls back to default
    }
    with tempfile.TemporaryDirectory() as tmp:
        legacy_path = Path(tmp) / "saved_parameters.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = M.load(legacy_path)

    # Unknown key dropped
    assert "BogusKeyFromOldVersion" not in loaded, "unknown key leaked through load()"
    # Switch coerced to bool
    assert loaded["RestorerSwitch"] is True, f"got {loaded['RestorerSwitch']!r}"
    # Slider coerced from str to float
    assert isinstance(loaded["RestorerSlider"], float) and loaded["RestorerSlider"] == 42.0, (
        f"slider not coerced: {loaded['RestorerSlider']!r}"
    )
    # Select with valid mode preserved
    assert loaded["DetectTypeTextSel"] == "SCRDF", f"got {loaded['DetectTypeTextSel']!r}"
    # Select with invalid mode falls back to default
    merge_default = P.PARAMETER_BY_NAME["MergeTextSel"].default
    assert loaded["MergeTextSel"] == merge_default, (
        f"invalid mode fallback failed: got {loaded['MergeTextSel']!r}, want {merge_default!r}"
    )
    print(f"  ok  legacy roundtrip: loaded {len(loaded)} keys, dropped 1 unknown, coerced types")


def test_save_filters_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        M.save({"RestorerSwitch": True, "InventedKey": 1}, path)
        round_tripped = json.loads(path.read_text())
        assert "RestorerSwitch" in round_tripped
        assert "InventedKey" not in round_tripped, "save() did not filter unknown key"
    print("  ok  save() drops unknown keys")


def main() -> int:
    print("rope.qt.parameters smoke tests")
    test_every_param_resolves_schema()
    test_no_duplicate_names()
    test_default_values_filter_by_scope()
    test_legacy_roundtrip()
    test_save_filters_unknown()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
