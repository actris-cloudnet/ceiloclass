"""Tests for instrument discovery and selection (no network)."""

import builtins
from types import SimpleNamespace

import pytest

from ceiloclass import cli
from ceiloclass.download import LidarSource, _group_sources, _label


def _inst(instrument_id, pid, name, serial=None):
    return SimpleNamespace(
        instrument_id=instrument_id,
        pid=pid,
        uuid=f"uuid-{pid}",
        name=name,
        serial_number=serial,
    )


def _meta(filename, instrument):
    return SimpleNamespace(filename=filename, instrument=instrument)


class _Parser:
    """Minimal stand-in for argparse.ArgumentParser used by _select_source."""

    prog = "ceiloclass"

    def error(self, message):
        raise SystemExit(f"error: {message}")

    def exit(self, status=0, message=""):
        raise SystemExit(message.strip())


# --- grouping ---------------------------------------------------------------


def test_group_sources_one_per_instrument():
    md = [
        _meta("a.nc", _inst("cl61d", "pid-1", "Vaisala CL61")),
        _meta("b1.nc", _inst("chm15kx", "pid-2", "Lufft CHM 15k")),
        _meta("b2.nc", _inst("chm15kx", "pid-2", "Lufft CHM 15k")),
    ]
    sources = _group_sources(md, raw=True)
    assert [s.reader for s in sources] == ["cl61", "chm15k"]
    assert [len(s.metadata) for s in sources] == [1, 2]


def test_group_sources_splits_same_model_by_pid():
    md = [
        _meta("x.nc", _inst("cl51", "pid-A", "Vaisala CL51", "SN-A")),
        _meta("y.nc", _inst("cl51", "pid-B", "Vaisala CL51", "SN-B")),
    ]
    assert len(_group_sources(md, raw=True)) == 2


def test_group_sources_lidar_product_has_no_reader():
    md = [_meta("L.nc", _inst("cl61d", "pid-1", "Vaisala CL61"))]
    assert _group_sources(md, raw=False)[0].reader is None


def test_label_includes_serial_and_file_count():
    inst = _inst("cl51", "pid-A", "Vaisala CL51", "42")
    assert _label(inst, 3) == "cl51 — Vaisala CL51, SN 42 (3 files)"
    assert _label(inst, 1).endswith("(1 file)")


# --- selection --------------------------------------------------------------


def test_select_single_source_is_automatic():
    sources = [LidarSource("cl61", "cl61d (1 file)", [object()])]
    assert cli._select_source(sources, _Parser()) is sources[0]


def test_select_many_non_interactive_errors(monkeypatch):
    sources = [
        LidarSource("cl51", "cl51 — A (1 file)", [object()]),
        LidarSource("cl61", "cl61d — B (1 file)", [object()]),
    ]
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="several instruments available"):
        cli._select_source(sources, _Parser())


def test_select_many_interactive_prompts(monkeypatch):
    sources = [
        LidarSource("cl51", "cl51 — A (1 file)", [object()]),
        LidarSource("cl61", "cl61d — B (1 file)", [object()]),
    ]
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["0", "nope", "2"])  # invalid, invalid, then valid
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    assert cli._select_source(sources, _Parser()).reader == "cl61"
