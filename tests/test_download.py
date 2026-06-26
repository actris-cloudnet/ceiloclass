"""Tests for instrument discovery and selection (no network)."""

import builtins
from types import SimpleNamespace

import pytest

from ceiloclass import cli, download
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


def _harmonized_client(monkeypatch):
    """Patch download.APIClient with one lidar + one doppler-lidar product."""
    queried = []
    cl61 = _meta("L.nc", _inst("cl61d", "p-cl", "Vaisala CL61"))
    halo = _meta("D.nc", _inst("halo-doppler-lidar", "p-halo", "HALO StreamLine"))

    class FakeClient:
        def files(self, *, product_id, **kwargs):
            queried.append(product_id)
            return {"lidar": [cl61], "doppler-lidar": [halo]}.get(product_id, [])

    monkeypatch.setattr(download, "APIClient", FakeClient)
    return queried


def test_list_harmonized_sources_searches_all_products(monkeypatch):
    queried = _harmonized_client(monkeypatch)
    sources = download.list_harmonized_sources("uto", "2024-12-31")
    assert set(queried) == set(download.HARMONIZED_PRODUCTS)
    assert len(sources) == 2  # ceilometer and doppler-lidar both listed
    assert all(s.reader is None for s in sources)  # all read with read_lidar


def test_list_harmonized_sources_filters_by_instrument_substring(monkeypatch):
    _harmonized_client(monkeypatch)
    sources = download.list_harmonized_sources("uto", "2024-12-31", "halo")
    assert len(sources) == 1
    assert "HALO" in sources[0].label


def test_list_harmonized_sources_errors_when_filter_matches_nothing(monkeypatch):
    _harmonized_client(monkeypatch)
    with pytest.raises(ValueError, match="matching 'cs135'"):
        download.list_harmonized_sources("uto", "2024-12-31", "cs135")


def test_invalid_site_becomes_clean_value_error(monkeypatch):
    """A portal CloudnetAPIError (e.g. unknown site) surfaces as a plain ValueError.

    The portal reports it with a list message; it must be flattened to one line,
    not leak as an uncaught traceback.
    """

    class FailingClient:
        def raw_files(self, **kwargs):
            raise download.CloudnetAPIError(["Invalid site: notarealsite"])

    monkeypatch.setattr(download, "APIClient", FailingClient)
    with pytest.raises(ValueError, match="^Invalid site: notarealsite$"):
        download.list_raw_sources("notarealsite", "2025-05-25")


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
