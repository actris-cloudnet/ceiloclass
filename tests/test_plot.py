import datetime
import warnings
from types import SimpleNamespace

import netCDF4
import numpy as np
from numpy import ma

from ceiloclass.plot import (
    _edges,
    _pixel_cells,
    _take,
    _time_cells,
    plot_classification,
)

MINUTE = 1 / 1440  # in days, the unit of matplotlib date numbers


def test_pixel_cells_keeps_every_cell_at_native_resolution():
    edges = _edges(np.arange(4.0))
    assert _pixel_cells(edges, 4).tolist() == [0, 1, 2, 3]


def test_pixel_cells_thins_and_repeats():
    edges = _edges(np.arange(9.0))
    assert _pixel_cells(edges, 3).tolist() == [1, 4, 7]
    assert _pixel_cells(edges[:3], 4).tolist() == [0, 0, 1, 1]


def test_pixel_cells_stretches_cells_across_a_time_gap():
    # The cells either side of the gap meet in its middle.
    edges = _edges(np.array([0.0, 1.0, 9.0, 10.0]))
    cells = _pixel_cells(edges, 22)
    assert np.bincount(cells).tolist() == [2, 9, 9, 2]


def test_depol_fill_values_under_the_mask_do_not_warn(tmp_path):
    day = datetime.datetime(2025, 6, 24)
    classification = SimpleNamespace(
        time=np.array([day + datetime.timedelta(minutes=m) for m in range(3)]),
        range=np.array([100.0, 200.0, 300.0, 400.0]),
        target=np.zeros((3, 4), dtype=np.int8),
        t0_alt=np.full(3, 250.0),
        strong_beta=1e-6,
    )
    depol = ma.masked_greater(np.full((3, 4), 0.1, dtype=np.float32), 1)
    depol[1, 2] = ma.masked
    depol.data[1, 2] = netCDF4.default_fillvals["f4"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_classification(classification, tmp_path / "plot.png", depol=depol)


def test_time_cells_without_gaps_are_plain_edges():
    time = np.arange(5) * MINUTE
    edges, profile = _time_cells(time)
    np.testing.assert_allclose(edges, _edges(time))
    assert profile.tolist() == [0, 1, 2, 3, 4]


def test_time_cells_put_an_empty_cell_in_a_gap():
    time = np.array([0, 1, 2, 30, 31]) * MINUTE
    edges, profile = _time_cells(time)
    assert profile.tolist() == [0, 1, 2, -1, 3, 4]
    np.testing.assert_allclose(edges / MINUTE, [-0.5, 0.5, 1.5, 2.5, 29.5, 30.5, 31.5])


def test_time_cells_keep_the_usual_width_next_to_a_leading_gap():
    time = np.array([0, 30, 31, 32]) * MINUTE
    edges, profile = _time_cells(time)
    assert profile.tolist() == [0, -1, 1, 2, 3]
    np.testing.assert_allclose(edges[:3] / MINUTE, [-0.5, 0.5, 29.5])


def test_time_cells_do_not_split_a_coarse_even_grid():
    # 15 min bins exceed the gap limit but are the usual step, so not gaps.
    edges, profile = _time_cells(np.arange(4) * 15 * MINUTE)
    assert profile.tolist() == [0, 1, 2, 3]


def test_take_masks_gap_profiles():
    data = np.arange(6).reshape(3, 2)
    out = _take(data, np.array([0, -1, 2]), np.array([1, 0]))
    assert out.tolist() == [[1, 0], [None, None], [5, 4]]
