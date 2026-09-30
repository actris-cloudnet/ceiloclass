import datetime
import warnings
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from numpy import ma

from ceiloclass import plot
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
    assert _pixel_cells(edges, edges[0], edges[-1], 4).tolist() == [0, 1, 2, 3]


def test_pixel_cells_thins_and_repeats():
    edges = _edges(np.arange(9.0))
    assert _pixel_cells(edges, edges[0], edges[-1], 3).tolist() == [1, 4, 7]
    assert _pixel_cells(edges, edges[0], edges[2], 4).tolist() == [0, 0, 1, 1]


def test_pixel_cells_stretches_cells_across_a_time_gap():
    # The cells either side of the gap meet in its middle.
    edges = _edges(np.array([0.0, 1.0, 9.0, 10.0]))
    cells = _pixel_cells(edges, edges[0], edges[-1], 22)
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


def test_window_thins_again_to_the_zoomed_view(monkeypatch):
    plt.switch_backend("Agg")
    n_time, n_range = 3000, 900
    day = datetime.datetime(2025, 6, 24)
    classification = SimpleNamespace(
        time=np.array(
            [day + datetime.timedelta(seconds=20 * i) for i in range(n_time)]
        ),
        range=10.0 + 10 * np.arange(n_range),
        target=np.zeros((n_time, n_range), dtype=np.int8),
        t0_alt=np.full(n_time, 2000.0),
        strong_beta=1e-6,
    )
    # A distinct value per cell, to tell which native cells an image holds.
    beta = 1e-12 * (1 + np.arange(n_time * n_range).reshape(n_time, n_range))
    time = mdates.date2num(classification.time)
    half_step = (time[1] - time[0]) / 2

    def zoom():
        ax = plt.gcf().axes[0]
        image = ax.images[0]
        full_extent = image.get_extent()
        assert image.get_array().shape[1] < n_time

        view = (time[100] - half_step, time[104] + half_step, 0.105, 0.135)
        ax.set_xlim(view[:2])
        ax.set_ylim(view[2:])
        np.testing.assert_allclose(image.get_extent(), view)
        np.testing.assert_array_equal(
            np.unique(image.get_array().compressed()), beta[100:105, 10:13].ravel()
        )

        ax.set_xlim(time[0] - 1, time[-1] + 1)
        ax.set_ylim(-5, 50)
        np.testing.assert_allclose(image.get_extent(), full_extent)

    monkeypatch.setattr(plot.plt, "show", zoom)
    plot_classification(classification, beta=beta, show=True, histogram=False)
