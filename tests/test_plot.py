import datetime
import warnings
from types import SimpleNamespace

import netCDF4
import numpy as np
from numpy import ma

from ceiloclass.plot import _edges, _pixel_cells, plot_classification


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
