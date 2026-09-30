import numpy as np

from ceiloclass.plot import _edges, _pixel_cells


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
