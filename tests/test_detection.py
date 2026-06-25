import numpy as np
from numpy import ma

from ceiloclass._interp import interpolate_along_time, local_maxima
from ceiloclass.detection import (
    correct_supercooled,
    fill_thin_clouds,
    find_depol_ice,
    find_freezing_region,
    find_liquid,
    grow_liquid,
)
from ceiloclass.model import T0


def _liquid_layer(scale=1.0, center=40, n_gates=200):
    # Sharp-topped layer, as a real liquid cloud appears in lidar backscatter.
    profile = np.zeros(n_gates)
    shape = np.array([5e-7, 1.5e-6, 4e-6, 1.3e-5, 6e-6, 1.2e-6, 3e-7])
    profile[center - 3 : center + 4] = shape * scale
    return ma.array(np.tile(profile, (4, 1)))


def test_local_maxima_matches_hand_count():
    data = np.array([[0, 1, 0, 2, 5, 2, 0, 1, 0, 0]], dtype=float)
    # order=1: strict peaks at indices 1, 4, 7
    peaks = local_maxima(data, order=1, axis=1)
    assert list(np.nonzero(peaks[0])[0]) == [1, 4, 7]
    # order=4: only index 4 is greater than all four neighbours each side
    peaks4 = local_maxima(data, order=4, axis=1)
    assert list(np.nonzero(peaks4[0])[0]) == [4]


def test_find_liquid_detects_proper_layer():
    height = np.arange(200) * 30.0  # 30 m gates
    beta = _liquid_layer(center=40)  # sharp layer at 1200 m
    is_liquid = find_liquid(beta, height)
    assert is_liquid.any()
    assert is_liquid[:, 40].all()  # the peak gate is liquid


def _surface_fog(n_gates=200):
    # Fog / very low stratus: strongest backscatter at the lowest gate, falling
    # off upward. The peak sits on the ground, in the blind zone of the search.
    profile = np.zeros(n_gates)
    profile[:5] = [1.3e-5, 1e-5, 6e-6, 1.2e-6, 3e-7]
    return ma.array(np.tile(profile, (4, 1)))


def test_find_liquid_detects_surface_fog():
    height = np.arange(200) * 30.0  # 30 m gates
    beta = _surface_fog()
    # A gate-0 maximum is never a strict local maximum (it equals its clipped
    # self), so the standard peak search finds nothing here.
    assert not local_maxima(ma.filled(beta, 0), order=4, axis=1).any()
    is_liquid = find_liquid(beta, height)
    assert is_liquid[:, 0].all()  # the surface peak gate is liquid
    assert not is_liquid[:, 10].any()  # but only the fog layer, not the column


def test_find_liquid_rejects_weak_peak():
    height = np.arange(200) * 30.0
    beta = _liquid_layer(scale=0.1, center=40)  # peak below peak_amp 1e-6
    assert not find_liquid(beta, height).any()


def test_find_liquid_rejects_wide_layer():
    height = np.arange(200) * 30.0
    gates = np.arange(200)
    wide = 8e-6 * np.exp(-((gates - 80) ** 2) / (2 * 12.0**2))  # ~700 m wide
    beta = ma.array(np.tile(wide, (4, 1)))
    assert not find_liquid(beta, height).any()


def test_find_freezing_region_crossing():
    height = np.arange(100) * 100.0  # 0..9900 m
    tw = np.linspace(290, 240, 100)[np.newaxis, :].repeat(3, axis=0)
    cold = find_freezing_region(tw, height)
    # Tw crosses T0 at gate ~33.6 -> below is warm, above is cold
    crossing = np.argmax(tw[0] < T0)
    assert not cold[:, :crossing].any()
    assert cold[:, crossing:].all()


def test_correct_supercooled_removes_very_cold_liquid():
    droplet = np.ones((2, 3), dtype=bool)
    tw = np.array([[300.0, T0 - 30, T0 - 50], [T0 - 50, 280.0, 275.0]])
    out = correct_supercooled(droplet, tw)
    # only the gate below -38 degC (T0-38) is removed
    assert out.tolist() == [[True, True, False], [False, True, True]]


def test_grow_liquid_absorbs_signal_fringe():
    #            mask:  gap  sig  sig  sig  sig  gap
    signal = np.array([[False, True, True, True, True, False]])
    droplet = np.array([[False, False, True, True, False, False]])
    blocked = np.zeros((1, 6), dtype=bool)
    height = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])  # 10 m gates
    # one gate (10 m) of signal above and below the core is absorbed; gaps are not
    grown = grow_liquid(droplet, signal, blocked, height, grow_distance=10.0)
    assert grown.tolist() == [[False, True, True, True, True, False]]


def test_grow_liquid_does_not_cross_blocked_or_gaps():
    signal = np.array([[True, True, True, True, True]])
    droplet = np.array([[False, False, True, False, False]])
    blocked = np.array([[False, True, False, False, False]])  # ice just below core
    height = np.array([0.0, 10.0, 20.0, 30.0, 40.0])  # 10 m gates -> 30 m = 3 gates
    grown = grow_liquid(droplet, signal, blocked, height, grow_distance=30.0)
    # growth stops at the blocked gate below, extends up through signal
    assert grown.tolist() == [[False, False, True, True, True]]


def test_fill_thin_clouds_fills_thin_layer():
    height = np.array([[0, 100, 200, 300, 400, 500.0]]).ravel()
    signal = np.array([[False, True, True, True, True, False]])
    droplet = np.array([[False, False, True, False, False, False]])
    blocked = np.zeros((1, 6), dtype=bool)
    # run spans gates 1..4 (100..400 m, thickness 300 m) <= max -> whole run filled
    out = fill_thin_clouds(droplet, signal, blocked, height, max_thickness=400)
    assert out.tolist() == [[False, True, True, True, True, False]]


def test_fill_thin_clouds_skips_thick_layer():
    height = np.arange(6) * 200.0  # run thickness 600 m
    signal = np.array([[False, True, True, True, True, False]])
    droplet = np.array([[False, False, True, False, False, False]])
    blocked = np.zeros((1, 6), dtype=bool)
    out = fill_thin_clouds(droplet, signal, blocked, height, max_thickness=400)
    assert out.tolist() == droplet.tolist()  # too thick, untouched


def test_fill_thin_clouds_stops_at_ice():
    height = np.arange(6) * 100.0
    signal = np.array([[True, True, True, True, True, True]])
    droplet = np.array([[False, False, True, False, False, False]])
    blocked = np.array([[False, False, False, True, False, False]])  # ice at gate 3
    out = fill_thin_clouds(droplet, signal, blocked, height, max_thickness=1000)
    # the ice gate breaks the run; only gates 0..2 (containing droplet) fill
    assert out.tolist() == [[True, True, True, False, False, False]]


def test_find_depol_ice_separates_ice_from_liquid():
    # Liquid is spherical (low depol); ice is non-spherical (high depol).
    depol = ma.array([[0.02, 0.05, 0.35, 0.45]])  # two liquid, two ice gates
    beta_mask = np.zeros((1, 4), dtype=bool)
    np.testing.assert_array_equal(
        find_depol_ice(depol, beta_mask), [[False, False, True, True]]
    )


def test_find_depol_ice_requires_signal():
    # High depol with no backscatter (masked) is not a target.
    depol = ma.array([[0.4, 0.4]])
    beta_mask = np.array([[True, False]])
    np.testing.assert_array_equal(find_depol_ice(depol, beta_mask), [[False, True]])


def test_interpolate_along_time_clamps():
    time = np.array([0.0, 10.0])
    values = np.array([[0.0, 100.0], [10.0, 200.0]])
    new = interpolate_along_time(np.array([-5.0, 5.0, 15.0]), time, values)
    assert np.allclose(new[0], [0.0, 100.0])  # clamped to first
    assert np.allclose(new[1], [5.0, 150.0])  # midpoint
    assert np.allclose(new[2], [10.0, 200.0])  # clamped to last
