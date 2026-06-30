import numpy as np
from numpy import ma

from ceiloclass._interp import interp_extrap, interpolate_along_time, local_maxima
from ceiloclass.detection import (
    _find_t0_alt,
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


def test_find_liquid_surface_pass_can_be_disabled():
    # The surface pass can be turned off (e.g. unreliable near-surface overlap):
    # the blind-zone fog is then not detected, while the rest of the search runs.
    height = np.arange(200) * 30.0
    beta = _surface_fog()
    assert find_liquid(beta, height)[:, 0].all()  # on by default
    assert not find_liquid(beta, height, surface_pass=False).any()  # off -> nothing


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


def _haze_ramp_layer(peak=5e-6):
    # A gradual sub-cloud aerosol ramp (gates 90-99) rising into a sharp-topped
    # peak at gate 100, as on a hazy day. `_ind_base` follows the ramp's gentle
    # gradient down, stretching the base into the sub-cloud haze.
    profile = np.zeros(300)  # 15 m gates, as on chm15k
    profile[90:101] = np.linspace(2e-6, peak, 11)
    profile[101:103] = [1e-6, 3e-7]
    return ma.array(np.tile(profile, (4, 1)))


def test_find_liquid_trims_base_to_cloud_strength():
    # With strong_beta the over-deep base (into sub-cloud haze) is raised to the
    # layer's lowest cloud-strength gate; the peak itself stays liquid.
    height = np.arange(300) * 15.0
    beta = _haze_ramp_layer(peak=5e-6)  # peak above the 4e-6 threshold
    loose = find_liquid(height=height, beta=beta)
    tight = find_liquid(height=height, beta=beta, strong_beta=4e-6)
    assert loose[:, 100].all() and tight[:, 100].all()  # peak liquid either way
    loose_base = int(np.argmax(loose[0]))
    tight_base = int(np.argmax(tight[0]))
    assert tight_base > loose_base  # base lifted out of the haze ramp
    assert (ma.filled(beta, 0)[0, loose_base:tight_base] < 4e-6).all()  # trimmed haze


def test_find_liquid_rejects_layer_below_cloud_strength():
    # A weak haze bump whose whole "layer" stays below strong_beta is not liquid:
    # without the threshold it is wrongly detected; with it, it is rejected.
    height = np.arange(300) * 15.0
    beta = _haze_ramp_layer(peak=3.5e-6)  # peak below the 4e-6 threshold
    assert find_liquid(height=height, beta=beta).any()  # peak_amp alone accepts it
    assert not find_liquid(height=height, beta=beta, strong_beta=4e-6).any()


def test_find_freezing_region_crossing():
    height = np.arange(100) * 100.0  # 0..9900 m
    tw = np.linspace(290, 240, 100)[np.newaxis, :].repeat(3, axis=0)
    cold = find_freezing_region(tw, height)
    # Tw crosses T0 at gate ~33.6 -> below is warm, above is cold
    crossing = np.argmax(tw[0] < T0)
    assert not cold[:, :crossing].any()
    assert cold[:, crossing:].all()


def test_find_t0_alt_warm_surface_takes_lowest_crossing():
    height = np.arange(8) * 1000.0  # 0..7000 m
    # Warm surface, cold layer aloft, then a warm layer, then cold again.
    prof = np.array([T0 + 4, T0 + 1, T0 - 1, T0 - 1, T0 + 1, T0 - 1, T0 - 3, T0 - 5])
    alt = _find_t0_alt(prof[np.newaxis, :], height)
    # Lowest crossing is between gate 1 (1000 m) and gate 2 (2000 m), not aloft.
    assert alt[0] == 1500.0


def test_find_t0_alt_cold_surface_takes_topmost_crossing():
    height = np.arange(8) * 1000.0
    # Sub-freezing surface (winter inversion): cold, warm aloft, cold again.
    prof = np.array([T0 - 2, T0 + 1, T0 + 3, T0 + 1, T0 - 1, T0 - 3, T0 - 5, T0 - 7])
    alt = _find_t0_alt(prof[np.newaxis, :], height)
    # Topmost crossing is just above the warm layer (gate 3 -> 4), not at ground.
    assert alt[0] == 3500.0


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
    grown = grow_liquid(droplet, signal, blocked, height, grow_up=10.0, grow_down=10.0)
    assert grown.tolist() == [[False, True, True, True, True, False]]


def test_grow_liquid_grows_more_upward_than_down():
    signal = np.ones((1, 7), dtype=bool)
    droplet = np.zeros((1, 7), dtype=bool)
    droplet[0, 3] = True
    blocked = np.zeros((1, 7), dtype=bool)
    height = np.arange(7) * 10.0  # 10 m gates
    grown = grow_liquid(droplet, signal, blocked, height, grow_up=20.0, grow_down=10.0)
    # 2 gates up (indices 4, 5), only 1 gate down (index 2)
    assert grown.tolist() == [[False, False, True, True, True, True, False]]


def test_grow_liquid_does_not_cross_blocked_or_gaps():
    signal = np.array([[True, True, True, True, True]])
    droplet = np.array([[False, False, True, False, False]])
    blocked = np.array([[False, True, False, False, False]])  # ice just below core
    height = np.array([0.0, 10.0, 20.0, 30.0, 40.0])  # 10 m gates -> 30 m = 3 gates
    grown = grow_liquid(droplet, signal, blocked, height, grow_up=30.0, grow_down=30.0)
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


def test_interp_extrap_continues_edge_slope():
    xp = np.array([0.0, 1000.0, 2000.0])
    fp = np.array([283.16, 273.16, 263.16])  # -0.01 K/m
    out = interp_extrap(np.array([-100.0, 500.0, 2500.0]), xp, fp)
    assert np.isclose(out[0], 284.16)  # extrapolated below, not clamped to 283.16
    assert np.isclose(out[1], 278.16)  # interpolated within range
    assert np.isclose(out[2], 258.16)  # extrapolated above, not clamped to 263.16


def test_interp_extrap_single_point_clamps():
    out = interp_extrap(np.array([-1.0, 0.0, 1.0]), np.array([0.0]), np.array([5.0]))
    assert np.allclose(out, 5.0)  # nothing to extrapolate from


def test_interpolate_along_time_clamps():
    time = np.array([0.0, 10.0])
    values = np.array([[0.0, 100.0], [10.0, 200.0]])
    new = interpolate_along_time(np.array([-5.0, 5.0, 15.0]), time, values)
    assert np.allclose(new[0], [0.0, 100.0])  # clamped to first
    assert np.allclose(new[1], [5.0, 150.0])  # midpoint
    assert np.allclose(new[2], [10.0, 200.0])  # clamped to last
