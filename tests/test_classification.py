import datetime

import numpy as np
from ceilopyter.ceilo import Ceilo
from ceilopyter.ceilo_raw import CeiloRaw
from numpy import ma

from ceiloclass.classification import (
    Target,
    _adaptive_strong_beta,
    _assemble,
    _despeckle,
    _extend_cold_to_ice,
    classify,
)
from ceiloclass.model import T0, Model


def test_assemble_precedence():
    shape = (1, 6)
    false = np.zeros(shape, dtype=bool)
    droplet, cold, ice, rain, aerosol = (false.copy() for _ in range(5))
    aerosol[0, 0] = True
    ice[0, 1] = True
    rain[0, 2] = True
    droplet[0, 3] = True  # warm droplet
    droplet[0, 4] = cold[0, 4] = True  # supercooled (cold droplet)
    rain[0, 5] = droplet[0, 5] = True  # liquid on top overwrites rain
    out = _assemble(droplet, cold, ice, rain, aerosol)
    assert out[0].tolist() == [
        Target.AEROSOL,
        Target.ICE,
        Target.DRIZZLE_OR_RAIN,
        Target.DROPLET,
        Target.SUPERCOOLED,
        Target.DROPLET,
    ]


# --- speckle filter ---------------------------------------------------------


def test_despeckle_clears_isolated_pixel():
    target = np.zeros((5, 5), dtype=int)
    target[2, 2] = Target.ICE  # single isolated pixel
    assert _despeckle(target.copy(), 3)[2, 2] == Target.CLEAR


def test_despeckle_keeps_cluster():
    target = np.zeros((5, 5), dtype=int)
    target[2, 2] = target[2, 3] = target[3, 2] = Target.ICE  # 3-pixel cluster
    out = _despeckle(target.copy(), 3)
    assert (out == Target.ICE).sum() == 3


def test_despeckle_disabled():
    target = np.zeros((5, 5), dtype=int)
    target[2, 2] = Target.ICE
    assert _despeckle(target.copy(), 1)[2, 2] == Target.ICE


# --- adaptive backscatter threshold ----------------------------------------


def _beta(values):
    return ma.masked_array(np.asarray(values, dtype=float).reshape(1, -1))


def test_adaptive_strong_beta_bimodal_picks_valley():
    rng = np.random.default_rng(0)
    aerosol = rng.lognormal(np.log(2e-7), 0.18, 20000)
    cloud = rng.lognormal(np.log(1e-5), 0.18, 5000)
    cutoff = _adaptive_strong_beta(_beta(np.concatenate([aerosol, cloud])))
    assert 3e-7 < cutoff < 5e-6  # in the gap between the two modes


def test_adaptive_strong_beta_bimodal_aerosol_not_split_as_cloud():
    # A dry day with two aerosol modes (e.g. boundary-layer + lofted dust), both
    # at aerosol-level backscatter. The upper mode must stay below the threshold
    # (classified aerosol), not be split off as cloud at the valley between them.
    rng = np.random.default_rng(4)
    low = rng.lognormal(np.log(4e-7), 0.15, 20000)
    dust = rng.lognormal(np.log(1e-6), 0.15, 8000)
    cutoff = _adaptive_strong_beta(_beta(np.concatenate([low, dust])))
    assert cutoff > 1.1e-6  # above the dust mode, not in the valley below it


def test_adaptive_strong_beta_anchors_on_lower_peak():
    # Cloud mode out-numbers aerosol; the threshold must still anchor on the
    # lower aerosol mode, not the taller cloud mode.
    rng = np.random.default_rng(1)
    aerosol = rng.lognormal(np.log(2e-7), 0.15, 5000)
    cloud = rng.lognormal(np.log(1e-5), 0.15, 15000)
    cutoff = _adaptive_strong_beta(_beta(np.concatenate([aerosol, cloud])))
    assert cutoff < 2e-6


def test_adaptive_strong_beta_single_mode_uses_shoulder():
    rng = np.random.default_rng(2)
    aerosol = rng.lognormal(np.log(2e-6), 0.2, 20000)
    cutoff = _adaptive_strong_beta(_beta(aerosol))
    assert 2e-6 < cutoff <= 1e-5  # past the peak, below the cap


def test_adaptive_strong_beta_caps_runaway():
    # A single high mode with no separable aerosol mode would run off; the
    # absolute cap keeps it at the physical aerosol ceiling.
    rng = np.random.default_rng(3)
    high = rng.lognormal(np.log(3e-5), 0.25, 20000)
    assert _adaptive_strong_beta(_beta(high)) == 1e-5


def test_adaptive_strong_beta_too_few_samples_returns_default():
    assert _adaptive_strong_beta(_beta(np.full(100, 1e-6))) == 3e-6


# --- freezing-region anchor -------------------------------------------------


def test_extend_cold_to_ice_reaches_ice_base():
    n_time, n_gate = 20, 12
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 9:] = True  # model freezing region: gates 9+
    ice = np.zeros((n_time, n_gate), dtype=bool)
    ice[:, 6:] = True  # depol ice from gate 6 up (3 gates below the model line)
    extended = _extend_cold_to_ice(cold, ice, height)
    assert extended[:, 6:].all()  # cold reaches the ice base
    assert not extended[:, 5].any()  # below the ice base stays warm


def test_extend_cold_to_ice_drops_isolated_pillar():
    n_time, n_gate = 21, 12
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 9:] = True
    ice = np.zeros((n_time, n_gate), dtype=bool)
    ice[:, 7:] = True  # base at gate 7 in every profile
    ice[10, 1:] = True  # one noisy profile: ice almost to the ground
    extended = _extend_cold_to_ice(cold, ice, height)
    assert extended[0, 7:].all() and not extended[0, 6]  # normal base kept
    assert not extended[10, 1:7].any()  # pillar clipped to the smoothed base
    assert extended[10, 7:].all()


def test_extend_cold_to_ice_ignores_disconnected_ice():
    n_time, n_gate = 10, 12
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 9:] = True
    ice = np.zeros((n_time, n_gate), dtype=bool)
    ice[:, 2:5] = True  # ice well below, not touching the cold region
    extended = _extend_cold_to_ice(cold, ice, height)
    assert np.array_equal(extended, cold)  # no contiguous path -> unchanged


# --- classify integration ---------------------------------------------------


def _synthetic_ceilo(beta_masked, depol=None):
    time = np.array(
        [datetime.datetime(2025, 6, 14, 0, i) for i in range(beta_masked.shape[0])]
    )
    rng = np.arange(beta_masked.shape[1]) * 30.0
    raw = CeiloRaw(time, rng, beta_masked, 910.0, depol=depol)
    return Ceilo(raw, ma.filled(beta_masked, 0.0), beta_masked, 1.0)


def _model(tw):
    return Model(tw=tw, extrapolated=np.zeros(tw.shape, dtype=bool))


def _liquid_layer_beta(n_time=4, n_height=200):
    profile = np.zeros(n_height)
    profile[37:44] = [5e-7, 1.5e-6, 4e-6, 1.3e-5, 6e-6, 1.2e-6, 3e-7]  # sharp top
    beta = ma.masked_all((n_time, n_height))
    for t in range(n_time):
        beta[t, profile > 0] = profile[profile > 0]
    return beta


def test_classify_detects_warm_droplet():
    beta = _liquid_layer_beta()
    model = _model(np.full(beta.shape, 290.0))  # warm everywhere
    cls = classify(_synthetic_ceilo(beta), model)
    assert cls.target.shape == beta.shape
    assert cls.droplet[:, 40].all()
    assert (cls.target == Target.DROPLET).any()
    assert not cls.cold.any()  # warm column -> no ice


def test_classify_low_depol_cold_layer_is_supercooled():
    beta = _liquid_layer_beta()
    model = _model(np.full(beta.shape, T0 - 5))  # cold everywhere
    depol = ma.zeros(beta.shape)  # spherical -> liquid
    cls = classify(_synthetic_ceilo(beta, depol), model)
    assert (cls.target[:, 40] == Target.SUPERCOOLED).all()


def test_classify_high_depol_cold_layer_is_ice():
    beta = _liquid_layer_beta()
    model = _model(np.full(beta.shape, T0 - 5))
    depol = ma.array(np.full(beta.shape, 0.4))  # non-spherical -> ice
    cls = classify(_synthetic_ceilo(beta, depol), model)
    assert (cls.target[:, 40] == Target.ICE).all()
    assert not (cls.target == Target.SUPERCOOLED).any()


def test_classify_multiple_scattering_top_stays_liquid():
    # Multiple scattering lifts depolarization toward cloud top: low at the base
    # (single scattering), high above. Because the layer has a low-depol base it
    # is genuine liquid, so its high-depol upper gates must NOT be carved out as
    # ice -- contrast test_classify_high_depol_cold_layer_is_ice (uniform high
    # depol, which is real ice).
    beta = _liquid_layer_beta()
    model = _model(np.full(beta.shape, T0 - 5))  # cold -> supercooled
    depol = ma.zeros(beta.shape)
    depol[:, 41:] = 0.4  # high depol from just above the peak upward
    cls = classify(_synthetic_ceilo(beta, depol), model)
    assert (cls.target[:, 40] == Target.SUPERCOOLED).all()  # low-depol base
    assert (cls.target[:, 42] == Target.SUPERCOOLED).all()  # high-depol top, shielded
    assert not (cls.target[:, 37:44] == Target.ICE).any()  # no part of the layer is ice


def test_classify_ice_depol_limit_is_tunable():
    # A uniform depol-0.4 cold layer is ice by default; raising the limit above
    # 0.4 turns it back into liquid, confirming the threshold is wired through.
    beta = _liquid_layer_beta()
    model = _model(np.full(beta.shape, T0 - 5))
    depol = ma.array(np.full(beta.shape, 0.4))
    default = classify(_synthetic_ceilo(beta, depol), model)
    assert (default.target[:, 40] == Target.ICE).all()
    raised = classify(_synthetic_ceilo(beta, depol), model, ice_depol_limit=0.5)
    assert (raised.target[:, 40] == Target.SUPERCOOLED).all()


def test_classify_find_falling_does_not_cut_high_supercooled_top():
    # A cold (-20 C) low-depol supercooled layer whose attenuating top crosses
    # 2 km must not be chopped to ice by find_falling: with depolarization, depol
    # (not the -15 C / 2000 m heuristic) governs where liquid stops growing.
    n_height = 80
    profile = np.zeros(n_height)
    # sharp liquid peak ~1.86 km, attenuating low-depol tail up past 2 km
    profile[60:71] = [
        3e-6,
        1e-5,
        4e-5,
        6e-5,
        3e-5,
        1.5e-5,
        8e-6,
        4e-6,
        2.5e-6,
        1.6e-6,
        1.1e-6,
    ]
    beta = ma.masked_all((4, n_height))
    for t in range(4):
        beta[t, profile > 0] = profile[profile > 0]
    depol = ma.zeros((4, n_height))  # spherical -> liquid
    model = _model(np.full((4, n_height), T0 - 20))  # cold enough for find_falling
    cls = classify(_synthetic_ceilo(beta, depol), model)
    # gate 67 is 2010 m (above the 2000 m find_falling floor) and low-depol
    assert (cls.target[:, 67] == Target.SUPERCOOLED).all()


def test_classify_thin_cirrus_is_ice_via_depol():
    # Faint (sub-threshold) but strongly-depolarizing sub-freezing signal is ice,
    # not aerosol -- the depol-based cold-side split.
    n_time, n_height = 4, 200
    beta = ma.masked_all((n_time, n_height))
    beta[:, 100:110] = 8e-7  # below the strong threshold
    depol = ma.zeros((n_time, n_height))
    depol[:, 100:110] = 0.4
    model = _model(np.full((n_time, n_height), T0 - 20))
    cls = classify(_synthetic_ceilo(beta, depol), model, strong_beta=3e-6)
    assert (cls.target[:, 105] == Target.ICE).all()


def test_classify_strong_signal_splits_ice_and_rain_by_temperature():
    n_time, n_height = 4, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 20:36] = 5e-5  # bright, low (warm) -- wide, so not a liquid layer
    beta[:, 150:166] = 5e-5  # bright, high (cold)
    tw = np.tile(T0 + (100 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 100
    cls = classify(_synthetic_ceilo(beta), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 28] == Target.DRIZZLE_OR_RAIN).all()  # warm strong
    assert (cls.target[:, 158] == Target.ICE).all()  # cold strong


def test_classify_weak_signal_is_aerosol():
    n_time, n_height = 4, 200
    beta = ma.masked_all((n_time, n_height))
    beta[:, 20:30] = 8e-7  # weak and flat
    cls = classify(
        _synthetic_ceilo(beta),
        _model(np.full((n_time, n_height), 290.0)),
        strong_beta=3e-6,
    )
    assert (cls.target[:, 25] == Target.AEROSOL).all()
    assert cls.strong_beta == 3e-6
