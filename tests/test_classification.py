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
    _melt_band_below_ice,
    _source_connected,
    _thin_runs,
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


def test_melt_band_below_ice_reaches_t0_from_ice_base():
    n_time, n_gate = 8, 16
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 6:] = True  # 0 degC line at gate 6
    ice_like = np.zeros((n_time, n_gate), dtype=bool)
    ice_like[:, 9:] = True  # thick depol ice base at gate 9, above t0
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, 6:] = True
    band = _melt_band_below_ice(ice_like, cold, bright, height)
    assert band[:, 6:9].all()  # band fills from the ice base down to t0
    assert not band[:, 5].any()  # warm side untouched
    assert not band[:, 9:].any()  # the (thick) ice seed itself is excluded


def test_melt_band_below_ice_stops_at_t0():
    n_time, n_gate = 8, 12
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 6:] = True
    ice_like = np.zeros((n_time, n_gate), dtype=bool)
    ice_like[:, 9:] = True
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, :11] = True  # bright all the way to the ground
    band = _melt_band_below_ice(ice_like, cold, bright, height)
    assert band[:, 6:9].all()
    assert not band[:, :6].any()  # never crosses t0 into the warm region


def test_melt_band_below_ice_requires_ice_above():
    n_time, n_gate = 8, 12
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 6:] = True
    ice_like = np.zeros((n_time, n_gate), dtype=bool)  # no depol ice anywhere
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, 6:11] = True
    band = _melt_band_below_ice(ice_like, cold, bright, height)
    assert not band.any()  # isolated cold low-depol signal is left as-is


def test_melt_band_below_ice_excludes_patch_buried_in_ice():
    # A low-depol patch below ice but with no bright path down to the warm region
    # (a clear gap beneath it) is reachable from the ice above but NOT from the
    # warm region -> not a melting layer, so excluded. Regression guard: such
    # patches inside an ice cloud must not become drizzle.
    n_time, n_gate = 8, 16
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 4:] = True  # 0 degC at gate 4
    ice_like = np.zeros((n_time, n_gate), dtype=bool)
    ice_like[:, 9:15] = True  # ice cloud gates 9-14
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, 7:9] = True  # low-depol patch at 7-8, just below the ice
    bright[:, 9:15] = True  # the ice itself is bright
    # gates 4-6 are clear -> the patch has no link down to the warm region
    band = _melt_band_below_ice(ice_like, cold, bright, height)
    assert not band.any()  # buried patch is not melting


def test_thin_runs_keeps_only_thin_runs():
    height = np.arange(10) * 100.0  # 100 m gates
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 1:3] = True  # 100 m run (thin)
    mask[0, 5:9] = True  # 300 m run (thick)
    out = _thin_runs(mask, height, max_thickness=150.0)
    assert out[0, 1:3].all()
    assert not out[0, 5:9].any()


def test_melt_band_below_ice_bridges_thin_melting_enhancement():
    # The melting layer depolarizes -> a thin ice_like enhancement at the melt
    # level. It is bridged so the band still links the ice base down to the rain.
    n_time, n_gate = 8, 18
    height = np.arange(n_gate) * 100.0  # 100 m gates
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 4:] = True
    ice_like = np.zeros((n_time, n_gate), dtype=bool)
    ice_like[:, 5] = True  # thin (1-gate) melting enhancement just above t0
    ice_like[:, 11:17] = True  # thick real ice cloud
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, 4:17] = True
    band = _melt_band_below_ice(ice_like, cold, bright, height)
    assert band[:, 5:11].all()  # enhancement bridged; band linked to the rain
    assert not band[:, 11:].any()  # thick ice still excluded


def test_melt_band_below_ice_does_not_bridge_thick_ice():
    # A thick ice layer between t0 and a low-depol patch is NOT bridged: the patch
    # stays out (buried-patch protection preserved).
    n_time, n_gate = 8, 18
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 4:] = True
    ice_like = np.zeros((n_time, n_gate), dtype=bool)
    ice_like[:, 5:10] = True  # thick (400 m) ice barrier above t0
    ice_like[:, 13:17] = True  # upper ice
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, 4:17] = True
    band = _melt_band_below_ice(ice_like, cold, bright, height)
    assert not band[:, 10:13].any()  # patch above the thick barrier stays ice


def test_melt_band_below_ice_respects_max_depth():
    # A long low-depol column links the warm region to the ice base. A tight
    # max_depth stops the two floods (from the ice above, from the warm below)
    # from meeting, so no melt band forms; a large cap fills the whole column.
    n_time, n_gate = 8, 16
    height = np.arange(n_gate) * 100.0
    cold = np.zeros((n_time, n_gate), dtype=bool)
    cold[:, 2:] = True
    ice_like = np.zeros((n_time, n_gate), dtype=bool)
    ice_like[:, 14:] = True  # ice base at gate 14
    bright = np.zeros((n_time, n_gate), dtype=bool)
    bright[:, 2:14] = True  # 12-gate low-depol column from t0 to the ice base
    full = _melt_band_below_ice(ice_like, cold, bright, height, max_depth=1500.0)
    assert full[:, 2:14].all()  # large cap: floods meet, whole column is melt
    capped = _melt_band_below_ice(ice_like, cold, bright, height, max_depth=250.0)
    assert not capped.any()  # tight cap: floods cannot span the column


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


def test_classify_weak_depol_boundary_layer_not_flooded_to_ice():
    # A daytime boundary layer of weak (sub-threshold) but strongly-depolarizing
    # aerosol (dust/pollen) below the model 0 degC line must stay aerosol: the
    # cold region anchors to falling ice only through CLOUD-STRENGTH backscatter,
    # so this weak column does not drag the freezing region down to the ground.
    n_time, n_height = 30, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, :120] = 1.5e-6  # weak signal from ground up past the 0 degC line
    depol = ma.zeros((n_time, n_height))
    depol[:, :120] = 0.4  # strongly depolarizing throughout (would be ice_like)
    tw = np.tile(T0 + (100 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 100
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=3e-6)
    assert (cls.target[:, :100] == Target.AEROSOL).all()  # warm side stays aerosol
    assert not (cls.target[:, :100] == Target.ICE).any()


def test_classify_strong_signal_splits_ice_and_rain_by_temperature():
    n_time, n_height = 4, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 20:36] = 5e-5  # bright, low (warm) -- wide, so not a liquid layer
    beta[:, 36:150] = 5e-6  # signal bridging the warm layer up to the cold ice
    beta[:, 150:166] = 5e-5  # bright, high (cold)
    tw = np.tile(T0 + (100 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 100
    cls = classify(_synthetic_ceilo(beta), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 28] == Target.DRIZZLE_OR_RAIN).all()  # warm strong
    assert (cls.target[:, 158] == Target.ICE).all()  # cold strong


def test_classify_melt_band_below_ice_is_drizzle():
    # A bright, wide (non-liquid-layer) column with a depol-ice cap on top and the
    # model 0 degC below the ice base: the cold low-depol band between t0 and the
    # ice base is ice melting into rain, so it classifies as drizzle/rain, not ice.
    n_time, n_height = 4, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 100:140] = 5e-5  # wide flat strong column -> not a liquid layer
    depol = ma.zeros((n_time, n_height))
    depol[:, 130:140] = 0.4  # depol ice cap on top
    tw = np.tile(T0 + (110 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 110
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 120] == Target.DRIZZLE_OR_RAIN).all()  # melt band -> rain
    assert (cls.target[:, 135] == Target.ICE).all()  # the depol-ice cap stays ice


def test_classify_ice_rain_boundary_follows_depol_phase_change():
    # A bright precipitation shaft crossing the model 0 degC line: solid ice above
    # (depol ~0.45) melts into rain below, where depolarization drops to ~0.2 --
    # still above the ice/liquid limit but well below the ice-core limit. The
    # ice/rain boundary must follow that depol drop, not flood ice down through
    # the still-depolarizing rain shaft below the melt.
    n_time, n_height = 8, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 60:160] = 5e-5  # wide flat bright shaft (not a liquid layer)
    depol = ma.zeros((n_time, n_height))
    depol[:, 60:130] = 0.2  # melting / rain below t0: depol dropped, still > 0.15
    depol[:, 130:160] = 0.45  # solid ice above t0
    tw = np.tile(T0 + (130 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 130
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 145] == Target.ICE).all()  # solid ice stays ice
    # gate 100 is below the melt, depol 0.2 -> rain, not ice dragged down
    assert not (cls.target[:, 100] == Target.ICE).any()
    assert (cls.target[:, 100] == Target.DRIZZLE_OR_RAIN).all()


def test_classify_extends_ice_through_solid_ice_below_model_t0():
    # The legitimate case _extend_cold_to_ice exists for: the model 0 degC is
    # biased high and real, solid ice (depol ~0.45) sits below it not yet melted.
    # The ice-core depol gate must still let the freezing region flood down through
    # it -- so this stays ice, in contrast to the dropped-depol rain shaft above.
    n_time, n_height = 8, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 60:160] = 5e-5
    depol = ma.zeros((n_time, n_height))
    depol[:, 60:160] = 0.45  # solid ice all the way down (model t0 biased high)
    tw = np.tile(T0 + (130 - gate) * 0.1, (n_time, 1))  # model 0 degC at gate 130
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 100] == Target.ICE).all()  # solid ice below t0 -> ice


def test_classify_ice_core_limit_does_not_strip_cold_ice_above_t0():
    # The ice-core (0.30) limit is for the DOWNWARD extension only. Above the 0 degC
    # line a moderate depol (0.15-0.30) is still cold ice, not rain: a sub-freezing
    # ice cloud at depol 0.2 must stay ICE, not be relabelled drizzle/rain. (Guards
    # against keying the melt-band removal on the ice-core limit -- regression for
    # the lindenberg-melting-above-t0 case.)
    n_time, n_height = 8, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 120:160] = 5e-5  # bright cold cloud, entirely above t0
    depol = ma.zeros((n_time, n_height))
    depol[:, 120:160] = 0.2  # ice depol, below the ice-core limit but above 0.15
    tw = np.tile(T0 + (100 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 100
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 140] == Target.ICE).all()  # cold 0.2-depol cloud -> ice
    assert not (cls.target[:, 120:160] == Target.DRIZZLE_OR_RAIN).any()


def test_classify_cold_strong_signal_without_ice_above_stays_ice():
    # The same column with no depol ice cap: nothing seeds the band, so the cold
    # strong column stays ice (the boundary only moves where depol ice is above).
    n_time, n_height = 4, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 100:140] = 5e-5
    depol = ma.zeros((n_time, n_height))  # no high depol anywhere
    tw = np.tile(T0 + (110 - gate) * 0.1, (n_time, 1))
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=1e-5)
    # The cold region stays ice (the boundary is not raised without depol ice).
    assert (cls.target[:, 115:140] == Target.ICE).all()


def test_classify_warm_bright_layer_without_cloud_above_is_aerosol():
    # A bright warm layer with no hydrometeor source above it is aerosol, not
    # drizzle -- precipitation must fall from a cloud. This is the bright,
    # humidified marine boundary layer (sea-salt haze) at Mindelo, whose
    # backscatter clears the cloud threshold yet has no parent cloud.
    n_time, n_height = 6, 200
    band = ma.masked_all((n_time, n_height))
    band[:, 20:36] = 5e-6  # bright, wide, flat warm layer -> not a liquid layer
    model = _model(np.full((n_time, n_height), 290.0))  # warm everywhere

    lone = classify(_synthetic_ceilo(band), model, strong_beta=3e-6)
    assert (lone.target[:, 28] == Target.AEROSOL).all()  # no source above
    assert not (lone.target == Target.DRIZZLE_OR_RAIN).any()

    # Disabling the gate restores the old behaviour (any bright warm = drizzle).
    ungated = classify(
        _synthetic_ceilo(band), model, strong_beta=3e-6, drizzle_source_window=-1
    )
    assert (ungated.target[:, 28] == Target.DRIZZLE_OR_RAIN).all()


def test_classify_warm_bright_layer_below_cloud_stays_drizzle():
    # The same bright warm layer, now CONNECTED through continuous signal up to a
    # liquid cloud above it, has a hydrometeor source and is kept as drizzle --
    # the fix must not break real sub-cloud drizzle.
    n_time, n_height = 6, 200
    beta = ma.masked_all((n_time, n_height))
    beta[:, 20:36] = 5e-6  # bright warm layer (the would-be drizzle)
    beta[:, 36:60] = 8e-7  # weak signal bridging it up to the cloud base
    beta[:, 60:67] = [5e-7, 1.5e-6, 4e-6, 1.3e-5, 6e-6, 1.2e-6, 3e-7]  # liquid cloud
    model = _model(np.full((n_time, n_height), 290.0))
    cls = classify(_synthetic_ceilo(beta), model, strong_beta=3e-6)
    assert (cls.target[:, 63] == Target.DROPLET).all()  # the cloud above
    assert (cls.target[:, 28] == Target.DRIZZLE_OR_RAIN).all()  # drizzle below it


def test_classify_bright_blob_below_disconnected_cirrus_is_aerosol():
    # A near-surface bright warm blob with clear air between it and an isolated
    # high cirrus must stay aerosol: the cirrus is not its source. Guards against
    # the "any ice anywhere above" rule that classified ground aerosol as drizzle
    # at Mindelo because of a single despeckled cirrus pixel kilometres above.
    n_time, n_height = 6, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 5:18] = 5e-6  # bright warm near-surface blob
    beta[:, 150:153] = 5e-6  # isolated cirrus, far above, clear air between
    # warm below, sub-freezing only up where the cirrus sits -> cirrus is ice
    tw = np.tile(T0 + (120 - gate) * 0.2, (n_time, 1))  # 0 degC at gate 120
    cls = classify(_synthetic_ceilo(beta), _model(tw), strong_beta=3e-6)
    assert (cls.target[:, 151] == Target.ICE).all()  # the cirrus is ice
    assert (cls.target[:, 11] == Target.AEROSOL).all()  # blob not sourced by it
    assert not (cls.target[:, :120] == Target.DRIZZLE_OR_RAIN).any()


def test_source_connected_bridges_small_gap_not_large():
    # A thin clear gap (a screened melting-layer notch) between a column and the
    # cloud above is bridged up to `max_gap` gates; a longer gap (Mindelo) is not.
    signal = np.ones((1, 20), dtype=bool)
    cloud = np.zeros((1, 20), dtype=bool)
    cloud[0, 15] = True  # the source sits above the gap
    signal[0, 8:11] = False  # 3-gate clear gap below the cloud
    assert not _source_connected(cloud, signal, 0)[0, 5]  # default: gap breaks it
    assert _source_connected(cloud, signal, 0, max_gap=3)[0, 5]  # bridged
    signal[0, 8:13] = False  # widen to a 5-gate gap
    assert not _source_connected(cloud, signal, 0, max_gap=3)[0, 5]  # too wide


def test_classify_drizzle_bridges_thin_melt_gap():
    # A drizzle column capped by depol ice, with a thin masked notch (the screened
    # melting-layer backscatter minimum) between them: the notch must not sever the
    # drizzle from its ice source and drop it to aerosol (the Lindenberg case).
    n_time, n_height = 4, 200
    gate = np.arange(n_height)
    beta = ma.masked_all((n_time, n_height))
    beta[:, 80:120] = 5e-5  # bright wide warm column (drizzle), below the notch
    beta[:, 122:140] = 5e-5  # bright column above the notch, up to the ice
    # gates 120:122 left masked = the screened melt notch (~60 m at 30 m gates)
    depol = ma.zeros((n_time, n_height))
    depol[:, 130:140] = 0.4  # depol ice cap on top
    tw = np.tile(T0 + (128 - gate) * 0.1, (n_time, 1))  # 0 degC at gate 128
    cls = classify(_synthetic_ceilo(beta, depol), _model(tw), strong_beta=1e-5)
    assert (cls.target[:, 135] == Target.ICE).all()  # the depol-ice cap
    assert (cls.target[:, 100] == Target.DRIZZLE_OR_RAIN).all()  # drizzle below notch
    assert not (cls.target[:, 80:120] == Target.AEROSOL).any()


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
