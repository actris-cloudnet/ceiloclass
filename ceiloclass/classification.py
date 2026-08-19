"""Simple target classification from ceilometer + model data (no radar).

Compute the 0 degC isotherm from the model and detect liquid droplet layers
(split into warm `DROPLET` and sub-freezing `SUPERCOOLED` by the 0 degC level).
Strong, non-liquid signal is cloud/precipitation -- `ICE` above the freezing
level (the sub-freezing air, anchored to observed ice, not just the model) and
`DRIZZLE_OR_RAIN` below it. Every other signal-bearing pixel is `AEROSOL`. There
is no melting class: the freezing anchor already places the ice/rain boundary at
the observed melt.
"""

from dataclasses import dataclass
from enum import IntEnum
from os import PathLike
from typing import Literal

import numpy as np
import numpy.typing as npt
from ceilopyter import Ceilo
from numpy import ma

from .detection import (
    DEEP_COLD_LIMIT,
    ICE_CORE_DEPOL_LIMIT,
    ICE_DEPOL_LIMIT,
    _fill_runs,
    _find_t0_alt,
    _grow_range,
    _iter_runs,
    _n_elements,
    _window_count,
    correct_supercooled,
    fill_thin_clouds,
    find_depol_ice,
    find_falling,
    find_freezing_region,
    find_liquid,
    grow_liquid,
)
from .model import Model, read_model

MS_TAIL = 80.0
"""Distance above a liquid layer over which rising depolarization is read as
multiple scattering, not ice (m).

Inside a liquid layer depolarization climbs from multiple scattering and falls
back to the background just above the cloud; this is the height over which that
elevated-depol tail is shielded from the ice veto. The default is the ~90th
percentile of the tail observed across several CL61 days."""

DRIZZLE_SOURCE_MAX_GAP = 100.0
"""Clear-air gap a drizzle column may span to reach its cloud source (m).

A melting layer has a backscatter notch that often falls below the screening
threshold, leaving a thin masked band between the rain and the ice above it. The
drizzle-source link (`_source_connected`) bridges clear gaps up to this distance
so that notch does not sever a drizzle column from its cloud and drop it to
aerosol. Stays well below the kilometre-scale clear air that must still reject an
unconnected near-surface layer (e.g. Mindelo haze under a distant cirrus)."""

MAX_PHYSICAL_BETA = 1e-2
"""Attenuated backscatter above this is instrument junk, not signal (sr-1 m-1).

Even a dense fog or a hard cloud base tops out around 1e-3; harmonized files
occasionally carry unmasked fill/garbage values (up to ~1e18 at Payerne) that
would otherwise enter the histogram and the classification as 'bright'."""

DEEP_RAIN_THICKNESS = 2000.0
"""A contiguous cloud-strength warm column at least this deep is rain (m).

Heavy rain extinguishes the lidar beam below the melting level, so neither the
parent ice nor a liquid/bright-band peak survives to be detected and the
column itself is the only evidence left. Aerosol does not sustain
cloud-strength backscatter over kilometres of depth -- the thickest genuine
aerosol runs across the regression days are ~0.74 km (Mindelo marine haze) and
~0.85 km (Granada dust) -- while attenuated rain columns run 2.5-3.8 km
(Juelich) or reach the ground at 1.3-1.9 km (Kenttarova). A run this deep is
therefore precipitation in its own right and seeds the drizzle flood even with
nothing detected above it."""

DEEP_RAIN_PASSABLE_THICKNESS = 1200.0
"""Column depth that makes a warm-strong run passable to the drizzle flood (m).

Weaker sibling of `DEEP_RAIN_THICKNESS`: a run this deep is not proof of rain
on its own (it stays aerosol in isolation) but is admitted into a shaft when
2D-connected to sourced rain -- an attenuated ground-reaching column beside a
sourced one. Sits above the thickest genuine aerosol runs observed (~0.85 km)
so a haze or dust layer touching a drizzle shaft still cannot ride the flood."""

SATURATION_FRACTION = 0.67
"""Fraction of the saturated backscatter integral at which the beam counts as gone.

The column integral of attenuated backscatter, I(z) = int beta' dz, is bounded:
I = (1 - T^2) / (2 eta S), so it saturates at I_sat = 1 / (2 eta S) once the
two-way transmission T^2 has vanished -- about 0.03 sr-1 for liquid with a
ceilometer (S ~ 18.8 sr, multiple-scattering factor eta ~ 0.7-1). I / I_sat is
therefore the fraction of the beam lost by that height. At 0.67 two thirds of
the beam are gone, so only a bright target could still be seen above; the
void above counts as attenuated. Factory-calibrated Vaisala ceilometers reach
I_sat consistently (0.02-0.035 across sites), which is why the plateau can be
estimated per file from the fully attenuating liquid tops (`_beam_saturation`)
instead of relying on the absolute calibration."""

LIQUID_EXTINCTION_GAP = 500.0
"""How far above its highest liquid gate a profile's signal may still run (m).

An extinguishing liquid layer kills the signal within a few hundred metres of
its top (the beam decays through the cloud and is lost below the noise floor);
`grow_liquid` has already absorbed the halo. Signal running on much further --
kilometres of aerosol above a surface-pass liquid speck in a dust column -- shows
the beam went on, and such a profile is neither attenuated by that liquid nor a
sample of the saturation plateau."""

MIN_SATURATION_PROFILES = 200
"""Liquid-topped profiles needed to estimate the saturation integral per file.

With fewer, the median of the liquid-top integrals could be a handful of thin
broken cumuli rather than the saturation plateau, and an underestimated plateau
would over-mark attenuation; the integral rule is then skipped."""


class Target(IntEnum):
    """Target classification categories."""

    CLEAR = 0
    DROPLET = 1
    DRIZZLE_OR_RAIN = 2
    ICE = 3
    SUPERCOOLED = 4
    AEROSOL = 5
    ATTENUATED = 6


@dataclass
class Classification:
    """Result of `classify`.

    Attributes:
        time: Time (from the ceilometer).
        range: Range (m).
        target: Target category per pixel (`Target` values), time x range.
        droplet: Liquid droplet layers.
        cold: Sub-freezing region (above the 0 degC level).
        aerosol: Aerosol (all other signal).
        attenuated: Beam extinguished below: the signal-free void above a profile's
            top where the lidar could not see (see `_find_attenuated`).
        quality: True where model temperature was extrapolated (lower quality).
        tw: Model wet-bulb temperature on the (time, range) grid (K).
        t0_alt: Altitude of the 0 degC isotherm per profile (m), time.
        strong_beta: Backscatter threshold used to split cloud/precip from aerosol.
        beam_saturation: Integrated attenuated backscatter at which the beam is
            extinguished (sr-1), as used by the attenuation rule; None when the
            integral rule was not applied (no estimate available or disabled).
    """

    time: npt.NDArray[np.object_]
    range: npt.NDArray[np.floating]
    target: npt.NDArray[np.integer]
    droplet: npt.NDArray[np.bool_]
    cold: npt.NDArray[np.bool_]
    ice: npt.NDArray[np.bool_]
    rain: npt.NDArray[np.bool_]
    aerosol: npt.NDArray[np.bool_]
    attenuated: npt.NDArray[np.bool_]
    quality: npt.NDArray[np.bool_]
    tw: npt.NDArray[np.floating]
    t0_alt: npt.NDArray[np.floating]
    strong_beta: float
    beam_saturation: float | None = None


def classify(
    ceilo: Ceilo,
    model: str | PathLike | Model,
    *,
    altitude: float | None = None,
    use_wet_bulb: bool = True,
    strong_beta: float | None = None,
    speckle_min: int = 3,
    ice_depol_limit: float = ICE_DEPOL_LIMIT,
    ms_tail: float = MS_TAIL,
    drizzle_source_window: int = 0,
    find_surface_liquid: bool = True,
    beam_saturation: float | Literal["auto"] | None = "auto",
) -> Classification:
    """Classify ceilometer targets: liquid layers + 0 degC line, rest aerosol.

    Strong backscatter (`beta > strong_beta`) that is not a liquid layer is
    precipitation/cloud: drizzle/rain where the air is above 0 degC, ice where it
    is below. Weaker signal is aerosol. A speckle filter then clears isolated
    pixels left by screening noise, and the signal-free void above a profile
    whose beam was extinguished (liquid cloud, precipitation without a visible
    source, an abrupt cloud-bright top, or a saturated backscatter integral) is
    marked `ATTENUATED` rather than clear.

    Args:
        ceilo: A `Ceilo` with screened `beta` (any instrument except LD40).
        model: A Cloudnet model file path, or a pre-built `Model`.
        altitude: Site altitude (m a.s.l.) to align the model profile onto the
            ceilometer grid; see `read_model`. Ignored if `model` is a `Model`.
        use_wet_bulb: Use wet-bulb temperature (recommended) instead of dry-bulb.
        strong_beta: Backscatter above which signal is cloud/precipitation rather
            than aerosol (sr-1 m-1). `None` (default) picks it from the data,
            just past the aerosol peak (see `_adaptive_strong_beta`), so it adapts
            to each site/day's aerosol load instead of a fixed value.
        speckle_min: Minimum number of classified (non-clear) pixels in the 3x3
            neighbourhood, including the pixel itself, for it to survive; below
            this it is cleared as speckle. Set to 1 to disable.
        ice_depol_limit: Depolarization above which a target is ice rather than
            liquid (CL61 only). See `ICE_DEPOL_LIMIT`.
        ms_tail: Distance above a liquid layer over which rising depolarization is
            treated as multiple scattering, not ice (m). See `MS_TAIL`.
        drizzle_source_window: Drizzle/rain is kept only where it connects, through
            continuous signal, to a hydrometeor source (a liquid layer or ice)
            directly above it -- precipitation and its parent cloud are one
            contiguous column (see `_source_connected`). The default (0) requires
            the source in the same profile. A positive value dilates the source
            mask by +/- that many profiles in time first, recovering drizzle at a
            ragged cloud edge where the base flickers out for a profile -- but keep
            it small: a wide window re-admits cloud-free bright aerosol as drizzle
            wherever a cloud passes within the window. The gate decides per
            shaft, not per pixel: gated rain is flooded through the
            2D-connected strong warm pixels that have a cloud above them in
            their own profile (see `_flood_connected` and the comment at the
            call site), so a broken cloud deck whose source path flickers
            profile to profile does not shred the shaft into drizzle/aerosol
            stripes. A negative value disables the gate entirely (any bright
            warm signal is drizzle).
        find_surface_liquid: Detect fog / low stratus from the lowest range gates
            (the surface pass of `find_liquid`). Disable it when the instrument's
            near-surface overlap correction is unreliable and would otherwise flag
            a spurious surface liquid layer.
        beam_saturation: Integrated attenuated backscatter (sr-1) at which the
            beam is extinguished, for the integral attenuation rule (see
            `SATURATION_FRACTION`). `"auto"` (default) estimates it per file from
            the profiles topped by a liquid layer (`_beam_saturation`), which
            makes the rule independent of the absolute calibration; a float
            fixes it (e.g. 0.03 for a calibrated ceilometer); `None` disables
            the integral rule, leaving the other attenuation rules in place.

    Returns:
        A `Classification` on the ceilometer time/range grid.
    """
    if ceilo.beta is None:
        msg = "Ceilo has no screened beta; cannot classify"
        raise ValueError(msg)
    if not isinstance(model, Model):
        model = read_model(
            model, ceilo.time, ceilo.range, altitude=altitude, use_wet_bulb=use_wet_bulb
        )

    beta = ma.masked_greater(ma.asarray(ceilo.beta), MAX_PHYSICAL_BETA)
    depol = None if ceilo.depol is None else ma.asarray(ceilo.depol)
    tw = model.tw
    height = np.asarray(ceilo.range, dtype=float)
    beta_mask = ma.getmaskarray(beta)
    if strong_beta is None:
        strong_beta = _adaptive_strong_beta(beta)

    signal = ~beta_mask
    bright = signal & (ma.filled(beta, 0.0) > strong_beta)
    freezing = find_freezing_region(tw, height)
    cold = freezing
    droplet = find_liquid(
        beta, height, surface_pass=find_surface_liquid, strong_beta=strong_beta
    )
    # High-confidence ice, used only to stop liquid from growing into obvious ice.
    blocked = find_falling(beta, height, tw)
    ice_like = None
    if depol is not None:
        droplet, ice_like, cold = _depol_adjustments(
            depol,
            droplet,
            blocked,
            freezing,
            bright,
            beta_mask,
            height,
            ice_depol_limit=ice_depol_limit,
            ms_tail=ms_tail,
        )
        # Depol-confirmed ice, not find_falling's altitude/temperature heuristic,
        # is the barrier to liquid growth (see _depol_adjustments).
        blocked = ice_like
    droplet = fill_thin_clouds(droplet, ~beta_mask, blocked, height)
    droplet = grow_liquid(droplet, ~beta_mask, blocked, height)
    droplet = correct_supercooled(droplet, tw)

    if ice_like is not None:
        # CL61 only: a depol-confirmed ice base sitting ABOVE the model 0 degC line
        # means the real melting level is higher than the model t0 (biased low):
        # the cold, cloud-strength, low-depol band between them is ice melting into
        # rain, not ice. Raise the ice/rain boundary to the observed ice base by
        # dropping that band from `cold`, so it classifies as drizzle/rain -- the
        # symmetric counterpart of _extend_cold_to_ice. The band is keyed on the
        # LOW ice/liquid depol limit, not the ice-core limit the downward extension
        # uses: above the (biased-low) t0 the air is genuinely warm and the melted
        # drops are near-spherical (depol < 0.15), whereas a 0.15-0.30 depol there
        # is still ice and must not be stripped to rain. Keep any find_liquid
        # supercooled droplets (they have a real backscatter peak) as cold.
        melt_band = _melt_band_below_ice(ice_like, freezing, bright, height)
        cold = cold & ~(melt_band & ~droplet)
    strong = bright & ~droplet
    # Ice is strong sub-freezing signal; with depolarization, also faint but
    # strongly-depolarizing sub-freezing signal -- thin cirrus the backscatter
    # threshold misses but whose non-spherical scattering marks it as ice.
    ice = strong & cold
    if ice_like is not None:
        ice = ice | (cold & ice_like)
    else:
        # No depolarization: faint upper-tropospheric cirrus can sit below the
        # adaptive backscatter threshold and would fall through to aerosol. Deep
        # cold (< -25 degC) elevated signal is ice regardless of strength -- a
        # temperature guard that recovers such cirrus while staying well clear of
        # lofted dust, which only reaches ~-19 degC even when elevated.
        deep_ice = find_falling(beta, height, tw, cold_limit=DEEP_COLD_LIMIT)
        ice = ice | (cold & deep_ice)
        # Faint cold signal contiguous with that ice is the same cloud: fill it
        # as ice too (virga below the deep-cold line / the bright core), see
        # _extend_ice_to_cloud_base.
        ice = _extend_ice_to_cloud_base(ice, cold & signal, height)
    # Drizzle/rain is strong warm signal that connects up through continuous
    # signal to a cloud source (see _source_connected); bright warm signal with
    # no cloud above is aerosol, not drizzle (e.g. the cloud-free marine haze at
    # Mindelo).
    rain = strong & ~cold
    if drizzle_source_window >= 0:
        sourced = rain & _source_connected(
            droplet | ice,
            signal,
            drizzle_source_window,
            max_gap=_n_elements(height, DRIZZLE_SOURCE_MAX_GAP),
        )
        # Columns deep enough to be rain on the evidence of their depth alone
        # (fully attenuated profiles, nothing detectable above them) also seed
        # the flood; see DEEP_RAIN_THICKNESS.
        sourced |= rain & ~_thin_runs(rain, height, DEEP_RAIN_THICKNESS)
        # A precipitation shaft under a broken cloud deck is one contiguous
        # bright region, but the per-pixel source path flickers (the cloud base
        # drops out of detection, or a screened gap severs the column), so the
        # strict gate shreds the shaft into drizzle/aerosol stripes. Let the
        # gate decide per shaft instead: flood the sourced rain through the
        # 2D-connected strong warm region, but only across pixels that have a
        # cloud somewhere above them in their own profile (continuity to it no
        # longer required). That confinement keeps the flood out of a
        # persistent cloud-free bright layer (the Mindelo haze, one connected
        # region spanning the whole day) that touches a genuine drizzle column
        # somewhere; a fully attenuated profile inside a shaft still rides on
        # the `sourced` pixels themselves being passable.
        # Runs deep enough to plausibly be attenuated rain (though not deep
        # enough to prove it, see DEEP_RAIN_PASSABLE_THICKNESS) are passable
        # too, so a ground-reaching column beside a sourced one joins the shaft.
        cloud_above = np.zeros_like(rain)
        cloud_above[:, :-1] = (
            np.cumsum((droplet | ice)[:, ::-1], axis=1)[:, ::-1][:, 1:] > 0
        )
        passable = (
            sourced
            | cloud_above
            | ~_thin_runs(rain, height, DEEP_RAIN_PASSABLE_THICKNESS)
        )
        rain = _flood_connected(sourced, rain & passable)
    aerosol = signal & ~droplet & ~ice & ~rain

    target = _assemble(droplet, cold, ice, rain, aerosol)
    target = _despeckle(target, speckle_min)
    integral = _beta_integral(beta, height)
    if beam_saturation == "auto":
        beam_saturation = _beam_saturation(target, integral, height)
    attenuated = _find_attenuated(
        target, bright, height, integral=integral, saturation=beam_saturation
    )
    target = np.where(attenuated, Target.ATTENUATED, target)

    return Classification(
        time=ceilo.time,
        range=ceilo.range,
        target=target,
        droplet=droplet,
        cold=cold,
        ice=ice,
        rain=rain,
        aerosol=aerosol,
        attenuated=attenuated,
        quality=model.extrapolated,
        tw=tw,
        t0_alt=_find_t0_alt(tw, height),
        strong_beta=strong_beta,
        beam_saturation=beam_saturation,
    )


def _depol_adjustments(
    depol: ma.MaskedArray,
    droplet: npt.NDArray[np.bool_],
    blocked: npt.NDArray[np.bool_],
    freezing: npt.NDArray[np.bool_],
    bright: npt.NDArray[np.bool_],
    beta_mask: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    ice_depol_limit: float,
    ms_tail: float,
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    """Use depolarization (CL61 only) to split ice from liquid and anchor cold air.

    Strong depolarization marks ice, not liquid droplets -- except inside a
    liquid layer, where rising depolarization is MULTIPLE SCATTERING, not ice; a
    flat threshold would carve out the layer's densest part. Genuine liquid
    layers (those with a low-depol single-scattering part, which pure ice lacks)
    are therefore shielded from the depol veto as a whole, plus a short
    scattering tail above them (`ms_tail`).

    The freezing region is then anchored to observed falling ice (see
    `_extend_cold_to_ice`), flooding only through CLOUD-STRENGTH, SOLID ice.
    Cloud-strength (`bright`) excludes a daytime boundary layer of
    weakly-backscattering yet strongly-depolarizing aerosol (dust, pollen). The
    ice-core depol limit (not the lower ice/liquid limit) marks SOLID ice: as
    ice falls past the melting level depolarization drops from solid ice
    ~0.4-0.5 to the ~0.2 of the wet/rain shaft below it. Flooding the freezing
    region down only through solid ice therefore STOPS at that phase change,
    instead of running on through the still-depolarizing rain below 0 degC
    (which would drag ice far into warm air). This higher limit is for the
    downward extension only: above the 0 degC line a 0.2 depol is still cold
    ice, not rain (see the melt-band block in `classify`).

    Returns:
        Updated ``(droplet, ice_like, cold)`` masks. The caller should also make
        `ice_like` the barrier to liquid growth: with depolarization we know ice
        directly, so it -- not `find_falling`'s -15 degC / 2000 m heuristic --
        blocks growth. That heuristic would otherwise cut a genuine supercooled
        cloud top off at altitude (e.g. at high-elevation Troll, where the layer
        sits near the 2000 m line); depolarization keeps growth out of real ice
        instead. `find_falling` stays the barrier only for instruments without
        depol.
    """
    ice_like = find_depol_ice(depol, beta_mask, ice_depol_limit=ice_depol_limit)
    liquid = _fill_runs(droplet & ~ice_like, droplet, height)
    ms_protected = grow_liquid(
        liquid, ~beta_mask, blocked, height, grow_up=ms_tail, grow_down=0.0
    )
    ice_like = ice_like & ~ms_protected
    droplet = droplet & ~ice_like
    ice_core = ice_like & (ma.filled(depol, 0.0) > ICE_CORE_DEPOL_LIMIT)
    cold = _extend_cold_to_ice(freezing, ice_core & bright, height, ~beta_mask, bright)
    return droplet, ice_like, cold


def _extend_cold_to_ice(
    cold: npt.NDArray[np.bool_],
    ice_like: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    signal: npt.NDArray[np.bool_] | None = None,
    bright: npt.NDArray[np.bool_] | None = None,
    *,
    max_depth: float = 1500.0,
    smooth_window: int = 10,
    bridge: float = 150.0,
) -> npt.NDArray[np.bool_]:
    """Extend the freezing region downward through ice connected to it.

    A biased-high model 0 degC level leaves depol-confirmed ice on its warm side
    (falling ice not yet melted). Starting from the model freezing region, flood
    downward through `ice_like` gates -- but no more than `max_depth` below the
    original boundary, so a deep depolarizing layer (e.g. lofted dust touching
    cloud) cannot drag the whole column sub-freezing.

    Strict gate-by-gate contiguity is too fragile for two real interruptions
    (the Payerne melting-level case):

    - a thin low-depol band inside the ice -- an embedded supercooled layer, or
      near-t0 crystals below the ice-core limit -- severs the chain right below
      the cold base. Non-ice runs no thicker than `bridge` metres are therefore
      passable; the rain shaft below the true melt is a kilometre-scale
      low-depol run and still blocks.
    - heavy precipitation extinguishes the beam before the freezing region, so
      the whole observed ice cloud hangs in clear (masked) air below the model
      cold region. With `signal` and `bright` given, the masked void above each
      profile's signal top is passable -- but only when the beam died abruptly
      in cloud-bright signal (extinction inside cloud/precip). An aerosol layer
      instead fades gradually below the threshold before masking, so its top is
      not bright and clear air above it remains a barrier.

    The extension is claimed only down to the lowest `ice_like` gate actually
    reached (a bridged gap never dangles below the ice), and the resulting base
    is smoothed against single-profile pillars (a noisy depolarizing column
    flooding to the ground): the base height is replaced by its rolling median
    over +/-`smooth_window` profiles and the extension is clipped to it. This
    keys on the base being an outlier, not on a profile count, so it is
    insensitive to the time averaging and to clustered pillars (the median
    tolerates a minority of them).
    """
    max_steps = max(_n_elements(height, max_depth), 1)
    passable = ice_like | _thin_runs(~ice_like, height, bridge)
    if signal is not None and bright is not None:
        has, top = _signal_top(signal)
        # Abrupt extinction: still cloud-bright at (or one gate below) the top.
        abrupt = has & _at_top(bright, top)
        void = ~signal & (
            np.arange(signal.shape[1])[np.newaxis, :] > top[:, np.newaxis]
        )
        passable |= void & abrupt[:, np.newaxis]
    extended = _grow_range(cold, passable, max_steps, up=False)
    # Claim only down to the lowest ice gate reached: a bridged or void gap must
    # lead to ice, never end the extension inside itself.
    reached = extended & ice_like
    floor = np.where(reached.any(axis=1), np.argmax(reached, axis=1), extended.shape[1])
    keep = np.arange(extended.shape[1])[np.newaxis, :] >= floor[:, np.newaxis]
    extended = cold | (extended & keep)
    if not (extended & ~cold).any():
        return extended
    return _clip_to_median_base(cold, extended, smooth_window)


def _clip_to_median_base(
    original: npt.NDArray[np.bool_],
    extended: npt.NDArray[np.bool_],
    smooth_window: int,
) -> npt.NDArray[np.bool_]:
    """Clip a downward extension of `original` against its rolling-median base.

    Guards against single-profile pillars (a noisy column flooding toward the
    ground): the per-profile base (lowest True gate of `extended`) is replaced
    by its rolling median over +/-`smooth_window` profiles, and extension gates
    below that floor are dropped (`original` itself is never clipped). This keys
    on the base being an outlier, not on a profile count, so it is insensitive
    to the time averaging and to clustered pillars (the median tolerates a
    minority of them). Profiles with nothing in `extended` are NaN and skipped
    by the windowed median rather than dragging the floor upward; an all-NaN
    window leaves the floor at 0 (keep the extension), and a profile with a base
    always contributes its own value, so its mask is never fully clipped.
    """
    n_time, n_gate = extended.shape
    base = np.where(extended.any(axis=1), np.argmax(extended, axis=1), np.nan)
    floor = np.zeros(n_time, dtype=int)
    for t in range(n_time):
        window = base[max(0, t - smooth_window) : t + smooth_window + 1]
        if not np.all(np.isnan(window)):
            floor[t] = int(np.ceil(np.nanmedian(window)))
    keep = np.arange(n_gate)[np.newaxis, :] >= floor[:, np.newaxis]
    return original | (extended & keep)


def _extend_ice_to_cloud_base(
    ice: npt.NDArray[np.bool_],
    allowed: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    smooth_window: int = 10,
) -> npt.NDArray[np.bool_]:
    """Extend ice through each contiguous cold-signal run that contains ice.

    No-depol instruments only (with depolarization faint ice is confirmed
    directly). Faint cold signal attached, through unbroken backscatter, to
    confirmed ice in the same profile is the same cloud -- ice virga hanging
    below a cirrus deck -- yet on a low-biased instrument (e.g. the DA10 DIAL,
    ~20% below the CL61) it sits under the cloud threshold, and below the
    deep-cold line the -25 degC guard cannot recover it either: the deck then
    ends in a conspicuously flat aerosol boundary at the -25 degC isotherm.
    Connectivity is calibration-independent, so filling the whole run as ice
    sidesteps the bias. Not in CloudnetPy, whose radar sees the virga directly.

    The run's own bounds are the physical guards: a clear-air gap or the 0 degC
    level (via the cold mask in `allowed`) ends the fill, which keeps warm
    columns ice-free without an altitude floor (that would hurt polar sites,
    where cold air reaches low); a run-thickness cap would not separate cirrus
    from dust and is deliberately absent. No upward growth is needed: above the
    -25 degC line a signal run is already deep-cold ice except gates the
    speckle filter dropped on purpose. The extension's base is clipped against
    its rolling median (`_clip_to_median_base`), so a transient single-profile
    contact between e.g. a dust top and an overlying cloud base cannot flood
    the dust layer; a cold aerosol run that touches no ice at all is never
    filled in the first place.
    """
    filled = _fill_runs(ice, allowed, height)
    if not (filled & ~ice).any():
        return ice
    return _clip_to_median_base(ice, filled, smooth_window)


def _thin_runs(
    mask: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    max_thickness: float,
) -> npt.NDArray[np.bool_]:
    """Mark gates in vertical `mask` runs no thicker than `max_thickness` metres."""
    h = np.asarray(height, dtype=float)
    out = np.zeros_like(mask)
    for i in np.nonzero(mask.any(axis=1))[0]:
        for j, k in _iter_runs(mask[i]):
            if h[k - 1] - h[j] <= max_thickness:
                out[i, j:k] = True
    return out


def _melt_band_below_ice(
    ice_like: npt.NDArray[np.bool_],
    freezing: npt.NDArray[np.bool_],
    bright: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    max_depth: float = 1500.0,
    bridge: float = 150.0,
) -> npt.NDArray[np.bool_]:
    """Mark the melting band: low-depol signal linking a depol-ice base to t0.

    Symmetric counterpart of `_extend_cold_to_ice`: when the depol ice base sits
    ABOVE the model 0 degC line, the real melting level is higher than the model
    t0, and the cold (per the model), cloud-strength, low-depol band between them
    is ice melting into rain, mislabelled as ice.

    The band is the cloud-strength signal in the freezing region that is reachable
    BOTH downward from the depol ice base (`ice_like & freezing`) AND upward from
    the warm region below t0 (`~freezing`) -- a continuous column linking the ice
    down to the rain across the melting level. Requiring both ends excludes
    low-depol patches buried inside an ice cloud well above t0 (reachable from ice
    but not from the warm region) and supercooled layers with no ice above them.

    The melting layer itself depolarizes (wet, irregular particles), so a thin
    high-depol enhancement at the melt level would otherwise read as ice and break
    the link. A vertically thin `ice_like` run (no thicker than `bridge` metres)
    is therefore treated as part of the band, while a thick, coherent ice cloud
    still blocks. Each flood is capped at `max_depth`.
    """
    max_steps = max(_n_elements(height, max_depth), 1)
    # The melting enhancement is a thin ice_like run; bridge it but keep thick ice
    # (the real cloud) as a barrier.
    thin_ice = _thin_runs(ice_like, height, bridge)
    passable = freezing & bright & (~ice_like | thin_ice)
    # Flood down from the depol ice base, up from the warm region below t0; the
    # band is where the two floods meet.
    down = _grow_range(ice_like & freezing, passable, max_steps, up=False)
    up = _grow_range(~freezing, passable, max_steps, up=True)
    return down & up & passable


def _flood_connected(
    seed: npt.NDArray[np.bool_],
    allowed: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Flood `seed` through 4-connected `allowed` pixels (time-range).

    Marks every `allowed` pixel whose connected region contains a seed. Runs
    alternating full-line fills along range and time (each pass fills whole
    contiguous runs, not single steps) until a fixpoint, so convergence takes a
    few passes even for shafts spanning hours.
    """
    grown = seed & allowed
    while True:
        prev = grown
        grown = _fill_axis_runs(grown, allowed)
        grown = _fill_axis_runs(grown.T, allowed.T).T
        if (grown == prev).all():
            return grown


def _fill_axis_runs(
    seed: npt.NDArray[np.bool_],
    allowed: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Fill each contiguous run of `allowed` along the last axis that has a seed."""
    out = seed.copy()
    for i in np.nonzero(seed.any(axis=1))[0]:
        for j, k in _iter_runs(allowed[i]):
            if seed[i, j:k].any():
                out[i, j:k] = True
    return out


def _source_connected(
    cloud: npt.NDArray[np.bool_],
    signal: npt.NDArray[np.bool_],
    time_window: int,
    max_gap: int = 0,
) -> npt.NDArray[np.bool_]:
    """Mark gates with a cloud ABOVE them, reachable through (near-)continuous signal.

    Drizzle/rain hangs *below* its cloud: the precipitation and its source form
    one contiguous signal column with the cloud on top. A gate is "sourced" only
    when a `cloud` gate (a liquid layer or ice) lies above it and every gate
    between carries `signal`. This rejects two false sources:

    - a bright layer separated from everything above by clear air -- e.g. a
      near-surface aerosol blob far below an unrelated cirrus (the path is broken
      by the clear air);
    - a cloud *below* the layer -- e.g. haze sitting above a shallow surface fog
      (the cloud is not above it).

    A clear-air run no longer than `max_gap` gates, bounded by signal on both
    sides, is bridged before the path is traced: a melting layer's backscatter
    notch is often screened out, and that thin masked band must not sever a
    drizzle column from the ice above it (a large clear gap, with a longer
    unbridged core, is still broken). The cloud mask is first dilated by
    +/-`time_window` profiles in time, so a single-profile gap in cloud detection
    at a ragged cloud edge does not drop a genuine drizzle shaft beside it.
    """
    src = cloud
    for _ in range(max(time_window, 0)):
        grown = src.copy()
        grown[1:] |= src[:-1]
        grown[:-1] |= src[1:]
        src = grown
    if max_gap > 0:
        signal = signal.copy()
        n_gate = signal.shape[1]
        for i in range(signal.shape[0]):
            for j, k in _iter_runs(~signal[i]):
                if j > 0 and k < n_gate and k - j <= max_gap:
                    signal[i, j:k] = True
    # Propagate "a cloud sits above, through unbroken signal" downward gate by
    # gate. Gate g inherits from the gate above (g+1) only when that gate carries
    # signal, so a clear-air gate breaks the path.
    above = np.zeros_like(src)
    for g in range(src.shape[1] - 2, -1, -1):
        above[:, g] = signal[:, g + 1] & (src[:, g + 1] | above[:, g + 1])
    return above


def _adaptive_strong_beta(
    beta: ma.MaskedArray,
    *,
    n_bins: int = 60,
    shoulder_frac: float = 0.05,
    prominence_frac: float = 0.03,
    valley_frac: float = 0.5,
    max_peak_ratio: float = 25.0,
    max_strong_beta: float = 1e-5,
    min_cloud_beta: float = 3e-6,
    min_aerosol_beta: float = 1e-7,
    default: float = 3e-6,
) -> float:
    """Pick the cloud/aerosol backscatter threshold from the data distribution.

    Ceilometer backscatter has a low-value aerosol/background mode; cloud and
    precipitation form a weaker high tail or a second, higher mode. We anchor on
    the aerosol mode -- the *lowest* prominent peak, not necessarily the tallest
    (on a cloudy day the cloud mode can hold more pixels) -- and place the
    threshold past it. Modes below `min_aerosol_beta` are never anchor
    candidates: aerosol backscatter lives around 1e-7..1e-5, so a lower mode is
    molecular scattering or residual noise -- a 532 nm PollyXT sees the Rayleigh
    return (~2e-9 at sea level) in clean-air gaps that a 910 nm ceilometer's
    noise floor never resolves. Anchored on such a mode, the `max_peak_ratio`
    cap would land mid-aerosol-mode and turn the whole aerosol population
    "bright" (marine haze becoming drizzle at Mindelo). The threshold is placed:

    - if a higher *cloud-bright* mode exists (bimodal), at the **valley** (lowest
      count) between the aerosol mode and that next mode. The higher mode must be
      both >2x the aerosol value and above `min_cloud_beta`: a second mode still
      at aerosol-level backscatter is layered aerosol (e.g. lofted Saharan dust
      over Granada), not cloud, and must not be split off as drizzle;
    - otherwise at the aerosol mode's right **shoulder** (where its count first
      falls below `shoulder_frac` of the peak).

    This adapts to each site/day's aerosol load (e.g. a dusty Granada day vs a
    clean one) instead of a fixed value. The result is capped two ways: at
    `max_peak_ratio` times the peak, and at the absolute `max_strong_beta`. The
    latter matters when aerosol and cloud are not cleanly separated (a polar
    winter continuum of low cloud, where the anchor can land on the cloud mode
    and the threshold would otherwise run off): aerosol backscatter does not
    physically exceed ~1e-5 sr-1 m-1, so anything above that is cloud. The former
    (`max_peak_ratio`) is a heuristic, load-scaling backstop -- not a physical or
    tuned value: it keeps the threshold within a factor of the aerosol mode when
    no shoulder/valley is found, and is the tighter bound for a faint mode where
    the absolute cap is too generous. It is deliberately loose (~1.4 decades above
    the mode) and never binds on the regression cases, so its exact value is not
    critical.

    Uses the whole column: the threshold is anchored on the low aerosol peak and
    the high tail is ignored, so cloud aloft must stay in the distribution for the
    aerosol->cloud valley to be found (restricting to low gates removes it and
    collapses the threshold onto the aerosol mode). Returns `default` when there
    are too few samples.
    """
    values = ma.filled(ma.asarray(beta), np.nan).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 1000:
        return default
    lo, hi = np.percentile(values, [1.0, 99.9])
    if not hi > lo:
        return default
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    # Light smoothing so noise does not create spurious peaks/troughs.
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    # Aerosol peak: default to the tallest mode, but prefer a lower-value mode
    # when one exists and a real valley separates it from the tallest peak (a
    # cloudy day's cloud mode can out-count the aerosol mode -- Ny-Alesund). The
    # valley test rejects mere noise bumps on the aerosol mode's rising edge
    # (Cluj/Granada), which have no dip between them and the peak.
    higher_left = np.r_[True, smooth[1:] > smooth[:-1]]
    higher_right = np.r_[smooth[:-1] > smooth[1:], True]
    is_peak = higher_left & higher_right & (smooth >= smooth.max() * prominence_frac)
    peak_idx = np.flatnonzero(is_peak)
    # Sub-aerosol modes (molecular return / residual noise, see docstring) are
    # never anchor candidates.
    peak_idx = peak_idx[centers[peak_idx] >= min_aerosol_beta]
    tallest = int(np.argmax(smooth))
    peak = tallest
    if centers[tallest] < min_aerosol_beta and peak_idx.size:
        # Even the tallest mode is sub-aerosol (a pristine molecular day):
        # anchor on the lowest genuine mode instead.
        peak = int(peak_idx[0])
    for p in peak_idx:
        if p >= tallest:
            break
        if smooth[p : tallest + 1].min() <= valley_frac * smooth[p]:
            peak = int(p)
            break
    # A higher, cloud-bright mode (>2x the aerosol value and above
    # `min_cloud_beta`) is cloud/precip: put the threshold at the valley between
    # the two. A second mode still at aerosol-level backscatter is layered aerosol,
    # not cloud, so fall through to the aerosol shoulder instead.
    higher = peak_idx[
        (peak_idx > peak)
        & (centers[peak_idx] > 2 * centers[peak])
        & (centers[peak_idx] > min_cloud_beta)
    ]
    if higher.size:
        cloud = int(higher[0])
        threshold = centers[peak + int(np.argmin(smooth[peak : cloud + 1]))]
    else:
        # Aerosol only (possibly layered). Anchor the shoulder on the *highest*
        # aerosol mode, so a secondary aerosol layer (e.g. lofted dust) stays
        # below the threshold rather than being split off by the shoulder landing
        # in the valley beneath it.
        aerosol = peak_idx[centers[peak_idx] <= min_cloud_beta]
        anchor = int(aerosol[-1]) if aerosol.size else peak
        threshold = centers[-1]
        shoulder = smooth[anchor] * shoulder_frac
        for i in range(anchor + 1, len(smooth)):
            if smooth[i] < shoulder:
                threshold = centers[i]
                break
    return float(min(threshold, centers[peak] * max_peak_ratio, max_strong_beta))


def _signal_top(
    signal: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.intp]]:
    """Per profile: whether any signal exists, and the index of its highest gate."""
    n_gate = signal.shape[1]
    has = signal.any(axis=1)
    top = np.where(has, n_gate - 1 - np.argmax(signal[:, ::-1], axis=1), 0)
    return has, top


def _at_top(
    mask: npt.NDArray[np.bool_], top: npt.NDArray[np.intp]
) -> npt.NDArray[np.bool_]:
    """Per profile: `mask` at the top gate or one gate below it."""
    rows = np.arange(mask.shape[0])
    return mask[rows, top] | mask[rows, np.maximum(top - 1, 0)]


def _beta_integral(
    beta: ma.MaskedArray, height: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Cumulative integral of attenuated backscatter from the bottom gate up (sr-1).

    Masked (no-signal) and negative gates contribute nothing.
    """
    values = np.clip(ma.filled(beta, 0.0), 0.0, None)
    return np.cumsum(values * np.gradient(height), axis=1)


def _topmost_runs(
    target: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Per profile: has-signal flag, highest signal gate, and the run ending there.

    The run is the contiguous block of classified gates ending at the top gate.
    """
    signal = target != Target.CLEAR
    has, top = _signal_top(signal)
    idx = np.arange(target.shape[1])[np.newaxis, :]
    # Base of the run: one above the highest clear gate below the top (or the
    # bottom of the profile).
    clear_below = ~signal & (idx <= top[:, np.newaxis])
    base = np.where(clear_below, idx, -1).max(axis=1) + 1
    in_run = (idx >= base[:, np.newaxis]) & (idx <= top[:, np.newaxis])
    return has, top, in_run


def _liquid_topped(
    target: npt.NDArray[np.integer],
    top: npt.NDArray[np.intp],
    in_run: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    max_gap: float = LIQUID_EXTINCTION_GAP,
) -> npt.NDArray[np.bool_]:
    """Profiles whose topmost run holds liquid within `max_gap` metres of its top."""
    liquid = in_run & np.isin(target, [Target.DROPLET, Target.SUPERCOOLED])
    idx = np.arange(target.shape[1])[np.newaxis, :]
    highest = np.where(liquid, idx, -1).max(axis=1)
    gap = height[top] - height[np.maximum(highest, 0)]
    return (highest >= 0) & (gap <= max_gap)


def _beam_saturation(
    target: npt.NDArray[np.integer],
    integral: npt.NDArray[np.floating],
    height: npt.NDArray[np.floating],
    *,
    min_profiles: int = MIN_SATURATION_PROFILES,
) -> float | None:
    """Estimate the saturated backscatter integral from this file's liquid tops.

    Profiles whose signal ends just above a liquid layer (see `_liquid_topped`)
    are, as a population, beams extinguished in liquid: their top-of-run
    integrals pile up at the instrument's saturation plateau I_sat = 1/(2 eta S)
    (the O'Connor et al. 2004 calibration principle). The median of that
    population is the plateau in the file's own calibration units -- a thin
    cumulus pulls below it and a rain column above it, but the bulk sits on the
    plateau. Returns None with fewer than `min_profiles` such profiles, when the
    estimate would be unreliable (see `MIN_SATURATION_PROFILES`).
    """
    has, top, in_run = _topmost_runs(target)
    liquid_topped = has & _liquid_topped(target, top, in_run, height)
    if liquid_topped.sum() < min_profiles:
        return None
    rows = np.arange(target.shape[0])
    return float(np.median(integral[rows, top][liquid_topped]))


def _find_attenuated(
    target: npt.NDArray[np.integer],
    bright: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    integral: npt.NDArray[np.floating] | None = None,
    saturation: float | None = None,
    saturation_fraction: float = SATURATION_FRACTION,
) -> npt.NDArray[np.bool_]:
    """Mark the void above each profile's signal top where the beam was extinguished.

    A masked pixel above a cloud is ambiguous on its own: clear sky and an
    extinguished beam both return nothing. The decision therefore comes from the
    *topmost signal run* of the profile (the contiguous classified gates ending at
    its highest one) -- only the void above that run is ever marked, since any
    signal higher up proves the beam got through. The run extinguishes the beam
    when any of these hold:

    - it ends within `LIQUID_EXTINCTION_GAP` of a **liquid** layer (droplet or
      supercooled): a liquid cloud of even modest water path is optically thick
      enough (tau ~ 3) to kill a lidar beam, and `find_liquid` keys on the
      peak-then-sharp-decay signature that is this very attenuation. A liquid
      layer with signal running on well above it -- a separate higher run, or
      kilometres of aerosol over a surface-pass speck -- evidently let the beam
      through and is not used;
    - its highest hydrometeor is **drizzle/rain** with no droplet or ice above it
      in the run: precipitation needs a cloud above, so not seeing one means the
      beam died inside the rain -- the same reasoning as `DEEP_RAIN_THICKNESS`;
    - it ends **abruptly** while still cloud-bright in a hydrometeor class at (or
      one gate below) its top, e.g. a thick ice/snow column. Aerosol fades
      gradually below the threshold before masking, so the void above an aerosol
      layer stays clear (bright-but-unsourced aerosol such as marine haze is
      excluded by requiring a hydrometeor class); likewise faint cirrus;
    - with `integral` and `saturation` given, the backscatter **integral** at its
      top has reached `saturation_fraction` of the saturation plateau, i.e. that
      fraction of the beam is demonstrably lost (see `SATURATION_FRACTION`). This
      is the physical criterion; it catches extinguished columns where no liquid
      peak was detected (a polar low-cloud continuum, broken-cumulus edges).

    Works on the despeckled `target` so screening noise in the void neither
    defines the signal top nor counts as signal above a cloud.
    """
    n_time, n_gate = target.shape
    has, top, in_run = _topmost_runs(target)
    idx = np.arange(n_gate)[np.newaxis, :]
    rows = np.arange(n_time)
    hydrometeor = np.isin(
        target,
        [Target.DROPLET, Target.SUPERCOOLED, Target.ICE, Target.DRIZZLE_OR_RAIN],
    )
    run_hyd = in_run & hydrometeor
    highest_hyd = np.where(run_hyd, idx, -1).max(axis=1)
    rain_topped = (highest_hyd >= 0) & (
        target[rows, np.maximum(highest_hyd, 0)] == Target.DRIZZLE_OR_RAIN
    )
    extinguished = (
        _liquid_topped(target, top, in_run, height)
        | rain_topped
        | _at_top(bright & hydrometeor, top)
    )
    if integral is not None and saturation is not None:
        extinguished |= integral[rows, top] >= saturation_fraction * saturation
    return (has & extinguished)[:, np.newaxis] & (idx > top[:, np.newaxis])


def _despeckle(
    target: npt.NDArray[np.integer], min_neighbours: int
) -> npt.NDArray[np.integer]:
    """Clear classified pixels with too few classified neighbours (speckle).

    Counts non-clear pixels in each pixel's 3x3 neighbourhood (itself included)
    and resets to `CLEAR` those below `min_neighbours`. A no-op when
    `min_neighbours <= 1`.
    """
    if min_neighbours <= 1:
        return target
    classified = target != Target.CLEAR
    counts = _window_count(_window_count(classified, half=1, axis=0), half=1, axis=1)
    speckle = classified & (counts < min_neighbours)
    return np.where(speckle, Target.CLEAR, target)


def _assemble(
    droplet: npt.NDArray[np.bool_],
    cold: npt.NDArray[np.bool_],
    ice: npt.NDArray[np.bool_],
    rain: npt.NDArray[np.bool_],
    aerosol: npt.NDArray[np.bool_],
) -> npt.NDArray[np.integer]:
    """Combine category bits into target codes (later rules overwrite earlier).

    Liquid layers sit on top: `droplet & cold` is supercooled liquid. Strong
    non-liquid signal is ice (cold) or drizzle/rain (warm); the rest is aerosol.
    """
    out = np.zeros(droplet.shape, dtype=int)
    out[aerosol] = Target.AEROSOL
    out[ice] = Target.ICE
    out[rain] = Target.DRIZZLE_OR_RAIN
    out[droplet & ~cold] = Target.DROPLET
    out[droplet & cold] = Target.SUPERCOOLED
    return out
