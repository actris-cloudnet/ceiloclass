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


class Target(IntEnum):
    """Target classification categories."""

    CLEAR = 0
    DROPLET = 1
    DRIZZLE_OR_RAIN = 2
    ICE = 3
    SUPERCOOLED = 4
    AEROSOL = 5


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
        quality: True where model temperature was extrapolated (lower quality).
        t0_alt: Altitude of the 0 degC isotherm per profile (m), time.
        strong_beta: Backscatter threshold used to split cloud/precip from aerosol.
    """

    time: npt.NDArray[np.object_]
    range: npt.NDArray[np.floating]
    target: npt.NDArray[np.integer]
    droplet: npt.NDArray[np.bool_]
    cold: npt.NDArray[np.bool_]
    ice: npt.NDArray[np.bool_]
    rain: npt.NDArray[np.bool_]
    aerosol: npt.NDArray[np.bool_]
    quality: npt.NDArray[np.bool_]
    t0_alt: npt.NDArray[np.floating]
    strong_beta: float


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
) -> Classification:
    """Classify ceilometer targets: liquid layers + 0 degC line, rest aerosol.

    Strong backscatter (`beta > strong_beta`) that is not a liquid layer is
    precipitation/cloud: drizzle/rain where the air is above 0 degC, ice where it
    is below. Weaker signal is aerosol. A speckle filter then clears isolated
    pixels left by screening noise.

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
            wherever a cloud passes within the window. A negative value disables
            the gate entirely (any bright warm signal is drizzle).
        find_surface_liquid: Detect fog / low stratus from the lowest range gates
            (the surface pass of `find_liquid`). Disable it when the instrument's
            near-surface overlap correction is unreliable and would otherwise flag
            a spurious surface liquid layer.

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

    beta = ma.asarray(ceilo.beta)
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
    # Drizzle/rain is strong warm signal that connects up through continuous
    # signal to a cloud source (see _source_connected); bright warm signal with
    # no cloud above is aerosol, not drizzle (e.g. the cloud-free marine haze at
    # Mindelo).
    rain = strong & ~cold
    if drizzle_source_window >= 0:
        rain &= _source_connected(
            droplet | ice,
            signal,
            drizzle_source_window,
            max_gap=_n_elements(height, DRIZZLE_SOURCE_MAX_GAP),
        )
    aerosol = signal & ~droplet & ~ice & ~rain

    target = _assemble(droplet, cold, ice, rain, aerosol)
    target = _despeckle(target, speckle_min)

    return Classification(
        time=ceilo.time,
        range=ceilo.range,
        target=target,
        droplet=droplet,
        cold=cold,
        ice=ice,
        rain=rain,
        aerosol=aerosol,
        quality=model.extrapolated,
        t0_alt=_find_t0_alt(tw, height),
        strong_beta=strong_beta,
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
    cold = _extend_cold_to_ice(freezing, ice_core & bright, height)
    return droplet, ice_like, cold


def _extend_cold_to_ice(
    cold: npt.NDArray[np.bool_],
    ice_like: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    max_depth: float = 1500.0,
    smooth_window: int = 10,
) -> npt.NDArray[np.bool_]:
    """Extend the freezing region downward through ice connected to it.

    A biased-high model 0 degC level leaves depol-confirmed ice on its warm side
    (falling ice not yet melted). Starting from the model freezing region, flood
    downward through contiguous `ice_like` gates -- but no more than `max_depth`
    below the original boundary, so a deep depolarizing layer (e.g. lofted dust
    touching cloud) cannot drag the whole column sub-freezing.

    The resulting ice base is then smoothed against single-profile pillars (a
    noisy depolarizing column flooding to the ground): the base height is replaced
    by its rolling median over +/-`smooth_window` profiles and the extension is
    clipped to it. This keys on the base being an outlier, not on a profile count,
    so it is insensitive to the time averaging and to clustered pillars (the
    median tolerates a minority of them).
    """
    max_steps = max(_n_elements(height, max_depth), 1)
    extended = _grow_range(cold, ice_like, max_steps, up=False)
    if not (extended & ~cold).any():
        return extended
    # Lowest cold gate per profile (the ice base); profiles with no ice are NaN
    # so they are skipped by the windowed median rather than dragging the floor
    # upward (a window of mostly clear profiles must not clip a neighbour's ice).
    n_time, n_gate = cold.shape
    has_cold = extended.any(axis=1)
    base = np.where(has_cold, np.argmax(extended, axis=1), np.nan)
    floor = np.zeros(n_time, dtype=int)
    for t in range(n_time):
        lo = max(0, t - smooth_window)
        hi = min(n_time, t + smooth_window + 1)
        window = base[lo:hi]
        # All-NaN windows leave floor at 0 (keep the extension); a profile that
        # has ice always contributes its own non-NaN base here, so its ice is
        # never fully clipped.
        if not np.all(np.isnan(window)):
            floor[t] = int(np.ceil(np.nanmedian(window)))
    keep = np.arange(n_gate)[np.newaxis, :] >= floor[:, np.newaxis]
    return cold | (extended & keep)


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
    default: float = 3e-6,
) -> float:
    """Pick the cloud/aerosol backscatter threshold from the data distribution.

    Ceilometer backscatter has a low-value aerosol/background mode; cloud and
    precipitation form a weaker high tail or a second, higher mode. We anchor on
    the aerosol mode -- the *lowest* prominent peak, not necessarily the tallest
    (on a cloudy day the cloud mode can hold more pixels) -- and place the
    threshold past it:

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
    tallest = int(np.argmax(smooth))
    peak = tallest
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
