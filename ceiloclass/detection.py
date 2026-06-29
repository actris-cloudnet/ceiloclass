"""Lidar + model target detection.

Ported from CloudnetPy's categorize code, restricted to the parts that work
without cloud radar (liquid droplets, freezing region, cold ice, aerosol).
"""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from numpy import ma

from ._interp import local_maxima
from .model import T0

COLD_LIMIT = T0 - 15
"""Temperature below which elevated lidar signal is treated as ice (K)."""

DESPECKLE_ALTITUDE = 4000.0
"""Altitude above which isolated single-gate ice signal is treated as noise (m)."""

SUPERCOOLED_LIMIT = T0 - 38
"""Temperature below which liquid droplets cannot exist (K)."""

ICE_DEPOL_LIMIT = 0.15
"""Lidar depolarization above which a target is non-spherical ice, not liquid.

Liquid droplets are spherical and barely depolarize (depol < 0.1); ice crystals
are non-spherical and depolarize strongly (depol ~ 0.3-0.5). The threshold sits
between the two populations. Only the CL61 provides depolarization.
"""

_PEAK_ORDER = 4
"""Neighbours compared each side when flagging a backscatter peak.

A gate within `_PEAK_ORDER` of the profile edge can never be a strict local
maximum, so the lowest `2 * _PEAK_ORDER` gates form a blind zone that the main
peak search cannot see into (see the surface pass in `find_liquid`).
"""


def find_depol_ice(
    depol: ma.MaskedArray,
    beta_mask: npt.NDArray[np.bool_],
    *,
    ice_depol_limit: float = ICE_DEPOL_LIMIT,
) -> npt.NDArray[np.bool_]:
    """Mark signal whose depolarization is too high to be liquid (i.e. ice).

    This is the depolarization information CloudnetPy gets from radar but a bare
    ceilometer cannot: it both vetoes `find_liquid` false positives (ice virga
    read as droplet layers) and recovers ice that `find_falling` misses below
    its altitude/temperature cutoffs (e.g. ice descending toward the 0 degC
    level). CL61 only; callers pass `depol=None` for other instruments.

    Args:
        depol: Linear depolarization ratio (masked), time x range.
        beta_mask: Mask of the screened backscatter (True where no signal).
        ice_depol_limit: Depolarization above which a target is ice.

    Returns:
        Boolean array, True where signal is present and strongly depolarizing.
    """
    return ~beta_mask & (ma.filled(ma.asarray(depol), 0.0) > ice_depol_limit)


def find_liquid(
    beta: ma.MaskedArray,
    height: npt.NDArray[np.floating],
    *,
    peak_amp: float = 1e-6,
    max_width: float = 250,
    min_points: int = 3,
    min_top_der: float = 1e-7,
    min_alt: float = 100,
) -> npt.NDArray[np.bool_]:
    """Detect liquid droplet layers from attenuated backscatter.

    Port of CloudnetPy's `droplet.find_liquid`, dropping the liquid-water-path
    and radar top-correction steps (not available without a radiometer/radar),
    and extended with a surface pass that recovers fog / very low stratus whose
    peak sits in the blind zone of the local-maximum search.

    Args:
        beta: Screened range-corrected backscatter (sr-1 m-1), time x range.
        height: Range (m).
        peak_amp: Minimum backscatter peak amplitude (sr-1 m-1).
        max_width: Maximum layer thickness (m).
        min_points: Minimum non-zero gates in a layer.
        min_top_der: Minimum backscatter gradient above the peak.
        min_alt: Minimum peak altitude above the lowest gate (m). Not applied to
            surface peaks, which are liquid sitting on the ground by definition.

    Returns:
        Boolean array, True in liquid layers.
    """
    height = np.asarray(height, dtype=float)
    n_height = height.shape[0]
    is_liquid = np.zeros(beta.shape, dtype=bool)
    base_below_peak = _n_elements(height, 200)
    top_above_peak = _n_elements(height, 150)
    beta_diff = ma.array(np.diff(beta, axis=1)).filled(0)
    beta_filled = ma.filled(beta, 0)
    is_peak = local_maxima(beta_filled, order=_PEAK_ORDER, axis=1) & (
        beta_filled > peak_amp
    )
    for n, peak in zip(*np.nonzero(is_peak), strict=True):
        lprof = beta_filled[n]
        dprof = beta_diff[n]
        try:
            base = _ind_base(dprof, peak, base_below_peak, 4)
            top = min(_ind_top(dprof, peak, n_height, top_above_peak, 4), n_height - 1)
        except (IndexError, ValueError):
            continue
        if _is_valid_peak(
            lprof, height, base, peak, top, max_width, min_points, min_top_der, min_alt
        ):
            is_liquid[n, base : top + 1] = True

    # Surface pass: a fog / very-low-stratus layer peaking within the lowest
    # `2 * _PEAK_ORDER` gates is invisible to `local_maxima` (an edge gate is
    # never a strict maximum), so the loop above finds nothing and the signal
    # would fall through to aerosol. Take the strongest gate in that blind zone,
    # anchor the base at the ground and walk the top upward as usual.
    blind_zone = 2 * _PEAK_ORDER
    for n in range(beta.shape[0]):
        if is_liquid[n, :blind_zone].any():
            continue
        lprof = beta_filled[n]
        if lprof[:blind_zone].max() <= peak_amp:
            continue
        peak = int(np.argmax(lprof[:blind_zone]))
        try:
            top = min(
                _ind_top(beta_diff[n], peak, n_height, top_above_peak, 4), n_height - 1
            )
        except (IndexError, ValueError):
            continue
        if _is_valid_peak(
            lprof, height, 0, peak, top, max_width, min_points, min_top_der, min_alt=0
        ):
            is_liquid[n, : top + 1] = True
    return is_liquid


def _is_valid_peak(
    lprof: npt.NDArray[np.floating],
    height: npt.NDArray[np.floating],
    base: int,
    peak: int,
    top: int,
    max_width: float,
    min_points: int,
    min_top_der: float,
    min_alt: float,
) -> bool:
    """Check a bounded backscatter peak against the liquid-layer criteria."""
    if height[top] == height[peak]:
        return False
    npoints = np.count_nonzero(lprof[base : top + 1])
    peak_width = height[top] - height[base]
    peak_alt = height[peak] - height[0]
    top_der = (lprof[peak] - lprof[top]) / (height[top] - height[peak])
    return bool(
        npoints >= min_points
        and peak_width < max_width
        and top_der > min_top_der
        and peak_alt >= min_alt
    )


def grow_liquid(
    droplet: npt.NDArray[np.bool_],
    signal: npt.NDArray[np.bool_],
    blocked: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    grow_up: float = 100.0,
    grow_down: float = 10.0,
) -> npt.NDArray[np.bool_]:
    """Extend liquid layers into the adjacent signal halo (cloud edges).

    `find_liquid` marks only the sharp backscatter core of a liquid layer; the
    weaker gates hugging its base and top are still cloud but fall outside the
    gradient bounds and would otherwise become aerosol. Dilate the droplet mask
    along range into gates that carry signal and are not `blocked` (ice), so the
    thin fringe is absorbed into the cloud. Distances are in metres (converted to
    a gate count for this instrument's resolution), so a layer grows the same
    physical amount whatever the range sampling.

    Growth is asymmetric. The lidar enters the cloud at its base and is attenuated
    going up, so the detected top sits below the true top -- grow generously
    upward to recover it. Below the base is usually sub-cloud aerosol or drizzle,
    so grow downward only a little, to avoid absorbing it as liquid.

    Args:
        droplet: Liquid droplet layers (time x range).
        signal: True where the backscatter is not masked (lidar signal present).
        blocked: Gates the growth must not enter (e.g. the ice region).
        height: Range (m), used to convert distances into gate counts.
        grow_up: Maximum distance to grow toward cloud top (m).
        grow_down: Maximum distance to grow toward cloud base (m).

    Returns:
        The droplet mask grown into its connected signal halo.
    """
    allowed = signal & ~blocked
    out = _grow_range(droplet, allowed, _n_elements(height, grow_up), up=True)
    out = _grow_range(out, allowed, _n_elements(height, grow_down), up=False)
    return out


def _grow_range(
    mask: npt.NDArray[np.bool_],
    allowed: npt.NDArray[np.bool_],
    n_gates: int,
    *,
    up: bool,
) -> npt.NDArray[np.bool_]:
    """Dilate `mask` by `n_gates` along range in one direction, through `allowed`.

    With `up`, a gate joins when the gate just below it is set (growth toward
    higher range); otherwise the gate just above drives growth (toward the
    surface).
    """
    out = mask.copy()
    for _ in range(n_gates):
        neighbour = np.zeros_like(out)
        if up:
            neighbour[:, 1:] |= out[:, :-1]
        else:
            neighbour[:, :-1] |= out[:, 1:]
        grown = out | (neighbour & allowed)
        if grown.sum() == out.sum():
            break
        out = grown
    return out


def fill_thin_clouds(
    droplet: npt.NDArray[np.bool_],
    signal: npt.NDArray[np.bool_],
    blocked: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    max_thickness: float = 100.0,
) -> npt.NDArray[np.bool_]:
    """Relabel whole thin signal layers that contain liquid as liquid.

    A thin liquid cloud is a single contiguous run of signal; `find_liquid`
    lights up only its sharp core, leaving the weaker edges to fall through to
    aerosol. Where a contiguous run of signal (broken by `blocked` ice gates and
    by clear-air gaps) contains a droplet gate and is no thicker than
    `max_thickness`, the whole run is that cloud and is marked droplet. Thicker
    runs — a cloud sitting on a deep aerosol/drizzle column with no clear-air gap
    — are left untouched, so genuine aerosol below the cloud is not absorbed.

    Args:
        droplet: Liquid droplet layers (time x range).
        signal: True where the backscatter is not masked.
        blocked: Gates that break a run and are never filled (e.g. the ice
            region).
        height: Range (m).
        max_thickness: Maximum run thickness to fill (m).

    Returns:
        The droplet mask with thin liquid-bearing signal runs filled.
    """
    return _fill_runs(droplet, signal & ~blocked, height, max_thickness=max_thickness)


def _fill_runs(
    seed: npt.NDArray[np.bool_],
    run: npt.NDArray[np.bool_],
    height: npt.NDArray[np.floating],
    *,
    max_thickness: float | None = None,
) -> npt.NDArray[np.bool_]:
    """Fill each contiguous run of `run` that contains a `seed` gate.

    Runs thicker than `max_thickness` (m) are skipped; `None` fills any
    thickness. Equivalent to flood-filling `seed` along range through `run`, but
    in a single pass.
    """
    height = np.asarray(height, dtype=float)
    out = seed.copy()
    active = seed & run
    for i in np.nonzero(active.any(axis=1))[0]:
        for j, k in _iter_runs(run[i]):
            thin = max_thickness is None or height[k - 1] - height[j] <= max_thickness
            if thin and seed[i, j:k].any():
                out[i, j:k] |= run[i, j:k]
    return out


def _iter_runs(row: npt.NDArray[np.bool_]) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) index spans of the contiguous True runs in a 1-D `row`."""
    n = row.shape[0]
    j = 0
    while j < n:
        if not row[j]:
            j += 1
            continue
        k = j + 1
        while k < n and row[k]:
            k += 1
        yield j, k
        j = k


def find_freezing_region(
    tw: npt.NDArray[np.floating], height: npt.NDArray[np.floating]
) -> npt.NDArray[np.bool_]:
    """Find the region colder than 0 degC from model temperature.

    Per profile, the 0 degC altitude is found by linear interpolation of the
    wet-bulb temperature across `T0`, and everything above it is marked cold.

    Args:
        tw: Wet-bulb temperature (K), time x range.
        height: Range (m).

    Returns:
        Boolean array, True in the sub-freezing region.
    """
    height = np.asarray(height, dtype=float)
    freezing_alt = _find_t0_alt(tw, height)
    return height[np.newaxis, :] > freezing_alt[:, np.newaxis]


def find_falling(
    beta: ma.MaskedArray,
    height: npt.NDArray[np.floating],
    tw: npt.NDArray[np.floating],
    *,
    cold_limit: float = COLD_LIMIT,
    min_altitude: float = 2000,
    despeckle_above: float = DESPECKLE_ALTITUDE,
) -> npt.NDArray[np.bool_]:
    """Detect cold elevated signal (ice / falling hydrometeors) from lidar.

    Lidar-only branch of CloudnetPy's `falling` detection: backscatter present
    in cold air (< -15 degC) above 2000 m is treated as ice. Above 4000 m an
    isolated single gate is dropped to avoid mislabelling aerosol as ice.

    Args:
        beta: Screened backscatter, time x range.
        height: Range (m).
        tw: Wet-bulb temperature (K), time x range.
        cold_limit: Temperature threshold (K).
        min_altitude: Minimum altitude for ice (m).
        despeckle_above: Above this altitude require a non-isolated signal (m).

    Returns:
        Boolean array, True for cold elevated signal.
    """
    height = np.asarray(height, dtype=float)
    is_signal = ~ma.getmaskarray(ma.asarray(beta))
    is_cold = tw < cold_limit
    is_high = (height > min_altitude)[np.newaxis, :]
    falling = is_signal & is_cold & is_high
    high = height > despeckle_above
    if high.any():
        isolated = falling & (_window_count(is_signal, half=3) < 2)
        falling &= ~(isolated & high[np.newaxis, :])
    return falling


def correct_supercooled(
    droplet: npt.NDArray[np.bool_],
    tw: npt.NDArray[np.floating],
    *,
    t_limit: float = SUPERCOOLED_LIMIT,
) -> npt.NDArray[np.bool_]:
    """Remove liquid droplets colder than the supercooled limit (-38 degC)."""
    return droplet & (tw >= t_limit)


def _ind_base(dprof: npt.NDArray, ind_peak: int, dist: int, lim: float) -> int:
    start = max(ind_peak - dist, 0)
    diffs = dprof[start:ind_peak]
    mind = np.argmax(diffs)
    return int(start + np.where(diffs > diffs[mind] / lim)[0][0])


def _ind_top(
    dprof: npt.NDArray, ind_peak: int, nprof: int, dist: int, lim: float
) -> int:
    end = min(ind_peak + dist, nprof)
    diffs = dprof[ind_peak:end]
    mind = np.argmin(diffs)
    return int(ind_peak + np.where(diffs < diffs[mind] / lim)[0][-1] + 1)


def _n_elements(height: npt.NDArray[np.floating], distance: float) -> int:
    return int(round(distance / np.median(np.diff(height))))


def _find_t0_alt(
    temperature: npt.NDArray[np.floating], height: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Altitude of the 0 degC level per profile, keeping the warm air below it.

    With a warm surface (the usual case) we take the *lowest* 0 degC crossing, so
    the boundary tracks the melting level smoothly even through warm-cold-warm
    layering aloft. Only when the surface itself is sub-freezing -- a winter
    surface inversion, where the lowest crossing would collapse to the ground and
    mislabel the warm column above as ice -- do we anchor on the *topmost*
    crossing instead. CloudnetPy always uses the lowest crossing.
    """
    alt = np.empty(temperature.shape[0])
    for i, prof in enumerate(temperature):
        if prof[0] >= T0:
            cold = np.where(prof < T0)[0]
            if len(cold) == 0:
                alt[i] = height[-1]  # warm throughout: no crossing within range
            else:
                j = cold[0]  # lowest cold gate; the crossing is just below it
                alt[i] = _interp_t0_alt(height, prof, j - 1, j)
        else:
            warm = np.where(prof >= T0)[0]
            if len(warm) == 0:
                alt[i] = height[0]  # whole column sub-freezing
            elif warm[-1] == len(height) - 1:
                alt[i] = height[-1]  # warm to the top: no crossing within range
            else:
                j = warm[-1]  # topmost warm gate; the crossing is just above it
                alt[i] = _interp_t0_alt(height, prof, j, j + 1)
    return alt


def _interp_t0_alt(
    height: npt.NDArray[np.floating], prof: npt.NDArray[np.floating], a: int, b: int
) -> float:
    """Linear-interpolate the altitude of the `T0` crossing between gates a and b."""
    slope = (height[b] - height[a]) / (prof[b] - prof[a])
    return float(height[a] + (T0 - prof[a]) * slope)


def _window_count(mask: npt.NDArray[np.bool_], half: int) -> npt.NDArray[np.intp]:
    """Count True values within +/- `half` gates along the range axis."""
    cumulative = np.cumsum(mask.astype(int), axis=1)
    padded = np.pad(cumulative, ((0, 0), (1, 0)))
    n = mask.shape[1]
    idx = np.arange(n)
    hi = np.minimum(idx + half + 1, n)
    lo = np.maximum(idx - half, 0)
    return padded[:, hi] - padded[:, lo]
