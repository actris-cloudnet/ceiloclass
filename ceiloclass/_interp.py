"""Small numpy-only helpers for the classification code."""

import numpy as np
import numpy.typing as npt


def local_maxima(
    data: npt.NDArray[np.floating], order: int = 4, axis: int = 1
) -> npt.NDArray[np.bool_]:
    """Find strict local maxima along an axis.

    Equivalent to ``scipy.signal.argrelextrema(data, np.greater, order=order,
    axis=axis)`` with the default ``mode="clip"``: a point is a maximum when it
    is strictly greater than each of its ``order`` neighbours on both sides.
    Indices beyond the edges are clipped, so border points are never maxima.

    Args:
        data: Input array.
        order: Number of neighbours to compare on each side.
        axis: Axis along which to search.

    Returns:
        Boolean array, True at local maxima.
    """
    n = data.shape[axis]
    idx = np.arange(n)
    result = np.ones(data.shape, dtype=bool)
    for shift in range(1, order + 1):
        right = np.take(data, np.clip(idx + shift, 0, n - 1), axis=axis)
        left = np.take(data, np.clip(idx - shift, 0, n - 1), axis=axis)
        result &= (data > right) & (data > left)
    return result


def interp_extrap(
    x: npt.NDArray[np.floating],
    xp: npt.NDArray[np.floating],
    fp: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Linear interpolation that extrapolates beyond the data range.

    Like ``numpy.interp`` within ``[xp[0], xp[-1]]`` but, instead of clamping,
    continues the edge slope outside it -- matching
    ``scipy.interpolate.interp1d(xp, fp, fill_value="extrapolate")``. ``xp`` must
    be increasing and have at least two points.

    Args:
        x: Query points.
        xp: Sample coordinates (1-D, increasing).
        fp: Sample values (1-D, same length as ``xp``).

    Returns:
        Interpolated/extrapolated values, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    out = np.interp(x, xp, fp)
    if len(xp) < 2:
        return out  # nothing to extrapolate from; np.interp already clamped
    below = x < xp[0]
    out[below] = fp[0] + (x[below] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    above = x > xp[-1]
    out[above] = fp[-1] + (x[above] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return out


def interpolate_along_time(
    new_time: npt.NDArray[np.floating],
    time: npt.NDArray[np.floating],
    values: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Linearly interpolate a 2-D field along its first (time) axis.

    Values outside the source time range are clamped to the nearest edge (no
    extrapolation).

    Args:
        new_time: Target time coordinates (1-D, length M).
        time: Source time coordinates (1-D, increasing, length N).
        values: Source values of shape (N, K).

    Returns:
        Interpolated values of shape (M, K).
    """
    time = np.asarray(time, dtype=float)
    new_time = np.asarray(new_time, dtype=float)
    upper = np.clip(np.searchsorted(time, new_time), 1, len(time) - 1)
    lower = upper - 1
    span = time[upper] - time[lower]
    # Guard the division so a zero span (e.g. a single source time) doesn't warn;
    # weight stays 0 there, clamping to the lower (only) sample.
    weight = np.zeros_like(new_time, dtype=float)
    np.divide(new_time - time[lower], span, out=weight, where=span > 0)
    weight = np.clip(weight, 0.0, 1.0)[:, np.newaxis]
    return values[lower] * (1 - weight) + values[upper] * weight
