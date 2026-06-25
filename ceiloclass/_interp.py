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
    weight = np.where(span > 0, (new_time - time[lower]) / span, 0.0)
    weight = np.clip(weight, 0.0, 1.0)[:, np.newaxis]
    return values[lower] * (1 - weight) + values[upper] * weight
