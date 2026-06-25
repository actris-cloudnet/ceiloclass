"""Cloudnet model file handling for classification.

Reads temperature, pressure and specific humidity from a Cloudnet model netCDF
file, computes wet-bulb temperature, and interpolates it onto the ceilometer
time/range grid.
"""

import datetime
import logging
from dataclasses import dataclass
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt
from cftime import num2pydate
from numpy import ma

from ._interp import interpolate_along_time

T0 = 273.16
"""Triple point of water (K), used as the freezing threshold."""

_MIN_WET_BULB_PRESSURE = 1000.0
"""Pressure floor for wet-bulb computation (Pa, ~30 km — above any ceilometer)."""


@dataclass
class Model:
    """Model temperature interpolated onto the ceilometer grid.

    Attributes:
        tw: Wet-bulb (or dry-bulb) temperature (K), shape time x range.
        extrapolated: True where tw falls outside the model's coverage in time
            or above the highest model level (i.e. the value is clamped).
    """

    tw: npt.NDArray[np.floating]
    extrapolated: npt.NDArray[np.bool_]


def read_model(
    path: str | PathLike,
    time: npt.NDArray[np.object_],
    range: npt.NDArray[np.floating],
    *,
    use_wet_bulb: bool = True,
) -> Model:
    """Read model temperature and interpolate it onto the ceilometer grid.

    Args:
        path: Cloudnet model netCDF file.
        time: Ceilometer time (datetime objects).
        range: Ceilometer range (m).
        use_wet_bulb: Compute wet-bulb temperature instead of dry-bulb.

    Returns:
        A `Model` with temperature on the (time, range) grid.
    """
    with netCDF4.Dataset(path) as nc:
        model_time = num2pydate(nc["time"][:], nc["time"].units)
        height = nc["height"][:]
        temperature = nc["temperature"][:]
        if use_wet_bulb:
            try:
                from atmoslib import wet_bulb_temperature  # noqa: PLC0415
            except ImportError as e:
                msg = "Wet-bulb temperature requires atmoslib (pip install atmoslib)"
                raise ImportError(msg) from e
            pressure = nc["pressure"][:]
            specific_humidity = nc["q"][:]
            temp = temperature.astype(float)
            # Solve wet-bulb only in dense enough air. The model's near-vacuum
            # top levels (p ~ 1 Pa, tens of km up, far above any ceilometer)
            # make atmoslib's iterative solver diverge; they are discarded by
            # the interpolation below anyway, so leave them as dry-bulb.
            dense = pressure > _MIN_WET_BULB_PRESSURE
            temp[dense] = wet_bulb_temperature(
                temperature[dense], pressure[dense], specific_humidity[dense]
            )
        else:
            temp = temperature

    obs_range = np.asarray(range, dtype=float)
    rows, tops, valid = [], [], []
    for h_i, t_i in zip(height, temp, strict=True):
        h = ma.filled(ma.masked_invalid(h_i), np.nan)
        v = ma.filled(ma.masked_invalid(t_i), np.nan)
        finite = np.isfinite(h) & np.isfinite(v)
        valid.append(bool(finite.any()))
        if finite.any():
            # A model time step with no usable values (occasionally seen in
            # HARMONIE files) is dropped here and bridged by the time
            # interpolation below, rather than failing the whole classification.
            rows.append(np.interp(obs_range, h[finite], v[finite]))
            tops.append(h[finite].max())
    if not rows:
        msg = "Model has no finite temperature/height values"
        raise ValueError(msg)
    n_dropped = valid.count(False)
    if n_dropped:
        logging.warning(
            "Dropping %d/%d model time step(s) with no finite temperature/height",
            n_dropped,
            len(valid),
        )
    model_time = model_time[np.array(valid)]
    temp_on_range = np.array(rows)
    model_top = np.array(tops)

    ref = model_time[0]
    model_seconds = _to_seconds(model_time, ref)
    obs_seconds = _to_seconds(time, ref)
    tw = interpolate_along_time(obs_seconds, model_seconds, temp_on_range)

    top = np.interp(obs_seconds, model_seconds, model_top)
    above_top = obs_range[np.newaxis, :] > top[:, np.newaxis]
    outside_time = (obs_seconds < model_seconds[0]) | (obs_seconds > model_seconds[-1])
    extrapolated = above_top | outside_time[:, np.newaxis]

    return Model(tw, extrapolated)


def _to_seconds(
    times: npt.NDArray[np.object_], reference: datetime.datetime
) -> npt.NDArray[np.floating]:
    return np.array([(t - reference).total_seconds() for t in times], dtype=float)
