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

_G = 9.80665
"""Standard gravity (m s-2), to convert surface geopotential to height."""


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
    altitude: float | None = None,
    use_wet_bulb: bool = True,
) -> Model:
    """Read model temperature and interpolate it onto the ceilometer grid.

    The model height is height above the model grid cell's surface, the
    ceilometer range is height above the instrument; the two surfaces can differ
    by hundreds of metres in complex terrain. Given the site `altitude` and the
    model's own surface height, both grids are aligned in absolute (a.s.l.)
    coordinates before interpolating, matching CloudnetPy. Without either, both
    are treated as height-above-ground (no correction).

    Args:
        path: Cloudnet model netCDF file.
        time: Ceilometer time (datetime objects).
        range: Ceilometer range (m).
        altitude: Site altitude above mean sea level (m); see `read_altitude`.
        use_wet_bulb: Compute wet-bulb temperature instead of dry-bulb.

    Returns:
        A `Model` with temperature on the (time, range) grid.
    """
    with netCDF4.Dataset(path) as nc:
        model_time = num2pydate(nc["time"][:], nc["time"].units)
        height = nc["height"][:]
        temperature = nc["temperature"][:]
        surface_altitude = _model_surface_altitude(nc)
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
    # Per-step shift to align model height-above-ground onto the observation grid
    # in a.s.l. coordinates; zero unless both surface heights are known.
    if altitude is not None and surface_altitude is not None:
        correction = surface_altitude - altitude
    else:
        correction = np.zeros(len(model_time))
    rows, tops, valid = [], [], []
    for h_i, t_i, corr in zip(height, temp, correction, strict=True):
        h = ma.filled(ma.masked_invalid(h_i), np.nan)
        v = ma.filled(ma.masked_invalid(t_i), np.nan)
        finite = np.isfinite(h) & np.isfinite(v)
        valid.append(bool(finite.any()))
        if finite.any():
            # A model time step with no usable values (occasionally seen in
            # HARMONIE files) is dropped here and bridged by the time
            # interpolation below, rather than failing the whole classification.
            rows.append(np.interp(obs_range - corr, h[finite], v[finite]))
            tops.append(h[finite].max() + corr)
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


def read_altitude(path: str | PathLike) -> float | None:
    """Read the site altitude (m above mean sea level) from a netCDF, if present."""
    try:
        with netCDF4.Dataset(path) as nc:
            if "altitude" in nc.variables:
                return float(np.asarray(nc["altitude"][:]).ravel()[0])
    except OSError:
        return None
    return None


def _model_surface_altitude(
    nc: netCDF4.Dataset,
) -> npt.NDArray[np.floating] | None:
    """Model grid-cell surface altitude per time step (m a.s.l.), or None."""
    if "sfc_height" in nc.variables:
        return np.asarray(nc["sfc_height"][:], dtype=float)
    if "sfc_geopotential" in nc.variables:
        return np.asarray(nc["sfc_geopotential"][:], dtype=float) / _G
    return None


def _to_seconds(
    times: npt.NDArray[np.object_], reference: datetime.datetime
) -> npt.NDArray[np.floating]:
    return np.array([(t - reference).total_seconds() for t in times], dtype=float)
