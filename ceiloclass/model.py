"""Cloudnet model file handling for classification.

Reads temperature, pressure and specific humidity from a Cloudnet model netCDF
file, interpolates them onto the ceilometer time/range grid, and computes
wet-bulb temperature there.
"""

import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike

import netCDF4
import numpy as np
import numpy.typing as npt
from cftime import num2pydate
from numpy import ma

from ._interp import interp_extrap, interpolate_along_time

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
        extrapolated: True where tw was extrapolated above the highest model
            level or outside the model's coverage in time. Extrapolation below
            the lowest model level is not flagged: it is deliberate (it gives a
            site below the model surface a sensible near-ground value).
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

    Temperature (and, for wet-bulb, pressure and specific humidity) are
    interpolated onto the ceilometer grid first; wet-bulb is then computed from
    the interpolated fields, as CloudnetPy does -- not by interpolating a
    wet-bulb field built on the model grid (the two differ, wet-bulb being
    nonlinear in its inputs).

    Args:
        path: Cloudnet model netCDF file.
        time: Ceilometer time (datetime objects).
        range: Ceilometer range (m).
        altitude: Site altitude above mean sea level (m); see `read_altitude`.
        use_wet_bulb: Compute wet-bulb temperature instead of dry-bulb.

    Returns:
        A `Model` with temperature on the (time, range) grid.
    """
    wet_bulb = _import_wet_bulb() if use_wet_bulb else None
    with netCDF4.Dataset(path) as nc:
        model_time = num2pydate(nc["time"][:], nc["time"].units)
        height = nc["height"][:]
        surface_altitude = _model_surface_altitude(nc)
        fields = {"temperature": nc["temperature"][:]}
        if wet_bulb is not None:
            fields["pressure"] = nc["pressure"][:]
            fields["q"] = nc["q"][:]

    obs_range = np.asarray(range, dtype=float)
    # Per-step shift to align model height-above-ground onto the observation grid
    # in a.s.l. coordinates; zero unless both surface heights are known.
    if altitude is not None and surface_altitude is not None:
        correction = surface_altitude - altitude
    else:
        correction = np.zeros(len(model_time))

    heights = [ma.filled(ma.masked_invalid(h), np.nan) for h in height]
    temps = [ma.filled(ma.masked_invalid(t), np.nan) for t in fields["temperature"]]
    finite = [
        np.isfinite(h) & np.isfinite(t) for h, t in zip(heights, temps, strict=True)
    ]
    # A model time step with no usable values (occasionally seen in HARMONIE
    # files) is dropped here and bridged by the time interpolation below, rather
    # than failing the whole classification.
    valid = np.array([m.any() for m in finite])
    if not valid.any():
        msg = "Model has no finite temperature/height values"
        raise ValueError(msg)
    n_dropped = int((~valid).sum())
    if n_dropped:
        logging.warning(
            "Dropping %d/%d model time step(s) with no finite temperature/height",
            n_dropped,
            len(valid),
        )
    kept = np.nonzero(valid)[0]
    model_time = model_time[valid]
    ref = model_time[0]
    model_seconds = _to_seconds(model_time, ref)
    obs_seconds = _to_seconds(time, ref)

    def _to_obs(field: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # Stage 1: each model profile onto the obs range. Extrapolate (not clamp)
        # outside the model levels, as CloudnetPy does, so a site below the model
        # surface gets a sensible near-ground value.
        on_range = np.array(
            [
                interp_extrap(
                    obs_range - correction[i],
                    heights[i][finite[i]],
                    ma.filled(ma.masked_invalid(field[i]), np.nan)[finite[i]],
                )
                for i in kept
            ]
        )
        # Stage 2: along time.
        return interpolate_along_time(obs_seconds, model_seconds, on_range)

    on_grid = {k: _to_obs(v) for k, v in fields.items()}

    if wet_bulb is not None:
        tw = on_grid["temperature"].copy()
        # Solve wet-bulb only in dense enough air; the obs grid is well within
        # this, but it guards any extrapolated near-vacuum gates above the model.
        dense = on_grid["pressure"] > _MIN_WET_BULB_PRESSURE
        tw[dense] = wet_bulb(
            on_grid["temperature"][dense],
            on_grid["pressure"][dense],
            on_grid["q"][dense],
        )
    else:
        tw = on_grid["temperature"]

    model_top = np.array([heights[i][finite[i]].max() + correction[i] for i in kept])
    top = np.interp(obs_seconds, model_seconds, model_top)
    above_top = obs_range[np.newaxis, :] > top[:, np.newaxis]
    outside_time = (obs_seconds < model_seconds[0]) | (obs_seconds > model_seconds[-1])
    extrapolated = above_top | outside_time[:, np.newaxis]

    return Model(tw, extrapolated)


def _import_wet_bulb() -> Callable[..., npt.NDArray[np.floating]]:
    try:
        from atmoslib import wet_bulb_temperature  # noqa: PLC0415
    except ImportError as e:
        msg = "Wet-bulb temperature requires atmoslib (pip install atmoslib)"
        raise ImportError(msg) from e
    return wet_bulb_temperature


def read_altitude(path: str | PathLike) -> float | None:
    """Read the site altitude (m above mean sea level) from a netCDF, if present."""
    try:
        with netCDF4.Dataset(path) as nc:
            if "altitude" in nc.variables:
                return float(np.asarray(nc["altitude"][:]).ravel()[0])
    except OSError:
        return None
    return None


def read_geolocation(
    path: str | PathLike,
) -> tuple[float | None, float | None, str | None]:
    """Read (latitude, longitude, location name) from a netCDF, when present.

    Any of the three is `None` if absent. Latitude/longitude are taken from the
    first grid point (Cloudnet single-site model files carry one).
    """
    lat = lon = None
    location = None
    try:
        with netCDF4.Dataset(path) as nc:
            if "latitude" in nc.variables:
                lat = float(np.asarray(nc["latitude"][:]).ravel()[0])
            if "longitude" in nc.variables:
                lon = float(np.asarray(nc["longitude"][:]).ravel()[0])
            if "location" in nc.ncattrs():
                location = str(nc.location)
    except OSError:
        return None, None, None
    return lat, lon, location


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
