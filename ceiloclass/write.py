"""Write a `Classification` to a CF-compliant, compressed netCDF4 file."""

import datetime
import uuid
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import netCDF4
import numpy as np

from .classification import Classification, Target
from .version import __version__


def _cvar(
    nc: netCDF4.Dataset,
    name: str,
    dtype: str,
    dims: tuple[str, ...],
    *,
    fill_value: float | None = None,
) -> netCDF4.Variable:
    """Create a zlib-compressed (shuffled) data variable."""
    return nc.createVariable(
        name, dtype, dims, zlib=True, complevel=4, shuffle=True, fill_value=fill_value
    )


_TARGET_DEFINITION = "\n".join(
    (
        "Value 0: Clear sky.",
        "Value 1: Cloud liquid droplets.",
        "Value 2: Drizzle or rain.",
        "Value 3: Ice particles.",
        "Value 4: Supercooled liquid droplets.",
        "Value 5: Aerosol particles.",
    )
)

# Single-word CF flag_meanings, in Target value order.
_FLAG_MEANINGS = (
    "clear_sky cloud_droplets drizzle_or_rain ice supercooled_droplets aerosol"
)


def write_classification(
    classification: Classification,
    filename: str | PathLike,
    *,
    wavelength: float | None = None,
    altitude: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    location: str | None = None,
    source_files: Sequence[str | PathLike] | None = None,
) -> str:
    """Write a classification to a compressed, CF-1.8 netCDF4 file.

    Args:
        classification: The result to write.
        filename: Output path (overwritten if it exists).
        wavelength: Lidar wavelength (nm), if known.
        altitude: Site altitude (m above mean sea level), if known.
        latitude: Site latitude (degrees north), if known. Use the true
            instrument coordinate (e.g. from the Cloudnet portal), not a model
            file's offset NWP grid point.
        longitude: Site longitude (degrees east), if known.
        location: Human-readable site name, if known.
        source_files: Input files; their base names are recorded in the
            `source_files` global attribute.

    Returns:
        The file's UUID (also stored as the `file_uuid` global attribute).
    """
    time = classification.time
    day = time[0]
    midnight = day.replace(hour=0, minute=0, second=0, microsecond=0)
    hours = np.array([(t - midnight).total_seconds() / 3600 for t in time], dtype="f8")
    file_uuid = str(uuid.uuid4())

    with netCDF4.Dataset(filename, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(time))
        nc.createDimension("range", len(classification.range))

        t = nc.createVariable("time", "f8", ("time",))
        t.units = f"hours since {midnight:%Y-%m-%d} 00:00:00 +00:00"
        t.long_name = "Time UTC"
        t.standard_name = "time"
        t.calendar = "standard"
        t.axis = "T"
        t[:] = hours

        rng = _cvar(nc, "range", "f4", ("range",))
        rng.units = "m"
        rng.long_name = "Range above instrument"
        rng.axis = "Z"
        rng.positive = "up"
        rng[:] = np.asarray(classification.range, dtype="f4")

        tc = _cvar(nc, "target_classification", "i1", ("time", "range"))
        tc.units = "1"
        tc.long_name = "Target classification"
        tc.flag_values = np.array([t.value for t in Target], dtype="i1")
        tc.flag_meanings = _FLAG_MEANINGS
        tc.definition = _TARGET_DEFINITION
        tc.comment = (
            "Radar-free target classification from a single lidar/ceilometer and a "
            "model temperature field."
        )
        tc[:] = np.asarray(classification.target, dtype="i1")

        tq = _cvar(nc, "temperature_quality", "i1", ("time", "range"))
        tq.units = "1"
        tq.long_name = "Model temperature quality flag"
        tq.flag_values = np.array([0, 1], dtype="i1")
        tq.flag_meanings = "reliable extrapolated"
        tq.comment = (
            "Flags pixels where the model wet-bulb temperature was extrapolated "
            "above the model top or outside its temporal coverage."
        )
        tq[:] = np.asarray(classification.quality, dtype="i1")

        fill = float(netCDF4.default_fillvals["f4"])
        fl = _cvar(nc, "freezing_level", "f4", ("time",), fill_value=fill)
        fl.units = "m"
        fl.long_name = "Range of the 0 degrees Celsius isotherm above instrument"
        t0 = np.asarray(classification.t0_alt, dtype="f4")
        fl[:] = np.where(np.isnan(t0), fill, t0)

        threshold = nc.createVariable("backscatter_threshold", "f4")
        threshold.units = "sr-1 m-1"
        threshold.long_name = (
            "Backscatter threshold separating cloud/precipitation from aerosol"
        )
        threshold[...] = np.float32(classification.strong_beta)

        if wavelength is not None:
            wl = nc.createVariable("wavelength", "f4")
            wl.units = "nm"
            wl.long_name = "Laser wavelength"
            wl[...] = np.float32(wavelength)

        _write_geolocation(nc, altitude, latitude, longitude)
        _write_global_attributes(nc, midnight, file_uuid, location, source_files)

    return file_uuid


def _write_geolocation(
    nc: netCDF4.Dataset,
    altitude: float | None,
    latitude: float | None,
    longitude: float | None,
) -> None:
    """Add scalar altitude/latitude/longitude variables where known."""
    if altitude is not None:
        v = nc.createVariable("altitude", "f4")
        v.units = "m"
        v.long_name = "Altitude of site"
        v.standard_name = "altitude"
        v[...] = np.float32(altitude)
    if latitude is not None:
        v = nc.createVariable("latitude", "f4")
        v.units = "degree_north"
        v.long_name = "Latitude of site"
        v.standard_name = "latitude"
        v[...] = np.float32(latitude)
    if longitude is not None:
        v = nc.createVariable("longitude", "f4")
        v.units = "degree_east"
        v.long_name = "Longitude of site"
        v.standard_name = "longitude"
        v[...] = np.float32(longitude)


def _write_global_attributes(
    nc: netCDF4.Dataset,
    day: datetime.datetime,
    file_uuid: str,
    location: str | None,
    source_files: Sequence[str | PathLike] | None,
) -> None:
    """Write CF and provenance global attributes."""
    now = datetime.datetime.now(datetime.timezone.utc)
    nc.Conventions = "CF-1.8"
    nc.title = "Ceilometer target classification" + (
        f" from {location}" if location else ""
    )
    nc.institution = "Cloudnet"
    nc.source = f"ceiloclass {__version__}"
    if source_files:
        nc.source_files = "\n".join(Path(f).name for f in source_files)
    nc.references = "https://github.com/actris-cloudnet/ceiloclass"
    nc.history = f"{now:%Y-%m-%d %H:%M:%S} +00:00 - file created by ceiloclass"
    nc.file_uuid = file_uuid
    nc.year = f"{day:%Y}"
    nc.month = f"{day:%m}"
    nc.day = f"{day:%d}"
    if location:
        nc.location = location
