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
    least_significant_digit: int | None = None,
) -> netCDF4.Variable:
    """Create a zlib-compressed (shuffled) data variable."""
    return nc.createVariable(
        name,
        dtype,
        dims,
        zlib=True,
        complevel=4,
        shuffle=True,
        fill_value=fill_value,
        least_significant_digit=least_significant_digit,
    )


_TARGET_DEFINITION = "\n".join(
    (
        "Value 0: Clear sky.",
        "Value 1: Cloud liquid droplets.",
        "Value 2: Drizzle or rain.",
        "Value 3: Ice particles.",
        "Value 4: Supercooled liquid droplets.",
        "Value 5: Aerosol particles.",
        "Value 6: Beam attenuated below (undetected).",
    )
)

# Single-word CF flag_meanings, in Target value order.
_FLAG_MEANINGS = (
    "clear_sky cloud_droplets drizzle_or_rain ice supercooled_droplets aerosol "
    "attenuated"
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
    instrument: str | None = None,
    used_depolarization: bool | None = None,
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
        instrument: Instrument identifier (e.g. "cl61d"), if known; recorded
            as the `instrument` global attribute.
        used_depolarization: Whether the depolarization path was used. The
            classification semantics differ between the two paths, so this is
            recorded as the `classification_path` global attribute.

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

        # Truncated to 0.01 K: far beyond model accuracy, and a smooth float
        # field stored at full precision would dominate the file size (~5x).
        tw = _cvar(nc, "Tw", "f4", ("time", "range"), least_significant_digit=2)
        tw.units = "K"
        tw.long_name = "Wet-bulb temperature"
        tw.standard_name = "wet_bulb_temperature"
        tw.comment = (
            "Model wet-bulb temperature interpolated onto the observation grid; "
            "the temperature field the classification was made against (at full "
            "precision; the stored copy is truncated to 0.01 K). See "
            "temperature_quality for extrapolated pixels."
        )
        tw[:] = np.asarray(classification.tw, dtype="f4")

        fill = float(netCDF4.default_fillvals["f4"])
        fl = _cvar(nc, "freezing_level", "f4", ("time",), fill_value=fill)
        fl.units = "m"
        fl.long_name = "Range of the 0 degrees Celsius isotherm above instrument"
        fl.comment = (
            "Altitude of the model wet-bulb 0 degC isotherm. On instruments with "
            "depolarization the classification anchors the effective ice/rain "
            "boundary to observed ice, so the boundary actually used can depart "
            "from this model level; see ice_rain_boundary."
        )
        t0 = np.asarray(classification.t0_alt, dtype="f4")
        fl[:] = np.where(np.isnan(t0), fill, t0)

        irb = _cvar(nc, "ice_rain_boundary", "f4", ("time",), fill_value=fill)
        irb.units = "m"
        irb.long_name = "Range of the effective ice/rain boundary above instrument"
        irb.comment = (
            "Base of the sub-freezing region the classification actually used: "
            "equals the freezing level except where depolarization anchored it "
            "to observed ice or the melt band. Gate-quantized; fill value where "
            "the whole column is warm."
        )
        cold = np.asarray(classification.cold, dtype=bool)
        rng_m = np.asarray(classification.range, dtype="f4")
        boundary = np.full(len(time), fill, dtype="f4")
        has_cold = cold.any(axis=1)
        boundary[has_cold] = rng_m[np.argmax(cold, axis=1)[has_cold]]
        irb[:] = boundary

        threshold = nc.createVariable("backscatter_threshold", "f4")
        threshold.units = "sr-1 m-1"
        threshold.long_name = (
            "Backscatter threshold separating cloud/precipitation from aerosol"
        )
        threshold[...] = np.float32(classification.strong_beta)

        if classification.beam_saturation is not None:
            sat = nc.createVariable("beam_saturation", "f4")
            sat.units = "sr-1"
            sat.long_name = "Integrated attenuated backscatter of an extinguished beam"
            sat.comment = (
                "Saturation plateau of the column integral of attenuated "
                "backscatter, estimated from this file's liquid-topped profiles; "
                "the attenuation rule marks the void above a profile whose "
                "integral reaches a fraction of it."
            )
            sat[...] = np.float32(classification.beam_saturation)

        if wavelength is not None:
            wl = nc.createVariable("wavelength", "f4")
            wl.units = "nm"
            wl.long_name = "Laser wavelength"
            wl[...] = np.float32(wavelength)

        _write_geolocation(nc, altitude, latitude, longitude)
        _write_global_attributes(nc, midnight, file_uuid, location, source_files)
        if instrument:
            nc.instrument = instrument
        if used_depolarization is not None:
            nc.classification_path = (
                "depolarization" if used_depolarization else "no-depolarization"
            )

    return file_uuid


def _write_geolocation(
    nc: netCDF4.Dataset,
    altitude: float | None,
    latitude: float | None,
    longitude: float | None,
) -> None:
    """Add scalar altitude/latitude/longitude variables where known."""
    fields = (
        ("altitude", altitude, "m", "Altitude of site"),
        ("latitude", latitude, "degree_north", "Latitude of site"),
        ("longitude", longitude, "degree_east", "Longitude of site"),
    )
    for name, value, units, long_name in fields:
        if value is None:
            continue
        v = nc.createVariable(name, "f4")
        v.units = units
        v.long_name = long_name
        v.standard_name = name
        v[...] = np.float32(value)


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
