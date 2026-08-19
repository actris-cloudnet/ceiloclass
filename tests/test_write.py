import datetime

import netCDF4
import numpy as np

from ceiloclass.classification import Classification, Target
from ceiloclass.write import write_classification


def _classification() -> Classification:
    """A tiny 3-profile x 4-gate classification with a NaN freezing level."""
    day = datetime.datetime(2025, 6, 24, tzinfo=datetime.timezone.utc)
    time = np.array([day + datetime.timedelta(hours=h) for h in (0.0, 1.5, 3.0)])
    rng = np.array([100.0, 200.0, 300.0, 400.0])
    target = np.array(
        [
            [Target.CLEAR, Target.AEROSOL, Target.ICE, Target.CLEAR],
            [Target.DROPLET, Target.DRIZZLE_OR_RAIN, Target.ICE, Target.SUPERCOOLED],
            [Target.CLEAR, Target.DROPLET, Target.ATTENUATED, Target.ATTENUATED],
        ],
        dtype=np.int64,
    )
    false = np.zeros(target.shape, dtype=bool)
    quality = np.zeros(target.shape, dtype=bool)
    quality[:, -1] = True  # top gate extrapolated
    cold = np.zeros(target.shape, dtype=bool)
    cold[0, 2:] = True  # profile 0 sub-freezing from gate 2 (300 m) up
    return Classification(
        time=time,
        range=rng,
        target=target,
        droplet=false,
        cold=cold,
        ice=false,
        rain=false,
        aerosol=false,
        attenuated=target == Target.ATTENUATED,
        quality=quality,
        tw=np.full(target.shape, 275.0),
        t0_alt=np.array([250.0, np.nan, 300.0]),
        strong_beta=1e-5,
    )


def test_write_roundtrip_and_cf(tmp_path):
    cls = _classification()
    path = tmp_path / "out.nc"
    file_uuid = write_classification(
        cls,
        path,
        wavelength=910.0,
        altitude=104.0,
        latitude=52.208,
        longitude=14.118,
        location="Lindenberg",
        source_files=["/data/raw_20250624.nc"],
        instrument="cl61d",
        used_depolarization=True,
    )

    with netCDF4.Dataset(path) as nc:
        assert nc.data_model == "NETCDF4"
        assert nc.Conventions == "CF-1.8"
        assert nc.file_uuid == file_uuid
        assert nc.location == "Lindenberg"
        assert (nc.year, nc.month, nc.day) == ("2025", "06", "24")
        # only the file name is recorded, not the full path
        assert nc.source_files == "raw_20250624.nc"

        assert {k: len(v) for k, v in nc.dimensions.items()} == {"time": 3, "range": 4}

        # target round-trips exactly, as int8
        tc = nc.variables["target_classification"]
        assert tc.dtype == np.int8
        np.testing.assert_array_equal(tc[:], cls.target)
        # CF discrete-flag attributes
        np.testing.assert_array_equal(tc.flag_values, [0, 1, 2, 3, 4, 5, 6])
        assert tc.flag_meanings.split()[0] == "clear_sky"
        assert len(tc.flag_meanings.split()) == len(Target)

        # every data variable is zlib-compressed
        for name in ("range", "target_classification", "temperature_quality", "Tw"):
            assert nc.variables[name].filters()["zlib"] is True

        # wet-bulb temperature grid round-trips (within its 0.01 K truncation)
        tw = nc.variables["Tw"]
        assert tw.units == "K"
        assert tw.least_significant_digit == 2
        np.testing.assert_allclose(tw[:], 275.0, atol=0.005)

        # effective boundary: cold base at 300 m in profile 0, fill elsewhere
        irb = nc.variables["ice_rain_boundary"][:]
        assert irb[0] == np.float32(300.0)
        assert irb.mask[1] and irb.mask[2]

        # provenance attributes
        assert nc.instrument == "cl61d"
        assert nc.classification_path == "depolarization"

        # time is CF-encoded hours since midnight of the day
        t = nc.variables["time"]
        assert t.units == "hours since 2025-06-24 00:00:00 +00:00"
        assert t.standard_name == "time"
        np.testing.assert_allclose(t[:], [0.0, 1.5, 3.0])

        # NaN freezing level becomes the fill value (masked on read)
        fl = nc.variables["freezing_level"][:]
        assert np.ma.is_masked(fl)
        assert fl.mask[1] and not fl.mask[0]

        # geolocation comes through as scalars
        assert nc.variables["latitude"][...] == np.float32(52.208)
        assert nc.variables["altitude"][...] == np.float32(104.0)
        assert nc.variables["wavelength"][...] == np.float32(910.0)


def test_write_without_optional_metadata(tmp_path):
    """Geolocation/wavelength variables are simply omitted when unknown."""
    path = tmp_path / "bare.nc"
    write_classification(_classification(), path)
    with netCDF4.Dataset(path) as nc:
        for absent in ("latitude", "longitude", "altitude", "wavelength"):
            assert absent not in nc.variables
        for attr in ("location", "instrument", "classification_path"):
            assert attr not in nc.ncattrs()
        assert "target_classification" in nc.variables
