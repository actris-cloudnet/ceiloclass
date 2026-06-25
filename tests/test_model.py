import datetime

import netCDF4
import numpy as np

from ceiloclass.model import read_model


def _write_model(path):
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", 2)
        nc.createDimension("level", 3)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2025-06-14 00:00:00 +00:00"
        t[:] = [0.0, 24.0]
        h = nc.createVariable("height", "f4", ("time", "level"))
        h[:] = np.array([[100.0, 1000.0, 5000.0], [100.0, 1000.0, 5000.0]])
        temp = nc.createVariable("temperature", "f4", ("time", "level"))
        temp[:] = np.array([[280.0, 270.0, 250.0], [282.0, 272.0, 252.0]])
        pressure = nc.createVariable("pressure", "f4", ("time", "level"))
        pressure[:] = np.array([[1.0e5, 9.0e4, 5.0e4], [1.0e5, 9.0e4, 5.0e4]])
        q = nc.createVariable("q", "f4", ("time", "level"))
        # Specific humidity (kg/kg), subsaturated and decreasing with height.
        q[:] = np.array([[0.004, 0.0015, 0.0002], [0.004, 0.0015, 0.0002]])


def test_read_model_wet_bulb_below_dry_bulb(tmp_path):
    path = tmp_path / "model.nc"
    _write_model(path)
    time = np.array([datetime.datetime(2025, 6, 14, 0, 0)])
    obs_range = np.array([100.0, 1000.0, 5000.0])
    dry = read_model(path, time, obs_range, use_wet_bulb=False)
    wet = read_model(path, time, obs_range, use_wet_bulb=True)
    assert np.all(wet.tw <= dry.tw + 1e-6)  # wet-bulb never exceeds dry-bulb
    assert np.all(np.isfinite(wet.tw))


def test_read_model_interpolates_to_grid(tmp_path):
    path = tmp_path / "model.nc"
    _write_model(path)
    time = np.array(
        [
            datetime.datetime(2025, 6, 14, 0, 0),
            datetime.datetime(2025, 6, 14, 12, 0),
        ]
    )
    obs_range = np.array([100.0, 1000.0, 5000.0])
    model = read_model(path, time, obs_range, use_wet_bulb=False)
    assert model.tw.shape == (2, 3)
    # First obs time == first model time -> equals the model temperature there.
    assert np.allclose(model.tw[0], [280.0, 270.0, 250.0], atol=0.5)
    # Midday is halfway between the two model steps.
    assert np.allclose(model.tw[1], [281.0, 271.0, 251.0], atol=0.5)
    assert not model.extrapolated.any()  # range within model coverage


def test_read_model_flags_extrapolation(tmp_path):
    path = tmp_path / "model.nc"
    _write_model(path)
    time = np.array([datetime.datetime(2025, 6, 14, 0, 0)])
    obs_range = np.array([100.0, 9000.0])  # 9000 m is above the model top
    model = read_model(path, time, obs_range, use_wet_bulb=False)
    assert not model.extrapolated[0, 0]
    assert model.extrapolated[0, 1]
