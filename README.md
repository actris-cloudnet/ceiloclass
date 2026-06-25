# ceiloclass

Cloud, aerosol and precipitation classification from ceilometer and lidar
backscatter and model temperature. Built on
[ceilopyter](https://github.com/actris-cloudnet/ceilopyter) for reading and
harmonizing the instrument data.

## Installation

Not yet on PyPI, so install from source:

```sh
git clone https://github.com/actris-cloudnet/ceiloclass.git
cd ceiloclass
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Usage

Fetch and classify a day of raw ceilometer data for a site (the instrument is
discovered automatically; if a site has several, you are prompted to pick one):

```sh
ceiloclass -s munich -d 2025-05-25 --show
```

Add `--lidar` to use the Cloudnet harmonized lidar product instead of raw data,
`-i` to narrow to a particular instrument, or pass local files directly:

```sh
ceiloclass --lidar -s munich -d 2025-05-25 --show
ceiloclass -i cl61 ceilo.nc -m model.nc --plot out.png
```

A fuller example — classify the CL61 lidar product at Kenttärova, averaging into
30 s bins, using the HARMONIE model and showing the plot:

```sh
ceiloclass -s kenttarova -d 2023-09-04 -a 30 --lidar -i cl61 -m harmonie-fmi-6-11 --show
```

## Arguments

| Argument               | Description                                                                                                                                                                                                      |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `files`                | Ceilometer data file(s). If omitted, files are fetched using `--site`/`--date`.                                                                                                                                  |
| `-i`, `--instrument`   | Instrument: `cl31`, `cl51`, `cl61`, `chm15k`, `cs135`, `ct25k`, `ld40`. Required for local raw files; when fetching it is optional and just narrows the search (you are prompted if several instruments remain). |
| `--lidar`              | Treat the input as a Cloudnet harmonized lidar product (calibrated, screened) rather than raw data.                                                                                                              |
| `-m`, `--model`        | Cloudnet model netCDF file, or a model id to fetch (e.g. `ecmwf`, `harmonie-fmi-6-11`) when using `--site`/`--date`.                                                                                             |
| `-s`, `--site`         | Cloudnet site id (to fetch raw files and/or model), e.g. `munich`.                                                                                                                                               |
| `-d`, `--date`         | Date `YYYY-MM-DD` (to fetch raw files and/or model).                                                                                                                                                             |
| `--download-dir`       | Directory for fetched files (default: current directory).                                                                                                                                                        |
| `--calibration-factor` | Override the backscatter calibration factor.                                                                                                                                                                     |
| `-a`, `--average`      | Average into time bins of this width (seconds) before classifying (faster).                                                                                                                                      |
| `--plot`               | Write a classification plot to this PNG file.                                                                                                                                                                    |
| `--show`               | Show the plot in a window.                                                                                                                                                                                       |
| `--max-y`              | Upper limit of the range axis in plots (km).                                                                                                                                                                     |

## License

MIT
