# ceiloclass

Cloud, aerosol and precipitation classification from ceilometer and lidar
backscatter and model temperature. Built on
[ceilopyter](https://github.com/actris-cloudnet/ceilopyter) for reading and
harmonizing the instrument data.

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

## License

MIT
