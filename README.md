# ceiloclass

Cloud, aerosol and precipitation classification from ceilometer and lidar
backscatter and model temperature. Built on
[ceilopyter](https://github.com/actris-cloudnet/ceilopyter) for reading and
harmonizing the instrument data.

## Install

```sh
pip install ceiloclass            # core classification
pip install ceiloclass[plot]      # + plotting (matplotlib)
pip install ceiloclass[download]  # + fetching raw/model files from Cloudnet
pip install ceiloclass[all]       # everything
```

## Usage

```sh
ceiloclass classify --lidar -s leipzig -d 2025-05-25 --show
```

## License

MIT
