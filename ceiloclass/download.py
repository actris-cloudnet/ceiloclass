"""Download raw ceilometer, lidar product and model files from Cloudnet.

Requires the optional `cloudnet-api-client` dependency
(``pip install ceiloclass[download]``). Files already present in the output
directory are not downloaded again.
"""

import datetime
import logging
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

INSTRUMENT_IDS: dict[str, tuple[str, ...]] = {
    "cl31": ("cl31",),
    "cl51": ("cl51",),
    "cl61": ("cl61d",),
    "chm15k": ("chm15k", "chm15kx"),
    "cs135": ("cs135",),
    "ct25k": ("ct25k",),
    "ld40": ("ld40",),
}
"""Maps instrument names to Cloudnet portal instrument ids."""


def fetch_raw(
    instrument: str,
    site_id: str,
    date: str | datetime.date,
    output_directory: str | PathLike = ".",
) -> list[Path]:
    """Download raw ceilometer files for an instrument, site and date.

    Args:
        instrument: Instrument name (a key of `INSTRUMENT_IDS`).
        site_id: Cloudnet site identifier (e.g. "hyytiala").
        date: Measurement date.
        output_directory: Where to save the files.

    Returns:
        Local paths to the raw files (downloaded or already present).
    """
    from cloudnet_api_client import APIClient  # noqa: PLC0415

    if instrument not in INSTRUMENT_IDS:
        msg = f"Unknown instrument: {instrument}"
        raise ValueError(msg)
    client = APIClient()
    metadata = client.raw_files(
        site_id=site_id,
        date=date,
        instrument_id=list(INSTRUMENT_IDS[instrument]),
    )
    if not metadata:
        msg = f"No {instrument} files found for {site_id} on {date}"
        raise ValueError(msg)
    return _download_missing(client, metadata, output_directory)


def fetch_lidar(
    site_id: str,
    date: str | datetime.date,
    output_directory: str | PathLike = ".",
    instrument: str | None = None,
) -> list[Path]:
    """Download the Cloudnet harmonized lidar product for a site and date.

    Args:
        site_id: Cloudnet site identifier.
        date: Measurement date.
        output_directory: Where to save the file.
        instrument: Optional instrument name to disambiguate the site's lidars.

    Returns:
        Local path(s) to the lidar product file(s).
    """
    from cloudnet_api_client import APIClient  # noqa: PLC0415

    client = APIClient()
    instrument_id = list(INSTRUMENT_IDS[instrument]) if instrument else None
    metadata = client.files(
        site_id=site_id, date=date, product_id="lidar", instrument_id=instrument_id
    )
    if not metadata:
        msg = f"No lidar product found for {site_id} on {date}"
        raise ValueError(msg)
    if len({m.instrument.instrument_id for m in metadata if m.instrument}) > 1:
        logging.warning(
            "Several lidars at %s; using %s (pass -i to choose)",
            site_id,
            metadata[0].filename,
        )
    return _download_missing(client, metadata[:1], output_directory)


def fetch_model(
    site_id: str,
    date: str | datetime.date,
    output_directory: str | PathLike = ".",
    model_id: str | None = None,
) -> Path:
    """Download a Cloudnet model file for a site and date.

    Args:
        site_id: Cloudnet site identifier (e.g. "lindenberg").
        date: Measurement date.
        output_directory: Where to save the file.
        model_id: Optional model identifier (e.g. "ecmwf", "harmonie-fmi-6-11");
            if omitted, the portal's default model for the site/date is used.

    Returns:
        Path to the model file (downloaded or already present).
    """
    from cloudnet_api_client import APIClient  # noqa: PLC0415

    client = APIClient()
    if model_id is not None:
        valid = [str(getattr(m, "id", m)) for m in client.models()]
        if model_id not in valid:
            msg = f"Unknown model '{model_id}'. Available models: {', '.join(valid)}"
            raise ValueError(msg)
    metadata = client.files(
        site_id=site_id, date=date, product_id="model", model_id=model_id
    )
    if not metadata:
        which = f" '{model_id}'" if model_id else ""
        msg = f"No{which} model file found for {site_id} on {date}"
        raise ValueError(msg)
    return _download_missing(client, metadata[:1], output_directory)[0]


def _download_missing(
    client: object, metadata: Iterable, output_directory: str | PathLike
) -> list[Path]:
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    metadata = list(metadata)
    missing = [m for m in metadata if not (out / m.filename).exists()]
    where = out.resolve()
    if missing:
        logging.info("Downloading %s to %s", _describe(missing), where)
        client.download(missing, output_directory=out)  # type: ignore[attr-defined]
    else:
        logging.info("Found %s already in %s", _describe(metadata), where)
    return [out / m.filename for m in metadata]


def _describe(metadata: list) -> str:
    """Human-readable summary of files: the name if one, else a count."""
    if len(metadata) == 1:
        return metadata[0].filename
    return f"{len(metadata)} files"
