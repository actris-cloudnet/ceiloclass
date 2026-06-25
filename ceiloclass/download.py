"""Download raw ceilometer, lidar product and model files from Cloudnet.

Files already present in the output directory are not downloaded again.
"""

import datetime
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from cloudnet_api_client import APIClient

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

_READER_BY_ID: dict[str, str] = {
    pid: name for name, ids in INSTRUMENT_IDS.items() for pid in ids
}
"""Reverse of `INSTRUMENT_IDS`: a portal instrument id maps to one reader name."""


@dataclass
class LidarSource:
    """One ceilometer/lidar available at a site/date, with its portal files.

    Attributes:
        reader: Reader name (a key of `INSTRUMENT_IDS`) for raw files, or `None`
            for a harmonized lidar product (always read with `read_lidar`).
        label: Human-readable description for prompting/logging.
        metadata: Portal file metadata to download for this instrument.
    """

    reader: str | None
    label: str
    metadata: list


def list_raw_sources(
    site_id: str,
    date: str | datetime.date,
    instrument: str | None = None,
) -> list[LidarSource]:
    """List the raw ceilometer/lidar instruments available at a site and date.

    Args:
        site_id: Cloudnet site identifier (e.g. "hyytiala").
        date: Measurement date.
        instrument: Optional reader name (a key of `INSTRUMENT_IDS`) to restrict
            the search to one instrument type.

    Returns:
        One `LidarSource` per distinct instrument, in portal order.

    Raises:
        ValueError: If `instrument` is unknown, or nothing is found.
    """
    ids = _portal_ids(instrument) if instrument else _ALL_PORTAL_IDS
    metadata = APIClient().raw_files(site_id=site_id, date=date, instrument_id=ids)
    sources = _group_sources(metadata, raw=True)
    if not sources:
        what = instrument or "ceilometer/lidar"
        msg = f"No {what} files found for {site_id} on {date}"
        raise ValueError(msg)
    return sources


def list_lidar_product_sources(
    site_id: str,
    date: str | datetime.date,
    instrument: str | None = None,
) -> list[LidarSource]:
    """List the harmonized lidar products available at a site and date.

    Args:
        site_id: Cloudnet site identifier.
        date: Measurement date.
        instrument: Optional reader name to restrict the search to one
            instrument type.

    Returns:
        One `LidarSource` per distinct instrument, in portal order.

    Raises:
        ValueError: If `instrument` is unknown, or nothing is found.
    """
    ids = _portal_ids(instrument) if instrument else None
    metadata = APIClient().files(
        site_id=site_id, date=date, product_id="lidar", instrument_id=ids
    )
    sources = _group_sources(metadata, raw=False)
    if not sources:
        msg = f"No lidar product found for {site_id} on {date}"
        raise ValueError(msg)
    return sources


def download_source(
    source: LidarSource, output_directory: str | PathLike = "."
) -> list[Path]:
    """Download a selected `LidarSource`'s files (skipping any already present)."""
    return _download_missing(APIClient(), source.metadata, output_directory)


_ALL_PORTAL_IDS = [pid for ids in INSTRUMENT_IDS.values() for pid in ids]


def _portal_ids(instrument: str) -> list[str]:
    if instrument not in INSTRUMENT_IDS:
        msg = f"Unknown instrument: {instrument}"
        raise ValueError(msg)
    return list(INSTRUMENT_IDS[instrument])


def _group_sources(metadata: Iterable, *, raw: bool) -> list[LidarSource]:
    """Group portal metadata into one `LidarSource` per physical instrument.

    Instruments are keyed by their persistent identifier (`pid`), so two units of
    the same model at one site become separate sources. Order follows first
    appearance in the metadata.
    """
    groups: dict[str, list] = {}
    for m in metadata:
        inst = m.instrument
        key = (inst.pid or str(inst.uuid)) if inst else m.filename
        groups.setdefault(key, []).append(m)
    sources = []
    for items in groups.values():
        inst = items[0].instrument
        reader = _READER_BY_ID.get(inst.instrument_id) if (raw and inst) else None
        sources.append(LidarSource(reader, _label(inst, len(items)), items))
    return sources


def _label(inst: object, n_files: int) -> str:
    files = f"{n_files} file" + ("s" if n_files != 1 else "")
    if inst is None:
        return f"unknown instrument ({files})"
    serial = f", SN {inst.serial_number}" if inst.serial_number else ""  # type: ignore[attr-defined]
    return f"{inst.instrument_id} — {inst.name}{serial} ({files})"  # type: ignore[attr-defined]


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
