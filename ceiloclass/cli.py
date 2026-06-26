"""Command-line interface for ceiloclass."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import cast

from ceilopyter import (
    Ceilo,
    average_time,
    read_chm15k,
    read_cl31,
    read_cl51,
    read_cl61,
    read_cs135,
    read_ct25k,
    read_ld40,
    read_lidar,
)

from .classification import Target, classify
from .download import (
    LidarSource,
    download_source,
    fetch_model,
    list_harmonized_sources,
    list_raw_sources,
)
from .plot import plot_classification

READERS = {
    "cl31": read_cl31,
    "cl51": read_cl51,
    "cl61": read_cl61,
    "chm15k": read_chm15k,
    "cs135": read_cs135,
    "ct25k": read_ct25k,
    "ld40": read_ld40,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``ceiloclass`` command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="ceiloclass",
        description="Classify ceilometer targets using model temperature.",
    )
    _add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        _run_classify(args, parser)
    except (ValueError, OSError) as e:
        # Expected failures (missing files, no data found, unreadable netCDF):
        # show a clean one-line message instead of a full traceback.
        parser.exit(1, f"{parser.prog}: error: {e}\n")


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "files",
        nargs="*",
        help="Ceilometer data file(s); if omitted, fetched using --site/--date",
    )
    p.add_argument(
        "-i",
        "--instrument",
        metavar="ID",
        help="Instrument. For raw input, the reader to use, one of: "
        f"{', '.join(sorted(READERS))}. When fetching a harmonized product it is "
        "optional and filters by instrument-id substring (e.g. 'halo', 'pollyxt', "
        "'cl61'); if several instruments remain you are prompted to pick one",
    )
    p.add_argument(
        "--harmonized",
        action="store_true",
        help="Use a Cloudnet harmonized backscatter product (ceilometers, PollyXT, "
        "doppler-lidars) instead of raw instrument data",
    )
    p.add_argument(
        "-m",
        "--model",
        help="Cloudnet model netCDF file, or a model id to fetch "
        "(e.g. ecmwf, harmonie-fmi-6-11) when using --site/--date",
    )
    p.add_argument(
        "-s", "--site", help="Cloudnet site id (to fetch raw files and/or model)"
    )
    p.add_argument(
        "-d", "--date", help="Date YYYY-MM-DD (to fetch raw files and/or model)"
    )
    p.add_argument(
        "--download-dir",
        default=".",
        help="Directory for fetched files (default: current directory)",
    )
    p.add_argument("--calibration-factor", type=float, help="Override calibration")
    p.add_argument(
        "-a",
        "--average",
        type=float,
        metavar="SECONDS",
        help="Average into time bins of this width before classifying (faster)",
    )
    p.add_argument("--plot", help="Write a classification plot to this PNG file")
    p.add_argument("--show", action="store_true", help="Show the plot in a window")
    p.add_argument(
        "--max-y",
        type=float,
        metavar="KM",
        help="Upper limit of the range axis in plots (km)",
    )
    p.add_argument(
        "--no-histogram",
        action="store_true",
        help="Omit the diagnostic backscatter histogram panel from plots",
    )


def _select_source(
    sources: list[LidarSource], parser: argparse.ArgumentParser
) -> LidarSource:
    """Pick one instrument: the only one, or prompt the user when there are many.

    With several candidates and no interactive terminal, error out listing them
    (the user can narrow with `-i`) rather than guessing.
    """
    if len(sources) == 1:
        return sources[0]
    listing = "\n".join(f"  {s.label}" for s in sources)
    if not sys.stdin.isatty():
        parser.error(
            "several instruments available; run interactively to choose, or pass "
            f"-i to narrow:\n{listing}"
        )
    print("Several instruments available:")
    for i, source in enumerate(sources, 1):
        print(f"  [{i}] {source.label}")
    while True:
        try:
            choice = input(f"Select [1-{len(sources)}]: ").strip()
        except EOFError:
            parser.exit(1, "\nno instrument selected\n")
        if choice.isdigit() and 1 <= int(choice) <= len(sources):
            return sources[int(choice) - 1]
        print("Please enter a number from the list.")


def _run_classify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    reader: Callable[..., Ceilo]
    files: list[str | PathLike]
    if args.files:
        # Local files: we can't introspect them, so the reader must be stated.
        if args.harmonized:
            reader = read_lidar
        elif args.instrument in READERS:
            reader = READERS[args.instrument]
        elif args.instrument:
            parser.error(
                f"unknown raw instrument {args.instrument!r}; "
                f"choose from: {', '.join(sorted(READERS))}"
            )
        else:
            parser.error("provide -i/--instrument, or --harmonized for a product")
        files = list(args.files)
    elif args.site and args.date:
        if args.harmonized:
            sources = list_harmonized_sources(args.site, args.date, args.instrument)
        else:
            sources = list_raw_sources(args.site, args.date, args.instrument)
        source = _select_source(sources, parser)
        files = cast("list[str | PathLike]", download_source(source, args.download_dir))
        if args.harmonized:
            reader = read_lidar
        elif source.reader is not None:
            reader = READERS[source.reader]
        else:
            parser.error(f"no raw reader for instrument: {source.label}")
    else:
        parser.error("provide data files, or both --site and --date to fetch them")
    ceilo: Ceilo = reader(files, args.calibration_factor)

    if args.average:
        ceilo = average_time(ceilo, args.average)

    if args.model and Path(args.model).exists():
        model = args.model
    elif args.site and args.date:
        # args.model (if given) is a model id to select which model to fetch.
        model = str(fetch_model(args.site, args.date, args.download_dir, args.model))
    else:
        parser.error("provide --model PATH, or both --site and --date to fetch one")

    result = classify(ceilo, model)

    total = result.target.size
    print(f"{result.target.shape[0]} profiles x {result.target.shape[1]} gates")
    print("\nclass fractions:")
    for target in Target:
        count = int((result.target == target.value).sum())
        if count:
            print(f"  {target.name:30s} {count / total * 100:6.2f}%")

    if args.plot or args.show:
        plot_kwargs = {}
        if args.max_y is not None:
            plot_kwargs["max_height"] = args.max_y * 1000
        plot_classification(
            result,
            args.plot,
            beta=ceilo.beta,
            depol=ceilo.depol,
            show=args.show,
            histogram=not args.no_histogram,
            **plot_kwargs,
        )
        if args.plot:
            print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
