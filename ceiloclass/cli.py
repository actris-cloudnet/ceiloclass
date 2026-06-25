"""Command-line interface for ceiloclass."""

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

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
from .download import fetch_lidar, fetch_model, fetch_raw

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
    parser = argparse.ArgumentParser(prog="ceiloclass")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_classify_parser(subparsers)
    args = parser.parse_args(argv)
    try:
        args.func(args, parser)
    except (ValueError, OSError) as e:
        # Expected failures (missing files, no data found, unreadable netCDF):
        # show a clean one-line message instead of a full traceback.
        parser.exit(1, f"{parser.prog}: error: {e}\n")


def _add_classify_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "classify", help="Classify ceilometer targets using model temperature"
    )
    p.add_argument(
        "files",
        nargs="*",
        help="Ceilometer data file(s); if omitted, fetched using --site/--date",
    )
    p.add_argument(
        "-i", "--instrument", choices=sorted(READERS), help="Instrument (raw input)"
    )
    p.add_argument(
        "--lidar",
        action="store_true",
        help="Input is a Cloudnet harmonized lidar product (calibrated, screened)",
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
    p.set_defaults(func=_run_classify)


def _run_classify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    reader: Callable[..., Ceilo]
    if args.lidar:
        reader = read_lidar
    elif args.instrument:
        reader = READERS[args.instrument]
    else:
        parser.error("provide -i/--instrument, or --lidar for a harmonized product")

    if args.files:
        files: list = list(args.files)
    elif args.site and args.date:
        if args.lidar:
            files = fetch_lidar(
                args.site, args.date, args.download_dir, args.instrument
            )
        else:
            files = fetch_raw(args.instrument, args.site, args.date, args.download_dir)
    else:
        parser.error("provide data files, or both --site and --date to fetch them")
    ceilo: Ceilo = reader(files, args.calibration_factor)

    if args.average:
        ceilo = average_time(ceilo, args.average)

    if args.model and Path(args.model).exists():
        model = args.model  # a local netCDF file
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
        from .plot import plot_classification  # noqa: PLC0415

        plot_kwargs = {}
        if args.max_y is not None:
            plot_kwargs["max_height"] = args.max_y * 1000
        plot_classification(
            result,
            args.plot,
            beta=ceilo.beta,
            depol=ceilo.depol,
            show=args.show,
            **plot_kwargs,
        )
        if args.plot:
            print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
