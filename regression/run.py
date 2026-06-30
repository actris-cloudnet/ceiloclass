#!/usr/bin/env python3
"""Regression / characterization suite for ceiloclass.

There is no pixel-level ground truth for these classifications, so this is not
assertion-based unit testing. Instead it runs a curated set of *well-known,
difficult* days spanning sites and lidars, classifies each with the current
code, and flags regressions two ways:

* **semantic checks** -- hand-written physical expectations per case (e.g.
  "Mindelo's bright marine haze must be aerosol, not drizzle"). These encode the
  bugs we have fixed and fail loudly if one comes back.
* **baseline drift** -- a committed numeric snapshot (``regression/baseline.json``)
  of each case's class fractions, ``strong_beta``, melting level and check
  values. Any drift beyond tolerance is reported, so an unrelated change that
  shifts the output is visible even where no check covers it.

Data is fetched from the Cloudnet portal and cached in ``--data-dir`` (default:
``regression/data/``, gitignored). The network is touched only for files not yet
present. A classification PNG per case is written to ``--plot-dir`` (default:
``regression/plots/``, gitignored) unless ``--no-plot`` is given.

Usage::

    python regression/run.py                  # all cases: checks + baseline diff
    python regression/run.py --only mindelo    # cases whose id contains "mindelo"
    python regression/run.py --list            # list cases and exit
    python regression/run.py --no-plot         # skip the per-case PNG files
    python regression/run.py --offline         # use only cached files, no network
    python regression/run.py --update-baseline # accept current output as baseline
    python regression/run.py --strict          # treat baseline drift as failure

Once a case has been resolved its filenames are recorded in
``<data-dir>/manifest.json``, so subsequent runs reuse the cached files without
querying the portal at all; ``--offline`` enforces this (and errors on a case
that is not yet cached).

Exit status is non-zero if any semantic check fails or a case errors (and, with
``--strict``, if anything drifts), so it doubles as a CI / pre-push gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ceilopyter import average_time, read_lidar

from ceiloclass import Target, classify, read_altitude
from ceiloclass.download import download_source, fetch_model, list_harmonized_sources

BASELINE_PATH = Path(__file__).with_name("baseline.json")
DEFAULT_DATA_DIR = Path(__file__).with_name("data")
DEFAULT_PLOT_DIR = Path(__file__).with_name("plots")


@dataclass(frozen=True)
class Check:
    """A semantic expectation: the fraction of one class in a time/height window.

    The window is the whole curtain unless ``hours`` (UTC, inclusive) and/or
    ``height_m`` (metres, inclusive) restrict it. The fraction is taken over all
    pixels in the window, or over the non-clear pixels when ``of="classified"``.
    The check passes when the measured fraction is within ``[min_frac, max_frac]``
    (either bound may be ``None``).
    """

    desc: str
    target: Target
    min_frac: float | None = None
    max_frac: float | None = None
    hours: tuple[float, float] | None = None
    height_m: tuple[float, float] | None = None
    of: str = "window"


@dataclass(frozen=True)
class Case:
    """One well-known day: where to get it, why it is hard, what to expect."""

    id: str
    site: str
    date: str
    instrument: str  # instrument-id substring, e.g. "cl61", "chm15k", "halo"
    note: str
    checks: list[Check] = field(default_factory=list)
    model: str | None = None
    average: float = 30.0


# --- The curated cases ------------------------------------------------------
# Add new ones here. Keep each case's `note` describing what makes it hard, and
# its `checks` encoding the physics we care about. After adding/adjusting, run
# with --update-baseline to record the numeric snapshot.

CASES: list[Case] = [
    Case(
        id="lindenberg-ice-flood",
        site="lindenberg",
        date="2025-05-22",
        instrument="cl61",
        note="Weak but strongly-depolarizing daytime boundary-layer aerosol must "
        "NOT flood the freezing region to the ground as a fake ice column. Also "
        "guards the melt-band logic: low-depol patches high inside an ice cloud "
        "(far above t0) must stay ice, not be relabelled drizzle.",
        checks=[
            Check(
                "no fake ice column in the warm boundary layer",
                Target.ICE,
                max_frac=0.05,
                hours=(16.5, 16.7),
                height_m=(0, 1000),
            ),
            Check(
                "no drizzle high inside the ice cloud (well above t0)",
                Target.DRIZZLE_OR_RAIN,
                max_frac=0.02,
                height_m=(2500, 5000),
            ),
        ],
    ),
    Case(
        id="lindenberg-melting-above-t0",
        site="lindenberg",
        date="2025-06-24",
        instrument="cl61",
        note="A depol-confirmed ice base sits ABOVE the model 0 degC line (t0 "
        "biased low): the cold low-depol band between t0 and that base is ice "
        "melting into rain, not ice. The ice/rain boundary follows the observed "
        "ice base (symmetric counterpart of _extend_cold_to_ice). CloudnetPy's "
        "radar classification for this day confirms a melting layer here.",
        checks=[
            Check(
                "the melt band below the ice base is drizzle/rain",
                Target.DRIZZLE_OR_RAIN,
                min_frac=0.10,
                hours=(11.0, 13.5),
                height_m=(2100, 2700),
            ),
            Check(
                "ice no longer floods the band down to the model t0",
                Target.ICE,
                max_frac=0.55,
                hours=(11.0, 13.5),
                height_m=(2100, 2700),
            ),
        ],
    ),
    Case(
        id="mindelo-marine-haze",
        site="mindelo",
        date="2026-06-24",
        instrument="cl61",
        note="Bright humidified marine boundary-layer aerosol (sea-salt haze) is "
        "above the cloud threshold but has no parent cloud: it must be aerosol, "
        "not drizzle. Also guards against isolated near-ground drizzle columns.",
        checks=[
            Check(
                # The bug made most of the surface band drizzle; the genuine
                # retained sub-cloud drizzle is a small minority of it.
                "marine haze is not drizzle",
                Target.DRIZZLE_OR_RAIN,
                max_frac=0.10,
                height_m=(0, 1000),
            ),
            Check(
                "the surface layer is classified (as aerosol)",
                Target.AEROSOL,
                min_frac=0.30,
                height_m=(0, 1000),
            ),
        ],
    ),
    Case(
        id="kenttarova-drizzle",
        site="kenttarova",
        date="2023-09-04",
        instrument="cl61",
        note="Real sub-cloud drizzle below liquid layers: the drizzle-source gate "
        "must keep it (regression guard against over-aggressive demotion).",
        checks=[
            Check(
                "real sub-cloud drizzle is retained",
                Target.DRIZZLE_OR_RAIN,
                min_frac=0.005,
            ),
        ],
    ),
    Case(
        id="troll-polar-ice",
        site="troll",
        date="2026-05-09",
        instrument="cl61",
        note="High-altitude polar site, low freezing level: ice-dominated with "
        "supercooled liquid aloft that find_falling must not cut off.",
        checks=[
            Check("ice-dominated column", Target.ICE, min_frac=0.03),
            Check("supercooled liquid present", Target.SUPERCOOLED, min_frac=0.003),
        ],
    ),
    Case(
        id="nyalesund-cloudy-arctic",
        site="ny-alesund",
        date="2025-08-02",
        instrument="cl51",
        note="Cloudy Arctic site (cl51, no depol): the cloud mode can out-count "
        "the aerosol mode in the backscatter histogram, so the adaptive threshold "
        "must anchor on the lower aerosol mode (here it caps at 1e-5) rather than "
        "land on the cloud mode and miss the clouds. strong_beta is tracked by "
        "the baseline.",
        checks=[
            Check("liquid clouds are detected", Target.DROPLET, min_frac=0.005),
            Check(
                "sub-cloud drizzle is detected",
                Target.DRIZZLE_OR_RAIN,
                min_frac=0.003,
            ),
        ],
    ),
    Case(
        id="granada-dust",
        site="granada",
        date="2022-08-09",
        instrument="chm15k",
        note="Dusty continental site, no depolarization (chm15k): lofted dust is a "
        "second aerosol mode that must stay aerosol, not be split off as "
        "drizzle/cloud by the adaptive threshold.",
        checks=[
            Check("lofted dust is not drizzle", Target.DRIZZLE_OR_RAIN, max_frac=0.03),
            Check("aerosol is present", Target.AEROSOL, min_frac=0.02),
        ],
    ),
    Case(
        id="granada-bimodal-aerosol",
        site="granada",
        date="2025-08-27",
        instrument="chm15k",
        note="Dry day with a bimodal aerosol population (chm15k, no depol): the "
        "second aerosol mode must not be mistaken for cloud and split off as "
        "drizzle by the adaptive backscatter threshold.",
        checks=[
            Check(
                "second aerosol mode is not drizzle",
                Target.DRIZZLE_OR_RAIN,
                max_frac=0.03,
            ),
            Check("aerosol is present", Target.AEROSOL, min_frac=0.02),
        ],
    ),
    Case(
        id="macehead-melting-level",
        site="mace-head",
        date="2025-06-24",
        instrument="chm15k",
        note="Warm marine day with a warm-cold-warm temperature structure aloft: "
        "the 0 degC level must track the lowest crossing smoothly (no kinks from "
        "latching onto an upper warm layer) and no spurious ice may appear below "
        "it. The melting level is tracked by the baseline (t0_median).",
        checks=[
            Check(
                "no spurious ice below the melting level", Target.ICE, max_frac=0.005
            ),
            Check("liquid clouds are detected", Target.DROPLET, min_frac=0.005),
        ],
    ),
    Case(
        id="leipzig-pollyxt",
        site="leipzig",
        date="2025-05-22",
        instrument="pollyxt",
        note="Instrument diversity: PollyXT harmonized product. Smoke test that it "
        "reads and classifies; specifics tracked via the baseline.",
        checks=[
            Check("something is classified", Target.CLEAR, max_frac=0.999),
        ],
    ),
    Case(
        id="bucharest-doppler",
        site="bucharest",
        date="2026-05-24",
        instrument="halo",
        note="Instrument diversity: HALO doppler-lidar (experimental backscatter "
        "product). Smoke test; specifics tracked via the baseline.",
        checks=[
            Check("something is classified", Target.CLEAR, max_frac=0.999),
        ],
    ),
    Case(
        id="lindenberg-da10-dial",
        site="lindenberg",
        date="2026-06-17",
        instrument="da10",
        note="Instrument diversity: Vaisala DA10 DIAL via the harmonized lidar "
        "product. Confirms a DIAL reads into a Ceilo and classifies on the same "
        "thresholds (ceilometer-scale backscatter, no depolarization -> no-depol "
        "path). High cirrus aloft is ice; an evening precip period gives liquid "
        "with sub-cloud drizzle below it.",
        checks=[
            Check(
                "high cirrus aloft is ice",
                Target.ICE,
                min_frac=0.02,
                hours=(9.0, 18.0),
                height_m=(4500, 8000),
            ),
            Check(
                "the boundary layer is classified (as aerosol)",
                Target.AEROSOL,
                min_frac=0.30,
                height_m=(0, 2000),
            ),
            Check(
                "evening sub-cloud precipitation is drizzle/rain",
                Target.DRIZZLE_OR_RAIN,
                min_frac=0.05,
                hours=(19.0, 24.0),
                height_m=(0, 2500),
            ),
        ],
    ),
]


# --- Running a case ---------------------------------------------------------


@dataclass
class Result:
    """Computed metrics for one case (also what the baseline stores)."""

    fractions: dict[str, float]
    strong_beta: float
    t0_median: float
    checks: dict[str, float]


def _hours(time: npt.NDArray[np.object_]) -> npt.NDArray[np.float64]:
    return np.array(
        [t.hour + t.minute / 60 + t.second / 3600 for t in time], dtype=float
    )


def _fractions(target: npt.NDArray[np.integer]) -> dict[str, float]:
    total = target.size
    return {t.name: float((target == t.value).sum()) / total for t in Target}


def _measure(
    target: npt.NDArray[np.integer],
    hours: npt.NDArray[np.float64],
    height: npt.NDArray[np.floating],
    check: Check,
) -> float:
    """Fraction of `check.target` within the check's time/height window."""
    tmask = np.ones(target.shape[0], dtype=bool)
    if check.hours is not None:
        tmask = (hours >= check.hours[0]) & (hours <= check.hours[1])
    hmask = np.ones(target.shape[1], dtype=bool)
    if check.height_m is not None:
        hmask = (height >= check.height_m[0]) & (height <= check.height_m[1])
    sub = target[np.ix_(tmask, hmask)]
    if sub.size == 0:
        return 0.0
    denom = int((sub != Target.CLEAR).sum()) if check.of == "classified" else sub.size
    if denom == 0:
        return 0.0
    return float((sub == check.target.value).sum()) / denom


class CacheMiss(RuntimeError):
    """A case's files are not cached while running with --offline."""


def _manifest_path(data_dir: Path) -> Path:
    return data_dir / "manifest.json"


def _load_manifest(data_dir: Path) -> dict:
    """Read the data-dir cache index (case id -> resolved filenames), if any."""
    path = _manifest_path(data_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(data_dir: Path, manifest: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    _manifest_path(data_dir).write_text(text + "\n")


def _snapshot(res: Result) -> dict:
    """Build a result's baseline entry, rounded to the precision the diff uses.

    The drift tolerances (~1e-3 on fractions, 5% on strong_beta, 50 m on t0) are
    far coarser than full float precision, so there is no point storing the noise.
    """
    return {
        "fractions": {k: round(v, 5) for k, v in res.fractions.items()},
        "strong_beta": float(f"{res.strong_beta:.3g}"),
        "t0_median": round(res.t0_median, 1),
        "checks": {k: round(v, 5) for k, v in res.checks.items()},
    }


def _resolve(
    case: Case, data_dir: Path, manifest: dict, *, offline: bool
) -> tuple[list[Path], Path]:
    """Return the case's (lidar files, model file), avoiding the portal if cached.

    On a cache hit -- the manifest lists this case and every listed file is
    present in `data_dir` -- no network is touched. On a miss it queries the
    portal, downloads what is missing, and records the resolved filenames in
    `manifest` for next time. With `offline` a miss raises instead of going out.
    """
    entry = manifest.get(case.id)
    if entry is not None:
        names = [*entry["files"], entry["model"]]
        if all((data_dir / n).exists() for n in names):
            files = [data_dir / n for n in entry["files"]]
            return files, data_dir / entry["model"]
    if offline:
        msg = f"{case.id}: not in the local cache ({data_dir}) and --offline is set"
        raise CacheMiss(msg)
    sources = list_harmonized_sources(case.site, case.date, case.instrument)
    if len(sources) > 1:
        labels = ", ".join(s.label for s in sources)
        print(f"    note: {len(sources)} instruments matched, using first ({labels})")
    files = download_source(sources[0], data_dir)
    model = fetch_model(case.site, case.date, data_dir, case.model)
    manifest[case.id] = {"files": [f.name for f in files], "model": model.name}
    return files, model


def run_case(
    case: Case,
    data_dir: Path,
    plot_dir: Path | None,
    manifest: dict,
    *,
    offline: bool,
) -> Result:
    """Resolve data (cache/download), classify, and compute the case metrics."""
    files, model = _resolve(case, data_dir, manifest, offline=offline)
    ceilo = read_lidar([str(f) for f in files], None)
    if case.average:
        ceilo = average_time(ceilo, case.average)
    cls = classify(ceilo, str(model), altitude=read_altitude(files[0]))

    hours = _hours(cls.time)
    height = np.asarray(cls.range, dtype=float)
    checks = {c.desc: _measure(cls.target, hours, height, c) for c in case.checks}

    if plot_dir is not None:
        from ceiloclass.plot import plot_classification

        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_classification(
            cls,
            str(plot_dir / f"{case.id}.png"),
            beta=ceilo.beta,
            depol=ceilo.depol,
            show=False,
        )

    return Result(
        fractions=_fractions(cls.target),
        strong_beta=float(cls.strong_beta),
        t0_median=float(np.nanmedian(cls.t0_alt)),
        checks=checks,
    )


# --- Comparison & reporting -------------------------------------------------


def _drifted(old: float, new: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(new - old) > max(abs_tol, rel_tol * abs(old))


def _drifts(case: Case, res: Result, base: dict) -> list[str]:
    """Human-readable list of metrics that drifted beyond tolerance."""
    out: list[str] = []
    for name, new in res.fractions.items():
        old = base.get("fractions", {}).get(name)
        if old is not None and _drifted(old, new, 0.003, 0.05):
            out.append(f"{name} {old * 100:.1f}% -> {new * 100:.1f}%")
    if (old := base.get("strong_beta")) is not None and _drifted(
        old, res.strong_beta, 0.0, 0.05
    ):
        out.append(f"strong_beta {old:.2e} -> {res.strong_beta:.2e}")
    if (old := base.get("t0_median")) is not None and _drifted(
        old, res.t0_median, 50.0, 0.0
    ):
        out.append(f"t0 {old:.0f}m -> {res.t0_median:.0f}m")
    for desc, new in res.checks.items():
        old = base.get("checks", {}).get(desc)
        if old is not None and _drifted(old, new, 0.003, 0.05):
            out.append(f"check[{desc}] {old * 100:.2f}% -> {new * 100:.2f}%")
    return out


def _check_status(case: Case, res: Result) -> list[tuple[bool, str]]:
    """Evaluate each semantic check; return (passed, message) per check."""
    out: list[tuple[bool, str]] = []
    for c in case.checks:
        frac = res.checks[c.desc]
        ok = (c.min_frac is None or frac >= c.min_frac) and (
            c.max_frac is None or frac <= c.max_frac
        )
        bound = []
        if c.min_frac is not None:
            bound.append(f">= {c.min_frac * 100:.2f}%")
        if c.max_frac is not None:
            bound.append(f"<= {c.max_frac * 100:.2f}%")
        msg = f"{c.desc}: {c.target.name} = {frac * 100:.2f}% ({', '.join(bound)})"
        out.append((ok, msg))
    return out


class _Style:
    """ANSI colours, disabled when stdout is not a terminal."""

    def __init__(self) -> None:
        self.on = sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def green(self, t: str) -> str:
        return self._c("32", t)

    def red(self, t: str) -> str:
        return self._c("31", t)

    def yellow(self, t: str) -> str:
        return self._c("33", t)

    def bold(self, t: str) -> str:
        return self._c("1", t)


def main(argv: list[str] | None = None) -> int:
    """Run the suite and return an exit code (0 = all good)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", help="run only cases whose id contains this string")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"cache directory for fetched files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--plot-dir",
        default=str(DEFAULT_PLOT_DIR),
        help=f"directory for the per-case classification PNG files "
        f"(default: {DEFAULT_PLOT_DIR})",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip writing classification PNG files"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; use only cached files (error if a case "
        "is not cached)",
    )
    parser.add_argument(
        "--update-baseline", action="store_true", help="overwrite the stored baseline"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat baseline drift as failure"
    )
    args = parser.parse_args(argv)

    style = _Style()
    cases = [c for c in CASES if not args.only or args.only in c.id]
    if not cases:
        print(f"no cases match {args.only!r}")
        return 1

    if args.list:
        for c in cases:
            print(f"{c.id:24s} {c.site} {c.date} [{c.instrument}]")
            print(f"    {c.note}")
        return 0

    baseline: dict = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())

    data_dir = Path(args.data_dir)
    plot_dir = None if args.no_plot else Path(args.plot_dir)
    manifest = _load_manifest(data_dir)
    new_baseline: dict = dict(baseline)
    n_fail = n_drift = n_error = 0

    for case in cases:
        print(style.bold(f"\n[{case.id}] {case.site} {case.date} [{case.instrument}]"))
        try:
            res = run_case(case, data_dir, plot_dir, manifest, offline=args.offline)
        except CacheMiss as e:
            n_error += 1
            print(style.red(f"  ERROR: {e}"))
            continue
        except Exception as e:  # noqa: BLE001 - report any case failure, keep going
            n_error += 1
            print(style.red(f"  ERROR: {e}"))
            traceback.print_exc()
            continue

        fr = res.fractions
        print(
            f"  strong_beta {res.strong_beta:.2e}  0C {res.t0_median:.0f} m  | "
            + "  ".join(
                f"{t.name[:4]} {fr[t.name] * 100:.1f}" for t in Target if fr[t.name] > 0
            )
        )
        for ok, msg in _check_status(case, res):
            if ok:
                print("  " + style.green("PASS ") + msg)
            else:
                n_fail += 1
                print("  " + style.red("FAIL ") + msg)

        base = baseline.get(case.id)
        if base is None:
            print("  " + style.yellow("NEW   no baseline yet"))
        else:
            drifts = _drifts(case, res, base)
            if drifts:
                n_drift += len(drifts)
                print("  " + style.yellow("DRIFT ") + "; ".join(drifts))
            else:
                print("  baseline: no drift")

        new_baseline[case.id] = _snapshot(res)

    # Persist the cache index so a later run with the files present can skip the
    # portal entirely (and so --offline knows what each case maps to).
    if not args.offline and manifest:
        _save_manifest(data_dir, manifest)

    if args.update_baseline:
        text = json.dumps(new_baseline, indent=2, sort_keys=True)
        BASELINE_PATH.write_text(text + "\n")
        print(style.bold(f"\nwrote baseline -> {BASELINE_PATH}"))

    print(
        style.bold(
            f"\n{len(cases)} cases: {n_fail} check failures, "
            f"{n_drift} drifts, {n_error} errors"
        )
    )
    bad = n_fail + n_error + (n_drift if args.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
