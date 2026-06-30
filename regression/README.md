# Regression / characterization suite

A curated set of **well-known, difficult** days across sites and lidars, run
against the current classifier to spot regressions and remaining problems.

There is no pixel-level ground truth, so this is not assertion-based unit
testing (those live in `tests/`). Instead each case is checked two ways:

- **semantic checks** — hand-written physical expectations (e.g. "Mindelo's
  bright marine haze must be aerosol, not drizzle"). They encode bugs we have
  fixed and fail loudly if one returns.
- **baseline drift** — a committed numeric snapshot (`baseline.json`) of each
  case's class fractions, `strong_beta`, melting level and check values. Any
  drift beyond tolerance is reported, so an unrelated change that shifts the
  output is visible even where no check covers it.

## Running

```sh
python regression/run.py                  # all cases: checks + baseline diff
python regression/run.py --only mindelo    # cases whose id contains "mindelo"
python regression/run.py --list            # list cases and exit
python regression/run.py --no-plot         # skip the per-case PNG files
python regression/run.py --offline         # use only cached files, no network
python regression/run.py --update-baseline # accept current output as baseline
python regression/run.py --strict          # treat baseline drift as failure too
```

Exit status is non-zero if any semantic check fails or a case errors (and, with
`--strict`, on any drift) — so it works as a pre-push / CI gate.

## Data and plots

Files are fetched from the Cloudnet portal and cached in `--data-dir` (default:
`regression/data/`). Each case's resolved filenames are recorded in
`<data-dir>/manifest.json`, so once the files are present later runs reuse them
**without contacting the portal at all**. `--offline` enforces that (and errors
on a case that is not yet cached). A classification PNG per case is written to
`--plot-dir` (default: `regression/plots/`) unless `--no-plot` is given. All of
this (data, manifest, plots) is gitignored.

## Adding a case

Edit the `CASES` list in `run.py`: give it an `id`, `site`, `date`, an
`instrument` id-substring (`cl61`, `chm15k`, `pollyxt`, `halo`, `da10`, …), a `note`
describing _what makes it hard_, and `checks` encoding the physics. Then run
`--update-baseline` to record its snapshot and commit the updated `baseline.json`.

**When a change is intentional** and shifts the numbers, re-run
`--update-baseline` and commit the new baseline alongside the code change, so the
diff documents the effect.
