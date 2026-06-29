"""Generate the adaptive-threshold illustration for the whitepaper.

Reproduces the four representative backscatter distributions from the
`_adaptive_strong_beta` regression tests (each standing for a documented
site/day), runs the real threshold finder, and plots each histogram coloured by
the resulting aerosol / cloud-precipitation split.

Run from the repo root:  python docs/adaptive_threshold_figure.py
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from numpy import ma

from ceiloclass.classification import _adaptive_strong_beta

OUT = "docs/adaptive_threshold.png"

AEROSOL_C = "#4c72b0"
CLOUD_C = "#dd8452"


# Builders mirror the regression tests verbatim (same seed, same single RNG),
# so each panel is exactly the distribution that test exercises.
def _bimodal() -> np.ndarray:
    rng = np.random.default_rng(0)
    aerosol = rng.lognormal(np.log(2e-7), 0.18, 20000)
    cloud = rng.lognormal(np.log(1e-5), 0.18, 5000)
    return np.concatenate([aerosol, cloud])


def _two_aerosol() -> np.ndarray:
    rng = np.random.default_rng(4)
    low = rng.lognormal(np.log(4e-7), 0.15, 20000)
    dust = rng.lognormal(np.log(1e-6), 0.15, 8000)
    return np.concatenate([low, dust])


def _single() -> np.ndarray:
    return np.random.default_rng(2).lognormal(np.log(2e-6), 0.2, 20000)


def _runaway() -> np.ndarray:
    return np.random.default_rng(3).lognormal(np.log(3e-5), 0.25, 20000)


CASES = [
    (
        "(a) Cloudy day: bimodal",
        "threshold at the valley between aerosol and cloud",
        _bimodal,
    ),
    (
        "(b) Dusty day: two aerosol modes",
        "upper (dust) mode kept as aerosol, not split as cloud",
        _two_aerosol,
    ),
    (
        "(c) Clean day: single aerosol mode",
        "threshold at the shoulder past the peak",
        _single,
    ),
    (
        "(d) Polar-winter low-cloud continuum",
        "no separable aerosol mode: capped at the 1e-5 ceiling",
        _runaway,
    ),
]


def _sci(value: float) -> str:
    r"""Compact LaTeX scientific label, e.g. 2e-6 -> $2{\times}10^{-6}$."""
    exp = math.floor(math.log10(value) + 1e-9)
    mant = round(value / 10**exp)
    if mant == 10:
        mant, exp = 1, exp + 1
    return rf"$10^{{{exp}}}$" if mant == 1 else rf"${mant}{{\times}}10^{{{exp}}}$"


def _clean_log_ticks(ax: plt.Axes, lo: float, hi: float) -> None:
    """Explicit log ticks at a density set by the span; drop minor labels."""
    decades = math.log10(hi / lo)
    mantissas = (1, 2, 3, 5) if decades <= 1.2 else (1, 3) if decades <= 2.5 else (1,)
    ticks = [
        m * 10.0**e
        for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
        for m in mantissas
        if lo * 0.95 <= m * 10.0**e <= hi * 1.05
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_sci(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def _panel(ax: plt.Axes, title: str, subtitle: str, values: np.ndarray) -> None:
    threshold = _adaptive_strong_beta(ma.asarray(values.reshape(1, -1)))
    lo, hi = np.percentile(values, [1.0, 99.9])
    edges = np.logspace(np.log10(lo), np.log10(hi), 61)
    counts, _ = np.histogram(values, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    colors = np.where(centers < threshold, AEROSOL_C, CLOUD_C)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colors)
    ax.axvline(threshold, color="black", ls="--", lw=1.3)

    # Keep the threshold label inside the axes: flip to the left near the edge.
    frac = (math.log10(threshold) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    right = frac > 0.55
    ax.annotate(
        rf"$\beta_{{\rm strong}}\!=\!{_sci(threshold)[1:-1]}$",
        xy=(threshold, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(-6 if right else 6, 0),
        textcoords="offset points",
        ha="right" if right else "left",
        va="top",
        fontsize=8.5,
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, loc="left", linespacing=1.4)
    ax.set_xscale("log")
    _clean_log_ticks(ax, lo, hi)
    ax.set_xlabel(r"backscatter $\beta$  (sr$^{-1}$ m$^{-1}$)", fontsize=9)
    ax.set_ylabel("pixel count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.margins(x=0.02)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    for ax, (title, subtitle, build) in zip(axes.ravel(), CASES, strict=True):
        _panel(ax, title, subtitle, build())
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=AEROSOL_C, label="aerosol (below threshold)"),
        plt.Rectangle((0, 0), 1, 1, color=CLOUD_C, label="cloud / precipitation"),
        plt.Line2D(
            [0], [0], color="black", ls="--", lw=1.3, label=r"$\beta_{\rm strong}$"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.5)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
