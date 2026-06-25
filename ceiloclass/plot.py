"""Plotting helpers for classification results.

Matplotlib is an optional dependency (``pip install ceiloclass[plot]``).
"""

from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy import ma

from .classification import Classification, Target

# Colors follow the CloudnetPy target-classification convention so these plots
# read the same way as standard Cloudnet figures.
_LABELS: dict[Target, tuple[str, str]] = {
    Target.CLEAR: ("Clear", "#ffffff"),
    Target.DROPLET: ("Liquid droplets", "#6cffec"),  # Cloudnet "cloud droplets"
    Target.DRIZZLE_OR_RAIN: ("Drizzle/rain", "#209ff3"),  # Cloudnet "drizzle/rain"
    Target.ICE: ("Ice", "#a0b0bb"),  # Cloudnet "ice" (lightsteel)
    Target.SUPERCOOLED: ("Supercooled liquid", "#464ab9"),  # Cloudnet "supercooled"
    Target.AEROSOL: ("Aerosol", "#cebc89"),  # Cloudnet "aerosol" (lightbrown)
}


def plot_classification(
    classification: Classification,
    path: str | PathLike | None = None,
    *,
    beta: npt.NDArray[np.floating] | None = None,
    depol: npt.NDArray[np.floating] | None = None,
    show: bool = False,
    max_height: float = 12000,
) -> None:
    """Plot a target-classification curtain, saving and/or showing it.

    Args:
        classification: A `Classification` result.
        path: Output image path; if given, the figure is saved here.
        beta: Optional backscatter to show in a top panel (e.g. ceilo.beta).
        depol: Optional depolarization ratio to show in a panel (CL61 only,
            e.g. ceilo.depol); useful for eyeballing ice vs liquid.
        show: Display the figure in an interactive window.
        max_height: Upper limit of the range axis (m).
    """
    try:
        import matplotlib  # noqa: PLC0415
    except ImportError as e:
        msg = "Plotting requires matplotlib; install with: pip install ceiloclass[plot]"
        raise ImportError(msg) from e
    if not show:
        matplotlib.use("Agg")  # headless: no display needed for saving
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import (  # noqa: PLC0415
        BoundaryNorm,
        ListedColormap,
        LogNorm,
    )

    time = classification.time
    # Only render up to the displayed height — plotting all gates (CL61 reaches
    # ~15 km) is the main cost even when viewing the lowest few km.
    keep = np.asarray(classification.range) <= max_height * 1.05
    rng_km = np.asarray(classification.range)[keep] / 1000
    target = classification.target[:, keep]
    if beta is not None:
        beta = ma.asarray(beta)[:, keep]
    if depol is not None:
        depol = ma.asarray(depol)[:, keep]
    cmap = ListedColormap([_LABELS[Target(i)][1] for i in range(len(Target))])
    norm = BoundaryNorm(np.arange(-0.5, len(Target) + 0.5, 1), cmap.N)

    n_curtain = 1 + (beta is not None) + (depol is not None)
    show_hist = beta is not None
    n_rows = n_curtain + int(show_hist)
    fig = plt.figure(figsize=(12, 3.6 * n_curtain + (3.0 if show_hist else 0.0)))
    gs = fig.add_gridspec(n_rows, 1)
    # Curtains share time and range; the histogram panel (if any) is independent.
    ax0 = fig.add_subplot(gs[0, 0])
    axes = [ax0] + [
        fig.add_subplot(gs[i, 0], sharex=ax0, sharey=ax0) for i in range(1, n_curtain)
    ]

    t0_km = np.asarray(classification.t0_alt) / 1000
    # Hide the isotherm when it sits at the ground for every profile (the whole
    # column is sub-freezing): a flat line on the axis floor is just noise.
    hide_t0 = bool(np.all(t0_km <= rng_km.min()))

    panel = 0
    if beta is not None:
        ax = axes[panel]
        panel += 1
        masked = ma.masked_less_equal(ma.array(beta), 0)
        mesh = ax.pcolormesh(
            time,
            rng_km,
            masked.T,
            norm=LogNorm(1e-7, 1e-4),
            cmap="viridis",
            shading="auto",
        )
        ax.set_title("Screened backscatter")
        ax.set_ylabel("Range (km)")
        fig.colorbar(mesh, ax=ax, label="beta (sr⁻¹ m⁻¹)", pad=0.01)
        _plot_t0(ax, time, t0_km, hide_t0)

    if depol is not None:
        ax = axes[panel]
        panel += 1
        masked = ma.masked_invalid(ma.array(depol))
        if beta is not None:
            # Hide clear-air depol noise: only show where backscatter survived.
            masked = ma.masked_where(
                ma.getmaskarray(ma.masked_less_equal(beta, 0)), masked
            )
        mesh = ax.pcolormesh(
            time,
            rng_km,
            masked.T,
            vmin=0,
            vmax=0.5,
            cmap="turbo",
            shading="auto",
        )
        ax.set_title("Depolarization ratio")
        ax.set_ylabel("Range (km)")
        fig.colorbar(mesh, ax=ax, label="depolarization", pad=0.01)
        _plot_t0(ax, time, t0_km, hide_t0)

    ax = axes[-1]
    mesh = ax.pcolormesh(time, rng_km, target.T, cmap=cmap, norm=norm, shading="auto")
    _plot_t0(ax, time, t0_km, hide_t0)
    ax.set_title("Target classification")
    ax.set_ylabel("Range (km)")
    ax.set_xlabel("Time (UTC)")
    # Show only the hour:minute on the shared time axis; the date is redundant
    # (a single day) and clutters the labels.
    import matplotlib.dates as mdates  # noqa: PLC0415

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(0, min(max_height / 1000, rng_km.max()))
    cbar = fig.colorbar(mesh, ax=ax, ticks=range(len(Target)), pad=0.01)
    cbar.ax.set_yticklabels(
        [_LABELS[Target(i)][0] for i in range(len(Target))], fontsize=7
    )

    hist_ax = None
    if beta is not None:
        hist_ax = fig.add_subplot(gs[n_curtain, 0])
        _plot_beta_hist(hist_ax, beta, classification.strong_beta)

    fig.tight_layout()
    if hist_ax is not None:
        # The curtain panels are narrowed by their colorbars; match the histogram
        # to a curtain's horizontal extent so all panels line up.
        ref = axes[0].get_position()
        pos = hist_ax.get_position()
        hist_ax.set_position((ref.x0, pos.y0, ref.width, pos.height))
    if path is not None:
        fig.savefig(path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def _plot_beta_hist(
    ax: Any, beta: npt.NDArray[np.floating], threshold: float | None = None
) -> None:
    """Plot a log-x histogram of the (positive, unmasked) screened backscatter.

    The cloud/precipitation vs aerosol threshold is drawn as a vertical line.
    """
    values = ma.filled(ma.asarray(beta), np.nan).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    ax.set_title("Screened backscatter histogram")
    ax.set_xlabel("beta (sr⁻¹ m⁻¹)")
    ax.set_ylabel("Count")
    ax.grid(True, which="both", alpha=0.25)
    if values.size == 0:
        return
    # Trim only the sparse low tail; keep the full high end so the rare but very
    # strong liquid-cloud pixels stay in view. A log count axis then makes those
    # low-population high-beta bins visible next to the dominant aerosol peak.
    lo = float(np.percentile(values, 0.5))
    hi = float(values.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 100)
    ax.hist(values, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    if threshold is not None:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"strong_beta = {threshold:.0e}",
        )
        ax.legend(loc="upper right", fontsize=7)


def _plot_t0(
    ax: Any,
    time: npt.NDArray[np.object_],
    t0_km: npt.NDArray[np.floating],
    hide: bool = False,
) -> None:
    """Overlay the 0 degC isotherm as a dashed line, readable on any colormap.

    Does nothing when `hide` is set (the isotherm is at the ground throughout).
    """
    if hide:
        return
    import matplotlib.patheffects as pe  # noqa: PLC0415

    (line,) = ax.plot(
        time,
        t0_km,
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        label="0 °C",
    )
    line.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    ax.legend(loc="upper right", fontsize=7)
