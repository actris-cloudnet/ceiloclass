"""Plotting helpers for classification results."""

from os import PathLike
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from numpy import ma

from .classification import MAX_PHYSICAL_BETA, Classification, Target

# Colors follow the CloudnetPy target-classification convention so these plots
# read the same way as standard Cloudnet figures.
_LABELS: dict[Target, tuple[str, str]] = {
    Target.CLEAR: ("Clear", "#ffffff"),
    Target.DROPLET: ("Liquid droplets", "#6cffec"),
    Target.DRIZZLE_OR_RAIN: ("Drizzle/rain", "#209ff3"),
    Target.ICE: ("Ice", "#a0b0bb"),
    Target.SUPERCOOLED: ("Supercooled liquid", "#464ab9"),
    Target.AEROSOL: ("Aerosol", "#cebc89"),
    Target.ATTENUATED: ("Beam attenuated", "#ededed"),
}

# Figure width and curtain panel height (in), and the saved resolution.
_WIDTH, _PANEL_HEIGHT, _DPI = 12.0, 3.6, 110

# Longest time step (days) drawn as continuous data, as in Cloudnet figures.
_MAX_GAP = 10 / 1440


def plot_classification(
    classification: Classification,
    path: str | PathLike | None = None,
    *,
    beta: npt.NDArray[np.floating] | None = None,
    depol: npt.NDArray[np.floating] | None = None,
    show: bool = False,
    max_height: float = 12000,
    histogram: bool = True,
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
        histogram: Add the diagnostic backscatter histogram panel (with the
            cloud/aerosol threshold) when `beta` is given. Set `False` to omit it.
    """
    if not show:
        matplotlib.use("Agg")  # headless: no display needed for saving

    # Matplotlib date numbers: pcolorfast needs numeric coordinates, and the
    # t0 line uses the same units so everything stays aligned.
    time = mdates.date2num(classification.time)
    # Only render up to the displayed height; range ascends, so slice (no copy).
    rng = np.asarray(classification.range)
    keep = slice(int(np.searchsorted(rng, max_height * 1.05, side="right")))
    rng_km = rng[keep] / 1000
    target = classification.target[:, keep]
    if beta is not None:
        beta = ma.asarray(beta)[:, keep]
    if depol is not None:
        depol = ma.asarray(depol)[:, keep]
    x, profiles = _time_cells(time)
    y = _edges(rng_km)
    gaps = np.c_[x[:-1], x[1:]][profiles < 0]

    def thin(
        xlim: tuple[float, float], ylim: tuple[float, float], width: int, height: int
    ) -> dict[str, ma.MaskedArray]:
        rows = profiles[_pixel_cells(x, *xlim, width)]
        return _curtains(target, beta, depol, rows, _pixel_cells(y, *ylim, height))

    # Draw only the cell under each pixel: more can't show and costs seconds.
    extent = (x[0], x[-1], y[0], y[-1])
    data = thin(
        extent[:2], extent[2:], round(_WIDTH * _DPI), round(_PANEL_HEIGHT * _DPI)
    )
    images = {}
    cmap = ListedColormap([_LABELS[Target(i)][1] for i in range(len(Target))])
    norm = BoundaryNorm(np.arange(-0.5, len(Target) + 0.5, 1), cmap.N)

    n_curtain = 1 + (beta is not None) + (depol is not None)
    show_hist = beta is not None and histogram
    n_rows = n_curtain + int(show_hist)
    fig = plt.figure(
        figsize=(_WIDTH, _PANEL_HEIGHT * n_curtain + (3.0 if show_hist else 0.0))
    )
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
        images["beta"] = _plot_curtain(
            fig,
            ax,
            extent,
            data["beta"],
            title="Screened backscatter",
            cbar_label="beta (sr⁻¹ m⁻¹)",
            norm=LogNorm(1e-7, 1e-4),
            cmap="viridis",
        )
        _plot_t0(ax, time, t0_km, hide_t0)

    if depol is not None:
        ax = axes[panel]
        panel += 1
        images["depol"] = _plot_curtain(
            fig,
            ax,
            extent,
            data["depol"],
            title="Depolarization ratio",
            cbar_label="depolarization",
            vmin=0,
            vmax=0.5,
            cmap="turbo",
        )
        _plot_t0(ax, time, t0_km, hide_t0)

    ax = axes[-1]
    mesh = images["target"] = ax.pcolorfast(
        extent[:2], extent[2:], data["target"].T, cmap=cmap, norm=norm
    )
    _plot_t0(ax, time, t0_km, hide_t0)
    ax.set_title("Target classification")
    ax.set_ylabel("Range (km)")
    ax.set_xlabel("Time (UTC)")
    # Show only the hour:minute on the shared time axis; the date is redundant
    # (a single day) and clutters the labels. The locator must be set explicitly
    # since the numeric pcolorfast coordinates don't install date units.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Fixed limits, so that re-thinning the images never rescales the axes.
    ax.set_xlim(extent[:2])
    ax.set_ylim(0, min(max_height / 1000, rng_km.max()))
    cbar = fig.colorbar(mesh, ax=ax, ticks=range(len(Target)), pad=0.01)
    cbar.ax.set_yticklabels(
        [_LABELS[Target(i)][0] for i in range(len(Target))], fontsize=7
    )

    for ax in axes:
        _plot_gaps(ax, gaps)

    hist_ax = None
    if show_hist and beta is not None:
        hist_ax = fig.add_subplot(gs[n_curtain, 0])
        _plot_beta_hist(hist_ax, beta, classification.strong_beta)

    fig.tight_layout()
    # tight_layout leaves a placeholder engine that makes savefig draw twice.
    fig.set_layout_engine(None)
    if hist_ax is not None:
        # The curtain panels are narrowed by their colorbars; match the histogram
        # to a curtain's horizontal extent so all panels line up.
        ref = axes[0].get_position()
        pos = hist_ax.get_position()
        hist_ax.set_position((ref.x0, pos.y0, ref.width, pos.height))
    if path is not None:
        fig.savefig(path, dpi=_DPI)
    if show:

        def rethin(_event: object = None) -> None:
            """Thin again to the visible part, so zooming in shows every cell."""
            xlim = max(ax0.get_xlim()[0], x[0]), min(ax0.get_xlim()[1], x[-1])
            ylim = max(ax0.get_ylim()[0], y[0]), min(ax0.get_ylim()[1], y[-1])
            if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
                return
            size = max(round(ax0.bbox.width), 1), max(round(ax0.bbox.height), 1)
            for name, array in thin(xlim, ylim, *size).items():
                images[name].set_data(array.T)
                images[name].set_extent((*xlim, *ylim))
            fig.canvas.draw_idle()

        ax0.callbacks.connect("xlim_changed", rethin)
        ax0.callbacks.connect("ylim_changed", rethin)
        fig.canvas.mpl_connect("resize_event", rethin)
        plt.show()
    plt.close(fig)


def _edges(centers: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Cell edges (n+1) around 1-D cell `centers`, as `shading="auto"` computes."""
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.r_[2 * c[0] - mid[0], mid, 2 * c[-1] - mid[-1]]


def _time_cells(
    time: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Time cell edges and each cell's profile index, -1 for an empty gap cell.

    A step longer than `_MAX_GAP` (and than the usual step) is a data gap:
    rather than stretching the profiles either side across it, they keep the
    usual width and an empty cell fills the gap.
    """
    edges = _edges(time)
    profile = np.arange(time.size)
    if time.size < 2:
        return edges, profile
    step = np.diff(time)
    half = float(np.median(step)) / 2
    is_gap = step > max(_MAX_GAP, 3 * half)
    gaps = np.flatnonzero(is_gap)
    # An outer cell mirrors its only step, which must not be a gap.
    if is_gap[0]:
        edges[0] = time[0] - half
    if is_gap[-1]:
        edges[-1] = time[-1] + half
    edges[gaps + 1] = time[gaps + 1] - half
    edges = np.insert(edges, gaps + 1, time[gaps] + half)
    return edges, np.insert(profile, gaps + 1, -1)


def _take(
    data: npt.NDArray[Any], rows: npt.NDArray[np.intp], cols: npt.NDArray[np.intp]
) -> ma.MaskedArray:
    """`data` at the given profiles and gates, masked where the profile is a gap."""
    out = ma.array(data[np.ix_(rows, cols)])
    out[rows < 0] = ma.masked
    return out


def _curtains(
    target: npt.NDArray[np.integer],
    beta: ma.MaskedArray | None,
    depol: ma.MaskedArray | None,
    rows: npt.NDArray[np.intp],
    cols: npt.NDArray[np.intp],
) -> dict[str, ma.MaskedArray]:
    """The curtain panels' data at the given profiles and gates, as displayed."""
    data = {"target": _take(target, rows, cols)}
    if beta is not None:
        # Junk screen mirrors classify: unmasked fill values would otherwise
        # paint over-range streaks.
        beta = ma.masked_greater(_take(beta, rows, cols), MAX_PHYSICAL_BETA)
        data["beta"] = ma.masked_less_equal(beta, 0)
    if depol is not None:
        depol = ma.masked_invalid(_take(depol, rows, cols))
        if beta is not None:
            # Hide clear-air depol noise: only show where backscatter survived.
            depol = ma.masked_where(ma.getmaskarray(data["beta"]), depol)
        # Matplotlib scales the hidden cells too, and their fill values overflow.
        data["depol"] = ma.array(depol.filled(0), mask=ma.getmaskarray(depol))
    return data


def _pixel_cells(
    edges: npt.NDArray[np.floating], lo: float, hi: float, n: int
) -> npt.NDArray[np.intp]:
    """Index of the cell under each of `n` even pixel centers from `lo` to `hi`."""
    centers = np.linspace(lo, hi, 2 * n + 1)[1::2]
    return np.clip(np.searchsorted(edges, centers) - 1, 0, edges.size - 2)


def _plot_curtain(
    fig: Any,
    ax: Any,
    extent: tuple[float, float, float, float],
    data: ma.MaskedArray,
    *,
    title: str,
    cbar_label: str,
    **mesh_kwargs: Any,
) -> Any:
    """Draw one time-range curtain panel with its title, y-label and colorbar.

    Uses `pcolorfast` (image-based, an order of magnitude faster to draw than
    `pcolormesh`'s per-cell quads on a full-day curtain) on the even pixel grid
    that `data` was thinned to, and returns the image.
    """
    mesh = ax.pcolorfast(extent[:2], extent[2:], data.T, **mesh_kwargs)
    ax.set_title(title)
    ax.set_ylabel("Range (km)")
    fig.colorbar(mesh, ax=ax, label=cbar_label, pad=0.01)
    return mesh


def _plot_gaps(ax: Any, gaps: npt.NDArray[np.floating]) -> None:
    """Hatch the time spans without data, as Cloudnet figures do."""
    for start, end in gaps:
        ax.axvspan(
            start,
            end,
            hatch="//",
            facecolor="whitesmoke",
            edgecolor="lightgrey",
            linewidth=0.5,
            hatch_linewidth=0.5,
        )


def _plot_beta_hist(
    ax: Any, beta: npt.NDArray[np.floating], threshold: float | None = None
) -> None:
    """Plot a log-x histogram of the (positive, unmasked) screened backscatter.

    The cloud/precipitation vs aerosol threshold is drawn as a vertical line.
    """
    values = ma.filled(ma.asarray(beta), np.nan).ravel()
    values = values[np.isfinite(values) & (values > 0) & (values <= MAX_PHYSICAL_BETA)]
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
    counts, _ = np.histogram(values, bins=bins)
    # A single stairs artist draws far faster than ax.hist's per-bin patches.
    ax.stairs(
        counts, bins, fill=True, color="#1f77b4", edgecolor="white", linewidth=0.3
    )
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    if threshold is not None:
        exponent = int(np.floor(np.log10(threshold)))
        mantissa = threshold / 10.0**exponent
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=rf"Cloud/aerosol threshold = ${mantissa:.1f}\times10^{{{exponent}}}$",
        )
        ax.legend(loc="upper right", fontsize=7)


def _plot_t0(
    ax: Any,
    time: npt.NDArray[np.floating],
    t0_km: npt.NDArray[np.floating],
    hide: bool = False,
) -> None:
    """Overlay the 0 degC isotherm as a dashed line, readable on any colormap.

    Does nothing when `hide` is set (the isotherm is at the ground throughout).
    """
    if hide:
        return
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
