"""
plot_utils.py — Shared Plotting Helpers
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def annotated_heatmap(matrix: pd.DataFrame, title: str, out_path: str,
                       cmap: str = "coolwarm", vmin=None, vmax=None,
                       fmt: str = "{:.2f}", figsize=None, cbar_label: str = None,
                       xlabel: str = None, ylabel: str = None,
                       nan_note: str = "Blank cell = no data for that pair x instrument (not zero)."):
    if matrix.empty:
        return False

    data = matrix.values.astype(float)
    n_rows, n_cols = data.shape
    figsize = figsize or (max(6, 0.6 * n_cols + 2), max(4, 0.5 * n_rows + 2))

    finite = data[np.isfinite(data)]
    vmin_ = vmin if vmin is not None else (float(np.min(finite)) if finite.size else 0.0)
    vmax_ = vmax if vmax is not None else (float(np.max(finite)) if finite.size else 1.0)
    if vmin_ == vmax_:
        vmax_ = vmin_ + 1e-6

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)

    cmap_obj = im.get_cmap()
    norm = im.norm
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            r, g, b, _ = cmap_obj(norm(val))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.55 else "black"
            outline = "black" if color == "white" else "white"
            txt = ax.text(j, i, fmt.format(val), ha="center", va="center",
                           color=color, fontsize=8, fontweight="bold")
            txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground=outline)])

    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")

    has_nan = np.isnan(data).any()
    if has_nan and nan_note:

        ax.annotate(nan_note, xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -55), textcoords="offset points",
                    ha="center", va="top", fontsize=8, style="italic",
                    color="dimgray", clip_on=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True


def grouped_bar(df: pd.DataFrame, x: str, y: str, group: str = None,
                 title: str = "", out_path: str = "", ylabel: str = "",
                 yerr_low: str = None, yerr_high: str = None,
                 horizontal: bool = False, figsize=None, rotation: int = 45):

    if df.empty:
        return False

    x_vals = list(pd.unique(df[x]))
    figsize = figsize or (max(7, 0.6 * len(x_vals) + 2), 5.5)
    fig, ax = plt.subplots(figsize=figsize)

    def _errors(sub):
        if yerr_low and yerr_high and yerr_low in sub.columns and yerr_high in sub.columns:
            lo = np.maximum(0, sub[y] - sub[yerr_low]).fillna(0).values
            hi = np.maximum(0, sub[yerr_high] - sub[y]).fillna(0).values
            return [lo, hi]
        return None

    if group is None:
        sub = df.set_index(x).loc[x_vals].reset_index()
        errs = _errors(sub)
        bar_fn = ax.barh if horizontal else ax.bar
        bar_fn(sub[x].astype(str), sub[y], xerr=errs if horizontal else None,
               yerr=None if horizontal else errs, capsize=4, color="steelblue",
               edgecolor="black", alpha=0.85)
    else:
        groups = list(pd.unique(df[group]))
        n_groups = len(groups)
        width = 0.8 / max(n_groups, 1)
        positions = np.arange(len(x_vals))
        cmap = plt.get_cmap("tab10")
        for gi, gval in enumerate(groups):
            sub = df[df[group] == gval].set_index(x).reindex(x_vals).reset_index()
            errs = _errors(sub)
            offset = (gi - (n_groups - 1) / 2.0) * width
            ax.bar(positions + offset, sub[y], width=width, yerr=errs, capsize=3,
                   label=str(gval), color=cmap(gi % 10), edgecolor="black", alpha=0.85)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(v) for v in x_vals], rotation=rotation, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title=group)

    if group is None:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
    ax.set_ylabel(ylabel or y)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True


def radar_chart(categories: list, series: dict, title: str, out_path: str,
                 figsize=(7.5, 7.5), fill_alpha: float = 0.15,
                 value_labels: bool = False, fmt: str = "{:.1f}",
                 ylim: tuple = None,
                 missing_note: str = "Vertices pulled to 0 mark missing data for that category — read the table for exact values."):
 
    if not categories or not series:
        return False

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)


    cleaned = {}
    had_missing = False
    for name, raw_vals in series.items():
        vals = []
        for v in raw_vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(0.0)
                had_missing = True
            else:
                vals.append(float(v))
        cleaned[name] = vals
    all_vals = [v for vals in cleaned.values() for v in vals]
    top = max(all_vals) if all_vals else 1.0
    n_series = len(cleaned)
    radial_step = top * 0.06  # spacing between stacked labels at one vertex

    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(si % 10) for si, name in enumerate(cleaned)}
    for name, vals in cleaned.items():
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, linewidth=2, label=str(name), color=colors[name], marker="o", markersize=4)
        ax.fill(angles, vals_closed, alpha=fill_alpha, color=colors[name])

    if value_labels:

        min_gap = max(radial_step, 1e-6)
        marker_clearance = top * 0.045  # keeps a label off its own marker dot
        for vi in range(n):
            ang = angles[vi]
            entries = sorted(
                ((name, vals[vi]) for name, vals in cleaned.items()),
                key=lambda e: e[1],
            )
            placed_radius = None
            for name, v in entries:
                base = v + marker_clearance
                target = base if placed_radius is None else max(base, placed_radius + min_gap)
                placed_radius = target
                txt = ax.text(ang, target, fmt.format(v), fontsize=7, color=colors[name],
                               ha="center", va="center", fontweight="bold")
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    if ylim:
        ax.set_ylim(*ylim)
    elif all_vals:
        top = max(all_vals)
        ax.set_ylim(0, top * 1.15 if top > 0 else 1)

    ax.set_title(title, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    ax.grid(alpha=0.4)

    if had_missing and missing_note:
        ax.annotate(missing_note, xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -55), textcoords="offset points",
                    ha="center", va="top", fontsize=8, style="italic",
                    color="dimgray", clip_on=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True
