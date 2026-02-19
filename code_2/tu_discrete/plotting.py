"""Plotting utilities for the discrete-time TU model.

This module owns all matplotlib usage so solver code stays focused on economics
and numerical methods.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from .config import PlotConfig, TypeGridConfig
from .results import SimulationResult


def _ensure_output_dir(path: str) -> None:
    """Create output directory if needed."""
    os.makedirs(path, exist_ok=True)


def plot_matching_set(
    result: SimulationResult,
    grid_config: TypeGridConfig,
    plot_config: PlotConfig,
) -> str:
    """Save the matching-set figure and return its path."""
    _ensure_output_dir(plot_config.output_dir)

    n = grid_config.n
    contributions = result.contributions
    l_density = result.l_density
    alpha_binary = result.alpha_binary

    color_matching = "Blues"
    X = np.linspace(1 / n / 2, 1 - 1 / n / 2, n)
    Y = X
    levels = [0.1, 1]

    plt.figure(figsize=(10, 10))
    plt.rc("axes", labelsize=40)
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)

    if grid_config.distribution == "uniform":
        plt.contour(X, Y, alpha_binary, levels, colors="k")
        plt.contourf(X, Y, alpha_binary, levels, cmap=color_matching)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%g"))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%g"))
        plt.gca().tick_params(axis="x", pad=10)
        plt.gca().tick_params(axis="y", pad=10)
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        rect_set = [left, bottom, width, height]
        rect_hist_x = [left, bottom_h, width, 0.2]
        rect_hist_y = [left_h, bottom, 0.2, height]

        ax_set = plt.axes(rect_set)
        ax_hist_x = plt.axes(rect_hist_x)
        ax_hist_y = plt.axes(rect_hist_y)

        ax_set.contour(X, Y, alpha_binary, levels, colors="k")
        ax_set.contourf(X, Y, alpha_binary, levels, cmap=color_matching)

        ax_hist_x.axis("off")
        ax_hist_y.axis("off")
        ax_hist_x.plot(contributions, l_density)
        ax_hist_x.fill_between(contributions, 0, l_density, alpha=0.2)
        ax_hist_y.plot(l_density, contributions)
        ax_hist_y.fill_betweenx(contributions, 0, l_density, alpha=0.2)

        ax_hist_x.set_xlim((0, 1))
        ax_hist_y.set_ylim((0, 1))
        ax_set.set_xlim((0, 1))
        ax_set.set_ylim((0, 1))
        ax_set.set_xticks(np.arange(0, 1.1, 0.2))
        ax_set.set_yticks(np.arange(0, 1.1, 0.2))
        ax_set.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_set.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_set.tick_params(axis="x", pad=10)
        ax_set.tick_params(axis="y", pad=10)
        ax_set.set_xlabel("x")
        ax_set.set_ylabel("y")

    fig_path = os.path.join(plot_config.output_dir, plot_config.matching_set_filename)
    plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return fig_path


def _plot_alpha_colormap(
    alpha_map: np.ndarray,
    output_path: str,
    title: str,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "alpha",
    show_ylabel: bool = True,
) -> None:
    """Plot a single alpha map as a colormap."""
    y_ticks = [0.0, 0.5, 1.0]
    y_tick_labels = ["0", "0.5", "1.0"]
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        alpha_map,
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("x")
    if show_ylabel:
        plt.ylabel("y")
    plt.yticks(y_ticks, y_tick_labels)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def plot_alpha_effort_comparison(result: SimulationResult, plot_config: PlotConfig) -> None:
    """Save alpha maps with effort, without effort, and their difference."""
    _ensure_output_dir(plot_config.output_dir)

    alpha_with_effort = result.alpha_binary
    alpha_without_effort = result.alpha_no_effort_binary
    alpha_diff = alpha_without_effort - alpha_with_effort

    _plot_alpha_colormap(
        alpha_map=alpha_with_effort,
        output_path=os.path.join(plot_config.output_dir, "alpha_map_with_effort.pdf"),
        title="Alpha Map (With Effort)",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar_label="alpha",
    )
    _plot_alpha_colormap(
        alpha_map=alpha_without_effort,
        output_path=os.path.join(plot_config.output_dir, "alpha_map_without_effort.pdf"),
        title="Alpha Map (No Effort)",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        cbar_label="alpha",
    )
    _plot_alpha_colormap(
        alpha_map=alpha_diff,
        output_path=os.path.join(plot_config.output_dir, "alpha_map_effort_difference.pdf"),
        title="Alpha Difference (No Effort - With Effort)",
        cmap="bwr",
        vmin=-1.0,
        vmax=1.0,
        cbar_label="difference",
        show_ylabel=False,
    )

    # Side-by-side quick comparison.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    panels = [
        (alpha_with_effort, "With Effort", "Blues", 0.0, 1.0, "alpha", True),
        (alpha_without_effort, "No Effort", "Greens", 0.0, 1.0, "alpha", True),
        (alpha_diff, "Difference", "bwr", -1.0, 1.0, "difference", False),
    ]
    for ax, (mat, title, cmap, vmin, vmax, cbar_label, show_ylabel) in zip(axes, panels):
        y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        y_tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        x_tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[0, 1, 0, 1],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_title(title)
        ax.grid()

    fig.savefig(
        os.path.join(plot_config.output_dir, "alpha_map_effort_comparison.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def plot_value_diagnostics(result: SimulationResult, plot_config: PlotConfig) -> None:
    """Save diagnostics for values, value gains, unmatched density, and colormaps."""
    _ensure_output_dir(plot_config.output_dir)

    x = result.contributions

    # A) V with effort vs no effort.
    plt.figure(figsize=(10, 6))
    plt.plot(x, result.V, lw=2, label="V (with effort)")
    plt.plot(x, result.V_no_effort, lw=2, linestyle="--", label="V (no effort)")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Unmatched Value by Type")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(plot_config.output_dir, "value_V_by_type.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # B) Delta value and one-step upgrade gain.
    plt.figure(figsize=(10, 6))
    plt.plot(x, result.deltaV, lw=2, label="ΔV = V - V_no_effort")
    plt.plot(x, result.gain_upgrade, lw=2, linestyle="--", label="p(e*) * [V(next) - V]")
    plt.xlabel("x")
    plt.ylabel("Value Difference")
    plt.title("Value Gain from Optimal Effort by Type")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(plot_config.output_dir, "value_deltaV_by_type.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # C) Unmatched density.
    plt.figure(figsize=(10, 6))
    plt.plot(x, result.u, lw=2, color="tab:orange")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Stationary Unmatched Density by Type")
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(plot_config.output_dir, "unmatched_density_u_by_type.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # D) 2x1: colormap + line plot for Delta V.
    fig, (ax_cm, ax_line) = plt.subplots(1, 2, figsize=(16, 3), gridspec_kw={"width_ratios": [1, 1]})
    im = ax_cm.imshow(
        result.deltaV[np.newaxis, :],
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[0, 1, 0, 1],
    )
    ax_cm.set_xlabel("x")
    ax_cm.set_yticks([])
    ax_cm.set_title("ΔV Colormap by Type")
    cbar = fig.colorbar(im, ax=ax_cm)
    cbar.set_label("ΔV")
    ax_line.plot(result.contributions, result.deltaV, lw=1.5, color="royalblue")
    ax_line.set_xlabel("x")
    ax_line.set_ylabel("ΔV")
    ax_line.set_title("ΔV by Type")
    ax_line.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_config.output_dir, "deltaV_colormap.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # E) 2x1: colormap + line plot for optimal effort.
    fig, (ax_cm, ax_line) = plt.subplots(1, 2, figsize=(16, 3), gridspec_kw={"width_ratios": [1, 1]})
    im = ax_cm.imshow(
        result.effort_policy[np.newaxis, :],
        aspect="auto",
        cmap="plasma",
        origin="lower",
        extent=[0, 1, 0, 1],
    )
    ax_cm.set_xlabel("x")
    ax_cm.set_yticks([])
    ax_cm.set_title("Optimal Effort Colormap by Type")
    cbar = fig.colorbar(im, ax=ax_cm)
    cbar.set_label("e*")
    ax_line.plot(result.contributions, result.effort_policy, lw=1.5, color="darkorange")
    ax_line.set_xlabel("x")
    ax_line.set_ylabel("e*")
    ax_line.set_title("Optimal Effort by Type")
    ax_line.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_config.output_dir, "effort_colormap.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_all_plots(
    result: SimulationResult,
    grid_config: TypeGridConfig,
    plot_config: PlotConfig,
) -> str:
    """Create all plots and return the path to the matching-set figure."""
    matching_path = plot_matching_set(result=result, grid_config=grid_config, plot_config=plot_config)
    plot_alpha_effort_comparison(result=result, plot_config=plot_config)
    plot_value_diagnostics(result=result, plot_config=plot_config)
    return matching_path
