"""
Teacher-hyperparameter selection and visualisation utilities.

Typical usage::

    from medal.selection import select_teacher_param, plot_reconstruction_error, plot_distortion_map

    df = results.load_metrics(X_test)
    opt_param = select_teacher_param(df, param_col="perplexity")

    # Tuning curve
    plot_reconstruction_error(df, opt_param, param_col="perplexity")

    # Distortion view at the optimal (or any chosen) hyperparameter
    emb_data = results.load_embeddings(y_train, X_test, y_test, params=[opt_param])
    plot_distortion_map(emb_data, opt_param, param_col="perplexity")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_RCPARAMS = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size":            11,
    "axes.labelsize":       11,
    "axes.titlesize":       12,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "lines.linewidth":      1.2,
    "lines.markersize":     3.5,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

_LINE_COLOR  = "#C0446A"
_BOX_COLOR   = "#4A90C4"
_LEFT_WSPACE = 0.25


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def select_teacher_param(
    df: pd.DataFrame,
    param_col: str,
    metric_col: str = "recon_loss",
    val_split: str = "Val",
    distill_threshold: float = 1e-5,
) -> object:
    """
    Choose the smallest teacher hyperparameter whose mean validation
    reconstruction loss lies within one SEM of the global minimum,
    considering only models that reached valid distillation during training.

    Two-step procedure:

    1. **Convergence filter** — keep only (param_col, seed) pairs whose
       Train distill_mse is below *distill_threshold*.  Pairs that never
       distilled properly are excluded from selection entirely.
    2. **One-SEM rule** — among the remaining configs, pick the smallest
       hyperparameter value whose mean Val loss is within one SEM of the
       global minimum.

    Parameters
    ----------
    df : pd.DataFrame
        Output of medal.sweep.SweepResults.load_metrics, containing
        columns [param_col, "seed", "split", metric_col, "distill_mse"].
    param_col : str
        Column name of the swept hyperparameter (e.g. "perplexity").
    metric_col : str
        Loss column to minimise (default "recon_loss").
    val_split : str
        Value in the "split" column to use for selection (default "Val").
    distill_threshold : float
        Maximum Train distillation MSE allowed for a model to be considered
        converged (default 1e-5).

    Returns
    -------
    opt_param : scalar
        The selected hyperparameter value.

    Raises
    ------
    ValueError
        If no (param_col, seed) pair passes the convergence filter.
    """
    import warnings

    # ── 1. Convergence filter ────────────────────────────────────────
    train_df = df[(df["split"] == "Train") & df["distill_mse"].notna()]

    if train_df.empty:
        warnings.warn(
            "No Train distill_mse values found in df — skipping convergence "
            "filter. Make sure load_metrics() was called with the Train split.",
            UserWarning,
            stacklevel=2,
        )
        filtered_df = df
    else:
        converged = train_df[train_df["distill_mse"] < distill_threshold][
            [param_col, "seed"]
        ]

        n_total = len(train_df)
        n_converged = len(converged)
        n_dropped = n_total - n_converged

        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped}/{n_total} (param, seed) pairs dropped: "
                f"Train distill_mse >= {distill_threshold:.0e}. "
                f"{n_converged} converged pairs remain.",
                UserWarning,
                stacklevel=2,
            )

        if converged.empty:
            raise ValueError(
                f"No (param, seed) pair reached Train distill_mse < "
                f"{distill_threshold:.0e}. Lower distill_threshold or check "
                "that training ran long enough."
            )

        filtered_df = df.merge(converged, on=[param_col, "seed"], how="inner")

    # ── 2. One-SEM rule on Val ───────────────────────────────────────
    val_df = filtered_df[filtered_df["split"] == val_split]
    stats  = val_df.groupby(param_col)[metric_col].agg(["mean", "sem"])
    # Add compatbility warning for custom embeddings
    if stats.empty:
        raise ValueError("No eligible validation results remain.")

    if stats["sem"].isna().any():
        raise ValueError(
            "The one-standard-error rule requires at least two "
            "eligible seed results per parameter value. "
            "Run additional seeds or use a separately defined "
            "single-seed selection rule."
        )
        
    argmin = stats["mean"].idxmin()
    best_mean, best_sem = stats.loc[argmin, ["mean", "sem"]]
    threshold = best_mean + best_sem
    candidates = stats[stats["mean"] <= threshold]
    return candidates.index.min()


def plot_reconstruction_error(
    df: pd.DataFrame,
    opt_param,
    param_col: str,
    show_boxplot: bool = False,
    xlabel: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 7),
    row_label_y: float = 1.15,
    legend_kw: Optional[dict] = None,
    xtick_fontsize: int = 10,
    share_x_lines: bool = True,
) -> "plt.Figure":
    """
    Plot reconstruction loss vs. swept hyperparameter (tuning curve).

    Parameters
    ----------
    df : pd.DataFrame
        Output of medal.sweep.SweepResults.load_metrics.
    opt_param : scalar
        Optimal parameter value returned by select_teacher_param.
    param_col : str
        Column name of the swept hyperparameter (e.g. "perplexity").
    show_boxplot : bool
        When True, adds a median boxplot column next to the mean line
        panel.  Default False (mean only).
    xlabel : str, optional
        x-axis label (defaults to param_col).
    figsize : tuple
    row_label_y, legend_kw, xtick_fontsize, share_x_lines
        Forwarded to _plot_line_box_panel.

    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update(_RCPARAMS)

    df = df.sort_values(by=param_col)
    unique_params = sorted(df[param_col].unique())

    val_df = df[df["split"] == "Val"]
    stats  = val_df.groupby(param_col)["recon_loss"].agg(["mean", "sem"])
    # Add compatibility fall back for custom embeddings
    if stats.empty or stats["sem"].isna().any():
        raise ValueError(
            "Plotting the one-standard-error interval requires "
            "at least two validation results per parameter value."
        )
    argmin = stats["mean"].idxmin()
    best_mean, best_sem = stats.loc[argmin, ["mean", "sem"]]
    one_std_range = (best_mean - best_sem, best_mean + best_sem)
    vline_x = unique_params.index(opt_param)

    xlabel = xlabel or param_col

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.15)
    gs = GridSpec(1, 1, figure=fig)
    _plot_line_box_panel(
        fig, gs[0, 0], df, param_col, unique_params,
        one_std_range, vline_x, xlabel,
        row_label_y=row_label_y, legend_kw=legend_kw,
        xtick_fontsize=xtick_fontsize,
        share_x_lines=share_x_lines,
        show_boxplot=show_boxplot,
    )

    plt.tight_layout()
    plt.show()
    return fig


def plot_distortion_map(
    emb_data: Dict[Any, Dict[str, tuple]],
    opt_param,
    param_col: str,
    param: Any = "best",
    palette=None,
    cmap: str = "magma",
    param_label: Optional[str] = None,
    colorbar_label: str = "Recon. MSE",
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Two-row distortion figure for a single hyperparameter value.

    Parameters
    ----------
    emb_data : dict
        {param_val: {"Train": (Z, recon_errors, labels), ...}}.
        Built by medal.sweep.SweepResults.load_embeddings.
    opt_param : scalar
        Optimal parameter value from :func:`select_teacher_param`.
    param_col : str
        Column name of the swept hyperparameter (used in the figure title).
    param : scalar or "best". Which hyperparameter value to display. "best" uses opt_param;
        pass any value present in emb_data to display that instead.
    palette : optional
        Colour palette for the label row (passed to seaborn).
    cmap : str
        Colourmap for the distortion row.
    param_label : str, optional
        Display name for the parameter in the figure title (defaults to
        param_col).
    colorbar_label : str
        Label for the distortion colorbar.
    figsize : tuple

    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update(_RCPARAMS)

    param_val = opt_param if param == "best" else param
    if param_val not in emb_data:
        raise ValueError(
            f"param={param_val!r} not found in emb_data. "
            f"Available: {sorted(emb_data)}"
        )

    param_label = param_label or param_col
    title_str = (
        f"Optimal {param_label}={param_val}"
        if param_val == opt_param else f"{param_label}={param_val}"
    )

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88)
    gs = GridSpec(1, 1, figure=fig)
    _plot_distortion_panel(
        fig, gs[0, 0], emb_data[param_val], title_str,
        palette=palette, cmap=cmap, colorbar_label=colorbar_label,
    )
    plt.show()
    return fig


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _plot_line_box_panel(
    fig, gs_element, analysis: pd.DataFrame, param_col: str,
    unique_params: List, one_std_range: Tuple[float, float], vline_x: int,
    xlabel: str,
    line_color: str = _LINE_COLOR,
    box_color: str  = _BOX_COLOR,
    left_wspace: float = _LEFT_WSPACE,
    row_label_y: float = 1.15,
    legend_kw: Optional[dict] = None,
    xtick_fontsize: int = 10,
    share_x_lines: bool = True,
    show_boxplot: bool = False,
):
    """Line (mean ± SEM) and optional boxplot (median) panels."""
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import seaborn as sns

    if legend_kw is None:
        legend_kw = {"frameon": False, "fontsize": 11, "loc": "upper left"}

    gs_kw   = {"wspace": left_wspace} if show_boxplot else {}
    gs_left = gs_element.subgridspec(3, 2 if show_boxplot else 1, hspace=0.45, **gs_kw)

    ax_line_train = fig.add_subplot(gs_left[0, 0])
    share_kw      = {"sharex": ax_line_train} if share_x_lines else {}
    ax_line_val   = fig.add_subplot(gs_left[1, 0], **share_kw)
    ax_line_test  = fig.add_subplot(gs_left[2, 0], **share_kw)
    line_axes     = [ax_line_train, ax_line_val, ax_line_test]

    if show_boxplot:
        ax_box_train = fig.add_subplot(gs_left[0, 1], sharey=ax_line_train)
        ax_box_val   = fig.add_subplot(gs_left[1, 1], sharex=ax_box_train, sharey=ax_line_val)
        ax_box_test  = fig.add_subplot(gs_left[2, 1], sharex=ax_box_train, sharey=ax_line_test)
        box_axes = [ax_box_train, ax_box_val, ax_box_test]
    else:
        box_axes = [None, None, None]

    row_label_x = 1 + left_wspace / 2 if show_boxplot else 0.5

    for ax_line, ax_box, split in zip(line_axes, box_axes, ["Train", "Val", "Test"]):
        subset = analysis[analysis["split"] == split]

        sns.pointplot(data=subset, x=param_col, y="recon_loss",
                      errorbar="se", color=line_color, ax=ax_line,
                      markersize=3, linewidth=1.2, err_kws={"linewidth": 0.6})
        sns.stripplot(data=subset, x=param_col, y="recon_loss",
                      color=line_color, size=1.8, alpha=0.3, ax=ax_line, zorder=0)

        if ax_box is not None:
            sns.boxplot(data=subset, x=param_col, y="recon_loss",
                        color=box_color, ax=ax_box, linewidth=0.6,
                        flierprops=dict(marker="o", markersize=1.5,
                                        markerfacecolor=box_color, alpha=0.4))

        for ax in [a for a in (ax_line, ax_box) if a is not None]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.6)
            ax.grid(axis="y", linewidth=0.35, color="gray", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=6)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        if ax_box is not None:
            ax_box.set_ylabel("")
            ax_box.tick_params(axis="y", labelleft=False)

        ax_line.text(row_label_x, row_label_y, split,
                     transform=ax_line.transAxes,
                     ha="center", va="center",
                     fontsize=12, fontweight="bold", clip_on=False)

        if split == "Val":
            ax_line.set_ylabel("Reconstruction Loss", labelpad=2,
                                fontsize=20, fontweight="bold")
            ax_line.fill_between(range(-1, len(unique_params) + 1),
                                  *one_std_range, color=line_color, alpha=0.12,
                                  label="1 SEM")
            ax_line.axvline(x=vline_x, ls="--", lw=0.8, color="#333333", zorder=5)
            ax_line.legend(**legend_kw)
            ax_line.set_xlim(-0.5, len(unique_params) - 0.5)
            if ax_box is not None:
                ax_box.axvline(x=vline_x, ls="--", lw=0.8, color="#333333", zorder=5)

    for ax in [ax_line_train, ax_line_val] + ([ax_box_train, ax_box_val] if show_boxplot else []):
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in [ax_line_test] + ([ax_box_test] if show_boxplot else []):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=xtick_fontsize)
        ax.set_xlabel(xlabel, labelpad=2, fontsize=10)


def _plot_distortion_panel(
    fig,
    gs_element,
    data: Dict[str, tuple],
    title_str: str,
    palette=None,
    cmap: str = "BuGn",
    colorbar_label: str = "Recon. MSE",
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    splits = ["Train", "Val", "Test"]
    n_cols = len(splits)

    gs = gs_element.subgridspec(
        2, n_cols + 1,
        width_ratios=[1] * n_cols + [0.07],
        hspace=0.10, wspace=0.08,
    )

    # Shared colour norm across all splits for the distortion row
    all_recon = np.concatenate([np.asarray(data[s][1]) for s in splits if s in data])
    norm = plt.Normalize(vmin=0, vmax=np.quantile(all_recon, 0.975))
    cax  = fig.add_subplot(gs[1, n_cols])

    sc = None
    for col, split in enumerate(splits):
        Z, recon, labs = data[split]
        Z     = Z.detach().numpy() if hasattr(Z, "detach") else np.asarray(Z)
        recon = np.asarray(recon)

        # Row 0 — label-coloured
        ax_lbl = fig.add_subplot(gs[0, col])
        sns.scatterplot(
            x=Z[:, 0], y=Z[:, 1], hue=labs, palette=palette,
            s=4, alpha=0.7, ax=ax_lbl, legend=False,
            linewidths=0, rasterized=True,
        )
        _clean_embedding_ax(ax_lbl)
        ax_lbl.set_title(split, fontsize=12, fontweight="bold", pad=3)
        if col == 0:
            ax_lbl.set_ylabel("Labels", fontsize=10, fontweight="bold", labelpad=4)

        # Row 1 — distortion heatmap
        ax_dst = fig.add_subplot(gs[1, col])
        sc = ax_dst.scatter(
            Z[:, 0], Z[:, 1], c=recon, cmap=cmap,
            norm=norm, s=4, alpha=0.7, linewidths=0, rasterized=True,
        )
        _clean_embedding_ax(ax_dst)
        if col == 0:
            ax_dst.set_ylabel("Distortion", fontsize=10, fontweight="bold", labelpad=4)

    cb = fig.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=8, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label(colorbar_label, fontsize=9, labelpad=2)
    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=0.98)


def _clean_embedding_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
