"""
Publication-style calibration curves for isotonic-recalibrated probabilities.

Features:
- Post-hoc isotonic recalibration fitted on a natural-prevalence validation cohort
- Quantile-bin reliability curves
- 95% Wilson confidence intervals for observed probabilities
- Bootstrap 95% CI for:
    * Brier score
    * Calibration slope
- Unified axis range across panels
- Low-risk focused layout suitable for imbalanced clinical prediction tasks
- Nature-like restrained red/blue palette
"""

from __future__ import annotations

# Project path bootstrap for direct script execution from nested folders.
from pathlib import Path
import sys

for _project_root in Path(__file__).resolve().parents:
    if (_project_root / "configs").is_dir() and (_project_root / "utils").is_dir():
        for _path in (_project_root, _project_root / "configs", _project_root / "utils"):
            _path_str = str(_path)
            if _path_str not in sys.path:
                sys.path.insert(0, _path_str)
        break

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression


# =========================
# Configuration / Styling
# =========================

# Nature-like restrained palette
COLOR_BLUE = "#1F5A91"       # main curve / markers
COLOR_BLUE_LIGHT = "#9EC3E6" # histogram / fill
COLOR_BLUE_SOFT = "#5F8FBD"  # error bars / secondary marks
COLOR_RED = "#B13A3A"        # validation curve / bars
COLOR_RED_LIGHT = "#E8B3B3"  # validation histogram fill
COLOR_ORANGE = "#D58B3E"     # counterpart histogram bars
COLOR_DELTA = "#A63B32"      # bin-wise count delta curve
COLOR_GREY = "#9A9A9A"       # ideal diagonal
COLOR_GRID = "#D9D9D9"
COLOR_TEXT = "#3F3F3F"
COLOR_AXIS = "#8E8E8E"
COLOR_WHITE = "#FFFFFF"

# Unified axis max for all panels
AXIS_UPPER = 0.08

# Bootstrap repeats
N_BOOT = 1000
BOOT_SEED = 42


@dataclass
class DatasetSpec:
    panel: str
    title_prefix: str
    csv_path: Path
    target_col: str
    prob_col: str
    n_bins: int = 8


# =========================
# Utility functions
# =========================

def _safe_clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _safe_clip_probs(p)
    return np.log(p / (1.0 - p))


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1.0 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _load_dataset(spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(spec.csv_path)

    if spec.target_col not in df.columns:
        raise KeyError(f"Missing target column '{spec.target_col}' in {spec.csv_path}")
    if spec.prob_col not in df.columns:
        raise KeyError(f"Missing probability column '{spec.prob_col}' in {spec.csv_path}")

    y_true = df[spec.target_col].to_numpy(dtype=int)
    y_prob = _safe_clip_probs(df[spec.prob_col].to_numpy(dtype=float))
    return y_true, y_prob, df


def _fit_validation_isotonic_calibrator(root: Path) -> Tuple[IsotonicRegression, pd.DataFrame, Path]:
    """
    Fit isotonic calibrator on a natural-prevalence internal validation cohort.
    """
    fit_path = root / "results_flat_0313" / "val_predictions.csv"
    df_fit = pd.read_csv(fit_path)

    if "target" not in df_fit.columns or "pred_student" not in df_fit.columns:
        raise KeyError(f"Calibration fit file missing required columns in {fit_path}")

    prevalence = float(df_fit["target"].mean())
    if not (0.0 < prevalence < 0.2):
        raise ValueError(
            "Calibration fitting set prevalence looks unrealistic for this task "
            f"(target mean={prevalence:.4f}). "
            "Use a natural (non-resampled) validation set to fit isotonic calibration."
        )

    x_fit = df_fit["pred_student"].to_numpy(dtype=float)
    y_fit = df_fit["target"].to_numpy(dtype=int)

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(x_fit, y_fit)

    return calibrator, df_fit, fit_path


def _apply_isotonic_calibration(calibrator: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    p_cal = calibrator.predict(np.asarray(y_prob, dtype=float))
    return _safe_clip_probs(p_cal)


# =========================
# Calibration statistics
# =========================

def _fit_calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Fit:
        logit(P(Y=1)) = intercept + slope * logit(p)
    Returns:
        slope
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _safe_clip_probs(y_prob)

    lp = _logit(y_prob).reshape(-1, 1)

    try:
        # Equivalent logistic calibration model:
        # logit(P(Y=1)) = intercept + slope * logit(p)
        # Disable regularization to match standard calibration slope/intercept estimation.
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=42)
        model.fit(lp, y_true)
        slope = float(model.coef_[0][0])
    except Exception:
        slope = np.nan

    return slope


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    brier_vals = np.empty(n_boot, dtype=float)
    slope_vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y_true[idx]
        ps = y_prob[idx]

        brier_vals[i] = brier_score_loss(ys, ps)
        slope_vals[i] = _fit_calibration_slope(ys, ps)

    def _ci(arr: np.ndarray) -> Tuple[float, float, float]:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan, np.nan, np.nan
        return (
            float(np.mean(arr)),
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        )

    return {
        "brier": _ci(brier_vals),
        "slope": _ci(slope_vals),
    }


# =========================
# Reliability binning
# =========================

def _bin_summary_quantile(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 8,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "p": y_prob.astype(float)})

    try:
        bins = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(df["p"], bins=min(n_bins, max(3, df["p"].nunique())), duplicates="drop")

    grouped = df.groupby(bins, observed=True)

    rows = []
    for _, g in grouped:
        n = len(g)
        k = int(g["y"].sum())
        obs = float(g["y"].mean())
        pred = float(g["p"].mean())
        lo, hi = _wilson_interval(k=k, n=n)

        rows.append(
            {
                "n": n,
                "k": k,
                "pred_mean": pred,
                "obs_rate": obs,
                "obs_lo": lo,
                "obs_hi": hi,
            }
        )

    out = pd.DataFrame(rows).sort_values("pred_mean").reset_index(drop=True)
    return out


def _sample_level_smooth_trend(y_true: np.ndarray, y_prob: np.ndarray, axis_upper: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unused placeholder retained for backward compatibility."""
    grid = np.linspace(0.0, axis_upper, 240)
    return grid, np.full_like(grid, np.nan, dtype=float)


def _smooth_trend_from_bins(bin_df: pd.DataFrame, axis_upper: float) -> Tuple[np.ndarray, np.ndarray]:
    """Monotone, shape-preserving trend line from bin summaries."""
    x = bin_df["pred_mean"].to_numpy(dtype=float)
    y = bin_df["obs_rate"].to_numpy(dtype=float)
    w = bin_df["n"].to_numpy(dtype=float)

    # Weighted isotonic fit removes jagged local dips while preserving monotonicity.
    iso = IsotonicRegression(y_min=0.0, y_max=axis_upper, out_of_bounds="clip")
    y_iso = iso.fit_transform(x, y, sample_weight=w)

    # Anchor near origin to improve visual behavior at the left boundary.
    xk = np.r_[0.0, x]
    yk = np.r_[0.0, y_iso]

    # Ensure strictly increasing x for PCHIP.
    x_unique, idx = np.unique(xk, return_index=True)
    y_unique = yk[idx]
    if len(x_unique) < 2:
        return x_unique, y_unique

    grid = np.linspace(float(x_unique.min()), float(x_unique.max()), 260)
    spline = PchipInterpolator(x_unique, y_unique, extrapolate=False)
    smooth = spline(grid)
    return grid, np.clip(smooth, 0.0, axis_upper)


# =========================
# Plotting
# =========================

def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT, labelsize=10.5)


def _add_metric_box(ax, stats: Dict[str, Tuple[float, float, float]]) -> None:
    brier_mean, brier_lo, brier_hi = stats["brier"]
    slope_mean, slope_lo, slope_hi = stats["slope"]

    text = (
        f"Brier: {brier_mean:.3f} ({brier_lo:.3f}–{brier_hi:.3f})\n"
        f"Slope: {slope_mean:.2f} ({slope_lo:.2f}–{slope_hi:.2f})"
    )


def _add_metric_box_labeled(
    ax,
    stats: Dict[str, Tuple[float, float, float]],
    label: str,
    color: str,
    x: float,
    y: float,
) -> None:
    brier_mean, brier_lo, brier_hi = stats["brier"]
    slope_mean, slope_lo, slope_hi = stats["slope"]

    text = (
        f"{label}: Brier {brier_mean:.3f} ({brier_lo:.3f}-{brier_hi:.3f})\n"
        f"Slope {slope_mean:.2f} ({slope_lo:.2f}-{slope_hi:.2f})"
    )

    ax.text(
        x,
        y,
        text,
        fontsize=8.7,
        transform=ax.transAxes,
        color=COLOR_TEXT,
        bbox={
            "facecolor": COLOR_WHITE,
            "alpha": 0.90,
            "edgecolor": color,
            "boxstyle": "round,pad=0.26",
            "linewidth": 0.9,
        },
        va="bottom",
        ha="left",
    )

    ax.text(
        0.47,
        0.09,
        text,
        fontsize=9.2,
        transform=ax.transAxes,
        color=COLOR_TEXT,
        bbox={
            "facecolor": COLOR_WHITE,
            "alpha": 0.92,
            "edgecolor": "#6E6E6E",
            "boxstyle": "round,pad=0.30",
            "linewidth": 0.8,
        },
        va="bottom",
        ha="left",
    )


def _draw_single_panel(
    ax_main,
    ax_hist,
    spec: DatasetSpec,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ref_prob: np.ndarray | None = None,
    ref_label: str | None = None,
    n_boot: int = 1000,
    axis_upper: float = 0.10,
) -> None:
    bin_df = _bin_summary_quantile(y_true, y_prob, n_bins=spec.n_bins)
    stats = _bootstrap_ci(y_true, y_prob, n_boot=n_boot, seed=BOOT_SEED)

    # Ideal line
    ax_main.plot(
        [0, axis_upper],
        [0, axis_upper],
        linestyle=(0, (4, 4)),
        color=COLOR_GREY,
        linewidth=1.3,
        zorder=1,
    )

    # Error bars
    yerr = np.vstack(
        [
            (bin_df["obs_rate"] - bin_df["obs_lo"]).to_numpy(),
            (bin_df["obs_hi"] - bin_df["obs_rate"]).to_numpy(),
        ]
    )

    ax_main.errorbar(
        bin_df["pred_mean"],
        bin_df["obs_rate"],
        yerr=yerr,
        fmt="o",
        color=COLOR_BLUE,
        ecolor=COLOR_BLUE_SOFT,
        elinewidth=1.4,
        capsize=0,
        markersize=4.8,
        markerfacecolor=COLOR_BLUE,
        markeredgecolor=COLOR_BLUE,
        markeredgewidth=0.3,
        zorder=4,
    )

    # Main smooth trend from monotone bin-based interpolation.
    gx, gy = _smooth_trend_from_bins(bin_df, axis_upper=axis_upper)
    ax_main.plot(
        gx,
        gy,
        color=COLOR_BLUE,
        linewidth=2.2,
        alpha=0.93,
        zorder=3,
    )

    # Subtle connector only; no smoothing on sparse binned points.
    x_bin = bin_df["pred_mean"].to_numpy(dtype=float)
    y_bin = bin_df["obs_rate"].to_numpy(dtype=float)

    # Keep a very thin bin-connection line as secondary evidence only.
    ax_main.plot(
        x_bin,
        y_bin,
        color=COLOR_BLUE_SOFT,
        linewidth=0.8,
        alpha=0.30,
        zorder=2,
    )

    ax_main.set_xlim(0.0, axis_upper)
    ax_main.set_ylim(0.0, axis_upper)
    ax_main.set_ylabel("Observed Probability", fontsize=12.5, color=COLOR_TEXT)
    ax_main.set_title(spec.title_prefix, fontsize=14, pad=10, color=COLOR_TEXT)

    ax_main.grid(True, linestyle="-", linewidth=0.55, alpha=0.30, color=COLOR_GRID)
    _add_metric_box(ax_main, stats)

    # Histogram: side-by-side bin bars (self vs counterpart) so differences are explicit.
    bins = np.linspace(0.0, axis_upper, 51)
    counts_self, edges = np.histogram(y_prob, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = (edges[1] - edges[0])

    if ref_prob is not None and len(ref_prob) > 0:
        counts_ref, _ = np.histogram(ref_prob, bins=bins)
        bar_w = bin_w * 0.42
        ax_hist.bar(
            centers - bar_w * 0.55,
            counts_self,
            width=bar_w,
            color=COLOR_BLUE_LIGHT,
            edgecolor=COLOR_BLUE,
            linewidth=0.30,
            alpha=0.95,
            label=spec.title_prefix,
            align="center",
        )
        ax_hist.bar(
            centers + bar_w * 0.55,
            counts_ref,
            width=bar_w,
            color="#F6D7B3",
            edgecolor=COLOR_ORANGE,
            linewidth=0.30,
            alpha=0.92,
            label=ref_label if ref_label else "Counterpart",
            align="center",
        )

        # Add a delta curve on a right y-axis to explicitly expose subtle
        # bin-wise differences even when absolute histograms look very close.
        delta = counts_self - counts_ref
        ax_delta = ax_hist.twinx()
        ax_delta.axhline(0.0, color=COLOR_DELTA, linewidth=0.8, alpha=0.55)
        ax_delta.plot(
            centers,
            delta,
            color=COLOR_DELTA,
            linewidth=1.1,
            alpha=0.95,
            zorder=6,
        )
        dmax = float(np.max(np.abs(delta)))
        dlim = max(10.0, dmax * 1.15)
        ax_delta.set_ylim(-dlim, dlim)
        ax_delta.set_ylabel("Delta", fontsize=8.5, color=COLOR_DELTA)
        ax_delta.tick_params(axis="y", colors=COLOR_DELTA, labelsize=8)
        ax_delta.spines["top"].set_visible(False)
        ax_delta.spines["left"].set_visible(False)
        ax_delta.spines["right"].set_color(COLOR_DELTA)

        # Small numeric cue for how different the two histograms are.
        abs_diff = int(np.sum(np.abs(delta)))
        diff_bins = int(np.sum(delta != 0))
        ax_hist.text(
            0.01,
            0.92,
            f"Diff bins: {diff_bins} | L1: {abs_diff}",
            transform=ax_hist.transAxes,
            fontsize=7.5,
            color=COLOR_TEXT,
            ha="left",
            va="top",
            bbox={
                "facecolor": COLOR_WHITE,
                "alpha": 0.82,
                "edgecolor": "none",
                "pad": 1.8,
            },
        )
    else:
        ax_hist.bar(
            centers,
            counts_self,
            width=bin_w * 0.82,
            color=COLOR_BLUE_LIGHT,
            edgecolor=COLOR_BLUE,
            linewidth=0.30,
            alpha=0.95,
            label=spec.title_prefix,
            align="center",
        )

    ax_hist.set_xlim(0.0, axis_upper)
    ax_hist.set_ylabel("Count", fontsize=9.6, color=COLOR_TEXT)
    ax_hist.set_xlabel("Predicted Probability", fontsize=12.5, color=COLOR_TEXT)
    ax_hist.grid(True, axis="x", linestyle="-", linewidth=0.50, alpha=0.26, color=COLOR_GRID)
    if ref_prob is not None and len(ref_prob) > 0:
        ax_hist.legend(loc="upper right", fontsize=7.8, frameon=False)

    _style_axes(ax_main)
    _style_axes(ax_hist)


# =========================
# Main
# =========================

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "results_flat"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit isotonic calibrator on internal validation set (0313/val_predictions.csv)
    calibrator, _, _ = _fit_validation_isotonic_calibrator(root)

    spec_val = DatasetSpec(
        panel="A",
        title_prefix="Internal validation",
        csv_path=root / "results_flat_0313" / "val_predictions.csv",
        target_col="target",
        prob_col="pred_student",
        n_bins=8,
    )
    spec_test = DatasetSpec(
        panel="B",
        title_prefix="Internal test",
        csv_path=root / "results_flat_0313" / "final_predictions.csv",
        target_col="target",
        prob_col="pred_student",
        n_bins=8,
    )

    y_val_true, y_val_raw, _ = _load_dataset(spec_val)
    y_test_true, y_test_raw, _ = _load_dataset(spec_test)
    y_val_prob = _apply_isotonic_calibration(calibrator, y_val_raw)
    y_test_prob = _apply_isotonic_calibration(calibrator, y_test_raw)

    val_bin = _bin_summary_quantile(y_val_true, y_val_prob, n_bins=spec_val.n_bins)
    test_bin = _bin_summary_quantile(y_test_true, y_test_prob, n_bins=spec_test.n_bins)
    val_stats = _bootstrap_ci(y_val_true, y_val_prob, n_boot=N_BOOT, seed=BOOT_SEED)
    test_stats = _bootstrap_ci(y_test_true, y_test_prob, n_boot=N_BOOT, seed=BOOT_SEED)

    fig = plt.figure(figsize=(7.4, 13.4), dpi=400)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[5.2, 5.2, 1.9], hspace=0.16)
    ax_val = fig.add_subplot(gs[0])
    ax_test = fig.add_subplot(gs[1], sharex=ax_val, sharey=ax_val)
    ax_hist = fig.add_subplot(gs[2], sharex=ax_val)

    def _draw_curve_panel(
        ax,
        bin_df: pd.DataFrame,
        stats: Dict[str, Tuple[float, float, float]],
        color_main: str,
        color_soft: str,
        title: str,
        panel: str,
    ) -> None:
        ax.plot(
            [0, AXIS_UPPER],
            [0, AXIS_UPPER],
            linestyle=(0, (4, 4)),
            color=COLOR_GREY,
            linewidth=1.3,
            zorder=1,
        )

        yerr = np.vstack(
            [
                (bin_df["obs_rate"] - bin_df["obs_lo"]).to_numpy(),
                (bin_df["obs_hi"] - bin_df["obs_rate"]).to_numpy(),
            ]
        )
        ax.errorbar(
            bin_df["pred_mean"],
            bin_df["obs_rate"],
            yerr=yerr,
            fmt="o",
            color=color_main,
            ecolor=color_soft,
            elinewidth=1.35,
            capsize=0,
            markersize=4.8,
            markerfacecolor=color_main,
            markeredgecolor=color_main,
            markeredgewidth=0.3,
            zorder=4,
        )

        gx, gy = _smooth_trend_from_bins(bin_df, axis_upper=AXIS_UPPER)
        ax.plot(gx, gy, color=color_main, linewidth=2.2, alpha=0.95, zorder=3)
        ax.plot(
            bin_df["pred_mean"].to_numpy(dtype=float),
            bin_df["obs_rate"].to_numpy(dtype=float),
            color=color_soft,
            linewidth=0.8,
            alpha=0.30,
            zorder=2,
        )

        ax.set_xlim(0.0, AXIS_UPPER)
        ax.set_ylim(0.0, AXIS_UPPER)
        ax.set_ylabel("Observed Probability", fontsize=12.3, color=COLOR_TEXT)
        ax.set_title(title, fontsize=14, pad=8, color=COLOR_TEXT)
        ax.grid(True, linestyle="-", linewidth=0.55, alpha=0.30, color=COLOR_GRID)
        _add_metric_box(ax, stats)
        ax.text(
            -0.12,
            1.02,
            panel,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            color=COLOR_TEXT,
        )

    _draw_curve_panel(ax_val, val_bin, val_stats, COLOR_RED, COLOR_RED_LIGHT, "Internal validation", "A")
    _draw_curve_panel(ax_test, test_bin, test_stats, COLOR_BLUE, COLOR_BLUE_SOFT, "Internal test", "B")
    plt.setp(ax_val.get_xticklabels(), visible=False)

    # Shared difference/comparison histogram (appears once).
    bins = np.linspace(0.0, AXIS_UPPER, 51)
    counts_val, edges = np.histogram(y_val_prob, bins=bins)
    counts_test, _ = np.histogram(y_test_prob, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = (edges[1] - edges[0])
    bar_w = bin_w * 0.42

    ax_hist.bar(
        centers - bar_w * 0.55,
        counts_val,
        width=bar_w,
        color=COLOR_RED_LIGHT,
        edgecolor=COLOR_RED,
        linewidth=0.30,
        alpha=0.95,
        label="Validation",
        align="center",
    )
    ax_hist.bar(
        centers + bar_w * 0.55,
        counts_test,
        width=bar_w,
        color=COLOR_BLUE_LIGHT,
        edgecolor=COLOR_BLUE,
        linewidth=0.30,
        alpha=0.95,
        label="Test",
        align="center",
    )

    ax_hist.set_xlim(0.0, AXIS_UPPER)
    ax_hist.set_ylabel("Count", fontsize=9.8, color=COLOR_TEXT)
    ax_hist.set_xlabel("Predicted Probability", fontsize=12.5, color=COLOR_TEXT)
    ax_hist.set_title("Difference histogram (single panel)", fontsize=12.0, pad=6, color=COLOR_TEXT)
    ax_hist.grid(True, axis="x", linestyle="-", linewidth=0.50, alpha=0.26, color=COLOR_GRID)
    ax_hist.legend(loc="upper right", fontsize=8.6, frameon=False)
    ax_hist.text(
        -0.12,
        1.02,
        "C",
        transform=ax_hist.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLOR_TEXT,
    )

    _style_axes(ax_val)
    _style_axes(ax_test)
    _style_axes(ax_hist)

    out_png = output_dir / "Figure_Calibration_Internal_Validation_Test_NatureStyle.png"
    out_pdf = output_dir / "Figure_Calibration_Internal_Validation_Test_NatureStyle.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()