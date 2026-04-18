import argparse
import os
import sys

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shap_plot_utils import prepare_shap_context, top_features_by_mean_abs, translate_feature_names


def configure_cjk_font() -> None:
    # Prefer common Chinese fonts on Windows to avoid square glyphs.
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def clustered_order_for_features(X: pd.DataFrame, features: list[str]) -> tuple[list[str], bool]:
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
    except Exception:
        return features, False

    corr = X[features].corr().fillna(0.0)
    dist = 1.0 - np.abs(corr.values)
    tri_upper = dist[np.triu_indices_from(dist, k=1)]
    if tri_upper.size == 0:
        return features, True

    Z = linkage(tri_upper, method="average")
    order = leaves_list(Z)
    return [features[i] for i in order], True


def draw_plain_bar(df_top: pd.DataFrame, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(df_top))
    ax.barh(y, df_top["mean_abs_shap"], color="#2B6CB0", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df_top["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_clustered_bar(X: pd.DataFrame, df_top: pd.DataFrame, out_path: str, title: str) -> None:
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
    except Exception:
        draw_plain_bar(df_top, out_path, title + " (SciPy missing, fallback plain bar)")
        return

    features = df_top["feature"].tolist()
    ordered_features, _ = clustered_order_for_features(X, features)
    ordered_df = df_top.set_index("feature").loc[ordered_features].reset_index()

    corr = X[ordered_features].corr().fillna(0.0)
    dist = 1.0 - np.abs(corr.values)
    tri_upper = dist[np.triu_indices_from(dist, k=1)]
    if tri_upper.size == 0:
        draw_plain_bar(ordered_df, out_path, title)
        return

    Z = linkage(tri_upper, method="average")

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.95])
    ax_tree = fig.add_subplot(gs[0, 0])
    dendrogram(Z, labels=ordered_features, orientation="left", ax=ax_tree, color_threshold=0)
    ax_tree.set_xticks([])
    ax_tree.set_title("Feature Correlation Tree")

    ax_bar = fig.add_subplot(gs[0, 1])
    y = np.arange(len(ordered_df))
    ax_bar.barh(y, ordered_df["mean_abs_shap"], color="#1A936F", alpha=0.9)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(ordered_df["feature"])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean |SHAP value|")
    ax_bar.set_title(title)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_cjk_font()
    parser = argparse.ArgumentParser(description="Plot Mean |SHAP| bar chart.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of top features to display.")
    parser.add_argument("--clustered", action="store_true", help="Use clustered tree + bar layout.")
    parser.add_argument("--output", type=str, default="", help="Output image path.")
    args = parser.parse_args()

    ctx = prepare_shap_context()
    X = ctx["X"]
    shap_values = ctx["shap_values"]
    cfg = ctx["cfg"]

    feature_names = X.columns.tolist()
    feature_names_en = translate_feature_names(feature_names)
    X_en = X.copy()
    X_en.columns = feature_names_en
    df_top = top_features_by_mean_abs(shap_values, feature_names_en, top_k=args.top_k)

    output_dir = str(cfg["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)
    out_name = "Figure_SHAP_MeanAbs_Bar_Clustered.png" if args.clustered else "Figure_SHAP_MeanAbs_Bar.png"
    out_path = args.output.strip() if args.output else os.path.join(output_dir, out_name)

    title = "Global Feature Importance (Mean |SHAP|)"
    if args.clustered:
        draw_clustered_bar(X_en, df_top, out_path, title)
    else:
        draw_plain_bar(df_top, out_path, title)

    csv_path = os.path.join(output_dir, "shap_mean_abs_top_features.csv")
    df_top.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[Saved] {out_path}")
    print(f"[Saved] {csv_path}")
    print("Top 5 features:")
    print(df_top.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
