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
import argparse
import os
import re
import sys

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shap_plot_utils import continuous_feature_candidates, prepare_shap_context, top_features_by_mean_abs, translate_feature_names


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


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def detect_interaction_feature(feat_en: str, shap_values: np.ndarray, X_en: pd.DataFrame) -> str:
    try:
        # SHAP may provide either approximate_interactions or potential_interactions.
        if hasattr(shap.utils, "approximate_interactions"):
            ranked = shap.utils.approximate_interactions(feat_en, shap_values, X_en)
            if len(ranked) > 0:
                idx = int(ranked[0])
                if 0 <= idx < X_en.shape[1]:
                    return str(X_en.columns[idx])
        if hasattr(shap.utils, "potential_interactions"):
            ranked = shap.utils.potential_interactions(feat_en, shap.Explanation(values=shap_values, data=X_en.values, feature_names=X_en.columns.tolist()))
            if len(ranked) > 0:
                idx = int(ranked[0])
                if 0 <= idx < X_en.shape[1]:
                    return str(X_en.columns[idx])
    except Exception:
        pass
    return "feature value"


def main() -> None:
    configure_cjk_font()
    parser = argparse.ArgumentParser(description="Plot SHAP dependence plots for top continuous features.")
    parser.add_argument("--top-k", type=int, default=3, help="How many continuous features to plot.")
    parser.add_argument("--max-candidates", type=int, default=20, help="Search top N SHAP features then keep continuous ones.")
    parser.add_argument("--feature", action="append", default=[], help="Manually add feature name to plot. Repeatable.")
    args = parser.parse_args()

    ctx = prepare_shap_context()
    X = ctx["X"]
    shap_values = np.asarray(ctx["shap_values"])
    feature_names_en = translate_feature_names(X.columns.tolist())
    name_map = dict(zip(X.columns.tolist(), feature_names_en))
    X_en = X.copy()
    X_en.columns = feature_names_en
    output_dir = str(ctx["cfg"]["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)

    top_df = top_features_by_mean_abs(shap_values, X.columns.tolist(), top_k=args.max_candidates)
    cont_cols = set(continuous_feature_candidates(X, min_unique=10))

    selected_orig = []
    for feat in top_df["feature"]:
        if feat in cont_cols:
            selected_orig.append(feat)
        if len(selected_orig) >= args.top_k:
            break

    reverse_map = {v: k for k, v in name_map.items()}
    for feat in args.feature:
        feat_orig = feat
        if feat_orig not in X.columns and feat in reverse_map:
            feat_orig = reverse_map[feat]
        if feat_orig in X.columns and feat_orig not in selected_orig:
            selected_orig.append(feat_orig)

    selected_orig = selected_orig[: max(args.top_k, len(args.feature))]
    if not selected_orig:
        raise ValueError("No continuous feature found. Use --feature to specify manually.")

    selected_en = [name_map[f] for f in selected_orig]

    for feat_en in selected_en:
        plt.figure(figsize=(8.5, 6.5))
        shap.dependence_plot(feat_en, shap_values, X_en, interaction_index="auto", show=False)
        fig = plt.gcf()
        if len(fig.axes) >= 2:
            interaction_name = detect_interaction_feature(feat_en, shap_values, X_en)
            fig.axes[-1].set_ylabel(f"Interaction: {interaction_name}")
        plt.title(f"SHAP Dependence - {feat_en}")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"Figure_SHAP_Dependence_{safe_filename(feat_en)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_path}")

    meta_path = os.path.join(output_dir, "shap_dependence_selected_features.csv")
    pd.DataFrame({"feature": selected_en}).to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {meta_path}")
    print("Selected features:")
    print("\n".join(selected_en))


if __name__ == "__main__":
    main()
