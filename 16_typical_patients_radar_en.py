import argparse
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw English radar chart for two typical patients (percentile-normalized)."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=r"d:\breast_cancer\results_flat_0313",
        help="Directory containing processed_test.csv, final_predictions.csv and typical_patients_selected.csv",
    )
    parser.add_argument(
        "--selection-csv",
        type=str,
        default="typical_patients_selected.csv",
        help="CSV with selected case rows including type and idx columns",
    )
    parser.add_argument(
        "--candidate-csv",
        type=str,
        default="shap_local_case_candidates.csv",
        help="Candidate case CSV used for auto-selecting a high-separation pair",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=["max_gap_candidates", "csv"],
        default="max_gap_candidates",
        help="Pair selection method: 'max_gap_candidates' (default) or preselected 'csv'",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        default="processed_test.csv",
        help="Feature table CSV",
    )
    parser.add_argument(
        "--prediction-csv",
        type=str,
        default="final_predictions.csv",
        help="Prediction table CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="typical_patients_radar_continuous_percentile_en.png",
        help="Output figure filename",
    )
    parser.add_argument(
        "--shap-top-csv",
        type=str,
        default="shap_mean_abs_top_features.csv",
        help="CSV containing SHAP feature ranking with a 'feature' column",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of SHAP top features to plot",
    )
    parser.add_argument(
        "--contrast-gamma",
        type=float,
        default=0.75,
        help="Contrast gamma (<1 exaggerates differences, =1 no change).",
    )
    parser.add_argument(
        "--bubble-output",
        type=str,
        default="typical_patients_bubble_top8_en.png",
        help="Output filename for bubble comparison chart",
    )
    parser.add_argument(
        "--lifestyle-top-k",
        type=int,
        default=6,
        help="Number of lifestyle-intervention features with strongest benign/malignant contrast",
    )
    parser.add_argument(
        "--lifestyle-output",
        type=str,
        default="typical_patients_lifestyle_contrast_en.png",
        help="Output filename for lifestyle-intervention contrast chart",
    )
    return parser


def _strip_duplicate_suffix(name: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", name).strip()


def pick_top_features_from_shap(
    full_columns: list[str],
    shap_path: Path,
    translate_feature_name,
    exact_feature_map: dict[str, str],
    top_k: int,
) -> tuple[list[str], list[str]]:
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP top feature file not found: {shap_path}")

    df_shap = pd.read_csv(shap_path)
    if "feature" not in df_shap.columns:
        raise ValueError(f"Missing 'feature' column in: {shap_path}")

    # User-requested exclusions from previous custom set.
    excluded_cn = {"每天食用蔬菜量", "每天食用水果量"}
    excluded_en = {translate_feature_name(cn) for cn in excluded_cn}

    # Build reverse map for English display names -> source column name.
    en_to_cn: dict[str, str] = {}
    for cn, en in exact_feature_map.items():
        if en not in en_to_cn:
            en_to_cn[en] = cn

    translated_lookup = {translate_feature_name(col): col for col in full_columns}

    selected_cn: list[str] = []
    selected_en: list[str] = []

    for raw_name in df_shap["feature"].dropna().astype(str).tolist():
        name = _strip_duplicate_suffix(raw_name)

        if name in excluded_cn or name in excluded_en:
            continue

        resolved_cn = None
        if name in full_columns:
            resolved_cn = name
        elif name in en_to_cn and en_to_cn[name] in full_columns:
            resolved_cn = en_to_cn[name]
        elif name in translated_lookup:
            resolved_cn = translated_lookup[name]

        if resolved_cn is None:
            continue

        resolved_en = translate_feature_name(resolved_cn)
        if resolved_cn in selected_cn:
            continue

        selected_cn.append(resolved_cn)
        selected_en.append(resolved_en)

        if len(selected_cn) >= top_k:
            break

    if len(selected_cn) < top_k:
        raise ValueError(
            f"Only found {len(selected_cn)} SHAP features after filtering; need {top_k}."
        )

    return selected_cn, selected_en


def prettify_label(label: str, width: int = 16) -> str:
    return textwrap.fill(label, width=width, break_long_words=False)


def choose_indices(
    base_dir: Path,
    selected: pd.DataFrame,
    selection_strategy: str,
    candidate_csv: str,
) -> tuple[int, int]:
    if selection_strategy == "csv":
        benign_idx = int(selected.loc[selected["type"] == "typical_benign", "idx"].iloc[0])
        malignant_idx = int(selected.loc[selected["type"] == "typical_malignant", "idx"].iloc[0])
        return benign_idx, malignant_idx

    cand_path = base_dir / candidate_csv
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {cand_path}")

    cands = pd.read_csv(cand_path)
    need_cols = {"idx", "group", "pred"}
    if not need_cols.issubset(set(cands.columns)):
        raise ValueError(f"Candidate CSV must include columns: {sorted(need_cols)}")

    benign_pool = cands[cands["group"] == "benign"].sort_values("pred", ascending=True)
    malignant_pool = cands[cands["group"] == "malignant"].sort_values("pred", ascending=False)
    if benign_pool.empty or malignant_pool.empty:
        raise ValueError("Candidate CSV does not contain both benign and malignant groups.")

    benign_idx = int(benign_pool.iloc[0]["idx"])
    malignant_idx = int(malignant_pool.iloc[0]["idx"])
    return benign_idx, malignant_idx


def enhance_contrast(values: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("contrast-gamma must be > 0")
    centered = 2.0 * np.asarray(values, dtype=float) - 1.0
    stretched = np.sign(centered) * (np.abs(centered) ** gamma)
    return np.clip((stretched + 1.0) / 2.0, 0.0, 1.0)


def pick_lifestyle_features_by_contrast(
    feat_df: pd.DataFrame,
    benign_idx: int,
    malignant_idx: int,
    translate_feature_name,
    top_k: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("lifestyle-top-k must be > 0")

    keywords_cn = [
        "吸烟",
        "饮酒",
        "运动",
        "锻炼",
        "步行",
        "睡眠",
        "作息",
        "蔬菜",
        "水果",
        "饮食",
        "盐",
        "油",
        "体重",
        "肥胖",
        "腰围",
        "压力",
    ]
    keywords_en = [
        "smok",
        "alcohol",
        "drink",
        "exercise",
        "activity",
        "walk",
        "sleep",
        "diet",
        "vegetable",
        "fruit",
        "salt",
        "oil",
        "weight",
        "obes",
        "bmi",
        "waist",
        "stress",
    ]
    category_tokens_cn = ["分类", "分组", "等级", "类别"]
    category_tokens_en = ["category", "class", "group", "grade", "level", "type"]

    lifestyle_cols: list[str] = []
    for col in feat_df.columns:
        if not pd.api.types.is_numeric_dtype(feat_df[col]):
            continue

        col_str = str(col)
        en_name = str(translate_feature_name(col_str))
        col_lower = col_str.lower()
        en_lower = en_name.lower()

        # Prefer primary modifiable indicators over derived category bins.
        is_category_like = any(k in col_str for k in category_tokens_cn) or any(
            k in col_lower or k in en_lower for k in category_tokens_en
        )
        if is_category_like:
            continue

        hit_cn = any(k in col_str for k in keywords_cn)
        hit_en = any(k in col_lower or k in en_lower for k in keywords_en)
        if hit_cn or hit_en:
            lifestyle_cols.append(col_str)

    if not lifestyle_cols:
        raise ValueError("No lifestyle-intervention features found from current indicators.")

    ranks = feat_df[lifestyle_cols].rank(method="average", pct=True)
    benign_vals = ranks.iloc[benign_idx].to_numpy(dtype=float)
    malignant_vals = ranks.iloc[malignant_idx].to_numpy(dtype=float)
    diffs = np.abs(benign_vals - malignant_vals)

    order = np.argsort(-diffs)
    selected_order = order[: min(top_k, len(lifestyle_cols))]

    selected_cn = [lifestyle_cols[i] for i in selected_order]
    selected_en = [translate_feature_name(cn) for cn in selected_cn]
    selected_benign = benign_vals[selected_order]
    selected_malignant = malignant_vals[selected_order]

    return selected_cn, selected_en, selected_benign, selected_malignant


def draw_bubble_chart(
    features_en: list[str],
    benign_vals: np.ndarray,
    malignant_vals: np.ndarray,
    out_path: Path,
) -> None:
    y = np.arange(len(features_en))
    fig, ax = plt.subplots(figsize=(11, 7.8), facecolor="white")

    for yi, xb, xm in zip(y, benign_vals, malignant_vals):
        ax.plot([xb, xm], [yi, yi], color="#A0A7B3", linewidth=1.6, alpha=0.75, zorder=1)

    size_b = 180 + 1050 * (benign_vals ** 1.4)
    size_m = 180 + 1050 * (malignant_vals ** 1.4)

    ax.scatter(
        benign_vals,
        y,
        s=size_b,
        c="#1f77b4",
        alpha=0.72,
        edgecolors="white",
        linewidths=1.2,
        label="Typical Benign",
        zorder=3,
    )
    ax.scatter(
        malignant_vals,
        y,
        s=size_m,
        c="#d62728",
        alpha=0.72,
        edgecolors="white",
        linewidths=1.2,
        label="Typical Malignant",
        zorder=3,
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Percentile-Normalized Feature Value", fontsize=12, fontweight="bold")
    ax.set_ylabel("SHAP Top Features", fontsize=12, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([prettify_label(x, width=26) for x in features_en], fontsize=11, fontweight="bold")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)], fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        prop={"size": 11, "weight": "bold"},
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_bubble_radar(
    features_en: list[str],
    benign_vals: np.ndarray,
    malignant_vals: np.ndarray,
    out_path: Path,
) -> None:
    n = len(features_en)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    benign_closed = np.concatenate([benign_vals, [benign_vals[0]]])
    malignant_closed = np.concatenate([malignant_vals, [malignant_vals[0]]])

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("white")

    # Keep light outlines for shape continuity.
    ax.plot(angles_closed, benign_closed, color="#1f77b4", linewidth=2.0, alpha=0.7, zorder=2)
    ax.plot(angles_closed, malignant_closed, color="#d62728", linewidth=2.0, alpha=0.7, zorder=2)

    # Bubble area increases with feature value.
    size_b = 70 + 1650 * (benign_vals ** 1.6)
    size_m = 70 + 1650 * (malignant_vals ** 1.6)

    ax.scatter(
        angles,
        benign_vals,
        s=size_b,
        c="#1f77b4",
        alpha=0.52,
        edgecolors="white",
        linewidths=1.2,
        label="Typical Benign",
        zorder=4,
    )
    ax.scatter(
        angles,
        malignant_vals,
        s=size_m,
        c="#d62728",
        alpha=0.52,
        edgecolors="white",
        linewidths=1.2,
        label="Typical Malignant",
        zorder=4,
    )

    ax.set_xticks(angles)
    label_text = [prettify_label(x, width=16) for x in features_en]
    tick_labels = ax.set_xticklabels(label_text, fontsize=11, fontweight="bold")
    for txt, angle in zip(tick_labels, angles):
        deg = np.degrees(angle)
        if 90 < deg < 270:
            txt.set_horizontalalignment("right")
        elif deg == 90 or deg == 270:
            txt.set_horizontalalignment("center")
        else:
            txt.set_horizontalalignment("left")
        txt.set_verticalalignment("center")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=10, fontweight="bold")
    ax.grid(alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        prop={"size": 12, "weight": "bold"},
        ncol=2,
    )

    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.92])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir)

    project_dir = Path(__file__).resolve().parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    from shap_plot_utils import EXACT_FEATURE_MAP, translate_feature_name  # local project utility

    feature_path = base_dir / args.feature_csv
    pred_path = base_dir / args.prediction_csv
    selection_path = base_dir / args.selection_csv

    feat = pd.read_csv(feature_path, dtype={"cohortid": "string"})
    pred = pd.read_csv(pred_path)
    selected = pd.read_csv(selection_path, dtype={"cohortid": "string"})

    full = pd.concat([feat.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

    benign_idx, malignant_idx = choose_indices(
        base_dir=base_dir,
        selected=selected,
        selection_strategy=args.selection_strategy,
        candidate_csv=args.candidate_csv,
    )

    shap_top_path = base_dir / args.shap_top_csv
    features_cn, features_en = pick_top_features_from_shap(
        full_columns=full.columns.tolist(),
        shap_path=shap_top_path,
        translate_feature_name=translate_feature_name,
        exact_feature_map=EXACT_FEATURE_MAP,
        top_k=args.top_k,
    )

    missing = [c for c in features_cn if c not in full.columns]
    if missing:
        raise ValueError(f"Missing features in input table: {missing}")

    # Percentile normalization makes mixed-scale features directly comparable.
    ranks = full[features_cn].rank(method="average", pct=True)
    benign_vals = ranks.iloc[benign_idx].to_numpy(dtype=float)
    malignant_vals = ranks.iloc[malignant_idx].to_numpy(dtype=float)

    # Emphasize visual separation while preserving ordering.
    benign_vals = enhance_contrast(benign_vals, gamma=args.contrast_gamma)
    malignant_vals = enhance_contrast(malignant_vals, gamma=args.contrast_gamma)

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    out_path = base_dir / args.output
    bubble_path = base_dir / args.bubble_output
    lifestyle_path = base_dir / args.lifestyle_output

    lifestyle_cn, lifestyle_en, lifestyle_benign_vals, lifestyle_malignant_vals = pick_lifestyle_features_by_contrast(
        feat_df=feat,
        benign_idx=benign_idx,
        malignant_idx=malignant_idx,
        translate_feature_name=translate_feature_name,
        top_k=args.lifestyle_top_k,
    )
    lifestyle_benign_vals = enhance_contrast(lifestyle_benign_vals, gamma=args.contrast_gamma)
    lifestyle_malignant_vals = enhance_contrast(lifestyle_malignant_vals, gamma=args.contrast_gamma)

    draw_bubble_radar(
        features_en=features_en,
        benign_vals=benign_vals,
        malignant_vals=malignant_vals,
        out_path=out_path,
    )

    draw_bubble_chart(
        features_en=features_en,
        benign_vals=benign_vals,
        malignant_vals=malignant_vals,
        out_path=bubble_path,
    )

    draw_bubble_radar(
        features_en=lifestyle_en,
        benign_vals=lifestyle_benign_vals,
        malignant_vals=lifestyle_malignant_vals,
        out_path=lifestyle_path,
    )

    # Persist current selection for reproducibility and downstream scripts.
    out_selected = pd.DataFrame(
        [
            {"source": args.selection_strategy, "type": "typical_benign", "idx": benign_idx},
            {"source": args.selection_strategy, "type": "typical_malignant", "idx": malignant_idx},
        ]
    )
    out_selected.to_csv(selection_path, index=False, encoding="utf-8-sig")

    print(f"[Saved] {out_path}")
    print(f"[Saved] {bubble_path}")
    print(f"[Saved] {lifestyle_path}")
    print(f"[Saved] {selection_path}")
    print(f"[Info] Typical benign idx = {benign_idx}")
    print(f"[Info] Typical malignant idx = {malignant_idx}")
    print(f"[Info] SHAP top features used (CN): {features_cn}")
    print(f"[Info] SHAP top features used (EN): {features_en}")
    print(f"[Info] Lifestyle contrast features used (CN): {lifestyle_cn}")
    print(f"[Info] Lifestyle contrast features used (EN): {lifestyle_en}")


if __name__ == "__main__":
    main()
