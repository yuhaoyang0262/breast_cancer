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
import sys

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shap_plot_utils import pick_local_case_indices, prepare_shap_context, translate_feature_names


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


def save_waterfall(
    shap_values: np.ndarray,
    X,
    feature_names_en,
    explainer,
    idx: int,
    out_path: str,
    title: str,
    output_unit: str,
    max_display: int = 15,
) -> None:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=float(base_value),
        data=X.iloc[idx].to_numpy(),
        feature_names=feature_names_en,
    )

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    ax = plt.gca()
    if output_unit == "probability":
        ax.set_xlabel("Model output (probability)")
    else:
        ax.set_xlabel("Model output (log-odds)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def pick_consistent_case_indices(
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    surrogate_prob: np.ndarray,
    threshold: float = 0.5,
    penalty: float = 2.0,
) -> tuple[int, int]:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob)
    surrogate_prob = np.asarray(surrogate_prob)
    pred_label = (pred_prob >= threshold).astype(int)
    diff = np.abs(pred_prob - surrogate_prob)

    malignant_correct = np.where((y_true == 1) & (pred_label == 1))[0]
    benign_correct = np.where((y_true == 0) & (pred_label == 0))[0]

    if len(malignant_correct) > 0:
        malignant_score = pred_prob[malignant_correct] - penalty * diff[malignant_correct]
        malignant_idx = int(malignant_correct[np.argmax(malignant_score)])
    else:
        malignant_idx = int(np.argmax(pred_prob - penalty * diff))

    if len(benign_correct) > 0:
        benign_score = -(pred_prob[benign_correct] + penalty * diff[benign_correct])
        benign_idx = int(benign_correct[np.argmax(benign_score)])
    else:
        benign_idx = int(np.argmin(pred_prob + penalty * diff))

    return malignant_idx, benign_idx


def build_case_candidates(
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    surrogate_prob: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob)
    surrogate_prob = np.asarray(surrogate_prob)
    pred_label = (pred_prob >= threshold).astype(int)
    abs_diff = np.abs(pred_prob - surrogate_prob)

    df = pd.DataFrame(
        {
            "idx": np.arange(len(y_true), dtype=int),
            "target": y_true,
            "pred": pred_prob,
            "surrogate_pred": surrogate_prob,
            "abs_diff": abs_diff,
            "correct": (pred_label == y_true).astype(int),
        }
    )

    malignant = df[(df["target"] == 1) & (df["correct"] == 1)].sort_values(["abs_diff", "pred"], ascending=[True, False])
    benign = df[(df["target"] == 0) & (df["correct"] == 1)].sort_values(["abs_diff", "pred"], ascending=[True, True])
    return malignant.reset_index(drop=True), benign.reset_index(drop=True)


def choose_from_candidates(df: pd.DataFrame, max_abs_diff: float) -> int:
    if len(df) == 0:
        return -1
    within = df[df["abs_diff"] <= max_abs_diff]
    if len(within) > 0:
        return int(within.iloc[0]["idx"])
    return int(df.iloc[0]["idx"])


def infer_output_unit(base_value: float) -> str:
    # SHAP waterfall is additive in the explainer output space.
    # For a regression surrogate fitted on probabilities, this should be probability.
    if 0.0 <= float(base_value) <= 1.0:
        return "probability"
    return "log-odds"


def main() -> None:
    configure_cjk_font()
    parser = argparse.ArgumentParser(description="Plot local SHAP waterfall plots for representative cases.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold used for selecting correct cases.")
    parser.add_argument("--max-display", type=int, default=15, help="Max features displayed in each waterfall plot.")
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=0.03,
        help="Preferred max |orig-surrogate| for auto-selected representative cases.",
    )
    parser.add_argument("--malignant-idx", type=int, default=-1, help="Manually set malignant case index.")
    parser.add_argument("--benign-idx", type=int, default=-1, help="Manually set benign case index.")
    args = parser.parse_args()

    ctx = prepare_shap_context()
    X = ctx["X"]
    y = np.asarray(ctx["y"]).astype(int)
    pred = np.asarray(ctx["pred"])
    surrogate_pred = np.asarray(ctx["surrogate_pred"])
    shap_values = np.asarray(ctx["shap_values"])
    feature_names_en = translate_feature_names(X.columns.tolist())
    explainer = ctx["explainer"]
    fidelity_r2 = float(ctx["fidelity_r2"])
    fidelity_mae = float(ctx["fidelity_mae"])
    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]
    output_unit = infer_output_unit(float(base_value))
    output_dir = str(ctx["cfg"]["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)

    malignant_candidates, benign_candidates = build_case_candidates(y, pred, surrogate_pred, threshold=args.threshold)
    candidates_path = os.path.join(output_dir, "shap_local_case_candidates.csv")
    pd.concat(
        [
            malignant_candidates.assign(group="malignant"),
            benign_candidates.assign(group="benign"),
        ],
        axis=0,
        ignore_index=True,
    ).to_csv(candidates_path, index=False, encoding="utf-8-sig")

    auto_m_idx = choose_from_candidates(malignant_candidates, max_abs_diff=args.max_abs_diff)
    auto_b_idx = choose_from_candidates(benign_candidates, max_abs_diff=args.max_abs_diff)
    if auto_m_idx < 0 or auto_b_idx < 0:
        auto_m_idx, auto_b_idx = pick_consistent_case_indices(y, pred, surrogate_pred, threshold=args.threshold)

    malignant_idx = args.malignant_idx if args.malignant_idx >= 0 else auto_m_idx
    benign_idx = args.benign_idx if args.benign_idx >= 0 else auto_b_idx

    if args.malignant_idx < 0 and args.benign_idx < 0:
        fallback_m, fallback_b = pick_local_case_indices(y, pred, threshold=args.threshold)
        # Fallback if consistency selector unexpectedly yields same class/sample.
        if malignant_idx == benign_idx:
            malignant_idx, benign_idx = fallback_m, fallback_b

    if malignant_idx >= len(X) or benign_idx >= len(X):
        raise IndexError("Selected case index out of range.")

    out_malignant = os.path.join(output_dir, "Figure_SHAP_Waterfall_Malignant.png")
    out_benign = os.path.join(output_dir, "Figure_SHAP_Waterfall_Benign.png")

    save_waterfall(
        shap_values,
        X,
        feature_names_en,
        explainer,
        malignant_idx,
        out_malignant,
        title=(
            f"Local SHAP Waterfall - Malignant Case (idx={malignant_idx}, "
            f"orig={pred[malignant_idx]:.3f}, surrogate={surrogate_pred[malignant_idx]:.3f}, unit={output_unit})"
        ),
        output_unit=output_unit,
        max_display=args.max_display,
    )
    save_waterfall(
        shap_values,
        X,
        feature_names_en,
        explainer,
        benign_idx,
        out_benign,
        title=(
            f"Local SHAP Waterfall - Benign Case (idx={benign_idx}, "
            f"orig={pred[benign_idx]:.3f}, surrogate={surrogate_pred[benign_idx]:.3f}, unit={output_unit})"
        ),
        output_unit=output_unit,
        max_display=args.max_display,
    )

    print(f"[Saved] {out_malignant}")
    print(f"[Saved] {out_benign}")
    print(f"[Saved] {candidates_path}")
    print(f"Surrogate fidelity: R2={fidelity_r2:.4f}, MAE={fidelity_mae:.4f}")
    print(f"Waterfall output unit: {output_unit} (base={float(base_value):.4f})")
    print(f"Selection rule: prefer |orig-surrogate| <= {args.max_abs_diff:.3f} among correctly predicted cases")
    print(
        f"Malignant case: idx={malignant_idx}, target={y[malignant_idx]}, "
        f"orig={pred[malignant_idx]:.4f}, surrogate={surrogate_pred[malignant_idx]:.4f}, "
        f"abs_diff={abs(pred[malignant_idx]-surrogate_pred[malignant_idx]):.4f}"
    )
    print(
        f"Benign case: idx={benign_idx}, target={y[benign_idx]}, "
        f"orig={pred[benign_idx]:.4f}, surrogate={surrogate_pred[benign_idx]:.4f}, "
        f"abs_diff={abs(pred[benign_idx]-surrogate_pred[benign_idx]):.4f}"
    )


if __name__ == "__main__":
    main()
