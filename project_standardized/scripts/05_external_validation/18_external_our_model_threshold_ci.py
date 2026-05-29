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
import os
import numpy as np
import pandas as pd


SEED = 42
BOOTSTRAP_N = 1000
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.05
TARGET_SENSITIVITY = 0.80


DATASETS = [
    {
        "tag": "20260313_2",
        "pred_path": r"d:\\breast_cancer\\results_flat_20260313_2\\external_model_predictions_20260313_2.csv",
    },
    {
        "tag": "20260321",
        "pred_path": r"d:\\breast_cancer\\results_flat_20260321\\external_model_predictions_20260321.csv",
    },
]


def clinical_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return sens, spec, ppv, npv


def bootstrap_metrics_ci(y_true, y_prob, threshold, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    sens_list, spec_list, ppv_list, npv_list = [], [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        s, sp, p, n_ = clinical_metrics(y_b, p_b, threshold)
        sens_list.append(s)
        spec_list.append(sp)
        ppv_list.append(p)
        npv_list.append(n_)

    def _ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    return _ci(sens_list), _ci(spec_list), _ci(ppv_list), _ci(npv_list)


def build_threshold_grid(min_th=0.05, max_th=0.95, step=0.05):
    grid = np.arange(min_th, max_th + step, step)
    return np.clip(np.round(grid, 6), 0.0, 1.0)


def select_threshold_non_youden(df_metrics, target_sens=0.80):
    candidates = df_metrics[df_metrics["Sensitivity"] >= target_sens]
    if len(candidates) > 0:
        best = candidates.sort_values(["Specificity", "Threshold"], ascending=[False, False]).iloc[0]
        return float(best["Threshold"]), f"MaxSpecificity@Sensitivity>={target_sens:.2f}"

    best = df_metrics.sort_values(["Sensitivity", "Specificity", "Threshold"], ascending=[False, False, False]).iloc[0]
    return float(best["Threshold"]), "FallbackMaxSensitivity"


def format_ci(v, lo, hi):
    return f"{v:.4f} ({lo:.4f}-{hi:.4f})"


def run_one_dataset(tag, pred_path, thresholds, target_sens, n_boot, seed):
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    df = pd.read_csv(pred_path)
    if "target" not in df.columns:
        raise ValueError(f"'target' column is missing: {pred_path}")
    if "Our Proposed Model" not in df.columns:
        raise ValueError(f"'Our Proposed Model' column is missing: {pred_path}")

    y_true = df["target"].astype(int).values
    y_prob = df["Our Proposed Model"].astype(float).values

    rows = []
    for th in thresholds:
        sens, spec, ppv, npv = clinical_metrics(y_true, y_prob, th)
        (sens_lo, sens_hi), (spec_lo, spec_hi), (ppv_lo, ppv_hi), (npv_lo, npv_hi) = bootstrap_metrics_ci(
            y_true, y_prob, threshold=th, n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "ExternalTag": tag,
                "Model": "Our Proposed Model",
                "N": int(len(y_true)),
                "Prevalence": float(np.mean(y_true)),
                "Threshold": float(th),
                "Sensitivity": float(sens),
                "Sensitivity_CI_Low": sens_lo,
                "Sensitivity_CI_High": sens_hi,
                "Specificity": float(spec),
                "Specificity_CI_Low": spec_lo,
                "Specificity_CI_High": spec_hi,
                "PPV": float(ppv),
                "PPV_CI_Low": ppv_lo,
                "PPV_CI_High": ppv_hi,
                "NPV": float(npv),
                "NPV_CI_Low": npv_lo,
                "NPV_CI_High": npv_hi,
                "Sensitivity (95% CI)": format_ci(sens, sens_lo, sens_hi),
                "Specificity (95% CI)": format_ci(spec, spec_lo, spec_hi),
                "PPV (95% CI)": format_ci(ppv, ppv_lo, ppv_hi),
                "NPV (95% CI)": format_ci(npv, npv_lo, npv_hi),
            }
        )

    df_metrics = pd.DataFrame(rows)
    selected_th, selection_rule = select_threshold_non_youden(df_metrics, target_sens=target_sens)
    df_metrics["ThresholdSelectionRule"] = selection_rule
    df_metrics["Is_Selected"] = (np.isclose(df_metrics["Threshold"], selected_th)).astype(int)

    return df_metrics


def main():
    thresholds = build_threshold_grid(
        min_th=THRESHOLD_MIN,
        max_th=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )

    all_rows = []
    for cfg in DATASETS:
        df_one = run_one_dataset(
            tag=cfg["tag"],
            pred_path=cfg["pred_path"],
            thresholds=thresholds,
            target_sens=TARGET_SENSITIVITY,
            n_boot=BOOTSTRAP_N,
            seed=SEED,
        )
        all_rows.append(df_one)

    df_out = pd.concat(all_rows, axis=0, ignore_index=True)

    out_path = r"d:\\breast_cancer\\results\\external_our_model_threshold_metrics_with_ci.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"[OK] Saved: {out_path}")
    print("[OK] Selected thresholds (non-Youden):")
    selected = df_out[df_out["Is_Selected"] == 1][
        ["ExternalTag", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "ThresholdSelectionRule"]
    ]
    print(selected.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
