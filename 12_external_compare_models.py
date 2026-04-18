import os
import argparse
import warnings
import numpy as np
import joblib
import pandas as pd
import torch

import torch.nn as nn
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.stats import percentileofscore
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import config


warnings.filterwarnings("ignore")

BOOTSTRAP_N = 1000
SEED = 42
ROC_MARK = "1"
THRESHOLD_STEP = 0.005


class StudentNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(192, 96), dropouts=(0.25, 0.15)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h, d in zip(hidden_dims, dropouts):
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(d),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, y_prob[idx]))

    auc = float(roc_auc_score(y_true, y_prob))
    if len(aucs) == 0:
        return auc, np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return auc, float(lo), float(hi)


def build_threshold_grid(step=THRESHOLD_STEP):
    grid = np.arange(0.0, 1.0 + step, step)
    return np.clip(np.round(grid, 6), 0.0, 1.0)


def clinical_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return sens, spec, ppv, npv


def threshold_sweep_table(y_true, y_prob, model_name, thresholds):
    rows = []
    for th in thresholds:
        sens, spec, ppv, npv = clinical_metrics(y_true, y_prob, th)
        rows.append({
            "Model": model_name,
            "Threshold": float(th),
            "Sensitivity": float(sens),
            "Specificity": float(spec),
            "PPV": float(ppv),
            "NPV": float(npv),
            "Sens_minus_Spec": float(sens - spec),
            "Sens_gt_Spec": int(sens > spec),
        })
    return pd.DataFrame(rows)


def select_threshold_sens_gt_spec(y_true, y_prob, thresholds):
    df = threshold_sweep_table(y_true, y_prob, model_name="tmp", thresholds=thresholds)
    candidates = df[df["Sens_gt_Spec"] == 1]
    if len(candidates) > 0:
        best_idx = candidates["Sens_minus_Spec"].idxmax()
        return float(df.loc[best_idx, "Threshold"]), True
    best_idx = df["Sens_minus_Spec"].idxmax()
    return float(df.loc[best_idx, "Threshold"]), False


def build_embeddings(df_target, df_train_raw, features):
    views = {
        "Metabolic": ["BMI", "糖尿病", "代谢", "腰臀", "饮食", "肉"],
        "Hormonal": ["月经", "绝经", "生育", "哺乳", "激素"],
    }

    for name, keywords in views.items():
        cols = [c for c in features if any(k in c for k in keywords)]
        if len(cols) > 1:
            scaler = StandardScaler()
            pca = PCA(n_components=1)
            x_tr = df_train_raw[cols].fillna(df_train_raw[cols].mean())
            scaler.fit(x_tr)
            pca.fit(scaler.transform(x_tr))
            x_ext = df_target[cols].fillna(df_train_raw[cols].mean())
            df_target[f"Embedding_{name}"] = pca.transform(scaler.transform(x_ext))
        else:
            df_target[f"Embedding_{name}"] = 0.0

    km_scaler = StandardScaler()
    kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    x_tr_all = df_train_raw[features].fillna(df_train_raw[features].mean())
    km_scaler.fit(x_tr_all)
    kmeans.fit(km_scaler.transform(x_tr_all))

    x_ext_all = df_target[features].fillna(df_train_raw[features].mean())
    dists = kmeans.transform(km_scaler.transform(x_ext_all))
    for i in range(5):
        df_target[f"Embedding_Proto_Dist_{i}"] = dists[:, i]

    return df_target


def align_external_features(df_ext_valid):
    train_raw_path = config.TRAIN_FILE
    df_train = pd.read_csv(train_raw_path)
    train_features = [c for c in df_train.columns if c not in ["BC", "cohortid"]]
    train_means = df_train[train_features].mean()

    df_local = df_ext_valid.copy()
    if "年龄" in df_local.columns and "年龄分类（选一个）" not in df_local.columns:
        df_local["年龄分类（选一个）"] = pd.cut(
            df_local["年龄"],
            bins=[0, 34, 44, 54, 64, 200],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)

    df_aligned = pd.DataFrame(index=df_local.index)
    for col in train_features:
        if col in df_local.columns:
            df_aligned[col] = pd.to_numeric(df_local[col], errors="coerce")
        else:
            df_aligned[col] = train_means.get(col, 0.0)

    for col in train_features:
        df_aligned[col] = df_aligned[col].fillna(train_means.get(col, 0.0))

    df_aligned = build_embeddings(df_aligned, df_train[train_features], train_features)
    return df_aligned


def run_internal_models(df_ext_valid, device):
    model_student_path = os.path.join(config.OUTPUT_DIR, "model_student.pth")
    model_cls_path = os.path.join(config.OUTPUT_DIR, "model_stream1_cls.json")
    model_rank_path = os.path.join(config.OUTPUT_DIR, "model_stream2_rank.json")
    val_processed_path = config.PROCESSED_DATA_PREFIX + "val.csv"

    checkpoint = torch.load(model_student_path, map_location=device, weights_only=False)
    saved_feats = checkpoint["features"]
    input_dim = int(checkpoint["state_dict"]["net.0.weight"].shape[1])
    hidden_dims = tuple(checkpoint.get("hidden_dims", [192, 96]))
    dropouts = tuple(checkpoint.get("dropouts", [0.25, 0.15]))
    fusion_w = float(checkpoint.get("fusion_w", 0.5))

    model_cls = xgb.Booster()
    model_cls.load_model(model_cls_path)
    model_rank = xgb.Booster()
    model_rank.load_model(model_rank_path)

    df_aligned = align_external_features(df_ext_valid)

    x_ext_aligned = pd.DataFrame(index=df_aligned.index)
    for f in saved_feats:
        x_ext_aligned[f] = df_aligned[f] if f in df_aligned.columns else 0.0
    x_ext_base = x_ext_aligned.values.astype(np.float32)

    val_df = pd.read_csv(val_processed_path)
    x_val_for_cdf = pd.DataFrame(index=val_df.index)
    for f in saved_feats:
        x_val_for_cdf[f] = val_df[f] if f in val_df.columns else 0.0

    dval = xgb.DMatrix(x_val_for_cdf.values.astype(np.float32), feature_names=saved_feats)
    dext = xgb.DMatrix(x_ext_base, feature_names=saved_feats)

    pred_cls_prob_val = expit(model_cls.predict(dval))
    pred_rank_score_val = model_rank.predict(dval)

    pred_stream1 = expit(model_cls.predict(dext))
    pred_stream2_raw = model_rank.predict(dext)

    rank_p_cls_ext = np.array([percentileofscore(pred_cls_prob_val, x, kind="weak") / 100 for x in pred_stream1])
    rank_p_rnk_ext = np.array([percentileofscore(pred_rank_score_val, x, kind="weak") / 100 for x in pred_stream2_raw])
    pred_teacher = fusion_w * rank_p_cls_ext + (1.0 - fusion_w) * rank_p_rnk_ext

    x_ext_student_stacked = np.column_stack([
        x_ext_base,
        pred_stream1,
        rank_p_rnk_ext,
        pred_teacher,
    ]).astype(np.float32)

    if input_dim == x_ext_base.shape[1]:
        x_student = x_ext_base
    elif input_dim == x_ext_student_stacked.shape[1]:
        x_student = x_ext_student_stacked
    else:
        raise ValueError(
            f"Cannot match student input dims: state={input_dim}, base={x_ext_base.shape[1]}, stacked={x_ext_student_stacked.shape[1]}"
        )

    x_t = torch.FloatTensor(x_student).to(device)
    is_ensemble = bool(checkpoint.get("is_ensemble", False)) and ("ensemble_members" in checkpoint)

    if is_ensemble and len(checkpoint["ensemble_members"]) > 0:
        preds = []
        for member in checkpoint["ensemble_members"]:
            m_hidden = tuple(member.get("hidden_dims", list(hidden_dims)))
            m_drop = tuple(member.get("dropouts", list(dropouts)))
            model = StudentNet(input_dim=input_dim, hidden_dims=m_hidden, dropouts=m_drop).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(model(x_t)).cpu().numpy().flatten())
        pred_student = np.mean(preds, axis=0)
    else:
        model = StudentNet(input_dim=input_dim, hidden_dims=hidden_dims, dropouts=dropouts).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            pred_student = torch.sigmoid(model(x_t)).cpu().numpy().flatten()

    return {
        "Teacher (Fusion)": pred_teacher,
        "Our Proposed Model": pred_student,
        "_aligned_df": df_aligned,
    }


def load_baseline_feature_list(model_dir):
    feat_path = os.path.join(model_dir, "baseline_feature_columns.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing feature list: {feat_path}")
    return pd.read_csv(feat_path)["feature"].tolist()


def run_baseline_models(df_aligned, model_dir):
    registry_path = os.path.join(model_dir, "baseline_model_registry.csv")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Missing model registry: {registry_path}")

    feat_cols = load_baseline_feature_list(model_dir)
    x_ext = pd.DataFrame(index=df_aligned.index)
    for c in feat_cols:
        x_ext[c] = df_aligned[c] if c in df_aligned.columns else 0.0

    pred_map = {}
    reg = pd.read_csv(registry_path)
    for _, row in reg.iterrows():
        name = row["Model"]
        model_file = row["Model_File"]
        if not os.path.exists(model_file):
            print(f"[WARN] missing model file, skip: {model_file}")
            continue
        clf = joblib.load(model_file)
        pred_map[name] = clf.predict_proba(x_ext.values)[:, 1]
    return pred_map


def plot_external_roc(y_true, pred_map, save_path):
    items = []
    for name, pred in pred_map.items():
        try:
            auc = roc_auc_score(y_true, pred)
            items.append((auc, name, pred))
        except Exception:
            continue

    items.sort(key=lambda x: x[0], reverse=True)

    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")

    palette = [
        "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4",
        "#91D1C2", "#7E6148", "#B09C85", "#4B4B4B", "#A1C9F4"
    ]

    color_idx = 0
    for auc, name, pred in items:
        fpr, tpr, _ = roc_curve(y_true, pred)
        if name == "Our Proposed Model":
            ax.plot(fpr, tpr, color="#E64B35", lw=3.6, alpha=1.0, label=f"Our Proposed Model (AUC={auc:.3f})", zorder=12)
        elif name in ["Stream 1 (Classification)", "Stream 2 (Ranking)", "Teacher (Fusion)"]:
            stream_color = {
                "Stream 1 (Classification)": "#FF8C00",
                "Stream 2 (Ranking)": "#8A2BE2",
                "Teacher (Fusion)": "#20B2AA",
            }.get(name, "#555555")
            ax.plot(fpr, tpr, color=stream_color, lw=2.4, alpha=0.95, label=f"Ours: {name} (AUC={auc:.3f})", zorder=10)
        else:
            ax.plot(
                fpr,
                tpr,
                color=palette[color_idx % len(palette)],
                lw=1.4,
                alpha=0.75,
                label=f"{name} (AUC={auc:.3f})",
                zorder=5,
            )
            color_idx += 1

    ax.plot([0, 1], [0, 1], color="#8c8c8c", linestyle="--", lw=1.2, label="Random Guess", zorder=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic - External", fontsize=18, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8, color="#cfcfcf")
    ax.tick_params(axis="both", labelsize=11, width=1.2)

    ax.text(
        0.03,
        0.97,
        f"{ROC_MARK}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#303030",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#9a9a9a", alpha=0.9),
        zorder=20,
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.8)

    legend = ax.legend(
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        facecolor="#efefef",
        edgecolor="#8e8e8e",
        title="Models",
        title_fontsize=10,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
def main():
    parser = argparse.ArgumentParser(
        description="External validation for a custom external Excel (inference-only multi-model comparison)."
    )
    parser.add_argument(
        "--external-file",
        type=str,
        default=None,
        help="Optional cleaned external file path. Must be external_cleaned_*.xlsx.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="20260313_2",
        help="Suffix used for output folder: results_flat_{output_tag}.",
    )
    args = parser.parse_args()

    if args.external_file is None:
        ext_raw_path = os.path.join(config.BASE_DIR, f"results_flat_{args.output_tag}", f"external_cleaned_{args.output_tag}.xlsx")
    else:
        ext_raw_path = args.external_file

    ext_raw_path = os.path.abspath(ext_raw_path)
    if "external_cleaned_" not in os.path.basename(ext_raw_path):
        raise ValueError("Only cleaned external file is allowed. Please pass external_cleaned_*.xlsx")
    if not os.path.exists(ext_raw_path):
        raise FileNotFoundError(f"External Excel not found: {ext_raw_path}")

    output_dir = os.path.join(config.BASE_DIR, f"results_flat_{args.output_tag}")
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(config.OUTPUT_DIR, "baseline_models")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Baseline model dir not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] external file: {ext_raw_path}")
    print(f"[INFO] output dir: {output_dir}")

    df_ext = pd.read_excel(ext_raw_path, header=2, engine="openpyxl")
    for col in df_ext.columns:
        if col != "cohortid":
            df_ext[col] = pd.to_numeric(df_ext[col], errors="coerce")

    # Inference directly on provided external cohort.
    df_ext_valid = df_ext.dropna(subset=["BC"]).copy()
    y_true = df_ext_valid["BC"].astype(int).values
    print(f"[INFO] labeled samples: {len(y_true)}")

    internal_preds = run_internal_models(df_ext_valid, device=device)
    df_aligned = internal_preds.pop("_aligned_df")
    baseline_preds = run_baseline_models(df_aligned, model_dir=model_dir)

    all_preds = {}
    all_preds.update(baseline_preds)
    all_preds.update(internal_preds)

    threshold_grid = build_threshold_grid(step=THRESHOLD_STEP)
    threshold_rows = []
    for name, pred in all_preds.items():
        df_curve = threshold_sweep_table(y_true, pred, model_name=name, thresholds=threshold_grid)
        threshold_rows.append(df_curve)

    df_threshold = pd.concat(threshold_rows, axis=0, ignore_index=True)
    threshold_path = os.path.join(output_dir, f"external_threshold_sensitivity_specificity_curve_{args.output_tag}.csv")
    df_threshold.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    rows = []
    for name, pred in all_preds.items():
        auc, lo, hi = bootstrap_auc_ci(y_true, pred, n_boot=BOOTSTRAP_N, seed=SEED)
        threshold, strict_ok = select_threshold_sens_gt_spec(y_true, pred, threshold_grid)
        sens, spec, ppv, npv = clinical_metrics(y_true, pred, threshold)
        rows.append(
            {
                "Model": name,
                "AUC": round(auc, 4),
                "AUC_Low": round(lo, 4) if not np.isnan(lo) else np.nan,
                "AUC_High": round(hi, 4) if not np.isnan(hi) else np.nan,
                "AUC (95% CI)": f"{auc:.4f} ({lo:.4f}-{hi:.4f})" if not np.isnan(lo) else f"{auc:.4f}",
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
                "PPV": round(ppv, 4),
                "NPV": round(npv, 4),
                "Threshold": round(threshold, 4),
                "ThresholdRule": "Sensitivity>Specificity",
                "StrictCandidate": int(strict_ok),
                "N": len(y_true),
            }
        )

    df_comp = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    comp_path = os.path.join(output_dir, f"external_comparison_table_{args.output_tag}.csv")
    df_comp.to_csv(comp_path, index=False, encoding="utf-8-sig")

    pred_out = pd.DataFrame({"target": y_true})
    for k, v in all_preds.items():
        pred_out[k] = v
    pred_path = os.path.join(output_dir, f"external_model_predictions_{args.output_tag}.csv")
    pred_out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    roc_path = os.path.join(output_dir, f"external_roc_comparison_{args.output_tag}.png")
    plot_external_roc(y_true, all_preds, roc_path)

    print("=" * 72)
    print(f"[OK] comparison table: {comp_path}")
    print(f"[OK] threshold curve: {threshold_path}")
    print(f"[OK] external predictions: {pred_path}")
    print(f"[OK] external roc figure: {roc_path}")
    if len(df_comp) > 0:
        print(f"[OK] best model: {df_comp.loc[0, 'Model']} | AUC={df_comp.loc[0, 'AUC']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
