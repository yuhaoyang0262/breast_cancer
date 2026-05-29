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
"""
10_visualize_separate.py
Top-Tier Journal Quality Visualizations
Features: Exact 1-to-1 Feature Translation (No Regex Bugs), Square ROC, Magma UMAP.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

import config
warnings.filterwarnings("ignore")

try:
    import inter
except ImportError:
    umap = None

# ==============================================================================
# 1. 顶刊学术配置与双核心配色
# ==============================================================================
COLOR_STUDENT = "#E64B35"  # Nature Cherry Red (最终蒸馏模型)
COLOR_TEACHER = "#00A087"  # Nature Teal/Green (教师融合模型)
COLOR_TREAT_ALL = "#8491B4" # Muted Blue

PALETTE_BASELINES = [
    "#4DBBD5", "#3C5488", "#F39B7F", "#91D1C2",
    "#7E6148", "#B09C85", "#4B4B4B", "#999999", "#A6CEE3"
]

def set_scientific_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5

# 1对1精准模型名映射
MODEL_MAP = {
    "逻辑回归": "Logistic Regression", "决策树": "Decision Tree",
    "Extra": "Extra Trees", "GBDT": "Hist-GBDT", "贝叶斯": "Naive Bayes",
    "KNN": "KNN", "SVM": "Linear SVM", "神经网络": "MLP Neural Network",
    "AdaBoost": "AdaBoost", "Stream1": "Ours: Stream 1 (Cls)",
    "Stream2": "Ours: Stream 2 (Rank)", "Teacher": "Ours: Teacher (Fusion)",
    "Student": "Our Proposed Model", "pred_stream1": "Ours: Stream 1 (Cls)",
    "pred_stream2": "Ours: Stream 2 (Rank)", "pred_teacher": "Ours: Teacher (Fusion)",
    "pred_student": "Our Proposed Model", "Our Proposed Model": "Our Proposed Model"
}

# ==============================================================================
# 2. 数据处理与校准函数
# ==============================================================================
def get_best_pred_file(split_keyword):
    if not os.path.exists(config.OUTPUT_DIR): return None
    candidates = [f for f in os.listdir(config.OUTPUT_DIR)
                  if split_keyword in f.lower() and f.endswith('.csv') and 'prediction' in f.lower()]
    best_file, max_cols = None, 0
    for f in candidates:
        try:
            path = os.path.join(config.OUTPUT_DIR, f)
            df = pd.read_csv(path, nrows=1)
            if len(df.columns) > max_cols and 'target' in df.columns:
                max_cols, best_file = len(df.columns), path
        except: pass
    return best_file

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    bad_keywords = ['Random Forest', 'RF', 'LightGBM', 'CatBoost', 'XGBoost', '随机森林']
    cols_to_drop = [c for c in df.columns if any(k.lower() in c.lower() for k in bad_keywords)]
    df = df.drop(columns=cols_to_drop)

    # 精确匹配模型名称
    new_cols = []
    for c in df.columns:
        matched = False
        for k, v in MODEL_MAP.items():
            if k in c:
                new_cols.append(v)
                matched = True
                break
        if not matched: new_cols.append(c)
    df.columns = new_cols
    return df

def net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    out = []
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        out.append(0.0 if pt >= 1 else (tp / n - fp / n * pt / (1 - pt)))
    return np.asarray(out)

def get_calibrated_prob(df_val, df_target, our_col):
    y_val = df_val["target"].values
    x_val = df_val[our_col].values.reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0)
    calibrator.fit(x_val, y_val)
    x_target = df_target[our_col].values.reshape(-1, 1)
    return calibrator.predict_proba(x_target)[:, 1]

# ==============================================================================
# 3. 绘图引擎 (完美正方形 & 高级配色)
# ==============================================================================
def plot_professional_roc(df_pred: pd.DataFrame, split_label: str, save_path: str):
    if "target" not in df_pred.columns: return
    y_true = df_pred["target"].values
    model_cols = [c for c in df_pred.columns if c != "target"]

    model_stats = [(roc_auc_score(y_true, df_pred[c].values), c) for c in model_cols]
    model_stats.sort(key=lambda x: x[0], reverse=True)

    set_scientific_style()
    # 保持正方形
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect('equal', adjustable='box')

    color_idx = 0
    for auc, col in model_stats:
        fpr, tpr, _ = roc_curve(y_true, df_pred[col].values)
        if "Proposed Model" in col or "Student" in col:
            ax.plot(fpr, tpr, color=COLOR_STUDENT, lw=3.2, alpha=1.0, label=f"{col} (AUC={auc:.3f})", zorder=10)
        elif "Teacher" in col:
            ax.plot(fpr, tpr, color=COLOR_TEACHER, lw=2.5, alpha=0.95, label=f"Ours: Teacher (AUC={auc:.3f})", zorder=9)
        elif "Stream" in col:
            continue
        else:
            ax.plot(fpr, tpr, color=PALETTE_BASELINES[color_idx % len(PALETTE_BASELINES)], lw=1.2, alpha=0.65, label=f"{col} (AUC={auc:.3f})", zorder=5)
            color_idx += 1

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Random Guess', zorder=1)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14, fontweight='bold')
    ax.set_title(f"Receiver Operating Characteristic - {split_label}", fontsize=16, fontweight='bold', pad=15)

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor='gray', title="Models", title_fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_professional_dca(df_pred: pd.DataFrame, df_val: pd.DataFrame, split_label: str, save_path: str):
    if "target" not in df_pred.columns: return
    y_true = df_pred["target"].values

    col_stu = [c for c in df_pred.columns if "Proposed Model" in c or "Student" in c]
    if not col_stu: return

    cal_prob_stu = get_calibrated_prob(df_val, df_pred, col_stu[0])

    max_thresh = 0.06
    thresholds = np.linspace(0.001, max_thresh, 150)
    prevalence = y_true.mean()

    set_scientific_style()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    nb_all  = np.asarray([prevalence - (1 - prevalence) * pt / (1 - pt) for pt in thresholds])
    nb_none = np.zeros_like(thresholds)
    nb_stu  = net_benefit(y_true, cal_prob_stu, thresholds)

    ax.plot(thresholds, nb_all, color=COLOR_TREAT_ALL, linestyle="--", linewidth=2.0, label="Treat All", zorder=2)
    ax.plot(thresholds, nb_none, color="black", linestyle="-", linewidth=2.0, label="Treat None", zorder=2)
    ax.plot(thresholds, nb_stu, color=COLOR_STUDENT, linewidth=3.2, label="Our Proposed Model", zorder=10)

    ax.fill_between(
        thresholds, nb_stu, np.maximum(nb_all, nb_none),
        where=(nb_stu > np.maximum(nb_all, nb_none)),
        color=COLOR_STUDENT, alpha=0.12, label="Net Clinical Benefit", zorder=1
    )

    y_max = max(prevalence * 1.3, 0.02)
    ax.set_ylim(-0.001, y_max)
    ax.set_xlim(0, max_thresh)
    ax.set_xlabel("Threshold Probability (Risk Preference)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Net Benefit", fontsize=14, fontweight='bold')
    ax.set_title(f"Decision Curve Analysis - {split_label}", fontsize=16, fontweight='bold', pad=15)

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="upper right", fontsize=12, framealpha=1.0, edgecolor='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap(X_test: pd.DataFrame, df_pred: pd.DataFrame, save_path: str):
    if umap is None: return
    col_stu = [c for c in df_pred.columns if "Proposed Model" in c or "Student" in c]
    if not col_stu: return
    best_prob = df_pred[col_stu[0]].values

    print("   -> Running UMAP dimension reduction...")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2, random_state=config.SEED)
    emb = reducer.fit_transform(X_test)

    order = np.argsort(best_prob)
    emb_sorted = emb[order]
    prob_sorted = best_prob[order]

    set_scientific_style()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    sc = ax.scatter(emb_sorted[:, 0], emb_sorted[:, 1], c=prob_sorted,
                    s=25, cmap="magma", alpha=0.85, edgecolors="none")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Predicted Breast Cancer Risk", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP Dimension 1", fontsize=13, fontweight='bold')
    ax.set_ylabel("UMAP Dimension 2", fontsize=13, fontweight='bold')
    ax.set_title("Patient Risk Landscape (UMAP)", fontsize=15, fontweight="bold", pad=15)

    for side in ["top", "right", "left", "bottom"]: ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 4. 主执行流程
# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print("   Generating Top-Tier Journal Figures")
    print("=" * 70)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    test_feat_path = config.PROCESSED_DATA_PREFIX + "test.csv"
    if not os.path.exists(test_feat_path):
        print(f"[ERROR] Cannot find {test_feat_path}")
        return
    df_feat = pd.read_csv(test_feat_path)
    feature_cols = [c for c in df_feat.columns if c not in [config.TARGET_COL, "cohortid"]]
    X_test = df_feat[feature_cols]

    val_file = get_best_pred_file("val")
    if not val_file:
        print("[ERROR] Cannot find validation predictions.")
        return
    df_val = load_and_clean_data(val_file)

    for split in ["train", "val", "test"]:
        filepath = get_best_pred_file(split)
        if not filepath: continue

        split_name = "Train" if split == "train" else ("Validation" if split == "val" else "Test")
        print(f"\n>>> Processing {split_name} Cohort Data...")
        df_pred = load_and_clean_data(filepath)

        roc_path = os.path.join(config.OUTPUT_DIR, f"Figure_ROC_{split_name}.png")
        plot_professional_roc(df_pred, split_name, roc_path)
        print(f"   [SAVED] {roc_path}")

        dca_path = os.path.join(config.OUTPUT_DIR, f"Figure_DCA_{split_name}.png")
        plot_professional_dca(df_pred, df_val, split_name, dca_path)
        print(f"   [SAVED] {dca_path}")

        if split_name == "Test":
            print("\n>>> Generating Interpretability Plots (Test Cohort)...")
            umap_path = os.path.join(config.OUTPUT_DIR, "Figure_UMAP_Test.png")
            plot_umap(X_test, df_pred, umap_path)
            print(f"   [SAVED] {umap_path}")

    print("\n✅ 可视化生成完成（ROC / DCA / UMAP）\n")

if __name__ == "__main__":
    main()