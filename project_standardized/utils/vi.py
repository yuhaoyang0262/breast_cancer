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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import umap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# ==============================================================================
# 1. 顶刊基因植入：SciencePlots 学术画风与 Nature 官方配色
# ==============================================================================
# 推荐安装: pip install SciencePlots
try:
    import scienceplots
    plt.style.use(['science', 'nature', 'no-latex'])
except ImportError:
    print("[提示] 未检测到 SciencePlots，使用 Matplotlib 默认学术样式。")
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['tick.params.direction'] = 'in'

# Nature Publishing Group (NPG) 经典色系
NPG_PALETTE = ["#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
COLOR_CHAMPION = "#951909" # 樱桃红 (Student)
COLOR_TEACHER  = "#91D1C2" # 浅松石 (Teacher)
COLOR_DCA_GAIN = "#FEE0D2" # 极浅红 (DCA 阴影面积)

LABEL_STUDENT = "Student Network Distillation"
LABEL_TEACHER = "Teacher Network Fusion"
LABEL_STREAM1 = "Classification Stream"
LABEL_STREAM2 = "Ranking Stream"

RESULTS_DIR = r"D:\breast_cancer\results_flat_0313"


class StudentNet(nn.Module):
    """Training-consistent student architecture for inference."""
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropouts=(0.4, 0.3, 0.2)):
        super().__init__()
        h1, h2, h3 = hidden_dims
        d1, d2, d3 = dropouts
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(d2),
            nn.Linear(h2, h3), nn.BatchNorm1d(h3), nn.ReLU(), nn.Dropout(d3),
            nn.Linear(h3, 1),
        )

    def forward(self, x):
        return self.net(x)


def infer_hidden_dims_from_state_dict(state_dict):
    """Infer hidden layer sizes from Linear weights in a checkpoint."""
    linear_layers = []
    for k, v in state_dict.items():
        if k.startswith("net.") and k.endswith(".weight") and v.ndim == 2:
            layer_idx = int(k.split(".")[1])
            linear_layers.append((layer_idx, tuple(v.shape)))
    linear_layers.sort(key=lambda x: x[0])
    if len(linear_layers) < 4:
        return (256, 128, 64)
    # [(out1,in), (out2,out1), (out3,out2), (1,out3)]
    return (linear_layers[0][1][0], linear_layers[1][1][0], linear_layers[2][1][0])


def load_split_student_predictions(results_dir):
    """Load train/val/test student predictions.

    Priority:
    1) Use saved prediction CSVs produced by training if available.
    2) If train/val split prediction file is missing, infer Student probs from checkpoint.
    """
    ckpt_path = os.path.join(results_dir, "model_student.pth")

    # 1) Try saved prediction files first
    split_csv = {
        "Train": os.path.join(results_dir, "train_predictions.csv"),
        "Validation": os.path.join(results_dir, "val_predictions.csv"),
        "Test": os.path.join(results_dir, "final_predictions.csv"),
    }

    split_data = {}
    for split_name, fp in split_csv.items():
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "target" in df.columns and "pred_student" in df.columns:
            y = df["target"].values.astype(int)
            p = df["pred_student"].values.astype(float)
            split_data[split_name] = (y, p)

    # If all 3 are present, return immediately
    if all(k in split_data for k in ["Train", "Validation", "Test"]):
        print("[INFO] 已从训练保存预测文件读取 Train/Validation/Test。")
        return split_data

    # 2) Fallback: infer missing splits from student checkpoint
    if not os.path.exists(ckpt_path):
        missing = [k for k in ["Train", "Validation", "Test"] if k not in split_data]
        raise FileNotFoundError(f"Missing checkpoint for fallback inference: {ckpt_path}; missing splits={missing}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        feature_list = ckpt.get("features", None)
        input_dim = int(ckpt.get("input_dim", 0))
    else:
        state_dict = ckpt
        feature_list = None
        input_dim = 0

    hidden_dims = infer_hidden_dims_from_state_dict(state_dict)
    if input_dim <= 0:
        # Fallback to inferred input dimension from first linear layer
        first_w = state_dict.get("net.0.weight")
        input_dim = int(first_w.shape[1]) if first_w is not None else 41

    model = StudentNet(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    split_feature_files = {
        "Train": os.path.join(results_dir, "processed_train.csv"),
        "Validation": os.path.join(results_dir, "processed_val.csv"),
        "Test": os.path.join(results_dir, "processed_test.csv"),
    }

    for split_name, fp in split_feature_files.items():
        if split_name in split_data:
            continue
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing split file: {fp}")
        df = pd.read_csv(fp)

        if "BC" in df.columns:
            y = df["BC"].values.astype(int)
        elif "target" in df.columns:
            y = df["target"].values.astype(int)
        else:
            raise ValueError(f"No target column (BC/target) found in {fp}")

        if feature_list is not None:
            X_df = pd.DataFrame(index=df.index)
            for f in feature_list:
                X_df[f] = df[f] if f in df.columns else 0.0
            X = X_df.values.astype(np.float32)
        else:
            cols = [c for c in df.columns if c not in ["BC", "target", "cohortid"]]
            X = df[cols].values.astype(np.float32)

        if X.shape[1] != input_dim:
            raise ValueError(f"Input dim mismatch for {split_name}: got {X.shape[1]}, expected {input_dim}")

        with torch.no_grad():
            probs = torch.sigmoid(model(torch.FloatTensor(X))).cpu().numpy().flatten()

        split_data[split_name] = (y, probs)

    print("[INFO] 部分 split 预测缺失，已用 model_student.pth + processed_*.csv 自动补算 Student 概率。")

    return split_data

# ==============================================================================
# 2. 精准读取您的本地数据 (已适配您的真实列名)
# ==============================================================================
path_preds = os.path.join(RESULTS_DIR, "final_predictions.csv")
path_feats = os.path.join(RESULTS_DIR, "processed_test.csv")

print(">>> 正在加载并解析本地真实数据...")
df_preds = pd.read_csv(path_preds)
df_feats = pd.read_csv(path_feats)

# 完美匹配您的真实列名
y_test    = df_preds['target'].values
p_student = df_preds['pred_student'].values
p_teacher = df_preds['pred_teacher'].values
p_stream1 = df_preds['pred_stream1'].values
p_stream2 = df_preds['pred_stream2'].values

# 提取用于 UMAP 的 41 维特征矩阵 (自动剔除 target 列)
X_umap_raw = df_feats.drop(columns=['target'], errors='ignore').values
n_samples = len(y_test)
prevalence = np.mean(y_test)

print(f"✅ 数据加载成功! 样本数: {n_samples}, 患病率: {prevalence:.4f}, UMAP维度: {X_umap_raw.shape[1]}")

# ==============================================================================
# 3. 核心计算函数 (DCA)
# ==============================================================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n_total = len(y_true)
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if pt == 0: nb = (tp / n_total)
        elif pt == 1: nb = 0
        else: nb = (tp / n_total) - (fp / n_total) * (pt / (1 - pt))
        net_benefit.append(nb)
    return np.array(net_benefit)

# ==============================================================================
# 4. 生成 4 面板高保真图
# ==============================================================================
print(">>> 正在绘图，UMAP降维预计需要 1 分钟左右...")
fig = plt.figure(figsize=(16, 14), dpi=300)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1.2], hspace=0.25, wspace=0.2)

# --- Panel A: ROC ---
ax1 = fig.add_subplot(gs[0, 0])
fpr_ours, tpr_ours, _ = roc_curve(y_test, p_student)
ax1.plot(fpr_ours, tpr_ours, color=COLOR_CHAMPION, linewidth=2.5, label=f'Ours: {LABEL_STUDENT} (AUC={auc(fpr_ours, tpr_ours):.3f})', zorder=10)

models = [
    (p_teacher, f"{LABEL_TEACHER}", COLOR_TEACHER, 1.8),
]
for p_v, name, color, lw in models:
    fpr, tpr, _ = roc_curve(y_test, p_v)
    ax1.plot(fpr, tpr, color=color, linewidth=lw, alpha=0.8, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
ax1.plot([0, 1], [0, 1], color=NPG_PALETTE[4], linestyle='--', linewidth=1, alpha=0.5)

# 局部放大镜
axins = inset_axes(ax1, width="45%", height="40%", loc='lower right', borderpad=2)
axins.plot(fpr_ours, tpr_ours, color=COLOR_CHAMPION, linewidth=2.5)
for p_v, name, color, lw in models:
    fpr, tpr, _ = roc_curve(y_test, p_v)
    axins.plot(fpr, tpr, color=color, linewidth=lw, alpha=0.8)
axins.set_xlim(0.0, 0.25); axins.set_ylim(0.6, 0.95)
axins.tick_params(axis='both', which='major', labelsize=8, direction='in')
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle=':', alpha=0.5)

ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel("1 - Specificity", fontsize=12, fontweight='bold')
ax1.set_ylabel("Sensitivity", fontsize=12, fontweight='bold')
ax1.set_title("A. ROC Curves", fontsize=14, fontweight='bold', loc='left')
ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)

# --- Panel B: DCA ---
ax2 = fig.add_subplot(gs[0, 1])
dca_thresholds = np.linspace(0.001, 0.5, 100)
nb_ours = calculate_net_benefit(y_test, p_student, dca_thresholds)
nb_all = calculate_net_benefit(y_test, np.ones(n_samples), dca_thresholds)
nb_none = calculate_net_benefit(y_test, np.zeros(n_samples), dca_thresholds)

ax2.plot(dca_thresholds, nb_none, color=NPG_PALETTE[4], linewidth=1.5, label='Intervene None')
ax2.plot(dca_thresholds, nb_all, color=NPG_PALETTE[0], linewidth=1.5, linestyle='--', label='Intervene All')
ax2.plot(dca_thresholds, nb_ours, color=COLOR_CHAMPION, linewidth=2.5, label=f'Ours: {LABEL_STUDENT}', zorder=10)

clinical_gain_upper = nb_ours
clinical_gain_lower = np.maximum(nb_all, nb_none)
ax2.fill_between(dca_thresholds, clinical_gain_lower, clinical_gain_upper,
                 where=(clinical_gain_upper > clinical_gain_lower),
                 color=COLOR_DCA_GAIN, alpha=0.6, zorder=1, label='Model Guided Net Clinical Benefit')

ax2.set_xlim([0, 0.5]); ax2.set_ylim([-0.005, prevalence + 0.005])
ax2.set_xlabel("Threshold Probability", fontsize=12, fontweight='bold')
ax2.set_ylabel("Net Clinical Benefit", fontsize=12, fontweight='bold')
ax2.set_title("B. Decision Curve Analysis", fontsize=14, fontweight='bold', loc='left')
ax2.legend(frameon=False, loc='upper right', fontsize=10)

# --- Panel C: Calibration ---
ax3 = fig.add_subplot(gs[1, 0])
def plot_calib(y_t, p_v, name, color, marker, lw):
    true_f, pred_m = calibration_curve(y_t, p_v, n_bins=10, strategy='quantile')
    ax3.plot(pred_m, true_f, color=color, marker=marker, markersize=5, linewidth=lw, label=f'{name} (Brier={brier_score_loss(y_t, p_v):.4f})')

plot_calib(y_test, p_student, f'Ours: {LABEL_STUDENT}', COLOR_CHAMPION, 'o', 2.5)
plot_calib(y_test, p_teacher, f'{LABEL_TEACHER}', COLOR_TEACHER, '^', 1.5)
plot_calib(y_test, p_stream1, f'{LABEL_STREAM1}', NPG_PALETTE[4], 's', 1.0)
plot_calib(y_test, p_stream2, f'{LABEL_STREAM2}', NPG_PALETTE[2], 'x', 1.0)

ax3.plot([0, 1], [0, 1], color=NPG_PALETTE[4], linestyle='--', linewidth=1, alpha=0.5, label='Perfect Calibration')
ax3.set_xlim([-0.02, 1.02]); ax3.set_ylim([-0.02, 1.02])
ax3.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight='bold')
ax3.set_ylabel("Observed Fraction", fontsize=12, fontweight='bold')
ax3.set_title("C. Reliability Diagrams", fontsize=14, fontweight='bold', loc='left')
ax3.legend(frameon=False, loc='upper left', fontsize=9)

# --- Panel D: UMAP ---
ax4 = fig.add_subplot(gs[1, 1])
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X_umap_raw)

ptiles = np.percentile(p_student, [80, 95, 99])
size_score = np.ones(n_samples) * 3
size_score[p_student > ptiles[0]] = 8
size_score[p_student > ptiles[1]] = 20
size_score[p_student > ptiles[2]] = 40

sc = ax4.scatter(embedding[:, 0], embedding[:, 1], c=p_student, cmap='RdYlBu_r', s=size_score, alpha=0.7, edgecolors='none', zorder=2)

high_risk_idx = np.where(p_student > ptiles[1])[0]
if len(high_risk_idx) > 0:
    sns.kdeplot(x=embedding[high_risk_idx, 0], y=embedding[high_risk_idx, 1],
                ax=ax4, colors=COLOR_CHAMPION, levels=4, linewidths=1.2, alpha=0.7, zorder=3)

ax4.set_xticks([]); ax4.set_yticks([])
ax4.spines['left'].set_visible(False); ax4.spines['bottom'].set_visible(False)

axins_cb = inset_axes(ax4, width="3%", height="60%", loc='center left', borderpad=-3)
cb = plt.colorbar(sc, cax=axins_cb, orientation='vertical')
cb.set_label('Predicted Risk', fontsize=10, fontweight='bold')
cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
ax4.set_title("D. Topographical Risk Landscape (UMAP)", fontsize=14, fontweight='bold', loc='left')

# 保存
output_path = os.path.join(RESULTS_DIR, "Figure_Nature_Style_Final.pdf")
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"✅ 保存成功: {output_path}")

# ==============================================================================
# 5. 新增图：Train/Validation/Test AUC 曲线（顶刊风格）
# ==============================================================================
print(">>> 正在生成分割集 ROC 总览图 (Train/Validation/Test)...")

split_pred = load_split_student_predictions(RESULTS_DIR)

fig2, ax = plt.subplots(figsize=(9.8, 7.8), dpi=320)

# Subtle background for a polished journal look
bg_x = np.linspace(0, 1, 256)
bg_y = np.linspace(0, 1, 256)
gx, gy = np.meshgrid(bg_x, bg_y)
grad = 0.94 + 0.06 * (1 - (gx * 0.75 + gy * 0.25))
ax.imshow(
    grad,
    extent=[0, 1, 0, 1],
    origin='lower',
    cmap='Greys_r',
    alpha=0.08,
    aspect='auto',
    zorder=0,
)

split_styles = [
    ("Train", "#4DBBD5", 2.8),
    ("Validation", "#00A087", 2.8),
    ("Test", "#E64B35", 3.2),
]

for split_name, color, lw in split_styles:
    y_s, p_s = split_pred[split_name]
    fpr_s, tpr_s, _ = roc_curve(y_s, p_s)
    auc_s = auc(fpr_s, tpr_s)
    ax.plot(
        fpr_s,
        tpr_s,
        color=color,
        linewidth=lw,
        alpha=0.95,
        label=f"{split_name} ROC (AUC = {auc_s:.3f})",
        zorder=3,
    )

ax.plot([0, 1], [0, 1], linestyle='--', color='#7F7F7F', linewidth=1.2, alpha=0.7, zorder=1)

# Highlight clinically relevant high-specificity regime
ax.axvspan(0.0, 0.2, color="#FEE0D2", alpha=0.18, zorder=1)
ax.text(0.015, 0.09, "High-specificity\nregion", fontsize=9, color="#6B6B6B", zorder=5)

axins2 = inset_axes(ax, width="36%", height="34%", loc='lower right', borderpad=1.8)
for split_name, color, lw in split_styles:
    y_s, p_s = split_pred[split_name]
    fpr_s, tpr_s, _ = roc_curve(y_s, p_s)
    axins2.plot(fpr_s, tpr_s, color=color, linewidth=2.1, alpha=0.95)
axins2.plot([0, 1], [0, 1], linestyle='--', color='#9A9A9A', linewidth=0.9, alpha=0.6)
axins2.set_xlim(0.0, 0.25)
axins2.set_ylim(0.55, 1.0)
axins2.tick_params(axis='both', which='major', labelsize=8, direction='in')
mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5", linestyle=':', alpha=0.5)

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("1 - Specificity", fontsize=13, fontweight='bold')
ax.set_ylabel("Sensitivity", fontsize=13, fontweight='bold')
ax.set_title("Student ROC Across Train / Validation / Internal Test", fontsize=16, fontweight='bold', pad=12)
ax.legend(frameon=False, loc='lower right', fontsize=11)

split_out_pdf = os.path.join(RESULTS_DIR, "Figure_ROC_Train_Val_Test.pdf")
split_out_png = os.path.join(RESULTS_DIR, "Figure_ROC_Train_Val_Test.png")
fig2.savefig(split_out_pdf, format='pdf', bbox_inches='tight')
fig2.savefig(split_out_png, format='png', bbox_inches='tight', dpi=320)
print(f"✅ 保存成功: {split_out_pdf}")
print(f"✅ 保存成功: {split_out_png}")

plt.show()