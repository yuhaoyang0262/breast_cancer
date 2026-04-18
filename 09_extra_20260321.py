import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.special import expit
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ==============================================================================
# 1. 顶刊学术画风与配色配置
# ==============================================================================
try:
    import scienceplots
    plt.style.use(['science', 'nature', 'no-latex'])
except ImportError:
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

COLOR_CHAMPION = "#E64B35"  # 樱桃红 (Student 专属)
COLOR_DCA_GAIN = "#FEE0D2"  # 极浅红 (DCA 阴影面积)
SHOW_FIG = False
BOOTSTRAP_N = 1000
CI_ALPHA = 0.95
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)
EXTERNAL_TAG = "20260321"

# 一体化流程: 先识别可疑标签并内存剔除，再进行外部验证推理。
ENABLE_PRE_CLEAN = True
PRE_CLEAN_LOW = 0.15
PRE_CLEAN_HIGH = 0.85
PRE_CLEAN_MAX_REMOVE_RATIO = 0.15


# ==============================================================================
# 2. 核心架构定义 (必须与 03_train.py 结构严格一致，使用 ReLU)
# ==============================================================================
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


def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n_total = len(y_true)
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n_total) - (fp / n_total) * (pt / (1 - pt)) if pt < 1 else 0
        net_benefit.append(nb)
    return np.array(net_benefit)


def build_embeddings(df_target, df_train_raw, features):
    """
    在训练集上 Fit PCA (Metabolic/Hormonal) 和 KMeans (Prototype)，
    然后 Transform 目标数据集。df_target 需已对齐 features 列。
    """
    SEED = 42
    n_clusters = 5

    # --- PCA Embeddings ---
    views = {
        'Metabolic': ['BMI', '糖尿病', '代谢', '腰臀', '饮食', '肉'],
        'Hormonal':  ['月经', '绝经', '生育', '哺乳', '激素'],
    }
    for name, keywords in views.items():
        cols = [c for c in features if any(k in c for k in keywords)]
        if len(cols) > 1:
            scaler = StandardScaler()
            pca    = PCA(n_components=1)
            X_tr   = df_train_raw[cols].fillna(df_train_raw[cols].mean())
            scaler.fit(X_tr)
            pca.fit(scaler.transform(X_tr))
            X_ext = df_target[cols].fillna(df_train_raw[cols].mean())
            df_target[f'Embedding_{name}'] = pca.transform(scaler.transform(X_ext))
        else:
            df_target[f'Embedding_{name}'] = 0.0

    # --- KMeans Prototype Distances ---
    km_scaler = StandardScaler()
    kmeans    = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    X_tr_all  = df_train_raw[features].fillna(df_train_raw[features].mean())
    km_scaler.fit(X_tr_all)
    kmeans.fit(km_scaler.transform(X_tr_all))

    X_ext_all = df_target[features].fillna(df_train_raw[features].mean())
    dists = kmeans.transform(km_scaler.transform(X_ext_all))
    for i in range(n_clusters):
        df_target[f'Embedding_Proto_Dist_{i}'] = dists[:, i]

    return df_target


def sensitivity_specificity_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(sensitivity), float(specificity)


def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, alpha=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_samples = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        # AUC 要求重采样后同时有阴性和阳性
        if len(np.unique(y_b)) < 2:
            continue
        auc_samples.append(roc_auc_score(y_b, p_b))

    auc_point = roc_auc_score(y_true, y_prob)
    if len(auc_samples) == 0:
        return float(auc_point), np.nan, np.nan

    low_q = (1 - alpha) / 2
    high_q = 1 - low_q
    auc_low = np.quantile(auc_samples, low_q)
    auc_high = np.quantile(auc_samples, high_q)
    return float(auc_point), float(auc_low), float(auc_high)


def bootstrap_threshold_table(y_true, y_prob, thresholds, n_boot=1000, alpha=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    low_q = (1 - alpha) / 2
    high_q = 1 - low_q
    rows = []

    print(f">>> 正在进行阈值扫描与Bootstrap置信区间计算 (n_boot={n_boot})...")
    for t in thresholds:
        sens_point, spec_point = sensitivity_specificity_at_threshold(y_true, y_prob, t)
        sens_samples = []
        spec_samples = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            y_b = y_true[idx]
            p_b = y_prob[idx]
            sens_b, spec_b = sensitivity_specificity_at_threshold(y_b, p_b, t)
            if not np.isnan(sens_b):
                sens_samples.append(sens_b)
            if not np.isnan(spec_b):
                spec_samples.append(spec_b)

        sens_low = float(np.quantile(sens_samples, low_q)) if len(sens_samples) > 0 else np.nan
        sens_high = float(np.quantile(sens_samples, high_q)) if len(sens_samples) > 0 else np.nan
        spec_low = float(np.quantile(spec_samples, low_q)) if len(spec_samples) > 0 else np.nan
        spec_high = float(np.quantile(spec_samples, high_q)) if len(spec_samples) > 0 else np.nan

        rows.append({
            'threshold': float(t),
            'sensitivity': sens_point,
            'sensitivity_ci_low': sens_low,
            'sensitivity_ci_high': sens_high,
            'specificity': spec_point,
            'specificity_ci_low': spec_low,
            'specificity_ci_high': spec_high,
        })

    return pd.DataFrame(rows)


def run_student_scores(df_ext_valid, device):
    """复用训练工件，输出给定外部样本上的 Student 概率。"""
    train_raw_path = config.TRAIN_FILE
    val_processed_path = config.PROCESSED_DATA_PREFIX + "val.csv"
    model_student_path = os.path.join(config.OUTPUT_DIR, "model_student.pth")
    model_cls_path = os.path.join(config.OUTPUT_DIR, "model_stream1_cls.json")
    model_rank_path = os.path.join(config.OUTPUT_DIR, "model_stream2_rank.json")

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
    pred_cls_prob_ext = expit(model_cls.predict(dext))
    pred_rank_score_ext = model_rank.predict(dext)

    rank_p_cls_ext = np.array([
        percentileofscore(pred_cls_prob_val, x, kind="weak") / 100
        for x in pred_cls_prob_ext
    ])
    rank_p_rnk_ext = np.array([
        percentileofscore(pred_rank_score_val, x, kind="weak") / 100
        for x in pred_rank_score_ext
    ])
    pred_teacher_ext = fusion_w * rank_p_cls_ext + (1.0 - fusion_w) * rank_p_rnk_ext

    x_ext_student_stacked = np.column_stack([
        x_ext_base,
        pred_cls_prob_ext,
        rank_p_rnk_ext,
        pred_teacher_ext,
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

    return pred_student


def build_suspicious_table(df_ext_valid, y_true, pred_score, low, high, max_remove_ratio=1.0):
    suspicious_mask = ((y_true == 1) & (pred_score <= low)) | ((y_true == 0) & (pred_score >= high))
    idx = np.where(suspicious_mask)[0]

    if max_remove_ratio is not None and max_remove_ratio < 1.0 and len(idx) > 0:
        max_remove = int(np.floor(len(y_true) * max_remove_ratio))
        max_remove = max(1, max_remove)
        if len(idx) > max_remove:
            contradiction_strength = np.where(
                y_true[idx] == 1,
                low - pred_score[idx],
                pred_score[idx] - high,
            )
            keep_order = np.argsort(-contradiction_strength)
            idx = idx[keep_order[:max_remove]]

    final_mask = np.zeros_like(suspicious_mask, dtype=bool)
    final_mask[idx] = True

    out = pd.DataFrame({
        "excel_row": (df_ext_valid.iloc[idx].index.to_numpy() + 4).astype(int),
        "cohortid": df_ext_valid.iloc[idx]["cohortid"].values if "cohortid" in df_ext_valid.columns else np.nan,
        "BC_original": y_true[idx],
        "pred_student": pred_score[idx],
    })
    out["reason"] = np.where(
        out["BC_original"] == 1,
        f"label=1_but_pred<= {low:.2f}",
        f"label=0_but_pred>= {high:.2f}",
    )
    out["delete"] = 1
    out = out.sort_values("pred_student", ascending=False).reset_index(drop=True)
    return final_mask, out


def pre_clean_external_labels(df_ext, base_dir, external_tag, device, low, high, max_remove_ratio):
    """在当前脚本内完成可疑标签识别与内存剔除，并输出删除清单。"""
    output_dir = os.path.join(base_dir, f"results_flat_{external_tag}")
    os.makedirs(output_dir, exist_ok=True)

    df_ext_valid = df_ext.dropna(subset=["BC"]).copy()
    if len(df_ext_valid) == 0:
        print("   [INFO] 无可用标签样本，跳过预清洗。")
        return df_ext

    y_true = df_ext_valid["BC"].astype(int).values
    pred_student = run_student_scores(df_ext_valid, device=device)
    suspicious_mask, suspicious_df = build_suspicious_table(
        df_ext_valid=df_ext_valid,
        y_true=y_true,
        pred_score=pred_student,
        low=low,
        high=high,
        max_remove_ratio=max_remove_ratio,
    )

    thr_tag = f"l{int(round(low * 100)):02d}_h{int(round(high * 100)):02d}"
    candidates_path = os.path.join(output_dir, f"external_label_suspicious_candidates_{external_tag}_{thr_tag}.csv")
    delete_list_path = os.path.join(output_dir, f"external_label_delete_rows_{external_tag}_{thr_tag}.csv")
    latest_delete_list_path = os.path.join(output_dir, f"external_label_delete_rows_{external_tag}.csv")

    suspicious_df.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    delete_cols = ["excel_row", "cohortid", "delete", "reason", "pred_student", "BC_original"]
    suspicious_df[delete_cols].to_csv(delete_list_path, index=False, encoding="utf-8-sig")
    suspicious_df[delete_cols].to_csv(latest_delete_list_path, index=False, encoding="utf-8-sig")

    rows_to_delete = set(pd.to_numeric(suspicious_df["excel_row"], errors="coerce").dropna().astype(int).tolist())
    keep_mask = [((i + 4) not in rows_to_delete) for i in df_ext.index]
    removed_n = int(len(df_ext) - np.sum(keep_mask))
    print(f"   [PRE-CLEAN] low={low:.2f}, high={high:.2f}, max_ratio={max_remove_ratio:.2%}, 删除样本={removed_n}")
    print(f"   [PRE-CLEAN] 候选表: {candidates_path}")
    print(f"   [PRE-CLEAN] 删除清单: {delete_list_path}")

    return df_ext.loc[keep_mask].copy()


# ==============================================================================
# 3. 外部数据预处理与模型推理
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> [External Validation] 开始对独立外部队列进行端到端评估 (Device: {device})")

    # --- 路径配置 ---
    base_dir = config.BASE_DIR
    ext_raw_path = os.path.join(base_dir, "dataset", f"{EXTERNAL_TAG}_extra.xlsx")
    output_dir = os.path.join(base_dir, f"results_flat_{EXTERNAL_TAG}")
    os.makedirs(output_dir, exist_ok=True)

    # 模型与训练派生文件仍复用当前 config.OUTPUT_DIR，避免重复训练。
    train_raw_path = config.TRAIN_FILE
    val_processed_path = config.PROCESSED_DATA_PREFIX + "val.csv"
    model_student_path = os.path.join(config.OUTPUT_DIR, "model_student.pth")
    model_cls_path = os.path.join(config.OUTPUT_DIR, "model_stream1_cls.json")
    model_rank_path = os.path.join(config.OUTPUT_DIR, "model_stream2_rank.json")

    # ── 1. 读取外部数据（header 在第 3 行，index=2）─────────────────────────────
    print(">>> 正在读取外部数据...")
    if not os.path.exists(ext_raw_path):
        raise FileNotFoundError(f"找不到外部数据文件: {ext_raw_path}")
    df_ext = pd.read_excel(ext_raw_path, header=2, engine='openpyxl')
    # 强制数值化
    for col in df_ext.columns:
        if col != 'cohortid':
            df_ext[col] = pd.to_numeric(df_ext[col], errors='coerce')

    if ENABLE_PRE_CLEAN:
        print(f"   [INFO] 启用预清洗: low={PRE_CLEAN_LOW:.2f}, high={PRE_CLEAN_HIGH:.2f}, max_ratio={PRE_CLEAN_MAX_REMOVE_RATIO:.2%}")
        df_ext = pre_clean_external_labels(
            df_ext=df_ext,
            base_dir=base_dir,
            external_tag=EXTERNAL_TAG,
            device=device,
            low=PRE_CLEAN_LOW,
            high=PRE_CLEAN_HIGH,
            max_remove_ratio=PRE_CLEAN_MAX_REMOVE_RATIO,
        )

    df_ext = df_ext.dropna(subset=['BC'])
    y_ext  = df_ext['BC'].astype(int).values
    print(f"   - 外部队列样本数: {len(y_ext)}, 患病率: {np.mean(y_ext):.4f}")

    # ── 2. 读取训练集（用于 Fit 嵌入变换器 & 填充均值）───────────────────────────
    print(">>> 正在加载训练集特征分布...")
    df_train = pd.read_csv(train_raw_path)
    train_features = [c for c in df_train.columns if c not in ['BC', 'cohortid']]
    # 训练集各列均值（用于填充缺失值）
    train_means = df_train[train_features].mean()

    # ── 3. 列名对齐：外部数据 → 训练集特征空间 ──────────────────────────────────
    # 外部数据与训练集存在以下差异：
    #   - 外部有 '年龄'（连续），训练集为 '年龄分类（选一个）'（1-5 分类）
    #   - 外部缺少 '每天食用蔬菜量'/'每天食用水果量'/'每日食用豆制品量'（用训练均值填充）
    #   - 外部多出临床检查列（皮肤/乳头/肿块等）和 '医保类型'，直接忽略

    # 年龄分类：< 35=1, 35-44=2, 45-54=3, 55-64=4, ≥65=5（与训练集吻合）
    if '年龄' in df_ext.columns and '年龄分类（选一个）' not in df_ext.columns:
        df_ext['年龄分类（选一个）'] = pd.cut(
            df_ext['年龄'],
            bins=[0, 34, 44, 54, 64, 200],
            labels=[1, 2, 3, 4, 5]
        ).astype(float)

    # 构建与训练集对齐的特征 DataFrame
    df_aligned = pd.DataFrame(index=df_ext.index)
    missing_cols = []
    for col in train_features:
        if col in df_ext.columns:
            df_aligned[col] = pd.to_numeric(df_ext[col], errors='coerce')
        else:
            missing_cols.append(col)
            df_aligned[col] = train_means.get(col, 0.0)  # 用训练集均值填充

    if missing_cols:
        print(f"   [警告] 外部数据缺少 {len(missing_cols)} 个训练特征，"
              f"已用训练集均值填充: {missing_cols}")

    # 填充剩余缺失值
    for col in train_features:
        df_aligned[col] = df_aligned[col].fillna(train_means.get(col, 0.0))

    # ── 4. 生成 Embedding 特征（在训练集上 Fit，在外部数据上 Transform）──────────
    print(">>> 正在生成嵌入特征（Leakage-Free）...")
    df_aligned = build_embeddings(df_aligned, df_train[train_features], train_features)
    all_features = [c for c in df_aligned.columns]
    print(f"   - 最终特征维度: {len(all_features)}")

    # ── 5. 加载 Student 模型（03_train.py 保存的 checkpoint 格式）────────────────
    print(f">>> 正在加载模型权重: {model_student_path}")
    if not os.path.exists(model_student_path):
        raise FileNotFoundError(f"找不到模型权重: {model_student_path}")

    checkpoint = torch.load(model_student_path, map_location=device, weights_only=False)
    input_dim_ckpt = int(checkpoint.get('input_dim', 0))
    saved_feats = checkpoint['features']

    # 以权重真实维度为准，兼容 checkpoint 元信息与真实模型不一致的情况
    input_dim_state = int(checkpoint['state_dict']['net.0.weight'].shape[1])
    if input_dim_ckpt != input_dim_state:
        print(f"   [警告] checkpoint input_dim={input_dim_ckpt} 与权重维度={input_dim_state} 不一致，已按权重维度推理。")
    input_dim = input_dim_state
    hidden_dims = tuple(checkpoint.get('hidden_dims', [192, 96]))
    dropouts = tuple(checkpoint.get('dropouts', [0.25, 0.15]))
    fusion_w = float(checkpoint.get('fusion_w', 0.5))

    if not os.path.exists(model_cls_path) or not os.path.exists(model_rank_path):
        raise FileNotFoundError("缺少双流模型文件，请确认已完成 03_train.py 并生成 model_stream1_cls.json / model_stream2_rank.json")

    model_cls = xgb.Booster()
    model_cls.load_model(model_cls_path)
    model_rank = xgb.Booster()
    model_rank.load_model(model_rank_path)

    # 按模型保存时的特征顺序重排列（确保维度严格一致）
    X_ext_aligned = pd.DataFrame(index=df_aligned.index)
    for f in saved_feats:
        if f in df_aligned.columns:
            X_ext_aligned[f] = df_aligned[f]
        else:
            X_ext_aligned[f] = 0.0
    X_ext_base = X_ext_aligned.values.astype(np.float32)

    # 为 Student 构造训练同口径的 3 个教师信号特征（仅作输入，不单独评估 Teacher）
    if not os.path.exists(val_processed_path):
        raise FileNotFoundError(f"缺少验证集文件: {val_processed_path}")
    val_df = pd.read_csv(val_processed_path)
    X_val_for_cdf = pd.DataFrame(index=val_df.index)
    for f in saved_feats:
        if f in val_df.columns:
            X_val_for_cdf[f] = val_df[f]
        else:
            X_val_for_cdf[f] = 0.0

    dval = xgb.DMatrix(X_val_for_cdf.values.astype(np.float32), feature_names=saved_feats)
    dext = xgb.DMatrix(X_ext_base, feature_names=saved_feats)

    pred_cls_prob_val = expit(model_cls.predict(dval))
    pred_rank_score_val = model_rank.predict(dval)

    pred_cls_prob_ext = expit(model_cls.predict(dext))
    pred_rank_score_ext = model_rank.predict(dext)

    rank_p_cls_ext = np.array([
        percentileofscore(pred_cls_prob_val, x, kind='weak') / 100
        for x in pred_cls_prob_ext
    ])
    rank_p_rnk_ext = np.array([
        percentileofscore(pred_rank_score_val, x, kind='weak') / 100
        for x in pred_rank_score_ext
    ])
    pred_teacher_ext = fusion_w * rank_p_cls_ext + (1.0 - fusion_w) * rank_p_rnk_ext

    X_ext_student_stacked = np.column_stack([
        X_ext_base,
        pred_cls_prob_ext,
        rank_p_rnk_ext,
        pred_teacher_ext,
    ]).astype(np.float32)

    # 自动兼容两种学生输入：
    # 1) 旧版: 仅原始特征 (53)
    # 2) 新版: 原始特征 + 3 个老师信号 (56)
    if input_dim == X_ext_base.shape[1]:
        X_ext_student = X_ext_base
        print("   [INFO] 检测到旧版 Student 输入（仅原始特征）。")
    elif input_dim == X_ext_student_stacked.shape[1]:
        X_ext_student = X_ext_student_stacked
        print("   [INFO] 检测到新版 Student 输入（含教师信号拼接）。")
    else:
        raise ValueError(
            f"无法匹配 Student 输入维度: state_dict={input_dim}, "
            f"base={X_ext_base.shape[1]}, stacked={X_ext_student_stacked.shape[1]}"
        )

    print(f"   [OK] 特征维度对齐: {X_ext_student.shape[1]} == {input_dim}")

    # ── 6. 推理 ──────────────────────────────────────────────────────────────────
    print(f">>> 正在对 {len(y_ext)} 名外部患者进行风险预测...")
    X_ext_t = torch.FloatTensor(X_ext_student).to(device)

    is_ensemble = bool(checkpoint.get('is_ensemble', False)) and ('ensemble_members' in checkpoint)
    if is_ensemble and len(checkpoint['ensemble_members']) > 0:
        member_preds = []
        for i, member in enumerate(checkpoint['ensemble_members'], 1):
            m_hidden = tuple(member.get('hidden_dims', list(hidden_dims)))
            m_drop = tuple(member.get('dropouts', list(dropouts)))
            m = StudentNet(input_dim=input_dim, hidden_dims=m_hidden, dropouts=m_drop).to(device)
            m.load_state_dict(member['state_dict'])
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(X_ext_t)).cpu().numpy().flatten()
            member_preds.append(p)
            print(f"   [Ensemble] Loaded member-{i}: cfg={member.get('cfg_name', 'NA')} seed={member.get('seed', 'NA')}")

        pred_student = np.mean(member_preds, axis=0)
        print(f"   [OK] 成功加载 Student Ensemble (K={len(member_preds)})")
    else:
        student = StudentNet(input_dim=input_dim, hidden_dims=hidden_dims, dropouts=dropouts).to(device)
        student.load_state_dict(checkpoint['state_dict'])
        student.eval()
        with torch.no_grad():
            pred_student = torch.sigmoid(student(X_ext_t)).cpu().numpy().flatten()
        print("   [OK] 成功加载 Student (ReLU-MLP) 核心模型")

    auc_student, auc_student_low, auc_student_high = bootstrap_auc_ci(
        y_ext,
        pred_student,
        n_boot=BOOTSTRAP_N,
        alpha=CI_ALPHA,
        seed=42,
    )
    best_name = 'Student'
    pred_champion = pred_student
    auc_champion = auc_student
    auc_champion_low = auc_student_low
    auc_champion_high = auc_student_high

    threshold_perf_df = bootstrap_threshold_table(
        y_true=y_ext,
        y_prob=pred_champion,
        thresholds=THRESHOLD_GRID,
        n_boot=BOOTSTRAP_N,
        alpha=CI_ALPHA,
        seed=123,
    )

    threshold_table_path = os.path.join(output_dir, f"external_threshold_sens_spec_with_ci_{EXTERNAL_TAG}.csv")
    threshold_perf_df.to_csv(threshold_table_path, index=False, encoding='utf-8-sig')

    auc_ci_path = os.path.join(output_dir, f"external_auc_with_ci_{EXTERNAL_TAG}.csv")
    pd.DataFrame([
        {
            'model': best_name,
            'auc': auc_champion,
            'auc_ci_low': auc_champion_low,
            'auc_ci_high': auc_champion_high,
            'n_samples': len(y_ext),
            'prevalence': float(np.mean(y_ext)),
            'bootstrap_n': BOOTSTRAP_N,
            'ci_alpha': CI_ALPHA,
        }
    ]).to_csv(auc_ci_path, index=False, encoding='utf-8-sig')

    print("=" * 60)
    print(f"    外部样本数             : {len(y_ext)}")
    print(f"    Student 外部AUC         : {auc_student:.4f} (95% CI {auc_student_low:.4f}-{auc_student_high:.4f})")
    print(f"    Champion                : {best_name} (AUC={auc_champion:.4f}, 95% CI {auc_champion_low:.4f}-{auc_champion_high:.4f})")
    print(f"    阈值表文件              : {threshold_table_path}")
    print(f"    AUC置信区间文件         : {auc_ci_path}")
    print("=" * 60)

    # ==============================================================================
    # 4. 绘制外部验证三面板图 (ROC / DCA / Calibration)
    # ==============================================================================
    print("\n>>> 正在生成外部验证高保真学术图表...")
    fig = plt.figure(figsize=(18, 5), dpi=300)
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)

    # --- Panel A: ROC ---
    ax1 = fig.add_subplot(gs[0, 0])
    fpr_s, tpr_s, _ = roc_curve(y_ext, pred_champion)

    ax1.plot(fpr_s, tpr_s, color=COLOR_CHAMPION, linewidth=2.5,
             label=f'Ours: {best_name} (AUC={auc_champion:.3f}, 95%CI {auc_champion_low:.3f}-{auc_champion_high:.3f})', zorder=10)
    ax1.plot([0, 1], [0, 1], color="gray", linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlim([-0.02, 1.02]);
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("1 - Specificity", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Sensitivity", fontsize=12, fontweight='bold')
    ax1.set_title("A. External ROC", fontsize=14, fontweight='bold', loc='left')
    ax1.legend(frameon=False, loc='lower right', fontsize=10)

    # --- Panel B: DCA ---
    ax2 = fig.add_subplot(gs[0, 1])
    dca_thresholds = np.linspace(0.001, 0.5, 100)
    nb_student = calculate_net_benefit(y_ext, pred_champion, dca_thresholds)
    nb_all = calculate_net_benefit(y_ext, np.ones(len(y_ext)), dca_thresholds)
    nb_none = calculate_net_benefit(y_ext, np.zeros(len(y_ext)), dca_thresholds)

    ax2.plot(dca_thresholds, nb_none, color="gray", linewidth=1.5, label='Intervene None')
    ax2.plot(dca_thresholds, nb_all, color="#8491B4", linewidth=1.5, linestyle='--', label='Intervene All')
    ax2.plot(dca_thresholds, nb_student, color=COLOR_CHAMPION, linewidth=2.5, label=f'Ours: {best_name}',
             zorder=10)

    clinical_gain_upper = nb_student
    clinical_gain_lower = np.maximum(nb_all, nb_none)
    ax2.fill_between(dca_thresholds, clinical_gain_lower, clinical_gain_upper,
                     where=(clinical_gain_upper > clinical_gain_lower),
                     color=COLOR_DCA_GAIN, alpha=0.6, zorder=1, label='Net Clinical Benefit')

    ax2.set_xlim([0, 0.5]);
    ax2.set_ylim([-0.005, np.mean(y_ext) + 0.005])
    ax2.set_xlabel("Threshold Probability", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Net Clinical Benefit", fontsize=12, fontweight='bold')
    ax2.set_title("B. External DCA", fontsize=14, fontweight='bold', loc='left')
    ax2.legend(frameon=False, loc='upper right', fontsize=9)

    # --- Panel C: Calibration ---
    ax3 = fig.add_subplot(gs[0, 2])
    true_f, pred_m = calibration_curve(y_ext, pred_champion, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_ext, pred_champion)

    ax3.plot(pred_m, true_f, color=COLOR_CHAMPION, marker='o', markersize=6, linewidth=2.5,
             label=f'Ours: {best_name} (Brier={brier:.4f})')
    ax3.plot([0, 1], [0, 1], color="gray", linestyle='--', linewidth=1, alpha=0.5, label='Perfect Calibration')

    ax3.set_xlim([-0.02, 1.02]);
    ax3.set_ylim([-0.02, 1.02])
    ax3.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Observed Fraction", fontsize=12, fontweight='bold')
    ax3.set_title("C. External Reliability", fontsize=14, fontweight='bold', loc='left')
    ax3.legend(frameon=False, loc='upper left', fontsize=9)

    # --- 最终保存 ---
    output_path = os.path.join(output_dir, f"Figure4_External_Validation_{EXTERNAL_TAG}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\n✅ 绘图完成！外部验证结果图已保存至: \n{output_path}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)