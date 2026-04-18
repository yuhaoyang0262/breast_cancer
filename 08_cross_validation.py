# 08_cross_validation.py
# 5-Fold Stratified Cross-Validation：完整流水线 (预处理 → 双流 → 蒸馏)
# 关键: 特征工程 (PCA/KMeans Embedding) 在每折内部 fit，杜绝任何泄露
# 输出: cv_results.csv, cv_summary.txt

import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from scipy.special import expit
from scipy.stats import rankdata, percentileofscore
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import config
import os
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 特征工程 (每折重新 fit)
# ==========================================
VIEWS = {
    'Metabolic': ['BMI', '糖尿病', '代谢', '腰臀', '饮食', '肉'],
    'Hormonal':  ['月经', '绝经', '生育', '哺乳', '激素'],
}
N_CLUSTERS = 5


def generate_embeddings(df_train, df_val, df_test, raw_features, seed=42):
    """
    在 df_train 上 fit PCA + KMeans，transform 到 val/test。
    返回增强后的 (df_train, df_val, df_test)，含 Embedding_* 列。
    不修改原始 DataFrame，返回副本。
    """
    tr = df_train.copy()
    va = df_val.copy()
    te = df_test.copy()

    train_means = tr[raw_features].mean()

    # --- PCA Embeddings ---
    for view_name, keywords in VIEWS.items():
        cols = [c for c in raw_features if any(k in c for k in keywords)]
        if len(cols) <= 1:
            continue
        scaler = StandardScaler()
        pca = PCA(n_components=1)

        X_tr_sub = tr[cols].fillna(train_means[cols])
        scaler.fit(X_tr_sub)
        pca.fit(scaler.transform(X_tr_sub))

        for df in [tr, va, te]:
            X_sub = df[cols].fillna(train_means[cols])
            df[f'Embedding_{view_name}'] = pca.transform(scaler.transform(X_sub))

    # --- KMeans Prototype Distances ---
    km_scaler = StandardScaler()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=10)

    X_tr_all = tr[raw_features].fillna(train_means)
    km_scaler.fit(X_tr_all)
    km.fit(km_scaler.transform(X_tr_all))

    for df in [tr, va, te]:
        X_all = df[raw_features].fillna(train_means)
        dists = km.transform(km_scaler.transform(X_all))
        for i in range(N_CLUSTERS):
            df[f'Embedding_Proto_Dist_{i}'] = dists[:, i]

    return tr, va, te


# ==========================================
# 2. 模型组件
# ==========================================
def asymmetric_focal_loss_obj(pred, dtrain):
    y_true = dtrain.get_label()
    p = expit(pred)
    gamma, alpha = 2.0, 4.0
    residual = p - y_true
    mf = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma)
    aw = np.where(y_true == 1, alpha, 1.0)
    grad = aw * mf * residual + aw * gamma * (1 - p) * p * mf * np.log(
        np.clip(p, 1e-6, 1 - 1e-6)) * (2 * y_true - 1)
    hess = np.maximum(aw * mf * p * (1 - p), 1e-6)
    return grad, hess


class StudentNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_ranking_data(X, y, group_size=1000, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    Xs = X.iloc[perm].reset_index(drop=True)
    ys = y.iloc[perm].reset_index(drop=True)
    n = len(X)
    ng = int(np.ceil(n / group_size))
    gc = [group_size] * (ng - 1)
    gc.append(n - sum(gc))
    return Xs, ys, gc


# ==========================================
# 3. 单折完整流水线
# ==========================================
def run_one_fold(X_train, y_train, X_val, y_val, X_test, y_test,
                 seed, device, verbose=True):
    """
    完整单折: Stream1 → Stream2 → Teacher → Student
    返回测试集预测概率和各阶段 AUC。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gpu = device.type == 'cuda'
    dev_str = 'cuda' if gpu else 'cpu'
    features = list(X_train.columns)

    # --- Stream 1: Classification ---
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    m_cls = xgb.train(
        {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7,
         'colsample_bytree': 0.7, 'tree_method': 'hist', 'device': dev_str,
         'disable_default_eval_metric': 1, 'eval_metric': 'auc', 'seed': seed},
        dtrain, num_boost_round=1200, obj=asymmetric_focal_loss_obj,
        evals=[(dval, 'v')], early_stopping_rounds=100, verbose_eval=False,
    )
    cls_val = expit(m_cls.predict(dval))
    cls_test = expit(m_cls.predict(dtest))
    auc_cls = roc_auc_score(y_test, cls_test)

    # --- Stream 2: Ranking ---
    Xr, yr, groups = make_ranking_data(X_train, y_train, seed=seed)
    dr = xgb.DMatrix(Xr, label=yr)
    dr.set_group(groups)
    m_rnk = xgb.train(
        {'objective': 'rank:pairwise', 'eval_metric': 'auc',
         'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
         'tree_method': 'hist', 'device': dev_str, 'seed': seed},
        dr, num_boost_round=800, verbose_eval=False,
    )
    rnk_val = m_rnk.predict(dval)
    rnk_test = m_rnk.predict(dtest)
    auc_rnk = roc_auc_score(y_test, rnk_test)

    # --- Teacher Fusion ---
    rp_cv = rankdata(cls_val) / len(cls_val)
    rp_rv = rankdata(rnk_val) / len(rnk_val)
    rp_ct = np.array([percentileofscore(cls_val, x, kind='weak') / 100 for x in cls_test])
    rp_rt = np.array([percentileofscore(rnk_val, x, kind='weak') / 100 for x in rnk_test])

    best_w, best_auc_w = 0.5, -1
    for w in np.linspace(0, 1, 101):
        a = roc_auc_score(y_val, w * rp_cv + (1 - w) * rp_rv)
        if a > best_auc_w:
            best_auc_w, best_w = a, w
    teacher_test = best_w * rp_ct + (1 - best_w) * rp_rt
    auc_teacher = roc_auc_score(y_test, teacher_test)

    # --- Student Distillation ---
    soft_train = expit(m_cls.predict(xgb.DMatrix(X_train)))
    n_feat = len(features)
    X_tr_t = torch.FloatTensor(X_train.values).to(device)
    y_tr_t = torch.FloatTensor(y_train.values).to(device)
    t_soft = torch.FloatTensor(soft_train).reshape(-1, 1).to(device)
    X_va_t = torch.FloatTensor(X_val.values).to(device)
    X_te_t = torch.FloatTensor(X_test.values).to(device)

    ds = TensorDataset(X_tr_t, y_tr_t, t_soft)
    dl = DataLoader(ds, batch_size=1024, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))

    student = StudentNet(n_feat).to(device)
    opt = optim.AdamW(student.parameters(), lr=0.001, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_va, best_st = 0, None
    for ep in range(150):
        student.train()
        for bx, by, bt in dl:
            opt.zero_grad()
            logits = student(bx)
            loss = 0.2 * bce(logits, by.reshape(-1, 1)) + 0.8 * mse(torch.sigmoid(logits), bt)
            loss.backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            student.eval()
            with torch.no_grad():
                vp = torch.sigmoid(student(X_va_t)).cpu().numpy().flatten()
                va = roc_auc_score(y_val, vp)
                if va > best_va:
                    best_va = va
                    best_st = copy.deepcopy(student.state_dict())

    if best_st:
        student.load_state_dict(best_st)
    student.eval()
    with torch.no_grad():
        stu_val = torch.sigmoid(student(X_va_t)).cpu().numpy().flatten()
        stu_test = torch.sigmoid(student(X_te_t)).cpu().numpy().flatten()
    auc_stu = roc_auc_score(y_test, stu_test)

    # --- Clinical Metrics (Youden on val) ---
    fpr, tpr, ths = roc_curve(y_val, stu_val)
    th = float(ths[np.argmax(tpr - fpr)])
    y_pred = (stu_test >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    brier = brier_score_loss(y_test, stu_test)

    if verbose:
        print(f"      S1={auc_cls:.4f}  S2={auc_rnk:.4f}  "
              f"Teacher={auc_teacher:.4f}  Student={auc_stu:.4f}  "
              f"Sens={sens:.4f}  Spec={spec:.4f}  Brier={brier:.4f}")

    return {
        'AUC_Stream1': round(auc_cls, 4),
        'AUC_Stream2': round(auc_rnk, 4),
        'AUC_Teacher': round(auc_teacher, 4),
        'AUC_Student': round(auc_stu, 4),
        'Best_Val_AUC': round(best_va, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'PPV': round(ppv, 4),
        'NPV': round(npv, 4),
        'Brier': round(brier, 4),
        'Threshold': round(th, 4),
        'Fusion_w': round(best_w, 2),
    }


# ==========================================
# 4. 主流程
# ==========================================
def main():
    K = 5
    SEED = config.SEED

    print("\n" + "=" * 70)
    print(f"   [Step 8] {K}-Fold Stratified Cross-Validation")
    print("   完整流水线: 预处理(PCA+KMeans) → 双流XGB → Teacher → Student")
    print("   预处理在每折内部 fit，杜绝任何泄露")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # --- 加载原始数据 (未经特征工程的) ---
    print("\n>>> Loading raw data (pre-embedding) ...")
    train_raw = pd.read_csv(config.TRAIN_FILE)
    val_raw = pd.read_csv(config.VAL_FILE)
    test_raw = pd.read_csv(config.TEST_FILE)

    # 合并为一个完整数据集
    full = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)
    raw_features = [c for c in full.columns if c not in [config.TARGET_COL, 'cohortid']]
    y_full = full[config.TARGET_COL].values

    print(f"   Total samples: {len(full)}  Features: {len(raw_features)}  "
          f"Positive rate: {y_full.mean():.4f}")

    # --- K-Fold ---
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    all_results = []

    for fold_i, (train_val_idx, test_idx) in enumerate(skf.split(full, y_full), 1):
        t0 = time.time()
        print(f"\n   ── Fold {fold_i}/{K} ──")

        df_train_val = full.iloc[train_val_idx].reset_index(drop=True)
        df_test_fold = full.iloc[test_idx].reset_index(drop=True)

        # 从 train_val 中再分出 val (约 20%)
        y_tv = df_train_val[config.TARGET_COL].values
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold_i)
        tr_idx, va_idx = next(sss.split(df_train_val, y_tv))
        df_train_fold = df_train_val.iloc[tr_idx].reset_index(drop=True)
        df_val_fold = df_train_val.iloc[va_idx].reset_index(drop=True)

        n_tr = len(df_train_fold)
        n_va = len(df_val_fold)
        n_te = len(df_test_fold)
        pos_tr = df_train_fold[config.TARGET_COL].sum()
        pos_te = df_test_fold[config.TARGET_COL].sum()
        print(f"      Train: {n_tr} (pos={int(pos_tr)})  "
              f"Val: {n_va}  Test: {n_te} (pos={int(pos_te)})")

        # === 特征工程 (在 fold 内部 fit) ===
        df_train_e, df_val_e, df_test_e = generate_embeddings(
            df_train_fold, df_val_fold, df_test_fold, raw_features, seed=SEED
        )

        all_features = [c for c in df_train_e.columns
                        if c not in [config.TARGET_COL, 'cohortid']]

        X_tr = df_train_e[all_features]
        y_tr = df_train_e[config.TARGET_COL]
        X_va = df_val_e[all_features]
        y_va = df_val_e[config.TARGET_COL]
        X_te = df_test_e[all_features]
        y_te = df_test_e[config.TARGET_COL]

        print(f"      Features after embedding: {len(all_features)}")

        # === 训练完整流水线 ===
        res = run_one_fold(X_tr, y_tr, X_va, y_va, X_te, y_te,
                           seed=SEED, device=device, verbose=True)
        res['Fold'] = fold_i
        all_results.append(res)

        elapsed = time.time() - t0
        print(f"      Fold {fold_i} done in {elapsed:.1f}s")

    # ==========================================
    # 5. 汇总
    # ==========================================
    df = pd.DataFrame(all_results)
    cols_first = ['Fold']
    df = df[cols_first + [c for c in df.columns if c not in cols_first]]

    print("\n" + "=" * 100)
    print(f"   {K}-Fold Cross-Validation 结果汇总")
    print("=" * 100)
    print(df.to_string(index=False))
    print("-" * 100)

    # 统计量
    metrics = ['AUC_Stream1', 'AUC_Stream2', 'AUC_Teacher', 'AUC_Student',
               'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Brier']
    summary_rows = []
    for col in metrics:
        vals = df[col].values
        m, s = vals.mean(), vals.std()
        print(f"   {col:<18}  {m:.4f} ± {s:.4f}  "
              f"[{vals.min():.4f}, {vals.max():.4f}]")
        summary_rows.append({
            'Metric': col, 'Mean': round(m, 4), 'Std': round(s, 4),
            'Min': round(vals.min(), 4), 'Max': round(vals.max(), 4),
        })

    # --- 论文报告格式 ---
    stu = df['AUC_Student'].values
    sens = df['Sensitivity'].values
    spec = df['Specificity'].values
    bri = df['Brier'].values

    print("\n   ── 论文报告格式 (Methods / Results) ──")
    print(f"   \"In {K}-fold stratified cross-validation, the Student model achieved")
    print(f"    a mean AUROC of {stu.mean():.4f} ± {stu.std():.4f} "
          f"(range: {stu.min():.4f}–{stu.max():.4f}),")
    print(f"    sensitivity {sens.mean():.4f} ± {sens.std():.4f}, "
          f"specificity {spec.mean():.4f} ± {spec.std():.4f},")
    print(f"    and Brier score {bri.mean():.4f} ± {bri.std():.4f}.\"")

    # --- 保存 ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(config.OUTPUT_DIR, "cv_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n   [SAVED] 逐折结果 → {csv_path}")

    summary_csv = os.path.join(config.OUTPUT_DIR, "cv_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"   [SAVED] 汇总统计 → {summary_csv}")

    # 保存一份可读的 summary txt
    txt_path = os.path.join(config.OUTPUT_DIR, "cv_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{K}-Fold Stratified Cross-Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(full)}\n")
        f.write(f"Positive rate: {y_full.mean():.4f}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write("Per-fold results:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Summary (Mean ± Std):\n")
        f.write("-" * 60 + "\n")
        for col in metrics:
            vals = df[col].values
            f.write(f"  {col:<18}  {vals.mean():.4f} ± {vals.std():.4f}  "
                    f"[{vals.min():.4f}, {vals.max():.4f}]\n")
        f.write("\n")
        f.write(f"Paper: Student AUROC = {stu.mean():.4f} ± {stu.std():.4f}\n")
    print(f"   [SAVED] 文本摘要 → {txt_path}")

    print(f"\n   ✅ {K}-Fold 交叉验证完成！\n")


if __name__ == "__main__":
    main()
