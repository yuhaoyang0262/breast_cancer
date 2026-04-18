# 07_multi_seed.py
# 多种子稳定性实验：在 5 个不同随机种子下训练完整流水线
# 报告 AUC 均值 ± 标准差，证明模型性能非"偶然"
# 输出: multi_seed_results.csv

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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import config
import os
import time
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 复用核心组件 (与 06_ablation_study.py 相同)
# ==========================================

def asymmetric_focal_loss_obj(pred, dtrain):
    y_true = dtrain.get_label()
    p = expit(pred)
    gamma, alpha = 2.0, 4.0
    residual = p - y_true
    mf = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma)
    aw = np.where(y_true == 1, alpha, 1.0)
    grad = aw * mf * residual + aw * gamma * (1 - p) * p * mf * np.log(np.clip(p, 1e-6, 1 - 1e-6)) * (2 * y_true - 1)
    hess = np.maximum(aw * mf * p * (1 - p), 1e-6)
    return grad, hess


class StudentNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


def make_ranking_data(X, y, group_size=1000, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    X_shuff = X.iloc[perm].reset_index(drop=True)
    y_shuff = y.iloc[perm].reset_index(drop=True)
    n = len(X)
    n_g = int(np.ceil(n / group_size))
    gc = [group_size] * (n_g - 1)
    gc.append(n - sum(gc))
    return X_shuff, y_shuff, gc


def find_best_fusion_weight(p1, p2, y):
    grid = np.linspace(0.0, 1.0, 101)
    best_w, best_auc = 0.5, -np.inf
    for w in grid:
        auc = roc_auc_score(y, w * p1 + (1 - w) * p2)
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    return best_w


def youden_threshold(y_true, y_prob):
    fpr, tpr, th = roc_curve(y_true, y_prob)
    return float(th[np.argmax(tpr - fpr)])


# ==========================================
# 完整一次训练
# ==========================================
def run_full_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                      seed, device, verbose=True):
    """运行一次完整的双流→融合→蒸馏流水线，返回各阶段 AUC。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_gpu = device.type == 'cuda'
    device_str = 'cuda' if use_gpu else 'cpu'

    # Stream 1: Classification
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    params_cls = {
        'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7,
        'colsample_bytree': 0.7, 'tree_method': 'hist',
        'device': device_str, 'disable_default_eval_metric': 1, 'eval_metric': 'auc',
        'seed': seed,
    }
    m_cls = xgb.train(params_cls, dtrain, num_boost_round=1200, obj=asymmetric_focal_loss_obj,
                       evals=[(dval, 'v')], early_stopping_rounds=100, verbose_eval=False)
    cls_val = expit(m_cls.predict(dval))
    cls_test = expit(m_cls.predict(dtest))
    auc_cls = roc_auc_score(y_test, cls_test)

    # Stream 2: Ranking
    X_rank, y_rank, groups = make_ranking_data(X_train, y_train, seed=seed)
    dr = xgb.DMatrix(X_rank, label=y_rank)
    dr.set_group(groups)
    params_rnk = {
        'objective': 'rank:pairwise', 'eval_metric': 'auc',
        'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
        'tree_method': 'hist', 'device': device_str, 'seed': seed,
    }
    m_rnk = xgb.train(params_rnk, dr, num_boost_round=800, verbose_eval=False)
    rnk_val = m_rnk.predict(dval)
    rnk_test = m_rnk.predict(dtest)
    auc_rnk = roc_auc_score(y_test, rnk_test)

    # Teacher Fusion
    rp_cls_v = rankdata(cls_val) / len(cls_val)
    rp_rnk_v = rankdata(rnk_val) / len(rnk_val)
    rp_cls_t = np.array([percentileofscore(cls_val, x, kind='weak') / 100 for x in cls_test])
    rp_rnk_t = np.array([percentileofscore(rnk_val, x, kind='weak') / 100 for x in rnk_test])
    w = find_best_fusion_weight(rp_cls_v, rp_rnk_v, y_val)
    teacher_test = w * rp_cls_t + (1 - w) * rp_rnk_t
    auc_teacher = roc_auc_score(y_test, teacher_test)

    # Student Distillation
    soft_train = expit(m_cls.predict(xgb.DMatrix(X_train)))
    X_tr_t = torch.FloatTensor(X_train.values).to(device)
    y_tr_t = torch.FloatTensor(y_train.values).to(device)
    t_soft = torch.FloatTensor(soft_train).reshape(-1, 1).to(device)
    X_va_t = torch.FloatTensor(X_val.values).to(device)
    X_te_t = torch.FloatTensor(X_test.values).to(device)

    ds = TensorDataset(X_tr_t, y_tr_t, t_soft)
    dl = DataLoader(ds, batch_size=1024, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))

    student = StudentNet(X_train.shape[1]).to(device)
    opt = optim.AdamW(student.parameters(), lr=0.001, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    best_auc_val, best_state = 0, None
    for ep in range(150):
        student.train()
        for bx, by, bt in dl:
            opt.zero_grad()
            logits = student(bx)
            loss = 0.2 * bce_fn(logits, by.reshape(-1, 1)) + 0.8 * mse_fn(torch.sigmoid(logits), bt)
            loss.backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            student.eval()
            with torch.no_grad():
                vp = torch.sigmoid(student(X_va_t)).cpu().numpy().flatten()
                va = roc_auc_score(y_val, vp)
                if va > best_auc_val:
                    best_auc_val = va
                    best_state = copy.deepcopy(student.state_dict())

    if best_state:
        student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad():
        stu_val = torch.sigmoid(student(X_va_t)).cpu().numpy().flatten()
        stu_test = torch.sigmoid(student(X_te_t)).cpu().numpy().flatten()
    auc_student = roc_auc_score(y_test, stu_test)

    # Youden 阈值 + 临床指标
    th = youden_threshold(y_val, stu_val)
    y_pred = (stu_test >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    if verbose:
        print(f"   Seed={seed:>4}  "
              f"Stream1={auc_cls:.4f}  Stream2={auc_rnk:.4f}  "
              f"Teacher={auc_teacher:.4f}  Student={auc_student:.4f}  "
              f"Sens={sens:.4f}  Spec={spec:.4f}  w={w:.2f}")

    return {
        "seed": seed,
        "AUC_Stream1": round(auc_cls, 4),
        "AUC_Stream2": round(auc_rnk, 4),
        "AUC_Teacher": round(auc_teacher, 4),
        "AUC_Student": round(auc_student, 4),
        "Best_Val_AUC": round(best_auc_val, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "Fusion_w": round(w, 2),
    }


# ==========================================
# 主流程
# ==========================================
def main():
    SEEDS = [42, 123, 2024, 7777, 31415]

    print("\n" + "=" * 70)
    print("   [Step 7] Multi-Seed Stability Experiment (多种子稳定性)")
    print("=" * 70)
    print(f"   Seeds: {SEEDS}")
    print(f"   每个种子运行完整的: 特征工程 → 双流XGB → Teacher → Student 蒸馏\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}\n")

    # 加载数据
    train = pd.read_csv(config.PROCESSED_DATA_PREFIX + "train.csv")
    val = pd.read_csv(config.PROCESSED_DATA_PREFIX + "val.csv")
    test = pd.read_csv(config.PROCESSED_DATA_PREFIX + "test.csv")

    features = [c for c in train.columns if c not in [config.TARGET_COL, 'cohortid']]
    X_train, y_train = train[features], train[config.TARGET_COL]
    X_val, y_val = val[features], val[config.TARGET_COL]
    X_test, y_test = test[features], test[config.TARGET_COL]

    print(f"   Features: {len(features)}  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}\n")

    all_results = []
    for i, seed in enumerate(SEEDS, 1):
        t0 = time.time()
        print(f"   [{i}/{len(SEEDS)}] Running seed={seed} ...", flush=True)
        res = run_full_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                                seed=seed, device=device, verbose=True)
        elapsed = time.time() - t0
        print(f"         Done in {elapsed:.1f}s\n")
        all_results.append(res)

    # 汇总
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 90)
    print("   多种子稳定性结果汇总")
    print("=" * 90)
    print(df.to_string(index=False))
    print("-" * 90)

    for col in ["AUC_Stream1", "AUC_Stream2", "AUC_Teacher", "AUC_Student", "Sensitivity", "Specificity"]:
        vals = df[col].values
        print(f"   {col:<18}  Mean={vals.mean():.4f} ± {vals.std():.4f}  "
              f"[{vals.min():.4f}, {vals.max():.4f}]")

    print("\n   ── 论文报告格式 ──")
    stu = df["AUC_Student"].values
    sens = df["Sensitivity"].values
    spec = df["Specificity"].values
    print(f"   Student AUC: {stu.mean():.4f} ± {stu.std():.4f} (range: {stu.min():.4f}–{stu.max():.4f})")
    print(f"   Sensitivity: {sens.mean():.4f} ± {sens.std():.4f}")
    print(f"   Specificity: {spec.mean():.4f} ± {spec.std():.4f}")

    # 保存
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, "multi_seed_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n   [SAVED] {csv_path}")
    print("\n   ✅ 多种子稳定性实验完成！\n")


if __name__ == "__main__":
    main()
