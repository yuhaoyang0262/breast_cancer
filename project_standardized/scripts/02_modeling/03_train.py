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
# 03_train.py (Rigorous Argument Version)
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import config
import os
import warnings
import random

warnings.filterwarnings("ignore")
THRESHOLD_STEP = 0.005


# ==========================================
# 1. 核心组件定义
# ==========================================

# --- A. Asymmetric Focal Loss ---
def asymmetric_focal_loss_obj(pred, dtrain):
    y_true = dtrain.get_label()
    p = expit(pred)
    gamma = 2.0
    alpha = 4.0

    residual = p - y_true
    modulating_factor = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma)
    alpha_weight = np.where(y_true == 1, alpha, 1.0)

    grad = alpha_weight * modulating_factor * residual + \
           alpha_weight * gamma * (1 - p) * p * modulating_factor * np.log(np.clip(p, 1e-6, 1.0 - 1e-6)) * (
                       2 * y_true - 1)

    hess = alpha_weight * modulating_factor * p * (1 - p)
    hess = np.maximum(hess, 1e-6)
    return grad, hess


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


# ==========================================
# 2. 辅助函数
# ==========================================
def make_ranking_data(X, y, group_size=1000):
    perm = np.random.permutation(len(X))
    X_shuff = X.iloc[perm].reset_index(drop=True)
    y_shuff = y.iloc[perm].reset_index(drop=True)

    n_samples = len(X)
    n_groups = int(np.ceil(n_samples / group_size))
    group_counts = [group_size] * (n_groups - 1)
    group_counts.append(n_samples - sum(group_counts))

    return X_shuff, y_shuff, group_counts


def find_best_linear_fusion_weight(p1, p2, y, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    best_w = 0.5
    best_auc = -np.inf
    for w in grid:
        auc = roc_auc_score(y, w * p1 + (1.0 - w) * p2)
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)
    return best_w, float(best_auc)


def build_threshold_grid(step=THRESHOLD_STEP):
    grid = np.arange(0.0, 1.0 + step, step)
    return np.clip(np.round(grid, 6), 0.0, 1.0)


def binary_metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return sens, spec, ppv, npv


def threshold_sweep_table(y_true, y_prob, model_name, split_name, thresholds):
    rows = []
    for th in thresholds:
        sens, spec, ppv, npv = binary_metrics_at_threshold(y_true, y_prob, th)
        rows.append({
            "Model": model_name,
            "Split": split_name,
            "Threshold": float(th),
            "Sensitivity": float(sens),
            "Specificity": float(spec),
            "PPV": float(ppv),
            "NPV": float(npv),
            "Sens_minus_Spec": float(sens - spec),
            "Sens_gt_Spec": int(sens > spec),
        })
    return pd.DataFrame(rows)


def pick_threshold_sens_gt_spec(y_true, y_prob, thresholds):
    df = threshold_sweep_table(y_true, y_prob, model_name="tmp", split_name="tmp", thresholds=thresholds)
    cand = df[df["Sens_gt_Spec"] == 1]
    if len(cand) > 0:
        best_idx = cand["Sens_minus_Spec"].idxmax()
        return float(df.loc[best_idx, "Threshold"]), True
    best_idx = df["Sens_minus_Spec"].idxmax()
    return float(df.loc[best_idx, "Threshold"]), False


def train_student_once(
    X_train_np,
    y_train_np,
    t_soft_np,
    X_val_np,
    y_val_np,
    X_test_np,
    device,
    seed,
    cfg,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X_tr = torch.FloatTensor(X_train_np)
    y_tr = torch.FloatTensor(y_train_np)
    t_soft = torch.FloatTensor(t_soft_np).reshape(-1, 1)
    X_va = torch.FloatTensor(X_val_np).to(device)
    y_va = y_val_np
    X_te = torch.FloatTensor(X_test_np).to(device)

    train_ds = TensorDataset(X_tr, y_tr, t_soft)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)

    student = StudentNet(
        input_dim=X_train_np.shape[1],
        hidden_dims=cfg['hidden_dims'],
        dropouts=cfg['dropouts'],
    ).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    pos_cnt = float(np.sum(y_train_np))
    neg_cnt = float(len(y_train_np) - pos_cnt)
    pos_weight = torch.tensor([neg_cnt / max(pos_cnt, 1.0)], device=device)
    bce_hard = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_soft = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    no_improve = 0

    for _ in range(cfg['epochs']):
        student.train()
        for bx, by, bt in train_dl:
            bx = bx.to(device)
            by = by.to(device)
            bt = bt.to(device)

            optimizer.zero_grad()
            logits = student(bx)
            loss_hard = bce_hard(logits, by.reshape(-1, 1))
            loss_soft = bce_soft(logits, bt)
            loss = cfg['hard_w'] * loss_hard + cfg['soft_w'] * loss_soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=cfg['clip_norm'])
            optimizer.step()
        scheduler.step()

        student.eval()
        with torch.no_grad():
            va_probs = torch.sigmoid(student(X_va)).cpu().numpy().flatten()
            curr_auc = roc_auc_score(y_va, va_probs)

        if curr_auc > best_val_auc + 1e-4:
            best_val_auc = curr_auc
            best_state = copy.deepcopy(student.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg['patience']:
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    student.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(student(X_va)).cpu().numpy().flatten()
        test_pred = torch.sigmoid(student(X_te)).cpu().numpy().flatten()

    return {
        'state_dict': copy.deepcopy(student.state_dict()),
        'best_val_auc': float(roc_auc_score(y_va, val_pred)),
        'val_pred': val_pred,
        'test_pred': test_pred,
        'cfg': cfg,
    }


# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"\n>>> [Step 3] Training Dual-Stream Model (Device: {device})")

    if not os.path.exists(config.PROCESSED_DATA_PREFIX + "train.csv"):
        print("[ERROR] Data not found. Run 01_preprocess.py first.")
        return

    train = pd.read_csv(config.PROCESSED_DATA_PREFIX + "train.csv")
    val = pd.read_csv(config.PROCESSED_DATA_PREFIX + "val.csv")
    test = pd.read_csv(config.PROCESSED_DATA_PREFIX + "test.csv")

    features = [c for c in train.columns if c not in [config.TARGET_COL, 'cohortid']]
    print(f"   Features: {len(features)} (Includes Embeddings)")

    X_train, y_train = train[features], train[config.TARGET_COL]
    X_val, y_val = val[features], val[config.TARGET_COL]
    X_test, y_test = test[features], test[config.TARGET_COL]

    print("\n>>> Stream 1: Training Classification Model...")
    dtrain_cls = xgb.DMatrix(X_train, label=y_train)
    dval_cls = xgb.DMatrix(X_val, label=y_val)
    dtest_cls = xgb.DMatrix(X_test, label=y_test)

    params_cls = {
        'max_depth': 4,
        'learning_rate': 0.03,
        'min_child_weight': 6,
        'gamma': 0.2,
        'reg_lambda': 2.0,
        'reg_alpha': 0.3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu',
        'disable_default_eval_metric': 1,
        'eval_metric': 'auc'
    }

    model_cls = xgb.train(
        params_cls,
        dtrain_cls,
        num_boost_round=900,
        obj=asymmetric_focal_loss_obj,
        evals=[(dval_cls, 'val')],
        early_stopping_rounds=80,
        verbose_eval=False
    )

    pred_cls_prob_val = expit(model_cls.predict(dval_cls))
    auc_cls_val = roc_auc_score(y_val, pred_cls_prob_val)
    print(f"   [Stream 1] Classification AUC (Val): {auc_cls_val:.4f}")

    pred_cls_prob_test = expit(model_cls.predict(dtest_cls))
    auc_cls_test = roc_auc_score(y_test, pred_cls_prob_test)
    print(f"   [Stream 1] Classification AUC (Test): {auc_cls_test:.4f}")

    print("\n>>> Stream 2: Training Ranking Model...")
    X_rank, y_rank, groups = make_ranking_data(X_train, y_train, group_size=1000)
    dtrain_rank = xgb.DMatrix(X_rank, label=y_rank)
    dtrain_rank.set_group(groups)
    dval_rank = xgb.DMatrix(X_val, label=y_val)
    dtest_rank = xgb.DMatrix(X_test, label=y_test)

    params_rank = {
        'objective': 'rank:pairwise',
        'eval_metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 8,
        'gamma': 0.3,
        'reg_lambda': 3.0,
        'reg_alpha': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu',
    }

    model_rank = xgb.train(
        params_rank,
        dtrain_rank,
        num_boost_round=700,
        evals=[(dval_rank, 'val')],
        early_stopping_rounds=60,
        verbose_eval=False
    )

    pred_rank_score_val = model_rank.predict(dval_rank)
    auc_rank_val = roc_auc_score(y_val, pred_rank_score_val)
    print(f"   [Stream 2] Ranking AUC (Val): {auc_rank_val:.4f}")

    pred_rank_score_test = model_rank.predict(dtest_rank)
    auc_rank_test = roc_auc_score(y_test, pred_rank_score_test)
    print(f"   [Stream 2] Ranking AUC (Test): {auc_rank_test:.4f}")

    print("\n>>> Meta-Fusion: Generating Teacher Signals...")
    from scipy.stats import percentileofscore

    rank_p_cls_val = rankdata(pred_cls_prob_val) / len(pred_cls_prob_val)
    rank_p_rnk_val = rankdata(pred_rank_score_val) / len(pred_rank_score_val)

    rank_p_cls_test = np.array([
        percentileofscore(pred_cls_prob_val, x, kind='weak') / 100
        for x in pred_cls_prob_test
    ])
    rank_p_rnk_test = np.array([
        percentileofscore(pred_rank_score_val, x, kind='weak') / 100
        for x in pred_rank_score_test
    ])

    fusion_w, auc_fusion_val = find_best_linear_fusion_weight(
        rank_p_cls_val,
        rank_p_rnk_val,
        y_val,
        grid=np.linspace(0.0, 1.0, 101),
    )
    teacher_pred_test = fusion_w * rank_p_cls_test + (1.0 - fusion_w) * rank_p_rnk_test
    auc_fusion_test = roc_auc_score(y_test, teacher_pred_test)

    print(f"   [Teacher] Fusion weight chosen on Val: w={fusion_w:.2f}")
    print(f"   [Teacher] Dual-Stream Fusion AUC (Val):  {auc_fusion_val:.4f}")
    print(f"   [Teacher] Dual-Stream Fusion AUC (Test): {auc_fusion_test:.4f}")

    if auc_fusion_val > max(auc_cls_val, auc_rank_val):
        print("   [VALIDATION] Synergy holds on Val: Fusion > Single Models.")
    else:
        print("   [NOTE] Synergy not observed on Val (Fusion did not beat best single).")

    print("\n>>> Knowledge Distillation: Training Student Network...")
    dtrain_full = xgb.DMatrix(X_train)
    pred_cls_prob_train = expit(model_cls.predict(dtrain_full))
    pred_rank_score_train = model_rank.predict(dtrain_full)

    rank_p_cls_train = np.array([
        percentileofscore(pred_cls_prob_val, x, kind='weak') / 100
        for x in pred_cls_prob_train
    ])
    rank_p_rnk_train = np.array([
        percentileofscore(pred_rank_score_val, x, kind='weak') / 100
        for x in pred_rank_score_train
    ])
    teacher_train_prob = fusion_w * rank_p_cls_train + (1.0 - fusion_w) * rank_p_rnk_train

    teacher_val_prob = fusion_w * rank_p_cls_val + (1.0 - fusion_w) * rank_p_rnk_val

    X_train_student = np.column_stack([
        X_train.values,
        pred_cls_prob_train,
        rank_p_rnk_train,
        teacher_train_prob,
    ])
    X_val_student = np.column_stack([
        X_val.values,
        pred_cls_prob_val,
        rank_p_rnk_val,
        teacher_val_prob,
    ])
    X_test_student = np.column_stack([
        X_test.values,
        pred_cls_prob_test,
        rank_p_rnk_test,
        teacher_pred_test,
    ])

    search_space = [
        {
            'name': 'S1',
            'hidden_dims': (192, 96),
            'dropouts': (0.25, 0.15),
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'hard_w': 0.70,
            'soft_w': 0.30,
            'batch_size': 512,
            'epochs': 120,
            'patience': 20,
            'clip_norm': 5.0,
        },
        {
            'name': 'S2',
            'hidden_dims': (256, 128),
            'dropouts': (0.20, 0.10),
            'lr': 8e-4,
            'weight_decay': 8e-5,
            'hard_w': 0.65,
            'soft_w': 0.35,
            'batch_size': 384,
            'epochs': 140,
            'patience': 24,
            'clip_norm': 5.0,
        },
        {
            'name': 'S3',
            'hidden_dims': (320, 160),
            'dropouts': (0.18, 0.08),
            'lr': 7e-4,
            'weight_decay': 6e-5,
            'hard_w': 0.60,
            'soft_w': 0.40,
            'batch_size': 320,
            'epochs': 150,
            'patience': 25,
            'clip_norm': 6.0,
        },
        {
            'name': 'S4',
            'hidden_dims': (384, 192),
            'dropouts': (0.16, 0.06),
            'lr': 6e-4,
            'weight_decay': 5e-5,
            'hard_w': 0.58,
            'soft_w': 0.42,
            'batch_size': 256,
            'epochs': 170,
            'patience': 28,
            'clip_norm': 6.0,
        },
        {
            'name': 'S5',
            'hidden_dims': (320, 160, 80),
            'dropouts': (0.15, 0.08, 0.05),
            'lr': 5e-4,
            'weight_decay': 4e-5,
            'hard_w': 0.55,
            'soft_w': 0.45,
            'batch_size': 224,
            'epochs': 190,
            'patience': 32,
            'clip_norm': 6.5,
        },
    ]
    seed_candidates = [config.SEED, config.SEED + 17, config.SEED + 31, config.SEED + 59]

    all_results = []
    for cfg in search_space:
        for s in seed_candidates:
            result = train_student_once(
                X_train_student,
                y_train.values,
                teacher_train_prob,
                X_val_student,
                y_val.values,
                X_test_student,
                device,
                seed=s,
                cfg=cfg,
            )
            print(f"   [Student-Search] {cfg['name']} seed={s} val_auc={result['best_val_auc']:.4f}")
            result['seed'] = s
            all_results.append(result)

    all_results = sorted(all_results, key=lambda x: x['best_val_auc'], reverse=True)
    top_k = min(3, len(all_results))
    top_results = all_results[:top_k]

    for idx, r in enumerate(top_results, 1):
        print(
            f"   [Student-Ensemble] Member-{idx}: cfg={r['cfg']['name']} seed={r['seed']} "
            f"val_auc={r['best_val_auc']:.4f}"
        )

    val_student_pred = np.mean([r['val_pred'] for r in top_results], axis=0)
    final_student_pred = np.mean([r['test_pred'] for r in top_results], axis=0)
    best_student_auc = roc_auc_score(y_val, val_student_pred)
    auc_student_test = roc_auc_score(y_test, final_student_pred)
    print(f"   [Student-Ensemble] TopK={top_k}")
    print(f"   [Student] Best AUC (Val): {best_student_auc:.4f}")
    print(f"   [Student] Final AUC (Test): {auc_student_test:.4f}")

    anchor = top_results[0]
    student = StudentNet(
        input_dim=X_train_student.shape[1],
        hidden_dims=anchor['cfg']['hidden_dims'],
        dropouts=anchor['cfg']['dropouts'],
    ).to(device)
    student.load_state_dict(anchor['state_dict'])
    student.eval()
    with torch.no_grad():
        X_tr_student_t = torch.FloatTensor(X_train_student).to(device)
        train_student_pred = torch.sigmoid(student(X_tr_student_t)).cpu().numpy().flatten()

    print("\n" + "=" * 60)
    print("   RIGOROUSNESS CHECK (The Narrative)")
    print("=" * 60)
    print(f"1. Single Stream 1 (Test): {auc_cls_test:.4f}")
    print(f"2. Single Stream 2 (Test): {auc_rank_test:.4f}")
    print(f"3. Teacher Fusion (Test): {auc_fusion_test:.4f}")
    print(f"4. Student Model  (Test): {auc_student_test:.4f}")
    print("-" * 60)
    synergy_on_val = auc_fusion_val > max(auc_cls_val, auc_rank_val)
    print(f"Synergy on Val (Fusion > best single): {synergy_on_val}")
    print(f"Val AUCs: Stream1={auc_cls_val:.4f}, Stream2={auc_rank_val:.4f}, Fusion={auc_fusion_val:.4f}")
    print("=" * 60)

    print(f"\n>>> Saving models and predictions to {config.OUTPUT_DIR}...")

    model_cls_path = os.path.join(config.OUTPUT_DIR, "model_stream1_cls.json")
    model_cls.save_model(model_cls_path)
    print(f"   [SAVED] Stream 1 (cls)  → {model_cls_path}")

    model_rank_path = os.path.join(config.OUTPUT_DIR, "model_stream2_rank.json")
    model_rank.save_model(model_rank_path)
    print(f"   [SAVED] Stream 2 (rank) → {model_rank_path}")

    student_model_path = os.path.join(config.OUTPUT_DIR, "model_student.pth")
    torch.save({
        'state_dict': student.state_dict(),
        'input_dim': X_train_student.shape[1],
        'features': features,
        'fusion_w': fusion_w,
        'hidden_dims': list(anchor['cfg']['hidden_dims']),
        'dropouts': list(anchor['cfg']['dropouts']),
        'student_cfg_name': anchor['cfg']['name'],
        'student_seed': anchor['seed'],
        'is_ensemble': True,
        'ensemble_members': [
            {
                'state_dict': r['state_dict'],
                'hidden_dims': list(r['cfg']['hidden_dims']),
                'dropouts': list(r['cfg']['dropouts']),
                'cfg_name': r['cfg']['name'],
                'seed': int(r['seed']),
                'val_auc': float(r['best_val_auc']),
            }
            for r in top_results
        ],
    }, student_model_path)
    print(f"   [SAVED] Student model   → {student_model_path}")

    print(f"\n>>> Saving predictions to {config.PREDICTION_FILE}...")

    result_df = pd.DataFrame({
        'target': y_test.values,
        'pred_stream1': pred_cls_prob_test,
        'pred_stream2': rank_p_rnk_test,
        'pred_teacher': teacher_pred_test,
        'pred_student': final_student_pred
    })
    result_df.to_csv(config.PREDICTION_FILE, index=False)

    val_pred_path = os.path.join(config.OUTPUT_DIR, "val_predictions.csv")

    val_df = pd.DataFrame({
        'target': y_val.values,
        'pred_stream1': pred_cls_prob_val,
        'pred_stream2': rank_p_rnk_val,
        'pred_teacher': fusion_w * rank_p_cls_val + (1.0 - fusion_w) * rank_p_rnk_val,
        'pred_student': val_student_pred,
    })
    val_df.to_csv(val_pred_path, index=False)

    train_pred_path = os.path.join(config.OUTPUT_DIR, "train_predictions.csv")
    train_df = pd.DataFrame({
        'target': y_train.values,
        'pred_stream1': pred_cls_prob_train,
        'pred_stream2': rank_p_rnk_train,
        'pred_teacher': teacher_train_prob,
        'pred_student': train_student_pred,
    })
    train_df.to_csv(train_pred_path, index=False)

    thresholds = build_threshold_grid(step=THRESHOLD_STEP)
    split_tables = {
        "Train": train_df,
        "Validation": val_df,
        "Test": result_df,
    }
    model_cols = ["pred_stream1", "pred_stream2", "pred_teacher", "pred_student"]
    model_name_map = {
        "pred_stream1": "Stream 1 (Classification)",
        "pred_stream2": "Stream 2 (Ranking)",
        "pred_teacher": "Teacher (Fusion)",
        "pred_student": "Our Proposed Model",
    }

    all_sweeps = []
    for split_name, df_split in split_tables.items():
        y_true_split = df_split["target"].values
        for col in model_cols:
            all_sweeps.append(
                threshold_sweep_table(
                    y_true_split,
                    df_split[col].values,
                    model_name=model_name_map[col],
                    split_name=split_name,
                    thresholds=thresholds,
                )
            )

    threshold_sweep_df = pd.concat(all_sweeps, axis=0, ignore_index=True)
    threshold_sweep_path = os.path.join(config.OUTPUT_DIR, "threshold_sensitivity_specificity_curve.csv")
    threshold_sweep_df.to_csv(threshold_sweep_path, index=False, encoding="utf-8-sig")

    rec_rows = []
    for col in model_cols:
        model_name = model_name_map[col]
        th_best, has_strict = pick_threshold_sens_gt_spec(
            val_df["target"].values,
            val_df[col].values,
            thresholds,
        )
        for split_name, df_split in split_tables.items():
            sens, spec, ppv, npv = binary_metrics_at_threshold(df_split["target"].values, df_split[col].values, th_best)
            rec_rows.append({
                "Model": model_name,
                "Selected_On": "Validation",
                "Split": split_name,
                "Threshold": float(th_best),
                "Sensitivity": float(sens),
                "Specificity": float(spec),
                "PPV": float(ppv),
                "NPV": float(npv),
                "Sens_gt_Spec": int(sens > spec),
                "Strict_Candidate_Exists_On_Validation": int(has_strict),
            })

    rec_df = pd.DataFrame(rec_rows)
    rec_path = os.path.join(config.OUTPUT_DIR, "threshold_recommendation_sens_gt_spec.csv")
    rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")

    print(f"   [SAVED] test  → {config.PREDICTION_FILE}")
    print(f"   [SAVED] val   → {val_pred_path}")
    print(f"   [SAVED] train → {train_pred_path}")
    print(f"   [SAVED] threshold sweep → {threshold_sweep_path}")
    print(f"   [SAVED] threshold recommendation → {rec_path}")
    print("Done.")


if __name__ == "__main__":
    main()
