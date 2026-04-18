# 02_benchmark.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import config
import os
import torch
import warnings

# 过滤警告
warnings.filterwarnings("ignore")

# 尝试导入 CatBoost (替代 TabPFN)
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    print("[WARN] CatBoost not installed. Run `pip install catboost`.")


def main():
    # --- 0. GPU 检测 ---
    use_gpu = torch.cuda.is_available()
    device_name = "cuda" if use_gpu else "cpu"
    # CatBoost 的 task_type
    cb_task_type = "GPU" if use_gpu else "CPU"

    print("\n" + "=" * 60)
    print(f"   [Step 2] Running SOTA Benchmarks (Device: {device_name.upper()})")
    if use_gpu:
        print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # --- 1. 数据加载 ---
    train_path = config.PROCESSED_DATA_PREFIX + "train.csv"
    if not os.path.exists(train_path):
        print(f"[ERROR] Processed data not found. Run 01_preprocess.py first.")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(config.PROCESSED_DATA_PREFIX + "test.csv")

    features = [c for c in train.columns if c not in [config.TARGET_COL, 'cohortid']]
    print(f"   Features used: {len(features)}")
    print(f"   Train Size: {len(train)}, Test Size: {len(test)}")

    X_train, y_train = train[features], train[config.TARGET_COL]
    X_test, y_test = test[features], test[config.TARGET_COL]

    # ==========================================
    # 1. Standard XGBoost (The "Must-Beat" Baseline)
    # ==========================================
    print("\n>>> 1. Running Standard XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost 参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': config.SEED,
        'device': device_name,
        'tree_method': 'hist'
    }

    try:
        bst = xgb.train(params, dtrain, num_boost_round=1000, verbose_eval=False)
        xgb_pred = bst.predict(dtest)
        xgb_auc = roc_auc_score(y_test, xgb_pred)
        print(f"   [RESULT] XGBoost Baseline AUC: {xgb_auc:.4f}")
    except Exception as e:
        print(f"   [ERROR] XGBoost failed: {e}")
        xgb_auc = 0.5

    # ==========================================
    # 2. CatBoost (The "Strong Competitor")
    # ==========================================
    # 替代 TabPFN，CatBoost 是处理类别特征和表格数据的另一大神器
    if CatBoostClassifier:
        print("\n>>> 2. Running CatBoost (The Competitor)...")
        print(f"   Running on {cb_task_type}...")

        # 定义模型
        model_cb = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type=cb_task_type,  # GPU or CPU
            devices='0',  # 使用第一块GPU
            verbose=False,
            random_seed=config.SEED
        )

        try:
            model_cb.fit(X_train, y_train, verbose=False)
            cb_pred = model_cb.predict_proba(X_test)[:, 1]
            cb_auc = roc_auc_score(y_test, cb_pred)

            print(f"   [RESULT] CatBoost Baseline AUC: {cb_auc:.4f}")

            # 对比
            best_baseline = max(xgb_auc, cb_auc)
            print("-" * 50)
            print(f"   [SUMMARY] Best SOTA Baseline to Beat: {best_baseline:.4f}")
            print(f"   (Your Dual-Stream Model in Step 3 must exceed this value)")

        except Exception as e:
            print(f"   [ERROR] CatBoost Failed: {e}")
            # 如果GPU显存不足，回退到CPU
            if 'GPU' in str(e) or 'memory' in str(e).lower():
                print("   -> Retrying on CPU...")
                model_cb = CatBoostClassifier(iterations=500, task_type="CPU", verbose=False)
                model_cb.fit(X_train, y_train)
                cb_auc = roc_auc_score(y_test, model_cb.predict_proba(X_test)[:, 1])
                print(f"   [RESULT] CatBoost (CPU) AUC: {cb_auc:.4f}")

    else:
        print("\n[SKIP] CatBoost not installed.")


if __name__ == "__main__":
    main()