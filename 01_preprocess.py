# 01_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import config  # 导入配置文件
import os
import warnings

warnings.filterwarnings('ignore')


def main():
    print("\n" + "=" * 60)
    print("   [Step 1] Data Preprocessing & Embedding Generation   ")
    print("=" * 60)

    # 1. 加载数据
    print(">>> Loading Raw Data...")
    if not os.path.exists(config.TRAIN_FILE):
        print(f"[ERROR] File not found: {config.TRAIN_FILE}")
        return

    train = pd.read_csv(config.TRAIN_FILE)
    val = pd.read_csv(config.VAL_FILE)
    test = pd.read_csv(config.TEST_FILE)

    # 识别特征列（排除 ID 和 Target）
    features = [c for c in train.columns if c not in [config.TARGET_COL, 'cohortid']]
    print(f"   Original Feature Count: {len(features)}")

    # ==========================================
    # 1b. TABLE 0 — Dataset Statistics
    #     (Methods 可复现性声明：样本量、患病率、缺失率)
    # ==========================================
    print("\n" + "=" * 70)
    print("   TABLE 0: Dataset Statistics (for Methods Section)")
    print("=" * 70)
    print(f"  {'Split':<6} {'N':>7}  {'Pos':>5}  {'Prevalence':>11}  "
          f"{'Neg':>7}  {'Features':>9}  {'MeanMiss%':>10}  {'MaxMiss%':>9}")
    print("-" * 70)
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        n        = len(split_df)
        n_pos    = int(split_df[config.TARGET_COL].sum())
        n_neg    = n - n_pos
        prev_pct = n_pos / n * 100
        miss     = split_df[features].isnull().mean()
        mean_m   = miss.mean() * 100
        max_col  = miss.idxmax()
        max_m    = miss.max() * 100
        print(f"  {split_name:<6} {n:>7,}  {n_pos:>5,}  {prev_pct:>10.2f}%  "
              f"{n_neg:>7,}  {len(features):>9}  {mean_m:>9.2f}%  "
              f"{max_m:>8.2f}% ({max_col[:18]})")
    print("=" * 70)
    print("  [Imputation] All missing values filled with TRAIN-set feature means.")
    print("  [Scaling]    All StandardScaler/PCA/KMeans fitted on TRAIN only;")
    print("               Val and Test are transformed using train-derived parameters.")
    print("  [Seed]       SEED =", config.SEED)
    print("=" * 70)

    # ==========================================
    # 2. 生成 PCA 嵌入 (Metabolic & Hormonal Views)
    # ==========================================
    print("\n>>> Generating PCA Embeddings (Leakage-Free)...")

    # 定义视图关键词
    views = {
        'Metabolic': ['BMI', '糖尿病', '代谢', '腰臀', '饮食', '肉'],
        'Hormonal': ['月经', '绝经', '生育', '哺乳', '激素']
    }

    for name, keywords in views.items():
        # 筛选包含关键词的列
        cols = [c for c in features if any(k in c for k in keywords)]

        if len(cols) > 1:
            print(f"   - Processing View: {name} ({len(cols)} features)")

            # 初始化工具
            scaler = StandardScaler()
            pca = PCA(n_components=1)

            # 1. 仅在训练集上 Fit (用训练集均值填充缺失)
            X_train_sub = train[cols].fillna(train[cols].mean())
            scaler.fit(X_train_sub)
            X_train_scaled = scaler.transform(X_train_sub)
            pca.fit(X_train_scaled)

            # 2. 对所有数据集 Transform
            for df in [train, val, test]:
                # 注意：必须用训练集的均值填充，防止泄露
                X_sub = df[cols].fillna(train[cols].mean())
                X_scaled = scaler.transform(X_sub)

                # 生成新特征
                df[f'Embedding_{name}'] = pca.transform(X_scaled)
        else:
            print(f"   [WARN] View {name} has insufficient features, skipping.")

    # ==========================================
    # 3. 生成 KMeans 嵌入 (Clinical Prototypes)
    # ==========================================
    print("\n>>> Generating KMeans Prototype Embeddings...")
    # 这里的逻辑是：计算每个病人距离 5 个典型临床“原型”的距离

    n_clusters = 5
    kmeans_scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.SEED, n_init=10)

    # 1. 准备数据 (Fit on Train)
    # 使用所有原始特征
    X_train_all = train[features].fillna(train[features].mean())
    kmeans_scaler.fit(X_train_all)
    X_train_sc = kmeans_scaler.transform(X_train_all)

    kmeans.fit(X_train_sc)

    # 2. 转换数据 (Transform on All)
    for df_name, df in [("Train", train), ("Val", val), ("Test", test)]:
        X_all = df[features].fillna(train[features].mean())  # 同样用训练集均值
        X_sc = kmeans_scaler.transform(X_all)

        # 获取距离矩阵 (n_samples, n_clusters)
        dists = kmeans.transform(X_sc)

        # 将距离作为新特征添加
        for i in range(n_clusters):
            df[f'Embedding_Proto_Dist_{i}'] = dists[:, i]

    print(f"   - Added {n_clusters} prototype distance features to all datasets.")

    # ==========================================
    # 4. Table 1 审计 (验证数据分布)
    # ==========================================
    print("\n" + "=" * 60)
    print("   TABLE 1 AUDIT: Check for Distribution Shifts   ")
    print("=" * 60)
    print(f"{'Feature':<30} {'Train Mean':<10} {'Test Mean':<10} {'P-Value':<10}")

    # 检查所有 Embedding 特征的分布
    new_features = [c for c in train.columns if 'Embedding' in c]
    audit_cols = features[:5] + new_features  # 检查几个原始特征 + 新特征

    for col in audit_cols:
        t_v = train[col].dropna()
        test_v = test[col].dropna()
        if len(t_v) > 0 and len(test_v) > 0:
            stat, p = stats.ttest_ind(t_v, test_v, equal_var=False)
            mark = "*" if p < 0.05 else ""
            # 缩短特征名显示
            display_name = (col[:27] + '..') if len(col) > 27 else col
            print(f"{display_name:<30} {t_v.mean():<10.3f} {test_v.mean():<10.3f} {p:<10.3f} {mark}")

    # ==========================================
    # 5. 保存结果
    # ==========================================
    print("\n>>> Saving processed data to disk...")
    train.to_csv(config.PROCESSED_DATA_PREFIX + "train.csv", index=False)
    val.to_csv(config.PROCESSED_DATA_PREFIX + "val.csv", index=False)
    test.to_csv(config.PROCESSED_DATA_PREFIX + "test.csv", index=False)

    # 打印最终特征数量
    final_features = [c for c in train.columns if c not in [config.TARGET_COL, 'cohortid']]
    print(f"   Final Feature Count: {len(final_features)} (Original + {len(new_features)} Embeddings)")
    print("Done.")


if __name__ == "__main__":
    main()