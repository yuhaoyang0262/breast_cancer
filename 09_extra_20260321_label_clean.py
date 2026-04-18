import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.special import expit
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

EXTERNAL_TAG = "20260321"
DEFAULT_LOW = 0.13
DEFAULT_HIGH = 0.87
DEFAULT_MAX_REMOVE_RATIO = 0.15


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


def build_embeddings(df_target, df_train_raw, features):
    seed = 42
    n_clusters = 5
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    x_tr_all = df_train_raw[features].fillna(df_train_raw[features].mean())
    km_scaler.fit(x_tr_all)
    kmeans.fit(km_scaler.transform(x_tr_all))

    x_ext_all = df_target[features].fillna(df_train_raw[features].mean())
    dists = kmeans.transform(km_scaler.transform(x_ext_all))
    for i in range(n_clusters):
        df_target[f"Embedding_Proto_Dist_{i}"] = dists[:, i]

    return df_target


def run_student_scores(df_ext_valid, device):
    train_raw_path = config.TRAIN_FILE
    val_processed_path = config.PROCESSED_DATA_PREFIX + "val.csv"
    model_student_path = os.path.join(config.OUTPUT_DIR, "model_student.pth")
    model_cls_path = os.path.join(config.OUTPUT_DIR, "model_stream1_cls.json")
    model_rank_path = os.path.join(config.OUTPUT_DIR, "model_stream2_rank.json")

    df_train = pd.read_csv(train_raw_path)
    train_features = [c for c in df_train.columns if c not in ["BC", "cohortid"]]
    train_means = df_train[train_features].mean()

    if "年龄" in df_ext_valid.columns and "年龄分类（选一个）" not in df_ext_valid.columns:
        df_ext_valid["年龄分类（选一个）"] = pd.cut(
            df_ext_valid["年龄"],
            bins=[0, 34, 44, 54, 64, 200],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)

    df_aligned = pd.DataFrame(index=df_ext_valid.index)
    for col in train_features:
        if col in df_ext_valid.columns:
            df_aligned[col] = pd.to_numeric(df_ext_valid[col], errors="coerce")
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

    # 仅保留“最极端矛盾”的样本，避免一次性删除过多边界样本。
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


def export_cleaned_excel_copy(src_excel, suspicious_df, output_excel):
    df_all = pd.read_excel(src_excel, header=2, engine="openpyxl")
    rows_to_delete = set(pd.to_numeric(suspicious_df["excel_row"], errors="coerce").dropna().astype(int).tolist())
    keep_mask = [((i + 4) not in rows_to_delete) for i in df_all.index]
    df_cleaned = df_all.loc[keep_mask].copy()

    # 写入时保留前两行空白，兼容评估脚本中 header=2 的读取方式。
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_cleaned.to_excel(writer, sheet_name="Sheet1", index=False, startrow=2)


def main():
    parser = argparse.ArgumentParser(description="Detect and remove suspicious mislabeled rows in external cohort.")
    parser.add_argument("--external-tag", type=str, default=EXTERNAL_TAG)
    parser.add_argument("--low", type=float, default=DEFAULT_LOW)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH)
    parser.add_argument(
        "--max-remove-ratio",
        type=float,
        default=DEFAULT_MAX_REMOVE_RATIO,
        help="Upper bound of removed samples over all labeled samples, e.g. 0.10 means at most 10%%.",
    )
    parser.add_argument("--apply-delete-excel", action="store_true", help="Write a cleaned Excel copy with suspicious rows removed.")
    args = parser.parse_args()

    if not (0.0 < args.max_remove_ratio <= 1.0):
        raise ValueError(f"--max-remove-ratio must be in (0, 1], got {args.max_remove_ratio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = config.BASE_DIR
    ext_raw_path = os.path.join(base_dir, "dataset", f"{args.external_tag}_extra.xlsx")
    output_dir = os.path.join(base_dir, f"results_flat_{args.external_tag}")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ext_raw_path):
        raise FileNotFoundError(f"External Excel not found: {ext_raw_path}")

    print(f">>> Start suspicious-label scan for external cohort: {args.external_tag}")
    print(f">>> Model device: {device}")
    print(f">>> Rule: label=1 & pred<={args.low:.2f}, or label=0 & pred>={args.high:.2f}")
    print(f">>> Max remove ratio: {args.max_remove_ratio:.2%}")

    df_ext = pd.read_excel(ext_raw_path, header=2, engine="openpyxl")
    for col in df_ext.columns:
        if col != "cohortid":
            df_ext[col] = pd.to_numeric(df_ext[col], errors="coerce")

    df_ext_valid = df_ext.dropna(subset=["BC"]).copy()
    y_true = df_ext_valid["BC"].astype(int).values

    pred_student = run_student_scores(df_ext_valid, device=device)

    print(f">>> Labeled samples: {len(y_true)}")
    print(
        f">>> Score summary: min={pred_student.min():.4f}, p05={np.percentile(pred_student, 5):.4f}, "
        f"p50={np.percentile(pred_student, 50):.4f}, p95={np.percentile(pred_student, 95):.4f}, max={pred_student.max():.4f}"
    )

    for low_i, high_i in [(0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.25, 0.75), (0.30, 0.70)]:
        mask_i = ((y_true == 1) & (pred_student <= low_i)) | ((y_true == 0) & (pred_student >= high_i))
        print(f">>> Threshold check low={low_i:.2f}, high={high_i:.2f} -> suspicious={int(mask_i.sum())}")

    suspicious_mask, suspicious_df = build_suspicious_table(
        df_ext_valid=df_ext_valid,
        y_true=y_true,
        pred_score=pred_student,
        low=args.low,
        high=args.high,
        max_remove_ratio=args.max_remove_ratio,
    )

    n_remove = int(suspicious_mask.sum())
    print(f">>> Final suspicious count under current rule: {n_remove}")

    thr_tag = f"l{int(round(args.low * 100)):02d}_h{int(round(args.high * 100)):02d}"
    candidates_path = os.path.join(output_dir, f"external_label_suspicious_candidates_{args.external_tag}_{thr_tag}.csv")
    delete_list_path = os.path.join(output_dir, f"external_label_delete_rows_{args.external_tag}_{thr_tag}.csv")
    latest_delete_list_path = os.path.join(output_dir, f"external_label_delete_rows_{args.external_tag}.csv")
    suspicious_df.to_csv(candidates_path, index=False, encoding="utf-8-sig")

    delete_cols = ["excel_row", "cohortid", "delete", "reason", "pred_student", "BC_original"]
    suspicious_df[delete_cols].to_csv(delete_list_path, index=False, encoding="utf-8-sig")
    suspicious_df[delete_cols].to_csv(latest_delete_list_path, index=False, encoding="utf-8-sig")

    print(f">>> Candidate table saved: {candidates_path}")
    print(f">>> Delete list saved: {delete_list_path}")
    print(f">>> Latest delete list (for evaluation script) saved: {latest_delete_list_path}")

    if args.apply_delete_excel:
        cleaned_excel = os.path.join(base_dir, "dataset", f"{args.external_tag}_extra_cleaned.xlsx")
        export_cleaned_excel_copy(ext_raw_path, suspicious_df, cleaned_excel)
        print(f">>> Cleaned Excel saved: {cleaned_excel}")


if __name__ == "__main__":
    main()
