import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise ImportError("shap is required. Please install with: pip install shap") from exc


warnings.filterwarnings("ignore")


EXACT_FEATURE_MAP = {
    "年龄分类（选一个）": "Demographic Age Group",
    "年龄分类": "Demographic Age Group",
    "年龄": "Chronological Age",
    "良性乳腺病二分类": "Benign Breast Disease",
    "腰臀比": "Waist-to-Hip Ratio",
    "每天食用蔬菜量": "Daily Vegetable Intake",
    "是否绝经": "Menopause Status",
    "学历": "Education Level",
    "每天食用水果量": "Daily Fruit Intake",
    "活产次数": "Live Births",
    "绝经年龄": "Age at Menopause",
    "体重": "Weight (kg)",
    "初潮年龄": "Age at Menarche",
    "每日食用豆制品量": "Daily Soy Intake",
    "身高": "Height (cm)",
    "肉类消费": "Meat Consumption",
    "头胎年龄分类": "Age at First Birth",
    "医保类型": "Medical Insurance",
    "BMI": "BMI",
    "流产分类": "History of Abortion",
    "哺乳时间": "Breastfeeding Duration",
    "乳腺癌家族史": "Family History of BC",
    "口服避孕药": "Oral Contraceptives",
    "二手烟": "Secondhand Smoke",
    "饮酒": "Alcohol Consumption",
    "体育锻炼": "Physical Activity",
    "精神压力": "Mental Stress",
    "睡眠质量": "Sleep Quality",
    "夜班": "Night Shift Work",
    "高血压": "Hypertension",
    "糖尿病": "Diabetes",
    "高血脂": "Hyperlipidemia",
    "初次月经": "First Menstruation",
    "婚姻": "Marital Status",
    "人均月收入": "Monthly Income per Capita",
    "主观经济地位": "Subjective Economic Status",
    "主观社会地位": "Subjective Social Status",
    "初次月经年龄分类": "Age at Menarche (Category)",
    "月经是否规律": "Menstrual Regularity",
    "痛经": "Dysmenorrhea",
    "生育次数分类": "Parity Category",
    "哺乳时长分类": "Breastfeeding Duration (Category)",
    "BMI分类(推荐)": "BMI Category",
    "吸烟分类": "Smoking Status",
    "饮酒分类": "Alcohol Consumption Status",
    "高红肉饮食分类": "High Red Meat Diet",
    "维生素": "Vitamin Supplementation",
    "总静态行为时间": "Total Sedentary Time",
    "是否每周进行中高强度体育锻炼": "Weekly Mod-Vig Physical Activity",
    "睡眠时长": "Sleep Duration",
    "激素替代疗法": "Hormone Replacement Therapy",
    "恶性肿瘤家族史": "Family History of Malignancy",
    "二型糖尿病": "Type 2 Diabetes Mellitus",
    "代谢综合征（高血糖+高血压+高血脂）": "Metabolic Syndrome",
    "抑郁得分": "Depression Score",
    "皮肤": "Breast Skin Condition",
    "乳头": "Nipple Condition",
    "乳头溢液": "Nipple Discharge",
    "副乳": "Accessory Breast Tissue",
    "腺体": "Glandular Tissue",
    "肿块": "Breast Mass",
    "腋窝淋巴结": "Axillary Lymph Nodes",
    "锁骨上淋巴结": "Supraclavicular Lymph Nodes",
}


def translate_feature_name(name: str) -> str:
    return EXACT_FEATURE_MAP.get(name, name)


def translate_feature_names(names: List[str]) -> List[str]:
    translated: List[str] = []
    seen: Dict[str, int] = {}
    for name in names:
        base = translate_feature_name(name)
        if base not in seen:
            seen[base] = 1
            translated.append(base)
            continue
        seen[base] += 1
        translated.append(f"{base} ({seen[base]})")
    return translated


def load_runtime_config() -> Dict[str, object]:
    try:
        import config  # type: ignore

        output_dir = getattr(config, "OUTPUT_DIR", "results")
        data_prefix = getattr(config, "PROCESSED_DATA_PREFIX", os.path.join(output_dir, "processed_"))
        target_col = getattr(config, "TARGET_COL", "target")
        seed = int(getattr(config, "SEED", 42))
    except Exception:
        output_dir = "results"
        data_prefix = os.path.join(output_dir, "processed_")
        target_col = "target"
        seed = 42

    return {
        "OUTPUT_DIR": output_dir,
        "PROCESSED_DATA_PREFIX": data_prefix,
        "TARGET_COL": target_col,
        "SEED": seed,
    }


def load_input_tables(data_prefix: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = f"{data_prefix}test.csv"
    prediction_path = os.path.join(output_dir, "final_predictions.csv")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    df_features = pd.read_csv(feature_path)
    df_predictions = pd.read_csv(prediction_path)
    return df_features, df_predictions


def pick_target_column(df_features: pd.DataFrame, df_predictions: pd.DataFrame, configured_target: str) -> str:
    candidates = [configured_target, "target", "BC", "label", "y"]
    for col in candidates:
        if col in df_predictions.columns:
            return col
    for col in candidates:
        if col in df_features.columns:
            return col
    raise ValueError("Cannot find a target column in prediction/features table.")


def pick_prediction_column(df_predictions: pd.DataFrame) -> str:
    preferred = ["pred_student", "proposed", "student", "pred_teacher", "pred_stream1", "pred_stream2"]
    lowered = {c.lower(): c for c in df_predictions.columns}
    for key in preferred:
        if key in lowered:
            return lowered[key]

    prob_like = [c for c in df_predictions.columns if "pred" in c.lower() or "prob" in c.lower()]
    if not prob_like:
        raise ValueError("Cannot find prediction probability column in final_predictions.csv")
    return prob_like[-1]


def build_feature_matrix(df_features: pd.DataFrame, target_col: str) -> pd.DataFrame:
    drop_cols = {target_col, "target", "BC", "cohortid", "id", "ID"}
    feature_cols = [c for c in df_features.columns if c not in drop_cols]
    X = df_features[feature_cols].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.factorize(X[c])[0]
    return X


def compute_surrogate_shap(
    X: pd.DataFrame,
    pred_prob: np.ndarray,
    seed: int,
    n_estimators: int = 420,
    max_depth: int = 5,
    learning_rate: float = 0.03,
) -> Tuple[xgb.XGBRegressor, "shap.TreeExplainer", np.ndarray]:
    surrogate = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=4,
    )
    surrogate.fit(X, pred_prob)
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)
    return surrogate, explainer, np.asarray(shap_values)


def prepare_shap_context() -> Dict[str, object]:
    cfg = load_runtime_config()
    output_dir = str(cfg["OUTPUT_DIR"])
    data_prefix = str(cfg["PROCESSED_DATA_PREFIX"])
    seed = int(cfg["SEED"])

    df_features, df_predictions = load_input_tables(data_prefix, output_dir)
    target_col = pick_target_column(df_features, df_predictions, str(cfg["TARGET_COL"]))
    pred_col = pick_prediction_column(df_predictions)

    X = build_feature_matrix(df_features, target_col=target_col)

    if target_col in df_predictions.columns:
        y_true = df_predictions[target_col].to_numpy()
    else:
        y_true = df_features[target_col].to_numpy()

    pred_prob = df_predictions[pred_col].to_numpy()
    valid_mask = ~pd.isna(pred_prob)
    X_valid = X.loc[valid_mask].reset_index(drop=True)
    y_valid = y_true[valid_mask]
    p_valid = pred_prob[valid_mask]

    surrogate, explainer, shap_values = compute_surrogate_shap(X_valid, p_valid, seed=seed)
    surrogate_pred = surrogate.predict(X_valid)
    fidelity_r2 = r2_score(p_valid, surrogate_pred)
    fidelity_mae = mean_absolute_error(p_valid, surrogate_pred)

    return {
        "cfg": cfg,
        "X": X_valid,
        "y": y_valid,
        "pred": p_valid,
        "target_col": target_col,
        "pred_col": pred_col,
        "surrogate": surrogate,
        "explainer": explainer,
        "shap_values": shap_values,
        "surrogate_pred": surrogate_pred,
        "fidelity_r2": fidelity_r2,
        "fidelity_mae": fidelity_mae,
    }


def top_features_by_mean_abs(shap_values: np.ndarray, feature_names: List[str], top_k: int) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)


def pick_local_case_indices(y_true: np.ndarray, pred_prob: np.ndarray, threshold: float = 0.5) -> Tuple[int, int]:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob)
    pred_label = (pred_prob >= threshold).astype(int)

    malignant_correct = np.where((y_true == 1) & (pred_label == 1))[0]
    benign_correct = np.where((y_true == 0) & (pred_label == 0))[0]

    if len(malignant_correct) == 0:
        malignant_idx = int(np.argmax(pred_prob))
    else:
        malignant_idx = int(malignant_correct[np.argmax(pred_prob[malignant_correct])])

    if len(benign_correct) == 0:
        benign_idx = int(np.argmin(pred_prob))
    else:
        benign_idx = int(benign_correct[np.argmin(pred_prob[benign_correct])])

    return malignant_idx, benign_idx


def continuous_feature_candidates(X: pd.DataFrame, min_unique: int = 10) -> List[str]:
    candidates: List[str] = []
    for col in X.columns:
        s = X[col]
        if np.issubdtype(s.dtype, np.number) and s.nunique(dropna=True) >= min_unique:
            candidates.append(col)
    return candidates
