# config.py
import os

# === 路径设置 ===
# 请修改为你的实际路径
BASE_DIR = r"D:\breast_cancer"
RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed_data_0313")
OUTPUT_DIR = os.path.join(BASE_DIR, "results_flat_0313")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 文件路径
TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train.csv")
VAL_FILE = os.path.join(RAW_DATA_DIR, "val.csv")
TEST_FILE = os.path.join(RAW_DATA_DIR, "test.csv")

# 结果保存路径
PROCESSED_DATA_PREFIX = os.path.join(OUTPUT_DIR, "processed_") # processed_train.csv
PREDICTION_FILE = os.path.join(OUTPUT_DIR, "final_predictions.csv")

# === 参数设置 ===
TARGET_COL = 'BC'  # 目标变量列名
SEED = 42

# 生物学特征分组 (用于交互约束)
INTERACTION_GROUPS = [
    ['BMI', '糖尿病', '绝经', '代谢'],
    ['初次月经', '生育', '哺乳'],
    ['家族史', '良性']
]