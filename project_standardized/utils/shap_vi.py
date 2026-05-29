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
"""
18_plot_unified_shap_figure.py
Top-Tier Journal Quality: Unified Figure with SHAP Summary (A) and Nightingale Rose (B).
Features:
1. Single Figure Object for perfect alignment, DPI, and font consistency.
2. Shared Semantic Palette: Premium Pastel Red-Blue Gradient.
3. 100% Exact 1-to-1 Translation with Medical Terminology.
"""

import os
import sys
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# 0. 环境与配置
# ==============================================================================
try:
    if 'config' in sys.modules:
        config = __import__('config')
    else:
        import config
    # Set default configuration values if not found in the config module
    OUTPUT_DIR = getattr(config, 'OUTPUT_DIR', 'results/')
    DATA_PREFIX = getattr(config, 'PROCESSED_DATA_PREFIX', 'processed/')
    TARGET_COL = getattr(config, 'TARGET_COL', 'target')
    SEED = getattr(config, 'SEED', 42)
except ImportError:
    # Fallback configuration if the config module is missing entirely
    OUTPUT_DIR = "results/"
    DATA_PREFIX = "processed/"
    TARGET_COL = "target"
    SEED = 42

warnings.filterwarnings("ignore")
import xgboost as xgb
import shap

# ==============================================================================
# 🌟 Nature Level Premium, Shared Semantic Palette (Pastel Red-Blue)
# ==============================================================================
# Blue (low / benign) and red (high / malignant), consistent with recent SHAP figures
COLOR_BENIGN_LOW = "#113FD8"
COLOR_CANCER_HIGH = "#E6359C"

# Premium Light Gray: Used for the neutral middle point in the SHAP gradient to keep it looking clean
COLOR_NEUTRAL_MIDDLE = "#F0F0F0"

# Create the continuous colormap for SHAP summary using the pastel gradient
NATURE_CMAP = LinearSegmentedColormap.from_list(
    "Nature_Pastel_SHAP", 
    [COLOR_BENIGN_LOW, COLOR_NEUTRAL_MIDDLE, COLOR_CANCER_HIGH]
)

# ==============================================================================
# 1. 精准翻译字典
# ==============================================================================
EXACT_FEATURE_MAP = {
    '年龄分类（选一个）': 'Demographic Age Group',
    '年龄分类': 'Demographic Age Group',
    '年龄': 'Chronological Age',
    '良性乳腺病二分类': 'Benign Breast Disease',
    '腰臀比': 'Waist-to-Hip Ratio',
    '每天食用蔬菜量': 'Daily Vegetable Intake',
    '是否绝经': 'Menopause Status',
    '学历': 'Education Level',
    '每天食用水果量': 'Daily Fruit Intake',
    '活产次数': 'Live Births',
    '绝经年龄': 'Age at Menopause',
    '体重': 'Weight (kg)',
    '初潮年龄': 'Age at Menarche',
    '每日食用豆制品量': 'Daily Soy Intake',
    '身高': 'Height (cm)',
    '肉类消费': 'Meat Consumption',
    '头胎年龄分类': 'Age at First Birth',
    '医保类型': 'Medical Insurance',
    'BMI': 'BMI',
    '流产分类': 'History of Abortion',
    '哺乳时间': 'Breastfeeding Duration',
    '乳腺癌家族史': 'Family History of BC',
    '口服避孕药': 'Oral Contraceptives',
    '二手烟': 'Secondhand Smoke',
    '饮酒': 'Alcohol Consumption',
    '体育锻炼': 'Physical Activity',
    '精神压力': 'Mental Stress',
    '睡眠质量': 'Sleep Quality',
    '夜班': 'Night Shift Work',
    '高血压': 'Hypertension',
    '糖尿病': 'Diabetes',
    '高血脂': 'Hyperlipidemia',
    '初次月经': 'First Menstruation',
    '婚姻': 'Marital Status',
    '人均月收入': 'Monthly Income per Capita',
    '主观经济地位': 'Subjective Economic Status',
    '主观社会地位': 'Subjective Social Status',
    '初次月经年龄分类': 'Age at Menarche (Category)',
    '月经是否规律': 'Menstrual Regularity',
    '痛经': 'Dysmenorrhea',
    '生育次数分类': 'Parity Category',
    '哺乳时长分类': 'Breastfeeding Duration (Category)',
    'BMI分类(推荐)': 'BMI Category',
    '吸烟分类': 'Smoking Status',
    '饮酒分类': 'Alcohol Consumption Status',
    '高红肉饮食分类': 'High Red Meat Diet',
    '维生素': 'Vitamin Supplementation',
    '总静态行为时间': 'Total Sedentary Time',
    '是否每周进行中高强度体育锻炼': 'Weekly Mod-Vig Physical Activity',
    '睡眠时长': 'Sleep Duration',
    '激素替代疗法': 'Hormone Replacement Therapy',
    '恶性肿瘤家族史': 'Family History of Malignancy',
    '二型糖尿病': 'Type 2 Diabetes Mellitus',
    '代谢综合征（高血糖+高血压+高血脂）': 'Metabolic Syndrome',
    '抑郁得分': 'Depression Score',
    '皮肤': 'Breast Skin Condition',
    '乳头': 'Nipple Condition',
    '乳头溢液': 'Nipple Discharge',
    '副乳': 'Accessory Breast Tissue',
    '腺体': 'Glandular Tissue',
    '肿块': 'Breast Mass',
    '腋窝淋巴结': 'Axillary Lymph Nodes',
    '锁骨上淋巴结': 'Supraclavicular Lymph Nodes'
}

def set_scientific_style():
    # Configure matplotlib to use clean, scientific fonts suitable for publication
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def get_best_data_files():
    # Locate the processed test feature data file
    test_feature_path = os.path.join(DATA_PREFIX + "test.csv")
    if not os.path.exists(test_feature_path): 
        return None, None
        
    # Search for the prediction file with the most columns (indicating the final ensemble/model)
    prediction_candidates = [file_name for file_name in os.listdir(OUTPUT_DIR) if
                      file_name.endswith('.csv') and 'prediction' in file_name.lower() and
                      ('test' in file_name.lower() or 'final' in file_name.lower())]
                      
    best_prediction_file = None
    maximum_columns_found = 0
    
    for file_name in prediction_candidates:
        try:
            full_path = os.path.join(OUTPUT_DIR, file_name)
            # Read only the first row to efficiently check the column count
            dataframe_sample = pd.read_csv(full_path, nrows=1)
            if len(dataframe_sample.columns) > maximum_columns_found and 'target' in dataframe_sample.columns:
                maximum_columns_found = len(dataframe_sample.columns)
                best_prediction_file = full_path
        except: 
            pass
            
    if not best_prediction_file: 
        return None, None
        
    return test_feature_path, best_prediction_file

def draw_radial_text(axis_object, radial_angle, text_content, radial_distance):
    # Convert angle to degrees to determine the correct text rotation and alignment
    degrees = np.degrees(radial_angle % (2 * np.pi))
    if 90 < degrees < 270:
        horizontal_alignment = 'right'
        rotation_angle = degrees - 180
    else:
        horizontal_alignment = 'left'
        rotation_angle = degrees
        
    # Wrap text to ensure long feature names fit within the figure boundaries
    wrapped_text_content = textwrap.fill(text_content, width=20)
    
    # Render the text onto the polar axis
    axis_object.text(radial_angle, radial_distance, wrapped_text_content, 
                     ha=horizontal_alignment, va='center', rotation=rotation_angle,
                     rotation_mode='anchor', fontsize=11, fontweight='bold', color='#333333')

# ==============================================================================
# 主绘图流程
# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print("   Generating Unified Academic Figure (SHAP Summary + Nightingale Rose)")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    feature_file_path, test_prediction_file = get_best_data_files()
    if not feature_file_path or not test_prediction_file:
        print("[ERROR] Cannot find valid data CSVs.")
        return

    # Load feature data and exclude target identifiers
    dataframe_features = pd.read_csv(feature_file_path)
    feature_columns = [col for col in dataframe_features.columns if col not in [TARGET_COL, "target", "cohortid"]]
    X_test_features = dataframe_features[feature_columns]

    # Load predictions and locate the student/proposed model column
    dataframe_test_predictions = pd.read_csv(test_prediction_file)
    student_model_column = next((col for col in dataframe_test_predictions.columns if any(keyword in col.lower() for keyword in ["proposed", "student", "pred_student"])), None)

    if not student_model_column:
        print("[ERROR] Cannot find Student model predictions.")
        return

    # 2. 计算 SHAP
    # Filter out rows with missing predictions
    valid_prediction_mask = ~dataframe_test_predictions[student_model_column].isna()
    X_test_valid_features = X_test_features[valid_prediction_mask].reset_index(drop=True)
    raw_model_predictions = dataframe_test_predictions.loc[valid_prediction_mask, student_model_column].values
    true_target_labels = dataframe_test_predictions.loc[valid_prediction_mask, "target"].values
    total_samples = len(raw_model_predictions)

    print("   -> Computing Exact SHAP values...")
    # Train a surrogate XGBoost model to explain the predictions cleanly
    surrogate_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED)
    surrogate_model.fit(X_test_valid_features, raw_model_predictions)
    tree_explainer = shap.TreeExplainer(surrogate_model)
    calculated_shap_values = tree_explainer.shap_values(X_test_valid_features)

    # Prepare translated features for Panel A
    X_test_english_features = X_test_valid_features.copy().rename(columns=EXACT_FEATURE_MAP)

    # Prepare data for the Nightingale Rose chart in Panel B
    dataframe_shap_values = pd.DataFrame(calculated_shap_values, columns=X_test_valid_features.columns)
    
    # Calculate top features for the Cancer cohort (target == 1)
    shap_cancer_top_features = dataframe_shap_values.loc[true_target_labels == 1].abs().mean(axis=0).sort_values(ascending=False).head(12)
    # Calculate top features for the Benign cohort (target == 0)
    shap_benign_top_features = dataframe_shap_values.loc[true_target_labels == 0].abs().mean(axis=0).sort_values(ascending=False).head(12)
    
    cancer_sample_count = (true_target_labels == 1).sum()
    benign_sample_count = (true_target_labels == 0).sum()
    cancer_proportion = cancer_sample_count / total_samples
    benign_proportion = benign_sample_count / total_samples

    # ==============================================================================
    # 3. 创建超级画板 (Unified Figure)
    # ==============================================================================
    set_scientific_style()
    # Create an ultra-wide canvas to accommodate both standard and polar coordinate systems
    main_figure = plt.figure(figsize=(22, 11))
    grid_specification = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1])

    # --------------------------------------------------------------------------
    # Panel A: SHAP Summary Plot
    # --------------------------------------------------------------------------
    print("   -> Rendering Panel A (SHAP Summary)...")
    axis_panel_a = main_figure.add_subplot(grid_specification[0])
    plt.sca(axis_panel_a)

    # Render the SHAP plot using the custom pastel colormap
    shap.summary_plot(
        calculated_shap_values,
        X_test_english_features,
        max_display=15,
        show=False,
        alpha=0.7, # Slightly increased alpha since colors are lighter
        cmap=NATURE_CMAP,
        plot_size=None
    )

    # Force the figure size back to original dimensions to prevent SHAP from overriding it
    main_figure.set_size_inches(22, 11)

    # Beautify Panel A
    axis_panel_a.spines['top'].set_visible(False)
    axis_panel_a.spines['right'].set_visible(False)
    axis_panel_a.spines['left'].set_visible(False)
    axis_panel_a.tick_params(axis='y', length=0, labelsize=12)
    axis_panel_a.tick_params(axis='x', labelsize=12)

    axis_panel_a.axvline(x=0, color="#4B4B4B", linestyle="--", linewidth=1.2, zorder=0)
    axis_panel_a.xaxis.grid(True, color='gray', linestyle=':', alpha=0.3, zorder=0)
    axis_panel_a.set_xlabel("SHAP Value (Impact on Model Prediction)", fontsize=14, fontweight='bold')

    # Optimize the Colorbar generated by SHAP
    colorbar_axis = main_figure.axes[1] 
    colorbar_axis.set_ylabel('')
    colorbar_axis.tick_params(labelsize=12, length=0)
    colorbar_axis.set_yticks([0.02, 0.98])
    colorbar_axis.set_yticklabels(['Low', 'High'], fontsize=12, fontweight='bold')

    # Add Panel A label
    axis_panel_a.text(-0.35, 1.05, "A", transform=axis_panel_a.transAxes, fontsize=40, fontweight='bold', va='top')
    axis_panel_a.text(-0.25, 1.04, "Overall Feature Contributions", transform=axis_panel_a.transAxes, fontsize=18, fontweight='bold', va='top')

    # --------------------------------------------------------------------------
    # Panel B: Nightingale Rose Chart
    # --------------------------------------------------------------------------
    print("   -> Rendering Panel B (Nightingale Rose)...")
    axis_panel_b = main_figure.add_subplot(grid_specification[1], projection='polar')

    donut_inner_radius = 0.15
    donut_outer_radius = 0.40
    maximum_bar_length = 0.45

    # Render inner donut chart for Cancer cohort
    angles_donut_cancer = np.linspace(-np.pi/2, np.pi/2, 100)
    axis_panel_b.fill_between(angles_donut_cancer, donut_inner_radius, donut_outer_radius, color=COLOR_CANCER_HIGH, alpha=0.9, zorder=2)

    # Render inner donut chart for Benign cohort
    angles_donut_benign = np.linspace(np.pi/2, 3*np.pi/2, 100)
    axis_panel_b.fill_between(angles_donut_benign, donut_inner_radius, donut_outer_radius, color=COLOR_BENIGN_LOW, alpha=0.9, zorder=2)

    # Add text to the center of the donut
    axis_panel_b.text(0, 0, f"TOTAL PATIENTS\nN = {total_samples}", ha='center', va='center', fontsize=12, fontweight='bold')
    radius_text_center = (donut_inner_radius + donut_outer_radius) / 2.0
    
    axis_panel_b.text(0, radius_text_center, f"Cancer Cohort\n{cancer_proportion:.1%}", ha='center', va='center', rotation=-90, color='white', fontsize=11, fontweight='bold', zorder=10)
    axis_panel_b.text(np.pi, radius_text_center, f"Benign Cohort\n{benign_proportion:.1%}", ha='center', va='center', rotation=90, color='white', fontsize=11, fontweight='bold', zorder=10)

    # Add dividing lines inside the donut
    axis_panel_b.plot([np.pi/2, 3*np.pi/2], [donut_outer_radius, donut_outer_radius], color='white', lw=3, zorder=3)
    axis_panel_b.plot([np.pi/2, np.pi/2], [donut_inner_radius, donut_outer_radius], color='white', lw=3, zorder=3)
    axis_panel_b.plot([-np.pi/2, -np.pi/2], [donut_inner_radius, donut_outer_radius], color='white', lw=3, zorder=3)

    # Render outer radial bar chart for Cancer cohort
    angles_cancer_bars = np.linspace(-75 * np.pi/180, 75 * np.pi/180, 12)
    radial_bar_width = (150 * np.pi/180) / 12 * 0.8
    scale_factor_cancer = maximum_bar_length / shap_cancer_top_features.max()
    lengths_cancer_bars = shap_cancer_top_features.values * scale_factor_cancer

    axis_panel_b.bar(angles_cancer_bars, lengths_cancer_bars, width=radial_bar_width, bottom=donut_outer_radius,
           color=COLOR_CANCER_HIGH, alpha=0.9, edgecolor='white', linewidth=1.0, zorder=4)

    for angle, length, feature_key in zip(angles_cancer_bars, lengths_cancer_bars, shap_cancer_top_features.index):
        translated_feature_name = EXACT_FEATURE_MAP.get(feature_key, feature_key)
        draw_radial_text(axis_panel_b, angle, translated_feature_name, donut_outer_radius + length + 0.02)

    # Render outer radial bar chart for Benign cohort
    angles_benign_bars = np.linspace(105 * np.pi/180, 255 * np.pi/180, 12)
    scale_factor_benign = maximum_bar_length / shap_benign_top_features.max()
    lengths_benign_bars = shap_benign_top_features.values * scale_factor_benign

    axis_panel_b.bar(angles_benign_bars, lengths_benign_bars, width=radial_bar_width, bottom=donut_outer_radius,
           color=COLOR_BENIGN_LOW, alpha=0.9, edgecolor='white', linewidth=1.0, zorder=4)

    for angle, length, feature_key in zip(angles_benign_bars, lengths_benign_bars, shap_benign_top_features.index):
        translated_feature_name = EXACT_FEATURE_MAP.get(feature_key, feature_key)
        draw_radial_text(axis_panel_b, angle, translated_feature_name, donut_outer_radius + length + 0.02)

    # Beautify Panel B
    axis_panel_b.set_ylim(0, 1.25)
    axis_panel_b.grid(False)
    axis_panel_b.set_yticklabels([])
    axis_panel_b.set_xticks([])
    axis_panel_b.spines['polar'].set_visible(False)

    # Add background reference circles
    for reference_radius in [0.55, 0.70, 0.85]:
        background_circle = plt.Circle((0,0), reference_radius, transform=axis_panel_b.transData, edgecolor='gray', fill=False, linestyle=':', alpha=0.3, zorder=1)
        axis_panel_b.add_artist(background_circle)

    # Add Panel B label
    axis_panel_b.text(-0.15, 1.05, "B", transform=axis_panel_b.transAxes, fontsize=40, fontweight='bold', va='top')
    axis_panel_b.text(-0.05, 1.04, "Bipartite Cohort-Specific Importance", transform=axis_panel_b.transAxes, fontsize=18, fontweight='bold', va='top')

    # Add global legend for Panel B
    legend_patch_cancer = mpatches.Patch(color=COLOR_CANCER_HIGH, label='Top Features driving Cancer Risk')
    legend_patch_benign = mpatches.Patch(color=COLOR_BENIGN_LOW, label='Top Features driving Benign Diagnosis')
    axis_panel_b.legend(handles=[legend_patch_cancer, legend_patch_benign], loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=2, fontsize=12, frameon=False)

    # --------------------------------------------------------------------------
    # 4. 保存输出
    # --------------------------------------------------------------------------
    final_save_path = os.path.join(OUTPUT_DIR, "Figure4_Unified_Interpretability_NaturePastel.png")

    # Adjust layout spacing to ensure nothing is cut off
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.1, wspace=0.15)

    plt.savefig(final_save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   ✅ 同源渲染、统一调色的 Ultimate Figure 4 已生成: {final_save_path}\n")

if __name__ == "__main__":
    main()