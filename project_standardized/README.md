# Breast Cancer Project - Standardized Layout

This directory is an organized copy of the original flat `project` folder. The original files were left unchanged.

## Layout

- `configs/`: runtime configuration, including data and output paths.
- `scripts/01_data_preparation/`: preprocessing pipeline.
- `scripts/02_modeling/`: model training pipeline.
- `scripts/03_evaluation/`: model evaluation and baseline comparison.
- `scripts/04_experiments/`: ablation, multi-seed, and cross-validation experiments.
- `scripts/05_external_validation/`: external validation and threshold analysis scripts.
- `scripts/06_visualization/`: SHAP, calibration, radar, and other publication figures.
- `utils/`: reusable plotting, SHAP, interaction, and cleanup helpers.
- `legacy/`: duplicate, empty, or fragment files kept for traceability.
- `outputs/`: reserved for generated artifacts from the standardized copy.

## Running Scripts

The copied scripts include a small path bootstrap, so they can be run directly from their new subdirectories. For example:

```powershell
python scripts/01_data_preparation/01_preprocess.py
python scripts/02_modeling/03_train.py
python scripts/03_evaluation/04_evaluate_pro.py
```

Review `configs/config.py` before running if data or output paths need to change.

## Notes

- Duplicate file `01_preprocess - 副本.py` was renamed to `legacy/duplicates/01_preprocess_copy.py`.
- Empty file `02_benchmark.py` was moved to `legacy/empty/02_benchmark.py`.
- Fragment file `report.py` was preserved as `legacy/fragments/report_fragment.txt` because it is not standalone Python.
