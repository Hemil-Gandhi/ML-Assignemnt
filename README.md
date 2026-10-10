# Heart Disease ML Assignment

Two heart-disease prediction systems, built and compared side by side:

| | Tabular (Cardiovascular) | ECG (PTB-XL) |
|---|---|---|
| Input | Patient data (age, BP, cholesterol, lifestyle…) | Raw 12-lead ECG (1000 × 12) |
| Task | Binary: disease yes/no | Multi-label: NORM, MI, STTC, CD, HYP |
| Model | HistGradientBoosting (`tabular_final_model.joblib`) | 1D-CNN, PyTorch (`ptbxl_cnn_baseline.pt`) |
| Test result | Acc 0.737 · F1 0.722 · ROC-AUC 0.804 | Macro ROC-AUC 0.902 · Macro F1 0.704 |

## Project structure

```
src/            app.py (CLI demo), gui.py (Tkinter app), models.py, prediction.py, preprocessing.py
notebooks/      01_tabular.ipynb, 02_ecg.ipynb, 03_results.ipynb  (full story, runnable)
notebooks/archive/  original 15 notebooks, untouched
models/         tabular_final_model.joblib, logistic_regression_tuned.joblib, ptbxl_cnn_baseline.pt
reports/        final metrics, thresholds, comparison tables
data/processed/ train/test splits + preprocessor (source of truth; raw cardio CSV not in repo)
```

## Setup

```bash
pip install -r requirements.txt   # Python 3.12
```

## Run

```bash
python -m src.app   # CLI demo: sample patient + sample ECG record
python -m src.gui   # Tkinter window app
```

Or open `notebooks/01_tabular.ipynb` → `02_ecg.ipynb` → `03_results.ipynb` in order.

## How it was built

1. Explored both datasets (distributions, correlations, sample ECG signals).
2. Preprocessed: stratified 80/20 split, scaling/one-hot (tabular); per-lead normalization, train/val/test splits (ECG).
3. Trained baselines: Logistic Regression, Random Forest, HistGradientBoosting; 3-block Conv1D CNN for ECG.
4. Tuned: GridSearch (tabular) and per-class threshold sweep 0.05–0.95 (ECG, e.g. MI 0.21, HYP 0.23), lifting ECG macro-F1 0.647 → 0.704.
5. Evaluated on untouched test sets; interpreted via coefficients (tabular) and gradient saliency → top ECG leads per disease (MI→V2, HYP→V1).

## Notes

- `tabular_final_model` is identical to the tuned HGB model (verified); the 699MB RandomForest (worst ROC) and duplicate checkpoints were removed during simplification.
- `data/raw/cardiovascular/cardio_data_processed.csv` and `data/raw/ptb-xl/scp_statements.csv` are not in the repo, so notebooks load from `data/processed` / `reports` / `models` and do no retraining.
