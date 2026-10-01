import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_tuned.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
HGB_MODEL_PATH = MODELS_DIR / "histgradientboosting_tuned.joblib"
FINAL_TABULAR_MODEL_PATH = MODELS_DIR / "tabular_final_model.joblib"

PTBXL_MODEL_PATH = MODELS_DIR / "ptbxl_cnn_baseline.pt"
PTBXL_THRESHOLDS_PATH = REPORTS_DIR / "ptbxl_cnn_thresholds.json"


class ECGCNN(nn.Module):
    """CNN architecture used for the PTB-XL model."""

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def load_tabular_models():
    """Load all cardiovascular prediction models."""

    return {
        "Logistic Regression": joblib.load(LOGISTIC_MODEL_PATH),
        "Random Forest": joblib.load(RANDOM_FOREST_MODEL_PATH),
        "HistGradientBoosting": joblib.load(HGB_MODEL_PATH),
        "Final Model": joblib.load(FINAL_TABULAR_MODEL_PATH),
    }


def load_ptbxl_model():
    """Load the trained PTB-XL CNN."""

    model = ECGCNN(num_classes=5)

    state_dict = torch.load(
        PTBXL_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def load_ptbxl_thresholds():
    """Load class-specific PTB-XL thresholds."""

    with open(
        PTBXL_THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
