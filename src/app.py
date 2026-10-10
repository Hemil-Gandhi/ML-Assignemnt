"""Simplified CLI demo: single tabular model + PTB-XL CNN."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_ptbxl_model, load_ptbxl_thresholds, load_tabular_model
from src.prediction import (
    cardiovascular_majority_vote,
    load_ecg_record,
    predict_cardiovascular,
    predict_ptbxl,
)

SAMPLE_PATIENT = {
    "age_years": 50,
    "gender": 1,
    "height": 165,
    "weight": 70,
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}

SAMPLE_ECG = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "ptb-xl" / "records100" / "00000" / "00001_lr"
)


def main():
    print("Heart Disease ML Application (simplified)")
    print("-" * 40)

    model = load_tabular_model()
    ptbxl_model = load_ptbxl_model()
    thresholds = load_ptbxl_thresholds()
    print("Models loaded.")

    results = predict_cardiovascular({"Final Model": model}, SAMPLE_PATIENT)
    for name, r in results.items():
        print(f"{name}: prediction={r['prediction']}, prob={r['probability']:.4f}")
    print("Overall:", cardiovascular_majority_vote(results))

    if SAMPLE_ECG.with_suffix(".hea").exists():
        signal = load_ecg_record(SAMPLE_ECG)
        print("Signal shape:", signal.shape)
        for cls, r in predict_ptbxl(ptbxl_model, signal, thresholds).items():
            print(f"{cls}: prob={r['probability']:.4f}, positive={r['positive']}")
    else:
        print(f"Sample ECG not found, skipping ECG demo: {SAMPLE_ECG}")


if __name__ == "__main__":
    main()
