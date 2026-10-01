from pathlib import Path

from src.models import (
    load_ptbxl_model,
    load_ptbxl_thresholds,
    load_tabular_models,
)

from src.prediction import (
    load_ecg_record,
    predict_cardiovascular,
    predict_ptbxl,
    cardiovascular_majority_vote,
)


def main():
    print("Heart Disease ML Application")
    print("-" * 40)

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    tabular_models = load_tabular_models()

    ptbxl_model = load_ptbxl_model()

    thresholds = load_ptbxl_thresholds()

    print("All models loaded successfully.")

    # ---------------------------------------------------------
    # Test cardiovascular prediction
    # ---------------------------------------------------------

    patient = {
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

    cardiovascular_results = predict_cardiovascular(
        tabular_models,
        patient,
    )

    print("\nCardiovascular predictions")
    print("-" * 40)

    for model_name, result in cardiovascular_results.items():
        print(
            f"{model_name}: "
            f"prediction={result['prediction']}, "
            f"probability={result['probability']:.4f}"
        )

    # ---------------------------------------------------------
    # Overall cardiovascular result
    # ---------------------------------------------------------

    overall = cardiovascular_majority_vote(cardiovascular_results)

    print("\nOverall cardiovascular result")
    print("-" * 40)

    print(f"Positive votes: " f"{overall['positive_votes']}")

    print(f"Negative votes: " f"{overall['negative_votes']}")

    print(f"Overall prediction: " f"{overall['prediction']}")

    # ---------------------------------------------------------
    # Test PTB-XL ECG prediction
    # ---------------------------------------------------------

    record_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "ptb-xl"
        / "records100"
        / "00000"
        / "00001_lr"
    )

    signal = load_ecg_record(record_path)

    print("\nPTB-XL prediction")
    print("-" * 40)

    print("Signal shape:", signal.shape)

    ecg_results = predict_ptbxl(
        ptbxl_model,
        signal,
        thresholds,
    )

    for class_name, result in ecg_results.items():
        print(
            f"{class_name}: "
            f"probability={result['probability']:.4f}, "
            f"positive={result['positive']}"
        )


if __name__ == "__main__":
    main()
