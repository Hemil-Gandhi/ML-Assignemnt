import numpy as np
import pandas as pd
import torch
import wfdb

from src.preprocessing import normalize_ecg

PTBXL_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


def predict_cardiovascular(
    models,
    input_data,
):
    """Generate cardiovascular predictions."""

    data = pd.DataFrame([input_data])

    results = {}

    for model_name, model in models.items():

        probability = model.predict_proba(data)[0, 1]

        prediction = int(probability >= 0.5)

        results[model_name] = {
            "prediction": prediction,
            "probability": float(probability),
        }

    return results


def cardiovascular_majority_vote(results):
    """Calculate the overall cardiovascular result."""

    predictions = [result["prediction"] for result in results.values()]

    positive_votes = sum(predictions)

    negative_votes = len(predictions) - positive_votes

    return {
        "prediction": int(positive_votes > negative_votes),
        "positive_votes": positive_votes,
        "negative_votes": negative_votes,
    }


def load_ecg_record(record_path):
    """Load a PTB-XL ECG record."""

    record = wfdb.rdrecord(str(record_path))

    return record.p_signal.astype(np.float32)


def predict_ptbxl(
    model,
    signal,
    thresholds,
):
    """Generate PTB-XL diagnostic predictions."""

    if signal.ndim != 2:
        raise ValueError("ECG signal must be a 2D array.")

    if signal.shape[1] != 12:
        raise ValueError("ECG must contain exactly 12 leads.")

    signal = normalize_ecg(signal)

    # samples × leads → leads × samples
    signal = signal.T

    tensor = torch.tensor(
        signal,
        dtype=torch.float32,
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.sigmoid(logits)

    probabilities = probabilities.squeeze(0).numpy()

    results = {}

    for class_name, probability in zip(
        PTBXL_CLASSES,
        probabilities,
    ):

        threshold = thresholds[class_name]

        results[class_name] = {
            "probability": float(probability),
            "threshold": float(threshold),
            "positive": bool(probability >= threshold),
        }

    return results
