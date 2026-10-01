import numpy as np


def normalize_ecg(signal):
    """Normalize each ECG lead independently."""

    signal = signal.astype(np.float32)

    mean = signal.mean(
        axis=0,
        keepdims=True,
    )

    std = signal.std(
        axis=0,
        keepdims=True,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return (signal - mean) / std
