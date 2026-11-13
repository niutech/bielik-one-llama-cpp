from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Official calibration.json values published with Bielik-One-11B-v3.1-r1.
# These are fitted for BF16 and the reference prompt profile, not quantized GGUF.
DEFAULT_TEMPERATURES = {
    "choice": 2.106722071909671,
    "binary": 1.8986842419010377,
    "score": 2.732080513508791,
    "labels": 1.0,
}


def load_temperatures(path: str | None = None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_TEMPERATURES)
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    temperatures = raw.get("temperatures", raw)
    # An explicit file is a model-specific profile. Missing types must stay
    # uncalibrated (T=1), never inherit official 11B BF16 values.
    result = {key: 1.0 for key in DEFAULT_TEMPERATURES}
    for key in result:
        if key in temperatures:
            value = float(temperatures[key])
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"Invalid calibration temperature for {key}: {value}")
            result[key] = value
    return result


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("softmax expects a non-empty 1D array")
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not normalize decision logits")
    return exp_values / total


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    # Numerically stable sigmoid.
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out
