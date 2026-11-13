from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .model_profiles import detect_model_profile


class CalibrationDatasetError(ValueError):
    """Raised when the labelled JSONL dataset is incomplete or inconsistent."""


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 1e-7, 1.0 - 1e-7))


def _binary_logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p) - math.log1p(-p)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponents = np.exp(shifted)
    return exponents / exponents.sum()


def _temperature_loss(log_temperature: float, samples: list[tuple[np.ndarray, int]]) -> float:
    temperature = math.exp(log_temperature)
    losses: list[float] = []
    for probabilities, target in samples:
        logits = np.log(np.clip(probabilities, 1e-7, 1.0))
        calibrated = _softmax(logits / temperature)
        losses.append(-math.log(max(float(calibrated[target]), 1e-15)))
    return float(np.mean(losses))


def _labels_temperature_loss(
    log_temperature: float,
    samples: list[tuple[float, bool]],
) -> float:
    temperature = math.exp(log_temperature)
    losses: list[float] = []
    for probability, target in samples:
        scaled_logit = _binary_logit(probability) / temperature
        # Stable binary cross entropy with logits.
        losses.append(float(np.logaddexp(0.0, scaled_logit) - int(target) * scaled_logit))
    return float(np.mean(losses))


def _golden_section_minimize(function, *, lower: float, upper: float) -> float:
    """Minimize a scalar function without adding SciPy as a dependency."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = upper - ratio * (upper - lower)
    x2 = lower + ratio * (upper - lower)
    f1, f2 = function(x1), function(x2)
    for _ in range(120):
        if f1 <= f2:
            upper, x2, f2 = x2, x1, f1
            x1 = upper - ratio * (upper - lower)
            f1 = function(x1)
        else:
            lower, x1, f1 = x1, x2, f2
            x2 = lower + ratio * (upper - lower)
            f2 = function(x2)
    return (lower + upper) / 2.0


def fit_temperature(
    samples: list[tuple[np.ndarray, int]],
    *,
    min_samples: int = 20,
) -> tuple[float, dict[str, float | int | bool]]:
    """Fit one softmax temperature to multiclass probability vectors.

    ``samples`` contains (probabilities_in_model_option_order, gold_index).
    The optimiser minimizes mean negative log likelihood on the supplied
    validation split.  It never sees the training split.
    """
    if len(samples) < min_samples:
        return 1.0, {
            "fitted": False,
            "samples": len(samples),
            "reason": f"need at least {min_samples} labelled examples",
        }
    normalized: list[tuple[np.ndarray, int]] = []
    for probabilities, target in samples:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 1 or p.size < 2 or not np.all(np.isfinite(p)):
            raise CalibrationDatasetError("multiclass calibration expects finite probability vectors")
        if target < 0 or target >= p.size:
            raise CalibrationDatasetError("gold class index is outside the option range")
        p = np.clip(p, 1e-7, 1.0)
        p = p / p.sum()
        normalized.append((p, int(target)))

    baseline = _temperature_loss(0.0, normalized)
    optimum = _golden_section_minimize(
        lambda value: _temperature_loss(value, normalized),
        lower=math.log(0.05),
        upper=math.log(20.0),
    )
    temperature = float(math.exp(optimum))
    fitted = _temperature_loss(optimum, normalized)
    return temperature, {
        "fitted": True,
        "samples": len(normalized),
        "nll_before": baseline,
        "nll_after": fitted,
    }


def fit_labels_temperature(
    samples: list[tuple[float, bool]],
    *,
    min_samples: int = 20,
) -> tuple[float, dict[str, float | int | bool]]:
    """Fit a sigmoid temperature over independent label probabilities."""
    if len(samples) < min_samples:
        return 1.0, {
            "fitted": False,
            "samples": len(samples),
            "reason": f"need at least {min_samples} labelled label-examples",
        }
    baseline = _labels_temperature_loss(0.0, samples)
    optimum = _golden_section_minimize(
        lambda value: _labels_temperature_loss(value, samples),
        lower=math.log(0.05),
        upper=math.log(20.0),
    )
    temperature = float(math.exp(optimum))
    fitted = _labels_temperature_loss(optimum, samples)
    return temperature, {
        "fitted": True,
        "samples": len(samples),
        "nll_before": baseline,
        "nll_after": fitted,
    }


def _expected_index(question_type: str, expected: Any, option_keys: list[str]) -> int:
    if question_type == "noul":
        if isinstance(expected, bool):
            is_true = expected
        elif isinstance(expected, str) and expected.strip().lower() in {"true", "false"}:
            is_true = expected.strip().lower() == "true"
        else:
            raise CalibrationDatasetError("noul expected value must be true/false")
        return option_keys.index("true" if is_true else "false")

    if question_type == "score":
        if isinstance(expected, bool):
            raise CalibrationDatasetError("score expected value must be an integer index, not boolean")
        try:
            index = int(expected)
        except (TypeError, ValueError) as exc:
            raise CalibrationDatasetError("score expected value must be an integer index") from exc
        if str(index) != str(expected) and not isinstance(expected, int):
            raise CalibrationDatasetError("score expected value must be an integer index")
        if not 0 <= index < len(option_keys):
            raise CalibrationDatasetError("score expected index is outside criteria")
        return index

    if not isinstance(expected, str):
        raise CalibrationDatasetError(f"{question_type} expected value must be an option key")
    try:
        return option_keys.index(expected)
    except ValueError as exc:
        raise CalibrationDatasetError(f"unknown expected option key: {expected!r}") from exc


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationDatasetError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise CalibrationDatasetError(f"line {line_number} must contain a JSON object")
            yield line_number, item


def calibrate_model(
    engine,
    *,
    model_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    min_samples_per_type: int = 20,
) -> dict[str, Any]:
    """Run an uncalibrated engine on labelled JSONL and write model sidecar.

    Each non-empty JSONL line contains ``state``, ``questions`` and ``expected``.
    ``expected`` is keyed by question name: option key for choice, true/false for
    noul, zero-based option index for score, and a list of selected keys (or a
    key->bool map) for labels.  Prompt-profile flags must be identical on every
    row so the fitted temperatures are not silently mixed across prompt setups.
    """
    multiclass: dict[str, list[tuple[np.ndarray, int]]] = {
        "choice": [], "binary": [], "score": []
    }
    labels_samples: list[tuple[float, bool]] = []
    profile_signature: tuple[Any, ...] | None = None
    rows = 0

    for line_number, sample in _read_jsonl(dataset_path):
        rows += 1
        if "state" not in sample or not isinstance(sample.get("questions"), dict):
            raise CalibrationDatasetError(
                f"line {line_number}: expected 'state' and a non-empty 'questions' object"
            )
        questions = sample["questions"]
        expected_by_name = sample.get("expected")
        if not isinstance(expected_by_name, dict):
            raise CalibrationDatasetError(f"line {line_number}: 'expected' must be an object")
        signature = (
            sample.get("permutations", 1),
            sample.get("read_twice", False),
            sample.get("questions_before_state", False),
        )
        if profile_signature is None:
            profile_signature = signature
        elif signature != profile_signature:
            raise CalibrationDatasetError(
                "all validation rows must use identical permutations/read_twice/"
                "questions_before_state settings"
            )
        if signature[0] != 1:
            raise CalibrationDatasetError(
                "temperature fitting currently requires permutations=1; "
                "fit on the standard single-option-order profile"
            )

        request = {
            key: sample[key]
            for key in ("state", "questions", "permutations", "read_twice", "questions_before_state")
            if key in sample
        }
        request["temperature"] = 1.0  # collect raw softmax/sigmoid probabilities
        answers = engine.decide(request)["answers"]

        for name, question in questions.items():
            if name not in expected_by_name:
                raise CalibrationDatasetError(f"line {line_number}: missing expected value for {name!r}")
            question_type = str(question.get("type", "")).lower()
            if question_type not in {"choice", "noul", "score", "labels"}:
                continue
            answer = answers[name]
            probability_map = answer["probabilities"]

            if question_type == "labels":
                if isinstance(question.get("criteria"), dict):
                    option_keys = [str(key) for key in question["criteria"].keys()]
                elif isinstance(question.get("criteria"), list):
                    option_keys = [str(index) for index in range(len(question["criteria"]))]
                else:
                    raise CalibrationDatasetError(f"line {line_number}: labels criteria must be an object or array")

                expected_value = expected_by_name[name]
                if isinstance(expected_value, dict):
                    if any(not isinstance(value, bool) for value in expected_value.values()):
                        raise CalibrationDatasetError(
                            f"line {line_number}: labels expected map values must be booleans"
                        )
                    label_targets = {str(key): value for key, value in expected_value.items()}
                elif isinstance(expected_value, list):
                    positive = {str(key) for key in expected_value}
                    label_targets = {key: key in positive for key in option_keys}
                else:
                    raise CalibrationDatasetError(
                        f"line {line_number}: labels expected value must be a list or key->bool object"
                    )
                for key in option_keys:
                    if key not in label_targets:
                        raise CalibrationDatasetError(
                            f"line {line_number}: labels expected value is missing key {key!r}"
                        )
                    probability = float(probability_map[key])
                    if question.get("mode", "head") == "per_option":
                        # per_option uses the binary true/false softmax and the
                        # binary temperature, not the labels-head sigmoid T.
                        multiclass["binary"].append((
                            np.asarray([probability, 1.0 - probability], dtype=np.float64),
                            0 if label_targets[key] else 1,
                        ))
                    else:
                        labels_samples.append((probability, label_targets[key]))
                continue

            if question_type == "score":
                option_keys = [str(index) for index in range(len(question.get("criteria", [])))]
                ordered_probabilities = np.asarray(
                    [probability_map[key] for key in option_keys], dtype=np.float64
                )
            elif question_type == "noul":
                # The model's native order is true, false; API response uses
                # false, true insertion order for readability.
                option_keys = ["true", "false"]
                ordered_probabilities = np.asarray(
                    [probability_map["true"], probability_map["false"]], dtype=np.float64
                )
            else:
                criteria = question.get("criteria")
                if not isinstance(criteria, dict):
                    raise CalibrationDatasetError(f"line {line_number}: choice criteria must be an object")
                option_keys = [str(key) for key in criteria.keys()]
                ordered_probabilities = np.asarray(
                    [probability_map[key] for key in option_keys], dtype=np.float64
                )

            target = _expected_index(question_type, expected_by_name[name], option_keys)
            bucket = "binary" if question_type == "noul" else question_type
            multiclass[bucket].append((ordered_probabilities, target))

    if rows == 0:
        raise CalibrationDatasetError("calibration dataset is empty")

    temperatures: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float | int | bool | str]] = {}
    for question_type, samples in multiclass.items():
        temperature, details = fit_temperature(samples, min_samples=min_samples_per_type)
        temperatures[question_type] = temperature
        diagnostics[question_type] = details
    temperature, details = fit_labels_temperature(labels_samples, min_samples=min_samples_per_type)
    temperatures["labels"] = temperature
    diagnostics["labels"] = details

    model = Path(model_path)
    output = Path(output_path) if output_path is not None else model.with_suffix(".calibration.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = detect_model_profile(model)
    document: dict[str, Any] = {
        "schema_version": 2,
        "method": "temperature_scaling",
        "model_file": model.name,
        "model_profile": profile.as_dict(),
        "temperatures": temperatures,
        "samples": {key: int(value.get("samples", 0)) for key, value in diagnostics.items()},
        "fit": diagnostics,
        "valid_for": {
            "prompt_profile": {
                "permutations": profile_signature[0] if profile_signature else 1,
                "read_twice": profile_signature[1] if profile_signature else False,
                "questions_before_state": profile_signature[2] if profile_signature else False,
            },
            "dataset_rows": rows,
            "minimum_samples_per_type": min_samples_per_type,
        },
        "note": (
            "Fit on the supplied labelled validation set. Keep a separate held-out test set; "
            "do not fit temperatures on production traffic or the training split."
        ),
    }
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document["output"] = str(output)
    return document
