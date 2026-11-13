from __future__ import annotations

import json

import numpy as np

from bielik_one_llama_cpp.calibration_fit import fit_labels_temperature, fit_temperature
from bielik_one_llama_cpp.model_profiles import detect_model_profile, resolve_auto_calibration


def test_detects_11b_gguf_quantization():
    assert detect_model_profile("Bielik-One-11B-v3.1-r1-Q4_K_M.gguf").as_dict() == {
        "family": "11B", "quantization": "Q4_K_M"
    }
    assert detect_model_profile("Bielik-One-11B-v3.1-r1-Q8_0.gguf").as_dict() == {
        "family": "11B", "quantization": "Q8_0"
    }
    assert detect_model_profile("Bielik-One-11B-v3.1-r1-BF16.gguf").as_dict() == {
        "family": "11B", "quantization": "BF16"
    }


def test_detects_4b_gguf_quantization():
    assert detect_model_profile("Bielik-One-4B-r1-Q4_K_M.gguf").as_dict() == {
        "family": "4B", "quantization": "Q4_K_M"
    }
    assert detect_model_profile("Bielik-One-4B-r1-Q8_0.gguf").as_dict() == {
        "family": "4B", "quantization": "Q8_0"
    }
    assert detect_model_profile("Bielik-One-4B-r1-BF16.gguf").as_dict() == {
        "family": "4B", "quantization": "BF16"
    }


def test_auto_calibration_only_uses_official_values_for_11b_bf16(tmp_path):
    selection = resolve_auto_calibration(tmp_path / "Bielik-One-11B-v3.1-r1-BF16.gguf")
    assert selection.enabled
    assert selection.source == "official 11B BF16 temperatures"

    for name in (
        "Bielik-One-11B-v3.1-r1-Q4_K_M.gguf",
        "Bielik-One-11B-v3.1-r1-Q8_0.gguf",
        "Bielik-One-4B-r1-BF16.gguf",
        "Bielik-One-4B-r1-Q4_K_M.gguf",
        "Bielik-One-4B-r1-Q8_0.gguf",
    ):
        selection = resolve_auto_calibration(tmp_path / name)
        assert not selection.enabled, name
        assert selection.path is None


def test_auto_calibration_prefers_model_specific_sidecar(tmp_path):
    model = tmp_path / "Bielik-One-4B-r1-Q4_K_M.gguf"
    sidecar = model.with_suffix(".calibration.json")
    sidecar.write_text(json.dumps({"temperatures": {"choice": 1.2}}), encoding="utf-8")
    selection = resolve_auto_calibration(model)
    assert selection.enabled
    assert selection.path == str(sidecar)


def test_fit_multiclass_temperature_improves_nll_for_overconfident_data():
    # The raw model is systematically too sharp. Temperature scaling should
    # soften the distributions and improve the held-out negative log loss.
    samples = []
    for _ in range(40):
        samples.append((np.asarray([0.9, 0.1]), 0))
        samples.append((np.asarray([0.9, 0.1]), 1))
    temperature, report = fit_temperature(samples, min_samples=20)
    assert 1.0 < temperature <= 20.0
    assert report["fitted"] is True
    assert report["nll_after"] < report["nll_before"]


def test_fit_labels_temperature_improves_nll_for_overconfident_data():
    samples = [(0.9, True), (0.9, False)] * 30
    temperature, report = fit_labels_temperature(samples, min_samples=20)
    assert temperature > 1.0
    assert report["nll_after"] < report["nll_before"]


def test_calibrate_jsonl_creates_model_sidecar(tmp_path):
    from bielik_one_llama_cpp.calibration_fit import calibrate_model

    model = tmp_path / "Bielik-One-4B-r1-Q4_K_M.gguf"
    dataset = tmp_path / "validation.jsonl"
    sidecar = model.with_suffix(".calibration.json")
    row = {
        "state": "Example state",
        "questions": {
            "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "late": {"type": "noul", "criteria": {"true": "Yes", "false": "No"}},
            "level": {"type": "score", "criteria": ["low", "high"]},
            "tags": {"type": "labels", "criteria": {"x": "X", "y": "Y"}},
        },
        "expected": {
            "route": "a",
            "late": True,
            "level": 1,
            "tags": ["x"],
        },
    }
    dataset.write_text((json.dumps(row) + "\n") * 3, encoding="utf-8")

    class FakeEngine:
        def decide(self, request):
            assert request["temperature"] == 1.0
            return {"answers": {
                "route": {"probabilities": {"a": 0.8, "b": 0.2}},
                "late": {"probabilities": {"false": 0.1, "true": 0.9}},
                "level": {"probabilities": {"0": 0.1, "1": 0.9}},
                "tags": {"probabilities": {"x": 0.9, "y": 0.1}},
            }}

    result = calibrate_model(
        FakeEngine(),
        model_path=model,
        dataset_path=dataset,
        min_samples_per_type=2,
    )
    assert sidecar.is_file()
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert loaded["model_file"] == model.name
    assert set(loaded["temperatures"]) == {"choice", "binary", "score", "labels"}
    assert result["output"] == str(sidecar)


def test_calibrator_rejects_non_single_order_profile(tmp_path):
    import pytest
    from bielik_one_llama_cpp.calibration_fit import CalibrationDatasetError, calibrate_model

    model = tmp_path / "Bielik-One-11B-v3.1-r1-Q4_K_M.gguf"
    dataset = tmp_path / "validation.jsonl"
    dataset.write_text(json.dumps({
        "state": "state",
        "permutations": 2,
        "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        "expected": {"q": "a"},
    }) + "\n", encoding="utf-8")

    class NeverCalledEngine:
        def decide(self, request):
            raise AssertionError("profile validation must happen before model inference")

    with pytest.raises(CalibrationDatasetError, match="permutations=1"):
        calibrate_model(
            NeverCalledEngine(),
            model_path=model,
            dataset_path=dataset,
            min_samples_per_type=2,
        )


def test_per_option_labels_are_fitted_with_binary_temperature(tmp_path):
    from bielik_one_llama_cpp.calibration_fit import calibrate_model

    model = tmp_path / "Bielik-One-4B-r1-Q8_0.gguf"
    dataset = tmp_path / "validation.jsonl"
    row = {
        "state": "state",
        "questions": {
            "tags": {
                "type": "labels",
                "mode": "per_option",
                "criteria": {"x": "X", "y": "Y"},
            }
        },
        "expected": {"tags": ["x"]},
    }
    dataset.write_text((json.dumps(row) + "\n") * 12, encoding="utf-8")

    class FakeEngine:
        def decide(self, request):
            return {"answers": {"tags": {"probabilities": {"x": 0.9, "y": 0.1}}}}

    result = calibrate_model(
        FakeEngine(),
        model_path=model,
        dataset_path=dataset,
        min_samples_per_type=20,
    )
    assert result["fit"]["binary"]["fitted"] is True
    assert result["fit"]["labels"]["fitted"] is False


def test_partial_sidecar_defaults_missing_types_to_raw_temperature(tmp_path):
    from bielik_one_llama_cpp.calibration import DEFAULT_TEMPERATURES, load_temperatures

    sidecar = tmp_path / "partial.calibration.json"
    sidecar.write_text(json.dumps({"temperatures": {"choice": 1.4}}), encoding="utf-8")
    temperatures = load_temperatures(str(sidecar))
    assert temperatures["choice"] == 1.4
    assert temperatures["binary"] == 1.0
    assert temperatures["score"] == 1.0
    assert temperatures["labels"] == 1.0
    assert load_temperatures()["choice"] == DEFAULT_TEMPERATURES["choice"]
