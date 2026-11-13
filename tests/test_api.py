from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from bielik_one_llama_cpp.api import create_app
from bielik_one_llama_cpp.decision import DecisionEngine


class FakeBackend:
    def score_options(self, prompt, option_indices, *, shared_prefix=None):
        logits = np.zeros(max(option_indices) + 1, dtype=np.float64)
        logits[0] = 2.0
        if len(option_indices) > 1:
            logits[1] = 1.0
        return logits[list(option_indices)]


def test_health_and_systemone_api():
    app = create_app(DecisionEngine(FakeBackend(), calibration=False))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["engine"] == "llama-cpp-python"

        response = client.post("/v1/systemone", json={
            "state": "test state",
            "questions": {
                "kind": {
                    "type": "choice",
                    "instructions": "choose",
                    "criteria": {"alpha": "A", "beta": "B"},
                }
            },
        })
        assert response.status_code == 200
        assert response.json()["answers"]["kind"]["choice"] == "alpha"


def test_bad_json_is_structured_422():
    app = create_app(DecisionEngine(FakeBackend(), calibration=False))
    with TestClient(app) as client:
        response = client.post("/v1/systemone", content="not json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_REQUEST"


def test_api_rejects_invalid_request_level_options():
    app = create_app(DecisionEngine(FakeBackend(), calibration=False))
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json={
            "state": "x",
            "permutations": 17,
            "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        })
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"


def test_api_rejects_options_exceeding_model_capacity():
    backend = FakeBackend()
    backend.decision_token_ids = tuple(range(248320, 248836))
    app = create_app(DecisionEngine(backend, calibration=False))
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json={
            "state": "x",
            "questions": {"q": {"type": "choice", "criteria": {str(i): str(i) for i in range(517)}}},
        })
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
    assert "516 options" in response.json()["error"]
