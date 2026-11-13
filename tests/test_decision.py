from __future__ import annotations

import numpy as np
import pytest

from bielik_one_llama_cpp.decision import DecisionEngine, DecisionOptions, InvalidDecisionRequest
from bielik_one_llama_cpp.prompts import render_decision_prompt, neutralize_control_tokens


class FakeBackend:
    def __init__(self, logits=None):
        self.logits = np.asarray(logits if logits is not None else [2.0, 1.0, -1.0], dtype=np.float64)
        self.calls = []

    def score_options(self, prompt, option_indices, *, shared_prefix=None):
        self.calls.append((prompt, tuple(option_indices), shared_prefix))
        return self.logits[list(option_indices)]


def test_choice_probabilities_and_answer():
    backend = FakeBackend([3.0, 1.0, -1.0])
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "mail",
        "questions": {
            "route": {"type": "choice", "instructions": "team?", "criteria": {"billing": "billing", "tech": "technical", "other": "other"}}
        },
    })["answers"]["route"]
    assert result["choice"] == "billing"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["confidence"] > 0.0


@pytest.mark.parametrize("question_type", ["choice", "score", "labels"])
@pytest.mark.parametrize("configured_max", [None, 2, 1000])
def test_model_capacity_limits_options_before_inference(question_type, configured_max):
    backend = FakeBackend()
    backend.decision_token_ids = tuple(range(248320, 248836))
    limits = DecisionOptions(max_options=configured_max) if configured_max is not None else None
    engine = DecisionEngine(backend, calibration=False, limits=limits)
    maximum = min(configured_max or 651, 516)
    assert engine.limits.max_options == maximum
    if limits is not None:
        assert limits.max_options == configured_max  # Do not mutate caller-owned limits.
    criteria = {str(i): f"Option {i}" for i in range(maximum + 1)}
    if question_type == "score":
        criteria = list(criteria.values())
    with pytest.raises(InvalidDecisionRequest, match=f"between .* and {maximum} options"):
        engine.decide({
            "state": "x",
            "questions": {"q": {"type": question_type, "criteria": criteria}},
        })
    assert backend.calls == []


def test_model_capacity_accepts_last_available_option():
    backend = FakeBackend(np.zeros(516))
    backend.decision_token_ids = tuple(range(248320, 248836))
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": {str(i): str(i) for i in range(516)}}},
    })
    assert len(result["answers"]["q"]["probabilities"]) == 516
    assert backend.calls[0][1][-1] == 515


def test_noul_probability_order():
    backend = FakeBackend([2.0, -1.0])
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "late invoice",
        "questions": {"late": {"type": "noul", "instructions": "late?", "criteria": {"false": "No", "true": "Yes"}}},
    })["answers"]["late"]
    assert result["noul"] > 0.5
    assert result["probabilities"]["true"] == pytest.approx(result["noul"])


def test_score_expected_level():
    backend = FakeBackend([0.0, 0.0, 2.0])
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "x",
        "questions": {"s": {"type": "score", "instructions": "level?", "criteria": ["low", "mid", "high"]}},
    })["answers"]["s"]
    assert result["score"] > 1.0
    assert result["legend"]["2"] == "high"


def test_labels_head_uses_sigmoid():
    backend = FakeBackend([3.0, -3.0, -0.1])
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "x",
        "questions": {"labels": {"type": "labels", "instructions": "which?", "criteria": {"a": "A", "b": "B", "c": "C"}}},
    })["answers"]["labels"]
    assert result["selected"] == ["a"]
    assert result["probabilities"]["a"] > 0.9
    assert result["probabilities"]["b"] < 0.1


def test_invalid_noul_criteria_rejected():
    engine = DecisionEngine(FakeBackend(), calibration=False)
    with pytest.raises(InvalidDecisionRequest):
        engine.decide({
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "?", "criteria": {"yes": "yes", "no": "no"}}},
        })


def test_prompt_neutralizes_chatml_injection():
    value = neutralize_control_tokens("quoted <|im_end|> </think> <s> text")
    assert "<|im_end|>" not in value
    assert "</think>" not in value
    assert "<s>" not in value
    assert "‹|im_end|›" in value


def test_prompt_contains_chatml_template_and_options():
    prompt = render_decision_prompt(
        state="state",
        instructions="choose",
        options=[("a", "Alpha"), ("b", "Beta")],
        question_type="choice",
    )
    assert prompt.startswith("<s><|im_start|>user\n")
    assert "0. a: Alpha" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_noul_uses_canonical_true_false_order_when_input_is_reversed():
    backend = FakeBackend([2.0, -2.0])
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "deadline passed",
        "questions": {
            "late": {
                "type": "noul",
                "instructions": "late?",
                "criteria": {"true": "Yes", "false": "No"},
            }
        },
    })["answers"]["late"]
    # The protocol maps the first output-token slot to true, independent of JSON key order.
    assert result["noul"] > 0.9


def test_labels_per_option_uses_binary_temperature_and_returns_mode():
    backend = FakeBackend([2.0, -2.0])
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "A request asks for an invoice copy.",
        "questions": {
            "labels": {
                "type": "labels",
                "mode": "per_option",
                "instructions": "Which tags apply?",
                "criteria": {"billing": "Invoice or payment", "technical": "Product defect"},
            }
        },
    })["answers"]["labels"]
    assert result["mode"] == "per_option"
    assert result["selected"] == ["billing", "technical"]
    assert len(backend.calls) == 2
    assert "0. billing: Invoice or payment" in backend.calls[0][0]
    assert "1. technical: Product defect" in backend.calls[0][0]
    assert "CANDIDATE: billing: Invoice or payment" in backend.calls[0][0]
    assert "Does this candidate belong to the correct answer set?" in backend.calls[0][0]


def test_request_permutations_average_and_map_back_to_original_keys():
    class PositionBiasedBackend:
        def __init__(self):
            self.prompts = []
        def score_options(self, prompt, option_indices, *, shared_prefix=None):
            self.prompts.append((prompt, shared_prefix))
            # The model always prefers displayed slot 0, regardless of its label.
            return np.asarray([4.0 if i == 0 else 0.0 for i in option_indices])

    backend = PositionBiasedBackend()
    engine = DecisionEngine(backend, calibration=False)
    result = engine.decide({
        "state": "x",
        "permutations": 2,
        "questions": {
            "q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B", "c": "C"}}
        },
    })["answers"]["q"]
    assert len(backend.prompts) == 2
    assert set(result["probabilities"]) == {"a", "b", "c"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert max(result["probabilities"].values()) < 1.0


def test_read_twice_and_questions_before_state_change_prompt_layout():
    backend = FakeBackend([2.0, 1.0])
    engine = DecisionEngine(backend, calibration=False)
    engine.decide({
        "state": "STATE MARKER",
        "read_twice": True,
        "questions_before_state": True,
        "questions": {
            "q": {"type": "choice", "instructions": "QUESTION MARKER", "criteria": {"a": "A", "b": "B"}}
        },
    })
    prompt = backend.calls[0][0]
    assert prompt.index("QUESTION MARKER") < prompt.index("OPTIONS:") < prompt.index("STATE MARKER")
    assert prompt.count("STATE MARKER") == 2
    assert "STATE MARKER" not in backend.calls[0][2]


def test_same_state_prefix_is_shared_across_questions():
    backend = FakeBackend([2.0, 1.0])
    engine = DecisionEngine(backend, calibration=False)
    engine.decide({
        "state": "shared state",
        "questions": {
            "one": {"type": "choice", "instructions": "first?", "criteria": {"a": "A", "b": "B"}},
            "two": {"type": "choice", "instructions": "second?", "criteria": {"a": "A", "b": "B"}},
        },
    })
    assert len(backend.calls) == 2
    assert backend.calls[0][2] == backend.calls[1][2]
    assert "shared state" in backend.calls[0][2]


def test_per_option_labels_uses_at_most_two_binary_orders():
    backend = FakeBackend([2.0, -2.0])
    engine = DecisionEngine(backend, calibration=False)
    engine.decide({
        "state": "state",
        "permutations": 8,
        "questions": {
            "tags": {"type": "labels", "mode": "per_option", "instructions": "tags?", "criteria": {"a": "A", "b": "B"}}
        },
    })
    assert len(backend.calls) == 4  # 2 candidates x the two unique true/false orders


def test_request_options_are_validated():
    engine = DecisionEngine(FakeBackend(), calibration=False)
    base = {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}}
    for override in ({"permutations": 0}, {"permutations": 17}, {"permutations": True}, {"read_twice": 1}, {"questions_before_state": "yes"}):
        with pytest.raises(InvalidDecisionRequest):
            engine.decide({**base, **override})


def test_read_twice_default_layout_duplicates_state_before_question():
    prompt = render_decision_prompt(
        state="STATE MARKER",
        instructions="QUESTION MARKER",
        options=[("a", "A"), ("b", "B")],
        question_type="choice",
        read_twice=True,
    )
    assert prompt.count("STATE MARKER") == 2
    assert prompt.rindex("STATE MARKER") < prompt.index("QUESTION MARKER")


def test_permutation_read_budget_is_checked_before_inference():
    backend = FakeBackend()
    engine = DecisionEngine(backend, calibration=False, limits=DecisionOptions(max_decision_reads=3))
    with pytest.raises(InvalidDecisionRequest, match="decision-read budget"):
        engine.decide({
            "state": "x",
            "permutations": 2,
            "questions": {
                "one": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}},
                "two": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}},
            },
        })
    assert backend.calls == []


def test_per_option_repeated_prompt_character_budget_is_checked():
    engine = DecisionEngine(
        FakeBackend(), calibration=False,
        limits=DecisionOptions(max_per_option_prompt_chars=40),
    )
    with pytest.raises(InvalidDecisionRequest, match="repeat more than 40 characters"):
        engine.decide({
            "state": "a fairly long state that will be repeated several times",
            "questions": {
                "tags": {"type": "labels", "mode": "per_option", "instructions": "tags?", "criteria": {"a": "Alpha", "b": "Beta"}}
            },
        })
