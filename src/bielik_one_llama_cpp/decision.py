from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from .calibration import load_temperatures, sigmoid, softmax
from .config import DECISION_TOKEN_COUNT
from .prompts import as_text, render_decision_prompt_parts


class LogitBackend(Protocol):
    def score_options(
        self,
        prompt: str,
        option_indices: Sequence[int],
        *,
        shared_prefix: str | None = None,
    ) -> np.ndarray: ...


class InvalidDecisionRequest(ValueError):
    pass


@dataclass(slots=True)
class DecisionOptions:
    max_questions: int = 64
    max_options: int = DECISION_TOKEN_COUNT
    max_state_chars: int = 400_000
    max_question_chars: int = 400_000
    max_decision_reads: int = 1024
    max_per_option_prompt_chars: int = 2_000_000
    max_permutations: int = 16


class DecisionEngine:
    def __init__(
        self,
        backend: LogitBackend,
        *,
        calibration: bool = True,
        calibration_path: str | None = None,
        limits: DecisionOptions | None = None,
    ) -> None:
        self.backend = backend
        self.calibration_enabled = calibration
        self.temperatures = load_temperatures(calibration_path) if calibration else {
            "choice": 1.0, "binary": 1.0, "score": 1.0, "labels": 1.0
        }
        self.limits = limits or DecisionOptions()
        decision_token_ids = getattr(backend, "decision_token_ids", None)
        if decision_token_ids is not None:
            self.limits = replace(
                self.limits,
                max_options=min(self.limits.max_options, len(decision_token_ids)),
            )

    def _temperature(self, question_type: str, override: Any) -> float:
        key = "binary" if question_type == "noul" else question_type
        if override is None:
            return self.temperatures[key]
        if isinstance(override, Mapping):
            value = override.get(question_type, override.get(key, 1.0))
        else:
            value = override
        try:
            temperature = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidDecisionRequest("temperature must be a positive number or a map by type") from exc
        if not math.isfinite(temperature) or temperature <= 0:
            raise InvalidDecisionRequest("temperature values must be finite and greater than zero")
        return temperature

    @staticmethod
    def _items(criteria: Any, *, question_type: str) -> list[tuple[str, Any]]:
        if question_type == "score":
            if not isinstance(criteria, list):
                raise InvalidDecisionRequest("score criteria must be an ordered array")
            return [(str(index), value) for index, value in enumerate(criteria)]
        if question_type == "noul" and criteria is None:
            return [("true", "Yes"), ("false", "No")]
        if not isinstance(criteria, Mapping):
            raise InvalidDecisionRequest(f"{question_type} criteria must be an object")
        items = [(str(key), value) for key, value in criteria.items()]
        if question_type == "noul":
            # Binary decision slots have a canonical order, independent of JSON key order.
            by_key = {key.lower(): (key.lower(), value) for key, value in items}
            if set(by_key) == {"true", "false"}:
                return [by_key["true"], by_key["false"]]
        return items

    @staticmethod
    def _validate_options(options: list[tuple[str, Any]], question_type: str, max_options: int) -> None:
        min_count = 2 if question_type in {"noul", "score"} else 1
        if not min_count <= len(options) <= max_options:
            raise InvalidDecisionRequest(
                f"{question_type} expects between {min_count} and {max_options} options"
            )
        keys = [key for key, _ in options]
        if any(not key.strip() or "\n" in key or "\r" in key for key in keys):
            raise InvalidDecisionRequest("option keys must be non-empty single-line strings")
        if len({key.casefold() for key in keys}) != len(keys):
            raise InvalidDecisionRequest("option keys must be unique (case-insensitive)")
        if question_type == "noul" and set(k.lower() for k in keys) != {"true", "false"}:
            raise InvalidDecisionRequest("noul criteria keys must be exactly 'true' and 'false'")

    @staticmethod
    def _effective_order_count(option_count: int, requested: int) -> int:
        """Do not spend reads repeating an ordering when all unique orders are exhausted."""
        possible = 1
        for factor in range(2, option_count + 1):
            possible *= factor
            if possible >= requested:
                return requested
        return min(requested, possible)

    @classmethod
    def _permutation_orders(
        cls,
        option_count: int,
        requested: int,
        *,
        seed_material: str,
    ) -> list[tuple[int, ...]]:
        """Build reproducible, unique option orders; the original order is first."""
        target = cls._effective_order_count(option_count, requested)
        identity = tuple(range(option_count))
        if target <= 1:
            return [identity]
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        seen = {identity}
        orders = [identity]
        max_attempts = max(100, target * 100)
        attempts = 0
        while len(orders) < target and attempts < max_attempts:
            candidate = list(range(option_count))
            rng.shuffle(candidate)
            item = tuple(candidate)
            if item not in seen:
                seen.add(item)
                orders.append(item)
            attempts += 1
        # For small option sets, deterministic enumeration guarantees completion.
        if len(orders) < target:
            for item in itertools.permutations(range(option_count)):
                if item not in seen:
                    seen.add(item)
                    orders.append(item)
                    if len(orders) == target:
                        break
        return orders

    def _read(
        self,
        state: Any,
        instructions: Any,
        question_type: str,
        options: list[tuple[str, Any]],
        *,
        candidate: tuple[str, Any] | None = None,
        context_options: list[tuple[str, Any]] | None = None,
        read_twice: bool = False,
        questions_before_state: bool = False,
    ) -> np.ndarray:
        prefix, suffix = render_decision_prompt_parts(
            state=state,
            instructions=instructions,
            options=options,
            question_type=question_type,
            candidate=candidate,
            context_options=context_options,
            read_twice=read_twice,
            questions_before_state=questions_before_state,
        )
        indices = list(range(len(options))) if candidate is None else [0, 1]
        return self.backend.score_options(prefix + suffix, indices, shared_prefix=prefix)

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        order: Sequence[int],
        option_count: int,
    ) -> np.ndarray:
        aligned = np.empty(option_count, dtype=np.float64)
        for displayed_index, original_index in enumerate(order):
            aligned[original_index] = probabilities[displayed_index]
        return aligned

    @staticmethod
    def _confidence_choice(probabilities: np.ndarray) -> float:
        count = len(probabilities)
        if count <= 1:
            return 1.0
        value = (float(np.max(probabilities)) - 1.0 / count) / (1.0 - 1.0 / count)
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _confidence_score(probabilities: np.ndarray) -> float:
        count = len(probabilities)
        if count <= 1:
            return 1.0
        mode = int(np.argmax(probabilities))
        distances = np.abs(np.arange(count, dtype=np.float64) - mode)
        distance = float(np.dot(probabilities, distances))
        midpoint = (count - 1) / 2.0
        uniform_distance = float(np.mean(np.abs(np.arange(count, dtype=np.float64) - midpoint)))
        if uniform_distance <= 0:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - distance / uniform_distance)))

    @staticmethod
    def _validate_request_flags(request: Mapping[str, Any], limits: DecisionOptions) -> tuple[int, bool, bool]:
        permutations = request.get("permutations", 1)
        if isinstance(permutations, bool) or not isinstance(permutations, int):
            raise InvalidDecisionRequest("permutations must be an integer between 1 and 16")
        if not 1 <= permutations <= limits.max_permutations:
            raise InvalidDecisionRequest(f"permutations must be between 1 and {limits.max_permutations}")
        read_twice = request.get("read_twice", False)
        questions_before_state = request.get("questions_before_state", False)
        if not isinstance(read_twice, bool):
            raise InvalidDecisionRequest("read_twice must be a boolean")
        if not isinstance(questions_before_state, bool):
            raise InvalidDecisionRequest("questions_before_state must be a boolean")
        return permutations, read_twice, questions_before_state

    def _answer(
        self,
        state: Any,
        name: str,
        question: Mapping[str, Any],
        override_temperature: Any,
        *,
        permutations: int,
        read_twice: bool,
        questions_before_state: bool,
    ) -> dict[str, Any]:
        question_type = str(question.get("type", "")).lower()
        if question_type not in {"choice", "noul", "score", "labels"}:
            raise InvalidDecisionRequest(f"Question {name!r}: unsupported type {question_type!r}")
        instructions = question.get("instructions", "")
        options = self._items(question.get("criteria"), question_type=question_type)
        self._validate_options(options, question_type, self.limits.max_options)

        if question_type != "labels" and "mode" in question:
            raise InvalidDecisionRequest("mode is accepted only for labels questions")

        seed_material = f"{name}\0{question_type}\0" + "\0".join(key for key, _ in options)

        if question_type == "labels":
            mode = question.get("mode", "head")
            if mode not in {"head", "per_option"}:
                raise InvalidDecisionRequest("labels mode must be 'head' or 'per_option'")
            if mode == "head":
                orders = self._permutation_orders(len(options), permutations, seed_material=seed_material)
                temperature = self._temperature("labels", override_temperature)
                probability_reads: list[np.ndarray] = []
                for order in orders:
                    displayed_options = [options[index] for index in order]
                    raw = self._read(
                        state,
                        instructions,
                        "labels",
                        displayed_options,
                        read_twice=read_twice,
                        questions_before_state=questions_before_state,
                    )
                    displayed_probs = sigmoid(raw / temperature)
                    probability_reads.append(self._align_probabilities(displayed_probs, order, len(options)))
                probs = np.mean(np.stack(probability_reads), axis=0)
            else:
                # Each label is a binary decision. There are only two distinct
                # orders of its true/false answer options, even when N > 2.
                binary_orders = self._permutation_orders(2, permutations, seed_material=seed_material + "\0binary")
                context_orders = self._permutation_orders(
                    len(options), min(permutations, 2), seed_material=seed_material + "\0context"
                )
                temperature = self._temperature("noul", override_temperature)
                probs_list: list[float] = []
                for candidate_index, candidate in enumerate(options):
                    reads: list[float] = []
                    for order_index, binary_order in enumerate(binary_orders):
                        context_order = context_orders[order_index % len(context_orders)]
                        displayed_binary = [("true", "Yes"), ("false", "No")]
                        displayed_binary = [displayed_binary[index] for index in binary_order]
                        displayed_context = [options[index] for index in context_order]
                        raw = self._read(
                            state,
                            instructions,
                            "noul",
                            displayed_binary,
                            candidate=candidate,
                            context_options=displayed_context,
                            read_twice=read_twice,
                            questions_before_state=questions_before_state,
                        )
                        displayed_probs = softmax(raw / temperature)
                        aligned = self._align_probabilities(displayed_probs, binary_order, 2)
                        reads.append(float(aligned[0]))  # true slot in the canonical order
                    probs_list.append(float(np.mean(reads)))
                probs = np.asarray(probs_list, dtype=np.float64)

            try:
                threshold = float(question.get("threshold", 0.5))
            except (TypeError, ValueError) as exc:
                raise InvalidDecisionRequest("labels threshold must be a number between 0 and 1") from exc
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise InvalidDecisionRequest("labels threshold must be between 0 and 1")
            probability_map = {key: float(prob) for (key, _), prob in zip(options, probs, strict=True)}
            selected = [key for key, prob in probability_map.items() if prob >= threshold]
            exact_set_confidence = 1.0
            for key, prob in probability_map.items():
                exact_set_confidence *= prob if key in selected else (1.0 - prob)
            return {
                "type": "labels",
                "selected": selected,
                "probabilities": probability_map,
                "confidence": float(exact_set_confidence),
                "mode": mode,
            }

        orders = self._permutation_orders(len(options), permutations, seed_material=seed_material)
        temperature = self._temperature(question_type, override_temperature)
        probability_reads = []
        for order in orders:
            displayed_options = [options[index] for index in order]
            raw = self._read(
                state,
                instructions,
                question_type,
                displayed_options,
                read_twice=read_twice,
                questions_before_state=questions_before_state,
            )
            displayed_probs = softmax(raw / temperature)
            probability_reads.append(self._align_probabilities(displayed_probs, order, len(options)))
        probabilities = np.mean(np.stack(probability_reads), axis=0)
        probability_map = {key: float(prob) for (key, _), prob in zip(options, probabilities, strict=True)}

        if question_type == "choice":
            selected_index = int(np.argmax(probabilities))
            return {
                "type": "choice",
                "choice": options[selected_index][0],
                "probabilities": probability_map,
                "confidence": self._confidence_choice(probabilities),
            }

        if question_type == "noul":
            positive_index = next(i for i, (key, _) in enumerate(options) if key.lower() == "true")
            positive_probability = float(probabilities[positive_index])
            return {
                "type": "noul",
                "noul": positive_probability,
                "probabilities": {
                    "false": float(probabilities[1 - positive_index]),
                    "true": positive_probability,
                },
                "confidence": abs(2.0 * positive_probability - 1.0),
            }

        if question_type == "score":
            expected_score = float(np.dot(np.arange(len(probabilities), dtype=np.float64), probabilities))
            return {
                "type": "score",
                "score": expected_score,
                "probabilities": probability_map,
                "legend": {str(i): str(description) for i, (_, description) in enumerate(options)},
                "confidence": self._confidence_score(probabilities),
            }

        raise AssertionError("unreachable question type")

    def _estimate_read_budget(
        self,
        questions: Mapping[str, Any],
        *,
        permutations: int,
        state_chars: int,
        read_twice: bool,
    ) -> tuple[int, int]:
        estimated_reads = 0
        repeated_chars = 0
        for name, question in questions.items():
            question_type = str(question.get("type", "")).lower()
            if question_type not in {"choice", "noul", "score", "labels"}:
                continue
            options = self._items(question.get("criteria"), question_type=question_type)
            self._validate_options(options, question_type, self.limits.max_options)
            seed_material = f"{name}\0{question_type}\0" + "\0".join(key for key, _ in options)
            if question_type == "labels" and question.get("mode", "head") == "per_option":
                binary_orders = self._effective_order_count(2, permutations)
                estimated_reads += len(options) * binary_orders
                prompt_chars = (
                    state_chars * (2 if read_twice else 1)
                    + len(as_text(question.get("instructions", "")))
                    + sum(len(as_text(key)) + len(as_text(value)) for key, value in options)
                    + max(
                        (len(as_text(key)) + len(as_text(value)) for key, value in options),
                        default=0,
                    )
                )
                repeated_chars += len(options) * binary_orders * prompt_chars
            else:
                estimated_reads += len(self._permutation_orders(
                    len(options), permutations, seed_material=seed_material
                ))
        return estimated_reads, repeated_chars

    def decide(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise InvalidDecisionRequest("request body must be a JSON object")
        if "state" not in request:
            raise InvalidDecisionRequest("state is required")
        questions = request.get("questions")
        if not isinstance(questions, Mapping) or not questions:
            raise InvalidDecisionRequest("questions must be a non-empty object")
        if len(questions) > self.limits.max_questions:
            raise InvalidDecisionRequest(f"at most {self.limits.max_questions} questions are allowed")

        permutations, read_twice, questions_before_state = self._validate_request_flags(request, self.limits)
        state = request["state"]
        state_text = as_text(state)
        try:
            state_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InvalidDecisionRequest("state contains invalid Unicode") from exc
        if len(state_text) > self.limits.max_state_chars:
            raise InvalidDecisionRequest("state exceeds maximum allowed size")

        question_chars = 0
        for name, question in questions.items():
            if not isinstance(question, Mapping):
                raise InvalidDecisionRequest(f"question {name!r} must be an object")
            text_parts = [as_text(name), as_text(question.get("instructions", ""))]
            criteria = question.get("criteria", {})
            if isinstance(criteria, Mapping):
                text_parts.extend(as_text(key) + as_text(value) for key, value in criteria.items())
            elif isinstance(criteria, list):
                text_parts.extend(as_text(value) for value in criteria)
            try:
                for part in text_parts:
                    part.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise InvalidDecisionRequest(f"question {name!r} contains invalid Unicode") from exc
            question_chars += sum(len(part) for part in text_parts)
            if question_chars > self.limits.max_question_chars:
                raise InvalidDecisionRequest("questions exceed maximum allowed size")

        estimated_reads, repeated_chars = self._estimate_read_budget(
            questions, permutations=permutations, state_chars=len(state_text), read_twice=read_twice
        )
        if estimated_reads > self.limits.max_decision_reads:
            raise InvalidDecisionRequest("request exceeds maximum decision-read budget")
        if repeated_chars > self.limits.max_per_option_prompt_chars:
            raise InvalidDecisionRequest(
                f"per-option prompts repeat more than {self.limits.max_per_option_prompt_chars:,} characters"
            )

        temperature = request.get("temperature")
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            answers[str(name)] = self._answer(
                state,
                str(name),
                question,
                temperature,
                permutations=permutations,
                read_twice=read_twice,
                questions_before_state=questions_before_state,
            )
        return {"model": request.get("model", "decision"), "answers": answers}
