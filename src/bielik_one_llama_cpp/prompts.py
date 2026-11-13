from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

# User text must not be allowed to inject ChatML or model-control tokens.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^|\r\n]{1,100}\|>|</?(?:think|s)\s*>", re.IGNORECASE)


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def neutralize_control_tokens(text: str) -> str:
    """Keep untrusted strings literal rather than allowing control-token injection."""
    return _CONTROL_TOKEN_RE.sub(lambda match: match.group(0).replace("<", "‹").replace(">", "›"), text)


def _clean(value: Any) -> str:
    return neutralize_control_tokens(as_text(value)).strip()


def render_decision_prompt_parts(
    *,
    state: Any,
    instructions: Any,
    options: Sequence[tuple[str, Any]],
    question_type: str,
    candidate: tuple[str, Any] | None = None,
    context_options: Sequence[tuple[str, Any]] | None = None,
    read_twice: bool = False,
    questions_before_state: bool = False,
) -> tuple[str, str]:
    """Return ``(shared_prefix, question_suffix)`` for a decision read.

    With the default layout the complete state block is part of the shared
    prefix. Requests containing multiple questions can therefore retain its
    KV cache and only evaluate each question-specific suffix. With
    ``questions_before_state`` the state necessarily follows the question, so
    only the invariant ChatML/instruction header can be reused.

    ``read_twice`` duplicates the state block verbatim. This changes the
    prompt, not the number of model forwards.
    """
    state_text = _clean(state)
    instruction_text = _clean(instructions)
    visible_options = context_options if candidate is not None and context_options is not None else options

    header = (
        "<s><|im_start|>user\n"
        "Read the state and answer the decision question using the available options.\n"
        "Do not explain your answer. Select only from the options provided.\n\n"
    )
    state_block = f"STATE:\n{state_text}"
    if read_twice:
        state_block += f"\n\n{state_text}"

    question_lines = ["QUESTION:", instruction_text, "", "OPTIONS:"]
    for index, (key, description) in enumerate(visible_options):
        desc = _clean(description) if description is not None else key
        if not desc:
            desc = key
        question_lines.append(f"{index}. {key}: {desc}")

    if candidate is not None:
        key, description = candidate
        desc = _clean(description) if description is not None else key
        question_lines.extend(
            [
                "",
                f"CANDIDATE: {key}: {desc}",
                "Does this candidate belong to the correct answer set?",
            ]
        )
    elif question_type == "labels":
        question_lines.extend(["", "For each option, assess independently whether it belongs to the correct answer set."])
    elif question_type == "score":
        question_lines.extend(
            [
                "",
                "The options represent ordered levels from lowest to highest in the original request, "
                "regardless of their displayed order.",
            ]
        )

    question_block = "\n".join(question_lines)
    if questions_before_state:
        # The complete question (including its options) precedes the state.
        # Only the invariant header is shared between question reads.
        prefix = header
        suffix_body = question_block + "\n\n" + state_block
    else:
        # This is the important shared part across all questions for one state.
        prefix = header + state_block + "\n\n"
        suffix_body = question_block

    suffix = suffix_body + "<|im_end|>\n<|im_start|>assistant\n"
    return prefix, suffix


def render_decision_prompt(
    *,
    state: Any,
    instructions: Any,
    options: Sequence[tuple[str, Any]],
    question_type: str,
    candidate: tuple[str, Any] | None = None,
    context_options: Sequence[tuple[str, Any]] | None = None,
    read_twice: bool = False,
    questions_before_state: bool = False,
) -> str:
    """Render a complete prompt (convenience wrapper used by tests/tools)."""
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
    return prefix + suffix
