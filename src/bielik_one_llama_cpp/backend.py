from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from .config import DECISION_TOKEN_COUNT, DECISION_TOKEN_BASE_ID, EngineConfig


class BackendError(RuntimeError):
    pass


class ContextLimitError(BackendError):
    pass


class LlamaCppBackend:
    """In-process next-token logit reader for the merged Bielik-One GGUF model."""

    def __init__(self, config: EngineConfig):
        config.validate()
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - exercised only on runtime host
            raise RuntimeError(
                "llama-cpp-python is missing. Install the project and a build matching your CPU/GPU."
            ) from exc

        self.config = config
        self._llama_cpp = llama_cpp
        if not callable(getattr(llama_cpp, "llama_get_logits_ith", None)):
            raise RuntimeError(
                "This llama-cpp-python build does not expose llama_get_logits_ith; "
                "install version 0.3.16 or newer."
            )
        self._lock = threading.RLock()
        self.llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.n_threads,
            n_batch=config.n_batch,
            n_ubatch=config.n_ubatch,
            use_mmap=config.use_mmap,
            use_mlock=config.use_mlock,
            vocab_only=config.vocab_only,
            logits_all=False,
            verbose=config.verbose,
        )
        self.decision_token_ids = self._resolve_decision_tokens()
        self._cached_prefix_tokens: tuple[int, ...] | None = None

    def _resolve_decision_tokens(self) -> tuple[int, ...]:
        ids: list[int] = []
        for index in range(DECISION_TOKEN_COUNT):
            marker = f"<|systemone_decision_{index:04d}|>".encode("utf-8")
            encoded = self.llm.tokenize(marker, add_bos=False, special=True)
            if len(encoded) != 1:
                raise BackendError(
                    f"GGUF tokenizer does not expose native decision token {marker.decode()!r} as one token. "
                    "Use a GGUF converted from the exact merged HF repository, including its added tokens."
                )
            ids.append(int(encoded[0]))

        if len(set(ids)) != DECISION_TOKEN_COUNT:
            raise BackendError("Decision-token vocabulary is not one-to-one in this GGUF file")
        expected = list(range(DECISION_TOKEN_BASE_ID, DECISION_TOKEN_BASE_ID + DECISION_TOKEN_COUNT))
        # The ids are expected to be consecutive in the original HF checkpoint. A
        # different GGUF tokenizer layout may be valid, so only report the delta.
        self.decision_token_id_range_matches_hf = ids == expected
        return tuple(ids)

    def _tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def _rewind_to_prefix(self, prefix_length: int) -> bool:
        """Drop the previous question suffix while retaining prefix KV tensors."""
        context = getattr(self.llm, "_ctx", None)
        remove_suffix = getattr(context, "kv_cache_seq_rm", None)
        if not callable(remove_suffix) or not hasattr(self.llm, "n_tokens"):
            return False
        try:
            removed = remove_suffix(-1, prefix_length, -1)
            if removed is False:
                return False
            self.llm.n_tokens = prefix_length
            return True
        except Exception:
            return False

    def score_options(
        self,
        prompt: str,
        option_indices: Sequence[int],
        *,
        shared_prefix: str | None = None,
    ) -> np.ndarray:
        if not option_indices:
            raise ValueError("At least one option index is required")
        if min(option_indices) < 0 or max(option_indices) >= len(self.decision_token_ids):
            raise ValueError("Option index exceeds the Bielik-One decision vocabulary")

        prompt_tokens = self._tokenize(prompt)
        if shared_prefix is None:
            prefix_tokens: list[int] = []
            suffix_tokens = prompt_tokens
        else:
            requested_prefix_tokens = self._tokenize(shared_prefix)
            # Tokenizers may merge a trailing whitespace token with the first
            # character of the suffix. Keep the longest *actual* token prefix
            # of the fully tokenized prompt rather than assuming text-level
            # tokenization is compositional at the split boundary.
            shared_length = 0
            for expected, actual in zip(requested_prefix_tokens, prompt_tokens):
                if expected != actual:
                    break
                shared_length += 1
            prefix_tokens = prompt_tokens[:shared_length]
            suffix_tokens = prompt_tokens[shared_length:]
            if not suffix_tokens:
                raise BackendError("decision prompt suffix is empty")

        if len(prompt_tokens) > self.config.n_ctx:
            raise ContextLimitError(
                f"Prompt has {len(prompt_tokens)} tokens; configured context is {self.config.n_ctx}."
            )

        with self._lock:
            try:
                if shared_prefix is None:
                    self.llm.reset()
                    self._cached_prefix_tokens = None
                    self.llm.eval(prompt_tokens)
                else:
                    prefix_key = tuple(prefix_tokens)
                    if getattr(self, "_cached_prefix_tokens", None) == prefix_key:
                        if not self._rewind_to_prefix(len(prefix_tokens)):
                            # Do not risk reading stale KV state on an older
                            # llama-cpp-python build: recompute correctly instead.
                            self.llm.reset()
                            self.llm.eval(prefix_tokens)
                    else:
                        self.llm.reset()
                        self.llm.eval(prefix_tokens)
                    self.llm.eval(suffix_tokens)
                    self._cached_prefix_tokens = prefix_key

                # With logits_all=False llama-cpp-python deliberately does not
                # copy logits into Llama.scores. Read only the final-token row
                # directly from llama.cpp rather than allocating n_ctx x vocab.
                logits_ptr = self._llama_cpp.llama_get_logits_ith(self.llm.ctx, -1)
                if logits_ptr is None or not bool(logits_ptr):
                    raise BackendError("llama.cpp did not return logits for the final prompt token")
                scores = np.ctypeslib.as_array(
                    logits_ptr, shape=(self.llm.n_vocab(),)
                ).copy().astype(np.float64, copy=False)

                token_ids = [self.decision_token_ids[i] for i in option_indices]
                if max(token_ids) >= scores.shape[0]:
                    raise BackendError(
                        f"GGUF output vocabulary ({scores.shape[0]}) does not contain decision token id {max(token_ids)}"
                    )
                result = scores[token_ids]
            except Exception:
                # A failed decode may leave only part of a suffix in the KV
                # cache. Invalidate it so a later request can never reuse it.
                self._cached_prefix_tokens = None
                try:
                    self.llm.reset()
                except Exception:
                    pass
                raise

        if result.shape != (len(option_indices),) or not np.all(np.isfinite(result)):
            raise BackendError("Backend returned invalid decision logits")
        return result
