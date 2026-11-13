from __future__ import annotations

import ctypes
import re
import sys
import types

import numpy as np

from bielik_one_llama_cpp.backend import LlamaCppBackend
from bielik_one_llama_cpp.config import EngineConfig


class FakeContext:
    def __init__(self):
        self.removed = []
        self.owner = None
    def kv_cache_seq_rm(self, seq_id, start, end):
        self.removed.append((seq_id, start, end))
        if self.owner is not None and self.owner.evaluated is not None:
            self.owner.evaluated = self.owner.evaluated[:start]
        return True


class FakeLlama:
    def __init__(self, **kwargs):
        self.ctx = object()
        self._ctx = FakeContext()
        self._ctx.owner = self
        self.n_tokens = 0
        self._vocab_size = 32651
        self.scores = _ForbiddenScores()
        self.evaluated = None

    def tokenize(self, text, add_bos=False, special=True):
        match = re.fullmatch(rb"<\|systemone_decision_(\d{4})\|>", text)
        if match:
            return [32000 + int(match.group(1))]
        return [10, 11, 12]

    def reset(self):
        self.evaluated = None
        self.n_tokens = 0

    def eval(self, tokens):
        if self.evaluated is None:
            self.evaluated = []
        self.evaluated.extend(tokens)
        self.n_tokens += len(tokens)

    def n_vocab(self):
        return self._vocab_size


class _ForbiddenScores:
    def __getitem__(self, _index):
        raise AssertionError("Llama.scores must not be used when logits_all=False")


def test_backend_reads_last_logits_via_low_level_api(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    values = (ctypes.c_float * 32651)()
    values[32000] = 1.25
    values[32001] = -0.75
    calls = []

    def get_logits_ith(ctx, index):
        calls.append((ctx, index))
        return values

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama
    fake_module.llama_get_logits_ith = get_logits_ith
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    backend = LlamaCppBackend(EngineConfig(model_path=str(model_file), n_ctx=512))
    result = backend.score_options("prompt", [0, 1])

    np.testing.assert_allclose(result, [1.25, -0.75])
    assert calls == [(backend.llm.ctx, -1)]
    assert backend.llm.evaluated == [10, 11, 12]


def test_backend_reuses_kv_prefix_and_only_evaluates_question_suffix(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")
    values = (ctypes.c_float * 32651)()
    values[32000] = 1.25
    values[32001] = -0.75
    calls = []

    def get_logits_ith(ctx, index):
        calls.append((ctx, index))
        return values

    class PrefixLlama(FakeLlama):
        def tokenize(self, text, add_bos=False, special=True):
            if text == b"PREFIX":
                return [10, 11]
            if text == b"PREFIX-one":
                return [10, 11, 12]
            if text == b"PREFIX-two":
                return [10, 11, 13]
            return super().tokenize(text, add_bos, special)

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = PrefixLlama
    fake_module.llama_get_logits_ith = get_logits_ith
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    backend = LlamaCppBackend(EngineConfig(model_path=str(model_file), n_ctx=512))
    backend.score_options("PREFIX-one", [0, 1], shared_prefix="PREFIX")
    first_eval = list(backend.llm.evaluated)
    backend.score_options("PREFIX-two", [0, 1], shared_prefix="PREFIX")
    assert first_eval == [10, 11, 12]
    # Retain tokens 10,11 then append only 13; do not re-evaluate prefix.
    assert backend.llm.evaluated == [10, 11, 13]
    assert backend.llm._ctx.removed == [(-1, 2, -1)]
    assert len(calls) == 2
