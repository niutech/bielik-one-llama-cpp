from __future__ import annotations

import ctypes
import re
import sys
import types

import numpy as np
import pytest

from bielik_one_llama_cpp.backend import BackendError, LlamaCppBackend
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
    decision_token_count = 651
    decision_token_base_id = 32000

    def __init__(self, **kwargs):
        self.ctx = object()
        self._ctx = FakeContext()
        self._ctx.owner = self
        self.n_tokens = 0
        self._vocab_size = self.decision_token_base_id + self.decision_token_count
        self.scores = _ForbiddenScores()
        self.evaluated = None

    def tokenize(self, text, add_bos=False, special=True):
        match = re.fullmatch(rb"<\|systemone_decision_(\d{4})\|>", text)
        if match and int(match.group(1)) < self.decision_token_count:
            return [self.decision_token_base_id + int(match.group(1))]
        return [10, 11, 12]

    def detokenize(self, tokens, special=False):
        index = tokens[0] - self.decision_token_base_id
        if 0 <= index < self.decision_token_count:
            return f"<|systemone_decision_{index:04d}|>".encode()
        return b"ordinary token"

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
    assert len(backend.decision_token_ids) == 651
    assert backend.decision_token_id_range_matches_hf


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


def make_backend(tmp_path, monkeypatch, llama_class):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = llama_class
    fake_module.llama_get_logits_ith = lambda *_: None
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    return LlamaCppBackend(EngineConfig(model_path=str(model_file), n_ctx=512))


def test_backend_discovers_4b_vocabulary_and_reads_its_logits(tmp_path, monkeypatch):
    class FourBLlama(FakeLlama):
        decision_token_count = 516
        decision_token_base_id = 248320

    backend = make_backend(tmp_path, monkeypatch, FourBLlama)
    assert backend.decision_token_ids == tuple(range(248320, 248836))
    assert not backend.decision_token_id_range_matches_hf  # Legacy 11B diagnostic.

    values = (ctypes.c_float * backend.llm.n_vocab())()
    values[248320] = 1.25
    values[248835] = -0.75
    backend._llama_cpp.llama_get_logits_ith = lambda *_: values
    np.testing.assert_allclose(backend.score_options("prompt", [0, 515]), [1.25, -0.75])
    with pytest.raises(ValueError, match="exceeds"):
        backend.score_options("prompt", [516])


@pytest.mark.parametrize("token_count", [0, 1])
def test_backend_rejects_missing_decision_vocabulary(tmp_path, monkeypatch, token_count):
    class MissingLlama(FakeLlama):
        decision_token_count = token_count

    with pytest.raises(BackendError, match="at least two native decision tokens"):
        make_backend(tmp_path, monkeypatch, MissingLlama)


def test_backend_rejects_gaps_in_decision_vocabulary(tmp_path, monkeypatch):
    class GappedLlama(FakeLlama):
        def detokenize(self, tokens, special=False):
            if tokens == [32002]:
                return b"ordinary token"
            return super().detokenize(tokens, special=special)

    with pytest.raises(BackendError, match="gap.*0002"):
        make_backend(tmp_path, monkeypatch, GappedLlama)


@pytest.mark.parametrize("encoded", [[10, 11], [32000]])
def test_backend_rejects_split_or_aliased_decision_tokens(tmp_path, monkeypatch, encoded):
    class BrokenLlama(FakeLlama):
        def tokenize(self, text, add_bos=False, special=True):
            if text == b"<|systemone_decision_0001|>":
                return encoded
            return super().tokenize(text, add_bos=add_bos, special=special)

    with pytest.raises(BackendError, match="0001"):
        make_backend(tmp_path, monkeypatch, BrokenLlama)


def test_backend_accepts_nonconsecutive_token_ids(tmp_path, monkeypatch):
    class ReorderedLlama(FakeLlama):
        decision_token_count = 2

        def detokenize(self, tokens, special=False):
            if tokens == [42]:
                return b"<|systemone_decision_0001|>"
            if tokens == [32001]:
                return b"ordinary token"
            return super().detokenize(tokens, special=special)

        def tokenize(self, text, add_bos=False, special=True):
            if text == b"<|systemone_decision_0001|>":
                return [42]
            return super().tokenize(text, add_bos=add_bos, special=special)

    backend = make_backend(tmp_path, monkeypatch, ReorderedLlama)
    assert backend.decision_token_ids == (32000, 42)
