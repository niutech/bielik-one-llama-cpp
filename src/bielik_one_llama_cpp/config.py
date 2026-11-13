from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ID = "speakleash/Bielik-One-11B-v3.1-r1"
DEFAULT_PORT = 8026
DEFAULT_CONTEXT = 32768
DECISION_TOKEN_COUNT = 651
DECISION_TOKEN_BASE_ID = 32000  # HF config vocab_size=32651; standard tokenizer base=32000.


@dataclass(slots=True)
class EngineConfig:
    model_path: str
    n_ctx: int = DEFAULT_CONTEXT
    n_gpu_layers: int = -1
    n_threads: int | None = None
    n_batch: int = 2048
    n_ubatch: int = 512
    use_mmap: bool = True
    use_mlock: bool = False
    vocab_only: bool = False
    verbose: bool = False
    calibration_path: str | None = None
    use_calibration: bool = True

    def validate(self) -> None:
        path = Path(self.model_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {path}")
        if path.suffix.lower() != ".gguf":
            raise ValueError(
                "llama-cpp-python requires a GGUF file. Convert the HF safetensors model first; see README.md."
            )
        if self.n_ctx < 256:
            raise ValueError("n_ctx must be at least 256")
        if self.n_batch < 1 or self.n_ubatch < 1:
            raise ValueError("n_batch and n_ubatch must be positive")
