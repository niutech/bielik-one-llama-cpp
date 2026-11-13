from __future__ import annotations

import argparse
import json
from .api import create_app
from .backend import LlamaCppBackend
from .config import DEFAULT_PORT, EngineConfig
from .decision import DecisionEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bielik-one-llama-cpp", description="Bielik-One typed decision server using llama.cpp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="load a GGUF model and start the HTTP API")
    serve.add_argument("model", help="path to Bielik-One-11B-v3.1-r1 GGUF")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--n-ctx", type=int, default=32768)
    serve.add_argument("--n-gpu-layers", type=int, default=-1)
    serve.add_argument("--n-threads", type=int, default=None)
    serve.add_argument("--n-batch", type=int, default=2048)
    serve.add_argument("--n-ubatch", type=int, default=512)
    serve.add_argument("--calibration", default=None, help="path to calibration.json (default: shipped BF16 temperatures)")
    serve.add_argument("--no-calibration", action="store_true", help="use raw probabilities (recommended for uncalibrated quantized GGUF)")
    serve.add_argument("--verbose", action="store_true")

    check = subparsers.add_parser("check", help="verify native decision tokens in a GGUF model")
    check.add_argument("model")
    check.add_argument("--n-ctx", type=int, default=4096)
    check.add_argument("--n-gpu-layers", type=int, default=0)
    check.add_argument("--verbose", action="store_true")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "serve":
        config = EngineConfig(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
            verbose=args.verbose,
            calibration_path=args.calibration,
            use_calibration=not args.no_calibration,
        )
        backend = LlamaCppBackend(config)
        engine = DecisionEngine(
            backend,
            calibration=config.use_calibration,
            calibration_path=config.calibration_path,
        )
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Install project dependencies first: pip install -e .") from exc
        app = create_app(engine)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    if args.command == "check":
        config = EngineConfig(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            vocab_only=True,
            verbose=args.verbose,
        )
        backend = LlamaCppBackend(config)
        print(json.dumps({
            "status": "ok",
            "decision_tokens": len(backend.decision_token_ids),
            "hf_token_id_range_matches": backend.decision_token_id_range_matches_hf,
            "first_token_id": backend.decision_token_ids[0],
            "last_token_id": backend.decision_token_ids[-1],
        }, indent=2))
        return

    raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
