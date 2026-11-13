from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .api import create_app
from .backend import LlamaCppBackend
from .calibration_fit import calibrate_model
from .config import DEFAULT_PORT, EngineConfig
from .decision import DecisionEngine
from .model_profiles import detect_model_profile, resolve_auto_calibration


_LOG = logging.getLogger("bielik_one_llama_cpp")


def _add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_n_ctx: int = 32768,
    default_gpu_layers: int = -1,
) -> None:
    parser.add_argument("--n-ctx", type=int, default=default_n_ctx)
    parser.add_argument("--n-gpu-layers", type=int, default=default_gpu_layers)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")


def _make_backend(model_path: str, args, *, vocab_only: bool = False) -> LlamaCppBackend:
    config = EngineConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        vocab_only=vocab_only,
        verbose=args.verbose,
    )
    return LlamaCppBackend(config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bielik-one-llama-cpp", description="Bielik-One typed decision server using llama.cpp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="load a GGUF model and start the HTTP API")
    serve.add_argument("model", help="path to a merged Bielik-One GGUF")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    _add_runtime_arguments(serve)
    calibration = serve.add_mutually_exclusive_group()
    calibration.add_argument(
        "--calibration",
        default="auto",
        metavar="PATH|auto",
        help=(
            "calibration JSON; 'auto' (default) loads a model-specific sidecar, "
            "or official 11B BF16 temperatures only when applicable"
        ),
    )
    calibration.add_argument(
        "--no-calibration",
        dest="calibration",
        action="store_const",
        const="none",
        help="use raw probabilities even when a matching calibration file exists",
    )

    check = subparsers.add_parser("check", help="verify native decision tokens in a GGUF model")
    check.add_argument("model")
    _add_runtime_arguments(check, default_n_ctx=4096, default_gpu_layers=0)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="fit model-specific temperatures on a labelled JSONL validation set",
    )
    calibrate.add_argument("model", help="path to the GGUF that will be calibrated")
    calibrate.add_argument("--dataset", required=True, help="labelled validation data in JSONL format")
    calibrate.add_argument(
        "--output",
        default=None,
        help="output JSON (default: <model-stem>.calibration.json next to the GGUF)",
    )
    calibrate.add_argument(
        "--min-samples-per-type",
        type=int,
        default=20,
        help="minimum labelled examples required to fit each temperature (default: 20)",
    )
    _add_runtime_arguments(calibrate)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "serve":
        backend = _make_backend(args.model, args)
        if args.calibration == "none":
            use_calibration, calibration_path = False, None
            calibration_source = "disabled by --no-calibration"
        elif args.calibration == "auto":
            selection = resolve_auto_calibration(args.model)
            use_calibration, calibration_path = selection.enabled, selection.path
            calibration_source = selection.source
        else:
            calibration_path = args.calibration
            if not Path(calibration_path).is_file():
                raise SystemExit(f"Calibration file not found: {calibration_path}")
            use_calibration, calibration_source = True, "explicit calibration file"

        _LOG.info("Model profile: %s", detect_model_profile(args.model).as_dict())
        _LOG.info("Calibration: %s", calibration_source)
        engine = DecisionEngine(
            backend,
            calibration=use_calibration,
            calibration_path=calibration_path,
        )
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - exercised on runtime host
            raise SystemExit("Install project dependencies first: pip install -e .") from exc
        app = create_app(engine, model_id=Path(args.model).name)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    if args.command == "check":
        backend = _make_backend(args.model, args, vocab_only=True)
        print(json.dumps({
            "status": "ok",
            "model_profile": detect_model_profile(args.model).as_dict(),
            "decision_tokens": len(backend.decision_token_ids),
            "decision_vocabulary_validated": True,
            "token_ids_consecutive": list(backend.decision_token_ids) == list(range(
                backend.decision_token_ids[0],
                backend.decision_token_ids[0] + len(backend.decision_token_ids),
            )),
            "legacy_11b_token_id_range_matches": backend.decision_token_id_range_matches_hf,
            "hf_token_id_range_matches": backend.decision_token_id_range_matches_hf,
            "first_token_id": backend.decision_token_ids[0],
            "last_token_id": backend.decision_token_ids[-1],
        }, indent=2))
        return

    if args.command == "calibrate":
        if args.min_samples_per_type < 2:
            raise SystemExit("--min-samples-per-type must be at least 2")
        backend = _make_backend(args.model, args)
        # A calibration run must start from temperature=1.0 so its predictions
        # represent the uncalibrated GGUF, not the shipped 11B BF16 profile.
        engine = DecisionEngine(backend, calibration=False)
        result = calibrate_model(
            engine,
            model_path=args.model,
            dataset_path=args.dataset,
            output_path=args.output,
            min_samples_per_type=args.min_samples_per_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
