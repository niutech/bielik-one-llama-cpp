from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .backend import ContextLimitError, BackendError
from .decision import DecisionEngine, InvalidDecisionRequest


def create_app(engine: DecisionEngine, *, model_id: str = "decision") -> FastAPI:
    app = FastAPI(
        title="Bielik-One llama.cpp decision API",
        version=__version__,
        description="Typed decisions from Bielik-One via native decision-token logits.",
    )
    app.state.engine = engine
    app.state.model_id = model_id

    @app.exception_handler(InvalidDecisionRequest)
    async def invalid_request_handler(_: Request, exc: InvalidDecisionRequest) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": str(exc), "code": "INVALID_REQUEST"})

    @app.exception_handler(ContextLimitError)
    async def context_error_handler(_: Request, exc: ContextLimitError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": str(exc), "code": "CONTEXT_LIMIT"})

    @app.exception_handler(BackendError)
    async def backend_error_handler(_: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": str(exc), "code": "BACKEND_ERROR"})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "llama-cpp-python",
            "model": app.state.model_id,
            "calibrated": engine.calibration_enabled,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "decision", "object": "model", "owned_by": "speakleash"}],
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        max_body_bytes = 16 * 1024 * 1024
        body_chunks: list[bytes] = []
        body_size = 0
        async for chunk in request.stream():
            body_size += len(chunk)
            if body_size > max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"error": "request body too large", "code": "PAYLOAD_TOO_LARGE"},
                )
            body_chunks.append(chunk)
        body = b"".join(body_chunks)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidDecisionRequest("body must be valid UTF-8 JSON") from exc
        result = await run_in_threadpool(engine.decide, payload)
        return JSONResponse(content=result)

    return app
