"""Delveta-LayaChoice inference sidecar — a thin localhost HTTP wrapper.

The Intent Funnel's ``cap_router`` lane (``chat_cap_router_backend="cap_router"``)
selects a capability by asking this service ONE choice question per turn. The
service owns a single responsibility: load the fine-tuned LayaChoice checkpoint
once at startup and answer ``/v1/systemone`` requests from it. It contains NO
Delveta business logic — no candidate normalization (K), no NONE policy, no
argument extraction, no Registry, no Binder. Those live in the Delveta process;
here we only serve the model.

Wire (the TypeSafe Jev ``/v1/systemone`` shape):

    POST /v1/systemone
      {"state": <the user turn>, "questions": {"<qid>": {"type": "choice",
        "instructions": <str>, "criteria": {<label>: <card text>, ...}}},
       "max_len": <int, optional>, "head_max_len": <int, optional>}
      -> {"model": ..., "answers": {"<qid>": {"type": "choice", "choice": <label>,
          "probabilities": {...}, "confidence": ..., "answer_confidence": ...}},
          "usage": {...}}

    GET /health -> {"status": "ok", "model": <path>, "device": <dev>}

Configuration (environment):
    LAYA_CHOICE_MODEL_DIR  deployment directory of the fine-tuned checkpoint
                           (rl_agent_config.json + model.safetensors + encoder/
                           + tokenizer/). Mounted READ-ONLY; never baked into the
                           image. REQUIRED.
    LAYA_DEVICE            torch device for inference (default "cpu").
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request

from layachoice_sequence import HEAD_MAX_LEN, MAX_LEN, install

# One model, one forward pass at a time: the endpoints are sync so FastAPI runs
# them in its threadpool; this lock serializes access to the single Agent.
_AGENT_LOCK = threading.Lock()
_AGENT: Any = None
_MODEL_DIR: str = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _AGENT, _MODEL_DIR
    import laya

    _MODEL_DIR = (os.environ.get("LAYA_CHOICE_MODEL_DIR") or "").strip()
    if not _MODEL_DIR:
        raise RuntimeError("LAYA_CHOICE_MODEL_DIR is not set; mount the "
                           "LayaChoice checkpoint directory and set it")
    device = (os.environ.get("LAYA_DEVICE") or "cpu").strip() or "cpu"
    # The 256-token option cap MUST be installed before the Agent encodes anything.
    install()
    _AGENT = laya.Agent(_MODEL_DIR, device=device)
    # Warm the model with a tiny request so the first real turn is not billed for
    # the cold forward pass (the client's per-call timeout excludes model load).
    _AGENT.system_one("warmup", {"capability": {
        "type": "choice", "instructions": "Which capability should handle the user's request?",
        "criteria": {"warm": "### warm\ntool: warm\ndoes: warmup\n"
                             "negative examples:\n- (none curated)\nparams: none (arguments must be {})"}}},
        max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    yield
    _AGENT = None


app = FastAPI(title="Delveta-LayaChoice", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok" if _AGENT is not None else "loading",
            "model": _MODEL_DIR,
            # str(): the Agent's device is a torch.device, which FastAPI/pydantic
            # cannot serialize (it 500'd); the JSON surface only ever needed a label.
            "device": str(getattr(_AGENT, "device", None))}


@app.post("/v1/systemone")
async def systemone(request: Request) -> Dict[str, Any]:
    if _AGENT is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="request body must be valid JSON")
    if not isinstance(body, dict) or "questions" not in body:
        raise HTTPException(status_code=400,
                            detail="request body must be an object with a 'questions' field")
    state = body.get("state")
    questions = body["questions"]
    max_len = body.get("max_len") or MAX_LEN
    head_max_len = body.get("head_max_len") or HEAD_MAX_LEN
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=422, detail="'questions' must be a non-empty object")
    try:
        with _AGENT_LOCK:
            return _AGENT.system_one(state, questions, max_len=int(max_len),
                                     head_max_len=int(head_max_len))
    except ValueError as exc:
        # Laya's question validation errors name the question and what to fix.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:  # noqa: BLE001 -- never leak paths/weights/OOM text to clients
        raise HTTPException(status_code=500, detail="inference failed")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("LAYA_PORT", "8000")),
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
