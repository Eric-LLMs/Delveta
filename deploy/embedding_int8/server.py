"""INT8 BGE-M3 embedding service — drop-in for the TEI ``/embed`` contract.

Serves ``POST /embed`` with body ``{"inputs": [...], "normalize": true}`` and returns a
JSON array of 1024-dim L2-normalized vectors, i.e. the exact HTTP contract the app's
``core.infrastructure.vector.TEIEmbedder`` already speaks. It is backed by the EXACT
validated Delveta fine-tuned BGE-M3 INT8 ONNX graph (CLS pooling + L2 normalization are
inside the graph; output ``sentence_embedding``).

Encoding is batch = 1 (no padding), truncation = 64 — identical to the frozen
retrieval-layer evaluation protocol, so the served vectors match the validated artifact.
No app code changes: the app still reaches this service at ``http://localhost:18080``.
"""
from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
MAX_LEN = int(os.environ.get("MAX_LEN", "64"))

_sess = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"), providers=["CPUExecutionProvider"]
)
_tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
_tok.enable_truncation(max_length=MAX_LEN)

app = FastAPI(title="delveta-embedding-int8")


class EmbedRequest(BaseModel):
    inputs: list[str]
    normalize: bool = True


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/embed")
def embed(req: EmbedRequest) -> list[list[float]]:
    out: list[list[float]] = []
    for text in req.inputs:
        ids = _tok.encode(text).ids
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        vec = _sess.run(None, feed)[0][0]
        out.append([float(x) for x in vec])
    return out
