# embedding_int8 — Delveta-BGE-M3-v1 INT8 embedding service

The single DEFAULT embedding backend for Delveta. It replaces the base
`BAAI/bge-m3` TEI container on the same `POST /embed` contract (port 18080), so
the app needs **no code change** — `core.infrastructure.vector.TEIEmbedder` keeps
speaking the identical wire.

- **Model**: `eric-ml-nlp/Delveta-BGE-M3-v1` → `INT8/model.onnx`
  - ONNX sha256 `0e0ada7fec9367bca6d816b86953e4f9fe8531ebd4e8251f6de0721c0d2b1f0d`
  - tokenizer.json sha256 `abc75dc1694c03d1fbc4c622b8598061a072ae5d06c359191f194774cc54e497`
  - (both equal the Hugging Face LFS object ids)
- **Graph**: CLS pooling + L2 normalization are baked into the ONNX graph; output
  `sentence_embedding`, 1024-dim, L2-normalized. Encoding is batch=1, no padding,
  truncation 64 — identical to the frozen retrieval-layer evaluation protocol.
- **Weights are NOT baked into the image.** They are bind-mounted read-only from
  the host model store and pointed at by `MODEL_DIR` (default `/model`). This
  mirrors the `cap-router` service's read-only checkpoint mount.
- The image itself is `delveta-embedding-int8:latest`. Rebuild only if the image
  is missing: `docker build -t delveta-embedding-int8:latest deploy/embedding_int8`.

Failure is explicit: if `model.onnx` is absent/unreadable the ONNX session raises
at import time and the container exits — there is **no fallback to the base model**.
