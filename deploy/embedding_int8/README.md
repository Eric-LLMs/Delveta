# embedding_int8 — Delveta-BGE-M3-v1 INT8 embedding service

The single DEFAULT embedding backend for Delveta. It replaced the base
`BAAI/bge-m3` TEI container on the same `POST /embed` contract (port 18080), so
the app needed **no code change** — `core.infrastructure.vector.TEIEmbedder`
keeps speaking the identical wire.

- **Model**: `eric-ml-nlp/Delveta-BGE-M3-v1` → `INT8/model.onnx`
  - ONNX sha256 `0e0ada7fec9367bca6d816b86953e4f9fe8531ebd4e8251f6de0721c0d2b1f0d`
  - tokenizer.json sha256 `abc75dc1694c03d1fbc4c622b8598061a072ae5d06c359191f194774cc54e497`
  - (both equal the Hugging Face LFS object ids)
- **Graph**: CLS pooling + L2 normalization are baked into the ONNX graph; output
  `sentence_embedding`, 1024-dim, L2-normalized. Encoding is batch=1, no padding,
  truncation 64 — identical to the frozen retrieval-layer evaluation protocol.
- **Weights are NOT baked into the image.** The container owns the
  `delveta_embedding-model` named volume; on first boot `bootstrap.py` downloads
  the pinned revision (`625d25d…`) from Hugging Face, verifies every file's size
  + SHA256, and publishes it under `/model/current`. Later boots reuse the volume
  with no network access. An interrupted download resumes at the last complete
  byte instead of restarting. The image itself is
  `delveta-embedding-int8:latest`; rebuild from the `deploy` context:
  `docker build -f deploy/embedding_int8/Dockerfile -t delveta-embedding-int8:latest deploy`.
- **Model pins come from runtime configuration, not the image.**
  `docker-compose.yml` injects `MODEL_SPEC_JSON` from
  `deploy/model_config/embedding.env`. After editing that file, recreate the
  container so Compose re-reads the environment:

  ```bash
  docker compose up -d --force-recreate embedding
  ```

  A plain `docker compose restart` keeps the environment fixed at container
  creation and therefore keeps serving the old pin. No image rebuild is needed
  for a model change.
- **Cross-machine deploy** needs only Docker and network access on first boot —
  no host drive letter, project directory, or pre-existing model files.
  `docker compose down -v` deletes the volume, so the next boot re-downloads.
  To avoid the download, export the volume with
  `deploy/model_volume_bundle.sh export delveta_embedding-model …` and import it
  on the target machine.
- **Volume internals, lifecycle, and moving models between machines**:
  [../MODEL_VOLUMES.md](../MODEL_VOLUMES.md); the full runtime-configuration
  procedure is in [../../docs/model-deployment.en.md](../../docs/model-deployment.en.md).

Failure is explicit: if the download cannot be verified, `bootstrap.py` exits
non-zero before uvicorn binds the port, so the container never reports healthy
without a usable model. There is **no fallback to the base model**.
