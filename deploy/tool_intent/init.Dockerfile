# One-shot init service for the argument-extractor model.
#
# Downloads the pinned GGUF from Hugging Face into the `toolintent-cache` volume
# and verifies it. All download/verify/lock/resume logic lives in the shared
# `model_store` package — this image adds no second implementation. It exits 0
# once the bytes are published (or already present) and non-zero on any failure,
# and compose starts `tool-intent` only after a successful exit.
#
# Pinned by digest: the tag can move, the digest cannot.
FROM python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5

# Build context is ./deploy.
COPY model_store /app/model_store
COPY tool_intent/fetch_model.py /app/fetch_model.py
WORKDIR /app

ENTRYPOINT ["python", "/app/fetch_model.py"]
