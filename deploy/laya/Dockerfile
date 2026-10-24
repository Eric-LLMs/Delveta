# Delveta-LayaChoice sidecar: serves the fine-tuned Laya (System-1) decision
# model over the TypeSafe Jev /v1/systemone protocol for the Intent Funnel's
# cap_router lane. The ~1.3 GB checkpoint is NOT baked into the image: the
# entrypoint downloads the pinned revision into the Docker-managed volume and
# verifies it before the server starts.
# Pinned by digest: the tag can move, the digest cannot.
FROM python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5

WORKDIR /app
COPY laya/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Self-contained: the 256-token option builder the checkpoint was trained under
# (deliberately copied from the fine-tuning script so the service never imports
# it), the thin HTTP wrapper, and the volume bootstrap. Build context is ./deploy.
COPY model_store /app/model_store
COPY laya/layachoice_sequence.py laya/server.py laya/bootstrap.py ./

EXPOSE 8000

# Fail closed: a missing or unverifiable checkpoint exits non-zero before the
# HTTP server exists, so an empty volume is never reported as a healthy model.
ENTRYPOINT ["python", "/app/bootstrap.py"]
CMD ["python", "server.py"]
