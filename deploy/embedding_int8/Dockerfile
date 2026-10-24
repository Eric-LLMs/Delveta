# Pinned by digest: the tag can move, the digest cannot.
FROM python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5

RUN pip install --no-cache-dir \
    "onnxruntime==1.30.0" \
    "tokenizers>=0.20" \
    "fastapi" \
    "uvicorn" \
    "numpy"

# model_store is the shared volume bootstrap; the service owns only its own
# bootstrap.py specification. Build context is ./deploy.
COPY model_store /app/model_store
COPY embedding_int8/server.py /app/server.py
COPY embedding_int8/bootstrap.py /app/bootstrap.py
WORKDIR /app
EXPOSE 80

# The entrypoint initialises the Docker-managed model volume (download + verify
# + atomic publish) and exits non-zero if it cannot, before uvicorn ever binds.
ENTRYPOINT ["python", "/app/bootstrap.py"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
