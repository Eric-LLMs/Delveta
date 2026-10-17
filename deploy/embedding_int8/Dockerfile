FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "onnxruntime==1.30.0" \
    "tokenizers>=0.20" \
    "fastapi" \
    "uvicorn" \
    "numpy"

COPY server.py /app/server.py
WORKDIR /app
EXPOSE 80

# MODEL_DIR points at the read-only bind mount carrying the validated
# Delveta-BGE-M3-v1 INT8 ONNX graph (model.onnx + tokenizer.json).
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
