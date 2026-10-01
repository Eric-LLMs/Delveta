# Delveta-LayaChoice sidecar: serves the fine-tuned Laya (System-1) decision
# model over the TypeSafe Jev /v1/systemone protocol for the Intent Funnel's
# cap_router lane. The 1.3 GB checkpoint is NOT baked into the image — it is
# bind-mounted READ-ONLY and pointed at by LAYA_CHOICE_MODEL_DIR (see
# docker-compose.yml), so the image stays small and the weights stay out of git.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Self-contained: the 256-token option builder the checkpoint was trained under
# (deliberately copied from the fine-tuning script so the service never imports
# it), plus the thin HTTP wrapper.
COPY layachoice_sequence.py server.py ./

EXPOSE 8000
CMD ["python", "server.py"]
