# Argument-extractor sidecar: Ollama serving the fine-tuned Delveta PE model.
#
# The upstream image is kept stock (no curl, no apt): downloading is the
# `tool-intent-model` init service's job (deploy/tool_intent/init.Dockerfile,
# backed by the shared model_store). This image only registers the GGUF that the
# init service placed in the `toolintent-cache` volume, then serves.
#
# Pinned by digest: the tag can move, the digest cannot.
FROM ollama/ollama@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551

COPY tool_intent/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
