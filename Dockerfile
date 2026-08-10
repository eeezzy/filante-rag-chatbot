# BGE-M3 (~2GB weights) is baked into the image at build time, not
# downloaded on first request, so cold starts don't depend on HuggingFace
# Hub being reachable/fast. Same for the pre-built Qdrant index — it's
# tiny (~5MB) and regenerating it needs the full ingestion pipeline, so
# shipping the already-built data/processed + data/qdrant is far simpler
# than rebuilding the index in the container.
FROM python:3.12-slim

WORKDIR /app

# FlagEmbedding depends on torch, and a plain `pip install` resolves the
# default PyPI wheel — which bundles ~1GB+ of NVIDIA CUDA libraries meant
# for GPU machines. Completely wasted on a CPU-only Fly.io VM, and the
# actual cause of a build that looked like a flaky network timeout but was
# really just downloading gigabytes it didn't need. Installing the CPU-only
# build explicitly first keeps pip's resolver from ever reaching for it.
# This step doesn't depend on our source, so it stays cached independently
# of source/dependency-list changes below.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# A regular (non-editable) install, and — unlike an editable install, which
# only creates a link back to wherever the source lives at install time —
# needs the actual source present first, so pyproject.toml + src/ are
# copied together rather than pyproject.toml alone for better layer
# caching. (First pass at this had them in the wrong order: `pip install
# -e .` before `COPY src/`, which left the editable link pointing at
# nothing and crashed at import time — caught by actually running the
# built image, not just building it.)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[retrieval,api]"

# Bake the embedding model into the image. FlagEmbedding's loader always
# fetches the *entire* HF repo regardless of ignore_patterns — including a
# ~2.2GB ONNX export we never use (we load via PyTorch, not onnxruntime),
# which doubled the model cache to 4.3GB for zero benefit. Verified BGE-M3
# has no safetensors variant either, so pytorch_model.bin (~2.2GB) is
# unavoidable — but the onnx/ files are pure waste, so delete them (via
# their resolved blob, not just the symlink — HF's cache is content-
# addressed) once the model has already loaded into memory and no longer
# needs them on disk. Confirmed empirically: model still loads correctly
# afterward, cache drops from 4.3GB to 2.2GB.
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)" \
    && for f in $(find /root/.cache/huggingface -path "*/onnx/model.onnx*" -type l); do readlink -f "$f"; done | sort -u | xargs rm -f \
    && find /root/.cache/huggingface -path "*/onnx" -type d -exec rm -rf {} +

COPY data/processed/ ./data/processed/
COPY data/qdrant/ ./data/qdrant/

EXPOSE 8000

# --workers 1: Qdrant's local on-disk mode is single-process (see
# vector_store.py) — a real high-traffic deployment would move Qdrant to
# a server and could then scale workers.
CMD ["uvicorn", "filante_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
