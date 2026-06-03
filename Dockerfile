# Bird species classification pipeline container
# Includes all pipeline modules and their dependencies via Poetry
# EfficientNetB2 model is NOT baked in — it is loaded at runtime from an IVCAP artifact
FROM python:3.11-slim

LABEL description="Bird species classification pipeline with EfficientNetB2 (HuggingFace Transformers)"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Set working directory for application code
WORKDIR /app

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Install CPU-only PyTorch BEFORE Poetry resolves dependencies.
# The default PyPI torch wheel bundles CUDA support and pulls ~4 GB of
# nvidia-cuda-*, nvidia-cublas-*, nvidia-cudnn-* packages which bloat the
# image and stall builds on machines without GPU drivers.
# The pytorch.org/whl/cpu index provides the same torch at a fraction of the size.
RUN pip install --no-cache-dir "torch>=2.0.0" \
  --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies via Poetry.
# torch is already present in the environment so Poetry will skip it.
RUN poetry config virtualenvs.create false && \
  poetry install --only main --no-interaction --no-ansi

# Copy all pipeline scripts
COPY dispatcher.py .
COPY stage1_fetch.py .
COPY stage2_preprocess.py .
COPY stage3_classify.py .
COPY run.sh .

# Make run.sh executable
RUN chmod +x run.sh

# Create workspace directory for data/outputs
RUN mkdir -p /workspace

# VERSION INFORMATION
ARG VERSION=???
ENV VERSION=$VERSION

# The entry point will be called by the workflow with stage-specific arguments
# Example: ./run.sh --stage fetch --out-dir /workspace/outputs
ENTRYPOINT ["/app/run.sh"]

# Default command (can be overridden)
CMD ["--help"]
