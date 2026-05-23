# Image classification pipeline container
# Includes all pipeline modules and their dependencies via Poetry
FROM python:3.11-slim

LABEL description="Image classification pipeline with MobileNetV2 ONNX model"

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

# Install Python dependencies via Poetry
# Configure Poetry to not create a virtual environment (install globally in container)
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

# The entry point will be called by the workflow with stage-specific arguments
# Example: ./run.sh --stage fetch --out-dir /workspace/outputs
ENTRYPOINT ["/app/run.sh"]

# Default command (can be overridden)
CMD ["--help"]
