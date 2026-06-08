
# ── Configuration ──────────────────────────────────────────────────────────────

# IVCAP Service ID
SERVICE_ID := urn:ivcap:service:7c9e66d9-74fa-4c8e-8f55-1d39b8204f14

# Docker image name and tag
DOCKER_IMAGE := image-classify-app
DOCKER_TAG := latest

# Version information (derived from git and date)
GIT_VERSION := $(shell git describe --tags --always 2>/dev/null || echo "0.1.1")
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE := $(shell date -u '+%Y-%m-%dT%H:%M:%S%z' | sed 's/\(..\)$$/:\1/')
VERSION := $(GIT_VERSION)|$(GIT_COMMIT)|$(DATE)

# Default Python version
PYTHON := python3.11

# Run directory for all pipeline outputs (cleared before each run)
RUN_DIR ?= ./run
# Output directory for pipeline artifacts
OUT_DIR ?= $(RUN_DIR)/outputs
IN_DIR ?= $(OUT_DIR)

help:
	@echo "Bird Species Classification Pipeline - Available Commands"
	@echo ""
	@echo "One-time Setup (run before first pipeline execution):"
	@echo "  make prepare-model    Download EfficientNetB2 from HuggingFace & upload as IVCAP artifact"
	@echo "  make download-model   Download EfficientNetB2 locally only (no IVCAP upload)"
	@echo "  make prepare-data     Download sample bird images & upload each as a standalone IVCAP artifact in a collection"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install dependencies (production)"
	@echo "  make dev-install      Install dependencies (with dev tools)"
	@echo "  make poetry-init      Initialize Poetry (run once, if needed)"
	@echo "  make update           Update all dependencies"
	@echo ""
	@echo "Running Pipeline Stages (direct):"
	@echo "  make run-stage1       Run Stage 1: Fetch model and bird images"
	@echo "  make run-stage2       Run Stage 2: Preprocess images (EfficientNetImageProcessor)"
	@echo "  make run-stage3       Run Stage 3: Classify bird species (EfficientNetB2)"
	@echo "  make run-all          Run all stages sequentially"
	@echo ""
	@echo "Testing Dispatcher Pattern:"
	@echo "  make test-fetch       Test dispatcher: Stage 1 (fetch)"
	@echo "  make test-preprocess  Test dispatcher: Stage 2 (preprocess)"
	@echo "  make test-classify    Test dispatcher: Stage 3 (classify)"
	@echo "  make test-dispatcher  Run all dispatcher tests"
	@echo ""
	@echo "Development & Testing:"
	@echo "  make test             Run pytest tests"
	@echo "  make lint             Run flake8 linting"
	@echo "  make format           Format code with black"
	@echo "  make clean            Remove artifacts and cache"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image (local, multi-arch)"
	@echo "  make docker-run       Run pipeline in Docker"
	@echo ""
	@echo "IVCAP Upload & Service Management:"
	@echo "  make ivcap-docker-publish  Build linux/amd64 image & push to IVCAP registry"
	@echo "  make register-service    Build+push image, merge workflow, register service"
	@echo "  make ivcap-test-job      Submit a test job to IVCAP (uses ivcap-test-request.json)"
	@echo "  make ivcap-test-job-local  Submit a test job via local IVCAP context"
	@echo ""
	@echo "Variables:"
	@echo "  OUT_DIR               Output directory (default: ./run/outputs)"
	@echo "  CONTEXT               IVCAP context for local job submission (default: local-od)"

# ── One-time Data / Model Preparation ─────────────────────────────────────────

# Detect uploaded EfficientNetB2 model artifact (created by prepare-model)
MODEL_HF_UUID := $(shell ls data/efficientnet-birds-*.zip 2>/dev/null | sed 's|.*efficientnet-birds-||; s|\.zip||' | head -1 || echo '')
MODEL_HF_ARTIFACT_URN := urn:ivcap:artifact:$(MODEL_HF_UUID)

# Detect uploaded bird images collection URN (created by prepare-data, stored in manifest.json)
IMAGES_COLLECTION_URN := $(shell jq -r '.collection_urn // empty' data/manifest.json 2>/dev/null || echo '')

prepare-model:
	@echo "▶ Downloading Birds-Classifier-EfficientNetB2 from HuggingFace..."
	@echo "  This downloads ~35 MB of model weights and uploads them as an IVCAP artifact."
	@mkdir -p ./data
	@env IVCAP_URL=$(shell ivcap context get url) IVCAP_JWT=$(shell ivcap context get access-token) \
		poetry run $(PYTHON) prepare_model.py
	@echo "✓ Model artifact ready. URN is stored in data/model_artifact.json"

download-model:
	@echo "▶ Downloading Birds-Classifier-EfficientNetB2 from HuggingFace (local only)..."
	@mkdir -p ./data
	@poetry run $(PYTHON) prepare_model.py --no-upload
	@echo "✓ Model downloaded to ./data/model (no IVCAP upload)"

prepare-data:
	@echo "▶ Preparing bird images to ./data directory..."
	@env IVCAP_URL=$(shell ivcap context get url) IVCAP_JWT=$(shell ivcap context get access-token) \
		poetry run $(PYTHON) prepare_data.py
	@echo "✓ Bird images prepared and uploaded to IVCAP"

# ── Data Caching ──────────────────────────────────────────────────────────────

cache-data:
	@echo "▶ Caching bird images and model to ./data directory..."
	@mkdir -p ./data
	@env IVCAP_URL=$(shell ivcap context get url) IVCAP_JWT=$(shell ivcap context get access-token) \
		DATA_CACHE_DIR=./data poetry run $(PYTHON) stage1_fetch.py \
		--collection-urn $(IMAGES_COLLECTION_URN) \
		--model-artifact-urn $(MODEL_HF_ARTIFACT_URN)
	@echo "✓ Data cached successfully to ./data"


clean-cache:
	@echo "Cleaning data cache..."
	@rm -rf ./data
	@echo "✓ Data cache cleaned"

# ── Setup & Installation ──────────────────────────────────────────────────────

poetry-init:
	@echo "Initializing Poetry project..."
	@if [ ! -f pyproject.toml ]; then \
		poetry init --no-interaction --name image-classify-app --description "Bird Species Classification Pipeline" --author "Your Name"; \
	else \
		echo "pyproject.toml already exists"; \
	fi

install:
	@echo "Installing production dependencies..."
	poetry install

dev-install:
	@echo "Installing dependencies (including dev tools)..."
	poetry install

update:
	@echo "Updating all dependencies..."
	poetry update

# ── Pipeline Stages ──────────────────────────────────────────────────────────

reset-run:
	@echo "▶ Clearing run directory: $(RUN_DIR)..."
	@rm -rf $(RUN_DIR)
	@mkdir -p $(RUN_DIR)
	@echo "✓ Run directory reset"

run-stage1:
	@echo "▶ Stage 1: Fetching EfficientNetB2 model and bird images..."
	@mkdir -p $(OUT_DIR)
	@env IVCAP_URL=$(shell ivcap context get url) IVCAP_JWT=$(shell ivcap context get access-token) \
		OUT_DIR=$(OUT_DIR) poetry run $(PYTHON) stage1_fetch.py \
		--collection-urn $(IMAGES_COLLECTION_URN) \
		--model-artifact-urn $(MODEL_HF_ARTIFACT_URN)
	@echo "✓ Stage 1 complete"


run-stage2: run-stage1
	@echo "▶ Stage 2: Preprocessing images with EfficientNetImageProcessor..."
	@IN_DIR=$(OUT_DIR) OUT_DIR=$(OUT_DIR) poetry run $(PYTHON) stage2_preprocess.py
	@echo "✓ Stage 2 complete"

run-stage3: run-stage2
	@echo "▶ Stage 3: Classifying bird species..."
	@IN_DIR=$(OUT_DIR) OUT_DIR=$(OUT_DIR) poetry run $(PYTHON) stage3_classify.py
	@echo "✓ Stage 3 complete"

run-all: run-stage3
	@echo ""
	@echo "✓✓✓ Pipeline complete!"
	@echo "Results written to: $(OUT_DIR)/result.ivcap.json"
	@if [ -f $(OUT_DIR)/result.ivcap.json ]; then \
		echo ""; \
		echo "Preview:"; \
		head -30 $(OUT_DIR)/result.ivcap.json; \
	fi

# ── Dispatcher Pattern Tests ──────────────────────────────────────────────────

test-fetch:
	@echo "▶ Testing dispatcher: Stage 1 (fetch)..."
	@mkdir -p $(OUT_DIR)
	@env IVCAP_URL=$(shell ivcap context get url) IVCAP_JWT=$(shell ivcap context get access-token) \
		LOG_LEVEL=INFO poetry run $(PYTHON) dispatcher.py --stage fetch \
		--collection-urn $(IMAGES_COLLECTION_URN) \
		--model-artifact-urn $(MODEL_HF_ARTIFACT_URN) \
		--out-dir $(OUT_DIR)
	@echo "✓ Dispatcher fetch test complete"


test-preprocess: test-fetch
	@echo "▶ Testing dispatcher: Stage 2 (preprocess)..."
	@LOG_LEVEL=INFO poetry run $(PYTHON) dispatcher.py --stage preprocess --in-dir $(OUT_DIR) --out-dir $(OUT_DIR)
	@echo "✓ Dispatcher preprocess test complete"

test-classify: test-preprocess
	@echo "▶ Testing dispatcher: Stage 3 (classify)..."
	@LOG_LEVEL=INFO poetry run $(PYTHON) dispatcher.py --stage classify --in-dir $(OUT_DIR) --out-dir $(OUT_DIR)
	@echo "✓ Dispatcher classify test complete"

test-dispatcher: test-classify
	@echo ""
	@echo "✓✓✓ Dispatcher pipeline test complete!"
	@echo "Results written to: $(OUT_DIR)/result.ivcap.json"
	@if [ -f $(OUT_DIR)/result.ivcap.json ]; then \
		echo ""; \
		echo "Preview:"; \
		head -30 $(OUT_DIR)/result.ivcap.json; \
	fi

# ── Development & Testing ────────────────────────────────────────────────────

test:
	@echo "Running tests..."
	@poetry run pytest -v

lint:
	@echo "Running flake8 linting..."
	@poetry run flake8 stage*.py --max-line-length=100 --ignore=E501,W503

format:
	@echo "Formatting code with black..."
	@poetry run black stage*.py --line-length=100

clean:
	@echo "Cleaning up artifacts and cache..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache *.pyc
	@rm -rf $(OUT_DIR)/model $(OUT_DIR)/images $(OUT_DIR)/tensors $(OUT_DIR)/*.json
	@echo "✓ Clean complete"

# ── Docker helpers ─────────────────────────────────────────────────────────────

docker-build:
	@echo "Building Docker image..."
	@docker build -t $(DOCKER_IMAGE):latest .

docker-run: reset-run
	@echo "▶ Running Stage 1 (Fetch) in Docker..."
	@docker run --rm -v $(RUN_DIR):/workspace \
		-e IVCAP_URL=$(shell ivcap context get url) \
		-e IVCAP_JWT=$(shell ivcap context get access-token) \
		-e LOG_LEVEL=INFO \
		$(DOCKER_IMAGE):latest \
		--stage fetch --out-dir /workspace/outputs \
		--collection-urn $(IMAGES_COLLECTION_URN) \
		--model-artifact-urn $(MODEL_HF_ARTIFACT_URN) \
		--limit 5
	@echo "✓ Stage 1 complete"
	@echo ""
	@echo "▶ Running Stage 2 (Preprocess) in Docker..."
	@docker run --rm -v $(RUN_DIR):/workspace \
		-e LOG_LEVEL=INFO \
		$(DOCKER_IMAGE):latest \
		--stage preprocess --in-dir /workspace/outputs --out-dir /workspace/outputs
	@echo "✓ Stage 2 complete"
	@echo ""
	@echo "▶ Running Stage 3 (Classify) in Docker..."
	@docker run --rm -v $(RUN_DIR):/workspace \
		-e LOG_LEVEL=INFO \
		$(DOCKER_IMAGE):latest \
		--stage classify --in-dir /workspace/outputs --out-dir /workspace/outputs
	@echo "✓ Stage 3 complete"
	@echo ""
	@echo "✓✓✓ Docker pipeline complete!"
	@echo "Results written to: $(RUN_DIR)/outputs/result.ivcap.json"
	@if [ -f $(RUN_DIR)/outputs/result.ivcap.json ]; then \
		echo ""; \
		echo "Preview:"; \
		head -30 $(RUN_DIR)/outputs/result.ivcap.json; \
	fi

# ── Service Management ────────────────────────────────────────────────────────

DOCKER_TAG=${GIT_COMMIT}
register-service: ivcap-docker-publish
	@echo "▶ Merging IVCAP service definition with workflow..."
	./merge-ivcap-workflow.sh ivcap.yml image-classify-workflow.yaml ivcap-service-with-workflow.yaml
	@echo "▶ Replacing Docker image placeholder with $(DOCKER_IMAGE)..."
	@sed -i '' 's|@DOCKER_IMAGE@|$(shell ivcap package list $(DOCKER_IMAGE)_amd64:${DOCKER_TAG})|g' ivcap-service-with-workflow.yaml
	@echo ""
	@echo "✓ Service definition merged: ivcap-service-with-workflow.yaml"
	@echo "▶ Register service with IVCAP"
	ivcap df update ${SERVICE_ID} -f ivcap-service-with-workflow.yaml

DOCKER_TAG=${GIT_COMMIT}
ivcap-docker-publish:
	@echo "▶ Building Docker image for IVCAP service..."
	@echo "  Version: $(VERSION)"
	docker buildx build \
		-t $(DOCKER_IMAGE)_amd64:${DOCKER_TAG} \
		--platform linux/amd64 \
		--build-arg VERSION="$(VERSION)" \
		--build-arg BUILD_PLATFORM=linux/amd64 \
		-f Dockerfile \
		--load .
	@echo "------------------------------------------------------"
	@echo "▶ Uploading Docker image for IVCAP service..."
	ivcap package push $(DOCKER_IMAGE)_amd64:${DOCKER_TAG}

ivcap-test-job:
	@echo "▶ Submitting test job to IVCAP..."
	ivcap job create ${SERVICE_ID} -f ivcap-test-request.json --stream

CONTEXT=local-od
ivcap-test-job-local:
	@echo "▶ Submitting test job to IVCAP..."
	ivcap \
		--context ${CONTEXT} \
		--access-token $(shell ivcap context get access-token) \
		job create ${SERVICE_ID} -f ivcap-test-request.json --stream

# ── Info ───────────────────────────────────────────────────────────────────────

info:
	@echo "Project Information:"
	@echo "  Python version: $$($(PYTHON) --version)"
	@poetry --version
	@echo ""
	@echo "Artifact URNs (detected from data/ directory):"
	@echo "  Images collection : $(IMAGES_COLLECTION_URN)"
	@echo "  Model  : $(MODEL_HF_ARTIFACT_URN)"
	@echo ""
	@echo "Dependencies:"
	@poetry show --tree

.PHONY: help install dev-install update poetry-init prepare-model download-model prepare-data cache-data clean-cache \
        reset-run run-stage1 run-stage2 run-stage3 run-all \
        test-fetch test-preprocess test-classify test-dispatcher \
        test lint format clean docker-build docker-run info \
        ivcap-docker-publish register-service ivcap-test-job ivcap-test-job-local
