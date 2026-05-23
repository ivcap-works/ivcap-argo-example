# Image Classification Pipeline — Argo Workflow Demo

A self-contained, three-stage Argo Workflow that classifies images using a
pretrained **MobileNetV2** model via **ONNX Runtime** on CPU.
No GPU, no model training, no external model registry required.

---

## Overview

```
Stage 1 – fetch        Stage 2 – preprocess      Stage 3 – classify
─────────────────      ────────────────────────   ──────────────────────────
Download               Resize → 256 px            Load model (ONNX Runtime)
  mobilenetv2-12.onnx  Centre-crop → 224×224      Run inference (CPU)
  imagenet_classes.txt Normalise per channel       Apply softmax
  sample images        Convert → NCHW float32      Write top-5 results
       │                       │                          │
       └── manifest.json ──────┘                          │
                               └── tensors/*.npy ─────────┘
                                                          │
                                                   results.json
```

All three stages run as separate Kubernetes Pods and share a single
`emptyDir` volume (`/workspace`), so no object store (S3/MinIO) is needed
for a demo deployment.

---

## Architecture & Data Flow

This project follows a **producer-consumer pattern** using IVCAP artifacts:

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA PREPARATION (Local Development, One-Time Setup)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  prepare_data.py (local script, not in Docker)                     │
│      ↓                                                              │
│  1. Download MobileNetV2-12 ONNX model from Hugging Face          │
│     → ./data/model/mobilenetv2-12.onnx                             │
│                                                                     │
│  2. Download ImageNet class labels                                 │
│     → ./data/model/imagenet_classes.txt                            │
│                                                                     │
│  3. Download sample images from Imagenette dataset                 │
│     → ./data/images/image_XXXX.jpg                                 │
│                                                                     │
│  4. Create manifest.json (list of image filenames)                 │
│     → ./data/manifest.json                                         │
│                                                                     │
│  5. ZIP directories and upload to IVCAP                            │
│     → images-{UUID}.zip (IVCAP artifact)                           │
│     → model-{UUID}.zip (IVCAP artifact)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                       IVCAP DataFabric
                    (artifact storage)
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE EXECUTION (Argo Workflow on Kubernetes)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: Fetch (stage1_fetch.py, in Docker)                      │
│      Downloads from IVCAP artifacts using URNs:                    │
│      → Artifact URN: urn:ivcap:artifact:{images-UUID}              │
│      → Artifact URN: urn:ivcap:artifact:{model-UUID}               │
│      → Extracts to /workspace/outputs/ on emptyDir volume          │
│                                                                     │
│  Stage 2: Preprocess (stage2_preprocess.py, in Docker)            │
│      Reads images from /workspace/outputs/images/                  │
│      → Resizes, crops, normalizes → tensors/*.npy                  │
│                                                                     │
│  Stage 3: Classify (stage3_classify.py, in Docker)                │
│      Reads tensors and model from /workspace/outputs/              │
│      → Runs ONNX inference                                         │
│      → Produces results.json with top-5 predictions                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

**Artifacts**: ZIP files uploaded to IVCAP that contain reusable data
- **Images artifact**: Contains all sample images (image_0000.jpg, etc.)
- **Model artifact**: Contains the ONNX model and class labels

**Dispatcher Pattern**: The `dispatcher.py` script routes invocations to
different stage functions based on `--stage` argument. This allows the same
Docker image to run all three stages with different entry points.

---

## Files

| File | Purpose |
|------|---------|
| `image-classify-workflow.yaml` | Argo Workflow definition (entrypoint) |
| `dispatcher.py` | Unified entry point routing CLI invocations to stage functions |
| `stage1_fetch.py` | Download model and images from IVCAP artifacts |
| `stage2_preprocess.py` | Preprocess images into NCHW float32 tensors |
| `stage3_classify.py` | Run inference, produce `results.json` |
| `prepare_data.py` | **LOCAL ONLY**: Prepare & upload data to IVCAP (one-time setup) |
| `Dockerfile` | Docker image packaging all Python modules and dependencies |
| `Makefile` | Convenience commands for local development & testing |
| `README.md` | This file |

---

## Data Preparation — Creating the Artifacts

### Why Data Preparation?

Before running the workflow, you need to:
1. Download the MobileNetV2-12 ONNX model (~13 MB)
2. Download ImageNet class labels (~100 KB)
3. Obtain sample images for classification

The `prepare_data.py` script automates this and uploads the data to IVCAP as
**reusable artifacts**. This is a one-time setup step done locally before
deploying to Kubernetes.

### Prerequisites for Data Preparation

- Python 3.11+
- Poetry (for dependency management)
- IVCAP CLI configured and authenticated
- Internet access (to download model and images from Hugging Face & PyTorch)

Install dependencies:
```bash
poetry install --with dev
```

### Step 1: Prepare Data Locally

```bash
make prepare-data
```

Or with custom settings:
```bash
poetry run python prepare_data.py \
  --data-dir ./data \
  --num-images 5 \
  --no-upload  # Skip IVCAP upload (useful for testing)
```

**What happens:**

1. **Downloads model** (13.3 MB):
   ```
   🤖 Model:
     ✓ Downloading MobileNetV2-12 ONNX model... ✓ (13.3 MB)
   ```
   → `./data/model/mobilenetv2-12.onnx`

2. **Downloads ImageNet labels**:
   ```
   📝 ImageNet Labels:
     ✓ Downloading ImageNet class labels... ✓ (1000 classes)
   ```
   → `./data/model/imagenet_classes.txt`

3. **Downloads sample images** from Imagenette dataset (5 images by default):
   ```
   🖼️  Sample Images:
     ✓ Extracted image_0000.jpg (42.3 KB)
     ✓ Extracted image_0001.jpg (38.5 KB)
     ✓ Extracted image_0002.jpg (35.1 KB)
     ✓ Extracted image_0003.jpg (41.2 KB)
     ✓ Extracted image_0004.jpg (39.8 KB)
   ```
   → `./data/images/image_XXXX.jpg`

4. **Creates manifest** listing all images:
   ```json
   {
     "images": [
       "image_0000.jpg",
       "image_0001.jpg",
       "image_0002.jpg",
       "image_0003.jpg",
       "image_0004.jpg"
     ]
   }
   ```
   → `./data/manifest.json`

5. **ZIPs directories** for upload:
   ```
   📦 Creating zip archive...
     ✓ images.zip created (0.2 MB)
   ```
   ```
   📦 Creating model zip archive...
     ✓ model.zip created (13.4 MB)
   ```

6. **Uploads to IVCAP** and renames with artifact UUIDs:
   ```
   ☁️  Uploading artifact to IVCAP...
     ✓ Artifact uploaded with URN: urn:ivcap:artifact:fac6093e-d142-4958-9241-b02a81abb5d0
     ✓ Renamed to: images-fac6093e-d142-4958-9241-b02a81abb5d0.zip
   ```
   ```
   ☁️  Uploading artifact to IVCAP...
     ✓ Artifact uploaded with URN: urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90
     ✓ Renamed to: model-0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90.zip
   ```

After successful preparation, your `./data` directory contains:

```
data/
├── images/
│   ├── image_0000.jpg
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   ├── image_0003.jpg
│   ├── image_0004.jpg
│   └── images-{UUID}.zip  ← Artifact file (can be deleted, stored in IVCAP)
├── model/
│   ├── imagenet_classes.txt
│   ├── mobilenetv2-12.onnx
│   └── model-{UUID}.zip    ← Artifact file (can be deleted, stored in IVCAP)
└── manifest.json
```

### Step 2: Extract Artifact URNs

The Makefile automatically extracts artifact UUIDs from the ZIP filenames:

```bash
# View the extracted URNs
grep -o "urn:ivcap:artifact:[a-f0-9\-]*" Makefile | head -2
```

Output:
```
IMAGES_ARTIFACT_URN := urn:ivcap:artifact:fac6093e-d142-4958-9241-b02a81abb5d0
MODEL_ARTIFACT_URN := urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90
```

These URNs are used by `stage1_fetch.py` to download data from IVCAP during
pipeline execution.

---

## Model

**MobileNetV2-12 (fp32)** from the [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/mobilenetv2-12).

| Property | Value |
|----------|-------|
| Size | 13.3 MB |
| Input | `data` — shape `(1, 3, 224, 224)`, NCHW, float32 |
| Output | `output` — shape `(1, 1000)`, raw logits |
| Classes | 1000 ImageNet classes |
| Top-1 accuracy | 69.48 % |
| Top-5 accuracy | 89.26 % |
| License | Apache 2.0 |

Preprocessing (applied in Stage 2):
- Resize shortest side to 256 px, preserving aspect ratio
- Centre-crop to 224 × 224
- Scale to [0, 1], then normalise:
  ```
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
  ```

---

## Prerequisites

### For Local Development & Data Preparation

- Python 3.11+
- Poetry (for dependency management)
- IVCAP CLI and authentication (`ivcap context get url`, `ivcap context get access-token`)
- Internet access (to download model, labels, and images)
- Optional: `torchvision` for loading Imagenette dataset

### For Argo Workflow Deployment

- A running Argo Workflows installation (v3.x recommended):
  ```bash
  kubectl apply -n argo \
    -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
  ```
- `kubectl` configured to reach your cluster
- IVCAP access from pods (environment variables: `IVCAP_URL`, `IVCAP_JWT`)
- Outbound internet access from pods (to fetch IVCAP artifacts)

Python dependencies (installed per-pod at runtime):
- Stage 1: `ivcap-client` (to download artifacts)
- Stage 2: `Pillow==10.3.0`, `numpy`
- Stage 3: `onnxruntime==1.18.1`, `numpy`

---

## Quick Start

### 1. Setup (One-Time)

```bash
# Install dependencies
make dev-install

# Prepare data and upload artifacts to IVCAP
make prepare-data
```

### 2. Local Testing (Optional)

Run the pipeline locally to verify everything works:

```bash
# Run all three stages sequentially
make run-all

# Or test dispatcher pattern
make test-dispatcher
```

### 3. Deploy to Kubernetes

```bash
# Submit to Argo Workflows
argo submit image-classify-workflow.yaml --watch

# Or with custom parameters
argo submit image-classify-workflow.yaml \
  -p onnxruntime_version=1.18.1 \
  -p log_level=DEBUG \
  --watch
```

---

## Local Development & Testing

### Running Stages Individually

```bash
# Stage 1: Fetch (downloads from IVCAP)
make run-stage1

# Stage 2: Preprocess (creates tensors)
make run-stage2

# Stage 3: Classify (runs inference)
make run-stage3

# All stages in sequence
make run-all
```

All outputs are written to `./run/outputs/`.

### Testing the Dispatcher Pattern

The dispatcher pattern allows a single Docker image to run all three stages:

```bash
make test-dispatcher
```

This tests:
```bash
python dispatcher.py --stage fetch --out-dir ./run/outputs
python dispatcher.py --stage preprocess --in-dir ./run/outputs --out-dir ./run/outputs
python dispatcher.py --stage classify --in-dir ./run/outputs --out-dir ./run/outputs
```

### Cleaning Up

```bash
# Remove run outputs
make clean

# Remove data cache (model, images, zips)
make clean-cache
```

---

## Deploying the Workflow to Kubernetes

The workflow is deployed via a **single Docker image** that contains all pipeline
code and dependencies. Each stage invokes the same image with different
`--stage` parameters.

> **Note on Architecture**: This demo uses a single image for simplicity. In a more
> involved production pipeline, you would typically build **separate Docker images for
> each stage** (e.g., `fetch:v1.0`, `preprocess:v1.0`, `classify:v1.0`), each with
> only the dependencies it needs. This approach enables independent versioning,
> optimization, and scaling of each stage. The dispatcher pattern and Argo templates
> shown here would remain largely the same—just pointing to different image tags per stage.

### Building the Docker Image

Create a Dockerfile that installs dependencies via Poetry and includes all pipeline scripts:

```dockerfile
FROM python:3.11-slim

# Install Poetry
RUN pip install --no-cache-dir poetry

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (production only, skip dev dependencies)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Copy pipeline scripts
COPY dispatcher.py stage1_fetch.py stage2_preprocess.py stage3_classify.py ./
```

Build and push to your registry:

```bash
docker build -t myregistry/image-classify:latest .
docker push myregistry/image-classify:latest
```

### Argo Workflow Configuration

The Argo Workflow (`image-classify-workflow.yaml`) uses **pipeline parameters**
to pass artifact URNs and configuration to each stage:

```yaml
arguments:
  parameters:
    - name: images_artifact_urn
      value: "urn:ivcap:artifact:fac6093e-d142-4958-9241-b02a81abb5d0"
    - name: model_artifact_urn
      value: "urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90"
    - name: image_version
      value: "latest"
```

Each stage template invokes the same Docker image with different parameters:

```yaml
templates:
  - name: fetch
    container:
      image: "myregistry/image-classify:{{workflow.parameters.image_version}}"
      command: ["python", "dispatcher.py"]
      args:
        - "--stage=fetch"
        - "--images-artifact-urn={{workflow.parameters.images_artifact_urn}}"
        - "--model-artifact-urn={{workflow.parameters.model_artifact_urn}}"
        - "--out-dir=/workspace/outputs"

  - name: preprocess
    container:
      image: "myregistry/image-classify:{{workflow.parameters.image_version}}"
      command: ["python", "dispatcher.py"]
      args:
        - "--stage=preprocess"
        - "--in-dir=/workspace/outputs"
        - "--out-dir=/workspace/outputs"

  - name: classify
    container:
      image: "myregistry/image-classify:{{workflow.parameters.image_version}}"
      command: ["python", "dispatcher.py"]
      args:
        - "--stage=classify"
        - "--in-dir=/workspace/outputs"
        - "--out-dir=/workspace/outputs"
```

### Submitting the Workflow

Submit with default parameters:

```bash
argo submit image-classify-workflow.yaml --watch
```

Or override parameters at submission time:

```bash
argo submit image-classify-workflow.yaml \
  -p images_artifact_urn=urn:ivcap:artifact:xxx \
  -p model_artifact_urn=urn:ivcap:artifact:yyy \
  -p image_version=v1.0.0 \
  --watch
```

---

## Running the Workflow

```bash
# Submit with defaults (uses artifact URNs from Makefile)
argo submit image-classify-workflow.yaml --watch

# Override image version
argo submit image-classify-workflow.yaml \
  -p image_version=latest \
  --watch

# Check logs per stage
argo logs <workflow-name> fetch
argo logs <workflow-name> preprocess
argo logs <workflow-name> classify

# Watch workflow status
argo get <workflow-name>
argo get <workflow-name> --log
```

---

## Expected Output

`results.json` (written to `/workspace/outputs/`) contains top-5 predictions
for each image:

```json
{
  "model": "mobilenetv2-12",
  "results": [
    {
      "image": "image_0000.jpg",
      "inference_ms": 120.4,
      "top5": [
        { "rank": 1, "label": "tabby",        "score": 0.412 },
        { "rank": 2, "label": "tiger cat",    "score": 0.298 },
        { "rank": 3, "label": "Egyptian cat", "score": 0.181 },
        { "rank": 4, "label": "lynx",         "score": 0.031 },
        { "rank": 5, "label": "Persian cat",  "score": 0.018 }
      ]
    },
    {
      "image": "image_0001.jpg",
      "inference_ms": 118.2,
      "top5": [
        { "rank": 1, "label": "golden retriever", "score": 0.512 },
        { "rank": 2, "label": "Labrador retriever", "score": 0.298 },
        { "rank": 3, "label": "English springer", "score": 0.089 },
        { "rank": 4, "label": "pointer",         "score": 0.082 },
        { "rank": 5, "label": "Great Pyrenees",  "score": 0.019 }
      ]
    }
  ]
}
```

---

## Scaling to Larger Batches

To classify many images in parallel, modify `image-classify-workflow.yaml` to
use `withParam` loops:

```yaml
steps:
  - - name: preprocess
      template: preprocess-images
      arguments:
        parameters:
          - name: image_file
            value: "{{item}}"
      withParam: "{{steps.fetch.outputs.parameters.image-list}}"
```

This creates one pod per image, enabling horizontal scaling.

---

## Troubleshooting

### IVCAP Authentication Issues

Ensure `ivcap context` is configured:
```bash
ivcap context get url
ivcap context get access-token
```

The Makefile passes these via environment variables:
```bash
IVCAP_URL=$(ivcap context get url) \
IVCAP_JWT=$(ivcap context get access-token)
```

### Model/Image Download Failures

If `prepare_data.py` fails to download:
- Check internet connectivity
- Verify Hugging Face and PyTorch URLs are accessible
- Retry (script has automatic retry logic with exponential backoff)

### ONNX Runtime Compatibility

If you encounter ONNX Runtime build issues, see `requirements-optional.txt`
for platform-specific installation instructions or use conda:

```bash
conda install -c conda-forge onnxruntime
```

### Docker Build Issues

Ensure Python 3.11 base image is available:
```bash
docker build -t image-classify-app:latest .
```

---

## References

- [MobileNetV2 ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/mobilenetv2-12)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [Argo Workflows documentation](https://argoproj.github.io/argo-workflows/)
- [IVCAP Client documentation](https://docs.ivcap.io/)
- [ImageNet class labels](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)
- [Imagenette Dataset](https://github.com/fastai/imagenette)
- MobileNetV2 paper: [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
