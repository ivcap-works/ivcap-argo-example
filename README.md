# Bird Species Classification — IVCAP Argo Workflow

A three-stage **Argo Workflow** that classifies bird species in images using a
pretrained **EfficientNetB2** model
([dennisjooo/Birds-Classifier-EfficientNetB2](https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2))
via HuggingFace Transformers. The model is fetched from an IVCAP artifact at
runtime — it is never baked into the Docker image.

> For a full explanation of the architecture, design decisions, and IVCAP
> integration see **[DESIGN.md](DESIGN.md)**.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Prerequisites](#prerequisites)
- [One-Time Setup](#one-time-setup)
- [Local Development](#local-development)
  - [Build the Docker image](#build-the-docker-image)
  - [Run the full pipeline locally](#run-the-full-pipeline-locally)
  - [Run stages individually](#run-stages-individually)
- [Deploying to IVCAP / Kubernetes](#deploying-to-ivcap--kubernetes)
  - [Build and push the image](#build-and-push-the-image)
  - [Register the service](#register-the-service)
  - [Submit a job](#submit-a-job)
- [Expected Output](#expected-output)
- [Project Files](#project-files)
- [References](#references)

---

## Pipeline Overview

```mermaid
flowchart LR
    subgraph Setup["One-Time Setup (local)"]
        PM["make prepare-model\n→ uploads EfficientNetB2\n  as IVCAP artifact"]
        PD["make prepare-data\n→ uploads bird images\n  as IVCAP artifact"]
    end

    subgraph Pipeline["Argo Workflow (per job)"]
        S1["Stage 1\nfetch\n──────\ndownload artifacts\n→ /workspace"]
        S2["Stage 2\npreprocess\n──────\nEfficientNetImageProcessor\n→ .npy tensors"]
        S3["Stage 3\nclassify\n──────\nEfficientNetB2 inference\n→ result.ivcap.json"]
    end

    PM -->|model URN| S1
    PD -->|images URN| S1
    S1 -->|sequential| S2
    S2 -->|sequential| S3
```

Each stage runs in its own Kubernetes pod and shares a 2 Gi PVC at
`/workspace`. All three stages use the **same Docker image**, routed via
`dispatcher.py --stage <fetch|preprocess|classify>`.

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python 3.11+ | Local development and setup scripts |
| [Poetry](https://python-poetry.org/) | Dependency management |
| Docker | Build and run the pipeline container |
| [IVCAP CLI](https://docs.ivcap.io/) | Upload artifacts, manage services |
| `ivcap context` configured | Provides `IVCAP_URL` and access token |

Install Python dependencies:
```bash
poetry install
```

---

## One-Time Setup

These steps are run **once** before the first pipeline execution. They download
and upload data to IVCAP as reusable artifacts.

### 1. Prepare the model artifact

Downloads EfficientNetB2 (~35 MB) from HuggingFace Hub and uploads it to IVCAP:

```bash
make prepare-model
```

The artifact URN is stored in `data/model_artifact.json` and the zip is
retained locally so the Makefile can derive the URN automatically.

### 2. Prepare the images artifact

Packages sample bird images and uploads them to IVCAP:

```bash
make prepare-data
```

After both steps, `make info` shows the artifact URNs that will be used at
runtime:

```bash
make info
```

---

## Local Development

### Build the Docker image

```bash
make docker-build
```

### Run the full pipeline locally

Runs all three stages in Docker, mounting `./run` as the shared workspace:

```bash
make docker-run
```

Results are written to `./run/outputs/result.ivcap.json`.

### Run stages individually

Without Docker (using the local Poetry environment):

```bash
make run-stage1     # Stage 1: fetch artifacts from IVCAP
make run-stage2     # Stage 2: preprocess images (requires stage 1 output)
make run-stage3     # Stage 3: classify bird species (requires stage 2 output)
make run-all        # All three stages in sequence
```

Or test the dispatcher pattern:

```bash
make test-dispatcher
```

### Other useful targets

```bash
make clean          # Remove run outputs
make clean-cache    # Remove ./data cache (model, images, zips)
make lint           # flake8
make format         # black
make test           # pytest
```

---

## Deploying to IVCAP / Kubernetes

### Build and push the image

Builds a `linux/amd64` image and pushes it to the IVCAP container registry:

```bash
make ivcap-docker-publish
```

### Register the service

Merges the Argo workflow YAML into the IVCAP service definition and registers
it:

```bash
make register-service
```

This runs `merge-ivcap-workflow.sh` to combine `ivcap.yml` with
`image-classify-workflow.yaml`, then calls `ivcap df update`.

### Submit a job

```bash
# Using ivcap-test-request.json
make ivcap-test-job

# Or via local IVCAP context
make ivcap-test-job-local
```

To submit directly with the `argo` CLI:

```bash
argo submit image-classify-workflow.yaml \
  -p images_artifact_urn=urn:ivcap:artifact:<uuid> \
  -p model_artifact_urn=urn:ivcap:artifact:<uuid> \
  --watch
```

---

## Expected Output

`result.ivcap.json` contains top-5 bird species predictions per image:

```json
{
  "model": "dennisjooo/Birds-Classifier-EfficientNetB2",
  "results": [
    {
      "image": "image_0000.jpg",
      "inference_ms": 182.1,
      "top5": [
        { "rank": 1, "label": "SWINHOES PHEASANT",    "score": 0.9999 },
        { "rank": 2, "label": "BULWERS PHEASANT",     "score": 0.0001 },
        { "rank": 3, "label": "CRESTED FIREBACK",     "score": 0.0 },
        { "rank": 4, "label": "CRESTED WOOD PARTRIDGE","score": 0.0 },
        { "rank": 5, "label": "RED LEGGED HONEYCREEPER","score": 0.0 }
      ]
    }
  ]
}
```

---

## Project Files

| File | Purpose |
|---|---|
| `dispatcher.py` | Entry point — routes `--stage` to the correct stage function |
| `stage1_fetch.py` | Download model and images artifacts from IVCAP |
| `stage2_preprocess.py` | Preprocess images → `(1, 3, 260, 260)` float32 tensors |
| `stage3_classify.py` | Run EfficientNetB2 inference, write `result.ivcap.json` |
| `run.sh` | Thin shell wrapper: `exec python dispatcher.py "$@"` |
| `Dockerfile` | Container image definition |
| `image-classify-workflow.yaml` | Argo Workflow definition (3 stages, shared PVC) |
| `ivcap.yml` | IVCAP service definition (schema, request parameters) |
| `merge-ivcap-workflow.sh` | Merges `ivcap.yml` + workflow YAML for service registration |
| `prepare_model.py` | One-time: download EfficientNetB2 and upload as IVCAP artifact |
| `prepare_data.py` | One-time: package bird images and upload as IVCAP artifact |
| `Makefile` | Convenience targets for all common tasks |
| `DESIGN.md` | Architecture, design decisions, and detailed implementation notes |

---

## References

- [dennisjooo/Birds-Classifier-EfficientNetB2](https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2) — HuggingFace model
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Argo Workflows documentation](https://argoproj.github.io/argo-workflows/)
- [IVCAP documentation](https://docs.ivcap.io/)
