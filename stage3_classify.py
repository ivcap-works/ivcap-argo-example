#!/usr/bin/env python3
"""
Stage 3 – Classify bird images with EfficientNetB2 via HuggingFace Transformers.

Loads each preprocessed tensor from Stage 2, runs it through the
EfficientNetForImageClassification model (loaded from the model artifact
directory), and writes the top-5 bird species predictions per image to
result.ivcap.json.

Model input  : pixel_values, shape (1,3,260,260), dtype float32
Model output : logits,       shape (1, N_CLASSES)  where N_CLASSES ≤ 525 bird species
Labels       : model.config.id2label  (no external labels file needed)

Reference model: dennisjooo/Birds-Classifier-EfficientNetB2 (HuggingFace Hub)
The model artifact is fetched by Stage 1 — it is never baked into the Docker image.

Inputs  (from $IN_DIR, default /tmp/outputs):
  - model/config.json
  - model/model.safetensors  (or pytorch_model.bin)
  - tensors/<name>.npy
  - manifest.json

Outputs (written to $OUT_DIR, default /tmp/outputs):
  - result.ivcap.json   (path overridden by $IVCAP_RESULT_PATH; Argo sets it to
                         /result.ivcap.json so the executor can capture it as an
                         output parameter for the IVCAP controller)
"""

import os
import json
import time
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)


def classify_stage(in_dir: str = "/tmp/outputs", out_dir: str = "/tmp/outputs") -> None:
    """Run bird species inference on preprocessed tensors using EfficientNetB2."""
    model_dir = os.path.join(in_dir, "model")
    tensor_dir = os.path.join(in_dir, "tensors")
    top_k = 5

    # ── Load model ────────────────────────────────────────────────────────────────
    try:
        import torch
        from transformers import EfficientNetForImageClassification
    except ImportError:
        logger.error("transformers / torch not installed")
        sys.exit(1)

    logger.info(f"Loading EfficientNetForImageClassification from {model_dir}")
    model = EfficientNetForImageClassification.from_pretrained(model_dir)
    model.eval()

    n_labels = len(model.config.id2label)
    logger.info(f"  Model loaded  ({n_labels} bird species labels)")

    # ── Load manifest ─────────────────────────────────────────────────────────────
    with open(os.path.join(in_dir, "manifest.json")) as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    tensors = manifest.get("tensors", [])

    if len(images) != len(tensors):
        raise RuntimeError(
            f"manifest mismatch: {len(images)} images vs {len(tensors)} tensors"
        )

    def _fname(entry: "str | dict") -> str:
        """Return the filename string from a manifest entry (str or dict)."""
        return entry["filename"] if isinstance(entry, dict) else entry

    def _species(entry: "str | dict") -> "str | None":
        """Return the known species label if available (dict entries only)."""
        return entry.get("species") if isinstance(entry, dict) else None

    # ── Inference ─────────────────────────────────────────────────────────────────
    results = []

    for img_entry, tensor_file in zip(images, tensors):
        img_file = _fname(img_entry)
        known_species = _species(img_entry)
        tensor_path = os.path.join(tensor_dir, tensor_file)
        arr = np.load(tensor_path)  # (1, 3, 260, 260) float32

        logger.info(f"Running inference on {img_file}")
        t0 = time.perf_counter()

        with torch.no_grad():
            pixel_values = torch.from_numpy(arr)
            logits = model(pixel_values=pixel_values).logits  # (1, N_CLASSES)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(logits, dim=-1)[0]  # (N_CLASSES,)
        top_vals, top_idx = torch.topk(probs, k=top_k)

        top5 = [
            {
                "rank": int(i + 1),
                "label": model.config.id2label[int(idx)],
                "score": round(float(val), 4),
            }
            for i, (idx, val) in enumerate(zip(top_idx, top_vals))
        ]

        logger.info(
            f"  {elapsed_ms:.1f} ms  →  top-1: {top5[0]['label']} "
            f"({top5[0]['score']:.3f})"
        )

        result = {
            "image": img_file,
            "inference_ms": round(elapsed_ms, 1),
            "top5": top5,
        }
        if known_species is not None:
            result["known_species"] = known_species
        results.append(result)

    # ── Write results ─────────────────────────────────────────────────────────────
    # $IVCAP_RESULT_PATH is set to /result.ivcap.json by the Argo workflow template
    # so the executor captures it as an output parameter for the IVCAP controller.
    # When running locally the env var is unset and the file lands in out_dir.
    out_path = os.environ.get(
        "IVCAP_RESULT_PATH", os.path.join(out_dir, "result.ivcap.json")
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": "dennisjooo/Birds-Classifier-EfficientNetB2",
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Stage 3 complete. Results → {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    in_dir = os.environ.get("IN_DIR", "/tmp/outputs")
    out_dir = os.environ.get("OUT_DIR", "/tmp/outputs")
    classify_stage(in_dir, out_dir)
