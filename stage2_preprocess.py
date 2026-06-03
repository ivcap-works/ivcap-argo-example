#!/usr/bin/env python3
"""
Stage 2 – Preprocess bird images with EfficientNetImageProcessor.

Reads images from Stage 1 output and processes each one using the
EfficientNetImageProcessor loaded from the model artifact directory.
The processor handles all resizing and normalisation internally
(EfficientNet-B2 expects 260×260 input, managed automatically).

Preprocessed tensors are saved as float32 NumPy arrays in NCHW format
(shape 1×3×260×260) so they can be loaded directly by Stage 3.

Inputs  (from $IN_DIR, default /tmp/outputs):
  - model/preprocessor_config.json  (loaded by EfficientNetImageProcessor)
  - images/<name>.jpg
  - manifest.json

Outputs (written to $OUT_DIR, default /tmp/outputs):
  - tensors/<name>.npy   (one numpy array per image, shape 1×3×260×260, float32)
  - manifest.json        (updated in-place with 'tensors' key)
"""

import os
import json
import sys
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_stage(
    in_dir: str = "/tmp/outputs", out_dir: str = "/tmp/outputs"
) -> None:
    """Preprocess bird images into tensors using EfficientNetImageProcessor."""
    model_dir = os.path.join(in_dir, "model")
    image_dir = os.path.join(in_dir, "images")
    tensor_dir = os.path.join(out_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    # ── Load processor from artifact directory ────────────────────────────────────
    logger.info(f"Loading EfficientNetImageProcessor from {model_dir}")
    try:
        from transformers import EfficientNetImageProcessor
    except ImportError:
        logger.error("transformers not installed")
        sys.exit(1)

    processor = EfficientNetImageProcessor.from_pretrained(model_dir)
    logger.info(
        f"  Processor loaded  (size={processor.size}, "
        f"rescale={processor.do_rescale}, normalize={processor.do_normalize})"
    )

    # ── Load manifest ─────────────────────────────────────────────────────────────
    manifest_path = os.path.join(in_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    if not images:
        raise RuntimeError("manifest.json contains no images")

    def _fname(entry: "str | dict") -> str:
        """Return the filename string from a manifest entry (str or dict)."""
        return entry["filename"] if isinstance(entry, dict) else entry

    # ── Process each image ────────────────────────────────────────────────────────
    tensor_files = []
    for entry in images:
        filename = _fname(entry)
        src = os.path.join(image_dir, filename)
        stem = os.path.splitext(filename)[0]
        dst = os.path.join(tensor_dir, f"{stem}.npy")

        logger.info(f"Preprocessing {filename}")
        img = Image.open(src).convert("RGB")

        # EfficientNetImageProcessor returns {'pixel_values': torch.Tensor (1,3,H,W)}
        inputs = processor(img, return_tensors="pt")
        tensor = inputs["pixel_values"].numpy()  # (1, 3, 260, 260) float32

        np.save(dst, tensor)
        tensor_files.append(f"{stem}.npy")
        logger.info(f"  Saved {dst}  shape={tensor.shape}  dtype={tensor.dtype}")

    # ── Update manifest ───────────────────────────────────────────────────────────
    manifest["tensors"] = tensor_files
    out_manifest = os.path.join(out_dir, "manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Stage 2 complete. {len(tensor_files)} tensors → {tensor_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    in_dir = os.environ.get("IN_DIR", "/tmp/outputs")
    out_dir = os.environ.get("OUT_DIR", "/tmp/outputs")
    preprocess_stage(in_dir, out_dir)
