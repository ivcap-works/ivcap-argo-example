#!/usr/bin/env python3
"""
Stage 2 – Preprocess images.

Reads images from Stage 1 output and converts each one to a normalised
float32 NCHW tensor ready for MobileNetV2-12 inference.

Preprocessing follows the official ONNX Model Zoo spec for MobileNetV2:
  • Resize shortest side to 256 px (preserving aspect ratio)
  • Centre-crop to 224 × 224
  • Scale pixel values from [0, 255] to [0.0, 1.0]
  • Normalise per channel:
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]
  • Reorder to NCHW: shape (1, 3, 224, 224), dtype float32

Reference: https://huggingface.co/onnxmodelzoo/mobilenetv2-12

Inputs  (from $IN_DIR, default /tmp/outputs):
  - images/<name>.jpg
  - manifest.json

Outputs (written to $OUT_DIR, default /tmp/outputs):
  - tensors/<name>.npy   (one numpy array per image, shape 1×3×224×224)
  - manifest.json        (updated in-place with 'tensors' key)
"""

import os
import json
import numpy as np
from PIL import Image

# ImageNet normalisation constants (official, shared by all torchvision models)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RESIZE_TO = 256  # resize shortest side to this
CROP_SIZE = 224  # then centre-crop to this


def preprocess(image_path: str) -> np.ndarray:
    """Return a float32 NCHW tensor of shape (1, 3, 224, 224)."""
    img = Image.open(image_path).convert("RGB")

    # Resize: shortest side → RESIZE_TO, preserve aspect ratio
    w, h = img.size
    if h < w:
        new_h, new_w = RESIZE_TO, int(w * RESIZE_TO / h)
    else:
        new_h, new_w = int(h * RESIZE_TO / w), RESIZE_TO
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Centre crop
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # HWC uint8 → float32 [0,1]
    arr = np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)

    # Normalise per channel
    arr = (arr - MEAN) / STD  # (224, 224, 3)

    # HWC → CHW → NCHW
    arr = arr.transpose(2, 0, 1)  # (3, 224, 224)
    arr = np.expand_dims(arr, axis=0)  # (1, 3, 224, 224)

    return arr


def preprocess_stage(
    in_dir: str = "/tmp/outputs", out_dir: str = "/tmp/outputs"
) -> None:
    """Preprocess images into tensors."""
    image_dir = os.path.join(in_dir, "images")
    tensor_dir = os.path.join(out_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    # ── Load manifest ─────────────────────────────────────────────────────────────
    manifest_path = os.path.join(in_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    if not images:
        raise RuntimeError("manifest.json contains no images")

    # ── Process each image ────────────────────────────────────────────────────────
    tensor_files = []
    for filename in images:
        src = os.path.join(image_dir, filename)
        stem = os.path.splitext(filename)[0]
        dst = os.path.join(tensor_dir, f"{stem}.npy")

        print(f"Preprocessing {filename} …", flush=True)
        tensor = preprocess(src)
        np.save(dst, tensor)
        tensor_files.append(f"{stem}.npy")
        print(
            f"  saved {dst}  shape={tensor.shape}  dtype={tensor.dtype}",
            flush=True,
        )

    # ── Update manifest ───────────────────────────────────────────────────────────
    manifest["tensors"] = tensor_files
    out_manifest = os.path.join(out_dir, "manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nStage 2 complete. {len(tensor_files)} tensors → {tensor_dir}")


if __name__ == "__main__":
    in_dir = os.environ.get("IN_DIR", "/tmp/outputs")
    out_dir = os.environ.get("OUT_DIR", "/tmp/outputs")
    preprocess_stage(in_dir, out_dir)
