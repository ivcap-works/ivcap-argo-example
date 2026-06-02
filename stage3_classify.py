#!/usr/bin/env python3
"""
Stage 3 – Classify images with MobileNetV2-12 via ONNX Runtime.

Loads each preprocessed tensor, runs it through the model, applies softmax,
and writes the top-5 predictions per image to result.ivcap.json.

Model input  : name='data', shape=(1,3,224,224), dtype=float32  (NCHW)
Model output : name='output', shape=(1,1000),    dtype=float32  (raw logits)

Reference: https://huggingface.co/onnxmodelzoo/mobilenetv2-12

Inputs  (from $IN_DIR, default /tmp/outputs):
  - model/mobilenetv2-12.onnx
  - model/imagenet_classes.txt
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
import numpy as np
import onnxruntime as ort


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax normalization to logits."""
    e = np.exp(x - x.max())
    return e / e.sum()


def classify_stage(in_dir: str = "/tmp/outputs", out_dir: str = "/tmp/outputs") -> None:
    """Run inference on preprocessed tensors using MobileNetV2-12 model."""
    model_path = os.path.join(in_dir, "model", "mobilenetv2-12.onnx")
    labels_path = os.path.join(in_dir, "model", "imagenet_classes.txt")
    tensor_dir = os.path.join(in_dir, "tensors")
    top_k = 5

    # ── Load labels ───────────────────────────────────────────────────────────────
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(labels)} ImageNet labels", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}", flush=True)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(f"  input  : name='{input_meta.name}'  shape={input_meta.shape}", flush=True)
    print(f"  output : name='{output_meta.name}' shape={output_meta.shape}", flush=True)

    input_name = input_meta.name  # 'data'
    output_name = output_meta.name  # 'output'

    # ── Load manifest ─────────────────────────────────────────────────────────────
    with open(os.path.join(in_dir, "manifest.json")) as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    tensors = manifest.get("tensors", [])

    if len(images) != len(tensors):
        raise RuntimeError(
            f"manifest mismatch: {len(images)} images vs {len(tensors)} tensors"
        )

    # ── Inference ─────────────────────────────────────────────────────────────────
    results = []

    for img_file, tensor_file in zip(images, tensors):
        tensor_path = os.path.join(tensor_dir, tensor_file)
        tensor = np.load(tensor_path)  # (1, 3, 224, 224) float32

        print(f"Running inference on {img_file} …", flush=True)
        t0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logits = outputs[0][0]  # (1000,)
        probs = softmax(logits)  # (1000,)

        top_idx = np.argsort(probs)[::-1][:top_k]
        top5 = [
            {"rank": int(rank + 1), "label": labels[i], "score": float(probs[i])}
            for rank, i in enumerate(top_idx)
        ]

        print(
            f"  {elapsed_ms:.1f} ms  →  top-1: {top5[0]['label']} "
            f"({top5[0]['score']:.3f})",
            flush=True,
        )

        results.append(
            {
                "image": img_file,
                "inference_ms": round(elapsed_ms, 1),
                "top5": top5,
            }
        )

    # ── Write results ─────────────────────────────────────────────────────────────
    # $IVCAP_RESULT_PATH is set to /result.ivcap.json by the Argo workflow template
    # so the executor captures it as an output parameter for the IVCAP controller.
    # When running locally the env var is unset and the file lands in out_dir.
    out_path = os.environ.get(
        "IVCAP_RESULT_PATH", os.path.join(out_dir, "result.ivcap.json")
    )
    with open(out_path, "w") as f:
        json.dump({"model": "mobilenetv2-12", "results": results}, f, indent=2)
    print(f"\nStage 3 complete. Results → {out_path}", flush=True)


if __name__ == "__main__":
    in_dir = os.environ.get("IN_DIR", "/tmp/outputs")
    out_dir = os.environ.get("OUT_DIR", "/tmp/outputs")
    classify_stage(in_dir, out_dir)
