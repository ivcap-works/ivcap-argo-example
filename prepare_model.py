#!/usr/bin/env python3
"""
Prepare the Birds-Classifier-EfficientNetB2 model as an IVCAP artifact.

Downloads the model and image processor from Hugging Face Hub
(dennisjooo/Birds-Classifier-EfficientNetB2), saves them to a local directory,
packages everything into a zip archive, and uploads the archive as an IVCAP
artifact.

After a successful upload the zip file is renamed to include the artifact UUID
so that the Makefile can derive the URN automatically:

    data/efficientnet-birds-<UUID>.zip  →  urn:ivcap:artifact:<UUID>

Usage:
    python prepare_model.py [--data-dir ./data] [--no-upload]

This script is for LOCAL DEVELOPMENT / CI SETUP ONLY and is not included in
the Docker image — the model is loaded at runtime from the IVCAP artifact.
"""

import os
import re
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path

MODEL_ID = "dennisjooo/Birds-Classifier-EfficientNetB2"


# ── Download ──────────────────────────────────────────────────────────────────


def download_model(model_dir: Path) -> None:
    """Download EfficientNetB2 model + processor from Hugging Face and save locally."""
    try:
        from transformers import (
            EfficientNetImageProcessor,
            EfficientNetForImageClassification,
        )
    except ImportError:
        print("ERROR: transformers not installed.", file=sys.stderr)
        print(
            "  Install with: poetry add transformers torch safetensors", file=sys.stderr
        )
        sys.exit(1)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already downloaded
    if (model_dir / "config.json").exists():
        print(f"  ✓ Model already present in {model_dir} — skipping download")
        return

    print(f"  Downloading processor from {MODEL_ID}...", flush=True)
    processor = EfficientNetImageProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(str(model_dir))
    print(f"  ✓ Processor saved")

    print(f"  Downloading model weights from {MODEL_ID}...", flush=True)
    model = EfficientNetForImageClassification.from_pretrained(MODEL_ID)
    model.save_pretrained(str(model_dir))

    n_labels = len(model.config.id2label)
    print(f"  ✓ Model saved  ({n_labels} bird species labels)")


# ── Zip ───────────────────────────────────────────────────────────────────────


def zip_model_directory(model_dir: Path) -> Path:
    """Zip all files in model_dir, return path to zip."""
    zip_path = model_dir.parent / "efficientnet-birds.zip"

    print(f"\n📦 Creating zip archive of {model_dir} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(model_dir.glob("**/*")):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(model_dir))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"  ✓ efficientnet-birds.zip  ({size_mb:.1f} MB)")
    return zip_path


# ── Upload ────────────────────────────────────────────────────────────────────


def upload_to_ivcap(zip_path: Path) -> str:
    """Upload zip to IVCAP, return artifact UUID."""
    try:
        from ivcap_client import IVCAP
    except ImportError:
        print("ERROR: ivcap-client not installed.", file=sys.stderr)
        print("  Install with: poetry add ivcap-client", file=sys.stderr)
        raise

    print(f"\n☁️  Uploading {zip_path.name} to IVCAP …")
    ivcap = IVCAP()
    artifact = ivcap.upload_artifact(
        name="Birds-Classifier-EfficientNetB2 model", file_path=str(zip_path)
    )

    artifact_urn = str(artifact.id)
    match = re.search(r"([a-f0-9\-]+)$", artifact_urn)
    uuid = match.group(1) if match else artifact_urn

    print(f"  ✓ Uploaded  URN: {artifact_urn}")
    return uuid


# ── Main ──────────────────────────────────────────────────────────────────────


def prepare_model(data_dir: str = "./data", upload: bool = True) -> None:
    data_path = Path(data_dir)
    model_dir = data_path / "model"

    print(f"\n🤖  Bird Species Classifier — EfficientNetB2")
    print(f"   HuggingFace model : {MODEL_ID}")
    print(f"   Local model dir   : {model_dir.resolve()}\n")

    # 1. Download from HuggingFace
    download_model(model_dir)

    if not upload:
        print("\n⚠  --no-upload set: skipping IVCAP upload.")
        return

    # 2. Zip
    zip_path = zip_model_directory(model_dir)

    # 3. Upload to IVCAP
    try:
        uuid = upload_to_ivcap(zip_path)
    except Exception as exc:
        print(f"\n⚠  IVCAP upload failed: {exc}", file=sys.stderr)
        print(
            "   The zip is still available locally for manual upload.", file=sys.stderr
        )
        return

    # 4. Rename zip to embed UUID so the Makefile can auto-detect the URN
    named_zip = zip_path.parent / f"efficientnet-birds-{uuid}.zip"
    zip_path.rename(named_zip)
    print(f"  ✓ Renamed to {named_zip.name}")

    # 5. Write a small JSON sidecar so other scripts can read the URN without
    #    relying on filename glob parsing.
    sidecar = data_path / "model_artifact.json"
    with open(sidecar, "w") as f:
        json.dump(
            {"model_id": MODEL_ID, "artifact_urn": f"urn:ivcap:artifact:{uuid}"},
            f,
            indent=2,
        )
    print(f"  ✓ Artifact URN written to {sidecar}")

    print(f"\n✅  Model artifact ready!")
    print(f"   URN : urn:ivcap:artifact:{uuid}")
    print(f"   Use this URN as --model-artifact-urn when running the pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download EfficientNetB2 bird classifier and upload as IVCAP artifact"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "./data"),
        help="Directory to store model files (default: ./data)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Download model locally but skip IVCAP upload",
    )
    args = parser.parse_args()
    prepare_model(data_dir=args.data_dir, upload=not args.no_upload)
