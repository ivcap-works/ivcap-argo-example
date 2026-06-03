#!/usr/bin/env python3
"""
Stage 1 – Fetch bird images and EfficientNetB2 model from IVCAP artifacts.

Fetches data from IVCAP artifacts:
  - Images artifact (zip file containing .jpg bird images)
  - Model artifact  (zip file containing a saved HuggingFace EfficientNetB2 model:
                     config.json, model.safetensors / pytorch_model.bin,
                     preprocessor_config.json)

Artifact URNs are passed as command-line arguments:
  --images-artifact-urn: URN of the images artifact (e.g., urn:ivcap:artifact:xxx)
  --model-artifact-urn:  URN of the model artifact  (e.g., urn:ivcap:artifact:yyy)

The model artifact is created once with `make prepare-model` / `python prepare_model.py`
and then referenced by URN at pipeline runtime — the model is never baked into the
Docker image.

Outputs (written to ./outputs/):
  - model/config.json
  - model/preprocessor_config.json
  - model/model.safetensors  (or pytorch_model.bin)
  - images/<name>.jpg        (one per sample image)
  - manifest.json            (list of image filenames)
"""

import os
import json
import sys
import zipfile
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_artifact_and_extract(
    artifact_urn: str, extract_to: str, artifact_type: str = "data"
) -> None:
    """Fetch an IVCAP artifact and extract its contents.

    Args:
        artifact_urn: The artifact URN (e.g., urn:ivcap:artifact:xxx)
        extract_to: Directory to extract the artifact contents to
        artifact_type: Type of artifact for logging (e.g., "images", "model")

    Raises:
        ImportError: If ivcap-client is not installed
        FileNotFoundError: If artifact cannot be fetched
    """
    try:
        from ivcap_client import IVCAP
    except ImportError:
        logger.error("ivcap-client not installed")
        logger.error("  Install with: poetry add ivcap-client")
        sys.exit(1)

    logger.info(f"Fetching '{artifact_type}' artifact → {artifact_urn}")

    try:
        # Create IVCAP instance (reads IVCAP_URL and IVCAP_JWT from environment)
        ivcap = IVCAP()

        # Create temporary directory for the downloaded zip
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, f"{artifact_type}.zip")

            # Download the artifact
            logger.info(f"  Downloading '{artifact_type}' artifact...")
            artifact = ivcap.get_artifact(artifact_urn)
            temp_zip_path = artifact.as_local_file()

            # Check if the download was successful.
            if not os.path.exists(temp_zip_path):
                raise FileNotFoundError(f"Failed to download artifact {artifact_urn}")

            size_mb = os.path.getsize(temp_zip_path) / 1e6
            logger.info(f"  Downloaded: {size_mb:.1f} MB")

            # Extract the zip file
            logger.info(f"  Extracting '{artifact_type}' artifact...")
            os.makedirs(extract_to, exist_ok=True)

            with zipfile.ZipFile(temp_zip_path, "r") as zipf:
                zipf.extractall(extract_to)

            logger.info(f"  Extracted to {extract_to}")

    except Exception as exc:
        logger.error(f"Fetching '{artifact_type}' artifact failed: {exc}")
        sys.exit(1)


def fetch_stage(
    out_dir: str = "/tmp/outputs",
    images_artifact_urn: str = None,
    model_artifact_urn: str = None,
) -> None:
    """Fetch EfficientNetB2 model and bird images from IVCAP artifacts.

    Args:
        out_dir: Output directory for extracted files
        images_artifact_urn: URN of the bird images artifact (required)
        model_artifact_urn: URN of the EfficientNetB2 model artifact (required)

    Raises:
        SystemExit: If required artifact URNs are not provided or artifacts cannot be fetched
    """
    if not images_artifact_urn or not model_artifact_urn:
        logger.error("Both --images-artifact-urn and --model-artifact-urn are required")
        sys.exit(1)

    model_dir = os.path.join(out_dir, "model")
    image_dir = os.path.join(out_dir, "images")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # ── Fetch model artifact ──────────────────────────────────────────────────────
    logger.info("Stage 1: Fetching artifacts from IVCAP")
    fetch_artifact_and_extract(model_artifact_urn, model_dir, artifact_type="model")

    # ── Fetch images artifact ─────────────────────────────────────────────────────
    fetch_artifact_and_extract(images_artifact_urn, image_dir, artifact_type="images")

    # ── Create manifest ───────────────────────────────────────────────────────────
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    )

    if not image_files:
        logger.error("No .jpg images found in extracted images artifact")
        sys.exit(1)

    manifest = {"images": image_files}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Verify model files ────────────────────────────────────────────────────────
    # The model artifact must contain a HuggingFace-saved EfficientNetB2 directory
    # (produced by EfficientNetForImageClassification.save_pretrained + processor.save_pretrained).
    config_file = os.path.join(model_dir, "config.json")
    preprocessor_file = os.path.join(model_dir, "preprocessor_config.json")

    if not os.path.exists(config_file):
        logger.error(f"model config.json not found in {model_dir}")
        logger.error("  The model artifact should be created with: make prepare-model")
        sys.exit(1)

    if not os.path.exists(preprocessor_file):
        logger.error(f"preprocessor_config.json not found in {model_dir}")
        sys.exit(1)

    # Verify there is at least one set of model weights
    weight_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith(".safetensors") or f.endswith(".bin")
    ]
    if not weight_files:
        logger.error(f"No model weights (.safetensors or .bin) found in {model_dir}")
        sys.exit(1)

    # Read architecture from config so we can log it
    with open(config_file) as f:
        config = json.load(f)
    n_labels = len(config.get("id2label", {}))
    arch = config.get("model_type", "unknown")

    logger.info("Stage 1 complete.")
    logger.info(
        f"  Model: {arch}  ({n_labels} label classes)  weights={weight_files[0]}"
    )
    logger.info(f"  Images: {len(image_files)} bird images")
    logger.info(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage 1 - Fetch EfficientNetB2 model and bird images from IVCAP artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage1_fetch.py \\
    --images-artifact-urn urn:ivcap:artifact:178d14c4-e24a-4545-b9b0-60dc77593eaa \\
    --model-artifact-urn  urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90

Create the model artifact first (once):
  make prepare-model
        """,
    )

    parser.add_argument(
        "--images-artifact-urn",
        required=True,
        help="URN of the bird images artifact (e.g., urn:ivcap:artifact:xxx)",
    )
    parser.add_argument(
        "--model-artifact-urn",
        required=True,
        help="URN of the EfficientNetB2 model artifact (e.g., urn:ivcap:artifact:yyy)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: OUT_DIR env var or ./outputs)",
    )

    args = parser.parse_args()

    data_cache_dir = os.environ.get("DATA_CACHE_DIR")
    if data_cache_dir:
        out_dir = data_cache_dir
    elif args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.environ.get("OUT_DIR", "./outputs")

    fetch_stage(
        out_dir=out_dir,
        images_artifact_urn=args.images_artifact_urn,
        model_artifact_urn=args.model_artifact_urn,
    )
