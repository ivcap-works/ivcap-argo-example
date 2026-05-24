#!/usr/bin/env python3
"""
Stage 1 – Fetch model and sample images from IVCAP artifacts.

Fetches data from IVCAP artifacts:
  - Images artifact (zip file containing .jpg images)
  - Model artifact (zip file containing .onnx model and imagenet_classes.txt)

Artifact URNs are passed as command-line arguments:
  --images-artifact-urn: URN of the images artifact (e.g., urn:ivcap:artifact:xxx)
  --model-artifact-urn:  URN of the model artifact (e.g., urn:ivcap:artifact:yyy)

Outputs (written to ./outputs/):
  - model/mobilenetv2-12.onnx
  - model/imagenet_classes.txt
  - images/<name>.jpg   (one per sample image)
  - manifest.json       (list of image filenames)
"""

import os
import json
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path


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
        print("ERROR: ivcap-client not installed", file=sys.stderr)
        print("  Install with: poetry add ivcap-client", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching '{artifact_type}' artifact → {artifact_urn}", flush=True)

    try:
        # Create IVCAP instance (reads IVCAP_URL and IVCAP_JWT from environment)
        ivcap = IVCAP()

        # Create temporary directory for the downloaded zip
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, f"{artifact_type}.zip")

            # Download the artifact
            print(f"  Downloading '{artifact_type}' artifact...", flush=True)
            artifact = ivcap.get_artifact(artifact_urn)
            temp_zip_path = artifact.as_local_file()

            # Check if the download was successful.
            # The artifact download should have reported that already, but we check again to be sure.
            if not os.path.exists(temp_zip_path):
                raise FileNotFoundError(f"Failed to download artifact {artifact_urn}")

            size_mb = os.path.getsize(temp_zip_path) / 1e6
            print(f"  Downloaded: {size_mb:.1f} MB", flush=True)

            # Extract the zip file
            print(f"  Extracting '{artifact_type}' artifact...", flush=True)
            os.makedirs(extract_to, exist_ok=True)

            with zipfile.ZipFile(temp_zip_path, "r") as zipf:
                zipf.extractall(extract_to)

            print(f"  Extracted to {extract_to}", flush=True)

    except Exception as exc:
        print(f"  ERROR fetching '{artifact_type}' artifact: {exc}", file=sys.stderr)
        sys.exit(1)


def fetch_stage(
    out_dir: str = "/tmp/outputs",
    images_artifact_urn: str = None,
    model_artifact_urn: str = None,
) -> None:
    """Fetch model and sample images from IVCAP artifacts.

    Args:
        out_dir: Output directory for extracted files
        images_artifact_urn: URN of the images artifact (required)
        model_artifact_urn: URN of the model artifact (required)

    Raises:
        ValueError: If required artifact URNs are not provided
    """
    if not images_artifact_urn or not model_artifact_urn:
        print(
            "ERROR: Both --images-artifact-urn and --model-artifact-urn are required",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = os.path.join(out_dir, "model")
    image_dir = os.path.join(out_dir, "images")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # ── Fetch model artifact ──────────────────────────────────────────────────────
    print("Stage 1: Fetching artifacts from IVCAP", flush=True)
    fetch_artifact_and_extract(model_artifact_urn, model_dir, artifact_type="model")

    # ── Fetch images artifact ─────────────────────────────────────────────────────
    fetch_artifact_and_extract(images_artifact_urn, image_dir, artifact_type="images")

    # ── Create manifest ───────────────────────────────────────────────────────────
    # List all jpg files in the images directory
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    )

    if not image_files:
        print("ERROR: no images found in extracted artifact", file=sys.stderr)
        sys.exit(1)

    manifest = {"images": image_files}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Verify model files ────────────────────────────────────────────────────────
    model_file = os.path.join(model_dir, "mobilenetv2-12.onnx")
    labels_file = os.path.join(model_dir, "imagenet_classes.txt")

    if not os.path.exists(model_file):
        print(f"ERROR: model file not found: {model_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(labels_file):
        print(f"ERROR: labels file not found: {labels_file}", file=sys.stderr)
        sys.exit(1)

    with open(labels_file) as f:
        n_labels = sum(1 for _ in f)

    print("Stage 1 complete.")
    print(f"  Model: {model_file} ({os.path.getsize(model_file) / 1e6:.1f} MB)")
    print(f"  Labels: {n_labels} classes")
    print(f"  Images: {len(image_files)} images")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1 - Fetch model and images from IVCAP artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage1_fetch.py \\
    --images-artifact-urn urn:ivcap:artifact:178d14c4-e24a-4545-b9b0-60dc77593eaa \\
    --model-artifact-urn urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90
        """,
    )

    parser.add_argument(
        "--images-artifact-urn",
        required=True,
        help="URN of the images artifact (e.g., urn:ivcap:artifact:xxx)",
    )
    parser.add_argument(
        "--model-artifact-urn",
        required=True,
        help="URN of the model artifact (e.g., urn:ivcap:artifact:yyy)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: OUT_DIR env var or ./outputs)",
    )

    args = parser.parse_args()

    # Support DATA_CACHE_DIR for caching external data
    # If DATA_CACHE_DIR is set, use that; otherwise use OUT_DIR or --out-dir
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
