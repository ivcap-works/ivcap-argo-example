#!/usr/bin/env python3
"""
Prepare local data for development and testing.

Downloads:
  - MobileNetV2-12 ONNX model from Hugging Face (onnxmodelzoo/mobilenetv2-12)
  - ImageNet class labels from pytorch/hub
  - Sample images from Imagenette dataset (PyTorch torchvision)

Outputs (written to ./data/):
  - model/mobilenetv2-12.onnx
  - model/imagenet_classes.txt
  - images/image_XXXX.jpg   (one per sample image)
  - manifest.json           (list of image filenames)
  - images.zip              (zipped images directory, uploaded to IVCAP as artifact)

This script is for LOCAL DEVELOPMENT ONLY and is not included in Docker.
"""

import os
import json
import urllib.request
import sys
import urllib.error
import time
import random
import shutil
from pathlib import Path
import zipfile
import re


def download_file(
    url: str, dest_path: str, description: str = "file", retries: int = 3
) -> bool:
    """Download a file with proper headers, retry logic, and error handling."""
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    for attempt in range(retries):
        try:
            print(f"  Downloading {description}...", end=" ", flush=True)
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(dest_path, "wb") as out_file:
                    out_file.write(response.read())
            size_kb = os.path.getsize(dest_path) / 1024
            print(f"✓ ({size_kb:.1f} KB)")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                # Rate limited, wait and retry
                wait_time = 5 * (attempt + 1)
                print(f"rate limited, waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"✗ {exc}", file=sys.stderr)
                return False
        except Exception as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return False
    return False


def zip_images_directory(image_dir: Path) -> str:
    """Zip the images directory and return the path to the zip file.

    Args:
        image_dir: Path to the images directory

    Returns:
        Path to the created zip file
    """
    zip_path = image_dir.parent / "images.zip"

    print(f"\n📦 Creating zip archive...")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for image_file in sorted(image_dir.glob("*.jpg")):
                zipf.write(image_file, arcname=image_file.name)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"  ✓ images.zip created ({size_mb:.1f} MB)")
        return str(zip_path)
    except Exception as exc:
        print(f"  ✗ Failed to create zip: {exc}", file=sys.stderr)
        raise


def zip_model_directory(model_dir: Path) -> str:
    """Zip the model directory and return the path to the zip file.

    Args:
        model_dir: Path to the model directory

    Returns:
        Path to the created zip file
    """
    zip_path = model_dir.parent / "model.zip"

    print(f"\n📦 Creating model zip archive...")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for model_file in sorted(model_dir.glob("*")):
                if model_file.is_file():
                    zipf.write(model_file, arcname=model_file.name)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"  ✓ model.zip created ({size_mb:.1f} MB)")
        return str(zip_path)
    except Exception as exc:
        print(f"  ✗ Failed to create model zip: {exc}", file=sys.stderr)
        raise


def upload_artifact_to_ivcap(zip_path: str, artifact_name: str = "images.zip") -> str:
    """Upload a zip file as an IVCAP artifact.

    Args:
        zip_path: Path to the zip file to upload
        artifact_name: Name of the artifact (default: "images.zip")

    Returns:
        The artifact UUID (without urn: prefix)

    Raises:
        ImportError: If ivcap-client is not installed
        Exception: If upload fails
    """
    try:
        from ivcap_client import IVCAP
    except ImportError:
        print(
            "  ⚠ ivcap-client not installed. Skipping artifact upload.", file=sys.stderr
        )
        print("    Install with: pip install ivcap-client", file=sys.stderr)
        raise

    print(f"\n☁️  Uploading artifact to IVCAP...")
    try:
        # Create IVCAP instance (reads IVCAP_URL and IVCAP_JWT from environment)
        ivcap = IVCAP()

        # Upload the zip file as an artifact
        artifact = ivcap.upload_artifact(name=artifact_name, file_path=zip_path)

        # Extract UUID from artifact URN (format: urn:ivcap:artifact:xxx)
        artifact_urn = artifact.id
        match = re.search(r"([a-f0-9\-]+)$", str(artifact_urn))
        artifact_uuid = match.group(1) if match else str(artifact_urn)

        print(f"  ✓ Artifact uploaded with URN: {artifact_urn}")
        return artifact_uuid

    except Exception as exc:
        print(f"  ✗ Failed to upload artifact: {exc}", file=sys.stderr)
        raise


def rename_zip_with_uuid(
    zip_path: str, artifact_uuid: str, prefix: str = "images"
) -> str:
    """Rename the zip file to include the artifact UUID.

    Args:
        zip_path: Current path to the zip file
        artifact_uuid: The artifact UUID
        prefix: Prefix for the renamed file (default: "images")

    Returns:
        The new path to the renamed zip file
    """
    zip_path_obj = Path(zip_path)
    new_zip_path = zip_path_obj.parent / f"{prefix}-{artifact_uuid}.zip"

    try:
        zip_path_obj.rename(new_zip_path)
        print(f"  ✓ Renamed to: {prefix}-{artifact_uuid}.zip")
        return str(new_zip_path)
    except Exception as exc:
        print(f"  ✗ Failed to rename zip file: {exc}", file=sys.stderr)
        raise


def prepare_data(
    data_dir: str = "./data", num_images: int = 5, upload_artifact: bool = True
):
    """Prepare data directories and download model + sample images.

    Args:
        data_dir: Directory to store data (default: ./data)
        num_images: Number of sample images to download (default: 5)
        upload_artifact: Whether to zip and upload images as IVCAP artifact (default: True)
    """
    data_path = Path(data_dir)
    model_dir = data_path / "model"
    image_dir = data_path / "images"

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Data directory: {data_path.resolve()}")
    print(f"   ├─ model/   → {model_dir.resolve()}")
    print(f"   └─ images/  → {image_dir.resolve()}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model_path = model_dir / "mobilenetv2-12.onnx"
    model_url = (
        "https://huggingface.co/onnxmodelzoo/mobilenetv2-12"
        "/resolve/main/mobilenetv2-12.onnx"
    )

    print("🤖 Model:")
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        print(f"  ✓ mobilenetv2-12.onnx already present ({size_mb:.1f} MB)")
    else:
        if download_file(model_url, str(model_path), "MobileNetV2-12 ONNX model"):
            size_mb = os.path.getsize(model_path) / 1e6
            print(f"    Model size: {size_mb:.1f} MB")
        else:
            print("  ERROR: Failed to download model", file=sys.stderr)
            sys.exit(1)

    # ── ImageNet labels ───────────────────────────────────────────────────────
    labels_path = model_dir / "imagenet_classes.txt"
    labels_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )

    print("\n📝 ImageNet Labels:")
    if labels_path.exists():
        n_labels = sum(1 for _ in open(labels_path))
        print(f"  ✓ imagenet_classes.txt already present ({n_labels} classes)")
    else:
        if download_file(labels_url, str(labels_path), "ImageNet class labels"):
            n_labels = sum(1 for _ in open(labels_path))
            print(f"    {n_labels} class labels loaded")
        else:
            print("  ERROR: Failed to download labels", file=sys.stderr)
            sys.exit(1)

    # ── Sample images ─────────────────────────────────────────────────────────
    # Sample images from Imagenette dataset (PyTorch torchvision)
    print("\n🖼️  Sample Images:")

    # Check for existing images
    existing_images = sorted(image_dir.glob("image_*.jpg"))
    existing_count = len(existing_images)

    # If count differs from requested, delete all and download new ones
    if existing_count != num_images:
        if existing_count > 0:
            print(
                f"  Clearing {existing_count} existing images (requesting {num_images})"
            )
            for img_file in existing_images:
                img_file.unlink()
                print(f"    Removed {img_file.name}")
            existing_count = 0

    # If we already have the right number, reuse them
    downloaded = []
    newly_downloaded = 0

    if existing_count == num_images:
        # Use existing images
        for img_file in existing_images:
            size_kb = img_file.stat().st_size / 1024
            print(f"  ✓ {img_file.name} already present ({size_kb:.1f} KB)")
            downloaded.append(img_file.name)
    else:
        # Download new images from Imagenette dataset
        try:
            from torchvision.datasets import Imagenette

            print(f"  Loading Imagenette dataset...")
            # Load validation dataset
            dataset = Imagenette(
                root="./imagenette_data", split="val", size="320px", download=True
            )

            # Get N random indices
            indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

            # Extract images
            for i, idx in enumerate(indices):
                filename = f"image_{i:04d}.jpg"
                dest = image_dir / filename

                image, label = dataset[idx]
                image.save(str(dest))

                size_kb = dest.stat().st_size / 1024
                print(f"  ✓ Extracted {filename} ({size_kb:.1f} KB)")
                downloaded.append(filename)
                newly_downloaded += 1

        except ImportError:
            print(
                "  ERROR: torchvision not installed. Install with: pip install torchvision",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as exc:
            print(
                f"  ERROR: Failed to download images from Imagenette: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not downloaded:
        print("  ERROR: no images were processed", file=sys.stderr)
        sys.exit(1)

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {"images": downloaded}
    manifest_path = data_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Upload images to IVCAP ────────────────────────────────────────────────
    if upload_artifact and len(downloaded) > 0:
        try:
            # Zip the images directory
            zip_path = zip_images_directory(image_dir)

            # Upload the zip file as an IVCAP artifact
            artifact_uuid = upload_artifact_to_ivcap(zip_path)

            # Rename the zip file with the artifact UUID
            renamed_zip_path = rename_zip_with_uuid(zip_path, artifact_uuid)

            print(f"\n✓ Images artifact uploaded and renamed: {renamed_zip_path}")

        except Exception as exc:
            print(
                f"\n⚠ Warning: Artifact upload failed. Continuing without upload.",
                file=sys.stderr,
            )
            print(f"  Error: {exc}", file=sys.stderr)

    # ── Upload model to IVCAP ──────────────────────────────────────────────────
    if upload_artifact:
        try:
            # Zip the model directory
            zip_path = zip_model_directory(model_dir)

            # Upload the zip file as an IVCAP artifact
            artifact_uuid = upload_artifact_to_ivcap(
                zip_path, artifact_name="model.zip"
            )

            # Rename the zip file with the artifact UUID
            renamed_zip_path = rename_zip_with_uuid(
                zip_path, artifact_uuid, prefix="model"
            )

            print(f"\n✓ Model artifact uploaded and renamed: {renamed_zip_path}")

        except Exception as exc:
            print(
                f"\n⚠ Warning: Model artifact upload failed. Continuing without upload.",
                file=sys.stderr,
            )
            print(f"  Error: {exc}", file=sys.stderr)

    print(f"\n✅ Data preparation complete!")
    print(f"   {len(downloaded)} images ({newly_downloaded} downloaded)")
    print(f"   Manifest: {manifest_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare local data for development and testing"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "./data"),
        help="Directory to store data (default: ./data or DATA_DIR env var)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=int(os.environ.get("NUM_IMAGES", "5")),
        help="Number of sample images to download (default: 5 or NUM_IMAGES env var)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading images as IVCAP artifact",
    )

    args = parser.parse_args()
    prepare_data(
        data_dir=args.data_dir,
        num_images=args.num_images,
        upload_artifact=not args.no_upload,
    )
