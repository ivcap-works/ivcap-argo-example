#!/usr/bin/env python3
"""
Prepare bird images as an IVCAP artifact for development and testing.

Downloads sample bird images from the HuggingFace 525-Bird-Species dataset
(the same dataset used to train dennisjooo/Birds-Classifier-EfficientNetB2)
and uploads them as an IVCAP artifact so the pipeline can reference them by URN.

Dataset: yashikota/birds-525-species-image-classification
         (525 labelled bird species, features: image + label)

Outputs (written to ./data/):
  - images/image_XXXX.jpg         (one per sample image)
  - manifest.json                 (list of image filenames + species labels)
  - images-<UUID>.zip             (zipped images uploaded to IVCAP as artifact)

The model artifact is handled separately by prepare_model.py / make prepare-model.

This script is for LOCAL DEVELOPMENT ONLY and is not included in Docker.
"""

import os
import json
import sys
import random
import zipfile
import re
from pathlib import Path

# ── HuggingFace dataset that matches the EfficientNetB2 bird classifier ────────
# This is the 525-species bird dataset used to train the model.
# Change this constant if a different bird dataset is preferred.
BIRD_DATASET_ID = "yashikota/birds-525-species-image-classification"
BIRD_DATASET_SPLIT = "validation"  # "validation" split gives quick, varied samples


# ── Helpers ───────────────────────────────────────────────────────────────────


def zip_images_directory(image_dir: Path) -> Path:
    """Zip the images directory and return the path to the zip file."""
    zip_path = image_dir.parent / "images.zip"

    print(f"\n📦 Creating zip archive...")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for image_file in sorted(image_dir.glob("*.jpg")):
                zipf.write(image_file, arcname=image_file.name)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"  ✓ images.zip created ({size_mb:.1f} MB)")
        return zip_path
    except Exception as exc:
        print(f"  ✗ Failed to create zip: {exc}", file=sys.stderr)
        raise


def upload_artifact_to_ivcap(zip_path: Path, artifact_name: str = "images.zip") -> str:
    """Upload a zip file as an IVCAP artifact, return the artifact UUID."""
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
        ivcap = IVCAP()
        artifact = ivcap.upload_artifact(name=artifact_name, file_path=str(zip_path))

        artifact_urn = str(artifact.id)
        match = re.search(r"([a-f0-9\-]+)$", artifact_urn)
        artifact_uuid = match.group(1) if match else artifact_urn

        print(f"  ✓ Artifact uploaded with URN: {artifact_urn}")
        return artifact_uuid

    except Exception as exc:
        print(f"  ✗ Failed to upload artifact: {exc}", file=sys.stderr)
        raise


def rename_zip_with_uuid(
    zip_path: Path, artifact_uuid: str, prefix: str = "images"
) -> Path:
    """Rename the zip file to include the artifact UUID."""
    new_zip_path = zip_path.parent / f"{prefix}-{artifact_uuid}.zip"
    try:
        zip_path.rename(new_zip_path)
        print(f"  ✓ Renamed to: {new_zip_path.name}")
        return new_zip_path
    except Exception as exc:
        print(f"  ✗ Failed to rename zip file: {exc}", file=sys.stderr)
        raise


def download_bird_images(image_dir: Path, num_images: int) -> list[dict]:
    """Download bird images from HuggingFace 525-Bird-Species dataset.

    Returns a list of dicts with keys: filename, label, species.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "  ERROR: 'datasets' not installed. Install with: poetry install",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"  Loading {BIRD_DATASET_ID} (split='{BIRD_DATASET_SPLIT}') from HuggingFace..."
    )
    try:
        dataset = load_dataset(BIRD_DATASET_ID, split=BIRD_DATASET_SPLIT)
    except Exception as exc:
        print(
            f"  ERROR: Failed to load dataset '{BIRD_DATASET_ID}': {exc}",
            file=sys.stderr,
        )
        print(
            "  Tip: Check the dataset name at https://huggingface.co/datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(dataset)
    print(f"  Dataset loaded: {total} images across bird species")

    indices = random.sample(range(total), min(num_images, total))

    downloaded = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        filename = f"image_{i:04d}.jpg"
        dest = image_dir / filename

        # The image column is a PIL Image in most HF vision datasets
        img = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        if img is None:
            print(
                f"  WARNING: sample {idx} has no image field — skipping",
                file=sys.stderr,
            )
            continue

        # Convert to RGB (some PNGs are RGBA) and save as JPEG
        img = img.convert("RGB")
        img.save(str(dest), format="JPEG", quality=90)

        # Capture the label/species name
        label_id = sample.get("label", sample.get("labels", None))
        if label_id is not None and hasattr(
            dataset.features.get("label", dataset.features.get("labels")), "int2str"
        ):
            label_feature = dataset.features.get("label") or dataset.features.get(
                "labels"
            )
            species = label_feature.int2str(label_id)
        else:
            species = str(label_id) if label_id is not None else "unknown"

        size_kb = dest.stat().st_size / 1024
        print(f"  ✓ {filename}  species={species}  ({size_kb:.1f} KB)")
        downloaded.append({"filename": filename, "label": label_id, "species": species})

    return downloaded


# ── Main ──────────────────────────────────────────────────────────────────────


def prepare_data(
    data_dir: str = "./data", num_images: int = 5, upload_artifact: bool = True
):
    """Prepare bird image directories and upload them as an IVCAP artifact.

    Downloads sample images from the HuggingFace 525-Bird-Species dataset
    (matching the training data of dennisjooo/Birds-Classifier-EfficientNetB2).

    Args:
        data_dir: Directory to store data (default: ./data)
        num_images: Number of sample images to download (default: 5)
        upload_artifact: Whether to zip and upload images as IVCAP artifact (default: True)
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"

    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Data directory: {data_path.resolve()}")
    print(f"   Dataset        : {BIRD_DATASET_ID}")
    print(f"   └─ images/     → {image_dir.resolve()}\n")

    # ── Check for existing images ──────────────────────────────────────────────
    print("🐦  Bird Images (525-Species dataset):")

    existing_images = sorted(image_dir.glob("image_*.jpg"))
    existing_count = len(existing_images)

    # If count differs from requested, delete all and re-download
    if existing_count != num_images:
        if existing_count > 0:
            print(
                f"  Clearing {existing_count} existing images (requesting {num_images})"
            )
            for img_file in existing_images:
                img_file.unlink()
            existing_count = 0

    downloaded = []
    newly_downloaded = 0

    if existing_count == num_images:
        # Re-use existing images; try to read species from manifest if available
        manifest_path = data_path / "manifest.json"
        species_map = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                old_manifest = json.load(f)
            for entry in old_manifest.get("images", []):
                if isinstance(entry, dict):
                    species_map[entry["filename"]] = entry.get("species", "unknown")

        for img_file in sorted(image_dir.glob("image_*.jpg")):
            size_kb = img_file.stat().st_size / 1024
            species = species_map.get(img_file.name, "unknown")
            print(
                f"  ✓ {img_file.name} already present  species={species}  ({size_kb:.1f} KB)"
            )
            downloaded.append({"filename": img_file.name, "species": species})
    else:
        records = download_bird_images(image_dir, num_images)
        downloaded = records
        newly_downloaded = len(records)

    if not downloaded:
        print("  ERROR: no images were processed", file=sys.stderr)
        sys.exit(1)

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "dataset": BIRD_DATASET_ID,
        "images": downloaded,
    }
    manifest_path = data_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Upload images to IVCAP ────────────────────────────────────────────────
    if upload_artifact and len(downloaded) > 0:
        try:
            zip_path = zip_images_directory(image_dir)
            artifact_uuid = upload_artifact_to_ivcap(zip_path)
            rename_zip_with_uuid(zip_path, artifact_uuid)
        except Exception as exc:
            print(
                f"\n⚠ Warning: Artifact upload failed. Continuing without upload.",
                file=sys.stderr,
            )
            print(f"  Error: {exc}", file=sys.stderr)

    print(f"\n✅ Data preparation complete!")
    print(f"   {len(downloaded)} bird images ({newly_downloaded} downloaded)")
    print(f"   Manifest: {manifest_path}")
    print(f"")
    print(f"   To prepare the model artifact, run: make prepare-model")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare bird images from HuggingFace and upload as IVCAP artifact"
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
    parser.add_argument(
        "--dataset",
        default=BIRD_DATASET_ID,
        help=f"HuggingFace dataset ID to use (default: {BIRD_DATASET_ID})",
    )

    args = parser.parse_args()

    # Allow overriding the dataset constant at runtime
    if args.dataset != BIRD_DATASET_ID:
        BIRD_DATASET_ID = args.dataset

    prepare_data(
        data_dir=args.data_dir,
        num_images=args.num_images,
        upload_artifact=not args.no_upload,
    )
