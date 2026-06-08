#!/usr/bin/env -S poetry run python3
"""
Prepare bird images as individual IVCAP artifacts grouped in a collection.

Downloads sample bird images from the HuggingFace 525-Bird-Species dataset
(the same dataset used to train dennisjooo/Birds-Classifier-EfficientNetB2),
creates a named IVCAP collection, and uploads each image as a standalone
artifact with ``urn:ivcap:policy:open`` access, adding every artifact to
the collection.

Dataset: yashikota/birds-525-species-image-classification
         (525 labelled bird species, features: image + label)

Outputs (written to ./data/):
  - images/image_XXXX.jpg         (one per sample image)
  - manifest.json                 (filenames, species labels, artifact URNs)

The model artifact is handled separately by prepare_model.py / make prepare-model.

This script is for LOCAL DEVELOPMENT ONLY and is not included in Docker.
"""

import os
import json
import sys
import random
import uuid
from pathlib import Path

# ── HuggingFace dataset that matches the EfficientNetB2 bird classifier ────────
BIRD_DATASET_ID = "yashikota/birds-525-species-image-classification"
BIRD_DATASET_SPLIT = "validation"  # "validation" split gives quick, varied samples

# ── IVCAP policy constant ──────────────────────────────────────────────────────
OPEN_POLICY = "urn:ivcap:policy:ivcap.open.artifact"

# Namespace UUID for generating deterministic collection URNs from names
# (RFC 4122 UUID namespace for URLs — reused here as a stable namespace)
_COLLECTION_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_collection_urn(collection_name: str) -> str:
    """Generate a deterministic ``urn:ivcap:collection:<uuid>`` from a name.

    Using UUID5 with a fixed namespace ensures the same name always maps to
    the same URN, making repeated runs idempotent — ``create_collection`` is
    a PUT that updates name/description without disturbing existing items.
    """
    uid = uuid.uuid5(_COLLECTION_NS, f"ivcap:collection:{collection_name}")
    return f"urn:ivcap:collection:{uid}"


def create_ivcap_collection(ivcap, collection_name: str):
    """Create (or idempotently update) an IVCAP collection with open policy.

    Returns the :class:`ivcap_client.Collection` instance.
    """
    collection_urn = make_collection_urn(collection_name)
    print(f"\n📂 Creating/updating collection '{collection_name}'")
    print(f"   URN: {collection_urn}")
    try:
        collection = ivcap.create_collection(
            urn=collection_urn,
            name=collection_name,
            description=(
                f"Bird test images sampled from '{BIRD_DATASET_ID}' "
                f"(split: {BIRD_DATASET_SPLIT})"
            ),
            policy=OPEN_POLICY,
        )
        print(f"  ✓ Collection ready: {collection.urn}")
        return collection
    except Exception as exc:
        print(f"  ✗ Failed to create collection: {exc}", file=sys.stderr)
        raise


def upload_image_artifact(
    ivcap,
    image_path: Path,
    species: str,
    collection,
) -> str:
    """Upload a single JPEG as an IVCAP artifact and ensure it is in the collection.

    Upload and collection membership are kept as two distinct steps:

    1. ``upload_artifact`` handles the file upload with SDK-level path-based
       dedup caching (same file → same artifact URN, no re-upload).
    2. ``collection.add_item`` explicitly ensures membership using the SDK's
       ``Collection`` API, which performs a server-side dedup check before
       creating the ``collection-item.1`` aspect.  This is idempotent and
       correct even on repeat runs where the upload is satisfied from cache.

    Returns the artifact URN string.
    """
    # Files are named <species_slug>_<dataset_idx>.jpg — a stable, content-unique
    # name derived from the dataset row index — so the SDK's path-based dedup
    # cache works correctly and force_upload is not required.
    artifact = ivcap.upload_artifact(
        name=image_path.name,
        file_path=str(image_path),
        content_type="image/jpeg",
        policy=OPEN_POLICY,
    )
    artifact_urn = str(artifact.id)

    # Ensure collection membership.  The SDK checks for an existing
    # collection-item.1 aspect before creating a new one, so this is safe
    # to call on every run without creating duplicates.  Pass the same open
    # policy so the membership aspect itself is also publicly readable.
    collection.add_item(artifact_urn, policy=OPEN_POLICY)

    return artifact_urn


def upload_images_to_collection(
    ivcap, image_dir: Path, downloaded: list[dict], collection
) -> list[dict]:
    """Upload all downloaded images to IVCAP and add them to the collection.

    Updates each record in *downloaded* in-place with ``artifact_urn``.
    Returns the augmented list.
    """
    print(f"\n☁️  Uploading {len(downloaded)} image(s) to IVCAP collection...")
    print(f"   Collection: {collection.urn}")

    for record in downloaded:
        image_path = image_dir / record["filename"]
        species = record.get("species", "unknown")
        size_kb = image_path.stat().st_size / 1024

        try:
            artifact_urn = upload_image_artifact(ivcap, image_path, species, collection)
            record["artifact_urn"] = artifact_urn
            print(
                f"  ✓ {record['filename']}  species={species}  "
                f"({size_kb:.1f} KB)  → {artifact_urn}"
            )
        except Exception as exc:
            print(
                f"  ✗ Failed to upload {record['filename']}: {exc}",
                file=sys.stderr,
            )
            raise

    return downloaded


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
    for idx in indices:
        sample = dataset[idx]

        # The image column is a PIL Image in most HF vision datasets
        img = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        if img is None:
            print(
                f"  WARNING: sample {idx} has no image field — skipping",
                file=sys.stderr,
            )
            continue

        # Capture the label/species name first so we can build the filename
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

        # Name the file using the species + dataset row index.
        # The dataset index is stable (same index → same image always), so the
        # SDK's path-based dedup cache works correctly and force_upload is not needed.
        species_slug = species.lower().replace(" ", "_").replace("/", "-")
        filename = f"{species_slug}_{idx:05d}.jpg"
        dest = image_dir / filename

        # Convert to RGB (some PNGs are RGBA) and save as JPEG
        img = img.convert("RGB")
        img.save(str(dest), format="JPEG", quality=90)

        size_kb = dest.stat().st_size / 1024
        print(f"  ✓ {filename}  ({size_kb:.1f} KB)")
        downloaded.append({"filename": filename, "label": label_id, "species": species})

    return downloaded


# ── Main ──────────────────────────────────────────────────────────────────────


def prepare_data(
    data_dir: str = "./data",
    count: int = 5,
    upload_artifact: bool = True,
    collection_name: str = "bird-test-1",
):
    """Prepare bird images and upload each as a standalone IVCAP artifact.

    Downloads sample images from the HuggingFace 525-Bird-Species dataset,
    creates a named IVCAP collection, and uploads every image as an
    individual artifact (``urn:ivcap:policy:open``) added to the collection.

    Args:
        data_dir:        Directory to store data (default: ./data)
        count:           Number of sample images to download (default: 5)
        upload_artifact: Whether to upload images as IVCAP artifacts (default: True)
        collection_name: Name of the IVCAP collection to create/update
                         (default: bird-test-1)
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"

    image_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialise IVCAP client eagerly so auth failures surface before any
    #    time-consuming image downloading starts ────────────────────────────────
    ivcap = None
    if upload_artifact:
        try:
            from ivcap_client import IVCAP
        except ImportError:
            print(
                "  ⚠ ivcap-client not installed. Skipping artifact upload.",
                file=sys.stderr,
            )
            print("    Install with: poetry install", file=sys.stderr)
            upload_artifact = False
        else:
            ivcap = IVCAP()  # raises immediately if IVCAP_URL / IVCAP_JWT are missing

    print(f"\n📁 Data directory: {data_path.resolve()}")
    print(f"   Dataset        : {BIRD_DATASET_ID}")
    print(f"   Collection     : {collection_name}")
    print(f"   └─ images/     → {image_dir.resolve()}\n")

    # ── Check for existing images ──────────────────────────────────────────────
    print("🐦  Bird Images (525-Species dataset):")

    existing_images = sorted(image_dir.glob("*.jpg"))
    existing_count = len(existing_images)

    # If count differs from requested, delete all and re-download
    if existing_count != count:
        if existing_count > 0:
            print(f"  Clearing {existing_count} existing images (requesting {count})")
            for img_file in existing_images:
                img_file.unlink()
            existing_count = 0

    downloaded = []
    newly_downloaded = 0

    if existing_count == count:
        # Re-use existing images; read species/artifact info from manifest if available
        manifest_path = data_path / "manifest.json"
        img_meta: dict[str, dict] = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                old_manifest = json.load(f)
            for entry in old_manifest.get("images", []):
                if isinstance(entry, dict):
                    img_meta[entry["filename"]] = entry

        for img_file in sorted(image_dir.glob("*.jpg")):
            size_kb = img_file.stat().st_size / 1024
            meta = img_meta.get(img_file.name, {})
            species = meta.get("species", "unknown")
            print(
                f"  ✓ {img_file.name} already present  species={species}  ({size_kb:.1f} KB)"
            )
            rec = {"filename": img_file.name, "species": species}
            if "artifact_urn" in meta:
                rec["artifact_urn"] = meta["artifact_urn"]
            downloaded.append(rec)
    else:
        records = download_bird_images(image_dir, count)
        downloaded = records
        newly_downloaded = len(records)

    if not downloaded:
        print("  ERROR: no images were processed", file=sys.stderr)
        sys.exit(1)

    # ── Upload images to IVCAP ────────────────────────────────────────────────
    if upload_artifact and ivcap is not None and len(downloaded) > 0:
        try:
            # Create (or idempotently update) the collection
            collection = create_ivcap_collection(ivcap, collection_name)

            # Upload each image as a standalone artifact
            downloaded = upload_images_to_collection(
                ivcap, image_dir, downloaded, collection
            )

            print(f"\n  ✓ All images uploaded to collection: {collection.urn}")

        except Exception as exc:
            print(
                "\n⚠ Warning: Artifact upload failed. Continuing without upload.",
                file=sys.stderr,
            )
            print(f"  Error: {exc}", file=sys.stderr)

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "dataset": BIRD_DATASET_ID,
        "collection_name": collection_name,
        "collection_urn": make_collection_urn(collection_name),
        "images": downloaded,
    }
    manifest_path = data_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Data preparation complete!")
    print(f"   {len(downloaded)} bird images ({newly_downloaded} downloaded)")
    print(
        f"   Collection : {collection_name}  ({make_collection_urn(collection_name)})"
    )
    print(f"   Manifest   : {manifest_path}")
    print(f"")
    print(f"   To prepare the model artifact, run: make prepare-model")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare bird images from HuggingFace and upload each as a "
            "standalone IVCAP artifact added to a named collection"
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "./data"),
        help="Directory to store data (default: ./data or DATA_DIR env var)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(os.environ.get("COUNT", "5")),
        help="Number of sample images to download (default: 5 or COUNT env var)",
    )
    parser.add_argument(
        "--collection-name",
        default=os.environ.get("COLLECTION_NAME", "bird-test-1"),
        help=(
            "Name of the IVCAP collection to create/update "
            "(default: bird-test-1 or COLLECTION_NAME env var)"
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading images as IVCAP artifacts",
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
        count=args.count,
        upload_artifact=not args.no_upload,
        collection_name=args.collection_name,
    )
