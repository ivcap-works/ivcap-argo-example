#!/usr/bin/env python3
"""
Stage 1 – Fetch bird images from an IVCAP collection and EfficientNetB2 model artifact.

Fetches data from IVCAP:
  - Images collection  (individual .jpg artifacts grouped in an IVCAP collection,
                        created by prepare_data.py / ``make prepare-data``)
  - Model artifact     (zip file containing a saved HuggingFace EfficientNetB2 model:
                        config.json, model.safetensors / pytorch_model.bin,
                        preprocessor_config.json)

Arguments:
  --collection-urn:    URN of the IVCAP collection that contains the bird image artifacts
                       (e.g., urn:ivcap:collection:xxx)
  --model-artifact-urn: URN of the model artifact (e.g., urn:ivcap:artifact:yyy)
  --limit:             Optional maximum number of images to fetch from the collection
                       (0 or omitted = fetch all)

The model artifact is created once with `make prepare-model` / `python prepare_model.py`
and then referenced by URN at pipeline runtime — the model is never baked into the
Docker image.

Outputs (written to ./outputs/):
  - model/config.json
  - model/preprocessor_config.json
  - model/model.safetensors  (or pytorch_model.bin)
  - images/<name>.jpg        (one per image fetched from the collection)
  - manifest.json            (list of image filenames)
"""

import os
import json
import sys
import shutil
import zipfile
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_model_artifact(
    artifact_urn: str,
    extract_to: str,
) -> None:
    """Fetch the model IVCAP artifact (zip) and extract its contents.

    Args:
        artifact_urn: The artifact URN (e.g., urn:ivcap:artifact:xxx)
        extract_to:   Directory to extract the artifact contents to

    Raises:
        SystemExit: If the artifact cannot be fetched or extracted
    """
    try:
        from ivcap_client import IVCAP
    except ImportError:
        logger.error("ivcap-client not installed")
        logger.error("  Install with: poetry add ivcap-client")
        sys.exit(1)

    logger.info(f"Fetching 'model' artifact → {artifact_urn}")

    try:
        ivcap = IVCAP()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, "model.zip")

            logger.info("  Downloading 'model' artifact...")
            artifact = ivcap.get_artifact(artifact_urn)
            temp_zip_path = artifact.as_local_file()

            if not os.path.exists(temp_zip_path):
                raise FileNotFoundError(f"Failed to download artifact {artifact_urn}")

            size_mb = os.path.getsize(temp_zip_path) / 1e6
            logger.info(f"  Downloaded: {size_mb:.1f} MB")

            logger.info("  Extracting 'model' artifact...")
            os.makedirs(extract_to, exist_ok=True)

            with zipfile.ZipFile(temp_zip_path, "r") as zipf:
                zipf.extractall(extract_to)

            logger.info(f"  Extracted to {extract_to}")

    except Exception as exc:
        logger.error(f"Fetching 'model' artifact failed: {exc}")
        sys.exit(1)


def fetch_images_from_collection(
    collection_urn: str,
    image_dir: str,
    limit: int | None = None,
) -> list[str]:
    """Fetch individual image artifacts from an IVCAP collection.

    Each item in the collection is expected to be a standalone JPEG artifact
    (as produced by ``prepare_data.py`` / ``make prepare-data``).  The function
    iterates over the collection, downloads every artifact via the ivcap-client,
    and writes it into *image_dir*.

    Args:
        collection_urn: URN of the IVCAP collection (e.g., urn:ivcap:collection:xxx)
        image_dir:      Directory to save the downloaded images
        limit:          Maximum number of images to fetch; ``None`` or ``0`` means all

    Returns:
        Sorted list of image filenames (relative to *image_dir*).

    Raises:
        SystemExit: On any unrecoverable error (missing client, bad URN, …)
    """
    try:
        from ivcap_client import IVCAP
    except ImportError:
        logger.error(
            "ivcap-client not installed — install with: poetry add ivcap-client"
        )
        sys.exit(1)

    effective_limit = limit if (limit and limit > 0) else None

    ivcap = IVCAP()

    logger.info(f"Fetching images from IVCAP collection: {collection_urn}")
    if effective_limit:
        logger.info(f"  Limit: {effective_limit} image(s)")

    try:
        collection = ivcap.get_collection(collection_urn)
        coll_name = getattr(collection, "name", collection_urn)
        logger.info(f"  Collection name: {coll_name}")
    except Exception as exc:
        logger.error(f"Failed to get collection '{collection_urn}': {exc}")
        sys.exit(1)

    os.makedirs(image_dir, exist_ok=True)

    image_files: list[str] = []
    count = 0

    try:
        # Pass effective_limit to collection.items() so the API respects it.
        # Default is 10 which would silently cap large collections.
        # CollectionItem.item (= .urn) is the artifact URN;
        # CollectionItem.id is the aspect-record URN (urn:ivcap:aspect:…).
        for item in collection.items(limit=effective_limit):
            # item.urn is an alias for item.item — the member artifact URN
            artifact_urn = item.urn

            logger.info(f"  [{count + 1}] Fetching artifact: {artifact_urn}")

            try:
                artifact = ivcap.get_artifact(artifact_urn)
                local_path = artifact.as_local_file()

                # Determine a sensible destination filename.
                # Prefer the artifact's own name (set at upload time by prepare_data.py),
                # then fall back to the basename of the cached local path, then a
                # zero-padded index.
                artifact_name: str = getattr(artifact, "name", None) or ""
                src = Path(local_path)
                img_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

                if Path(artifact_name).suffix.lower() in img_exts:
                    filename = artifact_name
                elif src.suffix.lower() in img_exts:
                    filename = src.name
                else:
                    filename = f"image_{count:05d}.jpg"

                dest = Path(image_dir) / filename
                shutil.copy2(local_path, str(dest))

                size_kb = dest.stat().st_size / 1024
                logger.info(f"       → {filename}  ({size_kb:.1f} KB)")
                image_files.append(filename)
                count += 1

            except Exception as exc:
                logger.error(f"  Failed to fetch/save artifact {artifact_urn}: {exc}")
                raise

    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"Error while iterating collection items: {exc}")
        sys.exit(1)

    if not image_files:
        logger.error(
            f"No images were fetched from collection '{collection_urn}'. "
            "Check that the collection contains image artifacts."
        )
        sys.exit(1)

    logger.info(f"  Fetched {count} image(s) from collection.")
    return sorted(image_files)


def fetch_stage(
    out_dir: str = "/tmp/outputs",
    collection_urn: str = None,
    model_artifact_urn: str = None,
    limit: int | None = None,
) -> None:
    """Fetch EfficientNetB2 model artifact and bird images from an IVCAP collection.

    Args:
        out_dir:            Output directory for extracted/downloaded files
        collection_urn:     URN of the IVCAP collection containing bird image artifacts
        model_artifact_urn: URN of the EfficientNetB2 model artifact
        limit:              Maximum number of images to fetch (None / 0 = all)

    Raises:
        SystemExit: If required URNs are not provided or resources cannot be fetched
    """
    if not collection_urn or not model_artifact_urn:
        logger.error("Both --collection-urn and --model-artifact-urn are required")
        sys.exit(1)

    model_dir = os.path.join(out_dir, "model")
    image_dir = os.path.join(out_dir, "images")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    logger.info("Stage 1: Fetching model artifact and collection images from IVCAP")

    # ── Fetch model artifact (zip) ────────────────────────────────────────────────
    fetch_model_artifact(model_artifact_urn, model_dir)

    # ── Fetch images from IVCAP collection ───────────────────────────────────────
    image_files = fetch_images_from_collection(collection_urn, image_dir, limit=limit)

    # ── Create manifest ───────────────────────────────────────────────────────────
    manifest = {"images": image_files}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Verify model files ────────────────────────────────────────────────────────
    config_file = os.path.join(model_dir, "config.json")
    preprocessor_file = os.path.join(model_dir, "preprocessor_config.json")

    if not os.path.exists(config_file):
        logger.error(f"model config.json not found in {model_dir}")
        logger.error("  The model artifact should be created with: make prepare-model")
        sys.exit(1)

    if not os.path.exists(preprocessor_file):
        logger.error(f"preprocessor_config.json not found in {model_dir}")
        sys.exit(1)

    weight_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith(".safetensors") or f.endswith(".bin")
    ]
    if not weight_files:
        logger.error(f"No model weights (.safetensors or .bin) found in {model_dir}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)
    n_labels = len(config.get("id2label", {}))
    arch = config.get("model_type", "unknown")

    logger.info("Stage 1 complete.")
    logger.info(
        f"  Model     : {arch}  ({n_labels} label classes)  weights={weight_files[0]}"
    )
    logger.info(f"  Images    : {len(image_files)} bird image(s) from collection")
    logger.info(f"  Manifest  : {manifest_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage 1 - Fetch EfficientNetB2 model and bird images from IVCAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage1_fetch.py \\
    --collection-urn     urn:ivcap:collection:5f3a9c12-1b2e-4d8a-9f7e-3c0b1d2e5f6a \\
    --model-artifact-urn urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90

  # Limit to the first 10 images:
  python stage1_fetch.py \\
    --collection-urn     urn:ivcap:collection:5f3a9c12-1b2e-4d8a-9f7e-3c0b1d2e5f6a \\
    --model-artifact-urn urn:ivcap:artifact:0b7d5c7e-ba9f-46a1-a80d-0dc6bb5a9b90 \\
    --limit 10

Create the model artifact first (once):
  make prepare-model
        """,
    )

    parser.add_argument(
        "--collection-urn",
        required=True,
        help="URN of the IVCAP collection containing bird image artifacts (e.g., urn:ivcap:collection:xxx)",
    )
    parser.add_argument(
        "--model-artifact-urn",
        required=True,
        help="URN of the EfficientNetB2 model artifact (e.g., urn:ivcap:artifact:yyy)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to fetch from the collection (0 = all, default: 0)",
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
        collection_urn=args.collection_urn,
        model_artifact_urn=args.model_artifact_urn,
        limit=args.limit if args.limit > 0 else None,
    )
