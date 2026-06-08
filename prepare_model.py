#!/usr/bin/env -S poetry run python3
"""
Prepare the Birds-Classifier-EfficientNetB2 model as an IVCAP artifact.

Downloads the model and image processor from Hugging Face Hub, saves them to a
local directory, packages everything (including the model card) into a zip
archive, and uploads the archive as an IVCAP artifact.

After a successful upload:
  - The zip file is renamed to include the artifact UUID so that the Makefile
    can derive the URN automatically:
        data/efficientnet-birds-<UUID>.zip  →  urn:ivcap:artifact:<UUID>
  - model_card.json (renamed from model_metadata.json) is attached as an aspect
    to the model artifact with schema urn:ivcap:schema:model-card.1

Usage:
    python prepare_model.py [--model ORG/MODEL] [--data-dir ./data] [--no-upload]

This script is for LOCAL DEVELOPMENT / CI SETUP ONLY and is not included in
the Docker image — the model is loaded at runtime from the IVCAP artifact.
"""

import json
import logging
import os
import re
import shutil
import sys
import zipfile
import argparse
from pathlib import Path

DEFAULT_MODEL_ID = "dennisjooo/Birds-Classifier-EfficientNetB2"
MODEL_CARD_SCHEMA = "urn:ivcap:schema:model-card.1"

logger = logging.getLogger(__name__)


# ── Download model ─────────────────────────────────────────────────────────────


def download_model(model_id: str, model_dir: Path) -> None:
    """Download EfficientNetB2 model + processor from Hugging Face and save locally."""
    try:
        from transformers import (
            EfficientNetImageProcessor,
            EfficientNetForImageClassification,
        )
    except ImportError:
        logger.error("transformers not installed.")
        logger.error("  Install with: poetry add transformers torch safetensors")
        sys.exit(1)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already downloaded
    if (model_dir / "config.json").exists():
        logger.info("Model already present in %s — skipping download", model_dir)
        return

    logger.info("Downloading processor from %s …", model_id)
    processor = EfficientNetImageProcessor.from_pretrained(model_id)
    processor.save_pretrained(str(model_dir))
    logger.info("Processor saved")

    logger.info("Downloading model weights from %s …", model_id)
    model = EfficientNetForImageClassification.from_pretrained(model_id)
    model.save_pretrained(str(model_dir))

    n_labels = len(model.config.id2label)
    logger.info("Model saved  (%d bird species labels)", n_labels)


# ── Download model card ────────────────────────────────────────────────────────


def download_model_card_for_artifact(
    model_id: str, model_dir: Path, data_path: Path
) -> Path | None:
    """
    Download the HuggingFace model card for *model_id*.

    - model_card.md  is written into *model_dir* so it is bundled in the zip.
    - model_metadata.json is also written to *model_dir*; a copy is placed at
      *data_path*/model_card.json for the IVCAP aspect upload.

    Returns the path to *data_path*/model_card.json, or None on failure.
    """
    try:
        from download_model_card import download_model_card
    except ImportError:
        logger.warning(
            "download_model_card module not found — skipping model card download"
        )
        return None

    try:
        logger.info("Downloading model card for %s …", model_id)
        download_model_card(model_id=model_id, out_dir=str(model_dir), quiet=True)

        # model_card.md + model_metadata.json are now in model_dir (zipped later).
        # Copy metadata JSON to data_path under the canonical name.
        src = model_dir / "model_metadata.json"
        dst = data_path / "model_card.json"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Model card JSON written to %s", dst)
            return dst
        else:
            logger.warning("model_metadata.json not found in %s", model_dir)
            return None
    except Exception as exc:
        logger.warning("Could not download model card: %s", exc)
        return None


# ── Zip ───────────────────────────────────────────────────────────────────────


def zip_model_directory(model_dir: Path) -> Path:
    """Zip all files in model_dir, return path to zip."""
    zip_path = model_dir.parent / "efficientnet-birds.zip"

    logger.info("Creating zip archive of %s …", model_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(model_dir.glob("**/*")):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(model_dir))

    size_mb = zip_path.stat().st_size / 1e6
    logger.info("efficientnet-birds.zip  (%.1f MB)", size_mb)
    return zip_path


# ── Upload artifact ────────────────────────────────────────────────────────────


def upload_to_ivcap(ivcap, zip_path: Path, model_id: str) -> tuple[str, str]:
    """
    Upload zip to IVCAP.

    Returns (artifact_urn, uuid).
    """
    logger.info("Uploading %s to IVCAP …", zip_path.name)
    artifact = ivcap.upload_artifact(name=f"{model_id} model", file_path=str(zip_path))

    artifact_urn = str(artifact.id)
    match = re.search(r"([a-f0-9\-]+)$", artifact_urn)
    uuid = match.group(1) if match else artifact_urn

    logger.info("Uploaded  URN: %s", artifact_urn)
    return artifact_urn, uuid


# ── Upload model card aspect ───────────────────────────────────────────────────


def upload_model_card_aspect(ivcap, artifact_urn: str, model_card_json: Path) -> None:
    """
    Attach the model card as an aspect on the model artifact entity.

    Schema: urn:ivcap:schema:model-card.1
    """
    if not model_card_json.exists():
        logger.warning(
            "model_card.json not found at %s — skipping aspect upload", model_card_json
        )
        return

    body = json.loads(model_card_json.read_text(encoding="utf-8"))
    ivcap.add_aspect(entity=artifact_urn, aspect=body, schema=MODEL_CARD_SCHEMA)
    logger.info("Model card aspect attached  (schema: %s)", MODEL_CARD_SCHEMA)


# ── Main ──────────────────────────────────────────────────────────────────────


def prepare_model(
    model_id: str = DEFAULT_MODEL_ID,
    data_dir: str = "./data",
    upload: bool = True,
) -> None:
    data_path = Path(data_dir)
    model_dir = data_path / "model"

    # ── Initialise IVCAP client eagerly so auth failures surface before the
    #    time-consuming model download starts ──────────────────────────────────
    ivcap = None
    if upload:
        try:
            from ivcap_client import IVCAP
        except ImportError:
            logger.error("ivcap-client not installed.")
            logger.error("  Install with: poetry add ivcap-client")
            sys.exit(1)
        ivcap = IVCAP()  # raises immediately if IVCAP_URL / IVCAP_JWT are missing

    logger.info("Bird Species Classifier — EfficientNetB2")
    logger.info("  HuggingFace model : %s", model_id)
    logger.info("  Local model dir   : %s", model_dir.resolve())

    # 1. Download model weights + processor from HuggingFace
    download_model(model_id, model_dir)

    # 2. Download model card (stored in model_dir → included in zip)
    model_card_json = download_model_card_for_artifact(model_id, model_dir, data_path)

    if not upload or ivcap is None:
        logger.warning("--no-upload set: skipping IVCAP upload.")
        return

    # 3. Zip model directory (includes model_card.md + model_metadata.json)
    zip_path = zip_model_directory(model_dir)

    # 4. Upload zip to IVCAP
    try:
        artifact_urn, uuid = upload_to_ivcap(ivcap, zip_path, model_id)
    except Exception as exc:
        logger.error("IVCAP upload failed: %s", exc)
        logger.error("The zip is still available locally for manual upload.")
        return

    # 5. Rename zip to embed UUID so the Makefile can auto-detect the URN
    named_zip = zip_path.parent / f"efficientnet-birds-{uuid}.zip"
    zip_path.rename(named_zip)
    logger.info("Renamed zip to %s", named_zip.name)

    # 6. Write a small JSON sidecar so other scripts can read the URN without
    #    relying on filename glob parsing.
    sidecar = data_path / "model_artifact.json"
    with open(sidecar, "w") as f:
        json.dump(
            {"model_id": model_id, "artifact_urn": artifact_urn},
            f,
            indent=2,
        )
    logger.info("Artifact URN written to %s", sidecar)

    # 7. Attach model card as an aspect on the artifact entity
    if model_card_json is not None:
        upload_model_card_aspect(ivcap, artifact_urn, model_card_json)

    logger.info("Model artifact ready!")
    logger.info("  URN : %s", artifact_urn)
    logger.info("  Use this URN as --model-artifact-urn when running the pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download EfficientNetB2 bird classifier and upload as IVCAP artifact"
    )
    parser.add_argument(
        "--model",
        metavar="ORG/MODEL",
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        help=f"HuggingFace model ID to download (default: {DEFAULT_MODEL_ID})",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    prepare_model(
        model_id=args.model,
        data_dir=args.data_dir,
        upload=not args.no_upload,
    )
