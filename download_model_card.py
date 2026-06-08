#!/usr/bin/env python3
"""
Download and parse the model card (README.md) and metadata for a HuggingFace
model using only the stdlib + pydantic — no huggingface_hub required.

HuggingFace public endpoints used:
  - Model card markdown : https://huggingface.co/{model_id}/raw/main/README.md
  - Model metadata JSON : https://huggingface.co/api/models/{model_id}

The resulting ModelCard Pydantic model captures the fields most relevant to
deciding whether a model is suitable for a particular use case:
  - task type, license, base model, intended uses
  - training dataset (name, URL, class count, split sizes, image resolution)
  - measured accuracy (train / validation / test)
  - preprocessing requirements
"""

import argparse
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

DEFAULT_MODEL_ID = "dennisjooo/Birds-Classifier-EfficientNetB2"
HF_BASE = "https://huggingface.co"

logger = logging.getLogger(__name__)


# ── Sub-models ─────────────────────────────────────────────────────────────────


class TrainingDataset(BaseModel):
    """Dataset used to train / evaluate the model."""

    name: str = Field(description="Dataset name or identifier")
    url: Optional[str] = Field(None, description="Canonical URL for the dataset")
    num_classes: Optional[int] = Field(
        None, description="Number of output classes / labels"
    )
    num_training_samples: Optional[int] = Field(
        None, description="Number of training examples"
    )
    num_validation_samples: Optional[int] = Field(
        None, description="Number of validation examples"
    )
    num_test_samples: Optional[int] = Field(None, description="Number of test examples")
    image_size: Optional[str] = Field(
        None, description="Input image resolution, e.g. '224x224'"
    )


class AccuracyMetrics(BaseModel):
    """Accuracy scores reported on each data split."""

    training: Optional[float] = Field(None, description="Accuracy on the training set")
    validation: Optional[float] = Field(
        None, description="Accuracy on the validation set"
    )
    test: Optional[float] = Field(None, description="Accuracy on the held-out test set")


class ModelCard(BaseModel):
    """
    Structured representation of a HuggingFace model card, focused on the
    information needed to evaluate suitability for a target use case.
    """

    # ── Identity ───────────────────────────────────────────────────────────────
    id: str = Field(description="HuggingFace model ID, e.g. 'org/model-name'")
    author: str = Field(description="Model author or organisation")
    url: str = Field(description="Canonical HuggingFace URL for this model")
    tags: List[str] = Field(
        default_factory=list, description="All tags from the HF API"
    )

    # ── Usage rights & task ────────────────────────────────────────────────────
    license: Optional[str] = Field(
        None, description="SPDX license identifier, e.g. 'apache-2.0'"
    )
    pipeline_tag: Optional[str] = Field(
        None, description="ML task type, e.g. 'image-classification'"
    )

    # ── Provenance ─────────────────────────────────────────────────────────────
    base_model: Optional[str] = Field(
        None, description="Pre-trained base model this was fine-tuned from"
    )

    # ── Suitability narrative ──────────────────────────────────────────────────
    model_description: Optional[str] = Field(
        None,
        description="Plain-text model description extracted from the README "
        "(code blocks stripped)",
    )
    intended_uses: Optional[str] = Field(
        None,
        description="Intended uses / limitations extracted from the README "
        "(code blocks stripped)",
    )
    preprocessing_info: Optional[str] = Field(
        None, description="Input preprocessing requirements extracted from the README"
    )

    # ── Data & performance ─────────────────────────────────────────────────────
    training_dataset: Optional[TrainingDataset] = Field(
        None, description="Structured information about the training dataset"
    )
    accuracy: Optional[AccuracyMetrics] = Field(
        None, description="Reported accuracy scores across data splits"
    )

    # ── Full text ──────────────────────────────────────────────────────────────
    description: str = Field(description="Full content of the model card README.md")

    # ── Derived fields ─────────────────────────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def derive_url(cls, data: dict) -> dict:
        """Compute `url` from `id` if not already provided."""
        if not data.get("url"):
            data["url"] = f"{HF_BASE}/{data['id']}"
        return data


# ── Markdown / YAML helpers ────────────────────────────────────────────────────


def _parse_frontmatter(readme: str) -> dict:
    """
    Parse the YAML frontmatter between the opening --- delimiters.
    Handles simple key: value pairs and list items (- value).
    """
    match = re.match(r"^---\n(.*?)\n---\n", readme, re.DOTALL)
    if not match:
        return {}
    result: dict = {}
    current_key: Optional[str] = None
    for line in match.group(1).splitlines():
        if line.startswith("- ") and current_key:
            if isinstance(result.get(current_key), list):
                result[current_key].append(line[2:].strip())
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val:
                result[key] = val
                current_key = None
            else:
                result[key] = []
                current_key = key
    return result


def _extract_section(readme: str, heading: str) -> Optional[str]:
    """
    Extract the body text of a markdown section matching *heading* (any level).
    Stops at the next heading of the same or higher level.
    """
    # Use {{}} to emit literal braces in the regex (f-string escaping)
    pattern = rf"#{{1,4}}\s+{re.escape(heading)}\s*\n(.*?)(?=\n#{{1,4}}\s|\Z)"
    match = re.search(pattern, readme, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _strip_code_blocks(text: str) -> str:
    """Replace fenced code blocks with a short placeholder."""
    return re.sub(
        r"```.*?```", "[see model card for code example]", text, flags=re.DOTALL
    ).strip()


def _parse_dataset(section: str) -> Optional[TrainingDataset]:
    """Extract structured dataset fields from the ### Data section text."""
    if not section:
        return None

    # First markdown link → dataset name + URL
    link = re.search(r"\[([^\]]+)\]\(([^)]+)\)", section)
    name = link.group(1) if link else "unknown"
    url = link.group(2) if link else None

    def _int(pattern: str) -> Optional[int]:
        m = re.search(pattern, section, re.IGNORECASE)
        return int(m.group(1).replace(",", "")) if m else None

    num_classes = _int(r"([\d,]+)\s+bird species")
    num_training = _int(r"([\d,]+)\s+training images")
    # "2,625 each for validation and test images" → same count for both splits
    num_val = _int(r"([\d,]+)\s+(?:each for )?validation")
    num_test = _int(r"([\d,]+)\s+(?:each for )?test\b")
    # If "X each for validation and test", num_test == num_val
    if num_test is None and num_val is not None:
        if re.search(r"each for validation and test", section, re.IGNORECASE):
            num_test = num_val

    size_match = re.search(r"(\d+)\s+by\s+(\d+)", section)
    image_size = f"{size_match.group(1)}x{size_match.group(2)}" if size_match else None

    return TrainingDataset(
        name=name,
        url=url,
        num_classes=num_classes,
        num_training_samples=num_training,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        image_size=image_size,
    )


def _parse_accuracy(readme: str) -> Optional[AccuracyMetrics]:
    """Extract train / val / test accuracy values from bold list items."""

    def _float(pattern: str) -> Optional[float]:
        m = re.search(pattern, readme, re.IGNORECASE)
        return float(m.group(1)) if m else None

    train = _float(r"\*\*Training\*\*[:\s]*([\d.]+)")
    val = _float(r"\*\*Validation\*\*[:\s]*([\d.]+)")
    test = _float(r"\*\*Test\*\*[:\s]*([\d.]+)")

    if any(v is not None for v in (train, val, test)):
        return AccuracyMetrics(training=train, validation=val, test=test)
    return None


# ── Network helper ─────────────────────────────────────────────────────────────


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


# ── Main ───────────────────────────────────────────────────────────────────────


def download_model_card(
    model_id: str, out_dir: str = ".", quiet: bool = False
) -> ModelCard:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch README.md ─────────────────────────────────────────────────────
    card_url = f"{HF_BASE}/{model_id}/raw/main/README.md"
    logger.info("Fetching model card from: %s", card_url)
    readme = _fetch(card_url).decode("utf-8")
    logger.debug("Retrieved model card (%d chars)", len(readme))

    # ── 2. Fetch API metadata ──────────────────────────────────────────────────
    api_url = f"{HF_BASE}/api/models/{model_id}"
    logger.info("Fetching model metadata from: %s", api_url)
    raw = json.loads(_fetch(api_url))
    logger.debug("Retrieved model metadata")

    # ── 3. Parse README ────────────────────────────────────────────────────────
    logger.debug("Parsing README frontmatter and sections")
    frontmatter = _parse_frontmatter(readme)

    model_desc_raw = _extract_section(readme, "Model Description")
    intended_raw = _extract_section(readme, "Intended Uses")
    preprocessing_raw = _extract_section(readme, "Preprocessing")
    data_section = _extract_section(readme, "Data")

    # ── 4. Build Pydantic model ────────────────────────────────────────────────
    logger.debug("Building ModelCard object")
    card = ModelCard(
        id=raw["id"],
        author=raw.get("author", ""),
        url="",  # derived by model_validator
        tags=raw.get("tags", []),
        license=frontmatter.get("license"),
        pipeline_tag=frontmatter.get("pipeline_tag") or raw.get("pipeline_tag"),
        base_model=frontmatter.get("base_model"),
        model_description=_strip_code_blocks(model_desc_raw)
        if model_desc_raw
        else None,
        intended_uses=_strip_code_blocks(intended_raw) if intended_raw else None,
        preprocessing_info=preprocessing_raw,
        training_dataset=_parse_dataset(data_section),
        accuracy=_parse_accuracy(readme),
        description=readme,
    )

    # ── 5. Persist ─────────────────────────────────────────────────────────────
    card_file = out_path / "model_card.md"
    card_file.write_text(readme, encoding="utf-8")
    logger.info("Saved model card     → %s", card_file)

    meta_file = out_path / "model_metadata.json"
    meta_file.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved model metadata → %s", meta_file)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    if not quiet:
        ds = card.training_dataset
        acc = card.accuracy
        print(f"""
── ModelCard suitability summary ────────────────────
  id            : {card.id}
  author        : {card.author}
  url           : {card.url}
  license       : {card.license}
  pipeline_tag  : {card.pipeline_tag}
  base_model    : {card.base_model}
  tags          : {", ".join(card.tags[:6])}{"…" if len(card.tags) > 6 else ""}

── Training data ─────────────────────────────────────""")
        if ds:
            print(f"  dataset       : {ds.name}")
            print(f"  url           : {ds.url}")
            print(f"  classes       : {ds.num_classes}")
            print(
                f"  train samples : {ds.num_training_samples:,}"
                if ds.num_training_samples
                else "  train samples : n/a"
            )
            print(
                f"  val   samples : {ds.num_validation_samples:,}"
                if ds.num_validation_samples
                else "  val   samples : n/a"
            )
            print(
                f"  test  samples : {ds.num_test_samples:,}"
                if ds.num_test_samples
                else "  test  samples : n/a"
            )
            print(f"  image size    : {ds.image_size}")
        if acc:
            print(f"\n── Accuracy ──────────────────────────────────────────")
            print(
                f"  train / val / test : {acc.training} / {acc.validation} / {acc.test}"
            )

    return card


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and parse a HuggingFace model card and metadata."
    )
    parser.add_argument(
        "--model",
        metavar="ORG/MODEL",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID to download (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default=".",
        help="Directory where model_card.md and model_metadata.json are written (default: .)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the suitability summary; only INFO/DEBUG log lines are emitted",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
    )

    download_model_card(model_id=args.model, out_dir=args.out_dir, quiet=args.quiet)
