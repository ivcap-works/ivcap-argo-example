#!/usr/bin/env python3
"""
Unified dispatcher for all bird species classification pipeline stages.

Routes command-line invocations to the appropriate stage function.
Each stage is called with explicit directory arguments.

Usage:
  python dispatcher.py --stage fetch \\
    --collection-urn     urn:ivcap:collection:xxx \\
    --model-artifact-urn urn:ivcap:artifact:yyy \\
    --out-dir /workspace/data
  python dispatcher.py --stage fetch \\
    --collection-urn     urn:ivcap:collection:xxx \\
    --model-artifact-urn urn:ivcap:artifact:yyy \\
    --limit 10 \\
    --out-dir /workspace/data
  python dispatcher.py --stage preprocess --in-dir /workspace/data --out-dir /workspace/data
  python dispatcher.py --stage classify   --in-dir /workspace/data --out-dir /workspace/data
"""

import argparse
import sys
import logging
import os

# Import all stage functions
from stage1_fetch import fetch_stage
from stage2_preprocess import preprocess_stage
from stage3_classify import classify_stage


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging based on LOG_LEVEL environment variable."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    # httpx (used by ivcap-client) logs every HTTP request at INFO level.
    # Suppress those to WARNING to keep pipeline output readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def main():
    """Main dispatcher: parse args and route to the appropriate stage."""
    parser = argparse.ArgumentParser(
        description="Bird species classification pipeline dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create the model artifact once (before the first pipeline run):
  python prepare_model.py

  # Then run the pipeline stages:
  python dispatcher.py --stage fetch \\
    --collection-urn     urn:ivcap:collection:xxx \\
    --model-artifact-urn urn:ivcap:artifact:yyy \\
    --out-dir /workspace/data

  # Limit to 10 images from the collection:
  python dispatcher.py --stage fetch \\
    --collection-urn     urn:ivcap:collection:xxx \\
    --model-artifact-urn urn:ivcap:artifact:yyy \\
    --limit 10 \\
    --out-dir /workspace/data

  python dispatcher.py --stage preprocess --in-dir /workspace/data --out-dir /workspace/data
  python dispatcher.py --stage classify   --in-dir /workspace/data --out-dir /workspace/data
        """,
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["fetch", "preprocess", "classify"],
        help="Which pipeline stage to run",
    )
    parser.add_argument(
        "--collection-urn",
        default=None,
        help=(
            "URN of the IVCAP collection containing bird image artifacts "
            "(required for fetch stage)"
        ),
    )
    parser.add_argument(
        "--model-artifact-urn",
        default=None,
        help="URN of the EfficientNetB2 model artifact (required for fetch stage)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of images to fetch from the collection "
            "(0 = no limit, default: 0; fetch stage only)"
        ),
    )
    parser.add_argument(
        "--in-dir",
        default=None,
        help="Input directory (used by preprocess and classify stages)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (used by all stages)",
    )

    args = parser.parse_args()

    # Get log level from environment
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logger = setup_logging(log_level)

    try:
        if args.stage == "fetch":
            out_dir = args.out_dir or os.environ.get("OUT_DIR", "/tmp/outputs")
            collection_urn = args.collection_urn or os.environ.get("COLLECTION_URN")
            model_urn = args.model_artifact_urn or os.environ.get("MODEL_ARTIFACT_URN")
            limit = args.limit if args.limit > 0 else None

            logger.info(f"Running fetch stage with OUT_DIR={out_dir}")
            logger.info(f"  Bird images collection   : {collection_urn}")
            logger.info(f"  EfficientNetB2 model artifact: {model_urn}")
            if limit:
                logger.info(f"  Image limit              : {limit}")

            fetch_stage(
                out_dir=out_dir,
                collection_urn=collection_urn,
                model_artifact_urn=model_urn,
                limit=limit,
            )

        elif args.stage == "preprocess":
            in_dir = args.in_dir or os.environ.get("IN_DIR", "/tmp/outputs")
            out_dir = args.out_dir or os.environ.get("OUT_DIR", "/tmp/outputs")
            logger.info(
                f"Running preprocess stage with IN_DIR={in_dir}, OUT_DIR={out_dir}"
            )
            preprocess_stage(in_dir=in_dir, out_dir=out_dir)

        elif args.stage == "classify":
            in_dir = args.in_dir or os.environ.get("IN_DIR", "/tmp/outputs")
            out_dir = args.out_dir or os.environ.get("OUT_DIR", "/tmp/outputs")
            logger.info(
                f"Running classify stage with IN_DIR={in_dir}, OUT_DIR={out_dir}"
            )
            classify_stage(in_dir=in_dir, out_dir=out_dir)

    except Exception as exc:
        logger.error(f"Stage '{args.stage}' failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
