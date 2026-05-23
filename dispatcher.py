#!/usr/bin/env python3
"""
Unified dispatcher for all image classification pipeline stages.

Routes command-line invocations to the appropriate stage function.
Each stage is called with explicit directory arguments.

Usage:
  python dispatcher.py --stage fetch --out-dir /workspace/data
  python dispatcher.py --stage preprocess --in-dir /workspace/data --out-dir /workspace/data
  python dispatcher.py --stage classify --in-dir /workspace/data --out-dir /workspace/data
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
    return logging.getLogger(__name__)


def main():
    """Main dispatcher: parse args and route to the appropriate stage."""
    parser = argparse.ArgumentParser(
        description="Image classification pipeline dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dispatcher.py --stage fetch \\
    --images-artifact-urn urn:ivcap:artifact:xxx \\
    --model-artifact-urn urn:ivcap:artifact:yyy
  python dispatcher.py --stage preprocess --in-dir /workspace/data --out-dir /workspace/data
  python dispatcher.py --stage classify --in-dir /workspace/data --out-dir /workspace/data
        """,
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["fetch", "preprocess", "classify"],
        help="Which pipeline stage to run",
    )
    parser.add_argument(
        "--images-artifact-urn",
        default=None,
        help="URN of the images artifact (required for fetch stage)",
    )
    parser.add_argument(
        "--model-artifact-urn",
        default=None,
        help="URN of the model artifact (required for fetch stage)",
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
            images_urn = args.images_artifact_urn or os.environ.get(
                "IMAGES_ARTIFACT_URN"
            )
            model_urn = args.model_artifact_urn or os.environ.get("MODEL_ARTIFACT_URN")

            logger.info(f"Running fetch stage with OUT_DIR={out_dir}")
            logger.info(f"  Images artifact: {images_urn}")
            logger.info(f"  Model artifact: {model_urn}")

            fetch_stage(
                out_dir=out_dir,
                images_artifact_urn=images_urn,
                model_artifact_urn=model_urn,
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
