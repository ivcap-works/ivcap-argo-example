#!/bin/bash
# merge-ivcap-workflow.sh
#
# Merges ivcap.yml and image-classify-workflow.yaml into a single YAML file
# Usage: ./merge-ivcap-workflow.sh > output.yaml
# Or:    ./merge-ivcap-workflow.sh ivcap.yml image-classify-workflow.yaml > merged.yaml

set -e

# Default files
IVCAP_FILE="${1:-ivcap.yml}"
WORKFLOW_FILE="${2:-image-classify-workflow.yaml}"
OUTPUT_FILE="${3:--}"  # Default to stdout

# Check if files exist
if [ ! -f "$IVCAP_FILE" ]; then
    echo "Error: $IVCAP_FILE not found" >&2
    exit 1
fi

if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "Error: $WORKFLOW_FILE not found" >&2
    exit 1
fi

# Use yq to merge files
# This reads ivcap.yml and injects the workflow under controller.workflow
if [ "$OUTPUT_FILE" = "-" ]; then
    # Output to stdout
    yq eval-all \
        "select(fileIndex==1) as \$workflow | select(fileIndex==0) | .controller.workflow = \$workflow" \
        "$IVCAP_FILE" "$WORKFLOW_FILE"
else
    # Output to file
    yq eval-all \
        "select(fileIndex==1) as \$workflow | select(fileIndex==0) | .controller.workflow = \$workflow" \
        "$IVCAP_FILE" "$WORKFLOW_FILE" > "$OUTPUT_FILE"
    echo "✓ Merged $IVCAP_FILE and $WORKFLOW_FILE → $OUTPUT_FILE" >&2
    echo "  Note: Use 'make register-service' to replace @DOCKER_IMAGE@ placeholder with actual Docker image" >&2
fi
