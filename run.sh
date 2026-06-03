#!/bin/bash
# Wrapper script to run dispatcher.py
# Dependencies are installed system-wide in the Docker image (poetry virtualenvs.create false),
# so we invoke python directly to avoid Poetry's "Skipping virtualenv creation" noise.

exec python dispatcher.py "$@"
