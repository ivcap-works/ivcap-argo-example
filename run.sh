#!/bin/bash
# Wrapper script to run dispatcher.py via poetry
# This simplifies Docker commands by collapsing 'poetry run python dispatcher.py' into a single call

exec poetry run python dispatcher.py "$@"
