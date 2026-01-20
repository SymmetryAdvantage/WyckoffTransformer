#!/bin/bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Error: This script must be executed, not sourced."
    echo "Usage: ./$(basename "${BASH_SOURCE[0]}")"
    return 1
fi

set -e

POETRY_LOCK_FILE="poetry.lock"
POETRY_LOCK_SRC="../poetry.lock.zeus"
PYPROJECT_TOML_FILE="pyproject.toml"
PYPROJECT_TOML_SRC="../pyproject.toml.zeus"


if [ ! -e "$POETRY_LOCK_FILE" ]; then
  cp $POETRY_LOCK_SRC $POETRY_LOCK_FILE
  echo "Initialized $POETRY_LOCK_FILE from $POETRY_LOCK_SRC"
else
  echo "Found $POETRY_LOCK_FILE. Diff to $POETRY_LOCK_SRC (if any):"
  diff $POETRY_LOCK_SRC $POETRY_LOCK_FILE
  echo "end diff"
fi

if [ ! -e "$PYPROJECT_TOML_FILE" ]; then
  cp $PYPROJECT_TOML_SRC $PYPROJECT_TOML_FILE
  echo "Initialized $PYPROJECT_TOML_FILE from $POETRY_LOCK_FILE (you likely need to update it for your environment)"
else
  echo "Found $PYPROJECT_TOML_FILE. Diff to $PYPROJECT_TOML_SRC (if any):"
  diff $PYPROJECT_TOML_SRC $PYPROJECT_TOML_FILE
  echo "end diff"
fi


cd .. && docker build -t wyckoff-transformer -f docker/Dockerfile .
