#!/bin/bash

# Initialize uv environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv sync