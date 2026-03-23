#!/bin/bash

poetry env use python
poetry run pip install --upgrade pip
poetry run pip install /mnt/hdd/torch_wheels/torch-2.10.0-cp312-cp312-linux_x86_64.whl /mnt/hdd/torch_wheels/torch_scatter-2.1.2-cp312-cp312-linux_x86_64.whl /mnt/hdd/torch_wheels/torch_sparse-0.6.18-cp312-cp312-linux_x86_64.whl /mnt/hdd/torch_wheels/pyg_lib-0.5.0-cp312-cp312-linux_x86_64.whl
poetry install