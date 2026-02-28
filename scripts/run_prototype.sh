#!/bin/bash
# Run the prototype app in the opedDev conda env
CONDA_ENV=opedDev_py311
APP=repositories/opedDev/prototype/app.py

echo "Starting prototype app (conda env: $CONDA_ENV)"
/home/pacificDev/.miniconda/bin/conda run -n "$CONDA_ENV" python -u "$APP"
