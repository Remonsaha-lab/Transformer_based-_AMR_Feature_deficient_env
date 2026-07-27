#!/usr/bin/env bash
set -euo pipefail

# DIODE-only EZ-SP installer.
# This project uses the repository virtual environment:
#   /home/remon/EZ_SP_CFDVri/.venv
# It does not create or activate the old Conda `spt` environment.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip)

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${PYTHON} was not found."
    echo "Create the environment first with:"
    echo "  python3.10 -m venv ${VENV_DIR}"
    exit 1
fi

PYTHON_VERSION="$(${PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
    echo "ERROR: This project environment expects Python 3.10; found ${PYTHON_VERSION}."
    exit 1
fi

CUDA_VERSION=""
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([^,]*\),.*/\1/p' | tail -1)"
fi
if [[ "${CUDA_VERSION}" != "12.1" ]]; then
    echo "ERROR: CUDA 12.1 was expected, but nvcc reported '${CUDA_VERSION:-not found}'."
    echo "The installed PyTorch/PyG wheels are cu121 wheels."
    exit 1
fi

echo "Using Python: $(${PYTHON} --version)"
echo "Using CUDA: ${CUDA_VERSION}"
echo "Using virtual environment: ${VENV_DIR}"

"${PIP[@]}" install --upgrade pip setuptools wheel
"${PIP[@]}" install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html \
    -r "${PROJECT_DIR}/requirements.txt"

# FRNN is compiled locally because it is not distributed as a normal project
# wheel. Reuse an existing checkout/build when it is already installed.
FRNN_DIR="${PROJECT_DIR}/src/dependencies/FRNN"
if ! "${PYTHON}" -c 'import frnn' >/dev/null 2>&1; then
    if [[ ! -d "${FRNN_DIR}/.git" ]]; then
        git clone --recursive https://github.com/lxxue/FRNN.git "${FRNN_DIR}"
    fi
    (
        cd "${FRNN_DIR}/external/prefix_sum"
        "${PIP[@]}" install .
    )
    (
        cd "${FRNN_DIR}"
        "${PIP[@]}" install .
    )
else
    echo "FRNN is already installed; skipping local compilation."
fi

echo
echo "DIODE environment installation completed."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "Verify it with: ${PYTHON} -c \"import torch, torch_geometric, pgeof, frnn; print(torch.__version__, torch.cuda.is_available())\""
