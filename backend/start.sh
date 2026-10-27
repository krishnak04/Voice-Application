#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
VENV_DIR="/home/i3admin/stt-base-en/venv"
# Discover pip-installed nvidia CUDA libraries inside the virtualenv
CUBLAS_LIB=$(find "$VENV_DIR/lib" -type d -path "*/nvidia/cublas/lib" 2>/dev/null | head -n 1)
CUDNN_LIB=$(find "$VENV_DIR/lib" -type d -path "*/nvidia/cudnn/lib" 2>/dev/null | head -n 1)
NVJIT_LIB=$(find "$VENV_DIR/lib" -type d -path "*/nvidia/nvjitlink/lib" 2>/dev/null | head -n 1)
CUDA_RT=$(find "$VENV_DIR/lib" -type d -path "*/nvidia/cuda_runtime/lib" 2>/dev/null | head -n 1)
export LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB:$NVJIT_LIB:$CUDA_RT:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="$VENV_DIR/bin:/usr/local/cuda/bin:$PATH"
exec "$VENV_DIR/bin/python" run_server.py
