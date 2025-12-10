#!/bin/bash
set -e  # Exit on any error

echo "=========================================="
echo "Block-Sparse Flash Attention Kernel Setup"
echo "=========================================="
echo ""

# Check if we're in the kernel_src directory
if [ ! -f "setup.py" ]; then
    echo "ERROR: This script must be run from the kernel_src/ directory"
    echo "Usage: cd kernel_src && ./setup.sh"
    exit 1
fi

echo "[1/4] Installing Python dependencies..."
pip install torch packaging ninja -q
echo "✓ Python dependencies installed"
echo ""

echo "[2/4] Cloning CUTLASS library..."
if [ -d "csrc/cutlass" ]; then
    echo "✓ CUTLASS already exists at csrc/cutlass"
else
    cd csrc/
    git clone https://github.com/NVIDIA/cutlass.git
    cd cutlass
    git checkout v3.5.1
    cd ../..
    echo "✓ CUTLASS v3.5.1 cloned successfully"
fi
echo ""

echo "[3/4] Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found in PATH"
    echo "If CUDA is installed, set CUDA_HOME and add to PATH:"
    echo "  export CUDA_HOME=/usr/local/cuda"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo ""
fi

# Check for A100 GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "A100"; then
        echo "✓ NVIDIA A100 GPU detected"
    else
        echo "WARNING: No A100 GPU detected. This kernel is optimized for A100 (SM_80)"
        echo "Detected GPUs:"
        nvidia-smi --query-gpu=name --format=csv,noheader
    fi
else
    echo "WARNING: nvidia-smi not found. Cannot verify GPU type"
fi
echo ""

echo "[4/4] Building kernel for A100 (SM_80)..."
echo "This may take 5-10 minutes..."
export FLASH_ATTN_CUDA_ARCHS="8.0"
export MAX_JOBS=8  # Limit parallel jobs to see errors more clearly

python setup.py install

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "Test the installation with:"
echo "  python -c 'from block_sparse_flash_attn import block_sparse_flash_attn_func; print(\"Success!\")'"
echo ""
