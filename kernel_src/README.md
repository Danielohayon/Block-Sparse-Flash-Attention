# Block-Sparse Flash Attention Kernel

A CUDA kernel implementation of Flash Attention 2 with block-sparse support using per-head per-block thresholds.

## Overview

This kernel enables efficient sparse attention by skipping K/V blocks based on learned threshold tensors. It uses fixed block sizes (Q=128, K=64) and always executes the row1 implementation for predictable performance.

**Key Features:**
- Block-sparse attention with configurable thresholds per head and position
- Optimized for NVIDIA A100 GPUs (SM_80)
- Based on Flash Attention v2.8.3
- Fixed block sizes for consistent behavior

## Requirements

- **GPU:** NVIDIA A100 (SM_80 architecture)
- **CUDA:** 11.6 or higher (tested with CUDA 12.1)
- **Python:** 3.8+
- **PyTorch:** With CUDA support
- **Docker Image (recommended):** `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`

## Installation

### 1. Clone CUTLASS Library
```bash
cd csrc/
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout v3.5.1 && cd ../..
```

### 2. Install Python Dependencies
```bash
pip install torch packaging ninja einops
```

### 3. Build the Kernel
```bash
export FLASH_ATTN_CUDA_ARCHS="8.0"
pip install -e . --no-build-isolation
```

Or use the automated setup script:
```bash
./setup.sh
```

## Usage

```python
import torch
from block_sparse_flash_attn import block_sparse_flash_attn_func

# Input tensors (batch, seqlen, nheads, headdim)
q = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device='cuda')
v = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device='cuda')

# Threshold tensor (nheads, max_blocks)
# Blocks: seqlen / 64 = 4096 / 64 = 64 blocks
thresholds = torch.full((32, 64), -5.0, dtype=torch.float32, device='cuda')

# Run block-sparse attention
output = block_sparse_flash_attn_func(q, k, v, thresholds)
```

**Threshold Semantics:**
- Shape: `(num_heads, max_blocks)` in `float32`
- A K/V block is skipped if `max(QK^T * scale) < threshold[head, block_idx]`
- Higher thresholds = more blocks skipped = faster computation

## Block Size Configuration

The kernel uses fixed block sizes:
- **Q blocks:** 128 tokens per block
- **K/V blocks:** 64 tokens per block
- **Total blocks:** `ceil(sequence_length / 64)`

Example: For a 32,768 token sequence, there are 512 K/V blocks.

## Testing Environment

This kernel was developed and tested on:
- **GPU:** NVIDIA A100 (40GB/80GB variants)
- **CUDA Version:** 12.1
- **PyTorch Version:** 2.4.1
- **OS:** Linux (Ubuntu 20.04/22.04)

## License

Based on Flash Attention 2, licensed under BSD-3-Clause.
See: https://github.com/Dao-AILab/flash-attention
