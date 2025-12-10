# Model Patching for Block-Sparse Attention

Runtime injection of custom attention mechanisms into HuggingFace Transformers models without modifying library code.

## Overview

This module enables testing block-sparse and other custom attention implementations on transformer models through runtime monkey-patching. Load your model once and switch between different attention configurations instantly.

## Quick Start

```python
from transformers import AutoModelForCausalLM
from custom_attention_injector import AttentionConfig, inject_custom_attention

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Configure block-sparse attention
config = AttentionConfig(
    mode="block_sparse",
    threshold_file="./thresholds.pt",
    block_sparse_topk=16,
)

# Inject custom attention
inject_custom_attention(model, config)

# Use model normally - now uses block-sparse attention
outputs = model.generate(...)
```

## Supported Attention Modes

### 1. Block-Sparse Attention

Uses the block-sparse Flash Attention kernel with learned thresholds.

```python
config = AttentionConfig(
    mode="block_sparse",
    threshold_file="./thresholds.pt",
    block_sparse_topk=16,
)
```

**Threshold file format:**
```python
{
    'thresholds': torch.Tensor,  # [num_k_values, num_layers, num_heads, max_blocks]
    'k_values': List[int],        # e.g., [8, 16, 32, 64]
}
```

### 2. SpargeAttn Attention

Sparse attention with adaptive masking.

```python
config = AttentionConfig(
    mode="sparge_attn",
    sparge_attn_params={
        "simthreshd1": -0.1,
        "topk": 0.35,
        "pvthreshd": 15,
    }
)
```

### 3. SDPA (Baseline)

Standard PyTorch scaled dot-product attention for comparison.

```python
config = AttentionConfig(mode="sdpa")
```

## Runtime Configuration Changes

Change attention configuration without reloading the model:

```python
from custom_attention_injector import update_block_sparse_topk

# Initial configuration
inject_custom_attention(model, AttentionConfig(
    mode="block_sparse",
    threshold_file="./thresholds.pt",
    block_sparse_topk=16,
))

# Test different sparsity levels
update_block_sparse_topk(model, 32)  # More blocks
update_block_sparse_topk(model, 8)   # Fewer blocks
```

## Common Use Cases

### Grid Search for Optimal Sparsity

```python
k_values = [8, 16, 32, 64]
results = {}

# Inject once
inject_custom_attention(model, AttentionConfig(
    mode="block_sparse",
    threshold_file="./thresholds.pt",
    block_sparse_topk=8,
))

# Test different values
for k in k_values:
    update_block_sparse_topk(model, k)
    results[k] = evaluate(model)

best_k = max(results, key=lambda k: results[k]["accuracy"])
print(f"Optimal k: {best_k}")
```

### Compare Attention Methods

```python
from custom_attention_injector import change_attention_config

# Test block-sparse
config_sparse = AttentionConfig(
    mode="block_sparse",
    threshold_file="./thresholds.pt",
    block_sparse_topk=16,
)
inject_custom_attention(model, config_sparse)
results_sparse = evaluate(model)

# Compare with baseline
config_baseline = AttentionConfig(mode="sdpa")
change_attention_config(model, config_baseline)
results_baseline = evaluate(model)

print(f"Block-sparse: {results_sparse}")
print(f"Baseline: {results_baseline}")
```

## Key Functions

- **`inject_custom_attention(model, config)`** - Initial injection of custom attention
- **`change_attention_config(model, new_config)`** - Switch attention mode completely
- **`update_block_sparse_topk(model, new_topk)`** - Quick update for block-sparse top-k value
- **`update_sparge_attn_params(model, **params)`** - Fine-tune SpargeAttn parameters
- **`get_current_attention_config(model)`** - Inspect current configuration

## Environment Variables

Set configuration via environment variables:

```bash
export CUSTOM_ATTENTION_MODE=block_sparse
export CUSTOM_THRESHOLD_FILE=./thresholds.pt
export CUSTOM_BLOCK_SPARSE_TOPK=16
```

```python
# Automatically reads from environment
inject_custom_attention(model)
```

## Requirements

- PyTorch with CUDA support
- HuggingFace Transformers
- Block-sparse Flash Attention kernel (for block_sparse mode)
- SpargeAttn kernel (for sparge_attn mode)

## Supported Models

Currently supports Llama architecture. Extensible to other models by adding model-specific injection functions.

## Performance Notes

- **Injection overhead:** ~2-3 seconds (one-time cost)
- **Runtime switching:** ~100ms
- **Forward pass:** No overhead vs native implementation

## License

This code is provided for research and experimentation. Follow the licenses of HuggingFace Transformers (Apache 2.0) and the respective attention kernel implementations.
