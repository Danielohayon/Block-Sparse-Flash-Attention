# Thresholds

Pre-computed and custom threshold calibration for block-sparse attention.

## Pre-computed Thresholds

Ready-to-use thresholds for Llama-3.1-8B-Instruct are in [`llama_3.1_8B_instruct/`](llama_3.1_8B_instruct/):

| File | Sequence Length | Size |
|------|-----------------|------|
| `thresholds_32k.pt` | Up to 32K tokens | 16 MB |
| `thresholds_64k.pt` | Up to 64K tokens | 31 MB |
| `thresholds_128k.pt` | Up to 128K tokens | 75 MB |

**Usage:**
```python
from custom_attention_injector import AttentionConfig

config = AttentionConfig(
    mode="block_sparse",
    threshold_file="./thresholds/llama_3.1_8B_instruct/thresholds_64k.pt",
    block_sparse_topk=64,
)
```

## Calibrating Custom Thresholds

To create thresholds for your own model or dataset, use `calibrate.py`:

```bash
python calibrate.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_file /path/to/calibration_data.json \
    --q_block_size 128 \
    --k_block_size 64 \
    --max_seq_len 65536 \
    --num_layers 32 \
    --num_heads 32 \
    --k_values "16,32,64,96,128" \
    --num_examples 16 \
    --thresholds_save_path ./my_thresholds.pt
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name | Required |
| `--data_file` | Path to calibration data (JSON) | Required |
| `--q_block_size` | Query block size | 128 |
| `--k_block_size` | Key block size | 64 |
| `--max_seq_len` | Maximum sequence length | 16384 |
| `--k_values` | Comma-separated k values | "16,32,64,96,128" |
| `--num_examples` | Number of calibration samples | 16 |

### Output Format

The script generates `*.inference.pt` files containing:
```python
{
    'thresholds': torch.Tensor,  # [num_k_values, num_layers, num_heads, max_blocks]
    'k_values': List[int],       # e.g., [16, 32, 64, 96, 128]
    'q_block_size': int,         # 128
    'k_block_size': int,         # 64
}
```

### Calibration Data Format

JSON file with a list of examples:
```json
[
    {"input_text": "Your first calibration sample...", "task_name": "task1"},
    {"input_text": "Your second calibration sample...", "task_name": "task1"},
    {"input_text": "Another sample...", "task_name": "task2"}
]
```

- `input_text`: The text to process (required)
- `task_name`: Optional grouping key. When using `--num_examples`, samples are taken per task.

Use 16+ samples of representative text at your target sequence length.

## Block Semantics

- **Q block size**: 128 tokens per query block
- **K block size**: 64 tokens per key/value block
- **Diagonal blocks**: Always included (local context)
- **Top-k**: Number of additional off-diagonal blocks to retain

Example with k=64: Each query attends to 64 off-diagonal K/V blocks plus diagonal blocks.
