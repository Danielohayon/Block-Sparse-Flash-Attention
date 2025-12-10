import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Any, Set, Optional
import os
from tqdm import tqdm
import argparse


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class ThresholdCalculator:
    """Calculate top-k thresholds for each query block position across all examples."""

    def __init__(
        self,
        k_values: Set[int],
        max_num_blocks: int,
        num_layers: int,
        num_heads: int,
        q_block_size: int = 128,
        k_block_size: int = 128,
        aggregation_method: str = 'median'
    ):
        """
        Initialize the threshold calculator.

        Args:
            k_values: Set of k values to calculate thresholds for
            max_num_blocks: Maximum number of query blocks to consider (e.g., 32 for 4096 seq_len with q_block_size 128)
            num_layers: Number of layers in the model
            num_heads: Number of attention heads per layer
            q_block_size: Block size for query blocks
            k_block_size: Block size for key blocks
            aggregation_method: Method to aggregate thresholds across examples ('mean' or 'median')
        """
        self.k_values = sorted(list(k_values))
        self.max_num_blocks = max_num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.q_block_size = q_block_size
        self.k_block_size = k_block_size
        self.aggregation_method = aggregation_method

        # Store thresholds per k, per layer, per head, per position, across examples
        # Structure: {layer_idx: {head_idx: {k: {position: [thresholds across examples]}}}}
        self.threshold_data = {}

    def calculate_threshold_for_position(
        self,
        query_pos: int,
        block_max_values: torch.Tensor,
        k: int
    ) -> float:
        """
        Calculate the threshold for a single query position.

        Args:
            query_pos: Position of the query block
            block_max_values: Tensor of shape [num_k_blocks] with max values
                             for this query block against all key blocks
            k: Desired number of top non-diagonal blocks to keep

        Returns:
            Threshold value (or -inf if not enough blocks to filter)
        """
        num_k_blocks = len(block_max_values)

        # Calculate which K blocks are diagonal (overlap with this Q block's token range)
        # Q block at query_pos covers tokens [query_pos * q_block_size, (query_pos + 1) * q_block_size)
        q_start_token = query_pos * self.q_block_size
        q_end_token = (query_pos + 1) * self.q_block_size - 1

        # K blocks that overlap with this Q block
        k_diagonal_start = q_start_token // self.k_block_size
        k_diagonal_end = q_end_token // self.k_block_size
        diagonal_k_blocks = set(range(k_diagonal_start, k_diagonal_end + 1))

        # Valid K blocks: those at or before the diagonal (causal), excluding diagonal itself
        # The diagonal blocks are always kept, so we only select from non-diagonal blocks
        valid_key_positions = [i for i in range(k_diagonal_end + 1)
                              if i not in diagonal_k_blocks]

        # If we need to keep all available non-diagonal blocks (or more), return -inf
        if len(valid_key_positions) <= k:
            return float('-inf')

        # Extract the valid key block max values (exclude diagonal and future blocks)
        valid_values = block_max_values[valid_key_positions].cpu().numpy()

        # Sort in descending order to find top-k
        sorted_values = np.sort(valid_values)[::-1]  # Descending order

        # The threshold should be between the k-th and (k+1)-th largest values
        # such that "max(block) > threshold" selects exactly k blocks
        kth_value = sorted_values[k - 1]  # k-th largest (0-indexed)
        k_plus_1_value = sorted_values[k] if k < len(sorted_values) else float('-inf')

        # Threshold is the mean of the range (k_plus_1_value, kth_value]
        # Any value in this range will select exactly k blocks
        threshold = (k_plus_1_value + kth_value) / 2.0

        return threshold

    def process_example_statistics(
        self,
        qk_block_max: torch.Tensor,
        layer_idx: int
    ):
        """
        Process the block max statistics for one example and calculate thresholds.
        
        Args:
            qk_block_max: Tensor of shape [batch, num_heads, num_q_blocks, num_k_blocks] 
                         with max Q@K^T values for each query-key block pair
            layer_idx: Index of the layer
        """
        if layer_idx not in self.threshold_data:
            self.threshold_data[layer_idx] = {}
        
        batch_size, num_heads, num_q_blocks, num_k_blocks = qk_block_max.shape
        
        # Process each example in the batch and each head separately
        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                if head_idx not in self.threshold_data[layer_idx]:
                    self.threshold_data[layer_idx][head_idx] = {k: {} for k in self.k_values}
                
                # Get block max values for this head: [num_q_blocks, num_k_blocks]
                block_maxes = qk_block_max[batch_idx, head_idx, :, :]
                
                # Calculate thresholds for each k value
                for k in self.k_values:
                    # Calculate threshold for each query position
                    for query_pos in range(num_q_blocks):
                        # Get the max values for this query block against all key blocks
                        query_block_maxes = block_maxes[query_pos, :]  # [num_k_blocks]
                        
                        threshold = self.calculate_threshold_for_position(
                            query_pos, query_block_maxes, k
                        )
                        
                        # Store threshold for this position
                        if query_pos not in self.threshold_data[layer_idx][head_idx][k]:
                            self.threshold_data[layer_idx][head_idx][k][query_pos] = []
                        
                        self.threshold_data[layer_idx][head_idx][k][query_pos].append(threshold)

    def compute_final_thresholds(self) -> torch.Tensor:
        """
        Compute final thresholds by aggregating across all examples.

        Returns:
            Tensor of shape [num_k_values, num_layers, num_heads, max_num_blocks]
        """
        # Initialize the final tensor
        final_thresholds = torch.full(
            (len(self.k_values), self.num_layers, self.num_heads, self.max_num_blocks),
            float('-inf'),
            dtype=torch.float32
        )

        # Fill in the thresholds
        for layer_idx in self.threshold_data:
            for head_idx in self.threshold_data[layer_idx]:
                for k_idx, k in enumerate(self.k_values):
                    # Get all positions that were actually computed
                    positions = sorted(self.threshold_data[layer_idx][head_idx][k].keys())

                    if len(positions) == 0:
                        continue

                    # Calculate aggregated threshold for each position that was seen
                    for pos in positions:
                        thresholds_list = self.threshold_data[layer_idx][head_idx][k][pos]

                        # Apply aggregation method
                        if self.aggregation_method == 'mean':
                            aggregated_threshold = np.mean(thresholds_list)
                        elif self.aggregation_method == 'median':
                            aggregated_threshold = np.median(thresholds_list)
                        else:
                            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

                        final_thresholds[k_idx, layer_idx, head_idx, pos] = float(aggregated_threshold)

                    # If we didn't reach max_num_blocks, repeat the last threshold
                    max_computed_pos = max(positions)
                    if max_computed_pos < self.max_num_blocks - 1:
                        last_threshold = final_thresholds[k_idx, layer_idx, head_idx, max_computed_pos]
                        # Fill positions from max_computed_pos+1 to max_num_blocks-1
                        final_thresholds[k_idx, layer_idx, head_idx, max_computed_pos+1:] = last_threshold

        return final_thresholds
    
    def save_mean_thresholds(self, save_path: str):
        """
        Save only the compact mean-aggregated thresholds used for inference.

        Args:
            save_path: Base path for saving. The mean inference file will be created at:
                      - {base}.inference.pt
        """
        base_path, ext = os.path.splitext(save_path)
        if ext != '.pt':
            ext = '.pt'

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Computing and saving mean thresholds (inference only)...")
        print(f"{'='*80}")

        original_method = self.aggregation_method
        self.aggregation_method = 'mean'

        final_thresholds = self.compute_final_thresholds()

        inference_path = f"{base_path}.inference{ext}"
        inference_save_data = {
            'thresholds': final_thresholds,
            'k_values': self.k_values,
            'max_num_blocks': self.max_num_blocks,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'q_block_size': self.q_block_size,
            'k_block_size': self.k_block_size,
            'aggregation_method': 'mean',
            'shape_description': '[num_k_values, num_layers, num_heads, max_num_q_blocks]',
        }
        torch.save(inference_save_data, inference_path)

        print(f"\nMean aggregation:")
        print(f"  Saved to: {inference_path}")

        for k_idx, k in enumerate(self.k_values):
            k_thresholds = final_thresholds[k_idx]
            finite_mask = torch.isfinite(k_thresholds)
            if finite_mask.any():
                finite_values = k_thresholds[finite_mask]
                print(f"    k={k}: mean={finite_values.mean():.4f}, "
                      f"std={finite_values.std():.4f}, "
                      f"min={finite_values.min():.4f}, "
                      f"max={finite_values.max():.4f}")
            else:
                print(f"    k={k}: all -inf")

        self.aggregation_method = original_method

        print(f"\n{'='*80}")
        print(f"Threshold tensor shape: [num_k_values={len(self.k_values)}, "
              f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
              f"max_num_q_blocks={self.max_num_blocks}]")
        print(f"Q block size: {self.q_block_size}, K block size: {self.k_block_size}")
        print(f"K values: {self.k_values}")

        total_threshold_samples = 0
        for layer_idx in self.threshold_data:
            for head_idx in self.threshold_data[layer_idx]:
                for k in self.threshold_data[layer_idx][head_idx]:
                    for pos in self.threshold_data[layer_idx][head_idx][k]:
                        total_threshold_samples += len(self.threshold_data[layer_idx][head_idx][k][pos])

        print(f"Total threshold samples collected: {total_threshold_samples}")
        print(f"{'='*80}")

        return inference_path

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AttentionStatisticsCollector:
    """Collector that monkey-patches LlamaAttention modules to gather Q@K block statistics."""

    def __init__(self, model, device, q_block_size: int = 128,
                 k_block_size: int = 128,
                 threshold_calculator: Optional[ThresholdCalculator] = None,
                 chunk_size: Optional[int] = None):
        """
        Initialize the collector.

        Args:
            model: The language model (standard HuggingFace Llama model)
            device: Device to run on
            q_block_size: Block size for query blocks (default 128)
            k_block_size: Block size for key blocks (default 128)
            threshold_calculator: Optional ThresholdCalculator to compute thresholds
            chunk_size: Number of query blocks to process at once (default None = process all at once)
        """
        self.model = model
        self.device = device
        self.q_block_size = q_block_size
        self.k_block_size = k_block_size
        self.threshold_calculator = threshold_calculator
        self.chunk_size = chunk_size
        self.hooks = []
        self.statistics = {}

        # Monkey-patch LlamaAttention modules
        self._register_hooks()

    def _register_hooks(self):
        """Monkey-patch LlamaAttention forward methods to capture Q, K states."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Get the LlamaAttention module directly
            attention_module = layer.self_attn

            # Store the original forward method
            original_forward = attention_module.forward

            # Create a wrapper that captures Q, K states
            # IMPORTANT: Pass attention_module as parameter to capture it in closure
            def create_wrapped_forward(layer_idx, original_forward, attention_module):
                def wrapped_forward(
                    hidden_states,
                    position_embeddings=None,
                    attention_mask=None,
                    past_key_values=None,
                    cache_position=None,
                    **kwargs
                ):
                    # Call our hook to process Q, K before actual forward
                    self._process_qk_states(
                        attention_module,
                        hidden_states,
                        position_embeddings,
                        layer_idx
                    )

                    # Call the original forward
                    return original_forward(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs
                    )
                return wrapped_forward

            # Replace the forward method - pass attention_module to capture it
            attention_module.forward = create_wrapped_forward(layer_idx, original_forward, attention_module)

            # Store the original for cleanup
            self.hooks.append((attention_module, original_forward))

            # Initialize statistics storage for this layer
            self.statistics[layer_idx] = []
    
    def _reshape_to_blocks(self, tensor: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Reshape tensor from [batch, num_heads, seq_len, head_dim]
        to [batch, num_heads, num_blocks, block_size, head_dim].
        
        Pads with the last token if seq_len is not a multiple of block_size.
        (Matches old script behavior for Q=K=128 debugging)

        Args:
            tensor: Input tensor of shape [batch, num_heads, seq_len, head_dim]
            block_size: Size of each block

        Returns:
            Reshaped tensor of shape [batch, num_heads, num_blocks, block_size, head_dim]
        """
        batch, num_heads, seq_len, head_dim = tensor.shape

        # Calculate number of blocks (pad if necessary)
        num_blocks = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_blocks * block_size
        
        # Pad sequence if necessary (like old script)
        if padded_seq_len > seq_len:
            padding_length = padded_seq_len - seq_len
            
            # Get the last token: [batch, num_heads, 1, head_dim]
            last_token = tensor[:, :, -1:, :]
            
            # Repeat the last token to fill the padding
            # [batch, num_heads, padding_length, head_dim]
            padding = last_token.repeat(1, 1, padding_length, 1)
            
            # Concatenate original tensor with padding
            tensor = torch.cat([tensor, padding], dim=2)

        # Reshape to blocks
        # [batch, num_heads, num_blocks * block_size, head_dim]
        # -> [batch, num_heads, num_blocks, block_size, head_dim]
        tensor_blocked = tensor.view(batch, num_heads, num_blocks, block_size, head_dim)

        return tensor_blocked

    def _compute_qk_block_max(
        self,
        q_blocked: torch.Tensor,
        k_blocked: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """
        Compute the maximum value of scaled Q@K^T for each query block against each key block.

        Memory-efficient version that processes query blocks in chunks to avoid OOM.

        Args:
            q_blocked: Query tensor of shape [batch, num_heads, num_q_blocks, q_block_size, head_dim]
            k_blocked: Key tensor of shape [batch, num_heads, num_k_blocks, k_block_size, head_dim]
            scaling: Scaling factor for attention scores

        Returns:
            Max values of shape [batch, num_heads, num_q_blocks, num_k_blocks]
        """
        batch, num_heads, num_q_blocks, q_block_size, head_dim = q_blocked.shape
        num_k_blocks = k_blocked.shape[2]
        k_block_size = k_blocked.shape[3]

        # Initialize output tensor
        qk_block_max = torch.zeros(
            (batch, num_heads, num_q_blocks, num_k_blocks),
            dtype=q_blocked.dtype,
            device=q_blocked.device
        )

        # Process Q blocks in chunks to avoid OOM
        # If chunk_size is None, process all blocks at once (no chunking)
        chunk_size = self.chunk_size if self.chunk_size is not None else num_q_blocks
        
        # DEBUGGING NOTE: If this still produces different thresholds than the old script,
        # the bug is likely in one of:
        # 1. Unified padding logic (lines 511-536) - try independent padding per tensor
        # 2. Number of samples processed (check if both scripts see same examples)
        # 3. Floating point precision (dtype differences between old/new transformers versions)

        for q_start in range(0, num_q_blocks, chunk_size):
            q_end = min(q_start + chunk_size, num_q_blocks)
            q_chunk = q_blocked[:, :, q_start:q_end, :, :].contiguous()  # [batch, heads, chunk, q_block_size, head_dim]

            # Reshape for efficient computation (use .view() like old script for identical behavior)
            q_reshaped = q_chunk.view(batch * num_heads, q_end - q_start, q_block_size, head_dim)
            k_reshaped = k_blocked.view(batch * num_heads, num_k_blocks, k_block_size, head_dim)

            # Transpose key: [batch*num_heads, num_k_blocks, head_dim, k_block_size]
            k_transposed = k_reshaped.transpose(-2, -1)

            # Expand dimensions
            q_expanded = q_reshaped.unsqueeze(2)  # [batch*heads, chunk, 1, q_block_size, head_dim]
            k_expanded = k_transposed.unsqueeze(1)  # [batch*heads, 1, num_k_blocks, head_dim, k_block_size]

            # Compute Q@K^T for this chunk
            # Result: [batch*num_heads, chunk, num_k_blocks, q_block_size, k_block_size]
            qk_chunk = torch.matmul(q_expanded, k_expanded)

            # Apply scaling and take max
            qk_chunk = qk_chunk * scaling
            qk_chunk_max = qk_chunk.amax(dim=(3, 4))  # [batch*heads, chunk, num_k_blocks]

            # Reshape and store
            qk_chunk_max = qk_chunk_max.view(batch, num_heads, q_end - q_start, num_k_blocks)
            qk_block_max[:, :, q_start:q_end, :] = qk_chunk_max

            # Clean up intermediate tensors
            del q_chunk, q_reshaped, k_transposed, q_expanded, k_expanded, qk_chunk, qk_chunk_max

        return qk_block_max

    def _process_qk_states(self, module, hidden_states, position_embeddings, layer_idx):
        """
        Process Q, K states from a LlamaAttention module.

        Args:
            module: The LlamaAttention module
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            layer_idx: Layer index for statistics storage
        """
        with torch.no_grad():
            # Project to Q, K
            batch_size, seq_len, _ = hidden_states.shape

            # Get Q, K projections
            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)

            # Reshape to [batch, num_heads, seq_len, head_dim]
            query_states = query_states.view(batch_size, seq_len, -1, module.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, seq_len, -1, module.head_dim).transpose(1, 2)

            # Apply RoPE if position_embeddings are provided
            if position_embeddings is not None:
                cos, sin = position_embeddings
                # Apply rotary embeddings
                cos = cos.unsqueeze(1)  # Add head dimension
                sin = sin.unsqueeze(1)
                # Simplified RoPE application (rotate_half is defined globally)
                query_states = (query_states * cos) + (rotate_half(query_states) * sin)
                key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            # Get scaling from the attention module
            scaling = module.scaling
            num_key_value_groups = module.num_key_value_groups

            # Apply GQA key repetition
            if num_key_value_groups > 1:
                key_states = repeat_kv(key_states, num_key_value_groups)

            # Block-based analysis of Q@K^T
            # DEBUGGING: Use independent padding like old script (Q=128, K=128 case)
            # This unified padding is needed for Q!=K cases, but for Q=K=128 debugging,
            # let's match the old script exactly
            batch, num_heads, seq_len, head_dim = query_states.shape

            q_blocked = self._reshape_to_blocks(query_states, self.q_block_size)
            k_blocked = self._reshape_to_blocks(key_states, self.k_block_size)
            
            # Compute max of scaled Q@K^T per query block
            qk_block_max = self._compute_qk_block_max(
                q_blocked,
                k_blocked,
                scaling
            )  # [batch, num_heads, num_q_blocks, num_k_blocks]

            # Move to CPU immediately and free GPU memory
            qk_block_max_cpu = qk_block_max.cpu()

            # Delete large intermediate tensors to free GPU memory
            del query_states, key_states, q_blocked, k_blocked, qk_block_max
            torch.cuda.empty_cache()

            # If threshold calculator is provided, process this example
            if self.threshold_calculator is not None:
                # Pass CPU tensor to avoid keeping GPU tensors around
                self.threshold_calculator.process_example_statistics(
                    qk_block_max_cpu.to(self.device),  # Move back to GPU only for computation
                    layer_idx
                )

            # Collect statistics (keep minimal data)
            stats = {
                'seq_len': seq_len,
                'num_q_blocks': qk_block_max_cpu.shape[2],
                'num_k_blocks': qk_block_max_cpu.shape[3],
                'q_block_size': self.q_block_size,
                'k_block_size': self.k_block_size,
                'scaling': scaling,
                'num_key_value_groups': num_key_value_groups,
            }

            self.statistics[layer_idx].append(stats)

    def remove_hooks(self):
        """Restore original forward methods."""
        for attention_module, original_forward in self.hooks:
            attention_module.forward = original_forward
        self.hooks = []
    
    def get_statistics(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get collected statistics."""
        return self.statistics
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all layers and examples.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_layers': len(self.statistics),
            'q_block_size': self.q_block_size,
            'k_block_size': self.k_block_size,
            'per_layer_summary': {}
        }
        
        for layer_idx, layer_stats in self.statistics.items():
            if len(layer_stats) == 0:
                continue
            
            # Aggregate QK block max values across all examples for this layer
            all_qk_block_max = torch.stack([s['qk_block_max'] for s in layer_stats])
            
            summary['per_layer_summary'][layer_idx] = {
                'num_examples': len(layer_stats),
                'qk_block_max_mean': all_qk_block_max.mean().item(),
                'qk_block_max_std': all_qk_block_max.std().item(),
                'qk_block_max_max': all_qk_block_max.max().item(),
                'qk_block_max_min': all_qk_block_max.min().item(),
            }
        
        return summary


def load_model_and_tokenizer(model_name: str, device: str):
    """Load the model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 for efficiency
        device_map=device,
    )
    
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def load_dataset(data_file: str, num_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the dataset from JSON file.

    Args:
        data_file: Path to JSON file containing examples
        num_examples: If specified, takes the first num_examples from EACH task_name.
                     For example, if num_examples=2 and there are 8 tasks, it will
                     load 16 examples total (2 per task).

    Returns:
        List of examples
    """
    print(f"Loading dataset from: {data_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    if num_examples is not None:
        # Group examples by task_name
        from collections import defaultdict
        task_groups = defaultdict(list)
        for example in data:
            task_name = example.get('task_name', 'unknown')
            task_groups[task_name].append(example)

        # Take first num_examples from each task
        limited_data = []
        for task_name in sorted(task_groups.keys()):
            task_examples = task_groups[task_name][:num_examples]
            limited_data.extend(task_examples)
            print(f"  {task_name}: {len(task_examples)} examples (from {len(task_groups[task_name])} total)")

        data = limited_data
        print(f"Loaded {len(data)} examples total (limited to {num_examples} per task)")
    else:
        print(f"Loaded {len(data)} examples")

    return data


def process_examples(model, tokenizer, data, collector, device, max_seq_len, batch_size=1):
    """Process all examples through the model with prefill only.

    Args:
        model: The language model
        tokenizer: The tokenizer
        data: List of examples to process
        collector: AttentionStatisticsCollector instance
        device: Device to run on
        max_seq_len: Maximum sequence length for tokenization
        batch_size: Batch size for processing (default 1 for simplicity)
    """
    print(f"Processing {len(data)} examples...")

    with torch.no_grad():  # No gradient computation needed
        for idx, example in enumerate(tqdm(data, desc="Processing examples")):
            # breakpoint()
            input_text = example['input_text']

            # Tokenize the input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            
            # Forward pass (prefill only - just process the input)
            # We don't generate, just pass through the model
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                use_cache=False,  # Don't cache since we're not generating
            )
            
            # The hooks will be called automatically during forward pass
            
            # Optional: Print progress every N examples
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(data)} examples")


def main():
    parser = argparse.ArgumentParser(
        description="Measure attention thresholds for block-sparse attention with configurable Q and K block sizes"
    )

    # Model and data configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSON file containing calibration examples"
    )
    parser.add_argument(
        "--thresholds_save_path",
        type=str,
        default="./top_k_thresholds.pt",
        help="Path to save computed thresholds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, uses cuda if available"
    )

    # Block size configuration
    parser.add_argument(
        "--q_block_size",
        type=int,
        default=128,
        help="Block size for query blocks (default: 128)"
    )
    parser.add_argument(
        "--k_block_size",
        type=int,
        default=128,
        help="Block size for key blocks (default: 128)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096*4,
        help="Maximum sequence length to support (default: 16384)"
    )

    # Model architecture configuration
    parser.add_argument(
        "--num_layers",
        type=int,
        default=32,
        help="Number of layers in the model (default: 32 for Llama-3-8B)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of attention heads per layer (default: 32 for Llama-3-8B)"
    )

    # K values configuration
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,3,4,6,8,12,16,20,24,28",
        help="Comma-separated list of k values to compute thresholds for (default: 1,2,3,4,6,8,12,16,20,24,28)"
    )

    # Dataset configuration
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="Number of examples to process from EACH task in the dataset. If not specified, processes all examples. "
             "For example, --num_examples 2 with 8 tasks will process 16 examples total (2 per task)."
    )
    
    # Performance configuration
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of query blocks to process at once in Q@K computation. Lower values use less memory. If not specified, processes all blocks at once (default: None)"
    )

    args = parser.parse_args()

    # Parse k_values from comma-separated string
    k_values = set(int(k.strip()) for k in args.k_values.split(','))

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Calculate max_num_blocks based on Q block size
    max_num_blocks = args.max_seq_len // args.q_block_size

    print("=" * 80)
    print("Threshold Measurement Configuration")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Data file: {args.data_file}")
    print(f"Device: {device}")
    print(f"Q block size: {args.q_block_size}")
    print(f"K block size: {args.k_block_size}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Max num blocks (Q): {max_num_blocks}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Number of heads: {args.num_heads}")
    print(f"K values: {sorted(k_values)}")
    print(f"Aggregation method: mean (inference file only)")
    print(f"Chunk size: {args.chunk_size if args.chunk_size is not None else 'all blocks at once'}")
    print(f"Thresholds save path: {args.thresholds_save_path}")
    print("="*80)
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    # Load dataset
    data = load_dataset(args.data_file, args.num_examples)

    # Create threshold calculator (mean aggregation for inference)
    threshold_calc = ThresholdCalculator(
        k_values=k_values,
        max_num_blocks=max_num_blocks,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        q_block_size=args.q_block_size,
        k_block_size=args.k_block_size,
        aggregation_method='mean'
    )

    # Create collector and register hooks
    collector = AttentionStatisticsCollector(
        model,
        device,
        q_block_size=args.q_block_size,
        k_block_size=args.k_block_size,
        threshold_calculator=threshold_calc,
        chunk_size=args.chunk_size
    )

    try:
        # Process all examples
        process_examples(model, tokenizer, data, collector, device, args.max_seq_len)

        # Calculate and save thresholds for inference
        inference_file = threshold_calc.save_mean_thresholds(args.thresholds_save_path)

        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)

        print(f"\nOutput files created:")
        print(f"  Mean inference:        {inference_file}")

        # Show file sizes
        import os as os_module
        inference_size = os_module.path.getsize(inference_file) / (1024**2)  # MB

        print(f"\nFile size:")
        print(f"  Mean inference:  {inference_size:.2f} MB")
        print("="*80)

    finally:
        # Clean up hooks
        collector.remove_hooks()
        print("\nHooks removed successfully")


if __name__ == "__main__":
    main()