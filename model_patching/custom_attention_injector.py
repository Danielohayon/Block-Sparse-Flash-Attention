"""
Custom Attention Injector for Transformers Models
==================================================

This module provides a way to inject custom attention implementations into
HuggingFace transformer models without modifying the transformers library.

It works by monkey-patching the forward methods of attention modules at runtime.

Usage Example:
--------------

    from transformers import AutoModelForCausalLM
    from custom_attention_injector import (
        AttentionConfig,
        inject_custom_attention,
    )

    # Load model normally
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-8b-hf")

    # Configure custom attention (block sparse example)
    config = AttentionConfig(
        mode="block_sparse",
        threshold_file="/path/to/thresholds.pt",
        block_sparse_topk=16,
    )

    # Or use sliding window
    config = AttentionConfig(
        mode="sliding_window",
        sliding_window_size=1024,
    )

    # Inject custom attention
    inject_custom_attention(model, config)

    # Use model normally - it will now use custom attention
    outputs = model.generate(...)

Supported Models:
-----------------
- Llama (and variants like Llama-2, Llama-3, etc.)
- Easy to extend to other models by adding new injection strategies

Environment Variables:
----------------------
You can also configure via environment variables:

    # Block sparse example
    export CUSTOM_ATTENTION_MODE=block_sparse
    export CUSTOM_THRESHOLD_FILE=/path/to/thresholds.pt
    export CUSTOM_BLOCK_SPARSE_TOPK=16

    # Sliding window example
    export CUSTOM_ATTENTION_MODE=sliding_window
    export CUSTOM_SLIDING_WINDOW_SIZE=1024

Then simply:

    from custom_attention_injector import inject_custom_attention
    inject_custom_attention(model)  # Reads from environment
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for custom attention kernels.

    Args:
        mode: One of "sdpa", "block_sparse", "sparge_attn", or "sliding_window"
            - "sdpa": Standard PyTorch scaled_dot_product_attention
            - "block_sparse": Block-sparse attention using flash_attn_func with thresholds
            - "sparge_attn": SpargeAttn attention kernel
            - "sliding_window": Causal sliding window attention
        threshold_file: Path to threshold file (required for block_sparse mode)
        block_sparse_topk: Top-k value for block sparse attention (required for block_sparse mode)
        sparge_attn_params: Dictionary of parameters for sparge_attn attention
        sliding_window_size: Size of the sliding window (required for sliding_window mode)
    """
    mode: str = "sdpa"
    threshold_file: Optional[str] = None
    block_sparse_topk: Optional[int] = None
    sparge_attn_params: Optional[dict] = None
    sliding_window_size: Optional[int] = None

    def __post_init__(self):
        valid_modes = {"sdpa", "block_sparse", "sparge_attn", "sliding_window"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")

        if self.mode == "block_sparse":
            if self.threshold_file is None:
                raise ValueError("threshold_file must be specified for block_sparse mode")
            if self.block_sparse_topk is None:
                raise ValueError("block_sparse_topk must be specified for block_sparse mode")

        if self.mode == "sparge_attn" and self.sparge_attn_params is None:
            # Set default parameters
            self.sparge_attn_params = {
                "simthreshd1": -0.1,
                "topk": 0.35,
                "pvthreshd": 15,
            }

        if self.mode == "sliding_window":
            if self.sliding_window_size is None:
                raise ValueError("sliding_window_size must be specified for sliding_window mode")


class CustomAttentionKernel(nn.Module):
    """Unified attention kernel supporting multiple backends."""

    def __init__(self, config: AttentionConfig, layer_idx: int, num_heads: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        # Load thresholds for block sparse attention
        if config.mode == "block_sparse":
            self._load_thresholds()
        else:
            self.thresholds_by_position = None

        # Import kernels lazily only when needed
        self.block_sparse_flash_attn_func = None
        self.sparge_attn = None

        if config.mode == "block_sparse":
            try:
                from block_sparse_flash_attn import block_sparse_flash_attn_func
                self.block_sparse_flash_attn_func = block_sparse_flash_attn_func
            except ImportError:
                raise ImportError(
                    "block_sparse_flash_attn is required for block_sparse mode. To install it follow the instructions in the kernel_src folder"
                )

        elif config.mode == "sparge_attn":
            try:
                from spas_sage_attn import spas_sage_attn_meansim_topk_cuda
                self.sparge_attn = spas_sage_attn_meansim_topk_cuda
            except ImportError:
                raise ImportError("sparge_attn is required for sparge_attn mode, clone and install https://github.com/thu-ml/SpargeAttn")

    def _load_thresholds(self):
        """Load thresholds for block sparse attention."""
        threshold_file = Path(self.config.threshold_file)

        if not threshold_file.exists():
            raise FileNotFoundError(
                f"Threshold file not found: {threshold_file}. "
                f"Please ensure the file exists at the specified path."
            )

        threshold_data = torch.load(threshold_file, map_location='cpu', weights_only=False)
        all_thresholds = threshold_data['thresholds']
        k_values = threshold_data['k_values']

        if self.config.block_sparse_topk not in k_values:
            raise ValueError(
                f"k={self.config.block_sparse_topk} not found in threshold file. "
                f"Available k values: {k_values}"
            )

        k_idx = k_values.index(self.config.block_sparse_topk)
        layer_thresholds = all_thresholds[k_idx, self.layer_idx, :, :].contiguous()

        # Register as buffer so it moves with the model
        self.register_buffer('thresholds_by_position', layer_thresholds, persistent=False)

        logger.info(
            f"Loaded block-sparse thresholds for layer {self.layer_idx}, "
            f"k={self.config.block_sparse_topk}: shape {layer_thresholds.shape}"
        )

    def _efficient_sliding_window(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        window_size: int,
        scaling: float,
    ) -> torch.Tensor:
        """Efficient sliding window attention.

        Uses flash_attn's native sliding window support for efficient computation.

        Args:
            query: [B, H, N, D]
            key: [B, H, N, D]
            value: [B, H, N, D]
            window_size: Size of the sliding window
            scaling: Attention scaling factor

        Returns:
            attn_output: [B, H, N, D]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Try to use flash_attn directly with sliding window support
        from flash_attn import flash_attn_func

        # flash_attn expects [B, N, H, D], so transpose
        query_t = query.transpose(1, 2).contiguous()  # [B, N, H, D]
        key_t = key.transpose(1, 2).contiguous()
        value_t = value.transpose(1, 2).contiguous()

        # Use flash_attn with sliding window
        # window_size=(left, right) where query i attends to keys in [i-left, i+right]
        # For causal sliding window: left=window_size-1, right=0
        attn_output = flash_attn_func(
            query_t,
            key_t,
            value_t,
            dropout_p=0.0,
            softmax_scale=scaling,
            causal=True,
            window_size=(window_size - 1, 0),  # (left, right)
        )

        # Transpose back to [B, H, N, D]
        attn_output = attn_output.transpose(1, 2)
        return attn_output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scaling: float,
    ) -> torch.Tensor:
        """
        Forward pass for attention.

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            is_causal: Whether to apply causal masking
            scaling: Attention scaling factor

        Returns:
            attn_output: [batch, seq_len, num_heads, head_dim]
        """

        if self.config.mode == "sdpa" or query.shape[2] <= 128:
            # PyTorch SDPA: expects [B, H, N, D], returns [B, H, N, D]
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                scale=scaling,
                is_causal=is_causal,
            )
            # Transpose to [B, N, H, D]
            attn_output = attn_output.transpose(1, 2).contiguous()

        elif self.config.mode == "block_sparse":
            # Block sparse: expects [B, N, H, D], returns [B, N, H, D]
            # Transpose to [B, N, H, D]
            query_t = query.transpose(1, 2).contiguous()
            key_t = key.transpose(1, 2).contiguous()
            value_t = value.transpose(1, 2).contiguous()

            # Padding is now handled inside block_sparse_flash_attn_func
            attn_output = self.block_sparse_flash_attn_func(
                query_t,
                key_t,
                value_t,
                self.thresholds_by_position,  # 4th positional argument
                dropout_p=0.0,
                softmax_scale=scaling,
                causal=is_causal,
            )

        elif self.config.mode == "sparge_attn":
            # SpargeAttn: expects [B, H, N, D], returns [B, H, N, D]

            attn_output = self.sparge_attn(
                q=query,
                k=key,
                v=value,
                is_causal=is_causal,
                scale=scaling,
                tensor_layout="HND",
                output_dtype=query.dtype,
                return_sparsity=False,
                **self.config.sparge_attn_params,
            )

            # Transpose to [B, N, H, D]
            attn_output = attn_output.transpose(1, 2).contiguous()

        elif self.config.mode == "sliding_window":
            # Sliding window: expects [B, H, N, D], returns [B, H, N, D]
            batch_size, num_heads, seq_len, head_dim = query.shape
            window_size = self.config.sliding_window_size

            # Validate config
            assert window_size is not None, "sliding_window_size must be set for sliding_window mode"

            if seq_len <= window_size:
                # Fast path: if sequence fits in window, use causal attention
                attn_output = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=None,
                    scale=scaling,
                    is_causal=True
                )
            else:
                # Efficient sliding window: use flash_attn's native support
                attn_output = self._efficient_sliding_window(
                    query, key, value, window_size, scaling
                )

            # Transpose to [B, N, H, D]
            attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output


def create_attention_config_from_env() -> Optional[AttentionConfig]:
    """Create AttentionConfig from environment variables.

    Environment variables:
        CUSTOM_ATTENTION_MODE: "sdpa", "block_sparse", "sparge_attn", or "sliding_window"
        CUSTOM_THRESHOLD_FILE: Path to threshold file (for block_sparse)
        CUSTOM_BLOCK_SPARSE_TOPK: Top-k value (for block_sparse)
        CUSTOM_SPARGE_SIMTHRESHD1: simthreshd1 parameter (for sparge_attn, default: -0.1)
        CUSTOM_SPARGE_TOPK: topk parameter (for sparge_attn, default: 0.35)
        CUSTOM_SPARGE_PVTHRESHD: pvthreshd parameter (for sparge_attn, default: 15)
        CUSTOM_SLIDING_WINDOW_SIZE: Window size (for sliding_window)

    Returns:
        AttentionConfig if CUSTOM_ATTENTION_MODE is set, None otherwise
    """
    mode = os.getenv("CUSTOM_ATTENTION_MODE")
    if mode is None:
        return None

    if mode == "block_sparse":
        threshold_file = os.getenv("CUSTOM_THRESHOLD_FILE")
        topk_str = os.getenv("CUSTOM_BLOCK_SPARSE_TOPK")
        topk = int(topk_str) if topk_str else None

        return AttentionConfig(
            mode=mode,
            threshold_file=threshold_file,
            block_sparse_topk=topk,
        )

    elif mode == "sparge_attn":
        sparge_params = {
            "simthreshd1": float(os.getenv("CUSTOM_SPARGE_SIMTHRESHD1", "-0.1")),
            "topk": float(os.getenv("CUSTOM_SPARGE_TOPK", "0.35")),
            "pvthreshd": int(os.getenv("CUSTOM_SPARGE_PVTHRESHD", "15")),
        }

        return AttentionConfig(
            mode=mode,
            sparge_attn_params=sparge_params,
        )

    elif mode == "sliding_window":
        window_size_str = os.getenv("CUSTOM_SLIDING_WINDOW_SIZE")
        window_size = int(window_size_str) if window_size_str else None

        return AttentionConfig(
            mode=mode,
            sliding_window_size=window_size,
        )

    else:
        return AttentionConfig(mode=mode)


# ============================================================================
# Model-specific injection strategies
# ============================================================================

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value states for grouped query attention.
    Goes from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_llama_attention_forward(
    original_forward: Callable,
    custom_attention: CustomAttentionKernel,
    num_key_value_groups: int,
    is_causal: bool,
    scaling: float,
) -> Callable:
    """Create a patched forward function for Llama attention modules."""

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Get Q, K, V from the original projections
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings

        # Import apply_rotary_pos_emb from the model's module
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Expand key/value for GQA
        key_states = _repeat_kv(key_states, num_key_value_groups)
        value_states = _repeat_kv(value_states, num_key_value_groups)

        # Determine if causal
        is_causal_mask = query_states.shape[2] > 1 and attention_mask is None and is_causal

        # Call custom attention kernel
        # Input: [batch, num_heads, seq_len, head_dim]
        # Output: [batch, seq_len, num_heads, head_dim]
        attn_output = custom_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            is_causal=is_causal_mask,
            scaling=scaling,
        )

        # Reshape and apply output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None  # Return None for attention weights

    return patched_forward


def inject_llama_attention(
    model: nn.Module,
    config: AttentionConfig,
) -> None:
    """Inject custom attention into a Llama model.

    Args:
        model: The Llama model to modify
        config: Attention configuration
    """
    logger.info(f"Injecting custom attention (mode={config.mode}) into Llama model")

    # Find all attention modules
    layer_idx = 0
    for name, module in model.named_modules():
        # Check if this is an attention module
        if module.__class__.__name__ == "LlamaAttention":
            logger.info(f"Patching attention layer {layer_idx}: {name}")

            # Create custom attention kernel for this layer
            custom_attention = CustomAttentionKernel(
                config=config,
                layer_idx=layer_idx,
                num_heads=module.config.num_attention_heads,
            )

            # Move to same device as the model
            custom_attention = custom_attention.to(next(module.parameters()).device)

            # Store the custom attention as an attribute
            module._custom_attention = custom_attention

            # Get attention parameters
            num_key_value_groups = (
                module.config.num_attention_heads // module.config.num_key_value_heads
            )
            scaling = module.head_dim ** -0.5
            is_causal = getattr(module, 'is_causal', True)

            # Create patched forward function
            patched_forward = create_llama_attention_forward(
                original_forward=module.forward,
                custom_attention=custom_attention,
                num_key_value_groups=num_key_value_groups,
                is_causal=is_causal,
                scaling=scaling,
            )

            # Replace the forward method
            module.forward = types.MethodType(patched_forward, module)

            layer_idx += 1

    logger.info(f"Successfully patched {layer_idx} attention layers")


# ============================================================================
# Main injection function
# ============================================================================

def inject_custom_attention(
    model: nn.Module,
    config: Optional[AttentionConfig] = None,
    model_type: Optional[str] = None,
) -> None:
    """Inject custom attention into a transformer model.

    Args:
        model: The model to modify
        config: Attention configuration. If None, will try to load from environment
        model_type: Model type ("llama", etc.). If None, will auto-detect

    Raises:
        ValueError: If model type cannot be detected or is not supported

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-8b-hf")
        >>>
        >>> config = AttentionConfig(
        ...     mode="block_sparse",
        ...     threshold_file="/path/to/thresholds.pt",
        ...     block_sparse_topk=16,
        ... )
        >>> inject_custom_attention(model, config)
    """
    # Try to load config from environment if not provided
    if config is None:
        config = create_attention_config_from_env()
        if config is None:
            raise ValueError(
                "No attention config provided and CUSTOM_ATTENTION_MODE environment "
                "variable not set. Please provide a config or set environment variables."
            )

    # Auto-detect model type if not provided
    if model_type is None:
        model_class_name = model.__class__.__name__.lower()
        if "llama" in model_class_name:
            model_type = "llama"
        else:
            raise ValueError(
                f"Could not auto-detect model type from class name '{model.__class__.__name__}'. "
                f"Please specify model_type explicitly. Supported types: llama"
            )

    # Inject based on model type
    if model_type == "llama":
        inject_llama_attention(model, config)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: llama"
        )

    logger.info(
        f"Custom attention injection complete for {model_type} model "
        f"with mode={config.mode}"
    )


# ============================================================================
# Runtime reconfiguration functions
# ============================================================================

def change_attention_config(
    model: nn.Module,
    new_config: AttentionConfig,
    model_type: Optional[str] = None,
) -> None:
    """Change the attention configuration of an already-injected model.

    This allows you to switch attention modes or change parameters without
    reloading the model. More efficient than re-injecting from scratch.

    Args:
        model: Model with already-injected custom attention
        new_config: New attention configuration
        model_type: Model type (will auto-detect if not provided)

    Example:
        >>> # Start with block-sparse, k=16
        >>> config1 = AttentionConfig(mode="block_sparse", threshold_file="thresh.pt", block_sparse_topk=16)
        >>> inject_custom_attention(model, config1)
        >>>
        >>> # Switch to k=32 without reloading model
        >>> config2 = AttentionConfig(mode="block_sparse", threshold_file="thresh.pt", block_sparse_topk=32)
        >>> change_attention_config(model, config2)
        >>>
        >>> # Switch to SpargeAttn
        >>> config3 = AttentionConfig(mode="sparge_attn")
        >>> change_attention_config(model, config3)
    """
    # Auto-detect model type if not provided
    if model_type is None:
        model_class_name = model.__class__.__name__.lower()
        if "llama" in model_class_name:
            model_type = "llama"
        else:
            raise ValueError(
                f"Could not auto-detect model type from class name '{model.__class__.__name__}'. "
                f"Please specify model_type explicitly."
            )

    logger.info(f"Changing attention config to mode={new_config.mode}")

    # Find and update all custom attention kernels
    layer_idx = 0
    for name, module in model.named_modules():
        if hasattr(module, '_custom_attention'):
            # Create new custom attention kernel
            new_custom_attention = CustomAttentionKernel(
                config=new_config,
                layer_idx=layer_idx,
                num_heads=module._custom_attention.num_heads,
            )

            # Move to same device as the old one
            device = next(module._custom_attention.parameters()).device if list(module._custom_attention.parameters()) else next(module.parameters()).device
            new_custom_attention = new_custom_attention.to(device)

            # Replace the custom attention kernel
            module._custom_attention = new_custom_attention

            # For Llama, we need to recreate the forward function with new parameters
            if model_type == "llama":
                num_key_value_groups = (
                    module.config.num_attention_heads // module.config.num_key_value_heads
                )
                scaling = module.head_dim ** -0.5
                is_causal = getattr(module, 'is_causal', True)

                patched_forward = create_llama_attention_forward(
                    original_forward=None,  # Not used in our implementation
                    custom_attention=new_custom_attention,
                    num_key_value_groups=num_key_value_groups,
                    is_causal=is_causal,
                    scaling=scaling,
                )

                module.forward = types.MethodType(patched_forward, module)

            layer_idx += 1

    logger.info(f"Successfully updated {layer_idx} attention layers")


def update_block_sparse_topk(
    model: nn.Module,
    new_topk: int,
    threshold_file: Optional[str] = None,
) -> None:
    """Update the top-k value for block-sparse attention without full re-injection.

    This is a convenience function for the common case of wanting to test
    different sparsity levels.

    Args:
        model: Model with block-sparse attention already injected
        new_topk: New top-k value
        threshold_file: Path to threshold file (uses existing if not provided)

    Example:
        >>> # Start with k=16
        >>> config = AttentionConfig(mode="block_sparse", threshold_file="thresh.pt", block_sparse_topk=16)
        >>> inject_custom_attention(model, config)
        >>>
        >>> # Test k=32
        >>> update_block_sparse_topk(model, new_topk=32)
        >>>
        >>> # Test k=8
        >>> update_block_sparse_topk(model, new_topk=8)
    """
    # Get threshold file from existing config if not provided
    if threshold_file is None:
        # Find first custom attention module to get threshold file
        for module in model.modules():
            if hasattr(module, '_custom_attention'):
                old_config = module._custom_attention.config
                if old_config.mode != "block_sparse":
                    raise ValueError(
                        f"Model is using {old_config.mode} attention, not block_sparse. "
                        f"Use change_attention_config() instead."
                    )
                threshold_file = old_config.threshold_file
                break

        if threshold_file is None:
            raise ValueError(
                "Could not find threshold file from existing config. "
                "Please provide threshold_file parameter."
            )

    logger.info(f"Updating block-sparse top-k from previous value to {new_topk}")

    # Create new config with updated top-k
    new_config = AttentionConfig(
        mode="block_sparse",
        threshold_file=threshold_file,
        block_sparse_topk=new_topk,
    )

    # Update the attention config
    change_attention_config(model, new_config)


def update_sparge_attn_params(
    model: nn.Module,
    **new_params,
) -> None:
    """Update SpargeAttn attention parameters without full re-injection.

    Args:
        model: Model with SpargeAttn attention already injected
        **new_params: Parameters to update (e.g., topk=0.5, simthreshd1=-0.2)

    Example:
        >>> # Start with default params
        >>> config = AttentionConfig(mode="sparge_attn")
        >>> inject_custom_attention(model, config)
        >>>
        >>> # Test different topk
        >>> update_sparge_attn_params(model, topk=0.5)
        >>>
        >>> # Test different threshold
        >>> update_sparge_attn_params(model, simthreshd1=-0.2, topk=0.3)
    """
    # Get existing params
    existing_params = None
    for module in model.modules():
        if hasattr(module, '_custom_attention'):
            old_config = module._custom_attention.config
            if old_config.mode != "sparge_attn":
                raise ValueError(
                    f"Model is using {old_config.mode} attention, not sparge_attn. "
                    f"Use change_attention_config() instead."
                )
            existing_params = old_config.sparge_attn_params.copy()
            break

    if existing_params is None:
        raise ValueError("Could not find existing SpargeAttn configuration")

    # Update with new params
    existing_params.update(new_params)

    logger.info(f"Updating SpargeAttn params: {new_params}")

    # Create new config with updated params
    new_config = AttentionConfig(
        mode="sparge_attn",
        sparge_attn_params=existing_params,
    )

    # Update the attention config
    change_attention_config(model, new_config)


def get_current_attention_config(model: nn.Module) -> Optional[AttentionConfig]:
    """Get the current attention configuration from a model.

    Args:
        model: Model to inspect

    Returns:
        AttentionConfig if custom attention is injected, None otherwise

    Example:
        >>> config = get_current_attention_config(model)
        >>> if config:
        ...     print(f"Current mode: {config.mode}")
        ...     if config.mode == "block_sparse":
        ...         print(f"Top-k: {config.block_sparse_topk}")
    """
    for module in model.modules():
        if hasattr(module, '_custom_attention'):
            return module._custom_attention.config

    return None


# ============================================================================
# Convenience function to restore original attention
# ============================================================================

def restore_original_attention(model: nn.Module) -> None:
    """Restore original attention implementation (not yet implemented).

    This would require storing the original forward methods before patching.
    For now, just reload the model if you need to restore original behavior.
    """
    raise NotImplementedError(
        "Restoring original attention is not yet implemented. "
        "Please reload the model from scratch if needed."
    )
