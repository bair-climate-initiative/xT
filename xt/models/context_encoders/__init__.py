from dataclasses import dataclass


@dataclass
class ContextEncoderConfig:
    enabled: bool = False
    """If True, use Transformer-XL as context mode."""
    no_memory: bool = False
    """If True, use Transformer-XL without memory."""
    n_layer: int = 12
    hidden_size: int = 768
    mem_chip: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    """Number of layers."""
    tiling: str = "naive"
    """Tiling strategy for XL."""
    context_patch_len: int = 100
    """Context Patch Length."""
    skip_connection: bool = False
    """Whether to add a skip connection passing XL layers"""
    classification_mode: bool = False
    """Ratio of chips have gradient traacked, must be between 0 and 1"""
    grad_ratio: float = 1.0
    """Ratio of sequence to maintain gradients for better memory usage."""
    attention_method: str = "hyper"
    """Number of in-context patches. Default -1 is infinity."""
    in_context_patches: int = -1