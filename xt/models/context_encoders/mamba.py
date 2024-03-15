from functools import partial
from typing import Optional

import torch
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.layers.drop import DropPath

from torch import Tensor, nn
from torch.nn import Dropout


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        residual_in_fp32=False,
        reverse=False,
        transpose=False,
        split_head=False,
        drop_path_rate=0.0,
        drop_rate=0.0,
        use_mlp=False,
        downsample=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.split_head = split_head
        self.reverse = reverse
        self.transpose = transpose
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.dropout = Dropout(p=drop_rate)


    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        **kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states + self.drop_path(
            self.mixer(hidden_states, inference_params=inference_params, **kwargs)
        )

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        return hidden_states, None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None,
    reverse=None,
    is_2d=False,
    drop_rate=0.1,
    drop_path_rate=0.1,
    use_mlp=False,
    transpose=False,
    split_head=False,
    use_nd=False,
    downsample=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
        reverse=reverse,
        transpose=transpose,
        drop_rate=drop_rate,
        use_mlp=use_mlp,
        drop_path_rate=drop_path_rate,
        split_head=split_head,
        downsample=downsample,
    )
    block.layer_idx = layer_idx
    return block


if __name__ == "__main__":
    ssm_cfg = {"d_state": 16}
    blk = create_block(
        d_model=768,
        ssm_cfg=ssm_cfg,
        residual_in_fp32=True,
        drop_rate=0.1,
        drop_path_rate=0.1,
        reverse=False,
        transpose=False,
        use_mlp=False,
        is_2d=False,
        rms_norm=False,
        split_head=False,
        use_nd=False,
        downsample=False,
    ).cuda()
    x = torch.rand(4, 322, 768).cuda()
    y, _ = blk(x)
    assert x.shape == y.shape
