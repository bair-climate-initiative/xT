import math

import torch
import torch.hub
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import nn
from torch.nn import Dropout2d

from ..context_encoders import ContextEncoderConfig
from ..context_encoders.attention import LLMAttention, ViTAttention
from .utils import get_2d_sincos_pos_embed, LlamaRMSNorm

default_decoder_filters = [48, 96, 176, 256]
default_last = 48


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ClassificationDecoder(nn.Module):
    def __init__(self, in_dim, num_classes, mlp_ratio=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim * mlp_ratio),
            nn.GELU(),
            nn.LayerNorm(in_dim * mlp_ratio),
            nn.Linear(in_dim * mlp_ratio, num_classes),
        )

    def forward(self, enc_results, mem):
        xx = enc_results[-1]  # N C H W
        xx = xx.mean((-1, -2))  # GAP->N X C
        logits = self.layers(xx)  # CLS
        return {"label": logits}


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.pretraining_tp = 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LLMLayer(nn.Module):
    def __init__(
        self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"
    ):
        super().__init__()
        num_heads = dim // 128
        if attention_method == "hyper":
            self.attn = LLMAttention(dim, dim, num_heads, causal=causal)
        else:
            self.attn = ViTAttention(dim, dim, num_heads, causal=causal)
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.mlp = LlamaMLP(dim, inner_dim)
        self.causal = causal

    def forward(self, hidden_states, residual_in=-1):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if residual_in != -1:
            return hidden_states, 0.0
        return hidden_states


class LLMClassificationDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        mlp_ratio=4,
        use_checkpoint=False,
        hidden_size=768,
        num_heads=8,
        n_layers=2,
        attention_method=None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.input_proj = nn.Linear(in_dim, hidden_size)
        assert attention_method in ["hyper", "naive", "mamba"]
        if attention_method == "mamba":
            from ..context_encoders.mamba import create_block

            ssm_cfg = {"d_state": 16}
            self.layers = nn.Sequential(
                *[
                    create_block(
                        d_model=hidden_size,
                        ssm_cfg=ssm_cfg,
                        fused_add_norm=False,
                        residual_in_fp32=True,
                        drop_rate=0.0,
                        drop_path_rate=0.0,
                        reverse=i % 2 == 0,
                        transpose=False,
                        use_mlp=False,
                        is_2d=False,
                        rms_norm=False,
                        split_head=False,
                        use_nd=False,
                        downsample=False,
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LLMLayer(
                        hidden_size,
                        hidden_size * mlp_ratio,
                        num_heads,
                        causal=True,
                        attention_method=attention_method,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.hidden_size = hidden_size
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, num_classes)
        )
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x, mem):
        x = x[-1]
        n, _, h, w = x.shape
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, h, cls_token=False)
        x = rearrange(x, "n c h w -> n (h w )c")

        # breakpoint()
        x = self.input_proj(x)
        x = x + torch.tensor(pos_embed).to(x)
        x = torch.cat([x, self.cls_token.repeat(n, 1, 1)], dim=1)
        residual = None
        if self.use_checkpoint:
            for i, blk in enumerate(self.layers):
                x, residual = checkpoint.checkpoint(blk, x, residual)
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
        else:
            for i, blk in enumerate(self.layers):
                x, residual = blk(x, residual)
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
        x = self.cls(x[:, -1])  # N C
        return {"label": x}


class EncoderDecoder(AbstractModel):
    def __init__(
        self,
        backbone: nn.Module,
        xl_config: ContextEncoderConfig,
        channels_last: bool = False,
        crop_size: int = 256,
        skip_decoder: bool = False,
        backbone_name: str = "swinv2_tiny_window16_256_timm",
        dataset: str = "inaturalist",
        num_classes: int = 9999,
        mlp_ratio: int = 4,
        cls_head: str = None,
        **kwargs,
    ):
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.skip_decoder = skip_decoder
        self.dataset = dataset
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.cls_head = cls_head
        self.grad_ratio = xl_config.grad_ratio
        self.xl_config = xl_config

        super().__init__()

        self.init_decoder()

        self.name = "u-{}".format(backbone_name)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x, mem=tuple(), cord=(0, 0), **kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        n_chip = x.shape[2] // self.crop_size
        # print(n_chip)
        if n_chip > 1:  # on gradient chipping
            x = rearrange(
                x,
                "N C (HP HC) (WP WC)-> (N HP WP) C HC WC ",
                HP=n_chip,
                WP=n_chip,
                HC=self.crop_size,
                WC=self.crop_size,
            )
            if self.grad_ratio >= 1.0:
                enc_results = self.encoder(x)
            else:
                n = x.shape[0]
                n_grad = math.ceil(n * self.grad_ratio)
                idx = torch.randperm(n)
                idx_inv = torch.argsort(idx)
                out_grad = self.encoder(x[:n_grad])
                with torch.no_grad():
                    out_stopgrad = self.encoder(x[n_grad:])
                enc_results = list(
                    [
                        torch.cat([a, b], dim=0)[idx_inv]
                        for a, b in zip(out_grad, out_stopgrad)
                    ]
                )
                del out_grad
                del out_stopgrad
            enc_results = list(
                [
                    rearrange(
                        i,
                        "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                        HP=n_chip,
                        WP=n_chip,
                    )
                    for i in enc_results
                ]
            )
        else:
            enc_results = self.encoder(x)

        output = self.decoder(enc_results, x_skip)
        return output

    def init_decoder(self):
        if self.dataset == "inaturalist":
            assert self.cls_head in ["naive", "xl"]

            clasifier = (
                ClassificationDecoder
                if self.cls_head == "naive"
                else LLMClassificationDecoder
            )
            extra_kwargs = {}
            if self.cls_head == "xl":
                extra_kwargs = dict(
                    hidden_size=self.xl_config.hidden_size,
                    n_layers=self.xl_config.n_layer,
                    attention_method=self.xl_config.attention_method,
                )
            self.decoder = clasifier(
                in_dim=self.filters[-1],
                num_classes=self.num_classes,
                mlp_ratio=self.mlp_ratio,
                **extra_kwargs,
            )
        else:
            raise Exception("Unknown dataset {}".format(self.dataset))
