import math

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from torch.nn import functional as F

# import xformers.ops as xops
from .atten_mask import build_attention_mask_adjacent
from .norm_4d import LayerNorm4d
from .window_attention import WindowAttention4D

# from xformers.components.attention import ScaledDotProduct


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attention_mode="default",
        temperature=1.0,
        rel_pos_bias=False,
        input_resolution=None,
        radius=1,
        grid=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attention_mode = attention_mode
        self.rel_pos_bias = rel_pos_bias
        self.input_resolution = input_resolution
        self.grid = grid
        if self.attention_mode == "cosine":
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        elif self.attention_mode == "xformer":
            self.radius = radius
            atten_bias = build_attention_mask_adjacent(
                self.input_resolution, radius=self.radius, grid=self.grid
            )
            mask = atten_bias == 0
            edges = torch.where(mask)
            self.register_buffer("edges", torch.stack(edges), persistent=False)
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((1, 1, num_heads)))
            )
            # atten_bias = build_attention_mask_adjacent(input_resolution,radius=1)
            # self.register_buffer("atten_bias", atten_bias, persistent=False)
        if self.rel_pos_bias:
            if self.rel_pos_bias == "abs":
                cbp_in_dim = 2 * len(self.input_resolution)
            elif self.rel_pos_bias == "rel":
                cbp_in_dim = len(self.input_resolution)
            else:
                raise NotImplemented
            self.cpb_mlp = nn.Sequential(
                nn.Linear(cbp_in_dim, 512, bias=True),
                nn.GELU(),
                nn.Linear(512, num_heads, bias=False),
            )
            abs_coords = [
                torch.arange(x, dtype=torch.float32)
                for x in self.input_resolution
            ]
            abs_coords = torch.stack(
                torch.meshgrid(abs_coords), dim=-1
            ).contiguous()  # T L H W 4
            for i, v in enumerate(self.input_resolution):
                abs_coords[..., i] /= v - 1 + 1e-9
            abs_coords = (abs_coords - 0.5) * 2  # T L H W 4
            abs_coords = abs_coords.flatten(0, -2)
            assert len(abs_coords.shape) == 2  # sanity check, L D
            if self.attention_mode == "xformer":
                self.register_buffer("abs_coords", abs_coords, persistent=False)
            else:
                abs_coords_x = abs_coords[:, None]
                abs_coords_y = abs_coords[None, :]
                abs_coords_x = abs_coords_x + abs_coords_y * 0.0
                abs_coords_y = abs_coords_y + abs_coords_x * 0.0
                abs_table = None
                if self.rel_pos_bias == "abs":
                    abs_table = torch.cat(
                        [abs_coords_x, abs_coords_y], dim=-1  # L L D
                    )  #
                elif self.rel_pos_bias == "rel":
                    abs_table = torch.cat(
                        abs_coords_x - abs_coords_y, dim=-1  # L L D
                    )  #
                else:
                    raise NotImplemented
                self.register_buffer("abs_table", abs_table, persistent=False)

    def add_rel_pos_bias(self, attn=None):
        if not self.rel_pos_bias:
            return 0.0
        elif self.rel_pos_bias in ["abs", "rel"]:
            nH = self.num_heads
            nL = np.product(self.input_resolution)
            # print(self.abs_table.max(),'abs_table')
            abs_position_bias = self.cpb_mlp(self.abs_table)  # L L H
            # print(abs_position_bias.max(),'abs_position_bias')
            abs_position_bias = abs_position_bias.permute(2, 0, 1)[None,]
            abs_position_bias = 16 * torch.sigmoid(abs_position_bias)
            # print(abs_position_bias.max(),'abs_position_bias')
            return abs_position_bias
        else:
            raise NotImplemented
        return 0.0

    def forward(self, x, mask=None):
        # print(x.max(),x.min(),'atten_in')
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        if self.attention_mode == "default":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn += self.add_rel_pos_bias(attn)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
        elif self.attention_mode == "cosine":
            # print(q.max(),k.max(),v.max(),'qkv')
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(
                -2, -1
            )
            # print(attn.max(),attn.min(),'attn')
            logit_scale = torch.clamp(
                self.logit_scale, max=math.log(1.0 / 0.01)
            ).exp()
            attn = attn * logit_scale
            # print(attn.max(),logit_scale.max(),'attn * logit_scale')
            attn += self.add_rel_pos_bias(attn)
            # print(attn.shape,attn.max(),'presoftmax')
            attn = attn.softmax(dim=-1)
            # print(attn.shape,attn.max(),'post')

            attn = self.attn_drop(attn)
        elif self.attention_mode == "xformer":
            q = q.permute(0, 2, 1, 3)  # N H L D -> N L H D
            k = k.permute(0, 2, 1, 3)  # N H L D -> N L H D
            v = v.permute(0, 2, 1, 3)  # N H L D -> N L H D
            edges = self.edges
            edge_features = torch.cat(
                [self.abs_coords[edges[0]], self.abs_coords[edges[1]]], dim=-1
            )
            abs_position_bias = self.cpb_mlp(edge_features)
            abs_position_bias = 16 * torch.sigmoid(abs_position_bias)
            # bias = self.add_rel_pos_bias(None) + self.atten_bias[None,None]

            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            logit_scale = torch.clamp(
                self.logit_scale, max=math.log(1.0 / 0.01)
            ).exp()

            qk = torch.einsum("nlhd,nlhd->nlh", q[:, edges[0]], k[:, edges[1]])
            # qk = qk * self.scale # N L_E, N_H Q^T
            qk = qk * logit_scale
            qk += abs_position_bias[None]  #

            # manual softmax
            qk = qk - qk.max(dim=1, keepdims=True).values
            qk = qk.exp()
            q_exp_sum = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                dtype=q.dtype,
                device=q.device,
            )
            q_exp_sum = q_exp_sum.scatter_add(
                dim=1,
                index=edges[0][None, :, None].expand(
                    q.shape[0], -1, q.shape[2]
                ),
                src=qk,
            )
            qk = qk / (
                q_exp_sum[:, edges[0]] + 1e-6
            )  # softmax (Q^T K / sqrt(d) + B)
            res = torch.einsum("nlh,nlhd->nlhd", qk, v[:, edges[1]])
            out = torch.zeros(q.shape, dtype=q.dtype, device=q.device)
            out = out.scatter_add(
                dim=1,
                index=edges[0][None, :, None, None].expand(
                    q.shape[0], -1, q.shape[2], q.shape[3]
                ),
                src=res,
            )
            # bias = bias.permute(0,2,1,3 ).expand(q.shape[0],-1,-1,-1)
            out = out.view(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out
            # attn = xops.memory_efficient_attention(q,k,v,attn_bias=bias)
        else:
            raise NotImplemented

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rel_pos_bias=False,
        num_patches=None,
        norm_mode="post",
        **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        input_resolution = num_patches
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            input_resolution=input_resolution,
            rel_pos_bias=rel_pos_bias,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        n, c, t, l, h, w = x.shape
        axis_len = dict(N=n, C=c, T=t, L=l, H=h, W=w)
        x = rearrange(x, "N C T L H W -> N (T L H W) C", **axis_len)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x = rearrange(x, "N (T L H W) C -> N C T L H W ", **axis_len)
        return x


class LayerNorm4d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N C T L H W -> N  T L H W  C
        return F.layer_norm(
            x.permute(0, 2, 3, 4, 5, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 5, 1, 2, 3, 4)


# todo try Gp norm or else


class GlobalLocalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm4d,
        rel_pos_bias=False,
        num_patches=None,
        norm_mode="post",
        **kwargs
    ):
        super().__init__()

        self.norm_mode = norm_mode
        assert self.norm_mode in ["pre", "post"]
        if self.norm_mode == "post":
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            self.norm3 = norm_layer(dim)
            self.norm4 = norm_layer(dim)
            self.prenorm1 = nn.Identity()
            self.prenorm2 = nn.Identity()
            self.prenorm3 = nn.Identity()
            self.prenorm4 = nn.Identity()
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
            self.norm4 = nn.Identity()
            self.prenorm1 = norm_layer(dim)
            self.prenorm2 = norm_layer(dim)
            self.prenorm3 = norm_layer(dim)
            self.prenorm4 = norm_layer(dim)

        input_resolution = num_patches
        self.attn_spatial = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_mode="cosine",
            rel_pos_bias=rel_pos_bias,
            input_resolution=(input_resolution[2], input_resolution[3]),
        )
        self.attn_local = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_mode="cosine",
            rel_pos_bias=rel_pos_bias,
            input_resolution=(input_resolution[0], input_resolution[1]),
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path3 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path4 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        x: N C T L H W
        """

        n, c, t, l, h, w = x.shape
        axis_len = dict(N=n, C=c, T=t, L=l, H=h, W=w)
        x = rearrange(x, "N C T L H W -> (N T L) (H W) C")
        x = x + self.drop_path(self.norm1(self.attn_spatial(self.prenorm1(x))))
        x = x + self.drop_path2(self.norm2(self.mlp1(self.prenorm2(x))))

        x = rearrange(x, "(N T L) (H W) C -> (N H W) ( T L) C", **axis_len)

        x = x + self.drop_path3(self.norm3(self.attn_local(self.prenorm3(x))))
        x = x + self.drop_path4(self.norm4(self.mlp2(self.prenorm4(x))))

        x = rearrange(x, "(N H W) (T L) C -> N C T L H W", **axis_len)
        return x


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm4d,
        rel_pos_bias=False,
        num_patches=None,
        norm_mode="pre",
        radius=1,
        grid=None,
        **kwargs
    ):
        super().__init__()

        self.norm_mode = norm_mode
        assert self.norm_mode in ["pre", "post"]
        if self.norm_mode == "post":
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            self.prenorm1 = nn.Identity()
            self.prenorm2 = nn.Identity()
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.prenorm1 = norm_layer(dim)
            self.prenorm2 = norm_layer(dim)

        input_resolution = num_patches
        self.attn_ = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_mode="xformer",
            rel_pos_bias=rel_pos_bias,
            input_resolution=(
                input_resolution[0],
                input_resolution[1],
                input_resolution[2],
                input_resolution[3],
            ),
            radius=radius,
            grid=grid,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        x: N C T L H W
        """

        n, c, t, l, h, w = x.shape
        axis_len = dict(N=n, C=c, T=t, L=l, H=h, W=w)
        x = rearrange(x, "N C T L H W -> N ( T L H W) C")
        x = x + self.drop_path(self.norm1(self.attn_(self.prenorm1(x))))
        x = x + self.drop_path2(self.norm2(self.mlp1(self.prenorm2(x))))
        x = rearrange(x, "N ( T L H W) C -> N C T L H W", **axis_len)
        return x


class ThreeWayAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm4d,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)

        self.attn_spatial = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_mode="cosine",
        )
        self.attn_pressure = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_mode="cosine",
        )
        self.attn_temporal = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_mode="cosine",
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp3 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path3 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path4 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path5 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path6 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        x: N C T L H W
        """

        n, c, t, l, h, w = x.shape
        axis_len = dict(N=n, C=c, T=t, L=l, H=h, W=w)
        x = rearrange(x, "N C T L H W -> (N T L) (H W) C", **axis_len)
        x = x + self.drop_path1(self.norm1(self.attn_spatial(x)))
        x = x + self.drop_path2(self.norm2(self.mlp1(x)))

        x = rearrange(x, "(N T L) (H W) C -> (N H W T) L C", **axis_len)

        x = x + self.drop_path3(self.norm3(self.attn_pressure(x)))
        x = x + self.drop_path4(self.norm4(self.mlp2(x)))

        x = rearrange(x, "(N H W T) L C -> (N H W L) T C", **axis_len)

        x = x + self.drop_path5(self.norm5(self.attn_temporal(x)))
        x = x + self.drop_path6(self.norm6(self.mlp3(x)))

        x = rearrange(x, "(N H W L) T C -> N C T L H W", **axis_len)
        return x


class WindowedAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        window_size=(2, 2, 2, 2),
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
        num_patches=(1, 1, 32, 64),
        roll=False,
        act_layer=nn.GELU,
        norm_layer=LayerNorm4d,
        bias_mode="earth",
        mask=True,
        **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.atten = WindowAttention4D(
            dim,
            window_size=window_size,
            input_resolution=num_patches,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            roll=roll,
            bias_mode=bias_mode,
            mask=mask,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        x: N C T L H W
        """

        n, c, t, l, h, w = x.shape
        axis_len = dict(N=n, C=c, T=t, L=l, H=h, W=w)
        x0 = rearrange(x, "N C T L H W -> N (T L H W) C")
        x = self.atten(x)
        x = rearrange(x, "N C T L H W -> N (T L H W) C")
        x = x0 + self.drop_path(self.norm1(x))
        x = x + self.drop_path2(self.norm2(self.mlp1(x)))

        x = rearrange(x, " N (T L H W) C -> N C T L H W", **axis_len)
        return x


class PanguWeatherBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        window_size=(2, 2, 2, 2),
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
        num_patches=(1, 1, 32, 64),
        act_layer=nn.GELU,
        norm_layer=LayerNorm4d,
        depth=1,
        use_checkpoint=False,
        inner_block=GlobalLocalAttentionBlock,
        bias_mode="earth",
        mask=True,
        **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                inner_block(
                    dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    drop=drop,
                    window_size=window_size,
                    num_patches=num_patches,
                    roll=(i % 2) == 0,
                    proj_drop=proj_drop,
                    act_layer=act_layer,
                    attn_drop=attn_drop,
                    qk_scale=qk_scale,
                    bias_mode=bias_mode,
                    mask=mask,
                    **kwargs
                )
                for i in range(depth)
            ]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        """
        x: N C T L H W
        """

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


registry = dict(
    GlobalLocalAttentionBlock=GlobalLocalAttentionBlock,
    WindowedAttentionBlock=WindowedAttentionBlock,
    ThreeWayAttentionBlock=ThreeWayAttentionBlock,
    ViTBlock=ViTBlock,
    LayerNorm4d=LayerNorm4d,
    PanguWeatherBlock=PanguWeatherBlock,
    LayerNorm=nn.LayerNorm,
    GraphAttentionBlock=GraphAttentionBlock,
)
