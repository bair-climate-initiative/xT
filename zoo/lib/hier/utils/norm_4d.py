import torch
from torch.nn import functional as F
from torch import nn

class LayerNorm4d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N C T L H W -> N  T L H W  C
        return F.layer_norm(
            x.permute(0, 2, 3,4,5, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 5, 1, 2,3,4)
# todo try Gp norm or else
