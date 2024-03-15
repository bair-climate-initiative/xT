from torch import nn

import hiera


class HieraWrapper(nn.Module):
    def __init__(self, model, hidden_size=768):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()
        self.feature_info = [dict(num_chs=hidden_size, reduction=32, module="cls_pool")]

    def forward(self, x):
        _, intermediates = self.model(x, return_intermediates=True)
        return [intermediates[-1].permute(0, 3, 1, 2)]  # n  768 7 7


def hiera_tiny(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = hiera.hiera_tiny(
        pretrained=True, checkpoint="mae_in1k_ft_in1k", input_size=input_size
    )
    return HieraWrapper(model)


def hiera_small(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = hiera.hiera_small(
        pretrained=True, checkpoint="mae_in1k_ft_in1k", input_size=input_size
    )
    return HieraWrapper(model)


def hiera_base(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = hiera.hiera_base(
        pretrained=True, checkpoint="mae_in1k_ft_in1k", input_size=input_size
    )
    return HieraWrapper(model)


def hiera_large(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = hiera.hiera_large(
        pretrained=True, checkpoint="mae_in1k_ft_in1k", input_size=input_size
    )
    return HieraWrapper(model)


def hiera_base_plus(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = hiera.hiera_base_plus(
        pretrained=True, checkpoint="mae_in1k_ft_in1k", input_size=input_size
    )
    return HieraWrapper(model, hidden_size=896)
