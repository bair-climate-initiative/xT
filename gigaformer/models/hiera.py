import hiera
from torch import nn
class HieraWrapper(nn.Module):

    def __init__(self,model,hidden_size=768):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()
        self.feature_info = [
            dict(num_chs=hidden_size, reduction=32, module="cls_pool")
        ]

    def forward(self,x):
        _, intermediates = self.model(x, return_intermediates=True)
        return [intermediates[-1].permute(0,3,1,2)] # n  768 7 7 
    
    

def get_hiera_model_base(*args,**kwargs):
    model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    return HieraWrapper(model)

def get_hiera_model_base_plus(*args,**kwargs):
    model = hiera.hiera_base_plus_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    return HieraWrapper(model,hidden_size=896)

def get_hiera_model_base_plus_448(*args,**kwargs):
    model = hiera.hiera_base_plus_224(pretrained=True, input_size=(448,448),checkpoint="mae_in1k_ft_in1k")
    return HieraWrapper(model,hidden_size=896)


def get_hiera_model_tiny(*args,**kwargs):
    model = hiera.hiera_tiny_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    return HieraWrapper(model)

def get_hiera_model_small(*args,**kwargs):
    model = hiera.hiera_small_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    return HieraWrapper(model)
get_hiera_model = get_hiera_model_base