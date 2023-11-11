import hiera
from torch import nn
class HieraWrapper(nn.Module):

    def __init__(self,model):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()
        self.feature_info = [
            dict(num_chs=768, reduction=32, module="cls_pool")
        ]

    def forward(self,x):
        _, intermediates = self.model(x, return_intermediates=True)
        return [intermediates[-1].permute(0,3,1,2)] # n  768 7 7 
    
    

def get_hiera_model(*args,**kwargs):
    model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    return HieraWrapper(model)
