import timm
from torch import nn
class SwinWrapper(nn.Module):

    def __init__(self,model,hidden_size=768):
        super().__init__()
        self.model = model
        # self.model.head = nn.Identity()
        # self.model.norm = nn.Identity()
        self.feature_info = list(model.feature_info)
    def forward(self,x):
        intermediates = self.model(x)
        intermediates = list([x.permute(0,3,1,2) for x in intermediates])
        return intermediates
    
    

def swinv2_base_window16_256_timm(*args,**kwargs):
    model = timm.create_model('swinv2_base_window16_256.ms_in1k',features_only=True,pretrained=True)
    return SwinWrapper(model)
