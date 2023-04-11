import torch.nn as nn
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from monai.losses import FocalLoss as FLoss

def get_weight(trainloader):
    pixel = []
    for x in trainloader:
        pixel.append(x['mask'].flatten().numpy())
    pixel = np.concatenate(pixel)
    weight = compute_class_weight(class_weight='balanced', classes=np.unique(pixel), y=pixel)
    weight = torch.tensor(weight, requires_grad=False)   
    return weight 

class FocalLoss(nn.Module):
    '''
    focal loss
    '''
    def __init__(self, weight=None, gamma=2):
        super().__init__()
        self.nllLoss = nn.NLLLoss(weight=weight)
        self.gamma = gamma
    
    def forward(self, outs):
        logit = outs['logit']
        target = outs['mask']
        logit, target = logit.flatten(-2, -1), target.flatten(-2, -1)
        softmax = torch.softmax(logit, dim=1)
        log_logits = torch.log_softmax(logit, dim=1) # stabalize the numerical computation
        fix_weights = (1-softmax) ** self.gamma
        logits = fix_weights * log_logits

        return self.nllLoss(logits, target)

# class FocalLoss:
#     def __init__(self, weight, **kwargs):
#         self.criterion = FLoss(weight=weight, to_onehot_y=True, **kwargs)
    
#     def __call__(self, outs):
#         logit, mask = outs['logit'], outs['mask']
#         return self.criterion(logit, mask.unsqueeze(1))