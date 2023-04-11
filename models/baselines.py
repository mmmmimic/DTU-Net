import torch
import torch.nn as nn
import torchvision
from ._modules import Conv1x1
from sononet import SonoNet
from .backbones import ResNet

'''
NB: According to arxiv.org/abs/2105.04289, among the three variants of CBM (joint, sequential, independent), 
only the independent variant has the three designed property. 
'''

class SonoNets(nn.Module):
    def __init__(self, config, num_labels, weights,
             features_only, in_channels):
        super().__init__()
        self.net = SonoNet(config, num_labels, weights,
             features_only, in_channels)
    
    def forward(self, x):
        image = x['image']
        logit = self.net(image)
        return {'logit': logit}    

if __name__ == "__main__":
    model = SonoNets(config='SN32', num_labels=8, weights=False, features_only=False, in_channels=3)

    model.train()
    data = {
        'image': torch.rand(2, 3, 224, 224).float(),
        'concept': torch.ones(2, 27).float(),
        'mask': torch.ones(2, 224, 224).long()
    }
    out = model(data)
    print(out['logit'].shape)
    print(out['logit'])
    print(out['concept_logit'].shape)






