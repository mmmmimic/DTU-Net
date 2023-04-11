import torch.nn as nn
import torchvision

def resnet(in_channels, out_channels, depth=18, weights=None):
    resnet = eval(f"torchvision.models.resnet{depth}(weights=weights, progress=True)")
    # for name, param in resnet.named_parameters():
    #     if 'fc' not in name: 
    #         param.requires_grad = False

    if in_channels != 3:
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if out_channels != 1000:
        resnet.fc = nn.Linear(resnet.fc.in_features, out_channels, bias=True)
    return resnet

def inception(in_channels, out_channels, weights):
    inception = torchvision.models.inception_v3(weights = weights)
    if in_channels != 3:
        inception.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    if out_channels != 1000:
        inception.fc = nn.Linear(2048, out_channels, bias=True)
        inception.AuxLogits.fc = nn.Linear(in_features=768, out_features=out_channels, bias=True)

    # for name, param in model.named_parameters():
    #     if 'fc' not in name:  # and 'Mixed_7c' not in name:
    #         param.requires_grad = False
    return inception

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = inception(in_channels, out_channels)
    
    def forward(self, x):
        x = self.model(x)

        return x

class ResNet(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.net = resnet(**args)
    
    def forward(self, x):
        image = x['image']
        logit = self.net(image)
        return {'logit': logit}

class IdendityMapping(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return {'logit': x['image']}