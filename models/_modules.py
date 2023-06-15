import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01), bn=True, **kwargs):
        super().__init__()
        if activation is not None:
            if bn:
                self.conv = nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False, **kwargs),
                    nn.BatchNorm1d(out_channels),
                    activation
                )
            else:
                self.conv = nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False, **kwargs),
                    activation
                )
        else:
            self.conv = nn.Linear(in_channels, out_channels, bias=True, **kwargs)
        

    def forward(self, x):
        return self.conv(x)
   
class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01), **kwargs):
        super().__init__()
        if activation is not None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, **kwargs),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True, **kwargs)
        

    def forward(self, x):
        return self.conv(x)

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01), bn=True, **kwargs):
        super().__init__()
        if activation is not None:
            if bn:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                    nn.BatchNorm2d(out_channels),
                    activation
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                    activation
                )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True, **kwargs)
        

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=nn.LeakyReLU(negative_slope=0.01),
                 dropout = 0.2,
                 **kwargs):

        super().__init__()

        self.conv = nn.Sequential(
            Conv3x3(in_channels, (in_channels+out_channels)//2, activation, **kwargs),
            nn.Dropout(p=dropout),
            Conv3x3((in_channels+out_channels)//2, out_channels, activation, **kwargs)
        )
        # self.conv =  Conv3x3(in_channels, out_channels, activation, **kwargs),

    def forward(self,x):
        return self.conv(x)

class DeConvBlock(nn.Module):
    def __init__(self, left_channels, right_channels, out_channels, interpolation=True, **kwargs):
        super().__init__()

        #self.up = nn.Upsample(scale_factor=2) if interpolation else nn.Sequential(nn.ConvTranspose2d(left_channels, 
        #                                                                    left_channels,  
        #                                                                    kernel_size=2, 
        #                                                                    stride=2,
        #                                                                    padding=0,
        #                                                                    bias=True),
        #                                                                    nn.ReLU())
        #self.conv = ConvBlock(left_channels+right_channels, out_channels, **kwargs) if interpolation else ConvBlock(right_channels+left_channels, 
        #                                                                                            out_channels, **kwargs)
         self.up = nn.Sequential(nn.Upsample(scale_factor=2), ConvBlock(right_channels, right_channels//2, **kwargs), nn.BatchNorm2d(right_channels//2), nn.ReLU()) if interpolation else nn.Sequential(nn.ConvTranspose2d(right_channels, 
                                                                             right_channels//2,  
                                                                             kernel_size=2, 
                                                                             stride=2,
                                                                             padding=0,
                                                                             bias=True),
                                                                             nn.ReLU())
         self.conv = ConvBlock(left_channels+right_channels//2, out_channels, **kwargs) if interpolation else ConvBlock(right_channels//2+left_channels, 
                                                                                                     out_channels, **kwargs)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        if x1 is not None:
            if x1.size(-1) != x2.size(-1):
                if x1.size(-1) < x2.size(-1):
                    x1 = F.interpolate(x1, (x2.size(-2), x2.size(-1)))
                else:
                    x2 = F.interpolate(x2, (x1.size(-2), x1.size(-1)))
            x = torch.cat((x1, x2), dim=1)
        else:
            x = x2
        x = self.conv(x)

        return x
