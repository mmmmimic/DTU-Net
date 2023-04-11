import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models as models

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        if activation is not None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        self.conv1 = Conv3x3(in_channels, out_channels//2)
        self.conv2 = Conv3x3(out_channels//2, out_channels, stride=stride, activation=None)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        x = out + self.short_cut(x)
        return F.leaky_relu(x, negative_slope=0.2)

class TripletModule(nn.Module):
    def __init__(self, embedding_size=1024):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),    
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True), 
            nn.AdaptiveMaxPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(-2, -1).squeeze(-1)
        x = self.fc(x)
        return x



if __name__ == "__main__":
    x = torch.rand(10, 1, 32, 32).cuda()
    model = TripletModule(512).cuda()
    x = model(x)
    print(x.shape)
