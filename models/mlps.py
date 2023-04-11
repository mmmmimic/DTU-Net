import torch
import torch.nn as nn
import math

class Conv1D1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.2), bn=True):
        super().__init__()
        if activation is not None:
            if bn:
                self.fc = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    activation
                )
            else:
                self.fc = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    activation
                )                
        else:
            self.fc = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
            )            
    
    def forward(self, x):
        x = self.fc(x)
        return x

class Conv2D1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        if activation is not None:
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            )            
    
    def forward(self, x):
        x = self.fc(x)
        return x

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.2), bn=True):
        super().__init__()
        if activation is not None:
            if bn:
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    nn.BatchNorm1d(out_channels),
                    activation
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    activation
                )       
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=True)
            )            
    
    def forward(self, x):
        x = self.fc(x)
        return x

class ResConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01), bn=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv1x1(in_channels, in_channels, activation, bn),
            Conv1x1(in_channels, out_channels, activation=None, bn=False)
        )        

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x = x1 + x2
        return self.act(x)

class ResConv1D1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01), bn=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv1D1x1(in_channels, in_channels, activation, bn),
            nn.Dropout1d(0.2),
            Conv1D1x1(in_channels, out_channels, activation=None, bn=False)
        )        

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x = x1 + x2
        return self.act(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, expand_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
                Conv1x1(in_channels, expand_dim//8),
                Conv1x1(expand_dim//8, expand_dim//4),
                Conv1x1(expand_dim//4, expand_dim//2),
                Conv1x1(expand_dim//2, expand_dim),
                Conv1x1(expand_dim, out_channels, activation=None)
        )
        # self.fc = Conv1x1(in_channels, out_channels, activation=None)

    def forward(self, x):
        return self.fc(x)

class SelectionMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mode='hard', expand_dim=128, cb_dim=8, head=256):
        super().__init__()
        self.mode = mode

        self.cb_dim = cb_dim

        self.head = head


        if mode == 'soft':
            self.combiner = nn.Linear(in_channels, cb_dim, bias=False)
            self.fc = nn.Sequential(
                Conv1x1(cb_dim, out_channels, activation=None)
        )
        elif mode == 'hard':
            self.combiner = nn.Sequential(
            Conv1x1(in_channels, 2*in_channels),
            Conv1x1(2*in_channels, 2*in_channels),
            Conv1x1(2*in_channels, cb_dim*head),
            )

            self.fc = nn.Sequential(
                Conv1D1x1(cb_dim*head+in_channels, 1024),
                Conv1D1x1(1024, out_channels, activation=None)
        )

        self.encoder = nn.Sequential(
                        Conv2D1x1(1, expand_dim//4),
                        Conv2D1x1(expand_dim//4, expand_dim//2),
                        # Conv2D1x1(expand_dim//2, expand_dim//4),
                        # Conv2D1x1(expand_dim//4, 1),
                        Conv2D1x1(expand_dim//2, expand_dim)
        )

        self.decoder = nn.Sequential(
                Conv1D1x1(expand_dim, expand_dim//2),
                Conv1D1x1(expand_dim//2, expand_dim//4),
                Conv1D1x1(expand_dim//4, 1, activation=None)            
        )

    def forward(self, x):
        # apply attention
        if self.mode == 'soft':
            # att = self.attention_layer(x)
            # x = x.unsqueeze(1)
            # x = self.encoder(x) # B, 256, N
            # x = att.unsqueeze(1) * x
            # x = torch.sum(x, dim=-1)
            x = self.combiner(x)
            x = self.encoder(x.unsqueeze(1).unsqueeze(1))
            x = x.squeeze(2)
        elif self.mode == 'hard':
            x_raw = x.clone()
            x_pow = x**2
            weight = self.combiner(x)
            weight = nn.functional.gumbel_softmax(weight.reshape(x.size(0), self.cb_dim, self.head, -1), dim=1, hard=True)
            x = x.unsqueeze(1).unsqueeze(1)
            x = weight * x
            x = torch.sum(x, dim=-1)
            x = x.flatten(-2)
            x_pow = weight * x_pow.unsqueeze(1).unsqueeze(1)
            x_pow = torch.sum(x_pow, dim=-1)
            x_pow = x_pow.flatten(-2)
            x = (x**2 - x_pow)/2
            x = torch.cat((x_raw, x), dim=-1)
            x = self.encoder(x.unsqueeze(1).unsqueeze(1))
            x = x.squeeze(2)

        x = x.transpose(1,2).contiguous()
        x = self.fc(x)
        x = x.transpose(1,2).contiguous()
        # x = x.squeeze(1)
        x = self.decoder(x).squeeze(1)
        return x
