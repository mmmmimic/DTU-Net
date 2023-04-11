#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:09:49 2022

@author: manli
"""

'''
imports
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
from .unet_family import RegUNet

'''
Build the network architecture
'''


'''
U-Net: best segmentation arch ever
'''
class ConvBlock(nn.Module):
    '''
    stacking conv and relu
    '''
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, 
                      kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(hid_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, 
                      kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        
    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    '''
    stacking deconv and relu
    '''
    def __init__(self, in_channels, hid_channels, out_channels, interp=True):
        super().__init__()
        self.interp = interp

        if interp:
            self.deconv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=3, padding=1, 
                                            stride=1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU()
                                        )
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                            kernel_size=3, padding=1, 
                                            stride=2, output_padding=1)
        self.conv = ConvBlock(out_channels + out_channels, hid_channels, out_channels)

    def forward(self, x1, x2):
        if self.interp:
            x1 = F.interpolate(x1, scale_factor=2)
        x1 = self.deconv(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)   

class UNet(nn.Module):
    '''
    build U-Net
    '''    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.left_conv1 = ConvBlock(in_channels, 64, 64)
        self.left_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, 128)
            )
        self.left_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, 256)
            )
        self.left_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, 512), 
            )        
        self.left_conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(512, 1024, 1024)           
            )
        
        self.right_conv5 = DeconvBlock(1024, 512, 512)
        self.right_conv4 = DeconvBlock(512, 256, 256)
        self.right_conv3 = DeconvBlock(256, 128, 128)
        self.right_conv2 = DeconvBlock(128, 64, 64)
        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1, 
                                   padding=0, stride=1)

    def forward(self, x):
        # x: [B, C, W, H]
        x = x['image']
        l1 = self.left_conv1(x)
        l2 = self.left_conv2(l1) 
        l3 = self.left_conv3(l2) 
        l4 = self.left_conv4(l3)
        l5 = self.left_conv5(l4)
        r5 = self.right_conv5(l5, l4)
        r4 = self.right_conv4(r5, l3)
        r3 = self.right_conv3(r4, l2)
        r2 = self.right_conv2(r3, l1)
        r1 = self.last_conv(r2)
        out = {}
        out['logit'] = r1
        out['emb'] = l5
        return out

class FlexUNet(nn.Module):
    def __init__(self, in_channels, out_channels, pair_num=4):
        super().__init__()
        # flexible U-Net, can play with the pair num

        init_channels = 64
        self.first_conv = ConvBlock(in_channels, init_channels, init_channels)

        self.pair_num = pair_num
        # self.register_buffer('pair_num', pair_num)

        self.convs = nn.ModuleDict()
        for i in range(pair_num):
            left_conv, right_conv = self._mask_pair_conv(init_channels)
            init_channels *= 2
            self.convs["left_conv%d"%(i+1)] = left_conv
            self.convs["right_conv%d"%(i+1)] = right_conv

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1, 
                                   padding=0, stride=1)

        self.r = None
        self.l1 = None

    def _mask_pair_conv(self, init_channels):
        left_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(init_channels, init_channels*2, init_channels*2)
            )
        right_conv =  DeconvBlock(init_channels*2, init_channels, init_channels)

        return left_conv, right_conv

    def forward(self, x):
        # x: [B, C, W, H]
        x = x['image']
        l = self.first_conv(x)
        self.l1 = l
        l_list = [l]

        # encoder
        for i in range(self.pair_num):
            l = self.convs["left_conv%d"%(i+1)](l)
            l_list.append(l)
        emb = l.clone()
        # decoder
        for i in range(self.pair_num, 0, -1):
            l = self.convs["right_conv%d"%(i)](l, l_list[i-1])

        r = self.last_conv(l)

        self.r = r

        out = {'logit': r, 'pixel_feature': l}
        return out

class DTUNet(nn.Module):
    def __init__(self, in_channels, out_channels, quality_head=True):
        super().__init__()
        # self.unet = FlexUNet(in_channels, out_channels, pair_num=3)
        quality_channel = 1 if quality_head else 0
        self.unet = RegUNet(in_channels, num_classes=out_channels + quality_channel)
        # toponet
        self.left_conv1 = ConvBlock(out_channels, 64, 64)
        self.left_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, 128)
            )
        self.left_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, 256)
            )
        self.left_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, 512)
            )        

        self.right_conv4 = DeconvBlock(512, 256, 256)
        self.right_conv3 = DeconvBlock(256, 128, 128)
        self.right_conv2 = DeconvBlock(128, 64, 64)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1, 
                                   padding=0, stride=1)     
        
        self.class_num = out_channels
        self.quality_head = quality_head

    def forward_one(self, x):
        l1 = self.left_conv1(x)
        l2 = self.left_conv2(l1) 
        l3 = self.left_conv3(l2) 
        l4 = self.left_conv4(l3)
        return l1, l2, l3, l4

    def _get_global_feature(self, x):
        x = F.adaptive_avg_pool2d(x, (1,1))
        return torch.sigmoid(x.squeeze())

    @staticmethod
    def random_erase(img):
        h, w = img.shape
        if not img.max():
            return 1 - img
        
        kernel_size = 8
        vals = torch.max_pool2d((img>0).float().view(1, 1, h, w), kernel_size=kernel_size, stride=kernel_size).squeeze()
        vals = vals.view(-1)
        idx = torch.nonzero(vals)

        patches = img.unfold(0, kernel_size, kernel_size).unfold(1, kernel_size, kernel_size)
        
        m, n, _, _ = patches.shape
        patches = patches.contiguous().view(-1, kernel_size, kernel_size)

        num = len(idx)
        index = np.array(range(num))
        np.random.shuffle(index)
        sample_index = idx[index[:int(num*0.2)]]
        patches[sample_index, ...] = 0

        patches = patches.view(m, n, kernel_size, kernel_size)
        patches = patches.contiguous()

        img = patches.permute(0,2,1,3).contiguous()
        img = img.view(h, w)

        return img

    def forward(self, x):
        # coarse segmentation
        unet_out = self.unet(x) # [B, C, H, W]
        coarse_logit = unet_out['logit']
        if self.quality_head:
            coarse_logit = coarse_logit[:,:-1, ...]
            quality_pred = coarse_logit[:,-1,...]

        coarse_score = torch.softmax(coarse_logit, dim=1)

        anchor_features = self.forward_one(coarse_score)

        if self.training: # during training, activate triplet loss
            mask = x['mask']
            crp_mask = torch.zeros_like(mask)
            for i in range(mask.size(0)):
                crp_mask[i,...] = self.random_erase(mask[i,...].clone())
            mask = F.one_hot(mask.long(), self.class_num).permute(0,3,1,2).float()
            crp_mask = F.one_hot(crp_mask.long(), self.class_num).permute(0,3,1,2).float()

            mask_features = self.forward_one(mask)
            crp_mask_features = self.forward_one(crp_mask)

            img_emb = anchor_features[-1]
            mask_emb = mask_features[-1]
            crp_mask_emb = crp_mask_features[-1]

            triplet_loss = nn.TripletMarginLoss(margin=0.1)(img_emb, mask_emb, crp_mask_emb)
        else:
            triplet_loss = torch.tensor([0]).to(coarse_score.device)

        # triplet_loss = 0
        l1, l2, l3, l4 = anchor_features
        b,c,h,w = l4.shape
        global_feature = self._get_global_feature(l4).view(b, c, 1, 1).repeat(1, 1, h, w)
        l4 = global_feature*l4
        r4 = self.right_conv4(l4, l3)
        r3 = self.right_conv3(r4, l2)
        r2 = self.right_conv2(r3, l1)
        
        r1 = self.last_conv(r2) 
        s1 = torch.sigmoid(r1)    
        
        bf_mask = s1.squeeze(1)

        bg, fg = torch.split(coarse_score, [1, coarse_score.size(1)-1], dim=1)
        bg, fg = (bg+(1-s1))/2, (fg+s1)/2
        # bg, fg = (1-s1)*(1-bg), fg*s1
        # bg, fg = bg * (1-s1), fg * s1
        logit = torch.cat((bg, fg), dim=1)   
        
        if self.quality_head:
            logit = torch.cat((logit, quality_pred.unsqueeze(1)), dim=1)

        out = {}
        out['coarse_logit'] = coarse_logit
        out['logit'] = logit
        out['topo_mask'] = bf_mask
        out['triplet_loss'] = triplet_loss

        return out

if __name__ == "__main__":
    img = torch.rand(2, 3, 256, 256).cuda()
    mask = torch.ones(2, 256, 256).cuda()
    x = {'image':img, 'mask':mask}
    model = DTUNet(3, 10).cuda()
    model.train()
    x = model(x)
    print(x['coarse_logit'].shape)
    print(x['logit'].shape)
    print(x['topo_mask'].shape)
    print(x['triplet_loss'])