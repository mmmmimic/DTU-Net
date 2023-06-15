import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import timm.models as models
import torchvision
from ._modules import Conv3x3, Conv1x1, ConvBlock, DeConvBlock

def get_regnet(in_channels=3,
                embedding_size=1024,
                model_size = '016',
                model_type = 'y',
                pretrained=True):
    assert model_size in ['004', '008', '016', '032']
    assert model_type in ['x', 'y']
    
    model = eval("models.regnet.regnet%s_%s(pretrained=pretrained, num_classes=embedding_size)"%(model_type, model_size))

    if in_channels != 3:
        model.stem.conv = torch.nn.Conv2d(in_channels=in_channels,
                                            out_channels=model.stem.conv.out_channels,
                                            kernel_size=model.stem.conv.kernel_size,
                                            stride = model.stem.conv.stride,
                                            padding=model.stem.conv.padding,
                                            bias = model.stem.conv.bias)

    return model.stem, model.s1, model.s2, model.s3, model.s4

def get_resnet(in_channels=3, model_size='resnet50', pretrained=True):
    model = eval("models.resnet.%s(pretrained=pretrained)"%model_size)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                            out_channels=model.conv1.out_channels,
                                            kernel_size=model.conv1.kernel_size,
                                            stride =model.conv1.stride,
                                            padding=model.conv1.padding,
                                            bias = model.conv1.bias)

    conv1 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool
    )
    conv2 = model.layer1
    conv3 = model.layer2
    conv4 = model.layer3
    conv5 = model.layer4
    return conv1, conv2, conv3, conv4, conv5    

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, channels=[64, 128, 256, 512, 1024], **kwargs):
        super().__init__()

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.down_convs.append(
            ConvBlock(in_channels=in_channels, out_channels=channels[0], stride=1, **kwargs) # init_conv
        )
        num_layers = len(channels) - 1
        for i in range(num_layers):
            self.down_convs.append(
                ConvBlock(in_channels=channels[i], out_channels=channels[i+1], stride=2, **kwargs)
            )
            self.up_convs.append(
                DeConvBlock(left_channels=channels[-(i+2)], right_channels = channels[-(i+1)], out_channels=channels[-(i+2)], **kwargs)
            )

        self.up_convs.append(
        DeConvBlock(left_channels=channels[0], right_channels=channels[1], out_channels=channels[0], **kwargs)
        )
        self.last_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = x['image']
        downs = []
        for down_conv in self.down_convs:
            x = down_conv(x)
            downs.append(x)
        downs = downs[:-1]
        downs = reversed(downs)
        for down, up_conv in zip(downs, self.up_convs):
            x = up_conv(down, x)

        x = self.last_conv(x)

        out_dict = {}
        out_dict['logit'] = x

        return out_dict

class RegUNet(nn.Module):
    def __init__(self, in_channels, num_classes, model_size = '016', model_type= 'y', interpolation=True, **kwargs):
        super().__init__()

        if model_size == '004':
            channels = [32,48,104,208,440]
        elif model_size == '008':
            channels = [32,64,128,320,768]
        elif model_size == '016':
            channels = [32,48,120,336,888]
        elif model_size == '032':
            channels = [32,72,216,576,1512]
        else:
            raise Exception(f"Invalid model_size '{model_size}'")
        
        # get regnet encoder
        self.down_conv0, self.down_conv1, self.down_conv2, self.down_conv3, self.down_conv4 = get_regnet(model_size=model_size, model_type=model_type, in_channels=in_channels, pretrained=True)

        self.up_conv4 = DeConvBlock(channels[-2], channels[-1], out_channels=channels[-2], interpolation=interpolation, **kwargs)

        self.up_conv3 = DeConvBlock(channels[-3], channels[-2], out_channels=channels[-3], interpolation=interpolation, **kwargs)

        self.up_conv2 = DeConvBlock(channels[-4], channels[-3], out_channels=channels[-4], interpolation=interpolation, **kwargs)

        self.up_conv1 = DeConvBlock(channels[-5], channels[-4], out_channels=channels[-5], interpolation=interpolation, **kwargs)

        self.up_conv0 = DeConvBlock(0, channels[-5], out_channels=channels[-5])
        self.fc = nn.Conv2d(channels[-5], num_classes, kernel_size=1)

        # self.up_conv0 = DeConvBlock(0, channels[-5], out_channels=num_classes, **kwargs)


    def forward(self, x):
        x = x['image']
        # left
        down1 = self.down_conv0(x)
        down2 = self.down_conv1(down1)
        down3 = self.down_conv2(down2)
        down4 = self.down_conv3(down3)

        # bottleneck
        down5 = self.down_conv4(down4)
        emb = torch.mean(down5.flatten(2), dim=-1)

        # right
        up4 = self.up_conv4(down4, down5)
        up3 = self.up_conv3(down3, up4)
        up2 = self.up_conv2(down2, up3)
        up1 = self.up_conv1(down1, up2)
        
        out = self.up_conv0(None, up1)
        out = self.fc(out)

        out_dict = {}
        out_dict['logit'] = out 
        out_dict['emb'] = emb

        return out_dict

class ResUNet(nn.Module):
    def __init__(self, in_channels, num_classes, model_size = 'resnet18', init_channels = 64, interpolation=True, **kwargs):
        super().__init__()

        if model_size == 'resnet18':
            channels = [64,64,128,256,512]
        elif model_size == 'resnet50':
            channels = [64,256,512,1024,2048]
        elif model_size == 'resnet101':
            channels = [64,256,512,1024,2048]
        else:
            raise Exception(f"Invalid model_size '{model_size}'")
        
        # get regnet encoder
        self.down_conv0, self.down_conv1, self.down_conv2, self.down_conv3, self.down_conv4 = get_resnet(model_size=model_size, in_channels=in_channels)

        self.skip_conv = ConvBlock(in_channels, init_channels, **kwargs)

        self.up_conv4 = DeConvBlock(channels[-2], channels[-1], out_channels=channels[-2], **kwargs)

        self.up_conv3 = DeConvBlock(channels[-3], channels[-2], out_channels=channels[-3], **kwargs)

        self.up_conv2 = DeConvBlock(channels[-4], channels[-3], out_channels=channels[-4], **kwargs)

        self.up_conv1 = DeConvBlock(channels[-5], channels[-4], out_channels=channels[-5], **kwargs)

        self.up_conv0 = DeConvBlock(init_channels, channels[-5], out_channels=num_classes, **kwargs)

    def forward(self, x):
        x = x['image']
        # left
        down1 = self.down_conv0(x)
        down2 = self.down_conv1(down1)
        down3 = self.down_conv2(down2)
        down4 = self.down_conv3(down3)

        # bottleneck
        down5 = self.down_conv4(down4)

        # right
        up4 = self.up_conv4(down4, down5)
        up3 = self.up_conv3(down3, up4)
        up2 = self.up_conv2(down2, up3)
        up1 = self.up_conv1(down1, up2)
        
        down0 = self.skip_conv(x)
        out = self.up_conv0(down0, up1)

        out_dict = {}
        out_dict['logit'] = out

        return out_dict

class FlexUNet(nn.Module):
    def __init__(self, in_channels, out_channels, pair_num=4, interpolation=True):
        super().__init__()
        # flexible U-Net, can play with the pair num

        init_channels = 64
        self.first_conv = ConvBlock(in_channels, init_channels)

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

    def _mask_pair_conv(self, init_channels):
        left_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(init_channels, init_channels*2)
            )
        right_conv =  DeConvBlock(init_channels*2, init_channels, init_channels, interpolation=interpolation)

        return left_conv, right_conv

    def forward(self, x):
        # x: [B, C, W, H]
        x = x['image']
        l = self.first_conv(x)
        l_list = [l]

        # encoder
        for i in range(self.pair_num):
            l = self.convs["left_conv%d"%(i+1)](l)
            l_list.append(l)
        # decoder
        for i in range(self.pair_num, 0, -1):
            l = self.convs["right_conv%d"%(i)](l_list[i-1], l)

        r = self.last_conv(l)

        out = {'logit': r}
        return out

class DTUNet(nn.Module):
    def __init__(self, in_channels, out_channels, interpolation=True): # , quality_head=True
        super().__init__()
        # self.unet = FlexUNet(in_channels, out_channels, pair_num=3, interpolation=interpolation)
        self.unet = RegUNet(in_channels, num_classes=out_channels, interpolation=interpolation)
        # toponet
        self.left_conv1 = ConvBlock(out_channels, 64)
        self.left_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128)
            )
        self.left_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256)
            )
        self.left_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512)
            )        

        self.right_conv4 = DeConvBlock(256, 512, 256, interpolation=interpolation)
        self.right_conv3 = DeConvBlock(128, 256, 128, interpolation=interpolation)
        self.right_conv2 = DeConvBlock(64, 128, 64, interpolation=interpolation)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1, 
                                   padding=0, stride=1)             

        self.class_num = out_channels

        # self.attention_module = nn.Sequential(
        #                 Conv3x3(in_channels+out_channels+1, 32),
        #                 Conv3x3(32, 32),
        #                 Conv3x3(32, 2)
        # )
        # self.attention_module = nn.Sequential(
        #                 Conv3x3(in_channels, 64),
        #                 Conv3x3(64, 32),
        #                 Conv3x3(32, 2, activation=None)
        # )

    @staticmethod
    def random_erase(img, lambda_=0.2):
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
        sample_index = idx[index[:int(num*lambda_)]]
        patches[sample_index, ...] = 0

        patches = patches.view(m, n, kernel_size, kernel_size)
        patches = patches.contiguous()

        img = patches.permute(0,2,1,3).contiguous()
        img = img.view(h, w)

        return img
    
    @staticmethod
    def random_noise_anatomy(img, mask, anatomy = 1, lambda_=0.2):
        # img: (C, H, W)
        # mask: (H, W)
        gray_img = torch.mean(img, dim=0) # shape (H, W)
        foregroud_pixels = (gray_img*(mask==anatomy)).flatten()
        avg, std = foregroud_pixels.mean(), torch.std(foregroud_pixels)
        lower_bound, upper_bound = avg - std, avg + std

        lower_mask = (gray_img >= lower_bound)
        upper_mask = (gray_img <= upper_bound)
        noise_mask = (lower_mask*upper_mask).long() # shape (H, W)

        # keep lambda% noise points
        idx = torch.nonzero(noise_mask)
        num = idx.shape[0]
        idx_mask = np.array(range(num))
        np.random.shuffle(idx_mask)
        sample_idx = idx[idx_mask[:int(num*lambda_)], :]

        h, w = mask.shape

        mask = mask.flatten()
        sample_idx = sample_idx[:,0]*w + sample_idx[:,1]

        # pick one random foreground class as noise
        mask.index_fill_(0, sample_idx, anatomy)

        mask = mask.view(h, w)

        return mask            

    def random_noise(self, img, mask, lambda_=0.2):
        # img: (C, H, W)
        # mask: (H, W)
        if not mask.max(): # if black mask
            return mask
        values = torch.unique(mask).detach().cpu().numpy()
        values = values[values!=0]
        v = np.random.choice(values)
        # for v in values:
        mask = self.random_noise_anatomy(img, mask, v, lambda_)
        return mask    

    def forward_once(self, x):
        l1 = self.left_conv1(x)
        l2 = self.left_conv2(l1)
        l3 = self.left_conv3(l2)
        l4 = self.left_conv4(l3)
        return l1, l2, l3, l4

    def forward(self, x):
        # coarse segmentation
        unet_out = self.unet(x) # [B, C, H, W]
        coarse_logit = unet_out['logit']
        emb = unet_out['emb']

        coarse_score = torch.softmax(coarse_logit, dim=1)

        anchor_features = self.forward_once(coarse_score)

        if self.training: # during training, activate triplet loss
            lambda_ = x['lambda_']
            mask = x['mask']
            crp_mask = torch.zeros_like(mask)
            lambda_ = 0.5-(x['epoch']-start_epoch)/total_epoch*0.4

            for i in range(mask.size(0)):
                crp_mask[i,...] = self.random_erase(mask[i,...].clone(), lambda_=lambda_)
                crp_mask[i,...] = self.random_noise(x['image'][i,...], crp_mask[i,...], lambda_=lambda_)
                
            mask = F.one_hot(mask.long(), self.class_num).permute(0,3,1,2).float()
            crp_mask = F.one_hot(crp_mask.long(), self.class_num).permute(0,3,1,2).float()

            mask_features = self.forward_once(mask)[-1]
            crp_mask_features = self.forward_once(crp_mask)[-1]
    
            triplet_loss = nn.TripletMarginLoss(margin=0.1)(anchor_features[-1], 
                            mask_features, crp_mask_features)
        else:
            triplet_loss = torch.tensor([0]).to(coarse_score.device)
        # triplet_loss = torch.tensor([0]).to(coarse_score.device)
        l1, l2, l3, l4 = anchor_features
        r3 = self.right_conv4(l3, l4)
        r2 = self.right_conv3(l2, r3)
        r1 = self.right_conv2(l1, r2)
        bf_mask = torch.sigmoid(self.last_conv(r1)) 

        if bf_mask.size(-1) != coarse_score.size(-1):
            bf_mask = F.interpolate(bf_mask, (coarse_score.size(-2), 
                                coarse_score.size(-1)))

        # weight = self.attention_module(torch.cat((coarse_score, bf_mask, x['image']), dim=1))
        # weight = self.attention_module(torch.cat((coarse_score, bf_mask), dim=1))
        # weight = self.attention_module(-x['image'])
        # weight = torch.softmax(weight, dim=1)
        # tex_weight, topo_weight = torch.split(weight, [1,1], dim=1)

        tex_weight = torch.ones_like(bf_mask)*0.3
        topo_weight = torch.ones_like(bf_mask)*0.7

        # bf_mask = bf_mask / (bf_mask.max() + 1e-12)

        # tex_weight = torch.ones_like(bf_mask)*0
        # topo_weight = torch.ones_like(bf_mask)*1

        bg, fg = torch.split(coarse_score, [1, coarse_score.size(1)-1], dim=1)
        # bg, fg = torch.split(coarse_score.detach(), [1, coarse_score.size(1)-1], dim=1)

        fg_sum = torch.sum(fg, dim=1, keepdims=True)+1e-12
        bg, fg = (tex_weight*bg+topo_weight*(1-bf_mask)), tex_weight*fg*fg_sum+topo_weight*bf_mask*fg
        # bg, fg = (tex_weight*bg+topo_weight*(1-bf_mask.detach())), tex_weight*fg*fg_sum+topo_weight*bf_mask.detach()*fg
        bg = bg * fg_sum

        # bg = bg*(1-bf_mask)
        # fg = fg*bf_mask

        logit = torch.cat((bg, fg), dim=1)   
        logit = torch.nn.functional.normalize(logit, dim=1, p=1)
        
        out = {}
        out['coarse_logit'] = coarse_logit
        out['logit'] = logit
        out['topo_mask'] = bf_mask.squeeze(1)
        out['triplet_loss'] = triplet_loss
        out['emb'] = emb
        out['topo_weight'] = topo_weight

        return out

if __name__ == "__main__":
    # model = RegUNet(1, 5).cuda()
    # model = UNet(1, 5).cuda()
    # model = ResUNet(1, 5).cuda()
    # model = DTUNet(1, 5).cuda()
    model = FlexUNet(1, 5).cuda()
    x = torch.rand(3, 1, 224, 224).cuda()
    x = {'image':x}
    model.eval()
    x = model(x)
    print(x['logit'].shape)
    
