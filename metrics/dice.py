import torch
import torch.nn as nn
from torch.nn import functional as F

def one_hot_embedding(y, num_classes):
    '''
    Embedding labels to one-hot form.

    Args:
        y: (LongTensor) class labels, sized [N, W, H]

    Returns:
        (tensor) encoded labels, sized [N, C, W, H].
    '''
    y = y.long()
    b, w, h = y.shape # [B, W, H]
    y = y.flatten() # [B*W*H]

    D = torch.eye(num_classes).to(y.device) # [C, C]

    y = D[y] # [N, C]

    y = y.view(b, w, h, -1).permute(0, 3, 1, 2) # [B, C, W, H]

    return y.float()

def get_dice(logit, mask, class_num, smooth=1, reduction='micro', exclude_background=True):
    # mask = nn.functional.one_hot(mask, class_num).permute(0,3,1,2)
    mask = one_hot_embedding(mask, num_classes=class_num)
    if smooth is not None:
        pred = torch.argmax(logit, dim=1)
        # pred = nn.functional.one_hot(pred, class_num).permute(0,3,1,2)
        pred = one_hot_embedding(pred, num_classes=class_num)
        if exclude_background:
            pred = pred[:, 1:, ...]
            mask = mask[:, 1:, ...]
        if reduction == 'macro':
            intersection = (pred*mask).flatten(-2, -1).sum(-1) # (B, C)
            union = (pred + mask).flatten(-2, -1).sum(-1)
            nonzero_num = torch.sum(union>0, dim=-1)
            dice_score = (2*intersection+smooth) / (union+smooth)
            dice_score = (dice_score.mean(-1)*intersection.size(-1) - (intersection.size(-1) - nonzero_num))/nonzero_num
            dice_score = dice_score.mean(-1)
        elif reduction == 'micro':
            intersection = torch.sum(pred*mask)
            dice_score = (2.*intersection + smooth) / (torch.sum(pred) + torch.sum(mask) + smooth)
        dice_score.requires_grad = True
    else:
        pred = torch.softmax(logit, dim=1)
        if exclude_background:
            pred = pred[:, 1:, ...]
            mask = mask[:, 1:, ...]
        if reduction == 'macro':
            intersection = (pred*mask).flatten(-2, -1).sum(-1) # (B, C)
            union = pred.flatten(-2, -1).sum(-1) + mask.flatten(-2, -1).sum(-1)
            nonzero_num = torch.sum(mask.flatten(-2,-1).sum(-1)>0, dim=-1)
            dice_score = (2*intersection / union)
            dice_score = dice_score.mean(-1)*intersection.size(1)/nonzero_num
            dice_score = dice_score.mean(-1)   
        elif reduction == 'micro':
            intersection = (pred*mask).sum()
            dice_score = (2.*intersection / (pred.sum() + mask.sum()))             

    return dice_score