from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .dicefocal_loss import DFLoss, DiceFocalLoss
import torch.nn as nn
import torch
from metrics import get_dice

class DTULoss:
    def __init__(self, weight, **kwargs):
        self.dicefocal = DFLoss(weight=weight)
        self.bce = nn.BCELoss()

        self.l1 = DiceFocalLoss(
                                include_background=True, 
                                softmax=False, 
                                sigmoid=False,
                                focal_weight=weight, 
                                to_onehot_y=True,
                                lambda_dice=1., lambda_focal=1.,
                                smooth_dr=1e-5, smooth_nr=1e-5,
                                squared_pred=False, 
                                )

    def __call__(self, outs):
        coarse_logit, mask, logit, topo_mask, triplet_loss = outs['coarse_logit'], outs['mask'], outs['logit'], outs['topo_mask'], outs['triplet_loss']  
        dicefocal_loss_coarse = self.dicefocal({'logit':coarse_logit, 'mask':mask})

        bce_loss = self.bce(topo_mask, (mask>0).float())

        dicefocal_loss_fine = self.l1(logit, mask.unsqueeze(1))

        loss = bce_loss + triplet_loss + dicefocal_loss_coarse + dicefocal_loss_fine # dicefocal_loss_fine is optional, with which the result is better

        return loss