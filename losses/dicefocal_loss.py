from monai.losses import DiceFocalLoss
from torch.nn import functional as F

class DFLoss:
    def __init__(self, weight, **kwargs):
        if 'sigmoid' in kwargs.keys():
            self.criterion = DiceFocalLoss(
                                        include_background=True, 
                                        # softmax=True, 
                                        sigmoid=True,
                                        focal_weight=weight, 
                                        # to_onehot_y=True,
                                        to_onehot_y=False,
                                        lambda_dice=1., lambda_focal=1.,
                                        smooth_dr=1e-5, smooth_nr=1e-5,
                                        squared_pred=False, 
                                        )
            self.sigmoid = True
        else:
            self.criterion = DiceFocalLoss(
                                        include_background=True, 
                                        softmax=True, 
                                        focal_weight=weight, 
                                        to_onehot_y=True,
                                        lambda_dice=1., lambda_focal=1.,
                                        smooth_dr=1e-5, smooth_nr=1e-5,
                                        squared_pred=False, 
                                        )
            self.sigmoid=False           

    def __call__(self, x):
        logit, mask = x['logit'], x['mask']
        # to onehot
        if not self.sigmoid:
            mask = mask.unsqueeze(1)
        loss = self.criterion(logit, mask)
        return loss