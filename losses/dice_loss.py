from metrics import get_dice
class DiceLoss:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, outs):
        logit, mask = outs['logit'], outs['mask']  
        dice_score = get_dice(logit, mask, **self.kwargs)

        return 1. - dice_score