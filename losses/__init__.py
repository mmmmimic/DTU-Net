from .focal_loss import FocalLoss, get_weight
from .dice_loss import DiceLoss
from .cse_loss import CSELoss
from .dtu_loss import DTULoss

class Criterion:
    def __init__(self, loss_dict, loss_configs, weights=None):
        self.loss_modules = []
        self.loss_weights = []
        self.loss_names = []

        for loss, weight in loss_dict.items():
            if loss in loss_configs.keys():
                kwargs = loss_configs[loss]
            else:
                kwargs = {}
            if loss == "dice loss":
                self.loss_modules.append(DiceLoss(**kwargs))
            elif loss == "focal loss":
                self.loss_modules.append(FocalLoss(weight=weights, **kwargs))
            elif loss == "dtu loss":
                self.loss_modules.append(DTULoss(weight=weights, **kwargs))
            elif loss == "dicefocal loss":
                self.loss_modules.append(DFLoss(weight=weights, **kwargs))
            else:
                raise NotImplementedError()
            self.loss_weights.append(weight)
            self.loss_names.append(loss)

    def __call__(self, outs):
        loss = {}
        for loss_criterion, weight, name in zip(self.loss_modules, self.loss_weights, self.loss_names):
            loss[name] = weight*loss_criterion(outs)

        loss['loss'] = sum(loss.values())
        return loss

    def __repr__(self):
        return str(self.loss_names)