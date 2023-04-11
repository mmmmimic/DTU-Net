import torch.nn as nn

class CSELoss:
    def __init__(self, weight, **kwargs):
        self.cse = nn.CrossEntropyLoss(weight=weight, **kwargs)

    def __call__(self, outs):
        logits = outs['logit']
        gt = outs['label']

        return self.cse(logits, gt)