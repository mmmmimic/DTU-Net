from sklearn.metrics import classification_report, accuracy_score, jaccard_score, confusion_matrix, top_k_accuracy_score, balanced_accuracy_score
from .dice import get_dice
from collections import defaultdict
import torch
import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(int)
        self.count = 0
        self.sum = defaultdict(int)
    
    def add(self, kwds, batch_size):
        for k, v in kwds.items():
            v = v.item()
            self.val[k] = v
            self.count += batch_size
            self.sum[k] += v*batch_size
            self.avg[k] = self.sum[k] / self.count

class ClassMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []
        self.logits = []

    def add(self, logit, pred, gt):
        '''
        logit (C, N, ...)
        pred (N, ...)
        gt (N, ...)
        '''
        self.preds.append(pred.detach().cpu().numpy())
        self.gts.append(gt.detach().cpu().numpy())
        self.logits.append(logit.detach().cpu().numpy())

    def gather(self):
        self.preds = np.concatenate(self.preds)
        self.gts = np.concatenate(self.gts)
        self.logits = np.concatenate(self.logits, axis=1).transpose()

    def get_acc(self):
        return accuracy_score(self.gts, self.preds)

    def get_acc2(self):
        return top_k_accuracy_score(self.gts, self.logits, k=2)

    def get_acc3(self):
        return top_k_accuracy_score(self.gts, self.logits, k=3)
    
    def get_acc5(self):
        return top_k_accuracy_score(self.gts, self.logits, k=5)    

    def get_fg_acc2(self):
        gts = self.gts
        logits = self.logits
        index = gts > 0
        for i in range(len(index)):
            if not i:
                index[i] = True
                break
        return top_k_accuracy_score(gts[index], logits[index, :], k=2)

    def get_fg_acc3(self):
        gts = self.gts
        logits = self.logits
        index = gts > 0
        for i in range(len(index)):
            if not i:
                index[i] = True
                break
        return top_k_accuracy_score(gts[index], logits[index, :], k=3)
    
    def get_fg_acc5(self):
        gts = self.gts
        logits = self.logits
        index = gts > 0
        for i in range(len(index)):
            if not i:
                index[i] = True
                break
        return top_k_accuracy_score(gts[index], logits[index, :], k=5)

    def get_micro_iou(self):
        return jaccard_score(self.gts, self.preds, average='micro')

    def get_macro_iou(self):
        return jaccard_score(self.gts, self.preds, average='macro')

    def get_fg_acc(self):
        gts = self.gts
        preds = self.preds
        index = gts > 0
        return accuracy_score(gts[index], preds[index])

    def get_avg_acc(self):
        return balanced_accuracy_score(self.gts, self.preds)

    def get_confusion_matrix(self):
        return confusion_matrix(self.gts, self.preds)

    def get_classification_report(self):
        return classification_report(self.gts, self.preds)

    
            
