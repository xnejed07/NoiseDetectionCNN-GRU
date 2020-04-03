
import torch
import numpy as np
from scipy.special import softmax,expit
from sklearn.metrics import f1_score,confusion_matrix,cohen_kappa_score,roc_curve,roc_auc_score,average_precision_score


class Statistics(object):
    def __init__(self):
        self.target = []
        self.logits = []

    def reset(self):
        self.target = []
        self.logits = []

    @staticmethod
    def idx2onehot(idx_array):
        y = np.zeros((idx_array.shape[0], idx_array.max() + 1))
        y[np.arange(y.shape[0]), idx_array] = 1
        return y

    @staticmethod
    def F1(conf):
        x0 = np.sum(conf, 0)
        x1 = np.sum(conf, 1)
        dg = np.diag(conf)
        f1 = 2 * dg / (x0 + x1)
        return f1

    @staticmethod
    def Kappa(conf):
        x0 = np.sum(conf, 0)
        x1 = np.sum(conf, 1)
        N = np.sum(np.sum(conf))
        ef = np.sum(x0 * x1 / N)
        dg = np.sum(np.diag(conf))
        K = (dg - ef) / (N - ef)
        return K

    def append(self,target,logits):
        self.logits.append(logits.data.cpu().numpy())
        self.target.append(target.data.cpu().numpy())

    @staticmethod
    def random_auprc(target):
        y_chance = np.zeros((target.max()+1,))
        for i in range(target.max()+1):
            y_chance[i] = len(target[target==i]) / len(target)

        return y_chance


    def evaluate(self):
        self.logits = np.concatenate(self.logits)
        self.target = np.concatenate(self.target).astype('int32')

        self.probs = softmax(self.logits,axis=1)
        self.argmax = np.argmax(self.probs,axis=1)

        CONF = np.array(confusion_matrix(y_true=self.target,y_pred=self.argmax))
        F1 = Statistics.F1(CONF)
        KPS = Statistics.Kappa(CONF)
        AUROC = roc_auc_score(y_true=Statistics.idx2onehot(self.target),y_score=self.probs,average=None)
        AUPRC = average_precision_score(y_true=Statistics.idx2onehot(self.target),y_score=self.probs,average=None)
        AUPRC_chance = self.random_auprc(self.target)

        print(CONF)
        print(F1)
        print(KPS)
        print(AUROC,np.mean(AUROC))
        print(AUPRC,np.mean(AUPRC))
        print(AUPRC_chance)

        self.reset()