# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class ConfusionMatrixCls(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist
    
    def update(self, label_trues, label_preds):
        # print('gt   ', label_trues)
        # print('pred ', label_preds)
        self.confusion_matrix += self._fast_hist(label_trues, label_preds, self.n_classes)
    
    def get_scores(self):
        """Return accuracy score evaluation result for classification
            - precision
            - recall
            - f1
            - macro_precision
            - macro_recall
            - macro_f1
        """
        TP = np.diag(self.confusion_matrix)
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        FP = np.sum(self.confusion_matrix, axis=0) - TP

        # print(self.confusion_matrix)
        # print('TP  ', TP)
        # print('FP  ', FP)
        # print('FN  ', FN)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * TP / (2*TP + FP + FN)
        iou = TP / (TP + FP + FN)

        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        mean_iou = np.mean(iou)

        return {"precision": precision,
                "recall": recall,
                "f1": f1,
                'iou': iou,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "mIOU": mean_iou,
                }
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
      



class ConfusionMatrixSeg(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along
        acc = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        acc_mean = np.mean(np.nan_to_num(acc[1:]))
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou = intersect / union
        mean_iou = np.mean(np.nan_to_num(iou[1:]))
        tm_iou = np.mean(np.nan_to_num(iou[2:]))
        
        dice = 2 * intersect / (hist.sum(axis=1) + hist.sum(axis=0))
        mean_dice = np.mean(np.nan_to_num(dice[1:]))

        freq = hist.sum(axis=1) / hist.sum() # freq of each target
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        freq_iou = (freq * iou).sum()

        return {'accuracy': acc,
                'accuracy_mean': acc_mean,
                'freqw_iou': freq_iou,
                'iou': iou, 
                'iou_mean': mean_iou, 
                "iou_tm": tm_iou,
                'dice': dice,
                'dice_mean': mean_dice,
                # 'IoU_threshold': np.mean(np.nan_to_num(self.iou_threshold)),
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        # self.iou = []
        # self.iou_threshold = []



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



if __name__ == "__main__":
    a = np.array([0, 1, 2, 2, 1, 1], dtype='uint8')
    b = np.array([0, 2, 0, 2, 1, 1], dtype='uint8')

    matrix = ConfusionMatrixCls(3)
    matrix.update(a, b)
    print(matrix.confusion_matrix)
    print(matrix.get_scores())