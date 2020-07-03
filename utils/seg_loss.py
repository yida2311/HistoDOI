import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovasz_loss import LovaszSoftmax

__all__ = ['CriterionAll', 'CrossEntropyLoss2d', 'FocalLoss', 'SoftCrossEntropyLoss2d']


class CriterionAll(nn.Module):
    def __init__(self, use_class_weight=False, lamda=1, num_classes=4):
        super(CriterionAll, self).__init__()
        self.use_class_weight = use_class_weight
        self.criterion = nn.CrossEntropyLoss()
        self.lovasz = LovaszSoftmax()
        self.kldiv = KLDivergenceLoss(T=1)
        self.lamda = lamda
        self.num_classes = num_classes
    
    def parsing_loss(self, preds, soft_preds, masks, cycle_n=None):
        """
        Loss function definition.
        Args:
            preds: preds of model
            soft_preds: preds of schp model
            masks: gt masks
        Returns:
            Calculated Loss.
        """
        h, w = masks.size(1), masks.size(2)

        # loss = self.lovasz(preds, masks)
        loss = self.criterion(preds, masks)
        if soft_preds is not None:
            soft_preds = F.softmax(soft_preds)
            soft_preds = moving_average(soft_preds, one_hot(masks, self.num_classes), alpha=1.0/(cycle_n+1))
            loss += self.lamda * self.kldiv(preds, soft_preds)
            loss /= (self.lamda + 1)

        return loss 
    
    def forward(self, preds, soft_preds, masks, cycle_n=None):
        loss = self.parsing_loss(preds, soft_preds, masks, cycle_n=cycle_n)
        return loss 

    def _generate_weights(self, masks, num_classes):
        """
        masks: torch.Tensor with shape [B, H, W]
        """
        masks_label = masks.data.cpu().numpy().astype(np.int64)
        pixel_nums = []
        tot_pixels = 0
        for i in range(num_classes):
            pixel_num_of_cls_i = np.sum(masks_label == i).astype(np.float)
            pixel_nums.append(pixel_num_of_cls_i)
            tot_pixels += pixel_num_of_cls_i
        weights = []
        for i in range(num_classes):
            weights.append(
                (tot_pixels - pixel_nums[i]) / tot_pixels / (num_classes - 1)
            )
        weights = np.array(weights, dtype=np.float)
        # weights = torch.from_numpy(weights).float().to(masks.device)
        return weights


class KLDivergenceLoss(nn.Module):
    def __init__(self, T=1):
        super(KLDivergenceLoss, self).__init__()
        self.T = T

    def forward(self, input, target):
        log_input_prob = F.log_softmax(input / self.T, dim=1)
        # target_porb = F.softmax(target / self.T, dim=1)
        loss = F.kl_div(log_input_prob, target_porb)
        return self.T*self.T*loss # balanced


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input, target):
        '''
        only support ignore at 0
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot: target = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class SoftCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index+1)], targets[range(index, index+1)])/(targets.size()[2] *
                                                                                             targets.size()[3])
        return loss

#-------------------------------- helper function ----------------------------------#

def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # index is flatten (during ignore) ##################
    # size = index.size()[:1] + (classes,)
    # view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)





