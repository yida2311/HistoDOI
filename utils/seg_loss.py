import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovasz_loss import LovaszSoftmax

__all__ = ['SchpLoss', 'CrossEntropyLoss', 'SymmetricCrossEntropyLoss', 
            'NormalizedSymmetricCrossEntropyLoss', 'FocalLoss', 'SoftCrossEntropyLoss2d']


class DecoupledSegLoss_v1(nn.Module):
    """Decoupled Segmentation loss 
        clean set: CrossEntropy, noisy set: SymmetricCrossEntropy
    """
    def __init__(self, num_classes=4, alpha=1, beta=1, ignore_index=-1):
        self.clean_loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.noisy_loss = SymmetricCrossEntropyLoss(alpha=alpha, beta=beta, num_classes=num_classes, ignore_index=ignore_index)
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        clean_targets, noisy_targets = self.decoupleTargets(targets)
        loss_clean = self.clean_loss(inputs, clean_targets)
        loss_noisy = self.noisy_loss(inputs, noisy_targets)
        return loss_clean + loss_noisy
        
    def decoupleTargets(self, targets):
        clean_mask = (targets==0) or (targets==1)
        noisy_mask = (targets==2) or (targets==3)
        clean_targets = targets
        clean_targets[noisy_mask] = self.ignore_index
        noisy_targets = targets
        noisy_targets[clean_mask] = self.ignore_index

        return clean_targets, noisy_targets


class DecoupledSegLoss_v2(nn.Module):
    """Decoupled Segmentation loss 
        clean set: Focal, noisy set: SymmetricCrossEntropy
    """
    def __init__(self, num_classes=4, alpha=1, beta=1, gamma=1, ignore_index=-1):
        self.clean_loss = FocalLoss(gamma=gamma, ignore=ignore_index)
        self.noisy_loss = SymmetricCrossEntropyLoss(alpha=alpha, beta=beta, num_classes=num_classes, ignore_index=ignore_index)
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        clean_targets, noisy_targets = self.decoupleTargets(targets)
        loss_clean = self.clean_loss(inputs, clean_targets)
        loss_noisy = self.noisy_loss(inputs, noisy_targets)
        return loss_clean + loss_noisy
        
    def decoupleTargets(self, targets):
        clean_mask = (targets==0) or (targets==1)
        noisy_mask = (targets==2) or (targets==3)
        clean_targets = targets
        clean_targets[noisy_mask] = self.ignore_index
        noisy_targets = targets
        noisy_targets[clean_mask] = self.ignore_index

        return clean_targets, noisy_targets


#======================= SCHP Loss ======================================
class SchpLoss(nn.Module):
    """ For self-correction human parsing"""
    def __init__(self, use_class_weight=False, lamda=1, num_classes=4):
        super(SchpLoss, self).__init__()
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


#================== Symmetric Cross Entropy Loss =========================
class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=4, ignore_index=-100):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropyLoss()
        self.rce_loss = ReverseCrossEntropyLoss(num_classes=num_classes)

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        rce = self.rce_loss(inputs, targets)

        loss = self.alpha * ce + self.beta * rce 
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ReverseCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100):
        super(ReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        targets_one_hot = one_hot(targets, self.num_classes)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)

        loss = -1 * torch.sum(pred * torch.log(targets_one_hot), dim=1)

        return loss.mean()


#===================== Normalized Symmetric Cross Entropy Loss =====================
class NormalizedSymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=4, ignore_index=-100):
        super(NormalizedSymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce_loss = NormalizedCrossEntropyLoss(num_classes=num_classes)
        self.nrce_loss = NormalizedReverseCrossEntropyLoss(num_classes=num_classes)

    def forward(self, inputs, targets):
        nce = self.nce_loss(inputs, targets)
        nrce = self.nrce_loss(inputs, targets)
        
        loss = self.alpha * nce + self.beta * nrce 
        return loss


class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100):
        super(NormalizedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        pred = F.log_softmax(inputs, dim=1)
        targets_one_hot = one_hot(targets, self.num_classes)
        ce =  -1 * torch.sum(targets_one_hot * pred, dim=1)
        C = -1 * torch.sum(pred, dim=1)
        nce = torch.div(ce, C)
        return nce.mean()


class NormalizedReverseCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100):
        super(NormalizedReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        targets_one_hot = one_hot(targets, self.num_classes)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)

        rce = -1 * torch.sum(pred * torch.log(targets_one_hot), dim=1)
        nrce = rce / (self.num_classes-1) / 4

        return nrce.mean()


#======================================= Focal Loss ======================================
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


#=============== Soft Cross Entropy Loss ========================
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





